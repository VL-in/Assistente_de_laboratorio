"""
Assistente de laboratório — interface Streamlit (MVP alinhado ao playbook biotech).

Ponto de entrada único da aplicação. Orquestra todos os subsistemas locais:

- ``projects_loader``: descobre projetos e arquivos nos volumes montados.
- ``rag``: extração, chunking, índice txtai e busca semântica.
- ``OpenAI`` (cliente SDK): conversa com o LM Studio rodando no host via API
  compatível com OpenAI.

Abas da UI (usuário final)
--------------------------
0. Conversa — chat com o agente (documentos e planilhas usados automaticamente).
1. Documentos — escanear pastas, inventário e atualização da base de conhecimento.

Aba Desenvolvimento (sub-abas internas)
---------------------------------------
Visão geral, busca semântica, índice vetorial (parâmetros RAG), planilhas (OLAP) e diagnóstico.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from llm_config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    get_llm_base_url_raw,
    get_llm_model,
    llm_runtime_config,
    normalize_openai_base_url,
)
from chat_router import classification_needs_llm, resolve_chat_routes
from qwen35_inference import (
    DEFAULT_CHAT_MAX_TOKENS,
    create_chat_completion,
    env_enable_thinking_default,
    is_qwen35_model,
    sanitize_history_message,
    select_chat_profile,
    strip_thinking_blocks,
)
from projects_loader import (
    DEFAULT_DOCUMENT_EXTENSIONS,
    ENV_PROJETOS_ROOT,
    ProjectScan,
    documents_by_project,
    projetos_root_from_env,
    running_inside_docker,
    scan_all_projects,
    validate_projetos_root,
)
from olap import (
    DEMO_TABLE,
    TABULAR_EXTENSIONS,
    check_duckdb,
    demo_aggregation,
    duckdb_data_root,
    duckdb_database_path,
    duckdb_library_version,
    has_ingested_tables,
    list_ingested_tables,
    olap_status,
    run_nl_olap_query,
    seed_demo_data,
    sync_tabular_from_scans,
)
from olap.schema_catalog import build_schema_catalog_text
from rag import (
    EMBEDDING_MODEL_ID,
    build_index,
    format_context_for_llm,
    index_mtime,
    index_ready,
    manifest_exists,
    manifest_path,
    search_with_backend,
    txtai_data_root,
    txtai_index_path,
)

# ── System prompt ────────────────────────────────────────────────────────────
# Instrução base enviada ao LLM em todas as conversas. Quando o RAG está ativo,
# o contexto recuperado é **concatenado** a este prompt (não substitui), para
# que o modelo mantenha o papel e as restrições definidas aqui.

CHAT_SYSTEM_PROMPT = (
    """<role>
Você é um assistente de laboratório experiente que atua em pesquisa e desenvolvimento de imunodiagnósticos, principalmente ELISA. Você passou muitos anos trabalhando com dados laboratoriais, planejamento de ensaios, interpretação de resultados. Já viu inúmeros erros por desatenção e sabe que documentação e rastreabilidade de informação é crucial em projetos de desenvolvimento. Você entende que existem múltiplos documentos de experimentos que representam linhas de raciocínio contínua, tratando, por tanto, os documentos não só isoladamente, mas uma sequência.
</role>

<context>
Estamos trabalhando em um laboratório de p&D que está com projeto de ELISA ativa. Os pesquisadores planejam e documentam por meio de arquivos docx dados como materiais e insumos, lotes e validades. Os documentos possuem padrões, e os insumos se apresentam na ordem de "nome"/"Fabricante ou código"/ "Lote" ou "Ativo" do equipamento/"Validade". Os pesquisadores vão vir até você para fazer perguntas sobre o que foi feito ou usado nos experimentos passados. O seu trabalho é identificar o que o usuário está buscando e, através dos resultados, apresentar as informações relevantes. Perceba que a mesma informação pode aparecer em diferentes documentos, que podem compor a resposta retornada.
</context>

<constraints>
- Nunca invente dados, busque por retrieval as respostas quando a pergunta é voltada para os ensaios.
- Analise todos os chunks e entenda que a resposta pode ser composta por dados de diferentes documentos.
- Não altere nenhum dado dos documentos.
- Retorne as informações junto ao título do documento e a sua data de planejamento.
- Ao final de cada resposta, seja cordial e pergunte se pode ajudar com mais alguma dúvida.
- Faça sempre uma pergunta de cada vez.
- Caso não tenha encontrado respostas nos documentos, expresse isso educadamente.
</constraints>

<goals>
- Identifique o objetivo do usuário.
- Se o usuário perguntar sobre nome de insumo, fabricante, lote ou validade, lembre que os dados estão sempre descritos nessa ordem.
- Sintetizar uma resposta objetiva que contenha o dado referente às perguntas feitas, sempre referenciando o documento.
</goals>

<invocation>
Sempre use a mesma língua do usuário. Por padrão, utilize português brasileiro. Seja cordial, profissional, objetivo e educado.
</invocation>"""
)

st.set_page_config(
    page_title="Assistente Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Espaçamento mais compacto para toggles na aba de desenvolvimento.
st.markdown(
    """
    <style>
    div[data-testid="stToggle"] label p { font-size: 0.85rem; margin: 0; }
    div[data-testid="stTabs"] button { padding: 0.35rem 0.75rem; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers de configuração e conexão ───────────────────────────────────────

def _project_root_override() -> str:
    """Caminho digitado na UI (aba Documentos ou Desenvolvimento)."""
    for key in ("path_override", "path_override_dev"):
        value = (st.session_state.get(key) or "").strip()
        if value:
            return value
    return ""


def _root_from_session() -> Path:
    """Retorna a raiz de projetos considerando o override digitado na interface."""
    override = _project_root_override()
    if override:
        return Path(override).expanduser().resolve()
    return projetos_root_from_env()


def _parse_extensions(ext_input: str) -> frozenset[str]:
    """Converte a string de extensões da UI em frozenset normalizado (ex.: ``'.docx'``)."""
    parts = [p.strip().lower() for p in ext_input.split(",") if p.strip()]
    exts = frozenset(p if p.startswith(".") else f".{p}" for p in parts)
    return exts if exts else DEFAULT_DOCUMENT_EXTENSIONS


def _is_timeout_error(exc: BaseException) -> bool:
    """Detecta timeout vindo de urlopen (URLError.reason, encadeamento ou exceção direta)."""
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException) and _is_timeout_error(reason):
        return True
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException) and _is_timeout_error(cause):
        return True
    return False


# ── Cache do backend txtai ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando índice vetorial (txtai)…")
def _txtai_backend_cached(mtime_key: float):
    """
    Carrega e armazena a instância ``Embeddings`` do txtai em cache de processo.

    O ``st.cache_resource`` mantém a instância em memória entre reruns do
    Streamlit, evitando recarregar o modelo de ~500 MB a cada interação.

    O parâmetro ``mtime_key`` é o timestamp de modificação do manifesto
    (obtido via ``index_mtime()``). Quando o índice é reconstruído, esse valor
    muda e o Streamlit descarta o cache, forçando o carregamento do índice
    atualizado. Após um rebuild explícito, ``_txtai_backend_cached.clear()``
    também é chamado diretamente para garantia imediata.

    Retorna ``None`` quando não há índice (``mtime_key <= 0``), evitando
    tentativa de carregar um caminho inexistente.
    """
    if mtime_key <= 0:
        return None
    from txtai import Embeddings

    from rag.index_txtai import embeddings_config

    emb = Embeddings(embeddings_config())
    emb.load(str(txtai_index_path()))
    return emb


def rag_semantic_search(query: str, limit: int) -> list[dict]:
    """
    Executa uma busca semântica usando o backend txtai em cache.

    Separa responsabilidades: ``_txtai_backend_cached`` gerencia o ciclo de
    vida do modelo (carregamento caro, feito uma vez); esta função apenas
    executa a busca (barata, feita a cada pergunta).
    """
    if not index_ready():
        return []
    mt = index_mtime()
    emb = _txtai_backend_cached(mt)
    if emb is None:
        return []
    return search_with_backend(emb, query, limit)


def _check_openai_compatible_models(base_url: str, *, timeout_s: float = 5.0) -> tuple[bool, str]:
    """GET {base}/v1/models — compatível com LM Studio no host."""
    b = normalize_openai_base_url(base_url.strip())
    if not b:
        return False, "URL base vazia."
    url = f"{b}/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                return False, f"HTTP {resp.status}: {raw[:500]}"
            try:
                data = json.loads(raw)
                ids = []
                for item in data.get("data", [])[:12]:
                    mid = item.get("id")
                    if mid:
                        ids.append(mid)
                preview = ", ".join(ids) if ids else raw[:800]
                return True, preview
            except json.JSONDecodeError:
                return True, raw[:800]
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        if _is_timeout_error(e):
            return False, f"Tempo esgotado ({timeout_s}s), favor atualize a página e tente novamente."
        return False, f"Rede/URL: {e.reason!r}"
    except TimeoutError:
        return False, f"Tempo esgotado ({timeout_s}s)."
    except OSError as e:
        if _is_timeout_error(e):
            return False, f"Tempo esgotado ({timeout_s}s), favor atualize a página e tente novamente."
        return False, str(e)


def _trim_chat_history(messages: list[dict], max_turns: int) -> list[dict]:
    """
    Retorna os últimos ``max_turns`` pares (user + assistant) do histórico.

    Cada "turno" corresponde a 2 mensagens (1 do usuário + 1 do assistente),
    portanto o limite em mensagens é ``max_turns × 2``.

    Sem truncagem, o histórico cresce sem limite e o prompt total (system
    prompt + contexto RAG até 12 k chars + histórico) pode ultrapassar a
    janela de contexto do modelo, causando respostas truncadas ou erros
    silenciosos dependendo do backend LLM.
    """
    if max_turns <= 0:
        return []
    max_msgs = max_turns * 2  # cada turno = 1 mensagem do usuário + 1 do assistente
    if len(messages) <= max_msgs:
        return messages
    return messages[-max_msgs:]


# ── Barra lateral ────────────────────────────────────────────────────────────

def _run_project_scan(*, show_errors_in_sidebar: bool = False) -> None:
    """Escaneia projetos, atualiza inventário na sessão e sincroniza planilhas."""
    root = _root_from_session()
    ok, msg = validate_projetos_root(root)
    if not ok:
        st.session_state["scan_error"] = msg
        if show_errors_in_sidebar:
            st.sidebar.error(msg)
        return
    st.session_state.pop("scan_error", None)
    ext_input = str(st.session_state.get("ext_input") or "")
    exts = _parse_extensions(ext_input)
    compute_hashes = bool(st.session_state.get("compute_hashes", True))
    with st.spinner("Lendo pastas e atualizando planilhas…"):
        scans = scan_all_projects(
            root,
            extensions=exts,
            compute_hashes=compute_hashes,
        )
        tabular_scans = scan_all_projects(
            root,
            extensions=TABULAR_EXTENSIONS,
            compute_hashes=compute_hashes,
        )
        olap_stats = sync_tabular_from_scans(tabular_scans)
    st.session_state["last_scan"] = scans
    st.session_state["last_root"] = str(root)
    st.session_state["last_exts"] = exts
    st.session_state["last_olap_sync"] = olap_stats
    st.rerun()


def _render_sidebar() -> None:
    """Barra lateral enxuta: status da sessão e atalho para Documentos."""
    st.sidebar.markdown("### Assistente Lab")
    st.sidebar.caption(
        "Use a aba **Documentos** para escanear pastas e atualizar a base. "
        "Cada subpasta na raiz é um projeto."
    )
    scans = st.session_state.get("last_scan")
    if scans is not None:
        total = sum(s.file_count for s in scans)
        st.sidebar.metric("Arquivos catalogados", total)
        if index_ready():
            st.sidebar.success("Base de conhecimento pronta")
        else:
            st.sidebar.info("Atualize a base em Documentos")
    err = st.session_state.get("scan_error")
    if err:
        st.sidebar.error(err)


# ── Abas da UI ───────────────────────────────────────────────────────────────

def _tab_inicio() -> None:
    """Aba 0 — Visão geral do produto e checklist de fases do playbook."""
    st.header("Agente de IA para documentos de P&D")
    st.markdown(
        "Este produto segue o **playbook MVP biotech**: análise documental local (`docx`, `xlsx`, `xlsm`, …), "
        "**RAG** com índice vetorial (**txtai**), consultas **OLAP** (**DuckDB**) e geração via **LM Studio** no host, "
        "tudo orquestrado aqui no **Streamlit** (sem obrigatoriedade de API REST separada)."
    )
    st.subheader("O que o agente faz (visão)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**1. Gerir fontes** — Lê diretórios locais montados no contêiner; cada subpasta de primeiro nível é um projeto."
        )
        st.markdown(
            "**2. Preparar evidências** — Inventário com `project_id` + caminho relativo, base para chunks com citação."
        )
    with c2:
        st.markdown("**3. Indexar (RAG)** — Embeddings + metadados de origem; reprocessamento quando o hash mudar.")
        st.markdown("**4. Responder com contexto** — Recuperação + LM Studio; respostas devem citar arquivo/projeto.")
    st.subheader("Fluxo do pipeline (resumo)")
    st.code(
        "Pasta local → inventário (projects_loader) → extração/chunking → txtai → chat com citações\n"
        "                                    ↘ metadados / DuckDB (OLAP)",
        language="text",
    )
    with st.expander("Fases do plano (checklist)"):
        st.markdown(
            "- **Fase 1** — Diagnóstico, LM Studio, DuckDB olá mundo (aba OLAP), metadados.\n"
            "- **Fase 2** — Parsing, hash, página de status de ingestão.\n"
            "- **Fase 3** — txtai + chat com citações.\n"
            "- **Fase 4** — RBAC, auditoria, guardrails."
        )


def _tab_documentos(root: Path, root_ok: bool, root_msg: str) -> None:
    """Aba do usuário — escanear pastas, ver inventário e atualizar a base de conhecimento."""
    st.header("Documentos")
    st.caption(
        "Coloque os arquivos nas pastas de projeto no computador (ou volume Docker), "
        "escaneie para ver o que foi encontrado e atualize a base para o assistente usar no chat."
    )

    st.subheader("1. Pasta dos projetos")
    default_hint = os.environ.get(ENV_PROJETOS_ROOT, "").strip() or str(projetos_root_from_env())
    st.text_input(
        "Caminho da pasta raiz (opcional)",
        key="path_override",
        placeholder=default_hint,
        help="Deixe vazio para usar o caminho configurado no servidor.",
    )
    if not root_ok:
        st.error(root_msg)
        st.info(
            f"Ajuste o caminho acima ou peça ao administrador para configurar `{ENV_PROJETOS_ROOT}`."
        )
        return

    scan_col, _ = st.columns([1, 3])
    with scan_col:
        if st.button("Escanear pastas", type="primary", key="btn_scan_documentos"):
            _run_project_scan()
    err = st.session_state.get("scan_error")
    if err:
        st.error(err)

    scans: list[ProjectScan] | None = st.session_state.get("last_scan")
    if scans is None:
        st.info("Nenhum escaneamento nesta sessão. Clique em **Escanear pastas** para começar.")
        return

    st.subheader("2. Arquivos encontrados")
    last_root_raw = (st.session_state.get("last_root") or "").strip()
    if last_root_raw:
        try:
            if Path(last_root_raw).resolve() != root.resolve():
                st.warning(
                    "O caminho da pasta mudou desde o último escaneamento. "
                    "Escaneie de novo para atualizar a lista."
                )
        except OSError:
            pass

    total_files = sum(s.file_count for s in scans)
    m1, m2 = st.columns(2)
    m1.metric("Projetos", len(scans))
    m2.metric("Arquivos", total_files)

    if not scans:
        st.info("Nenhuma subpasta de projeto encontrada nesta raiz.")
        return

    records = []
    for s in scans:
        for f in s.files:
            records.append(
                {
                    "projeto": f.project_id,
                    "arquivo": f.relative_path,
                    "tamanho (bytes)": f.size_bytes,
                }
            )
    df = pd.DataFrame.from_records(records)
    options = sorted({s.project_id for s in scans})
    selected = st.multiselect(
        "Filtrar por projeto",
        options=options,
        default=options,
        key="docs_filter_projects",
    )
    active_projects = selected if selected else options
    df = df[df["projeto"].isin(active_projects)]
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("3. Atualizar base de conhecimento")
    if not index_ready():
        st.caption("Ainda não há base indexada — a primeira atualização pode levar alguns minutos.")
    else:
        st.caption(
            "Somente arquivos novos ou alterados são reprocessados (comparação por impressão digital)."
        )

    if st.button("Atualizar base agora", type="primary", key="btn_update_knowledge"):
        with st.spinner("Preparando documentos para o assistente…"):
            stats = build_index(
                scans,
                project_ids=None,
                max_chars=520,
                overlap=80,
                max_doc_chars=2_000_000,
                batch_size=64,
                replace_existing=not index_ready(),
                progress=lambda msg: None,
            )
        _txtai_backend_cached.clear()
        st.session_state["last_index_stats"] = stats
        if stats.incremental:
            st.success(
                f"Base atualizada — {stats.chunks_written} trecho(s) novo(s), "
                f"{stats.files_skipped_unchanged} arquivo(s) sem alteração."
            )
        else:
            st.success(
                f"Base criada — {stats.chunks_written} trecho(s) de "
                f"{stats.files_extracted} arquivo(s)."
            )
        if stats.errors:
            with st.expander("Avisos durante a atualização"):
                st.code("\n".join(stats.errors[:50]), language="text")
        st.rerun()


def _tab_fontes(root: Path, root_ok: bool, root_msg: str) -> None:
    """Inventário técnico (desenvolvimento) — tabela com hash e metadados."""
    st.header("Fontes e inventário")
    st.caption(
        "Módulo `projects_loader`: descobre projetos e arquivos que alimentarão a **indexação para RAG** "
        "(filtro por `project_id` para o agente não misturar evidências de projetos diferentes)."
    )
    if not root_ok:
        st.error(root_msg)
        st.info(
            f"Ajuste a raiz na barra lateral ou defina `{ENV_PROJETOS_ROOT}`. Caminho atual: `{root}`"
        )
        return

    scans: list[ProjectScan] | None = st.session_state.get("last_scan")
    if scans is None:
        st.warning("Ainda não há escaneamento nesta sessão.")
        st.markdown("Use **Escanear pastas** na aba **Documentos** para carregar o inventário.")
        return

    last_root_raw = (st.session_state.get("last_root") or "").strip()
    if last_root_raw:
        try:
            if Path(last_root_raw).resolve() != root.resolve():
                st.warning(
                    "A **pasta** na barra lateral mudou desde o último escaneamento. "
                    "O inventário abaixo ainda reflete a pasta anterior — use **Escanear pastas agora** para atualizar."
                )
        except OSError:
            st.warning(
                "Não foi possível comparar a pasta atual com a do último escaneamento. "
                "Se você alterou o caminho, execute **Escanear pastas agora**."
            )

    st.success(f"Raiz usada no último escaneamento: `{st.session_state.get('last_root', root)}`")
    total_files = sum(s.file_count for s in scans)
    m1, m2, m3 = st.columns(3)
    m1.metric("Projetos", len(scans))
    m2.metric("Arquivos catalogados", total_files)
    m3.metric(
        "Extensões no filtro",
        len(st.session_state.get("last_exts") or DEFAULT_DOCUMENT_EXTENSIONS),
    )

    if not scans:
        st.info("Nenhuma subpasta de projeto encontrada nesta raiz.")
        return

    by_proj = documents_by_project(scans)
    st.metric("Pronto para segmentação RAG", f"{len(by_proj)} chaves `project_id`")

    records = []
    for s in scans:
        for f in s.files:
            records.append(
                {
                    "projeto": f.project_id,
                    "arquivo (relativo ao projeto)": f.relative_path,
                    "tamanho (bytes)": f.size_bytes,
                    "hash SHA-256": f.content_hash_sha256 or "—",
                }
            )
    df = pd.DataFrame.from_records(records)
    options = sorted({s.project_id for s in scans})
    selected = st.multiselect(
        "Filtrar por projeto",
        options=options,
        default=options,
        key="dev_fontes_filter_projects",
    )
    # Se o usuário desmarcar tudo, tratamos como "todos" (evita tabela vazia com listagem cheia).
    active_projects = selected if selected else options
    df = df[df["projeto"].isin(active_projects)]

    st.subheader("Documentos por projeto")
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Listagem por projeto (texto)"):
        for s in scans:
            if s.project_id not in active_projects:
                continue
            st.markdown(f"**{s.project_id}** — `{s.root}` — {s.file_count} arquivo(s)")
            if s.files:
                st.code("\n".join(f.relative_path for f in s.files[:200]), language="text")
                if len(s.files) > 200:
                    st.caption(f"… e mais {len(s.files) - 200} arquivo(s).")


def _tab_indexacao_rag(scans: list[ProjectScan] | None) -> None:
    """Aba 2 — Construção e atualização do índice txtai."""
    st.header("Indexação RAG")
    st.caption(
        f"Pipeline: extração (docx/xlsx/pdf/txt/md/csv) → chunking → **`txtai`** + "
        f"[`{EMBEDDING_MODEL_ID}`](https://huggingface.co/{EMBEDDING_MODEL_ID}). "
        f"Índice em `{txtai_index_path()}`."
    )

    st.markdown(
        f"- **Volume dados txtai:** `{txtai_data_root()}`\n"
        f"- **Índice salvo:** `{txtai_index_path()}`\n"
        f"- **Manifesto (hash):** `{manifest_path()}` — "
        f"{'presente' if manifest_exists() else 'ausente'}\n"
        f"- **Índice pronto:** {'sim' if index_ready() else 'não'}"
    )

    if scans is None:
        st.warning("Faça um escaneamento na aba **Documentos** antes de indexar.")
        return
    if not scans:
        st.info("Inventário vazio — nada para indexar.")
        return

    last_root_raw = (st.session_state.get("last_root") or "").strip()
    if last_root_raw:
        try:
            if Path(last_root_raw).resolve() != _root_from_session().resolve():
                st.warning(
                    "A pasta de projetos na barra lateral **não coincide** com a pasta do último escaneamento. "
                    "Reescaneie antes de indexar."
                )
        except OSError:
            pass

    total = sum(s.file_count for s in scans)
    st.success(f"**{total}** arquivo(s) em **{len(scans)}** projeto(s) no inventário atual.")

    options = sorted({s.project_id for s in scans})
    selected = st.multiselect(
        "Projetos a incluir no índice",
        options=options,
        default=options,
        help="Vazio = todos os projetos listados.",
        key="dev_rag_index_projects",
    )
    project_filter = set(selected) if selected else None

    c1, c2, c3 = st.columns(3)
    with c1:
        chunk_chars = st.number_input(
            "Tamanho máx. do chunk (caracteres)",
            min_value=200,
            max_value=4000,
            value=520,
            step=20,
            help="O modelo MPNet usa até ~128 tokens; ~520 caracteres é um valor seguro para pt-BR.",
            key="dev_rag_chunk_chars",
        )
    with c2:
        overlap = st.number_input(
            "Sobreposição entre chunks",
            min_value=0,
            max_value=800,
            value=80,
            step=10,
            key="dev_rag_chunk_overlap",
        )
    with c3:
        batch_size = st.number_input(
            "Tamanho do lote (index txtai)",
            min_value=8,
            max_value=512,
            value=64,
            step=8,
            key="dev_rag_batch_size",
        )

    max_doc = st.number_input(
        "Limite de caracteres por arquivo (extração)",
        min_value=50_000,
        max_value=10_000_000,
        value=2_000_000,
        step=100_000,
        help="Evita carregar PDF/planilhas enormes inteiros na memória.",
        key="dev_rag_max_doc_chars",
    )

    # Desmarca automaticamente após a primeira indexação bem-sucedida, pois o
    # modo incremental já é o comportamento desejado em execuções subsequentes.
    replace_default = not index_ready()
    replace = st.checkbox(
        "Substituir índice existente",
        value=replace_default,
        help="Apaga índice e manifesto e reconstrói tudo. Desmarque para **reindexação incremental por hash**.",
        key="dev_rag_replace_index",
    )
    if not replace and index_ready():
        st.caption(
            "Modo **incremental**: só arquivos com hash SHA-256 diferente (ou novos/removidos) "
            "são reprocessados. Após atualização do app que muda a extração de `.docx`, os "
            "arquivos Word são reindexados automaticamente na próxima execução."
        )
    elif not replace:
        st.caption("Primeira indexação: será criado índice e manifesto do zero.")
    st.caption(
        "Documentos Word: tabelas de insumos/validade entram na indexação (não só parágrafos). "
        "Se o índice foi criado antes dessa correção, use **Substituir índice** uma vez."
    )

    status_box = st.empty()

    def _progress(msg: str) -> None:
        status_box.caption(msg)

    if st.button("Construir índice agora", type="primary", key="btn_build_index"):
        with st.spinner("Indexando documentos… pode levar vários minutos na primeira vez (download do modelo)."):
            stats = build_index(
                scans,
                project_ids=project_filter,
                max_chars=int(chunk_chars),
                overlap=int(overlap),
                max_doc_chars=int(max_doc),
                batch_size=int(batch_size),
                replace_existing=replace,
                progress=_progress,
            )
        _txtai_backend_cached.clear()
        st.session_state["last_index_stats"] = stats
        if stats.incremental:
            st.success(
                f"**Incremental** — chunks gravados: **{stats.chunks_written}** · "
                f"removidos: **{stats.chunks_deleted}** · "
                f"reindexados: **{stats.files_reindexed}** · "
                f"inalterados (pulados): **{stats.files_skipped_unchanged}** · "
                f"arquivos removidos do disco: **{stats.files_removed}** · "
                f"vazios/erro: **{stats.files_empty}**"
            )
        else:
            st.success(
                f"Chunks gravados: **{stats.chunks_written}** · "
                f"Arquivos com texto: **{stats.files_extracted}** · "
                f"Vazios / ignorados: **{stats.files_empty}**"
            )
        if stats.incremental and not stats.chunks_written and not stats.chunks_deleted:
            st.info(
                "Nenhuma alteração detectada — todos os arquivos do inventário "
                "coincidem com o manifesto (hash SHA-256)."
            )
        if stats.errors:
            with st.expander("Avisos e erros (detalhe)"):
                st.code("\n".join(stats.errors[:200]), language="text")
                if len(stats.errors) > 200:
                    st.caption("Lista truncada.")
        st.rerun()

    prev = st.session_state.get("last_index_stats")
    if prev:
        if getattr(prev, "incremental", False):
            st.info(
                f"Última execução (incremental): **{prev.chunks_written}** chunks novos · "
                f"**{prev.files_skipped_unchanged}** inalterados · "
                f"**{prev.files_reindexed}** reindexados."
            )
        else:
            st.info(
                f"Última execução nesta sessão: **{prev.chunks_written}** chunks · "
                f"{prev.files_extracted} arquivos com texto · {prev.files_empty} vazios."
            )

    with st.expander("Metadados por chunk (campos armazenados com o texto)"):
        st.json(
            {
                "text": "conteúdo puro — usado para gerar o embedding",
                "cited": "[Projeto: X] [Arquivo: Y] [Chunk N]\\n<conteúdo> — exibido ao usuário e enviado ao LLM",
                "project_id": "nome da pasta do projeto",
                "relative_path": "caminho dentro do projeto",
                "chunk_index": 0,
                "extract_detail": "ex.: xlsx: abas e linhas",
            }
        )


def _tab_rag_dev() -> None:
    """Aba 3 — Busca semântica direta no índice txtai, sem passar pelo LLM."""
    st.header("Teste RAG (desenvolvedor)")
    st.caption(
        "Consulta direta ao índice **txtai** (mesmo backend do chat com RAG). "
        "Use para validar chunking, metadados e relevância antes de acoplar ao LM Studio."
    )

    st.markdown(
        f"- Modelo de embedding: `{EMBEDDING_MODEL_ID}`\n"
        f"- Índice: `{txtai_index_path()}` — **pronto:** {'sim' if index_ready() else 'não'}"
    )

    if not index_ready():
        st.warning("Construa um índice na aba **Documentos** ou em **Índice vetorial** (dev) antes de testar.")
        return

    q = st.text_input(
        "Consulta semântica",
        placeholder="Ex.: validade do reagente X no projeto ELISA",
        key="dev_rag_search_query",
    )
    top_k = st.slider("Top-K", min_value=1, max_value=25, value=8, key="dev_rag_search_top_k")

    if st.button("Executar busca", type="primary", key="btn_rag_dev_search"):
        if not q.strip():
            st.error("Informe uma consulta.")
        else:
            with st.spinner("Buscando…"):
                hits = rag_semantic_search(q, top_k)
            if not hits:
                st.info("Nenhum resultado para esta consulta.")
            else:
                st.metric("Resultados", len(hits))
                for i, h in enumerate(hits, start=1):
                    score = h.get("score")
                    # "cited" contém o texto completo com prefixo de projeto/arquivo.
                    # Cai em "text" para compatibilidade com índices antigos.
                    body = (h.get("cited") or h.get("text") or "").strip()
                    title = f"#{i} · score={score}"
                    with st.expander(title, expanded=(i <= 3)):
                        st.markdown(body if body else "_(sem texto)_")
                        with st.popover("JSON bruto"):
                            st.json(h)


def _chat_effective_options() -> dict:
    """
    Parâmetros efetivos do chat.

    Na aba Conversa, documentos e planilhas são usados automaticamente quando
    disponíveis. A aba Desenvolvimento pode sobrescrever via ``dev_chat_*``.
    """
    dev_override = st.session_state.get("dev_chat_override", False)
    use_rag = (
        bool(st.session_state.get("dev_chat_use_rag", True))
        if dev_override
        else index_ready()
    )
    use_olap = (
        bool(st.session_state.get("dev_chat_use_olap", True))
        if dev_override
        else has_ingested_tables()
    )
    return {
        "use_rag": use_rag and index_ready(),
        "use_olap": use_olap and has_ingested_tables(),
        "rag_top_k": int(st.session_state.get("dev_chat_rag_top_k", 6)),
        "max_tokens": int(st.session_state.get("dev_chat_max_tokens", DEFAULT_CHAT_MAX_TOKENS)),
        "max_history_turns": int(st.session_state.get("dev_chat_max_history_turns", 8)),
        "use_thinking": bool(st.session_state.get("dev_chat_use_thinking", env_enable_thinking_default())),
        "use_stream": bool(st.session_state.get("chat_use_stream", True)),
    }


def _tab_chat_dev_controls() -> None:
    """Controles avançados do chat (aba Desenvolvimento)."""
    st.subheader("Parâmetros do chat")
    st.caption(
        "Com *configuração manual*, os toggles substituem o roteador LLM. "
        "Sem isso, um classificador leve decide se cada mensagem consulta documentos/planilhas."
    )
    base, model, _ = llm_runtime_config()
    st.caption(f"Servidor: `{base}` · Modelo: `{model}`")

    st.toggle(
        "Usar configuração manual",
        key="dev_chat_override",
        help="Desligado: documentos e planilhas entram automaticamente quando existirem.",
    )

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.toggle(
            "Documentos",
            value=True,
            key="dev_chat_use_rag",
            disabled=not st.session_state.get("dev_chat_override"),
        )
    with c1:
        st.toggle(
            "Planilhas",
            value=True,
            key="dev_chat_use_olap",
            disabled=not st.session_state.get("dev_chat_override"),
        )
    with c2:
        st.toggle(
            "Raciocínio",
            value=env_enable_thinking_default(),
            key="dev_chat_use_thinking",
            disabled=not is_qwen35_model(model) or not st.session_state.get("dev_chat_override"),
        )
    with c3:
        st.toggle("Streaming", value=True, key="chat_use_stream")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.number_input("Trechos", 1, 20, 6, key="dev_chat_rag_top_k")
    with c5:
        st.number_input("Máx. tokens", 256, 32768, DEFAULT_CHAT_MAX_TOKENS, step=256, key="dev_chat_max_tokens")
    with c6:
        st.number_input("Turnos no histórico", 1, 30, 8, key="dev_chat_max_history_turns")


def _tab_chat() -> None:
    """
    Aba principal — conversa com o agente.

    O contexto de documentos/planilhas é injetado no system prompt quando
    disponível, sem expor parâmetros técnicos ao usuário final.
    """
    st.header("Conversa")
    st.caption("Pergunte sobre experimentos, insumos e resultados documentados no laboratório.")

    base, model, api_key = llm_runtime_config()
    opts = _chat_effective_options()
    use_rag = opts["use_rag"]
    use_olap = opts["use_olap"]
    rag_top_k = opts["rag_top_k"]
    max_tokens = opts["max_tokens"]
    max_history_turns = opts["max_history_turns"]
    use_thinking = opts["use_thinking"]
    use_stream = opts["use_stream"]

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    status_a, status_b, actions = st.columns([2, 2, 1])
    with status_a:
        if index_ready():
            st.caption("Documentos prontos para consulta")
        else:
            st.caption("Atualize os documentos na aba **Documentos**")
    with status_b:
        if has_ingested_tables():
            st.caption("Planilhas disponíveis para consulta")
        else:
            st.caption("Nenhuma planilha indexada ainda")
    with actions:
        if st.button("Limpar", key="btn_clear_chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    chat_box = st.container(height=460, border=True)
    prompt = st.chat_input("Escreva sua pergunta…")

    if not prompt:
        with chat_box:
            if not st.session_state.chat_messages:
                st.markdown(
                    "_Nenhuma mensagem ainda. Use o campo abaixo para começar a conversa._"
                )
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        show_dev_details = bool(st.session_state.get("dev_chat_override"))

        client = OpenAI(base_url=base, api_key=api_key, timeout=120.0)

        history_before = st.session_state.chat_messages[:-1]
        _route_kwargs = dict(
            message=prompt,
            history=history_before,
            client=client,
            model=model,
            documents_enabled=use_rag and index_ready(),
            spreadsheets_enabled=use_olap and has_ingested_tables(),
            manual_override=show_dev_details,
        )
        needs_llm_router = not show_dev_details and classification_needs_llm(prompt)
        if needs_llm_router:
            with st.spinner("Analisando pergunta…"):
                route = resolve_chat_routes(**_route_kwargs)
        else:
            route = resolve_chat_routes(**_route_kwargs)
        run_rag = route.use_documents
        run_olap = route.use_spreadsheets
        if show_dev_details:
            st.caption(
                f"Roteador ({route.source}): documentos={'sim' if run_rag else 'não'} · "
                f"planilhas={'sim' if run_olap else 'não'}"
            )

        system_prompt = CHAT_SYSTEM_PROMPT
        if run_rag:
            hits = rag_semantic_search(prompt, int(rag_top_k))
            ctx = format_context_for_llm(hits)
            if ctx:
                system_prompt += (
                    "\n\n### Contexto recuperado dos documentos do laboratório\n"
                    + ctx
                    + "\n\nBaseie respostas sobre ensaios neste contexto quando for relevante; "
                    "cite projeto e arquivo como nos cabeçalhos [n]. Se o contexto não ajudar, diga claramente."
                )
            else:
                system_prompt += (
                    "\n\n(Nenhum trecho relevante foi recuperado do índice para esta pergunta — "
                    "não invente dados de ensaios.)"
                )

        olap_result = None
        if run_olap:
            with st.spinner("Consultando planilhas…"):
                olap_result = run_nl_olap_query(prompt, client=client, model=model)
            if olap_result.ok and olap_result.context_for_llm:
                system_prompt += "\n\n" + olap_result.context_for_llm
                system_prompt += (
                    "\n\nAo responder com dados tabulares acima, cite projeto (_project_id) "
                    "e arquivo (_source_file). Não invente valores fora do resultado SQL."
                )
                if show_dev_details:
                    with st.expander("SQL e dados (dev)", expanded=False):
                        if olap_result.sql:
                            st.code(olap_result.sql, language="sql")
                        if olap_result.dataframe is not None:
                            st.dataframe(
                                olap_result.dataframe,
                                use_container_width=True,
                                hide_index=True,
                            )
            elif olap_result.error:
                if show_dev_details:
                    st.warning(f"Planilhas: {olap_result.error}")
                    with st.expander("Detalhe da consulta (dev)", expanded=False):
                        if olap_result.sql:
                            st.code(olap_result.sql, language="sql")
                        if olap_result.raw_llm_response:
                            st.code(olap_result.raw_llm_response, language="text")
                system_prompt += (
                    f"\n\n### Dados tabulares (falha na consulta)\n{olap_result.error}\n"
                    "Explique o problema ao usuário sem inventar números de planilhas."
                )

        history_trimmed = _trim_chat_history(
            st.session_state.chat_messages, int(max_history_turns)
        )
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages += [
            {
                "role": m["role"],
                "content": sanitize_history_message(
                    m["role"], m["content"], model_id=model
                ),
            }
            for m in history_trimmed
        ]

        chat_profile = select_chat_profile(model_id=model, use_thinking=use_thinking and is_qwen35_model(model))

        with chat_box:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            with st.chat_message("assistant"):
                try:
                    if use_stream:
                        stream = create_chat_completion(
                            client,
                            messages=api_messages,
                            model=model,
                            profile=chat_profile,
                            max_tokens=int(max_tokens),
                            stream=True,
                        )

                        def _token_stream():
                            for chunk in stream:
                                if not chunk.choices:
                                    continue
                                delta = chunk.choices[0].delta
                                if delta and delta.content:
                                    yield delta.content

                        full_text = st.write_stream(_token_stream)
                        answer = (
                            strip_thinking_blocks(full_text or "")
                            if is_qwen35_model(model)
                            else (full_text or "")
                        )
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        with st.spinner("Gerando resposta…"):
                            completion = create_chat_completion(
                                client,
                                messages=api_messages,
                                model=model,
                                profile=chat_profile,
                                max_tokens=int(max_tokens),
                                stream=False,
                            )
                        raw = (completion.choices[0].message.content or "").strip()
                        text = (
                            strip_thinking_blocks(raw)
                            if is_qwen35_model(model)
                            else raw
                        )
                        st.markdown(text)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": text}
                        )
                except Exception as exc:
                    st.error(f"Não foi possível obter resposta do assistente: {exc}")
                    if (
                        st.session_state.chat_messages
                        and st.session_state.chat_messages[-1].get("role") == "user"
                    ):
                        st.session_state.chat_messages.pop()
                    st.caption(
                        "A última mensagem foi removida; verifique o servidor de IA e tente de novo."
                    )


def _tab_desenvolvimento(
    root: Path,
    root_ok: bool,
    root_msg: str,
    scans: list[ProjectScan] | None,
) -> None:
    """Aba única para time técnico — sub-abas com ferramentas de diagnóstico e tuning."""
    st.header("Desenvolvimento")
    st.caption("Ferramentas para administradores e desenvolvedores. O usuário final não precisa desta aba.")

    dev_tabs = st.tabs(
        [
            "Visão geral",
            "Parâmetros do chat",
            "Busca semântica",
            "Índice vetorial",
            "Planilhas",
            "Diagnóstico",
        ]
    )
    with dev_tabs[0]:
        _tab_inicio()
        st.divider()
        st.subheader("Escaneamento técnico")
        default_hint = os.environ.get(ENV_PROJETOS_ROOT, "").strip() or str(projetos_root_from_env())
        st.text_input(
            "Raiz dos projetos (override)",
            key="path_override_dev",
            placeholder=default_hint,
            help="Mesmo campo da aba Documentos; use um dos dois.",
        )
        st.checkbox("Calcular SHA-256 no escaneamento", key="compute_hashes", value=True)
        st.text_input(
            "Extensões (vírgula)",
            key="ext_input",
            value=", ".join(sorted(DEFAULT_DOCUMENT_EXTENSIONS)),
        )
        if st.button("Escanear (dev)", key="btn_scan_dev"):
            _run_project_scan()
    with dev_tabs[1]:
        _tab_chat_dev_controls()
    with dev_tabs[2]:
        _tab_rag_dev()
    with dev_tabs[3]:
        _tab_indexacao_rag(scans)
    with dev_tabs[4]:
        _tab_olap(scans)
    with dev_tabs[5]:
        _tab_diagnostico(root, root_ok, root_msg)
        st.divider()
        _tab_fontes(root, root_ok, root_msg)


def _tab_olap(scans: list[ProjectScan] | None) -> None:
    """Aba 5 — DuckDB: ingestão de planilhas e catálogo."""
    st.header("Dados tabulares (OLAP)")
    st.caption(
        f"Ingestão automática de **{', '.join(sorted(TABULAR_EXTENSIONS))}** ao escanear pastas "
        "(aba **Documentos**). Na **Conversa**, planilhas são consultadas automaticamente."
    )

    status = olap_status()
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- Biblioteca DuckDB: `{status['library_version']}`")
        st.write(f"- Diretório de dados: `{status['data_root']}`")
    with c2:
        st.write(f"- Arquivo: `{status['database_path']}`")
        st.write(
            f"- Planilhas ingeridas: **{status['ingested_tables']}** tabela(s), "
            f"**{status['ingested_rows']}** linha(s) no total"
        )

    if scans is None:
        st.warning("Faça **Escanear pastas** na aba **Documentos** (ou em Desenvolvimento) para ingerir planilhas.")
    else:
        if st.button("Sincronizar planilhas com DuckDB agora", type="primary", key="btn_olap_sync"):
            root = _root_from_session()
            with st.spinner("Carregando CSV/XLSX/XLSM nos projetos…"):
                tabular_scans = scan_all_projects(
                    root,
                    extensions=TABULAR_EXTENSIONS,
                    compute_hashes=bool(st.session_state.get("compute_hashes")),
                )
                stats = sync_tabular_from_scans(tabular_scans)
            st.session_state["last_olap_sync"] = stats
            if getattr(stats, "aborted_empty_scan", False):
                st.error(
                    "Sincronização **abortada**: o escaneamento não encontrou "
                    "planilhas, mas o DuckDB tem tabelas. Nenhuma tabela foi "
                    "apagada. Confira a raiz de projetos na barra lateral."
                )
            else:
                st.success(
                    f"{stats.tables_touched} tabela(s) criada(s)/atualizada(s), "
                    f"{stats.tables_removed} removida(s), "
                    f"{stats.indexes_ensured} índice(s) de metadados garantidos."
                )
            if stats.errors:
                with st.expander("Erros / avisos da ingestão"):
                    st.code("\n".join(stats.errors[:30]), language="text")
            st.rerun()

    catalog = list_ingested_tables()
    st.subheader("Catálogo de tabelas (planilhas)")
    if catalog.empty:
        st.info(
            "Nenhuma planilha no banco. Confirme que existem arquivos "
            f"{', '.join(sorted(TABULAR_EXTENSIONS))} nos projetos e escaneie de novo."
        )
    else:
        st.dataframe(catalog, use_container_width=True, hide_index=True)

    with st.expander("Catálogo de schema (enviado ao LLM)"):
        schema_text = build_schema_catalog_text()
        if schema_text.startswith("("):
            st.warning(schema_text)
        else:
            st.code(schema_text, language="text")

    with st.expander("Dados de demonstração (opcional)"):
        if st.button("Criar / recriar demo", key="btn_olap_seed"):
            seed_demo_data(force=True)
            st.rerun()
        if status["demo_ready"]:
            st.dataframe(demo_aggregation(), use_container_width=True, hide_index=True)


def _tab_diagnostico(root: Path, root_ok: bool, root_msg: str) -> None:
    """Aba 6 — Versões, caminhos, status do índice e teste de conectividade LLM."""
    st.header("Diagnóstico")
    st.subheader("Runtime")
    st.write(f"- Python `{sys.version.split()[0]}` em **{platform.system()}**")
    st.write(f"- Streamlit `{st.__version__}`")
    if running_inside_docker():
        st.success("Processo dentro de **contêiner Docker** (`/.dockerenv` presente).")
    else:
        st.write("Fora do Docker (modo desenvolvimento local).")

    st.subheader("Caminhos")
    st.code(f"Raiz resolvida (UI): {root}\nVálida: {root_ok} — {root_msg}", language="text")
    st.markdown(
        "No Compose, volumes típicos: `/data/projetos` (ingestão), `/data/txtai`, `/data/duckdb`, `/data/sqlite`."
    )

    st.subheader("txtai / RAG")
    st.write(f"- `ASSISTENTE_TXTAI_DIR` / dados: `{txtai_data_root()}`")
    st.write(f"- Índice: `{txtai_index_path()}`")
    st.write(f"- Índice pronto: **{'sim' if index_ready() else 'não'}** · modelo: `{EMBEDDING_MODEL_ID}`")

    st.subheader("DuckDB / OLAP")
    olap = olap_status()
    st.write(f"- `ASSISTENTE_DUCKDB_DIR` / dados: `{duckdb_data_root()}`")
    st.write(f"- Arquivo: `{duckdb_database_path()}` · versão lib: `{duckdb_library_version()}`")
    st.write(
        f"- Banco no disco: **{'sim' if olap['database_exists'] else 'não'}** · "
        f"planilhas: **{olap['ingested_tables']}** tabela(s) · "
        f"demo: **{'sim' if olap['demo_ready'] else 'não'}**"
    )
    if st.button("Testar conexão DuckDB (SELECT 1)", key="btn_duckdb_ping"):
        ok, detail = check_duckdb()
        if ok:
            st.success(detail)
        else:
            st.error(detail)

    st.subheader("LM Studio (API compatível com OpenAI)")
    llm_base = get_llm_base_url_raw()
    llm_model = get_llm_model()
    resolved_base, _, _ = llm_runtime_config()
    env_base = os.environ.get("LLM_BASE_URL", "").strip()
    env_model = os.environ.get("LLM_MODEL", "").strip()
    st.text_input(
        "LLM_BASE_URL (efetiva)",
        value=llm_base,
        disabled=True,
        key="dev_diag_llm_base_url",
    )
    if env_base:
        st.caption("Origem: variável de ambiente `LLM_BASE_URL`.")
    else:
        st.caption(f"Origem: padrão do projeto (`{DEFAULT_LLM_BASE_URL}`).")
    if resolved_base != llm_base.rstrip("/"):
        st.caption(f"URL usada nas chamadas (com `/v1`): `{resolved_base}`")
    st.text_input(
        "LLM_MODEL (efetivo)",
        value=llm_model,
        disabled=True,
        key="dev_diag_llm_model",
    )
    if env_model:
        st.caption("Origem: variável de ambiente `LLM_MODEL`.")
    else:
        st.caption(f"Origem: padrão do projeto (`{DEFAULT_LLM_MODEL}`).")
    if st.button("Testar GET /v1/models (timeout 5s)", key="btn_llm_ping"):
        ok, detail = _check_openai_compatible_models(llm_base)
        if ok:
            st.success("Servidor respondeu. Modelos (amostra):")
            st.code(detail, language="text")
        else:
            st.error(detail)


# ── Ponto de entrada ─────────────────────────────────────────────────────────
# O Streamlit re-executa este arquivo inteiro a cada interação do usuário.
# Funções cacheadas (_txtai_backend_cached) e o st.session_state preservam
# estado entre execuções dentro da mesma sessão de browser.

_render_sidebar()

root = _root_from_session()
root_ok, root_msg = validate_projetos_root(root)

st.title("Assistente Lab")
st.caption("Assistente para documentos e dados do laboratório — uso local e offline.")

tabs = st.tabs(["Conversa", "Documentos", "Desenvolvimento"])

scans: list[ProjectScan] | None = st.session_state.get("last_scan")

with tabs[0]:
    _tab_chat()
with tabs[1]:
    _tab_documentos(root, root_ok, root_msg)
with tabs[2]:
    _tab_desenvolvimento(root, root_ok, root_msg, scans)
