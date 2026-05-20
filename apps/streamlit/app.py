"""
Assistente de laboratório — interface Streamlit (MVP alinhado ao playbook biotech).

Ponto de entrada único da aplicação. Orquestra todos os subsistemas locais:

- ``projects_loader``: descobre projetos e arquivos nos volumes montados.
- ``rag``: extração, chunking, índice txtai e busca semântica.
- ``OpenAI`` (cliente SDK): conversa com o LM Studio rodando no host via API
  compatível com OpenAI.

Abas da UI
----------
0. Início — visão geral do produto e fases do playbook.
1. Fontes e inventário — escaneamento e tabela de documentos por projeto.
2. Indexação RAG — construção/atualização do índice txtai com controles de chunk.
3. Teste RAG (dev) — busca semântica direta sem passar pelo LLM.
4. Chat — conversa com RAG + LM Studio; streaming opcional.
5. OLAP — DuckDB em volume; dados demo e agregações read-only.
6. Diagnóstico — versões, caminhos, status txtai, teste de conectividade LLM.
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
from dotenv import load_dotenv
from openai import OpenAI

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
    check_duckdb,
    demo_aggregation,
    demo_detail,
    duckdb_data_root,
    duckdb_database_path,
    duckdb_library_version,
    olap_status,
    seed_demo_data,
)
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

load_dotenv()


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
    page_title="Assistente Lab — Agente documental",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers de configuração e conexão ───────────────────────────────────────

def _root_from_session() -> Path:
    """Retorna a raiz de projetos considerando o override digitado na barra lateral."""
    override = (st.session_state.get("path_override") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return projetos_root_from_env()


def _parse_extensions(ext_input: str) -> frozenset[str]:
    """Converte a string de extensões da UI em frozenset normalizado (ex.: ``'.docx'``)."""
    parts = [p.strip().lower() for p in ext_input.split(",") if p.strip()]
    exts = frozenset(p if p.startswith(".") else f".{p}" for p in parts)
    return exts if exts else DEFAULT_DOCUMENT_EXTENSIONS


def _normalize_openai_base_url(url: str) -> str:
    """
    Garante que a URL termine com ``/v1``, exigido pelo SDK OpenAI.

    O LM Studio aceita chamadas sem o sufixo, mas o SDK Python da OpenAI
    concatena os endpoints (``/chat/completions``, etc.) diretamente a
    ``base_url``, portanto ``/v1`` precisa estar presente.
    """
    u = url.strip().rstrip("/")
    if not u:
        return u
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def _llm_runtime_config() -> tuple[str, str, str]:
    """
    Retorna ``(base_url_com_v1, model_id, api_key)`` lidos do ambiente.

    A ``api_key`` cai em ``"lm-studio"`` quando nenhuma variável está definida;
    o LM Studio aceita qualquer valor não vazio nesse campo.
    """
    raw_base = os.environ.get("LLM_BASE_URL", "").strip()
    base = _normalize_openai_base_url(raw_base)
    model = os.environ.get("LLM_MODEL", "").strip()
    key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
        or "lm-studio"
    )
    return base, model, key


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
    b = _normalize_openai_base_url(base_url.strip())
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

def _render_sidebar() -> None:
    """Renderiza os controles de escaneamento na barra lateral."""
    st.sidebar.markdown("### Fontes locais")
    st.sidebar.caption(
        "Pastas em volume (Docker) alimentam o inventário e, em seguida, a **indexação RAG**. "
        "Cada pasta **filha direta** da raiz = um projeto (planning/results ficam no mesmo projeto)."
    )
    default_hint = os.environ.get(ENV_PROJETOS_ROOT, "").strip() or str(projetos_root_from_env())
    st.sidebar.text_input(
        "Raiz dos projetos (opcional)",
        key="path_override",
        placeholder=default_hint,
        help="Vazio = ASSISTENTE_PROJETOS_DIR ou padrão do ambiente/Docker.",
    )
    st.sidebar.checkbox(
        "Calcular SHA-256 no escaneamento",
        key="compute_hashes",
        value=True,
        help="Acelera a indexação incremental (compara hash sem reler o arquivo). Pode ser lento no escaneamento.",
    )
    st.sidebar.text_input(
        "Extensões (vírgula)",
        key="ext_input",
        value=", ".join(sorted(DEFAULT_DOCUMENT_EXTENSIONS)),
        help="Filtro do inventário e da futura ingestão para embeddings.",
    )
    if st.sidebar.button("Escanear pastas agora", type="primary"):
        root = _root_from_session()
        ok, msg = validate_projetos_root(root)
        if not ok:
            st.session_state["sidebar_scan_error"] = msg
        else:
            st.session_state.pop("sidebar_scan_error", None)
            ext_input = str(st.session_state.get("ext_input") or "")
            exts = _parse_extensions(ext_input)
            compute_hashes = bool(st.session_state.get("compute_hashes"))
            with st.spinner("Lendo árvore de diretórios…"):
                scans = scan_all_projects(
                    root,
                    extensions=exts,
                    compute_hashes=compute_hashes,
                )
            st.session_state["last_scan"] = scans
            st.session_state["last_root"] = str(root)
            st.session_state["last_exts"] = exts
            st.rerun()
    err = st.session_state.get("sidebar_scan_error")
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


def _tab_fontes(root: Path, root_ok: bool, root_msg: str) -> None:
    """Aba 1 — Tabela do inventário de documentos por projeto."""
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
        st.markdown("Use **Escanear pastas agora** na barra lateral para carregar o inventário.")
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
    selected = st.multiselect("Filtrar por projeto", options=options, default=options)
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
        st.warning("Faça um escaneamento na aba **Fontes e inventário** (barra lateral) antes de indexar.")
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
        )
    with c2:
        overlap = st.number_input(
            "Sobreposição entre chunks",
            min_value=0,
            max_value=800,
            value=80,
            step=10,
        )
    with c3:
        batch_size = st.number_input(
            "Tamanho do lote (index txtai)",
            min_value=8,
            max_value=512,
            value=64,
            step=8,
        )

    max_doc = st.number_input(
        "Limite de caracteres por arquivo (extração)",
        min_value=50_000,
        max_value=10_000_000,
        value=2_000_000,
        step=100_000,
        help="Evita carregar PDF/planilhas enormes inteiros na memória.",
    )

    # Desmarca automaticamente após a primeira indexação bem-sucedida, pois o
    # modo incremental já é o comportamento desejado em execuções subsequentes.
    replace_default = not index_ready()
    replace = st.checkbox(
        "Substituir índice existente",
        value=replace_default,
        help="Apaga índice e manifesto e reconstrói tudo. Desmarque para **reindexação incremental por hash**.",
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
        st.warning("Construa um índice na aba **Indexação RAG** antes de testar buscas.")
        return

    q = st.text_input("Consulta semântica", placeholder="Ex.: validade do reagente X no projeto ELISA")
    top_k = st.slider("Top-K", min_value=1, max_value=25, value=8)

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


def _tab_chat() -> None:
    """
    Aba 4 — Chat com o LLM com injeção opcional de contexto RAG.

    Estratégia de RAG
    -----------------
    O contexto recuperado é injetado no **system prompt** (não na mensagem do
    usuário). Isso preserva o papel e as restrições definidas em
    ``CHAT_SYSTEM_PROMPT`` e sinaliza claramente ao modelo que o contexto é
    evidência documental, não instrução do usuário.

    Erro na chamada LLM
    -------------------
    Em caso de exceção, a última mensagem do usuário é removida do histórico
    em sessão. Isso permite que o usuário reenvie a mensagem após corrigir a
    configuração do servidor, sem duplicar a pergunta no histórico.
    """
    st.header("Chat com o agente")
    st.caption(
        "Conversa com o modelo **LM Studio** (API OpenAI-compatible). "
        "Opcionalmente injeta trechos recuperados pelo índice **txtai** (RAG)."
    )

    base, model, api_key = _llm_runtime_config()
    if not base or not model:
        st.warning(
            "Defina **`LLM_BASE_URL`** (ex.: `http://172.21.64.1:1234` ou `http://host.docker.internal:1234`) "
            "e **`LLM_MODEL`** (ex.: `qwen/qwen3.5-9b`) no `.env` / ambiente do Compose."
        )
        st.info("Use a aba **Diagnóstico** para testar `GET /v1/models` antes de conversar.")
        return

    st.caption(f"Servidor: `{base}` · Modelo: `{model}`")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    r1, r2, r3, r4 = st.columns([3, 1, 2, 2])
    with r1:
        use_rag = st.toggle(
            "Usar RAG (txtai)",
            value=index_ready(),
            disabled=not index_ready(),
            key="chat_use_rag",
            help="Recupera trechos do índice com a pergunta atual e envia como contexto ao modelo.",
        )
    with r2:
        rag_top_k = st.number_input(
            "Trechos RAG",
            min_value=1,
            max_value=20,
            value=6,
            key="chat_rag_top_k",
            disabled=not use_rag,
        )
    with r3:
        max_tokens = st.number_input(
            "Max. tokens (resposta)",
            min_value=128,
            max_value=8192,
            value=1024,
            step=128,
            key="chat_max_tokens",
            help="Limite de tokens gerados pelo modelo. Evita respostas truncadas em modelos com default conservador.",
        )
    with r4:
        use_stream = st.toggle("Resposta em streaming", value=True, key="chat_use_stream")

    col_hist, _ = st.columns([2, 3])
    with col_hist:
        max_history_turns = st.number_input(
            "Turnos no histórico enviado ao modelo",
            min_value=1,
            max_value=30,
            value=8,
            step=1,
            key="chat_max_history_turns",
            help="Limita quantos pares de mensagens (user + assistente) são incluídos no prompt. "
                 "Reduz o tamanho total do contexto enviado e evita exceder a janela do modelo.",
        )

    if use_rag and not index_ready():
        st.warning("Índice txtai indisponível. Construa em **Indexação RAG** ou desative o RAG.")

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Limpar conversa", key="btn_clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Envie uma mensagem para o modelo…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        client = OpenAI(base_url=base, api_key=api_key, timeout=120.0)

        system_prompt = CHAT_SYSTEM_PROMPT
        if use_rag and index_ready():
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

        history_trimmed = _trim_chat_history(
            st.session_state.chat_messages, int(max_history_turns)
        )
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages += [
            {"role": m["role"], "content": m["content"]}
            for m in history_trimmed
        ]

        with st.chat_message("assistant"):
            try:
                if use_stream:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=api_messages,
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
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": full_text or ""}
                    )
                else:
                    with st.spinner("Gerando resposta…"):
                        completion = client.chat.completions.create(
                            model=model,
                            messages=api_messages,
                            max_tokens=int(max_tokens),
                            stream=False,
                        )
                    text = (completion.choices[0].message.content or "").strip()
                    st.markdown(text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": text})
            except Exception as exc:
                st.error(f"Erro ao chamar o LM Studio: {exc}")
                if (
                    st.session_state.chat_messages
                    and st.session_state.chat_messages[-1].get("role") == "user"
                ):
                    st.session_state.chat_messages.pop()
                st.caption(
                    "A última mensagem sua foi removida do histórico; "
                    "ajuste o servidor ou o modelo e tente de novo."
                )


def _tab_olap() -> None:
    """Aba 5 — DuckDB em volume: seed demo e agregação read-only."""
    st.header("Dados tabulares (OLAP)")
    st.caption(
        "Motor **DuckDB** em arquivo no volume persistente. "
        "Esta fase traz um *olá mundo*: tabela demo e agregação por projeto."
    )

    status = olap_status()
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- Biblioteca DuckDB: `{status['library_version']}`")
        st.write(f"- Diretório de dados: `{status['data_root']}`")
    with c2:
        st.write(f"- Arquivo: `{status['database_path']}`")
        st.write(
            f"- Banco criado: **{'sim' if status['database_exists'] else 'não'}** · "
            f"demo: **{'sim' if status['demo_ready'] else 'não'}**"
        )

    col_seed, col_refresh = st.columns(2)
    with col_seed:
        if st.button("Criar / recriar dados de demonstração", key="btn_olap_seed"):
            created = seed_demo_data(force=True)
            if created:
                st.success(f"Tabela `{DEMO_TABLE}` populada com linhas de exemplo.")
            else:
                st.info("Dados demo já existiam; use recriar para substituir.")
            st.rerun()
    with col_refresh:
        if st.button("Atualizar painéis", key="btn_olap_refresh"):
            st.rerun()

    if not status["demo_ready"]:
        st.warning(
            "Nenhum dado demo ainda. Clique em **Criar / recriar dados de demonstração** "
            "para materializar o banco no volume (persiste entre reinícios do contêiner)."
        )
        return

    st.subheader("Agregação por projeto (read-only)")
    st.dataframe(demo_aggregation(), use_container_width=True, hide_index=True)

    with st.expander("Linhas brutas da tabela demo"):
        st.dataframe(demo_detail(), use_container_width=True, hide_index=True)

    st.caption(
        "Próximas fases: materializar resumos da ingestão de documentos neste mesmo arquivo DuckDB."
    )


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
        f"tabela demo: **{'sim' if olap['demo_ready'] else 'não'}**"
    )
    if st.button("Testar conexão DuckDB (SELECT 1)", key="btn_duckdb_ping"):
        ok, detail = check_duckdb()
        if ok:
            st.success(detail)
        else:
            st.error(detail)

    st.subheader("LM Studio (API compatível com OpenAI)")
    llm_base = os.environ.get("LLM_BASE_URL", "").strip()
    llm_model = os.environ.get("LLM_MODEL", "").strip()
    resolved_base = _normalize_openai_base_url(llm_base) if llm_base else ""
    st.text_input("LLM_BASE_URL (somente leitura na UI)", value=llm_base or "(não definido)", disabled=True)
    if resolved_base and resolved_base != llm_base.strip().rstrip("/"):
        st.caption(f"URL efetiva usada pelo app (com `/v1`): `{resolved_base}`")
    st.text_input("LLM_MODEL", value=llm_model or "(não definido)", disabled=True)
    if llm_base:
        if st.button("Testar GET /v1/models (timeout 5s)", key="btn_llm_ping"):
            ok, detail = _check_openai_compatible_models(llm_base)
            if ok:
                st.success("Servidor respondeu. Modelos (amostra):")
                st.code(detail, language="text")
            else:
                st.error(detail)
    else:
        st.warning("Defina `LLM_BASE_URL` no ambiente (ex.: `http://host.docker.internal:1234/v1` no Docker).")


# ── Ponto de entrada ─────────────────────────────────────────────────────────
# O Streamlit re-executa este arquivo inteiro a cada interação do usuário.
# Funções cacheadas (_txtai_backend_cached) e o st.session_state preservam
# estado entre execuções dentro da mesma sessão de browser.

_render_sidebar()

root = _root_from_session()
root_ok, root_msg = validate_projetos_root(root)

st.title("Assistente Lab")
st.caption("Agente documental local — inventário por projeto, RAG (txtai) e IA (LM Studio), conforme playbook MVP.")

tabs = st.tabs(
    [
        "Início",
        "Fontes e inventário",
        "Indexação RAG",
        "Teste RAG (dev)",
        "Chat",
        "OLAP",
        "Diagnóstico",
    ]
)

scans: list[ProjectScan] | None = st.session_state.get("last_scan")

with tabs[0]:
    _tab_inicio()
with tabs[1]:
    _tab_fontes(root, root_ok, root_msg)
with tabs[2]:
    _tab_indexacao_rag(scans)
with tabs[3]:
    _tab_rag_dev()
with tabs[4]:
    _tab_chat()
with tabs[5]:
    _tab_olap()
with tabs[6]:
    _tab_diagnostico(root, root_ok, root_msg)
