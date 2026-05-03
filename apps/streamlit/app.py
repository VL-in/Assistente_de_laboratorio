"""
Assistente de laboratório — UI Streamlit (MVP alinhado ao playbook biotech).

Orquestra fontes locais segmentadas por projeto (`projects_loader`), prepara terreno para
indexação RAG (txtai), chat com citações (LM Studio) e OLAP (DuckDB), conforme fases do plano.
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

load_dotenv()

st.set_page_config(
    page_title="Assistente Lab — Agente documental",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _root_from_session() -> Path:
    override = (st.session_state.get("path_override") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return projetos_root_from_env()


def _parse_extensions(ext_input: str) -> frozenset[str]:
    parts = [p.strip().lower() for p in ext_input.split(",") if p.strip()]
    exts = frozenset(p if p.startswith(".") else f".{p}" for p in parts)
    return exts if exts else DEFAULT_DOCUMENT_EXTENSIONS


def _check_openai_compatible_models(base_url: str, *, timeout_s: float = 5.0) -> tuple[bool, str]:
    """GET {base}/models — compatível com LM Studio no host."""
    b = base_url.strip().rstrip("/")
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
        return False, f"Rede/URL: {e.reason!r}"
    except TimeoutError:
        return False, f"Tempo esgotado ({timeout_s}s)."
    except OSError as e:
        return False, str(e)


def _render_sidebar() -> None:
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
        "Calcular SHA-256 (versionamento)",
        key="compute_hashes",
        help="Útil para reindexação incremental; pode ser lento.",
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


def _tab_inicio() -> None:
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
            "- **Fase 1** — Diagnóstico, LM Studio, DuckDB olá mundo, metadados.\n"
            "- **Fase 2** — Parsing, hash, página de status de ingestão.\n"
            "- **Fase 3** — txtai + chat com citações.\n"
            "- **Fase 4** — RBAC, auditoria, guardrails."
        )


def _tab_fontes(root: Path, root_ok: bool, root_msg: str) -> None:
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
    st.header("Indexação RAG")
    st.caption(
        "Próxima entrega (playbook): parsing → chunks com metadados (projeto, arquivo, aba) → **txtai** em volume persistente."
    )
    if scans is None:
        st.warning("Faça um escaneamento na aba **Fontes e inventário** (barra lateral) antes de planejar a indexação.")
        return
    if not scans:
        st.info("Inventário vazio — nada para indexar.")
        return
    total = sum(s.file_count for s in scans)
    st.success(f"Há **{total}** arquivo(s) em **{len(scans)}** projeto(s) elegíveis ao pipeline.")
    st.markdown(
        "Use `documents_by_project(scans)` do `projects_loader` para iterar por projeto sem vazar caminhos cruzados. "
        "Cada `ScannedFile` traz `absolute_path`, `relative_path` e `project_id` para anexar ao vetor store."
    )
    with st.expander("Contrato sugerido para o índice (txtai / metadados)"):
        st.json(
            {
                "exemplo_metadados_por_chunk": {
                    "project_id": "253 - ELISA indireto Dengue",
                    "source_relative": "results/amostras_dengue.xlsx",
                    "content_hash_sha256": "<opcional>",
                }
            }
        )


def _tab_chat() -> None:
    st.header("Chat com o agente")
    st.caption("Fase 3 do playbook: pergunta → recuperação (txtai) → LM Studio → resposta com **citação** de fonte.")
    st.info("Interface de chat em construção. Enquanto isso, valide o LM Studio na aba **Diagnóstico**.")


def _tab_olap() -> None:
    st.header("Dados tabulares (OLAP)")
    st.caption("Playbook: **DuckDB** em arquivo em volume; consultas read-only a partir desta UI.")
    st.info("Conexão DuckDB e painéis de exemplo entram na Fase 1/2 do plano.")


def _tab_diagnostico(root: Path, root_ok: bool, root_msg: str) -> None:
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

    st.subheader("LM Studio (API compatível com OpenAI)")
    llm_base = os.environ.get("LLM_BASE_URL", "").strip()
    llm_model = os.environ.get("LLM_MODEL", "").strip()
    st.text_input("LLM_BASE_URL (somente leitura na UI)", value=llm_base or "(não definido)", disabled=True)
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


# --- Layout principal ---

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
    _tab_chat()
with tabs[4]:
    _tab_olap()
with tabs[5]:
    _tab_diagnostico(root, root_ok, root_msg)
