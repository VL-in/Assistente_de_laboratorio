"""
Assistente de laboratório — interface Streamlit (MVP alinhado ao playbook biotech).

Ponto de entrada único da aplicação. Orquestra todos os subsistemas locais:

- ``projects_loader``: descobre projetos e arquivos nos volumes montados.
- ``rag``: extração, chunking, índice txtai e busca semântica.
- ``OpenAI`` (cliente SDK): conversa com o OpenRouter (API compatível com
  OpenAI) usando a chave em ``OPENROUTER_API_KEY``.

Abas da UI (usuário final)
--------------------------
0. Conversa — chat com o agente (documentos e planilhas usados automaticamente).
1. Documentos — escanear pastas, inventário e atualização da base de conhecimento.
2. ML tradicional — AutoML (FLAML), dicionário de colunas, exportação .pkl e predição.

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
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from llm_config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    get_llm_api_key,
    get_llm_base_url_raw,
    get_llm_model,
    is_openrouter_endpoint,
    llm_runtime_config,
    normalize_openai_base_url,
    openrouter_default_headers,
)
from observability import (
    chat_observation_context,
    langfuse_status,
    update_chat_trace_output,
    update_chat_trace_route,
)
from agents import (
    CrewRunResult,
    anonymize_messages_for_external,
    anonymize_pii,
    parallel_tools_enabled,
    run_crew_chat,
    sanitize_model_output,
    scan_user_input,
    trace_handoff_enabled,
)
from qwen35_inference import (
    chat_history_chars_per_message,
    chat_max_history_turns,
    chat_max_tokens,
    create_chat_completion,
    effective_chat_limits,
    env_enable_thinking_default,
    is_qwen35_model,
    iter_stream_answer_text,
    select_chat_profile,
    strip_thinking_blocks,
)
from projects_loader import (
    DEFAULT_DOCUMENT_EXTENSIONS,
    ENV_PROJETOS_ROOT,
    ProjectScan,
    documents_by_project,
    filter_scans_by_extensions,
    projetos_root_from_env,
    running_inside_docker,
    scan_all_projects,
    validate_projetos_root,
)
from olap import (
    TABULAR_EXTENSIONS,
    check_duckdb,
    demo_aggregation,
    duckdb_data_root,
    duckdb_database_path,
    duckdb_library_version,
    has_ingested_tables,
    list_ingested_tables,
    olap_status,
    seed_demo_data,
    sync_tabular_from_scans,
)
from olap.schema_catalog import build_schema_catalog_text
from ml.paths import chat_ml_model_available, chat_ml_model_path, ml_models_root
from ml.chat_infer import load_chat_model_bundle, ml_inference_status_message
from ml.kaggle_sources import kaggle_cache_root
from ml.training import flaml_available
from rag import (
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    EMBEDDING_MODEL_ID,
    RERANKER_MODEL_ID,
    RerankerLoadResult,
    build_index,
    default_retrieve_k,
    env_hybrid_enabled,
    env_rerank_enabled,
    hybrid_dense_weight,
    index_mtime,
    index_ready,
    load_reranker_safe,
    manifest_exists,
    manifest_path,
    rerank_hits,
    search_with_backend,
    txtai_data_root,
    txtai_index_path,
)
from ml.ui import render_ml_tab

# ── System prompts ───────────────────────────────────────────────────────────
# Os prompts do chat (geral e rota ML) vivem em ``agents/synthesizer.py``,
# único responsável por montar o contexto que vai ao LLM. Para editar o tom
# ou as constraints do assistente, vá lá.


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
    Streamlit, evitando recarregar o índice vetorial a cada interação.
    Os vetores são calculados pelo sentence-transformers in-process (multilingual-e5-small).

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


@st.cache_resource(show_spinner="Carregando modelo ML…")
def _ml_bundle_cached(model_path: str, mtime_key: float):
    """Carrega o ``ModelBundle`` do disco; invalida quando o .pkl muda."""
    if mtime_key <= 0:
        return None
    return load_chat_model_bundle(Path(model_path))


@st.cache_resource(show_spinner="Carregando reranker RAG (cross-encoder)…")
def _rag_reranker_bundle_cached() -> RerankerLoadResult:
    """
    Carrega o cross-encoder para rerank em cache de processo.

    Retorna ``RerankerLoadResult`` com ``model=None`` e ``error`` preenchido
    quando o modelo não puder ser carregado — a busca híbrida continua, mas
    sem rerank (evita falha silenciosa).
    """
    return load_reranker_safe()


def _format_rag_hit_title(hit: dict, rank: int) -> str:
    """Cabeçalho do expander na aba de teste RAG (rerank vs retrieval)."""
    score = hit.get("score")
    retrieval = hit.get("retrieval_score")
    parts = [f"#{rank}"]
    if hit.get("rerank_applied"):
        parts.append(f"rerank={score}")
        if retrieval is not None:
            parts.append(f"retrieval={retrieval}")
    else:
        parts.append(f"score={score}")
        if env_rerank_enabled():
            parts.append("sem rerank")
    mode = hit.get("search_mode")
    if mode:
        parts.append(str(mode))
    return " · ".join(parts)


def rag_semantic_search(
    query: str,
    limit: int,
    *,
    project_ids: set[str] | None = None,
    reranker: object | None = None,
    rerank_retrieve_k: int | None = None,
) -> list[dict]:
    """
    Executa busca híbrida (BM25 + semântica) com rerank opcional usando backends em cache.

    ``_txtai_backend_cached`` gerencia o índice txtai; ``_rag_reranker_bundle_cached``
    gerencia o cross-encoder quando o rerank está ativo.
    """
    if not index_ready():
        return []
    mt = index_mtime()
    emb = _txtai_backend_cached(mt)
    if emb is None:
        return []

    q = (query or "").strip()
    if not q:
        return []

    tk = max(int(limit), 1)
    reranker_loaded = reranker
    rerank_error: str | None = None
    if reranker_loaded is None and env_rerank_enabled():
        bundle = _rag_reranker_bundle_cached()
        reranker_loaded = bundle.model
        rerank_error = bundle.error
    use_rerank = reranker_loaded is not None

    retrieve_k = default_retrieve_k(tk, rerank_retrieve_k) if use_rerank else tk
    hits = search_with_backend(
        emb,
        q,
        tk,
        project_ids=project_ids,
        retrieve_limit=retrieve_k if use_rerank else None,
    )
    if use_rerank and hits:
        hits = rerank_hits(q, hits, reranker=reranker_loaded, top_k=tk)
    elif rerank_error and env_rerank_enabled():
        for hit in hits:
            hit["rerank_applied"] = False
            hit["rerank_error"] = rerank_error
    return hits


@st.cache_resource
def _openai_client_cached(
    base_url: str,
    api_key: str,
    timeout_s: float,
    default_headers_items: tuple[tuple[str, str], ...] = (),
) -> OpenAI:
    """
    Cliente OpenAI reutilizado entre mensagens (evita handshake repetido).

    ``default_headers_items`` chega como tupla de pares (key, value) porque o
    ``st.cache_resource`` precisa de argumentos hashable; é reconstruído como
    dict antes de passar ao SDK. Para o OpenRouter, aqui entram os cabeçalhos
    ``HTTP-Referer`` e ``X-Title`` (rankings).
    """
    headers = dict(default_headers_items) if default_headers_items else None
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_s,
        default_headers=headers,
    )


def _openai_client() -> OpenAI:
    base, _, api_key = llm_runtime_config()
    timeout_s = float(os.environ.get("LLM_TIMEOUT_S", "120"))
    headers = openrouter_default_headers() if is_openrouter_endpoint(base) else {}
    headers_items = tuple(sorted(headers.items()))
    return _openai_client_cached(base, api_key, timeout_s, headers_items)


def _langfuse_session_id() -> str:
    """ID estável por sessão Streamlit — agrupa traces no painel Sessions."""
    key = "langfuse_session_id"
    if key not in st.session_state:
        st.session_state[key] = str(uuid.uuid4())
    return str(st.session_state[key])


def _check_openai_compatible_models(base_url: str, *, timeout_s: float = 5.0) -> tuple[bool, str]:
    """
    GET ``{base}/v1/models`` — compatível com OpenRouter (e qualquer endpoint
    OpenAI-compatible). Quando a URL aponta para o OpenRouter, anexamos os
    headers de rankings (``HTTP-Referer``/``X-Title``) para refletir como o
    app aparece na plataforma.
    """
    b = normalize_openai_base_url(base_url.strip())
    if not b:
        return False, "URL base vazia."
    url = f"{b}/models"
    headers: dict[str, str] = {}
    api_key = get_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if is_openrouter_endpoint(b):
        for k, v in openrouter_default_headers().items():
            headers.setdefault(k, v)
    req = urllib.request.Request(url, method="GET", headers=headers)
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


def _run_chat_via_crew(
    *,
    prompt: str,
    history_before: list[dict],
    client: OpenAI,
    model: str,
    use_rag: bool,
    use_olap: bool,
    use_ml: bool,
    rag_top_k: int,
    rag_project_ids: set[str] | None,
    max_tokens: int,
    max_history_turns: int,
    use_thinking: bool,
    use_stream: bool,
    show_dev_details: bool,
    show_trace: bool,
    chat_box,
) -> None:
    """
    Pipeline multiagente do chat (Triage → Tools → Synthesizer).

    O Synthesizer é executado **aqui** (não dentro do Crew) porque
    ``st.write_stream`` precisa do gerador da chamada
    ``create_chat_completion(stream=True)`` neste contexto Streamlit. O Crew
    apenas prepara as ``messages``, contextos e a trilha.

    Os limites de histórico (``max_history_turns``, ``max_chars_per_message``)
    são propagados ao ``run_crew_chat`` — o próprio Synthesizer aplica a
    truncagem em ``synth.messages``, evitando duplicação de lógica no app.
    """
    documents_available = use_rag and index_ready()
    spreadsheets_available = use_olap and has_ingested_tables()
    ml_available = use_ml and chat_ml_model_available()

    rag_backend = None
    rag_reranker = None
    if documents_available:
        rag_backend = _txtai_backend_cached(index_mtime())
        if env_rerank_enabled():
            rag_reranker = _rag_reranker_bundle_cached().model

    ml_bundle = None
    ml_path: Path | None = None
    if ml_available:
        ml_path = chat_ml_model_path()
        mtime = ml_path.stat().st_mtime if ml_path.is_file() else 0.0
        ml_bundle = _ml_bundle_cached(str(ml_path), mtime)

    # Estimativa preliminar dos limites para passar ao runner. O Synthesizer
    # ajusta dinamicamente quando ``used_ml`` for true, mas precisamos passar
    # algum valor antes — usamos o limite "geral" e revisamos abaixo se a rota
    # for ML.
    pre_max_tokens, pre_hist_turns = effective_chat_limits(
        run_ml=False,
        max_tokens=int(max_tokens),
        max_history_turns=int(max_history_turns),
    )
    pre_hist_chars = chat_history_chars_per_message(ml_route=False)

    with st.spinner("Coordenando agentes…"):
        crew_result = run_crew_chat(
            prompt,
            history=history_before,
            client=client,
            model=model,
            rag_backend=rag_backend,
            rag_top_k=int(rag_top_k),
            rag_project_ids=rag_project_ids,
            rag_reranker=rag_reranker,
            force_routes=bool(st.session_state.get("dev_chat_override")),
            force_use_rag=bool(use_rag),
            force_use_olap=bool(use_olap),
            force_use_ml=bool(use_ml),
            documents_available=documents_available,
            spreadsheets_available=spreadsheets_available,
            ml_available=ml_available,
            ml_bundle=ml_bundle,
            ml_model_path=ml_path,
            parallel_tools=parallel_tools_enabled(),
            max_history_turns=int(pre_hist_turns),
            max_chars_per_message=int(pre_hist_chars),
        )

    update_chat_trace_route(
        greeting=crew_result.greeting_response is not None,
        tool_results=crew_result.tool_results,
    )

    if show_dev_details and crew_result.triage is not None:
        d = crew_result.triage
        st.caption(
            f"Crew ({d.source}): rag={'sim' if d.use_rag else 'não'} · "
            f"olap={'sim' if d.use_olap else 'não'} · "
            f"ml={'sim' if d.use_ml else 'não'}"
            + (f" · motivo: {d.reason}" if d.reason else "")
        )
        ml_tool_result = crew_result.tool_results.get("ml")
        if ml_tool_result is not None:
            with st.expander("Predição ML (dev)", expanded=False):
                preds = ml_tool_result.payload.get("predictions")
                if preds is not None:
                    st.dataframe(preds, use_container_width=True, hide_index=True)
                raw = ml_tool_result.payload.get("raw_llm_response")
                if raw:
                    st.code(raw, language="json")
                if ml_tool_result.error:
                    st.warning(ml_tool_result.error)
        olap_tool_result = crew_result.tool_results.get("olap")
        if olap_tool_result is not None and (
            olap_tool_result.ok or olap_tool_result.payload.get("sql")
        ):
            with st.expander("SQL e dados (dev)", expanded=False):
                sql = olap_tool_result.payload.get("sql")
                if sql:
                    st.code(sql, language="sql")
                df = olap_tool_result.payload.get("dataframe")
                if df is not None:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                if olap_tool_result.error:
                    st.warning(olap_tool_result.error)

    # Greeter rule-based — resposta determinística sem stream.
    if crew_result.greeting_response is not None:
        with chat_box:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            with st.chat_message("assistant"):
                st.markdown(crew_result.greeting_response)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": crew_result.greeting_response}
        )
        update_chat_trace_output(crew_result.greeting_response)
        if show_trace:
            _render_handoff_trace(crew_result)
        return

    if crew_result.synth is None:
        st.error("Crew não conseguiu preparar o prompt do Synthesizer.")
        if (
            st.session_state.chat_messages
            and st.session_state.chat_messages[-1].get("role") == "user"
        ):
            st.session_state.chat_messages.pop()
        if show_trace:
            _render_handoff_trace(crew_result)
        return

    used_ml = crew_result.synth.used_ml
    reply_max_tokens, history_turns = effective_chat_limits(
        run_ml=used_ml,
        max_tokens=int(max_tokens),
        max_history_turns=int(max_history_turns),
    )
    history_chars = chat_history_chars_per_message(ml_route=used_ml)

    # Rota ML: refaz o prompt com janela menor (synth.messages anterior usou
    # os limites "gerais" porque ainda não sabíamos a rota). Custo desprezível
    # — apenas reordena strings, sem novas chamadas LLM.
    if used_ml and (history_turns != pre_hist_turns or history_chars != pre_hist_chars):
        from agents.synthesizer import build_messages as _rebuild

        history_with_user = history_before + [{"role": "user", "content": prompt}]
        crew_result.synth = _rebuild(
            user_message=prompt,
            history=history_with_user,
            tool_results=crew_result.tool_results,
            model_id=model,
            max_history_turns=int(history_turns),
            max_chars_per_message=int(history_chars),
        )

    api_messages = crew_result.synth.messages
    chat_profile = select_chat_profile(
        model_id=model, use_thinking=use_thinking and is_qwen35_model(model)
    )

    if show_dev_details and used_ml:
        st.caption(
            f"Limites ML: max_tokens={reply_max_tokens} · "
            f"histórico={history_turns} turno(s) · "
            f"{history_chars} chars/msg"
        )

    with chat_box:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        with st.chat_message("assistant"):
            try:
                # Borda externa (P1-4): anonimiza PII em TODO o prompt (system +
                # contexto RAG/OLAP + histórico + pergunta) antes de sair para o
                # OpenRouter. A resposta ao usuário autenticado permanece íntegra.
                external_messages, pii_sent = anonymize_messages_for_external(
                    api_messages
                )
                if show_dev_details and pii_sent:
                    st.caption(
                        "PII anonimizada antes do envio externo: "
                        + ", ".join(f"{k}×{v}" for k, v in sorted(pii_sent.items()))
                    )
                if use_stream:
                    stream = create_chat_completion(
                        client,
                        messages=external_messages,
                        model=model,
                        profile=chat_profile,
                        max_tokens=int(reply_max_tokens),
                        stream=True,
                        generation_name="crew-synthesizer",
                    )

                    placeholder = st.empty()
                    chunks: list[str] = []
                    for piece in iter_stream_answer_text(stream, model_id=model):
                        chunks.append(piece)
                        # Render incremental seguro: sanitiza o acumulado a cada
                        # token para nunca exibir markdown de exfiltração cru.
                        partial = sanitize_model_output(
                            strip_thinking_blocks("".join(chunks))
                        ).text
                        placeholder.markdown(partial + "▌")
                    full_text = strip_thinking_blocks("".join(chunks))
                    out_guard = sanitize_model_output(full_text)
                    answer = out_guard.text
                    placeholder.markdown(answer)
                else:
                    with st.spinner("Gerando resposta…"):
                        completion = create_chat_completion(
                            client,
                            messages=external_messages,
                            model=model,
                            profile=chat_profile,
                            max_tokens=int(reply_max_tokens),
                            stream=False,
                            generation_name="crew-synthesizer",
                        )
                    raw = (completion.choices[0].message.content or "").strip()
                    out_guard = sanitize_model_output(strip_thinking_blocks(raw))
                    answer = out_guard.text
                    st.markdown(answer)
                if show_dev_details and out_guard.neutralized:
                    st.caption(
                        "Saída sanitizada (vetores neutralizados): "
                        + ", ".join(out_guard.neutralized)
                    )
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer}
                )
                # Trace de auditoria (Langfuse): também é borda externa — envia a
                # resposta já sanitizada e anonimizada.
                update_chat_trace_output(anonymize_pii(answer).text)
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

    if show_trace:
        _render_handoff_trace(crew_result)


def _render_handoff_trace(crew_result: CrewRunResult) -> None:
    """
    Exibe a trilha de handoff dos agentes em um expander (modo aprendizado).

    Mostra cada passo (Greeter, Triage, Tools, Synthesizer) com:
    - duração em ms,
    - resumo de entrada/saída,
    - metadados (rota escolhida, contagem de evidências, SQL gerado, etc.).
    """
    trace = crew_result.trace
    if trace.step_count == 0:
        return
    with st.expander(
        f"Trilha dos agentes (dev) · {trace.step_count} etapa(s) · "
        f"{trace.total_elapsed_ms:.0f} ms",
        expanded=False,
    ):
        for i, step in enumerate(trace.steps, start=1):
            status_icon = "OK" if step.ok else "ERRO"
            st.markdown(
                f"**{i}. {step.name}** · {step.elapsed_ms:.0f} ms · {status_icon}"
            )
            if step.input_summary:
                st.caption(f"entrada: {step.input_summary}")
            if step.output_summary:
                st.caption(f"saída: {step.output_summary}")
            if step.note:
                st.caption(f"obs: {step.note}")
            if step.metadata:
                with st.popover("metadados", use_container_width=False):
                    st.json(step.metadata)
            if i < trace.step_count:
                st.divider()


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
        tabular_scans = filter_scans_by_extensions(scans, TABULAR_EXTENSIONS)
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
        "**RAG** com índice vetorial (**txtai**), consultas **OLAP** (**DuckDB**) e geração via **OpenRouter** "
        "(API compatível com OpenAI), tudo orquestrado aqui no **Streamlit**."
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
        st.markdown("**4. Responder com contexto** — Recuperação + OpenRouter; respostas devem citar arquivo/projeto.")
    st.subheader("Fluxo do pipeline (resumo)")
    st.code(
        "Pasta local → inventário (projects_loader) → extração/chunking → txtai → chat com citações\n"
        "                                    ↘ metadados / DuckDB (OLAP)",
        language="text",
    )
    with st.expander("Fases do plano (checklist)"):
        st.markdown(
            "- **Fase 1** — Diagnóstico, integração LLM (OpenRouter), DuckDB olá mundo (aba OLAP), metadados.\n"
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
        help="Se restringir a um subconjunto, a aba Conversa usa só esses projetos na busca de documentos.",
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
                max_chars=DEFAULT_CHUNK_MAX_CHARS,
                overlap=DEFAULT_CHUNK_OVERLAP,
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
            value=DEFAULT_CHUNK_MAX_CHARS,
            step=20,
            help="E5-small suporta até ~512 tokens; 520 caracteres (~100–130 tokens) é o padrão.",
            key="dev_rag_chunk_chars",
        )
    with c2:
        overlap = st.number_input(
            "Sobreposição entre chunks",
            min_value=0,
            max_value=800,
            value=DEFAULT_CHUNK_OVERLAP,
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
    """Aba 3 — Busca híbrida direta no índice txtai, sem passar pelo LLM."""
    st.header("Teste RAG (desenvolvedor)")
    st.caption(
        "Consulta direta ao índice **txtai** (mesmo backend do chat com RAG). "
        "Use para validar chunking, metadados e relevância antes de acoplar ao LLM remoto."
    )

    hybrid_on = env_hybrid_enabled()
    rerank_bundle = _rag_reranker_bundle_cached() if env_rerank_enabled() else None
    rerank_line = f"- Reranker: `{RERANKER_MODEL_ID}`"
    if rerank_bundle is not None:
        if rerank_bundle.ok:
            rerank_line += " — **ativo** (cross-encoder carregado)"
        else:
            rerank_line += f" — **indisponível** ({rerank_bundle.error})"
    else:
        rerank_line += " — desligado (`RAG_RERANK_ENABLED=0`)"
    st.markdown(
        f"- Modelo de embedding: `{EMBEDDING_MODEL_ID}`\n"
        f"- Busca híbrida (BM25 + semântica): **{'sim' if hybrid_on else 'não'}**"
        + (f" — peso denso α=`{hybrid_dense_weight()}`" if hybrid_on else "")
        + f"\n{rerank_line}\n"
        f"- Chunk padrão: **{DEFAULT_CHUNK_MAX_CHARS}** chars · overlap **{DEFAULT_CHUNK_OVERLAP}**\n"
        f"- Índice: `{txtai_index_path()}` — **pronto:** {'sim' if index_ready() else 'não'}"
    )

    if not index_ready():
        st.warning("Construa um índice na aba **Documentos** ou em **Índice vetorial** (dev) antes de testar.")
        return

    q = st.text_input(
        "Consulta",
        placeholder="Ex.: tampão de amostra no protocolo ELISA",
        key="dev_rag_search_query",
    )
    top_k = st.slider("Top-K", min_value=1, max_value=25, value=8, key="dev_rag_search_top_k")
    st.caption(
        "Busca híbrida combina significado (E5) com termos exatos (BM25). "
        "O rerank com cross-encoder é aplicado automaticamente após a recuperação."
    )

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
                if hits and hits[0].get("rerank_applied"):
                    st.caption("Scores **rerank** = cross-encoder; **retrieval** = busca híbrida (E5+BM25).")
                elif env_rerank_enabled() and hits:
                    err = hits[0].get("rerank_error")
                    if err:
                        st.warning(f"Rerank não aplicado: {err}")
                for i, h in enumerate(hits, start=1):
                    body = (h.get("cited") or h.get("text") or "").strip()
                    title = _format_rag_hit_title(h, i)
                    with st.expander(title, expanded=(i <= 3)):
                        st.markdown(body if body else "_(sem texto)_")
                        with st.popover("JSON bruto"):
                            st.json(h)


def _effective_rag_project_ids(scans: list[ProjectScan] | None) -> set[str] | None:
    """
    Projetos visíveis na busca RAG do chat.

    - Modo dev manual: multiselect ``dev_chat_rag_projects``.
    - Usuário final: filtro da aba Documentos quando restringe um subconjunto.
    - Caso contrário: ``None`` (todos os projetos indexados).
    """
    if scans is None:
        return None
    all_ids = {s.project_id for s in scans}
    if not all_ids:
        return None

    if st.session_state.get("dev_chat_override"):
        selected = st.session_state.get("dev_chat_rag_projects")
        if selected:
            active = {p for p in selected if p in all_ids}
            return active or None
        return None

    selected = st.session_state.get("docs_filter_projects")
    if selected:
        active = {p for p in selected if p in all_ids}
        if active and len(active) < len(all_ids):
            return active
    return None


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
    use_ml = (
        bool(st.session_state.get("dev_chat_use_ml", True))
        if dev_override
        else chat_ml_model_available()
    )
    show_trace = bool(st.session_state.get("dev_chat_show_trace", trace_handoff_enabled()))
    return {
        "use_rag": use_rag and index_ready(),
        "use_olap": use_olap and has_ingested_tables(),
        "use_ml": use_ml and chat_ml_model_available(),
        "rag_top_k": int(st.session_state.get("dev_chat_rag_top_k", 6)),
        "max_tokens": int(
            st.session_state.get("dev_chat_max_tokens", chat_max_tokens(ml_route=False))
        ),
        "max_history_turns": int(
            st.session_state.get(
                "dev_chat_max_history_turns", chat_max_history_turns(ml_route=False)
            )
        ),
        "use_thinking": bool(st.session_state.get("dev_chat_use_thinking", env_enable_thinking_default())),
        "use_stream": bool(st.session_state.get("chat_use_stream", True)),
        "show_trace": show_trace,
    }


def _tab_chat_dev_controls() -> None:
    """Controles avançados do chat (aba Desenvolvimento)."""
    st.subheader("Parâmetros do chat")
    st.caption(
        "Com *configuração manual*, os toggles substituem o roteador LLM. "
        "Sem isso, um classificador leve decide se cada mensagem consulta documentos, planilhas ou predição ML."
    )
    base, model, _ = llm_runtime_config()
    st.caption(f"Servidor: `{base}` · Modelo: `{model}`")

    st.toggle(
        "Usar configuração manual",
        key="dev_chat_override",
        help="Desligado: documentos e planilhas entram automaticamente quando existirem.",
    )

    c0, c1, c2, c3, c4 = st.columns(5)
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
            "ML (predição)",
            value=True,
            key="dev_chat_use_ml",
            disabled=not st.session_state.get("dev_chat_override"),
        )
    with c3:
        st.toggle(
            "Raciocínio",
            value=env_enable_thinking_default(),
            key="dev_chat_use_thinking",
            disabled=not is_qwen35_model(model) or not st.session_state.get("dev_chat_override"),
        )
    with c4:
        st.toggle("Streaming", value=True, key="chat_use_stream")
    st.caption(f"Modelo ML do chat: `{chat_ml_model_path()}`")

    # Trilha de handoff (aprendizado / auditoria). O pipeline multiagente é o
    # único caminho do chat — não há mais toggle de Crew on/off.
    st.toggle(
        "Mostrar trilha do crew (modo aprendizado)",
        value=trace_handoff_enabled(),
        key="dev_chat_show_trace",
        help=(
            "Exibe um expander abaixo da resposta com cada etapa do Crew "
            "(Greeter, Triage, RAG/OLAP/ML, Synthesizer) e seus metadados."
        ),
    )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.number_input("Trechos", 1, 20, 6, key="dev_chat_rag_top_k")
    with c5:
        st.number_input(
            "Máx. tokens",
            256,
            32768,
            int(chat_max_tokens(ml_route=False)),
            step=256,
            key="dev_chat_max_tokens",
            help="Na rota de predição ML, o teto efetivo é CHAT_ML_MAX_TOKENS (padrão 768).",
        )
    with c6:
        st.number_input(
            "Turnos no histórico",
            1,
            30,
            int(chat_max_history_turns(ml_route=False)),
            key="dev_chat_max_history_turns",
            help="Na rota ML, usa no máximo CHAT_ML_MAX_HISTORY_TURNS (padrão 2).",
        )
    st.caption(
        f"Rerank RAG automático com `{RERANKER_MODEL_ID}` "
        "(candidatos: max(top_k×4, 20); override via `RAG_RERANK_RETRIEVE_K`)."
    )

    scans: list[ProjectScan] | None = st.session_state.get("last_scan")
    if scans and st.session_state.get("dev_chat_override"):
        options = sorted({s.project_id for s in scans})
        st.multiselect(
            "Projetos visíveis no RAG (dev)",
            options=options,
            default=options,
            help="Restringe trechos recuperados aos projetos selecionados. Vazio = todos.",
            key="dev_chat_rag_projects",
        )


def _tab_chat() -> None:
    """
    Aba principal — conversa com o agente.

    O contexto de documentos/planilhas é injetado no system prompt quando
    disponível, sem expor parâmetros técnicos ao usuário final.
    """
    st.header("Conversa")
    st.caption("Pergunte sobre experimentos, insumos e resultados documentados no laboratório.")

    _, model, _ = llm_runtime_config()
    opts = _chat_effective_options()
    use_rag = opts["use_rag"]
    use_olap = opts["use_olap"]
    use_ml = opts["use_ml"]
    rag_top_k = opts["rag_top_k"]
    max_tokens = opts["max_tokens"]
    max_history_turns = opts["max_history_turns"]
    use_thinking = opts["use_thinking"]
    use_stream = opts["use_stream"]
    show_trace = opts["show_trace"]

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    status_a, status_b, status_c, actions = st.columns([2, 2, 2, 1])
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
    with status_c:
        st.caption(ml_inference_status_message())
    with actions:
        if st.button("Limpar", key="btn_clear_chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.pop("langfuse_session_id", None)
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
        # Guardrail de entrada (P1-1): bloqueia prompt-injection/extração de
        # prompt e mensagens gigantes ANTES de chamar qualquer LLM (zero tokens).
        guard = scan_user_input(prompt)
        if not guard.allowed:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": guard.reason
                    or "Mensagem bloqueada pela camada de segurança.",
                }
            )
            with chat_box:
                for msg in st.session_state.chat_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            return

        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        show_dev_details = bool(st.session_state.get("dev_chat_override"))

        client = _openai_client()
        scans: list[ProjectScan] | None = st.session_state.get("last_scan")
        rag_project_ids = _effective_rag_project_ids(scans)

        history_before = st.session_state.chat_messages[:-1]

        # Pipeline multiagente: Triage → Tools (paralelas) → Synthesizer.
        # Greeter rule-based curto-circuita saudações sem chamar o LLM.
        with chat_observation_context(
            session_id=_langfuse_session_id(),
            input_text=prompt,
            metadata={"model": model},
            tags=["feature:chat"],
        ):
            _run_chat_via_crew(
                prompt=prompt,
                history_before=history_before,
                client=client,
                model=model,
                use_rag=use_rag,
                use_olap=use_olap,
                use_ml=use_ml,
                rag_top_k=rag_top_k,
                rag_project_ids=rag_project_ids,
                max_tokens=max_tokens,
                max_history_turns=max_history_turns,
                use_thinking=use_thinking,
                use_stream=use_stream,
                show_dev_details=show_dev_details,
                show_trace=show_trace,
                chat_box=chat_box,
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
            "Busca híbrida",
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
    elif st.button("Sincronizar planilhas com DuckDB agora", type="primary", key="btn_olap_sync"):
        with st.spinner("Carregando CSV/XLSX/XLSM nos projetos…"):
            tabular_scans = filter_scans_by_extensions(scans, TABULAR_EXTENSIONS)
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
        schema_text = build_schema_catalog_text(sample_rows=2)
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
        "No Compose, volumes típicos: `/data/projetos`, `/data/txtai`, `/data/duckdb`, "
        "`/data/ml`, `/data/sqlite`."
    )

    st.subheader("ML tradicional (FLAML)")
    flaml_ok, flaml_detail = flaml_available()
    st.write(f"- Modelos `.pkl`: `{ml_models_root()}`")
    st.write(f"- Cache Kaggle: `{kaggle_cache_root()}`")
    st.write(f"- FLAML disponível: **{'sim' if flaml_ok else 'não'}**")
    if not flaml_ok and flaml_detail:
        st.caption(flaml_detail)

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

    st.subheader("LLM remoto (OpenRouter — API compatível com OpenAI)")
    llm_base = get_llm_base_url_raw()
    llm_model = get_llm_model()
    resolved_base, _, _ = llm_runtime_config()
    env_base = os.environ.get("LLM_BASE_URL", "").strip()
    env_model = os.environ.get("LLM_MODEL", "").strip()
    api_key_present = bool(get_llm_api_key())
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
    if api_key_present:
        st.caption(
            "Chave de API: **configurada** "
            "(`OPENROUTER_API_KEY` ou `OPENAI_API_KEY`)."
        )
    else:
        st.warning(
            "Nenhuma chave de API foi encontrada. Defina `OPENROUTER_API_KEY` "
            "no `.env` e recrie o contêiner (`docker compose up -d --build`)."
        )
    if is_openrouter_endpoint(resolved_base):
        with st.expander("Headers do OpenRouter (rankings)", expanded=False):
            st.json(openrouter_default_headers())
    if st.button("Testar GET /v1/models (timeout 5s)", key="btn_llm_ping"):
        ok, detail = _check_openai_compatible_models(llm_base)
        if ok:
            st.success("Servidor respondeu. Modelos (amostra):")
            st.code(detail, language="text")
        else:
            st.error(detail)

    st.subheader("Observabilidade (Langfuse)")
    lf = langfuse_status()
    if lf["enabled"]:
        st.success(
            "Langfuse **ativo** — cada turno do chat envia traces para o projeto "
            f"configurado (`{lf['base_url']}`)."
        )
        st.caption(
            f"Sessão Streamlit atual: `{_langfuse_session_id()}` · "
            f"ambiente: `{lf['environment']}`"
            + (f" · release: `{lf['release']}`" if lf["release"] else "")
        )
        if lf["tags"]:
            st.caption(f"Tags: {', '.join(lf['tags'])}")
    else:
        st.info(
            "Langfuse **inativo**. Defina `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY` "
            "no `.env` (chaves em https://cloud.langfuse.com → Settings → API Keys) "
            "e recrie o contêiner. Use `LANGFUSE_ENABLED=0` para desligar sem apagar as chaves."
        )
        st.caption(
            "Chaves detectadas: "
            f"pública={'sim' if lf['public_key_set'] else 'não'} · "
            f"secreta={'sim' if lf['secret_key_set'] else 'não'}"
        )


# ── Ponto de entrada ─────────────────────────────────────────────────────────
# O Streamlit re-executa este arquivo inteiro a cada interação do usuário.
# Funções cacheadas (_txtai_backend_cached) e o st.session_state preservam
# estado entre execuções dentro da mesma sessão de browser.

_render_sidebar()

root = _root_from_session()
root_ok, root_msg = validate_projetos_root(root)

st.title("Assistente Lab")
st.caption("Assistente para documentos e dados do laboratório — uso local e offline.")

tabs = st.tabs(["Conversa", "Documentos", "ML tradicional", "Desenvolvimento"])

scans: list[ProjectScan] | None = st.session_state.get("last_scan")

with tabs[0]:
    _tab_chat()
with tabs[1]:
    _tab_documentos(root, root_ok, root_msg)
with tabs[2]:
    render_ml_tab()
with tabs[3]:
    _tab_desenvolvimento(root, root_ok, root_msg, scans)
