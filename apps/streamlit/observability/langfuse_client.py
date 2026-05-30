"""
Integração Langfuse para rastrear chamadas ao LLM (OpenRouter / OpenAI-compatível).

Segue o guia de instrumentação do skill Langfuse:
https://github.com/langfuse/skills/blob/main/skills/langfuse/references/instrumentation.md

- Import do patch OpenAI **depois** do ``load_dotenv`` (``llm_config._bootstrap_langfuse``).
- Wrapper OpenAI automático via ``langfuse.openai`` (modelo, tokens, latência).
- Trace raiz por turno de chat com ``session_id`` e tags de rota (rag/olap/ml).
- Spans aninhados para pipeline multiagente (Triage, Tools, Synthesizer).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

_openai_tracing_installed = False


def normalize_langfuse_env() -> None:
    """
    Alinha variáveis de ambiente com o SDK e o CLI Langfuse.

    O skill oficial usa ``LANGFUSE_HOST``; o SDK Python aceita ``LANGFUSE_BASE_URL``
    ou ``LANGFUSE_HOST``. Se só uma estiver definida, espelhamos para a outra.
    """
    host = os.environ.get("LANGFUSE_HOST", "").strip()
    base = os.environ.get("LANGFUSE_BASE_URL", "").strip()
    if host and not base:
        os.environ["LANGFUSE_BASE_URL"] = host
    elif base and not host:
        os.environ["LANGFUSE_HOST"] = base


def langfuse_enabled() -> bool:
    """
    Retorna True quando as chaves públicas/secretas estão configuradas.

    ``LANGFUSE_ENABLED=0`` desliga a integração sem remover as chaves do ``.env``.
    ``LANGFUSE_TRACING_ENABLED=false`` segue a flag nativa do SDK.
    """
    if os.environ.get("LANGFUSE_ENABLED", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "true").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


def ensure_openai_tracing() -> None:
    """Registra o wrapper Langfuse nas chamadas ``openai.*.create`` (idempotente)."""
    global _openai_tracing_installed
    if _openai_tracing_installed or not langfuse_enabled():
        return
    normalize_langfuse_env()
    import langfuse.openai  # noqa: F401 — efeito colateral: register_tracing()

    _openai_tracing_installed = True


def flush_langfuse() -> None:
    """Envia spans pendentes ao Langfuse (útil após cada turno no Streamlit)."""
    if not langfuse_enabled():
        return
    from langfuse import get_client

    get_client().flush()


def _langfuse_tags() -> list[str]:
    raw = os.environ.get("LANGFUSE_TAGS", "").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def crew_route_tags(
    *,
    use_rag: bool = False,
    use_olap: bool = False,
    use_ml: bool = False,
) -> list[str]:
    """Tags derivadas da rota do crew — filtro no dashboard Langfuse."""
    tags: list[str] = ["feature:chat"]
    if use_rag:
        tags.append("route:rag")
    if use_olap:
        tags.append("route:olap")
    if use_ml:
        tags.append("route:ml")
    if not (use_rag or use_olap or use_ml):
        tags.append("route:direct")
    return tags


def crew_route_tags_from_execution(
    *,
    greeting: bool = False,
    tool_results: dict[str, Any] | None = None,
) -> list[str]:
    """
    Tags baseadas nas Tools que de fato executaram (não nos toggles da UI).

    Preferir esta função após ``run_crew_chat`` para traces Langfuse fiéis.
    """
    tags: list[str] = ["feature:chat"]
    if greeting:
        tags.append("route:greeter")
        return tags
    executed = set(tool_results or {})
    use_rag = "rag" in executed
    use_olap = "olap" in executed
    use_ml = "ml" in executed
    if use_rag:
        tags.append("route:rag")
    if use_olap:
        tags.append("route:olap")
    if use_ml:
        tags.append("route:ml")
    if not (use_rag or use_olap or use_ml):
        tags.append("route:direct")
    return tags


def update_chat_trace_route(
    *,
    greeting: bool = False,
    tool_results: dict[str, Any] | None = None,
) -> None:
    """
    Atualiza tags do turno com a rota real executada no crew.

    Langfuse SDK v4 removeu ``update_current_trace()``; tags passam a ser
    atributos da observação corrente (``langfuse.trace.tags`` no span OTEL).
    """
    if not langfuse_enabled():
        return
    from langfuse import get_client
    from opentelemetry import trace as otel_trace

    tags = crew_route_tags_from_execution(
        greeting=greeting,
        tool_results=tool_results,
    )
    env_tags = _langfuse_tags()
    all_tags = list(dict.fromkeys([*tags, *env_tags]))

    client = get_client()
    client.update_current_span(metadata={"route_tags": all_tags})

    otel_span = otel_trace.get_current_span()
    if otel_span is not None and otel_span.is_recording():
        otel_span.set_attribute("langfuse.trace.tags", all_tags)


def langfuse_status() -> dict[str, Any]:
    """Resumo para a aba Desenvolvimento → Diagnóstico."""
    base_url = (
        os.environ.get("LANGFUSE_BASE_URL", "").strip()
        or os.environ.get("LANGFUSE_HOST", "").strip()
        or "https://cloud.langfuse.com"
    )
    return {
        "enabled": langfuse_enabled(),
        "openai_tracing_installed": _openai_tracing_installed,
        "public_key_set": bool(os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()),
        "secret_key_set": bool(os.environ.get("LANGFUSE_SECRET_KEY", "").strip()),
        "base_url": base_url,
        "environment": os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", "").strip()
        or "default",
        "release": os.environ.get("LANGFUSE_RELEASE", "").strip(),
        "tags": _langfuse_tags(),
    }


def update_chat_trace_output(output: str | None) -> None:
    """Define a saída do trace raiz (resposta do assistente, truncada)."""
    if not langfuse_enabled() or not output:
        return
    from langfuse import get_client

    text = output.strip()
    if not text:
        return
    get_client().update_current_span(output=text[:4000])


def record_trace_score(
    *,
    name: str,
    value: float | int | bool,
    comment: str | None = None,
) -> None:
    """
    Registra score no trace atual (ex.: feedback do usuário).

    Ver: https://langfuse.com/docs/scores/overview
    """
    if not langfuse_enabled():
        return
    from langfuse import get_client

    get_client().score_current_trace(
        name=name,
        value=float(value) if isinstance(value, bool) else value,
        comment=comment,
    )
    flush_langfuse()


@contextmanager
def langfuse_span(
    name: str,
    *,
    as_type: str = "span",
    input_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """
    Span filho nomeado (ex.: ``Triage``, ``RAG Tool``).

    ``as_type="tool"`` destaca Tools no UI Langfuse.
    """
    if not langfuse_enabled():
        yield None
        return

    ensure_openai_tracing()
    from langfuse import get_client

    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type=as_type,  # type: ignore[arg-type]
        name=name,
        input=input_data,
        metadata=metadata or None,
    ) as observation:
        yield observation


@contextmanager
def chat_observation_context(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str = "chat-turn",
    input_text: str | None = None,
    metadata: dict[str, str] | None = None,
    tags: list[str] | None = None,
) -> Iterator[None]:
    """
    Agrupa um turno do chat (Triage + Tools + Synthesizer) num trace Langfuse.

    ``session_id`` agrupa várias mensagens da mesma sessão Streamlit no painel
    Sessions do Langfuse. O ``input`` registra só a mensagem do usuário (sem
    chaves de API nem objetos internos).
    """
    if not langfuse_enabled():
        yield
        return

    ensure_openai_tracing()
    from langfuse import get_client, propagate_attributes

    langfuse = get_client()
    env_tags = _langfuse_tags()
    all_tags = list(dict.fromkeys([*(tags or []), *env_tags]))

    meta = dict(metadata or {})
    user_input = input_text.strip()[:2000] if input_text else None

    with langfuse.start_as_current_observation(
        as_type="span",
        name=trace_name,
        input={"user_message": user_input} if user_input else None,
        metadata=meta or None,
    ):
        prop: dict[str, Any] = {}
        if session_id:
            prop["session_id"] = session_id
        if user_id:
            prop["user_id"] = user_id
        if all_tags:
            prop["tags"] = all_tags
        if meta:
            prop["metadata"] = meta
        with propagate_attributes(**prop):
            try:
                yield
            finally:
                flush_langfuse()
