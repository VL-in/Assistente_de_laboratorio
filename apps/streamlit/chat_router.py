"""
Roteamento de intenção do chat — decide se RAG e/ou OLAP rodam nesta mensagem.

Usa regras rápidas para saudações óbvias e um classificador LLM leve para o resto.
Não aplica limiar de score no RAG (a busca só roda quando ``use_documents`` é true).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from qwen35_inference import (
    DEFAULT_ROUTER_MAX_TOKENS,
    PROFILE_CHAT_ROUTER,
    create_chat_completion,
    strip_thinking_blocks,
)

_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)

_ROUTER_SYSTEM = """Você classifica a intenção da última mensagem do usuário em um chat de laboratório de P&D (ELISA, documentos Word, planilhas).

Responda APENAS com um objeto JSON válido, sem markdown e sem texto extra:
{"documents": true ou false, "spreadsheets": true ou false}

documents=true quando a mensagem pede ou pressupõe informação de experimentos, insumos, lotes, validades, materiais, protocolos, documentos, histórico de ensaios, ou é continuação disso (ex.: "e a validade?", "qual lote?").

spreadsheets=true quando pede análise de dados tabulares: contagens, totais, médias, somas, comparações, rankings, filtros em planilhas/CSV, agregações.

Ambos false para saudação, despedida, agradecimento, conversa social, meta sobre o assistente, ou mensagem sem pedido de dado do laboratório.

Se ambos tipos se aplicarem, marque os dois como true."""

_SOCIAL_ONLY = re.compile(
    r"^\s*(?:"
    r"oi|olá|ola|hey|hi|hello|"
    r"bom\s+dia|boa\s+tarde|boa\s+noite|"
    r"tudo\s+bem|como\s+vai|e\s+aí|eai|"
    r"obrigad[oa]|valeu|brigad[oa]|"
    r"até\s+(?:mais|logo)|tchau|flw|"
    r"ok|okay|sim|não|nao"
    r")\s*[!.?…]*\s*$",
    re.IGNORECASE,
)

_LAB_DOC_HINT = re.compile(
    r"\b(?:"
    r"lote|validade|reagente|insumo|ensaio|elisa|experimento|"
    r"documento|protocolo|planejamento|fabricante|"
    r"material|amostra|dilui|placa|coating|"
    r"usamos|utilizamos|qual\s+foi|quando\s+foi|onde\s+está"
    r")\b",
    re.IGNORECASE,
)

_TABULAR_HINT = re.compile(
    r"\b(?:"
    r"planilha|tabela|csv|xlsx|"
    r"quantos|quantas|média|media|soma|total|contagem|"
    r"comparar|comparação|comparacao|ranking|agrupar|"
    r"por\s+projeto|linhas|colunas|registros"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatRouteDecision:
    """O que executar nesta rodada do chat."""

    use_documents: bool
    use_spreadsheets: bool
    source: str  # "rules" | "llm" | "rules_fallback" | "disabled"


def classification_needs_llm(message: str) -> bool:
    """True se esta mensagem deve passar pelo classificador LLM (não é saudação óbvia)."""
    return router_enabled() and not _is_social_only(message)


def router_enabled() -> bool:
    """``CHAT_INTENT_ROUTER=0|false`` desliga o classificador (comportamento antigo: sempre consulta)."""
    return os.environ.get("CHAT_INTENT_ROUTER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _normalize_message(text: str) -> str:
    return (text or "").strip()


def _is_social_only(message: str) -> bool:
    msg = _normalize_message(message)
    if not msg or len(msg) > 80:
        return False
    return _SOCIAL_ONLY.match(msg) is not None


def _format_history_snippet(history: list[dict], *, max_turns: int = 3) -> str:
    """Últimas mensagens user/assistant para desambiguar follow-ups."""
    if not history:
        return "(sem histórico anterior)"
    tail = history[-(max_turns * 2) :]
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "?")
        content = _normalize_message(str(m.get("content") or ""))
        if not content:
            continue
        label = "Usuário" if role == "user" else "Assistente"
        lines.append(f"{label}: {content[:400]}")
    return "\n".join(lines) if lines else "(sem histórico anterior)"


def _parse_router_json(raw: str) -> tuple[bool, bool] | None:
    text = strip_thinking_blocks((raw or "").strip())
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    docs = data.get("documents")
    sheets = data.get("spreadsheets")
    if not isinstance(docs, bool) or not isinstance(sheets, bool):
        return None
    return docs, sheets


def _rule_fallback_route(message: str) -> ChatRouteDecision:
    """Fallback conservador se o LLM falhar (sem limiar de score no RAG)."""
    if _is_social_only(message):
        return ChatRouteDecision(False, False, "rules_fallback")
    docs = bool(_LAB_DOC_HINT.search(message))
    sheets = bool(_TABULAR_HINT.search(message))
    if not docs and not sheets and len(message.split()) <= 3:
        return ChatRouteDecision(False, False, "rules_fallback")
    return ChatRouteDecision(docs, sheets, "rules_fallback")


def _apply_availability(
    decision: ChatRouteDecision,
    *,
    documents_available: bool,
    spreadsheets_available: bool,
) -> ChatRouteDecision:
    return ChatRouteDecision(
        use_documents=decision.use_documents and documents_available,
        use_spreadsheets=decision.use_spreadsheets and spreadsheets_available,
        source=decision.source,
    )


def classify_with_llm(
    message: str,
    *,
    history: list[dict],
    client: OpenAI,
    model: str,
) -> ChatRouteDecision | None:
    """Chama o LM Studio para classificar; retorna ``None`` se a resposta for inválida."""
    user_block = (
        f"Histórico recente:\n{_format_history_snippet(history)}\n\n"
        f"Última mensagem do usuário:\n{message.strip()}"
    )
    try:
        completion = create_chat_completion(
            client,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": user_block},
            ],
            model=model,
            profile=PROFILE_CHAT_ROUTER,
            max_tokens=DEFAULT_ROUTER_MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        return None

    parsed = _parse_router_json(raw)
    if parsed is None:
        return None
    docs, sheets = parsed
    return ChatRouteDecision(docs, sheets, "llm")


def classify_chat_routes(
    message: str,
    *,
    history: list[dict] | None = None,
    client: OpenAI | None = None,
    model: str = "",
    documents_available: bool = False,
    spreadsheets_available: bool = False,
) -> ChatRouteDecision:
    """
    Decide se esta mensagem deve acionar RAG (documentos) e/ou OLAP (planilhas).

    Parameters
    ----------
    message:
        Texto atual do usuário.
    history:
        Mensagens anteriores (sem a mensagem atual), para follow-ups.
    client, model:
        Obrigatórios para o classificador LLM (exceto saudações e fallback).
    documents_available / spreadsheets_available:
        Se o backend não estiver pronto, força false no respectivo flag.
    """
    msg = _normalize_message(message)
    hist = history or []

    if not router_enabled():
        return _apply_availability(
            ChatRouteDecision(True, True, "disabled"),
            documents_available=documents_available,
            spreadsheets_available=spreadsheets_available,
        )

    if _is_social_only(msg):
        return ChatRouteDecision(False, False, "rules")

    if client is not None and model:
        llm_decision = classify_with_llm(msg, history=hist, client=client, model=model)
        if llm_decision is not None:
            return _apply_availability(
                llm_decision,
                documents_available=documents_available,
                spreadsheets_available=spreadsheets_available,
            )

    fallback = _rule_fallback_route(msg)
    return _apply_availability(
        fallback,
        documents_available=documents_available,
        spreadsheets_available=spreadsheets_available,
    )


def resolve_chat_routes(
    message: str,
    *,
    history: list[dict] | None = None,
    client: OpenAI | None = None,
    model: str = "",
    documents_enabled: bool,
    spreadsheets_enabled: bool,
    manual_override: bool = False,
) -> ChatRouteDecision:
    """
    Ponto único usado pelo Streamlit.

    Com ``manual_override`` (aba Desenvolvimento), respeita os toggles sem LLM.
    """
    if manual_override:
        return ChatRouteDecision(
            use_documents=documents_enabled,
            use_spreadsheets=spreadsheets_enabled,
            source="manual",
        )
    return classify_chat_routes(
        message,
        history=history,
        client=client,
        model=model,
        documents_available=documents_enabled,
        spreadsheets_available=spreadsheets_enabled,
    )
