"""
Roteamento de intenção do chat — decide se RAG e/ou OLAP rodam nesta mensagem.

Usa regras rápidas para saudações óbvias e um classificador LLM leve para o resto.
Não aplica limiar de score no RAG (a busca só roda quando ``use_documents`` é true).
A inferência ML só roda quando ``use_ml`` é true (pedido explícito de predição).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from openai import OpenAI

from qwen35_inference import (
    DEFAULT_ROUTER_MAX_TOKENS,
    PROFILE_CHAT_ROUTER,
    chat_history_chars_per_message,
    chat_max_history_turns,
    create_chat_completion,
    format_history_snippet,
    strip_thinking_blocks,
)

_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)

_ROUTER_SYSTEM = """Você classifica a intenção da última mensagem do usuário em um chat de laboratório de P&D (ELISA, documentos Word, planilhas).

Responda APENAS com um objeto JSON válido, sem markdown e sem texto extra:
{"documents": true ou false, "spreadsheets": true ou false, "ml_prediction": true ou false}

documents=true quando a mensagem pede ou pressupõe informação de experimentos, insumos, lotes, validades, materiais, protocolos, documentos, histórico de ensaios, ou é continuação disso (ex.: "e a validade?", "qual lote?").

spreadsheets=true quando pede análise de dados tabulares: contagens, totais, médias, somas, comparações, rankings, filtros em planilhas/CSV, agregações.

ml_prediction=true SOMENTE quando o usuário pede EXPLICITAMENTE predição, inferência ou estimativa pelo modelo/algoritmo de ML treinado (ex.: prever log_Aff, afinidade Ab–Ag, "rode o modelo", "faça a predição"). NÃO marque true para perguntas gerais sobre experimentos, documentos ou planilhas sem pedido claro de predição ML.

Ambos false (e ml_prediction false) para saudação, despedida, agradecimento, conversa social, meta sobre o assistente, ou mensagem sem pedido de dado do laboratório.

Se vários tipos se aplicarem, marque cada um como true."""

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

_ML_HINT = re.compile(
    r"\b(?:"
    r"predi[çc][ãa]o|predizer|prever|previs[ãa]o|infer[êe]ncia|inferir|"
    r"modelo\s+ml|algoritmo\s+ml|machine\s+learning|"
    r"log_aff|log\s+aff|afinidade\s+ab|abrank|"
    r"estim(?:e|ar)\s+(?:a\s+)?afinidade|rodar\s+o\s+modelo"
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
    use_ml: bool
    source: str  # "rules" | "llm" | "rules_fallback" | "disabled" | "manual"


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


def _parse_router_json(raw: str) -> tuple[bool, bool, bool] | None:
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
    ml = data.get("ml_prediction", False)
    if not isinstance(docs, bool) or not isinstance(sheets, bool):
        return None
    if not isinstance(ml, bool):
        ml = False
    return docs, sheets, ml


def _rule_fallback_route(message: str) -> ChatRouteDecision:
    """Fallback conservador se o LLM falhar (sem limiar de score no RAG)."""
    if _is_social_only(message):
        return ChatRouteDecision(False, False, False, "rules_fallback")
    ml = bool(_ML_HINT.search(message))
    docs = bool(_LAB_DOC_HINT.search(message)) and not ml
    sheets = bool(_TABULAR_HINT.search(message)) and not ml
    if not docs and not sheets and not ml and len(message.split()) <= 3:
        return ChatRouteDecision(False, False, False, "rules_fallback")
    return ChatRouteDecision(docs, sheets, ml, "rules_fallback")


def _apply_availability(
    decision: ChatRouteDecision,
    *,
    documents_available: bool,
    spreadsheets_available: bool,
    ml_available: bool,
) -> ChatRouteDecision:
    return ChatRouteDecision(
        use_documents=decision.use_documents and documents_available,
        use_spreadsheets=decision.use_spreadsheets and spreadsheets_available,
        use_ml=decision.use_ml and ml_available,
        source=decision.source,
    )


def _prioritize_ml_when_requested(
    decision: ChatRouteDecision,
    message: str,
    *,
    ml_available: bool,
) -> ChatRouteDecision:
    """
    Se a mensagem pede predição ML e o .pkl existe, força só a rota ML.

    Evita que o classificador marque ``documents=true`` e o chat ignore o modelo.
    """
    if not ml_available or not _ML_HINT.search(message):
        return decision
    source = decision.source if decision.use_ml else f"{decision.source}+ml_hint"
    return ChatRouteDecision(False, False, True, source)


def classify_with_llm(
    message: str,
    *,
    history: list[dict],
    client: OpenAI,
    model: str,
) -> ChatRouteDecision | None:
    """Chama o LM Studio para classificar; retorna ``None`` se a resposta for inválida."""
    ml_hint = bool(_ML_HINT.search(message))
    hist_turns = chat_max_history_turns(ml_route=ml_hint)
    hist_chars = chat_history_chars_per_message(ml_route=ml_hint)
    user_block = (
        f"Histórico recente:\n"
        f"{format_history_snippet(history, max_turns=hist_turns, max_chars_per_message=hist_chars)}\n\n"
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
    docs, sheets, ml = parsed
    return ChatRouteDecision(docs, sheets, ml, "llm")


def classify_chat_routes(
    message: str,
    *,
    history: list[dict] | None = None,
    client: OpenAI | None = None,
    model: str = "",
    documents_available: bool = False,
    spreadsheets_available: bool = False,
    ml_available: bool = False,
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
    documents_available / spreadsheets_available / ml_available:
        Se o backend não estiver pronto, força false no respectivo flag.
    """
    msg = _normalize_message(message)
    hist = history or []

    def _finalize(decision: ChatRouteDecision) -> ChatRouteDecision:
        applied = _apply_availability(
            decision,
            documents_available=documents_available,
            spreadsheets_available=spreadsheets_available,
            ml_available=ml_available,
        )
        return _prioritize_ml_when_requested(applied, msg, ml_available=ml_available)

    if not router_enabled():
        return _finalize(ChatRouteDecision(True, True, False, "disabled"))

    if _is_social_only(msg):
        return ChatRouteDecision(False, False, False, "rules")

    if client is not None and model:
        llm_decision = classify_with_llm(msg, history=hist, client=client, model=model)
        if llm_decision is not None:
            return _finalize(llm_decision)

    return _finalize(_rule_fallback_route(msg))


def resolve_chat_routes(
    message: str,
    *,
    history: list[dict] | None = None,
    client: OpenAI | None = None,
    model: str = "",
    documents_enabled: bool,
    spreadsheets_enabled: bool,
    ml_enabled: bool = False,
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
            use_ml=ml_enabled,
            source="manual",
        )
    return classify_chat_routes(
        message,
        history=history,
        client=client,
        model=model,
        documents_available=documents_enabled,
        spreadsheets_available=spreadsheets_enabled,
        ml_available=ml_enabled,
    )
