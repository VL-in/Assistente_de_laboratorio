"""
Triage Agent — classifica intenção da mensagem e decide quais Tools acionar.

Usa o LM Studio com perfil ``PROFILE_CHAT_ROUTER`` (saída JSON curta, temp 0.2).
Não usa o LLM do CrewAI diretamente — chama o cliente OpenAI com os mesmos
parâmetros já validados em ``chat_router.classify_with_llm``, garantindo
compatibilidade com Qwen3.5 (``enable_thinking=False`` para JSON estável).

Saída: ``TriageDecision(use_rag, use_olap, use_ml, source, reason)``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from chat_router import (
    _ML_HINT,
    _is_social_only,
    _parse_router_json,
    _rule_fallback_route,
)
from qwen35_inference import (
    DEFAULT_ROUTER_MAX_TOKENS,
    PROFILE_CHAT_ROUTER,
    chat_history_chars_per_message,
    chat_max_history_turns,
    create_chat_completion,
    format_history_snippet,
    strip_thinking_blocks,
)

_TRIAGE_SYSTEM = """Você é o agente de triagem do assistente de laboratório.
Sua única tarefa: ler a última mensagem do usuário e decidir quais especialistas devem agir.

Especialistas disponíveis:
- documents (RAG): documentos Word/PDF de experimentos, protocolos, lotes, validades, materiais.
- spreadsheets (OLAP): planilhas e CSVs com dados tabulares — contagens, médias, somas, comparações, rankings.
- ml_prediction: modelo ML treinado para afinidade Ab–Ag (AbRank, regressão log_Aff). SOMENTE quando o usuário pedir explicitamente predição/inferência.

Responda APENAS com JSON válido (sem markdown, sem explicação):
{"documents": true|false, "spreadsheets": true|false, "ml_prediction": true|false, "reason": "frase curta em pt-BR"}

Regras:
- ml_prediction=true só com pedido EXPLÍCITO de predição/inferência ("rode o modelo", "preveja log_Aff", "estime a afinidade").
- Quando ml_prediction=true, force documents=false e spreadsheets=false (a predição é autocontida).
- Para saudação/agradecimento/conversa social: tudo false e reason="saudação".
- Vários especialistas podem ser true ao mesmo tempo se a pergunta misturar tópicos."""

_TRIAGE_USER_TEMPLATE = """Histórico recente:
{history}

Última mensagem do usuário:
{message}

Decida e responda em JSON."""


@dataclass(frozen=True)
class TriageDecision:
    """Decisão do agente de triagem após classificar a mensagem."""

    use_rag: bool
    use_olap: bool
    use_ml: bool
    source: str  # "rules" | "llm" | "rules_fallback" | "social"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_rag": self.use_rag,
            "use_olap": self.use_olap,
            "use_ml": self.use_ml,
            "source": self.source,
            "reason": self.reason,
        }


_REASON_FROM_JSON = re.compile(
    r"\"reason\"\s*:\s*\"([^\"]{0,200})\"",
    re.IGNORECASE,
)


def _extract_reason(raw: str) -> str:
    cleaned = strip_thinking_blocks(raw or "")
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            reason = data.get("reason")
            if isinstance(reason, str):
                return reason.strip()[:200]
    except json.JSONDecodeError:
        pass
    match = _REASON_FROM_JSON.search(cleaned)
    if match:
        return match.group(1).strip()[:200]
    return ""


def classify_intent(
    message: str,
    *,
    history: list[dict] | None = None,
    client: OpenAI | None = None,
    model: str = "",
    documents_available: bool = False,
    spreadsheets_available: bool = False,
    ml_available: bool = False,
) -> TriageDecision:
    """
    Classifica a intenção do usuário e devolve quais Tools devem rodar.

    Curto-circuito determinístico para saudação (sem chamar LLM).
    Em caso de falha do LLM, cai no ``_rule_fallback_route`` legado.
    """
    msg = (message or "").strip()
    hist = history or []

    if _is_social_only(msg):
        return TriageDecision(
            use_rag=False,
            use_olap=False,
            use_ml=False,
            source="social",
            reason="saudação",
        )

    decision: TriageDecision | None = None
    if client is not None and model:
        decision = _classify_with_llm(msg, hist, client, model)

    if decision is None:
        fallback = _rule_fallback_route(msg)
        decision = TriageDecision(
            use_rag=fallback.use_documents,
            use_olap=fallback.use_spreadsheets,
            use_ml=fallback.use_ml,
            source="rules_fallback",
            reason="fallback por regex",
        )

    if not ml_available and decision.use_ml:
        decision = TriageDecision(
            use_rag=decision.use_rag,
            use_olap=decision.use_olap,
            use_ml=False,
            source=decision.source,
            reason=decision.reason or "modelo ML indisponível",
        )

    if not documents_available:
        decision = TriageDecision(
            use_rag=False,
            use_olap=decision.use_olap,
            use_ml=decision.use_ml,
            source=decision.source,
            reason=decision.reason,
        )
    if not spreadsheets_available:
        decision = TriageDecision(
            use_rag=decision.use_rag,
            use_olap=False,
            use_ml=decision.use_ml,
            source=decision.source,
            reason=decision.reason,
        )

    if ml_available and _ML_HINT.search(msg):
        decision = TriageDecision(
            use_rag=False,
            use_olap=False,
            use_ml=True,
            source=f"{decision.source}+ml_hint",
            reason=decision.reason or "pedido explícito de predição",
        )

    return decision


def _classify_with_llm(
    message: str,
    history: list[dict],
    client: OpenAI,
    model: str,
) -> TriageDecision | None:
    """Chama LM Studio com o system de triagem; ``None`` se a resposta for inválida."""
    ml_hint = bool(_ML_HINT.search(message))
    hist_turns = chat_max_history_turns(ml_route=ml_hint)
    hist_chars = chat_history_chars_per_message(ml_route=ml_hint)
    user_block = _TRIAGE_USER_TEMPLATE.format(
        history=format_history_snippet(
            history,
            max_turns=hist_turns,
            max_chars_per_message=hist_chars,
        ),
        message=message.strip(),
    )

    try:
        completion = create_chat_completion(
            client,
            messages=[
                {"role": "system", "content": _TRIAGE_SYSTEM},
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
    return TriageDecision(
        use_rag=docs,
        use_olap=sheets,
        use_ml=ml,
        source="llm",
        reason=_extract_reason(raw),
    )
