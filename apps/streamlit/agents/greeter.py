"""
Greeter rule-based — atalho determinístico para saudações e mensagens sociais.

Não chama LLM, não consulta RAG/OLAP/ML. Só responde quando a mensagem é
estritamente social (oi/obrigado/tchau). Para qualquer coisa fora disso,
retorna ``None`` e o Crew assume.

A regex de detecção (``SOCIAL_ONLY``) vive em ``agents.intent_rules``,
compartilhada com o Triage para garantir comportamento idêntico.
"""

from __future__ import annotations

from agents.intent_rules import is_social_only as _is_social_only_rule

_GREETING_RESPONSE = (
    "Olá! Sou seu assistente de laboratório. Posso ajudar com:\n\n"
    "- Consultar **documentos** de experimentos (lotes, validades, protocolos).\n"
    "- Analisar **planilhas** ingeridas no DuckDB (contagens, médias, comparações).\n"
    "- Rodar **predições ML** (afinidade Ab–Ag pelo modelo treinado).\n\n"
    "Qual a sua pergunta?"
)

_THANKS_RESPONSE = (
    "Por nada! Posso ajudar com mais alguma dúvida sobre os experimentos, "
    "planilhas ou predições?"
)

_FAREWELL_RESPONSE = "Até logo! Quando precisar, é só voltar."

_AFFIRMATIVE_RESPONSE = (
    "Combinado. O que você gostaria de saber dos documentos, planilhas ou predições?"
)

_GREETING_PREFIXES = (
    "oi", "olá", "ola", "hey", "hi", "hello",
    "bom dia", "boa tarde", "boa noite",
    "tudo bem", "como vai", "e aí", "eai",
)
_THANKS_PREFIXES = ("obrigad", "valeu", "brigad")
_FAREWELL_PREFIXES = ("até", "tchau", "flw")


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def is_social_only(message: str) -> bool:
    """Atalho público — delega para ``agents.intent_rules.is_social_only``."""
    return _is_social_only_rule(message)


def handle_greeting(message: str) -> str | None:
    """
    Retorna resposta determinística quando a mensagem é só social.

    Devolve ``None`` para qualquer coisa diferente de saudação — sinalizando
    para o ``runner.run_crew_chat`` que precisa rodar o Crew completo.
    """
    if not is_social_only(message):
        return None
    norm = _normalize(message)
    if norm.startswith(_THANKS_PREFIXES):
        return _THANKS_RESPONSE
    if norm.startswith(_FAREWELL_PREFIXES):
        return _FAREWELL_RESPONSE
    if norm in {"ok", "okay", "sim", "não", "nao"}:
        return _AFFIRMATIVE_RESPONSE
    if any(norm.startswith(prefix) for prefix in _GREETING_PREFIXES):
        return _GREETING_RESPONSE
    return _GREETING_RESPONSE
