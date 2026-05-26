"""
Greeter rule-based — atalho determinístico para saudações e mensagens sociais.

Não chama LLM, não consulta RAG/OLAP/ML. Só responde quando a mensagem é
estritamente social (oi/obrigado/tchau). Para qualquer coisa fora disso,
retorna ``None`` e o Crew assume.
"""

from __future__ import annotations

import re

# Lista compatível com ``chat_router._SOCIAL_ONLY``; replicada aqui para que o
# Greeter não dependa do roteador legado quando ele for aposentado.
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
    """Replica a heurística do ``chat_router`` para curto-circuito do Crew."""
    msg = (message or "").strip()
    if not msg or len(msg) > 80:
        return False
    return _SOCIAL_ONLY.match(msg) is not None


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
