"""
Regras determinísticas de classificação de intenção do chat.

Este módulo concentra padrões regex e helpers puros (sem LLM) usados pelos
agentes ``triage`` e ``greeter``. Ele substitui as regras antigamente
hospedadas em ``chat_router`` (caminho legado, aposentado em 2026-05).

Como o LLM remoto (Triage) pode falhar (rede, formato), mantemos um
fallback rule-based aqui — mais previsível e barato.
"""

from __future__ import annotations

import json
import re

# ── Padrões regex compartilhados ────────────────────────────────────────────

# Saudações, agradecimentos, confirmações curtas. Casa quando a mensagem é
# *exclusivamente* social (curta e sem conteúdo de laboratório).
SOCIAL_ONLY = re.compile(
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

# Vocabulário típico de pergunta sobre documentos de experimentos/protocolos.
LAB_DOC_HINT = re.compile(
    r"\b(?:"
    r"lote|validade|reagente|insumo|ensaio|elisa|experimento|"
    r"documento|protocolo|planejamento|fabricante|"
    r"material|amostra|dilui|placa|coating|"
    r"usamos|utilizamos|qual\s+foi|quando\s+foi|onde\s+está"
    r")\b",
    re.IGNORECASE,
)

# Pedido explícito de predição/inferência via modelo ML treinado.
ML_HINT = re.compile(
    r"\b(?:"
    r"predi[çc][ãa]o|predizer|prever|previs[ãa]o|infer[êe]ncia|inferir|"
    r"modelo\s+ml|algoritmo\s+ml|machine\s+learning|"
    r"log_aff|log\s+aff|afinidade\s+ab|abrank|"
    r"estim(?:e|ar)\s+(?:a\s+)?afinidade|rodar\s+o\s+modelo"
    r")\b",
    re.IGNORECASE,
)

# Vocabulário de análise tabular: contagens, médias, agrupamentos.
TABULAR_HINT = re.compile(
    r"\b(?:"
    r"planilha|tabela|csv|xlsx|"
    r"quantos|quantas|média|media|soma|total|contagem|"
    r"comparar|comparação|comparacao|ranking|agrupar|"
    r"por\s+projeto|linhas|colunas|registros"
    r")\b",
    re.IGNORECASE,
)

# Objeto JSON candidato — usado para resgatar JSON envolto em narrativa do LLM.
_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)

# ── Helpers públicos ────────────────────────────────────────────────────────


def is_social_only(message: str) -> bool:
    """``True`` quando a mensagem inteira é uma saudação/confirmação curta."""
    msg = (message or "").strip()
    if not msg or len(msg) > 80:
        return False
    return SOCIAL_ONLY.match(msg) is not None


def parse_router_json(raw: str) -> tuple[bool, bool, bool] | None:
    """
    Lê a resposta JSON do classificador LLM e devolve ``(docs, sheets, ml)``.

    Aceita JSON puro ou JSON envolto em markdown/narrativa (extrai o primeiro
    objeto ``{...}`` encontrado). Retorna ``None`` se a resposta não puder ser
    interpretada — chamadores devem cair no ``rule_fallback``.

    O import tardio de ``strip_thinking_blocks`` evita ciclo de import com
    ``qwen35_inference`` (que não depende deste módulo).
    """
    from qwen35_inference import strip_thinking_blocks

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


def rule_fallback(message: str) -> tuple[bool, bool, bool]:
    """
    Fallback conservador quando o LLM do Triage falha ou retorna inválido.

    Retorna ``(use_rag, use_olap, use_ml)``. ``use_ml`` tem prioridade —
    quando casa, zera os outros (predição é autocontida).

    Mensagens muito curtas (≤ 3 tokens) sem nenhum dos hints são tratadas
    como conversa social: tudo ``False``.
    """
    msg = (message or "").strip()
    if is_social_only(msg):
        return False, False, False
    ml = bool(ML_HINT.search(msg))
    if ml:
        return False, False, True
    docs = bool(LAB_DOC_HINT.search(msg))
    sheets = bool(TABULAR_HINT.search(msg))
    if not docs and not sheets and len(msg.split()) <= 3:
        return False, False, False
    return docs, sheets, False
