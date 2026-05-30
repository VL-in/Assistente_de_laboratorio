"""
Configuração da busca híbrida RAG (semântica + BM25).

O txtai combina dois índices quando ``hybrid=True`` na configuração:
  - **Denso (semântico):** embeddings E5 via TEI — captura significado e sinônimos.
  - **Esparsa (lexical):** BM25 sobre o texto bruto — favorece termos exatos,
    incluindo nomes compostos como ``tampão de amostra``.

O parâmetro ``weight`` (α) controla o peso do índice denso na fusão final.
``1.0`` = só semântica; ``0.0`` = só BM25; padrão ``0.4`` favorece um pouco
a correspondência lexical para termos técnicos de laboratório.
"""

from __future__ import annotations

import os

ENV_HYBRID_ENABLED = "RAG_HYBRID_ENABLED"
ENV_HYBRID_WEIGHT = "RAG_HYBRID_WEIGHT"
DEFAULT_HYBRID_WEIGHT = 0.4


def env_hybrid_enabled() -> bool:
    """``RAG_HYBRID_ENABLED=0`` desliga BM25 (só busca semântica)."""
    raw = os.environ.get(ENV_HYBRID_ENABLED, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def hybrid_dense_weight(override: float | None = None) -> float:
    """
    Peso α do índice denso na fusão híbrida (0–1).

    Prioridade: ``override`` > ``RAG_HYBRID_WEIGHT`` > ``DEFAULT_HYBRID_WEIGHT``.
    """
    if override is not None:
        return _clamp_weight(float(override))
    raw = os.environ.get(ENV_HYBRID_WEIGHT, "").strip()
    if raw:
        try:
            return _clamp_weight(float(raw))
        except ValueError:
            pass
    return DEFAULT_HYBRID_WEIGHT


def _clamp_weight(value: float) -> float:
    return max(0.0, min(1.0, value))
