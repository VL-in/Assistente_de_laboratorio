"""
Reranking de candidatos RAG com cross-encoder.

A busca vetorial (bi-encoder) é rápida, mas o score de similaridade nem sempre
coloca os trechos mais úteis no topo. O rerank reavalia pares (pergunta, trecho)
com um modelo cross-encoder — mais lento, porém mais preciso — sobre um pool
maior de candidatos antes de montar o contexto para o LLM.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Multilingual MS MARCO (14 idiomas, incl. pt-BR) — rerank cross-encoder.
# O slug L12 é o publicado em https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
# (L6-H384 não existe no Hub e fazia o rerank falhar silenciosamente).
RERANKER_MODEL_ID = os.environ.get(
    "RAG_RERANK_MODEL_ID",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
).strip()

ENV_RERANK_ENABLED = "RAG_RERANK_ENABLED"
ENV_RERANK_RETRIEVE_K = "RAG_RERANK_RETRIEVE_K"
DEFAULT_RERANK_MULTIPLIER = 4
DEFAULT_RERANK_MIN_CANDIDATES = 20
RERANK_TEXT_MAX_CHARS = 2000


def env_rerank_enabled() -> bool:
    """``RAG_RERANK_ENABLED=0`` desliga o rerank (apenas escape hatch em dev/CI)."""
    return os.environ.get(ENV_RERANK_ENABLED, "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def default_retrieve_k(top_k: int, override: int | None = None) -> int:
    """
    Quantos candidatos buscar antes do rerank.

    Prioridade: ``override`` explícito > ``RAG_RERANK_RETRIEVE_K`` > heurística
    ``max(top_k * 4, 20)``.
    """
    tk = max(int(top_k), 1)
    if override is not None and int(override) > 0:
        return max(int(override), tk)

    env_raw = os.environ.get(ENV_RERANK_RETRIEVE_K, "").strip()
    if env_raw:
        try:
            env_k = int(env_raw)
            if env_k > 0:
                return max(env_k, tk)
        except ValueError:
            pass

    return max(tk * DEFAULT_RERANK_MULTIPLIER, DEFAULT_RERANK_MIN_CANDIDATES)


def hit_text_for_rerank(hit: dict) -> str:
    """Texto do chunk usado pelo cross-encoder (com truncagem por limite de tokens)."""
    body = (hit.get("cited") or hit.get("text") or "").strip()
    if len(body) > RERANK_TEXT_MAX_CHARS:
        return body[:RERANK_TEXT_MAX_CHARS]
    return body


def load_reranker(model_id: str | None = None) -> Any:
    """Carrega um ``CrossEncoder`` do sentence-transformers."""
    from sentence_transformers import CrossEncoder

    mid = (model_id or RERANKER_MODEL_ID).strip()
    if not mid:
        raise ValueError("RAG_RERANK_MODEL_ID vazio.")
    return CrossEncoder(mid)


@dataclass(frozen=True)
class RerankerLoadResult:
    """Resultado do carregamento do cross-encoder (modelo + erro opcional)."""

    model: object | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.model is not None and self.error is None


def load_reranker_safe(model_id: str | None = None) -> RerankerLoadResult:
    """Carrega o reranker sem propagar exceção — útil na UI e no cache Streamlit."""
    if not env_rerank_enabled():
        return RerankerLoadResult(None, "RAG_RERANK_ENABLED=0")
    try:
        return RerankerLoadResult(load_reranker(model_id), None)
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("Falha ao carregar reranker %s: %s", RERANKER_MODEL_ID, msg)
        return RerankerLoadResult(None, msg)


def rerank_hits(
    query: str,
    hits: list[dict],
    *,
    reranker: object,
    top_k: int,
) -> list[dict]:
    """
    Reordena hits pelo score do cross-encoder e devolve os ``top_k`` melhores.

    Preserva o score da busca vetorial em ``retrieval_score`` e substitui
    ``score`` pelo score do rerank (para exibição e metadados do handoff).
    """
    q = (query or "").strip()
    if not q or not hits or top_k <= 0:
        return hits[: max(top_k, 0)]

    pairs = [(q, hit_text_for_rerank(h)) for h in hits]
    raw_scores = reranker.predict(pairs)  # type: ignore[attr-defined]

    reranked: list[dict] = []
    for hit, score in zip(hits, raw_scores):
        out = dict(hit)
        if "retrieval_score" not in out:
            out["retrieval_score"] = hit.get("score")
        out["rerank_score"] = float(score)
        out["score"] = float(score)
        out["rerank_applied"] = True
        reranked.append(out)

    reranked.sort(key=lambda h: h.get("score") or 0.0, reverse=True)
    return reranked[: int(top_k)]
