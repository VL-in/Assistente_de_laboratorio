"""
Configuração de ambiente para avaliações DeepEval.

Importe (ou chame ``configure_eval_env``) **antes** de ``llm_config`` / ``harness``,
para evitar o patch Langfuse em respostas 429 e aplicar throttle no OpenRouter free.
"""

from __future__ import annotations

import os


def configure_eval_env(*, case_interval_s: float | None = None) -> None:
    """Defaults conservadores para evals end-to-end."""
    os.environ["LANGFUSE_ENABLED"] = "0"
    os.environ.setdefault("LLM_MIN_REQUEST_INTERVAL_S", "12")
    os.environ.setdefault("LLM_RETRY_MAX_ATTEMPTS", "10")
    os.environ.setdefault("LLM_RETRY_BASE_DELAY_S", "25")
    if case_interval_s is not None:
        os.environ["EVAL_CASE_INTERVAL_S"] = str(max(0.0, case_interval_s))


def eval_case_interval_s() -> float:
    raw = os.environ.get("EVAL_CASE_INTERVAL_S", "0") or "0"
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def eval_metrics_max_concurrent() -> int:
    """Casos avaliados em paralelo pelo DeepEval (padrão 1 = serial)."""
    raw = os.environ.get("EVAL_METRICS_MAX_CONCURRENT", "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def eval_metrics_throttle_s() -> float:
    """Pausa (s) entre casos na fase de métricas LLM-as-judge."""
    raw = os.environ.get("EVAL_METRICS_THROTTLE_S", "15").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 15.0
