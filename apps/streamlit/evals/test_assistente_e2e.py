"""
Testes end-to-end DeepEval do Assistente de Lab.

Executar (a partir da raiz do repositório)::

    pip install -r apps/streamlit/requirements-evals.txt
    deepeval test run apps/streamlit/evals/test_assistente_e2e.py

Variáveis úteis:

- ``EVAL_LIMIT`` — limita quantos goldens rodam (padrão: 3, smoke test).
- ``EVAL_CATEGORY`` — filtra categoria (rag, olap, ml, …).
- ``EVAL_SKIP_METRICS=1`` — só gera respostas, sem juiz LLM.

Exemplo rápido::

    set EVAL_LIMIT=2
    deepeval test run apps/streamlit/evals/test_assistente_e2e.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
STREAMLIT_ROOT = EVALS_DIR.parent
for path in (STREAMLIT_ROOT, EVALS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Bootstrap eval (Langfuse off + throttle) antes de importar harness/llm_config.
import run_assistente_eval  # noqa: E402, F401


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def test_assistente_lab_e2e() -> None:
    """Smoke test end-to-end: gera respostas e avalia com DeepEval."""
    from run_assistente_eval import run_evaluation

    limit = _env_int("EVAL_LIMIT", 3)
    category = os.environ.get("EVAL_CATEGORY", "").strip() or None
    skip_metrics = os.environ.get("EVAL_SKIP_METRICS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    require_ready = os.environ.get("EVAL_REQUIRE_READY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    exit_code = run_evaluation(
        limit=limit,
        category=category,
        skip_metrics=skip_metrics,
        require_ready=require_ready,
    )
    assert exit_code == 0, f"Eval falhou com código {exit_code}"
