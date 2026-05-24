"""
Caminhos persistentes para modelos ML (.pkl).
"""

from __future__ import annotations

import os
from pathlib import Path

from projects_loader import running_inside_docker

ENV_ML_DIR = "ASSISTENTE_ML_DIR"


def ml_models_root() -> Path:
    """
    Diretório onde modelos ``.pkl`` são gravados.

    Prioridade: ``ASSISTENTE_ML_DIR`` → ``/data/ml`` (Docker) → ``.ml_data/models``.
    """
    raw = os.environ.get(ENV_ML_DIR, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if running_inside_docker():
        return Path("/data/ml").resolve()
    return (Path(__file__).resolve().parent.parent / ".ml_data" / "models").resolve()


def ensure_ml_models_root() -> Path:
    root = ml_models_root()
    root.mkdir(parents=True, exist_ok=True)
    return root
