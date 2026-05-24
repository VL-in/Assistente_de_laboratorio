"""
Caminhos persistentes para modelos ML e dataset padrão (projeto 253 Dengue).
"""

from __future__ import annotations

import os
from pathlib import Path

from projects_loader import ENV_PROJETOS_ROOT, projetos_root_from_env, running_inside_docker

ENV_ML_DIR = "ASSISTENTE_ML_DIR"
ENV_ML_DENGUE_RESULTS = "ASSISTENTE_ML_DENGUE_RESULTS"

DENGUE_PROJECT_FOLDER = "253 - ELISA indireto Dengue"
DENGUE_RESULTS_SUBDIR = "results"

DEFAULT_DENGUE_RESULTS_WINDOWS = Path(
    r"D:\Vanessa\AI_project\Projetos\253 - ELISA indireto Dengue\results"
)


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


def resolve_dengue_results_dir() -> Path:
    """
    Pasta ``results`` do projeto 253.

    Prioridade:
    1. ``ASSISTENTE_ML_DENGUE_RESULTS`` (caminho explícito)
    2. ``{projetos_root}/{253 - ELISA indireto Dengue}/results``
    3. Caminho Windows de desenvolvimento (fallback local)
    """
    explicit = os.environ.get(ENV_ML_DENGUE_RESULTS, "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    projetos_root = projetos_root_from_env()
    candidate = projetos_root / DENGUE_PROJECT_FOLDER / DENGUE_RESULTS_SUBDIR
    if candidate.is_dir():
        return candidate.resolve()

    env_root = os.environ.get(ENV_PROJETOS_ROOT, "").strip()
    if env_root:
        env_candidate = Path(env_root).expanduser() / DENGUE_PROJECT_FOLDER / DENGUE_RESULTS_SUBDIR
        if env_candidate.is_dir():
            return env_candidate.resolve()

    if DEFAULT_DENGUE_RESULTS_WINDOWS.is_dir():
        return DEFAULT_DENGUE_RESULTS_WINDOWS.resolve()

    return candidate.resolve()
