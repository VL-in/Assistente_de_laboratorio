"""
Caminhos persistentes para modelos ML (.pkl).
"""

from __future__ import annotations

import os
from pathlib import Path

from projects_loader import running_inside_docker

ENV_ML_DIR = "ASSISTENTE_ML_DIR"
ENV_ML_CHAT_MODEL = "ASSISTENTE_ML_CHAT_MODEL"
DEFAULT_CHAT_MODEL_FILENAME = "modelo_20260524_224734_04768.pkl"


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


def chat_ml_model_path() -> Path:
    """
    Caminho do ``.pkl`` usado nas predições via chat.

    Prioridade: ``ASSISTENTE_ML_CHAT_MODEL`` → ``{ml_models_root}/modelo_20260524_224734_04768.pkl``.
    """
    raw = os.environ.get(ENV_ML_CHAT_MODEL, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (ml_models_root() / DEFAULT_CHAT_MODEL_FILENAME).resolve()


def chat_ml_model_available() -> bool:
    return chat_ml_model_path().is_file()


ENV_ESM_CACHE = "HF_HOME"
DEFAULT_ESM_CACHE_SUBDIR = "huggingface"


def esm_cache_root() -> Path:
    """
    Cache Hugging Face / ESM-2.

    Prioridade: ``HF_HOME`` → ``{ml_models_root}/huggingface``.
    """
    raw = os.environ.get("HF_HOME", "").strip() or os.environ.get("TRANSFORMERS_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (ml_models_root() / DEFAULT_ESM_CACHE_SUBDIR).resolve()


def ensure_esm_cache_root() -> Path:
    root = esm_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root
