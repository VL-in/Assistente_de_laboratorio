"""Caminhos persistentes para índice txtai e defaults por ambiente."""

from __future__ import annotations

import os
from pathlib import Path

from projects_loader import running_inside_docker

ENV_TXTAI_DIR = "ASSISTENTE_TXTAI_DIR"


def txtai_data_root() -> Path:
    """
    Diretório base para dados txtai (volume Docker ou pasta local de desenvolvimento).
    O índice salvo fica em ``txtai_index_path()`` dentro deste root.
    """
    raw = os.environ.get(ENV_TXTAI_DIR, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if running_inside_docker():
        return Path("/data/txtai").resolve()
    # Dev local: ao lado do app, ignorado pelo git via .gitignore recomendado
    return (Path(__file__).resolve().parent.parent / ".txtai_data").resolve()


def txtai_index_path() -> Path:
    """Diretório passado a ``Embeddings.save`` / ``load``."""
    return txtai_data_root() / "embeddings_index"


def ensure_txtai_parent_exists() -> None:
    """Garante que o diretório pai exista (mkdir -p)."""
    txtai_data_root().mkdir(parents=True, exist_ok=True)
