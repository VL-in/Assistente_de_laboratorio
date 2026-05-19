"""Pipeline RAG local: extração, chunking, índice txtai."""

from .index_txtai import (
    EMBEDDING_MODEL_ID,
    BuildStats,
    build_index,
    format_context_for_llm,
    index_mtime,
    index_ready,
    search_chunks,
)
from .manifest import manifest_exists, manifest_path
from .paths import ENV_TXTAI_DIR, txtai_data_root, txtai_index_path

__all__ = [
    "EMBEDDING_MODEL_ID",
    "BuildStats",
    "ENV_TXTAI_DIR",
    "build_index",
    "format_context_for_llm",
    "index_mtime",
    "index_ready",
    "manifest_exists",
    "manifest_path",
    "search_chunks",
    "txtai_data_root",
    "txtai_index_path",
]
