"""Extensões tabulares ingeridas automaticamente no DuckDB."""

from __future__ import annotations

TABULAR_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx", ".xlsm"})

INGEST_MANIFEST_TABLE = "_olap_ingest_manifest"
