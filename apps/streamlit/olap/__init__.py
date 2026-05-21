"""
Pacote OLAP — DuckDB em arquivo no volume persistente.

Ingestão automática de planilhas dos projetos, catálogo de schema e consultas
em linguagem natural (texto → SQL read-only) para o chat Streamlit.
"""

from .connection import (
    DEMO_TABLE,
    check_duckdb,
    demo_aggregation,
    demo_detail,
    duckdb_library_version,
    olap_status,
    open_duckdb,
    seed_demo_data,
)
from .constants import INGEST_MANIFEST_TABLE, TABULAR_EXTENSIONS
from .ingest import (
    IngestStats,
    has_ingested_tables,
    list_ingested_tables,
    sync_tabular_from_scans,
    table_name_for,
)
from .nl_query import OlapQueryResult, run_nl_olap_query
from .paths import ENV_DUCKDB_DIR, duckdb_data_root, duckdb_database_path

__all__ = [
    "DEMO_TABLE",
    "ENV_DUCKDB_DIR",
    "INGEST_MANIFEST_TABLE",
    "IngestStats",
    "OlapQueryResult",
    "TABULAR_EXTENSIONS",
    "check_duckdb",
    "demo_aggregation",
    "demo_detail",
    "duckdb_data_root",
    "duckdb_database_path",
    "duckdb_library_version",
    "has_ingested_tables",
    "list_ingested_tables",
    "olap_status",
    "open_duckdb",
    "run_nl_olap_query",
    "seed_demo_data",
    "sync_tabular_from_scans",
    "table_name_for",
]
