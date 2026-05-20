"""
Pacote OLAP — DuckDB em arquivo no volume persistente.

Exporta resolução de caminhos, conexão, seed de demonstração e consultas
read-only usadas pela aba OLAP e pelo Diagnóstico do ``app.py``.
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
from .paths import ENV_DUCKDB_DIR, duckdb_data_root, duckdb_database_path

__all__ = [
    "DEMO_TABLE",
    "ENV_DUCKDB_DIR",
    "check_duckdb",
    "demo_aggregation",
    "demo_detail",
    "duckdb_data_root",
    "duckdb_database_path",
    "duckdb_library_version",
    "olap_status",
    "open_duckdb",
    "seed_demo_data",
]
