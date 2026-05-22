"""
Índices DuckDB para acelerar filtros OLAP.

Cria índices ART nas colunas de metadados usadas nas consultas geradas pelo LLM
(``_project_id``, ``_source_file``, ``_sheet_name``) e no manifesto de ingestão.
Executa ``ANALYZE`` após criar índices para melhorar o plano de consulta.
"""

from __future__ import annotations

import re

import duckdb

from .constants import INGEST_MANIFEST_TABLE

# Colunas citadas em ``nl_query._TEXT_TO_SQL_SYSTEM`` para filtros.
METADATA_FILTER_COLUMNS = ("_project_id", "_source_file", "_sheet_name")

_MANIFEST_INDEX_SPECS: tuple[tuple[str, str], ...] = (
    ("idx_olap_manifest_project", "project_id"),
    ("idx_olap_manifest_table", "table_name"),
    ("idx_olap_manifest_path", "relative_path"),
)


def _sanitize_index_name(table_name: str, column: str) -> str:
    base = re.sub(r"[^a-z0-9_]+", "_", table_name.lower()).strip("_")[:48]
    col = re.sub(r"[^a-z0-9_]+", "_", column.lower()).strip("_")[:24]
    return f"idx_{base}_{col}"[:120]


def _table_has_column(
    conn: duckdb.DuckDBPyConnection, table_name: str, column: str
) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ? AND column_name = ?
        """,
        [table_name, column],
    ).fetchone()
    return bool(row and row[0] > 0)


def ensure_table_search_indexes(
    conn: duckdb.DuckDBPyConnection, table_name: str
) -> int:
    """
    Garante índices nas colunas de metadados de uma tabela ingerida.

    Retorna quantos índices foram criados ou já existiam (via IF NOT EXISTS).
    """
    created = 0
    for col in METADATA_FILTER_COLUMNS:
        if not _table_has_column(conn, table_name, col):
            continue
        idx = _sanitize_index_name(table_name, col)
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{table_name}" ("{col}")'
        )
        created += 1
    if created:
        conn.execute(f'ANALYZE "{table_name}"')
    return created


def ensure_manifest_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Índices no manifesto para listagens e resolução de tabela por projeto/arquivo."""
    for idx_name, col in _MANIFEST_INDEX_SPECS:
        if not _table_has_column(conn, INGEST_MANIFEST_TABLE, col):
            continue
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{idx_name}" '
            f'ON "{INGEST_MANIFEST_TABLE}" ("{col}")'
        )
    conn.execute(f'ANALYZE "{INGEST_MANIFEST_TABLE}"')


def ensure_indexes_for_all_ingested_tables(conn: duckdb.DuckDBPyConnection) -> int:
    """
    Percorre o manifesto e garante índices em cada tabela de dados.

    Útil após sync incremental (arquivos inalterados não recriam tabela, mas podem
    ainda não ter índices de versões antigas do app).
    """
    if not _table_has_column(conn, INGEST_MANIFEST_TABLE, "table_name"):
        return 0
    rows = conn.execute(
        f'SELECT DISTINCT table_name FROM "{INGEST_MANIFEST_TABLE}"'
    ).fetchall()
    total = 0
    for (tbl,) in rows:
        if tbl and tbl != INGEST_MANIFEST_TABLE:
            total += ensure_table_search_indexes(conn, str(tbl))
    return total
