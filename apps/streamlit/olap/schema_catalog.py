"""
Catálogo de schema DuckDB para prompts de texto-para-SQL.
"""

from __future__ import annotations

from .connection import open_duckdb
from .constants import INGEST_MANIFEST_TABLE
from .ingest import database_exists_quick


_EXCLUDED_TABLES = frozenset({"demo_experimentos"})


def _user_tables(conn) -> list[str]:
    """Retorna tabelas de planilhas ingeridas (exclui internas e demo)."""
    rows = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    ).fetchall()
    return [
        r[0]
        for r in rows
        if not r[0].startswith("_") and r[0] not in _EXCLUDED_TABLES
    ]


def build_schema_catalog_text(*, sample_rows: int = 2) -> str:
    """
    Texto legível com tabelas, colunas e amostra para o LLM gerar SQL.
    """
    if not database_exists_quick():
        return "(Nenhum banco DuckDB materializado ainda.)"

    conn = open_duckdb(read_only=True)
    try:
        tables = _user_tables(conn)
        if not tables:
            return "(Banco existe, mas não há tabelas de planilhas ingeridas.)"

        parts: list[str] = []
        for tbl in tables:
            cols = conn.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = ?
                ORDER BY ordinal_position
                """,
                [tbl],
            ).fetchall()
            col_lines = ", ".join(f"{c} ({t})" for c, t in cols)
            parts.append(f"TABLE \"{tbl}\": {col_lines}")

            try:
                sample = conn.execute(
                    f'SELECT * FROM "{tbl}" LIMIT ?',
                    [sample_rows],
                ).fetchdf()
                if not sample.empty:
                    parts.append(sample.to_string(index=False))
            except Exception:
                pass

        manifest_note = ""
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {INGEST_MANIFEST_TABLE}").fetchone()
            if n and n[0]:
                manifest_note = (
                    f"\nMetadados de origem: tabela {INGEST_MANIFEST_TABLE} "
                    "(project_id, relative_path, sheet_name → table_name)."
                )
        except Exception:
            pass

        intro = (
            "Colunas em toda tabela de planilha: "
            "_project_id (projeto), _source_file (caminho relativo), _sheet_name (aba ou 'csv'), "
            "demais colunas vêm do arquivo.\n"
        )
        return intro + "\n\n".join(parts) + manifest_note
    finally:
        conn.close()
