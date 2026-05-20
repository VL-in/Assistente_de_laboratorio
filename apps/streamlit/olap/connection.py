"""
Conexão DuckDB e dados de demonstração (olá mundo OLAP).

O banco fica em arquivo no volume persistente (``assistente_lab.duckdb``).
Consultas da UI usam conexão read-only quando possível; criação do schema
demo usa escrita explícita.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from .paths import DUCKDB_DATABASE_FILENAME, duckdb_database_path, ensure_duckdb_parent_exists

DEMO_TABLE = "demo_experimentos"

_DEMO_ROWS: list[tuple[str, str, str, int, str]] = [
    ("ELISA_2024", "Ensaio_01.docx", "anticorpo", 2, "2024-03-15"),
    ("ELISA_2024", "Ensaio_02.docx", "reagente", 5, "2024-04-02"),
    ("ELISA_2024", "Ensaio_02.docx", "placa", 1, "2024-04-02"),
    ("Anticorpos", "Protocolo.docx", "anticorpo", 1, "2024-05-10"),
]


def open_duckdb(*, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Abre (ou cria) o arquivo DuckDB no volume configurado.

    Garante que o diretório pai exista antes de conectar. Em modo read-only,
    o arquivo precisa já existir.
    """
    path = duckdb_database_path()
    if not read_only:
        ensure_duckdb_parent_exists()
    return duckdb.connect(str(path), read_only=read_only)


def database_exists() -> bool:
    return duckdb_database_path().is_file()


def duckdb_library_version() -> str:
    return str(duckdb.__version__)


def check_duckdb() -> tuple[bool, str]:
    """
    Smoke test: conecta e executa ``SELECT 1``.

    Retorna ``(ok, detalhe)`` para a aba Diagnóstico.
    """
    try:
        if database_exists():
            conn = open_duckdb(read_only=True)
        else:
            conn = open_duckdb(read_only=False)
        try:
            row = conn.execute("SELECT 1 AS ok").fetchone()
            if row and row[0] == 1:
                return True, f"DuckDB {duckdb_library_version()} — SELECT 1 ok"
            return False, "Consulta de teste retornou resultado inesperado."
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — diagnóstico na UI
        return False, f"{type(exc).__name__}: {exc}"


def demo_table_exists(conn: duckdb.DuckDBPyConnection) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [DEMO_TABLE],
    ).fetchone()
    return bool(row and row[0] > 0)


def seed_demo_data(*, force: bool = False) -> bool:
    """
    Cria a tabela de exemplo e insere linhas se ainda não existir.

    Retorna ``True`` se inseriu dados nesta chamada, ``False`` se já havia tabela
    (e ``force`` é falso).
    """
    conn = open_duckdb(read_only=False)
    try:
        if demo_table_exists(conn) and not force:
            return False
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE {DEMO_TABLE} (
                projeto_id VARCHAR,
                documento VARCHAR,
                tipo_insumo VARCHAR,
                quantidade INTEGER,
                data_planejamento DATE
            )
            """
        )
        conn.executemany(
            f"""
            INSERT INTO {DEMO_TABLE}
            (projeto_id, documento, tipo_insumo, quantidade, data_planejamento)
            VALUES (?, ?, ?, ?, ?::DATE)
            """,
            _DEMO_ROWS,
        )
        return True
    finally:
        conn.close()


def demo_aggregation() -> pd.DataFrame:
    """Agregação de exemplo: soma de quantidades por projeto (read-only)."""
    empty = pd.DataFrame(
        columns=["projeto_id", "total_quantidade", "tipos_insumo_distintos"]
    )
    if not database_exists():
        return empty
    conn = open_duckdb(read_only=True)
    try:
        if not demo_table_exists(conn):
            return empty
        return conn.execute(
            f"""
            SELECT
                projeto_id,
                SUM(quantidade) AS total_quantidade,
                COUNT(DISTINCT tipo_insumo) AS tipos_insumo_distintos
            FROM {DEMO_TABLE}
            GROUP BY projeto_id
            ORDER BY projeto_id
            """
        ).fetchdf()
    finally:
        conn.close()


def demo_detail(limit: int = 50) -> pd.DataFrame:
    """Linhas brutas da tabela demo (read-only)."""
    if not database_exists():
        return pd.DataFrame()
    conn = open_duckdb(read_only=True)
    try:
        if not demo_table_exists(conn):
            return pd.DataFrame()
        return conn.execute(
            f"SELECT * FROM {DEMO_TABLE} ORDER BY data_planejamento, projeto_id LIMIT ?",
            [limit],
        ).fetchdf()
    finally:
        conn.close()


def olap_status() -> dict[str, Any]:
    """Resumo para diagnóstico e cabeçalho da aba OLAP."""
    root = duckdb_database_path().parent
    demo_ready = False
    if database_exists():
        conn = open_duckdb(read_only=True)
        try:
            demo_ready = demo_table_exists(conn)
        finally:
            conn.close()
    return {
        "library_version": duckdb_library_version(),
        "data_root": root,
        "database_path": duckdb_database_path(),
        "database_filename": DUCKDB_DATABASE_FILENAME,
        "database_exists": database_exists(),
        "demo_ready": demo_ready,
    }
