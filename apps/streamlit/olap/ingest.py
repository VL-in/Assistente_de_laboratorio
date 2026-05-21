"""
Ingestão de planilhas dos projetos para tabelas DuckDB.

Varre os mesmos diretórios que ``projects_loader`` (via ``ProjectScan``),
carrega ``.csv``, ``.xlsx`` e ``.xlsm`` e materializa uma tabela por aba/planilha.
Metadados de rastreio ficam em ``_olap_ingest_manifest``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from projects_loader import ProjectScan, ScannedFile, compute_file_sha256

from .connection import open_duckdb
from .constants import INGEST_MANIFEST_TABLE, TABULAR_EXTENSIONS

_METADATA_COLS = ("_project_id", "_source_file", "_sheet_name")


@dataclass
class IngestStats:
    """Resumo de uma sincronização tabular → DuckDB."""

    tables_created: int = 0
    tables_updated: int = 0
    tables_removed: int = 0
    files_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def tables_touched(self) -> int:
        return self.tables_created + self.tables_updated


def _sanitize_identifier(raw: str, *, max_len: int = 48) -> str:
    """Converte texto livre em identificador SQL seguro (minúsculas, _)."""
    s = raw.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "x"
    if s[0].isdigit():
        s = f"c_{s}"
    return s[:max_len]


def _sanitize_column(name: str) -> str:
    col = _sanitize_identifier(str(name), max_len=56)
    if col in _METADATA_COLS:
        col = f"col_{col}"
    return col


def _deduplicate_columns(columns: list[str]) -> list[str]:
    """Adiciona sufixo numérico a colunas duplicadas após sanitização."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result


def ingest_key(project_id: str, relative_path: str, sheet_name: str) -> str:
    return f"{project_id}|{relative_path}|{sheet_name}"


def table_name_for(project_id: str, relative_path: str, sheet_name: str) -> str:
    """Nome estável: ``p_{projeto}__{caminho}__{aba}``."""
    rel = Path(relative_path)
    stem = _sanitize_identifier(rel.stem)
    parent = _sanitize_identifier(rel.parent.as_posix().replace("/", "_") if rel.parent.parts else "")
    sheet = _sanitize_identifier(sheet_name or "csv")
    proj = _sanitize_identifier(project_id, max_len=32)
    parts = [f"p_{proj}"]
    if parent and parent != "x":
        parts.append(parent)
    parts.extend([stem, sheet])
    return "__".join(parts)[:120]


def _file_content_hash(scanned: ScannedFile) -> str:
    if scanned.content_hash_sha256:
        return scanned.content_hash_sha256
    return compute_file_sha256(scanned.absolute_path)


def _ensure_manifest(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {INGEST_MANIFEST_TABLE} (
            ingest_key VARCHAR PRIMARY KEY,
            project_id VARCHAR,
            relative_path VARCHAR,
            sheet_name VARCHAR,
            table_name VARCHAR,
            content_hash VARCHAR,
            row_count BIGINT,
            ingested_at TIMESTAMP
        )
        """
    )


def _load_manifest(conn: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    if not conn.execute(
        """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [INGEST_MANIFEST_TABLE],
    ).fetchone()[0]:
        return {}
    rows = conn.execute(
        f"""
        SELECT ingest_key, project_id, relative_path, sheet_name, table_name,
               content_hash, row_count
        FROM {INGEST_MANIFEST_TABLE}
        """
    ).fetchall()
    return {
        r[0]: {
            "project_id": r[1],
            "relative_path": r[2],
            "sheet_name": r[3],
            "table_name": r[4],
            "content_hash": r[5],
            "row_count": r[6],
        }
        for r in rows
    }


def _read_csv(path: Path) -> pd.DataFrame | None:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def _read_excel_sheets(path: Path) -> dict[str, pd.DataFrame]:
    try:
        book = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception:
        return {}
    return {name: df for name, df in book.items() if df is not None and not df.empty}


def _prepare_dataframe(
    df: pd.DataFrame,
    *,
    project_id: str,
    relative_path: str,
    sheet_name: str,
) -> pd.DataFrame:
    out = df.copy()
    sanitized = [_sanitize_column(c) for c in out.columns]
    out.columns = _deduplicate_columns(sanitized)
    out.insert(0, "_sheet_name", sheet_name)
    out.insert(0, "_source_file", relative_path)
    out.insert(0, "_project_id", project_id)
    return out


def _register_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    df: pd.DataFrame,
) -> None:
    tmp = f"_tmp_{table_name}"
    conn.register(tmp, df)
    try:
        conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM "{tmp}"')
    finally:
        try:
            conn.unregister(tmp)
        except duckdb.CatalogException:
            pass


def _drop_table(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')


def _iter_tabular_files(scans: list[ProjectScan]) -> list[ScannedFile]:
    out: list[ScannedFile] = []
    for scan in scans:
        for f in scan.files:
            if f.absolute_path.suffix.lower() in TABULAR_EXTENSIONS:
                out.append(f)
    return out


def sync_tabular_from_scans(scans: list[ProjectScan]) -> IngestStats:
    """
    Sincroniza planilhas do inventário com o DuckDB.

    Arquivos novos ou com hash diferente recriam a tabela; arquivos removidos do
    inventário apagam tabela e registro no manifesto.
    """
    stats = IngestStats()
    conn = open_duckdb(read_only=False)
    try:
        _ensure_manifest(conn)
        manifest = _load_manifest(conn)
        current_keys: set[str] = set()

        for scanned in _iter_tabular_files(scans):
            path = scanned.absolute_path
            try:
                content_hash = _file_content_hash(scanned)
            except OSError as exc:
                stats.errors.append(f"{scanned.relative_path}: hash — {exc}")
                stats.files_skipped += 1
                continue

            suffix = path.suffix.lower()
            sheets: dict[str, pd.DataFrame] = {}
            if suffix == ".csv":
                df = _read_csv(path)
                if df is None or df.empty:
                    stats.files_skipped += 1
                    continue
                sheets["csv"] = df
            else:
                sheets = _read_excel_sheets(path)
                if not sheets:
                    stats.files_skipped += 1
                    continue

            for sheet_name, raw_df in sheets.items():
                key = ingest_key(scanned.project_id, scanned.relative_path, sheet_name)
                current_keys.add(key)
                tbl = table_name_for(scanned.project_id, scanned.relative_path, sheet_name)
                prev = manifest.get(key)
                if prev and prev.get("content_hash") == content_hash:
                    continue

                try:
                    df = _prepare_dataframe(
                        raw_df,
                        project_id=scanned.project_id,
                        relative_path=scanned.relative_path,
                        sheet_name=sheet_name,
                    )
                    _register_table(conn, tbl, df)
                    row_count = len(df)
                    conn.execute(
                        f"DELETE FROM {INGEST_MANIFEST_TABLE} WHERE ingest_key = ?",
                        [key],
                    )
                    conn.execute(
                        f"""
                        INSERT INTO {INGEST_MANIFEST_TABLE}
                        (ingest_key, project_id, relative_path, sheet_name, table_name,
                         content_hash, row_count, ingested_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        [
                            key,
                            scanned.project_id,
                            scanned.relative_path,
                            sheet_name,
                            tbl,
                            content_hash,
                            row_count,
                        ],
                    )
                    if prev:
                        stats.tables_updated += 1
                    else:
                        stats.tables_created += 1
                except Exception as exc:  # noqa: BLE001
                    stats.errors.append(f"{scanned.relative_path} [{sheet_name}]: {exc}")

        stale = [k for k in manifest if k not in current_keys]
        for key in stale:
            tbl = manifest[key]["table_name"]
            _drop_table(conn, tbl)
            conn.execute(
                f"DELETE FROM {INGEST_MANIFEST_TABLE} WHERE ingest_key = ?",
                [key],
            )
            stats.tables_removed += 1

        return stats
    finally:
        conn.close()


def _manifest_table_exists(conn: duckdb.DuckDBPyConnection) -> bool:
    """Verifica se a tabela de manifesto existe (sem tentar criar)."""
    row = conn.execute(
        """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [INGEST_MANIFEST_TABLE],
    ).fetchone()
    return bool(row and row[0] > 0)


def list_ingested_tables() -> pd.DataFrame:
    """Catálogo de tabelas ingeridas (manifesto)."""
    empty = pd.DataFrame(
        columns=[
            "project_id",
            "relative_path",
            "sheet_name",
            "table_name",
            "row_count",
        ]
    )
    if not database_exists_quick():
        return empty
    conn = open_duckdb(read_only=True)
    try:
        if not _manifest_table_exists(conn):
            return empty
        return conn.execute(
            f"""
            SELECT project_id, relative_path, sheet_name, table_name, row_count
            FROM {INGEST_MANIFEST_TABLE}
            ORDER BY project_id, relative_path, sheet_name
            """
        ).fetchdf()
    finally:
        conn.close()


def has_ingested_tables() -> bool:
    if not database_exists_quick():
        return False
    conn = open_duckdb(read_only=True)
    try:
        if not _manifest_table_exists(conn):
            return False
        n = conn.execute(f"SELECT COUNT(*) FROM {INGEST_MANIFEST_TABLE}").fetchone()[0]
        return bool(n and n > 0)
    except Exception:
        return False
    finally:
        conn.close()


def database_exists_quick() -> bool:
    from .paths import duckdb_database_path

    return duckdb_database_path().is_file()
