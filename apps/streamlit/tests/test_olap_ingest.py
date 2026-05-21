"""Testes de ingestão tabular e validação SQL OLAP."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olap.ingest import (  # noqa: E402
    _deduplicate_columns,
    has_ingested_tables,
    ingest_key,
    list_ingested_tables,
    sync_tabular_from_scans,
    table_name_for,
)
from olap.nl_query import (  # noqa: E402
    _extract_sql,
    _strip_think_blocks,
    execute_sql,
    validate_readonly_sql,
)
from projects_loader import scan_project  # noqa: E402


class OlapIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._duck_prev = os.environ.get("ASSISTENTE_DUCKDB_DIR")
        os.environ["ASSISTENTE_DUCKDB_DIR"] = self._tmpdir.name

    def tearDown(self) -> None:
        if self._duck_prev is None:
            os.environ.pop("ASSISTENTE_DUCKDB_DIR", None)
        else:
            os.environ["ASSISTENTE_DUCKDB_DIR"] = self._duck_prev
        self._tmpdir.cleanup()

    def test_table_name_stable(self) -> None:
        name = table_name_for("ELISA_2024", "results/dados.xlsx", "Plan1")
        self.assertTrue(name.startswith("p_elisa_2024"))
        self.assertIn("dados", name)

    def test_sync_csv_creates_table(self) -> None:
        proj = Path(self._tmpdir.name) / "ProjA"
        proj.mkdir()
        csv_path = proj / "medidas.csv"
        pd.DataFrame({"insumo": ["A"], "qtd": [3]}).to_csv(csv_path, index=False)

        scan = scan_project(proj, extensions=frozenset({".csv"}), compute_hashes=True)
        stats = sync_tabular_from_scans([scan])
        self.assertEqual(stats.tables_created, 1)
        self.assertEqual(stats.tables_updated, 0)

        key = ingest_key("ProjA", "medidas.csv", "csv")
        stats2 = sync_tabular_from_scans([scan])
        self.assertEqual(stats2.tables_created, 0)
        self.assertEqual(stats2.tables_updated, 0)

        csv_path.unlink()
        scan2 = scan_project(proj, extensions=frozenset({".csv"}), compute_hashes=True)
        stats3 = sync_tabular_from_scans([scan2])
        self.assertEqual(stats3.tables_removed, 1)
        self.assertNotIn(key, [])  # smoke: no exception


class NlQueryValidationTests(unittest.TestCase):
    def test_rejects_drop(self) -> None:
        ok, _ = validate_readonly_sql("DROP TABLE x")
        self.assertFalse(ok)

    def test_accepts_select(self) -> None:
        ok, _ = validate_readonly_sql('SELECT * FROM "p_x" LIMIT 10')
        self.assertTrue(ok)

    def test_rejects_insert(self) -> None:
        ok, _ = validate_readonly_sql("INSERT INTO x VALUES (1)")
        self.assertFalse(ok)

    def test_rejects_multiple_statements(self) -> None:
        ok, _ = validate_readonly_sql("SELECT 1; DROP TABLE x")
        self.assertFalse(ok)


class DeduplicateColumnsTests(unittest.TestCase):
    def test_no_duplicates(self) -> None:
        cols = ["a", "b", "c"]
        self.assertEqual(_deduplicate_columns(cols), ["a", "b", "c"])

    def test_handles_duplicates(self) -> None:
        cols = ["nome", "nome", "nome"]
        result = _deduplicate_columns(cols)
        self.assertEqual(result, ["nome", "nome_1", "nome_2"])

    def test_mixed_duplicates(self) -> None:
        cols = ["a", "b", "a", "c", "b"]
        result = _deduplicate_columns(cols)
        self.assertEqual(result, ["a", "b", "a_1", "c", "b_1"])


class ExtractSqlTests(unittest.TestCase):
    """Cobre os caminhos do parser de resposta do LLM em ``_extract_sql``."""

    def test_extracts_select_after_closed_think(self) -> None:
        raw = "<think>raciocínio aqui</think>\nSELECT 1"
        sql, truncated = _extract_sql(raw)
        self.assertEqual(sql, "SELECT 1")
        self.assertFalse(truncated)

    def test_extracts_from_sql_code_block(self) -> None:
        raw = "Resposta:\n```sql\nSELECT * FROM \"p_x\"\n```\nfim."
        sql, truncated = _extract_sql(raw)
        self.assertEqual(sql, 'SELECT * FROM "p_x"')
        self.assertFalse(truncated)

    def test_truncated_think_returns_empty_and_flag(self) -> None:
        # Modelo gastou todos os tokens dentro do <think> sem fechar a tag.
        raw = "<think>raciocinando, mas a resposta foi cortada antes de"
        sql, truncated = _extract_sql(raw)
        self.assertEqual(sql, "")
        self.assertTrue(truncated)

    def test_strip_handles_reasoning_alias(self) -> None:
        raw = "<reasoning>algo</reasoning>SELECT 2"
        cleaned, truncated = _strip_think_blocks(raw)
        self.assertEqual(cleaned.strip(), "SELECT 2")
        self.assertFalse(truncated)

    def test_empty_response_returns_empty(self) -> None:
        sql, truncated = _extract_sql("")
        self.assertEqual(sql, "")
        self.assertFalse(truncated)


class ReadOnlyFunctionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._duck_prev = os.environ.get("ASSISTENTE_DUCKDB_DIR")
        os.environ["ASSISTENTE_DUCKDB_DIR"] = self._tmpdir.name

    def tearDown(self) -> None:
        if self._duck_prev is None:
            os.environ.pop("ASSISTENTE_DUCKDB_DIR", None)
        else:
            os.environ["ASSISTENTE_DUCKDB_DIR"] = self._duck_prev
        self._tmpdir.cleanup()

    def test_list_ingested_tables_empty_db(self) -> None:
        df = list_ingested_tables()
        self.assertTrue(df.empty)

    def test_has_ingested_tables_no_db(self) -> None:
        self.assertFalse(has_ingested_tables())

    def test_execute_sql_no_db(self) -> None:
        df, err = execute_sql("SELECT 1")
        self.assertIsNone(df)
        self.assertIn("não existe", err)


if __name__ == "__main__":
    unittest.main()
