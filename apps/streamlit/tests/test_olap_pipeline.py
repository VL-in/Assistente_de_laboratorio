"""Testes de integração do pipeline OLAP: ingestão → catálogo → consulta."""

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

from olap.connection import open_duckdb  # noqa: E402
from olap.ingest import (  # noqa: E402
    has_ingested_tables,
    list_ingested_tables,
    sync_tabular_from_scans,
    table_name_for,
)
from olap.nl_query import execute_sql, validate_readonly_sql  # noqa: E402
from olap.schema_catalog import build_schema_catalog_text  # noqa: E402
from projects_loader import scan_project  # noqa: E402


class PipelineIntegrationTests(unittest.TestCase):
    """Testa o fluxo completo: arquivo → DuckDB → consulta."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._duck_prev = os.environ.get("ASSISTENTE_DUCKDB_DIR")
        os.environ["ASSISTENTE_DUCKDB_DIR"] = self._tmpdir.name

        self._proj_dir = Path(self._tmpdir.name) / "TestProject"
        self._proj_dir.mkdir()
        csv_path = self._proj_dir / "dados.csv"
        pd.DataFrame({
            "insumo": ["Anticorpo A", "Reagente B", "Placa C"],
            "quantidade": [10, 25, 5],
            "lote": ["L001", "L002", "L003"],
        }).to_csv(csv_path, index=False)

        self._xlsx_path = self._proj_dir / "planilha.xlsx"
        with pd.ExcelWriter(self._xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({
                "experimento": ["E1", "E2"],
                "resultado": [0.85, 0.92],
            }).to_excel(writer, sheet_name="Resultados", index=False)
            pd.DataFrame({
                "parametro": ["pH", "Temp"],
                "valor": [7.4, 37],
            }).to_excel(writer, sheet_name="Config", index=False)

    def tearDown(self) -> None:
        if self._duck_prev is None:
            os.environ.pop("ASSISTENTE_DUCKDB_DIR", None)
        else:
            os.environ["ASSISTENTE_DUCKDB_DIR"] = self._duck_prev
        self._tmpdir.cleanup()

    def test_full_ingest_pipeline(self) -> None:
        scan = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv", ".xlsx"}),
            compute_hashes=True,
        )
        self.assertEqual(len(scan.files), 2)

        stats = sync_tabular_from_scans([scan])
        self.assertEqual(stats.tables_created, 3)
        self.assertEqual(stats.tables_updated, 0)
        self.assertEqual(stats.tables_removed, 0)
        self.assertEqual(len(stats.errors), 0)

        self.assertTrue(has_ingested_tables())

        catalog_df = list_ingested_tables()
        self.assertEqual(len(catalog_df), 3)
        self.assertIn("dados", catalog_df["relative_path"].iloc[0])

    def test_catalog_contains_tables(self) -> None:
        scan = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv", ".xlsx"}),
            compute_hashes=True,
        )
        sync_tabular_from_scans([scan])

        catalog_text = build_schema_catalog_text()
        self.assertIn("_project_id", catalog_text)
        self.assertIn("_source_file", catalog_text)
        self.assertIn("insumo", catalog_text)
        self.assertIn("quantidade", catalog_text)
        self.assertNotIn("demo_experimentos", catalog_text)

    def test_execute_sql_on_ingested_data(self) -> None:
        scan = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv", ".xlsx"}),
            compute_hashes=True,
        )
        sync_tabular_from_scans([scan])

        tbl = table_name_for("TestProject", "dados.csv", "csv")
        sql = f'SELECT insumo, quantidade FROM "{tbl}" WHERE quantidade > 5'
        df, err = execute_sql(sql)

        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn("Anticorpo A", df["insumo"].values)
        self.assertIn("Reagente B", df["insumo"].values)

    def test_metadata_columns_present(self) -> None:
        scan = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv"}),
            compute_hashes=True,
        )
        sync_tabular_from_scans([scan])

        tbl = table_name_for("TestProject", "dados.csv", "csv")
        df, err = execute_sql(f'SELECT * FROM "{tbl}" LIMIT 1')

        self.assertIsNone(err)
        self.assertIn("_project_id", df.columns)
        self.assertIn("_source_file", df.columns)
        self.assertIn("_sheet_name", df.columns)
        self.assertEqual(df["_project_id"].iloc[0], "TestProject")
        self.assertEqual(df["_source_file"].iloc[0], "dados.csv")

    def test_incremental_update_on_file_change(self) -> None:
        scan = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv"}),
            compute_hashes=True,
        )
        stats1 = sync_tabular_from_scans([scan])
        self.assertEqual(stats1.tables_created, 1)

        stats2 = sync_tabular_from_scans([scan])
        self.assertEqual(stats2.tables_created, 0)
        self.assertEqual(stats2.tables_updated, 0)

        csv_path = self._proj_dir / "dados.csv"
        pd.DataFrame({
            "insumo": ["Novo Insumo"],
            "quantidade": [100],
            "lote": ["L999"],
        }).to_csv(csv_path, index=False)

        scan3 = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv"}),
            compute_hashes=True,
        )
        stats3 = sync_tabular_from_scans([scan3])
        self.assertEqual(stats3.tables_updated, 1)

        tbl = table_name_for("TestProject", "dados.csv", "csv")
        df, _ = execute_sql(f'SELECT * FROM "{tbl}"')
        self.assertEqual(len(df), 1)
        self.assertEqual(df["insumo"].iloc[0], "Novo Insumo")

    def test_file_removal_drops_table(self) -> None:
        scan = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv"}),
            compute_hashes=True,
        )
        sync_tabular_from_scans([scan])
        self.assertTrue(has_ingested_tables())

        csv_path = self._proj_dir / "dados.csv"
        csv_path.unlink()

        scan2 = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv"}),
            compute_hashes=True,
        )
        stats = sync_tabular_from_scans([scan2])
        self.assertEqual(stats.tables_removed, 1)

        catalog = list_ingested_tables()
        self.assertEqual(len(catalog), 0)

    def test_empty_scan_with_manifest_aborts_sync(self) -> None:
        # 1. sincroniza para popular o manifesto
        scan = scan_project(
            self._proj_dir,
            extensions=frozenset({".csv", ".xlsx"}),
            compute_hashes=True,
        )
        sync_tabular_from_scans([scan])
        self.assertTrue(has_ingested_tables())
        baseline_catalog = list_ingested_tables()
        self.assertGreater(len(baseline_catalog), 0)

        # 2. simula escaneamento vazio (pasta inacessivel / caminho errado)
        empty_scan = scan_project(
            Path(self._tmpdir.name) / "pasta_que_nao_existe",
            extensions=frozenset({".csv", ".xlsx"}),
            compute_hashes=True,
        )
        self.assertEqual(len(empty_scan.files), 0)

        stats = sync_tabular_from_scans([empty_scan])
        self.assertTrue(stats.aborted_empty_scan)
        self.assertEqual(stats.tables_removed, 0)
        self.assertEqual(stats.tables_created, 0)
        self.assertEqual(stats.tables_updated, 0)
        self.assertGreater(len(stats.errors), 0)
        self.assertIn("preservar", stats.errors[0].lower())

        # 3. catalogo deve estar intacto apos a tentativa abortada
        after_catalog = list_ingested_tables()
        self.assertEqual(len(after_catalog), len(baseline_catalog))

    def test_empty_scan_without_manifest_runs_normally(self) -> None:
        # Sem manifesto, escaneamento vazio nao precisa ser abortado
        # (nao ha nada a perder).
        empty_scan = scan_project(
            Path(self._tmpdir.name) / "vazio",
            extensions=frozenset({".csv"}),
            compute_hashes=True,
        )
        stats = sync_tabular_from_scans([empty_scan])
        self.assertFalse(stats.aborted_empty_scan)
        self.assertEqual(stats.tables_created, 0)
        self.assertEqual(stats.tables_removed, 0)


class SqlValidationTests(unittest.TestCase):
    """Testes de validação SQL mais abrangentes."""

    def test_with_clause_allowed(self) -> None:
        sql = 'WITH cte AS (SELECT 1 AS x) SELECT * FROM cte'
        ok, _ = validate_readonly_sql(sql)
        self.assertTrue(ok)

    def test_subquery_with_delete_blocked(self) -> None:
        sql = 'SELECT * FROM (DELETE FROM x RETURNING *)'
        ok, _ = validate_readonly_sql(sql)
        self.assertFalse(ok)

    def test_union_allowed(self) -> None:
        sql = 'SELECT 1 UNION SELECT 2'
        ok, _ = validate_readonly_sql(sql)
        self.assertTrue(ok)

    def test_line_comment_with_drop_now_allowed(self) -> None:
        # Comentários são removidos antes da checagem; o resto é só SELECT.
        sql = "SELECT 1 -- DROP TABLE x"
        ok, err = validate_readonly_sql(sql)
        self.assertTrue(ok, err)

    def test_string_literal_with_drop_keyword_allowed(self) -> None:
        # Antes: rejeitado por causa de 'DROP' dentro do literal. Agora: ok.
        sql = "SELECT * FROM x WHERE status = 'DROP-OUT'"
        ok, err = validate_readonly_sql(sql)
        self.assertTrue(ok, err)

    def test_string_literal_with_semicolon_allowed(self) -> None:
        sql = "SELECT * FROM x WHERE col = 'a;b'"
        ok, err = validate_readonly_sql(sql)
        self.assertTrue(ok, err)

    def test_string_literal_with_set_keyword_allowed(self) -> None:
        sql = "SELECT 'set' AS x"
        ok, err = validate_readonly_sql(sql)
        self.assertTrue(ok, err)

    def test_block_comment_with_create_allowed(self) -> None:
        sql = "SELECT 1 /* CREATE TABLE y AS SELECT * FROM z */"
        ok, err = validate_readonly_sql(sql)
        self.assertTrue(ok, err)

    def test_real_drop_outside_string_still_blocked(self) -> None:
        sql = "DROP TABLE x"
        ok, _ = validate_readonly_sql(sql)
        self.assertFalse(ok)

    def test_merge_still_blocked(self) -> None:
        sql = 'MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET x = 1'
        ok, _ = validate_readonly_sql(sql)
        self.assertFalse(ok)

    def test_vacuum_blocked(self) -> None:
        ok, _ = validate_readonly_sql("VACUUM")
        self.assertFalse(ok)

    def test_pragma_blocked(self) -> None:
        ok, _ = validate_readonly_sql("SELECT 1; PRAGMA database_size")
        self.assertFalse(ok)

    def test_double_quote_identifier_with_keyword_allowed(self) -> None:
        # Coluna chamada "drop_reason" entre aspas duplas e identificador valido.
        sql = 'SELECT "drop_reason" FROM "p_x"'
        ok, err = validate_readonly_sql(sql)
        self.assertTrue(ok, err)


if __name__ == "__main__":
    unittest.main()
