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

from olap.connection import open_duckdb  # noqa: E402
from olap.ingest import (  # noqa: E402
    _convert_br_number,
    _convert_intl_number,
    _deduplicate_columns,
    _read_csv,
    _smart_convert_single_value,
    _try_convert_column_to_numeric,
    has_ingested_tables,
    ingest_key,
    list_ingested_tables,
    sync_tabular_from_scans,
    table_name_for,
)
from olap.indexes import METADATA_FILTER_COLUMNS  # noqa: E402
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

    def test_sync_creates_metadata_indexes(self) -> None:
        proj = Path(self._tmpdir.name) / "ProjB"
        proj.mkdir()
        csv_path = proj / "dados.csv"
        pd.DataFrame({"insumo": ["A"], "qtd": [1]}).to_csv(csv_path, index=False)

        scan = scan_project(proj, extensions=frozenset({".csv"}), compute_hashes=True)
        stats = sync_tabular_from_scans([scan])
        self.assertGreaterEqual(stats.indexes_ensured, len(METADATA_FILTER_COLUMNS))

        catalog = list_ingested_tables()
        self.assertEqual(len(catalog), 1)
        tbl = catalog.iloc[0]["table_name"]

        conn = open_duckdb(read_only=True)
        try:
            n_idx = conn.execute(
                "SELECT COUNT(*) FROM duckdb_indexes() WHERE table_name = ?",
                [tbl],
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertGreaterEqual(int(n_idx), len(METADATA_FILTER_COLUMNS))


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


class BrazilianNumberConversionTests(unittest.TestCase):
    """Testes da conversão de números brasileiros (vírgula decimal)."""

    def test_convert_simple_decimal(self) -> None:
        self.assertAlmostEqual(_convert_br_number("0,98"), 0.98)
        self.assertAlmostEqual(_convert_br_number("1,5"), 1.5)
        self.assertAlmostEqual(_convert_br_number("-1,5"), -1.5)

    def test_convert_thousands_separator(self) -> None:
        self.assertAlmostEqual(_convert_br_number("1.234,56"), 1234.56)
        self.assertAlmostEqual(_convert_br_number("10.000,00"), 10000.0)
        self.assertAlmostEqual(_convert_br_number("1.234.567,89"), 1234567.89)

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(_convert_br_number("abc"))
        self.assertIsNone(_convert_br_number(""))

    def test_intl_conversion(self) -> None:
        self.assertAlmostEqual(_convert_intl_number("1.184"), 1.184)
        self.assertAlmostEqual(_convert_intl_number("1,234.56"), 1234.56)
        self.assertAlmostEqual(_convert_intl_number("-0.98"), -0.98)

    def test_smart_convert_detects_format(self) -> None:
        # Formato brasileiro (vírgula decimal)
        self.assertAlmostEqual(_smart_convert_single_value("0,998"), 0.998)
        # Formato internacional (ponto decimal)
        self.assertAlmostEqual(_smart_convert_single_value("1.184"), 1.184)
        # Brasileiro completo (ponto milhar, vírgula decimal)
        self.assertAlmostEqual(_smart_convert_single_value("1.234,56"), 1234.56)
        # Internacional completo (vírgula milhar, ponto decimal)
        self.assertAlmostEqual(_smart_convert_single_value("1,234.56"), 1234.56)
        # Inteiro
        self.assertAlmostEqual(_smart_convert_single_value("42"), 42.0)

    def test_column_conversion_mixed_formats(self) -> None:
        # Simula dados misturados como no problema real
        series = pd.Series(["0,998", "1.184", "3.498", "3,672"])
        converted = _try_convert_column_to_numeric(series)
        self.assertTrue(pd.api.types.is_numeric_dtype(converted))
        self.assertAlmostEqual(converted.iloc[0], 0.998)  # BR format
        self.assertAlmostEqual(converted.iloc[1], 1.184)  # INTL format
        self.assertAlmostEqual(converted.iloc[2], 3.498)  # INTL format
        self.assertAlmostEqual(converted.iloc[3], 3.672)  # BR format

    def test_column_conversion_brazilian(self) -> None:
        series = pd.Series(["0,998", "1,184", "3,498", "3,672"])
        converted = _try_convert_column_to_numeric(series)
        self.assertTrue(pd.api.types.is_numeric_dtype(converted))
        self.assertAlmostEqual(converted.iloc[0], 0.998)
        self.assertAlmostEqual(converted.iloc[1], 1.184)

    def test_column_conversion_standard_numbers(self) -> None:
        series = pd.Series(["1.5", "2.7", "3.14"])
        converted = _try_convert_column_to_numeric(series)
        self.assertTrue(pd.api.types.is_numeric_dtype(converted))
        self.assertAlmostEqual(converted.iloc[0], 1.5)

    def test_column_already_numeric_unchanged(self) -> None:
        series = pd.Series([1, 2, 3])
        converted = _try_convert_column_to_numeric(series)
        self.assertEqual(list(converted), [1, 2, 3])

    def test_column_text_stays_text(self) -> None:
        series = pd.Series(["Reagente A", "Placa B", "Anticorpo C"])
        converted = _try_convert_column_to_numeric(series)
        # Pandas >=2.x pode inferir StringDtype para colunas de texto; ambos são aceitáveis.
        self.assertFalse(pd.api.types.is_numeric_dtype(converted))
        self.assertEqual(converted.iloc[0], "Reagente A")


class CsvSeparatorAutodetectTests(unittest.TestCase):
    """``_read_csv`` deve detectar separador comum em planilhas BR/INTL."""

    def _write(self, content: str, encoding: str = "utf-8") -> Path:
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))
        p = Path(tmpdir) / "data.csv"
        p.write_text(content, encoding=encoding)
        return p

    def test_comma_separator(self) -> None:
        p = self._write("a,b,c\n1,2,3\n4,5,6\n")
        df = _read_csv(p)
        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(list(df.columns), ["a", "b", "c"])
        self.assertEqual(df.shape, (2, 3))

    def test_semicolon_separator_brazilian_excel(self) -> None:
        p = self._write("a;b;c\n1;2;3\n4;5;6\n")
        df = _read_csv(p)
        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(list(df.columns), ["a", "b", "c"])
        self.assertEqual(df.shape, (2, 3))

    def test_tab_separator(self) -> None:
        p = self._write("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        df = _read_csv(p)
        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(list(df.columns), ["a", "b", "c"])
        self.assertEqual(df.shape, (2, 3))

    def test_pipe_separator(self) -> None:
        p = self._write("a|b|c\n1|2|3\n4|5|6\n")
        df = _read_csv(p)
        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(list(df.columns), ["a", "b", "c"])

    def test_single_column_no_separator(self) -> None:
        p = self._write("nome\nAna\nBeatriz\nCarla\n")
        df = _read_csv(p)
        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(list(df.columns), ["nome"])
        self.assertEqual(df.shape, (3, 1))

    def test_latin1_encoding_with_semicolon(self) -> None:
        p = self._write("região;valor\nNorte;1\nSul;2\n", encoding="latin-1")
        df = _read_csv(p)
        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(df.shape, (2, 2))

    def test_empty_file_returns_none(self) -> None:
        p = self._write("")
        df = _read_csv(p)
        self.assertIsNone(df)


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

    def test_ignores_select_mentioned_mid_sentence(self) -> None:
        # SELECT no meio de frase em pt-BR nao deve ser confundido com a query.
        raw = "Vou usar SELECT mas precisa pensar...\nSELECT * FROM y LIMIT 10"
        sql, _ = _extract_sql(raw)
        self.assertEqual(sql, "SELECT * FROM y LIMIT 10")

    def test_trailing_semicolon_truncated(self) -> None:
        raw = "SELECT 1 FROM x;"
        sql, _ = _extract_sql(raw)
        self.assertEqual(sql, "SELECT 1 FROM x")

    def test_second_statement_after_semicolon_dropped(self) -> None:
        raw = "SELECT * FROM a;\nSELECT * FROM b"
        sql, _ = _extract_sql(raw)
        self.assertEqual(sql, "SELECT * FROM a")

    def test_semicolon_inside_string_preserved(self) -> None:
        raw = "SELECT * FROM x WHERE col = 'a;b'"
        sql, _ = _extract_sql(raw)
        self.assertEqual(sql, "SELECT * FROM x WHERE col = 'a;b'")

    def test_trailing_narrative_ptbr_stripped(self) -> None:
        raw = "SELECT * FROM x LIMIT 10\nEssa consulta retorna 10 linhas."
        sql, _ = _extract_sql(raw)
        self.assertEqual(sql, "SELECT * FROM x LIMIT 10")

    def test_trailing_narrative_enus_stripped(self) -> None:
        raw = "SELECT 1\nThis query returns one row."
        sql, _ = _extract_sql(raw)
        self.assertEqual(sql, "SELECT 1")

    def test_code_block_truncated_on_semicolon(self) -> None:
        raw = "```sql\nSELECT 1; SELECT 2\n```"
        sql, _ = _extract_sql(raw)
        self.assertEqual(sql, "SELECT 1")

    def test_with_clause_in_start_of_line_extracted(self) -> None:
        raw = "<think>...</think>\nWITH x AS (SELECT 1) SELECT * FROM x"
        sql, _ = _extract_sql(raw)
        self.assertEqual(sql, "WITH x AS (SELECT 1) SELECT * FROM x")


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
