"""Testes do pacote OLAP (DuckDB)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olap.connection import (  # noqa: E402
    demo_aggregation,
    demo_detail,
    duckdb_library_version,
    seed_demo_data,
)
from olap.paths import duckdb_data_root, duckdb_database_path  # noqa: E402


class OlapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("ASSISTENTE_DUCKDB_DIR")
        os.environ["ASSISTENTE_DUCKDB_DIR"] = self._tmpdir.name

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("ASSISTENTE_DUCKDB_DIR", None)
        else:
            os.environ["ASSISTENTE_DUCKDB_DIR"] = self._prev
        self._tmpdir.cleanup()

    def test_paths_under_env_override(self) -> None:
        root = duckdb_data_root()
        self.assertEqual(root, Path(self._tmpdir.name).resolve())
        self.assertTrue(str(duckdb_database_path()).endswith("assistente_lab.duckdb"))

    def test_seed_and_aggregation(self) -> None:
        self.assertTrue(seed_demo_data())
        self.assertFalse(seed_demo_data())
        agg = demo_aggregation()
        self.assertEqual(len(agg), 2)
        self.assertIn("total_quantidade", agg.columns)
        elisa = agg.loc[agg["projeto_id"] == "ELISA_2024", "total_quantidade"].iloc[0]
        self.assertEqual(int(elisa), 8)
        detail = demo_detail()
        self.assertGreaterEqual(len(detail), 4)

    def test_library_version_string(self) -> None:
        version = duckdb_library_version()
        self.assertTrue(version[0].isdigit())


if __name__ == "__main__":
    unittest.main()
