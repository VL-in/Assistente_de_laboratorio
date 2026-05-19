"""Testes de extração docx (tabelas de insumos)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.extract import _extract_docx  # noqa: E402


class ExtractDocxTests(unittest.TestCase):
    def test_tables_included_in_extraction(self) -> None:
        from docx import Document

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "planejamento.docx"
            doc = Document()
            doc.add_paragraph("Planejamento do ensaio ELISA")
            table = doc.add_table(rows=2, cols=4)
            hdr = table.rows[0].cells
            hdr[0].text = "Nome"
            hdr[1].text = "Fabricante ou código"
            hdr[2].text = "Lote"
            hdr[3].text = "Validade"
            row = table.rows[1].cells
            row[0].text = "Tampão de amostra"
            row[1].text = "FAB-001"
            row[2].text = "L2024-09"
            row[3].text = "31/12/2026"
            doc.save(str(path))

            outcome = _extract_docx(path, max_chars_total=500_000)
            self.assertTrue(outcome.ok, outcome.detail)
            self.assertIn("Planejamento", outcome.text)
            self.assertIn("Tampão de amostra", outcome.text)
            self.assertIn("31/12/2026", outcome.text)
            self.assertIn("tabela", outcome.detail.lower())

    def test_nested_table_in_cell(self) -> None:
        from docx import Document

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.docx"
            doc = Document()
            outer = doc.add_table(rows=1, cols=1)
            cell = outer.rows[0].cells[0]
            cell.text = "Reagente principal"
            inner = cell.add_table(rows=2, cols=2)
            inner.rows[0].cells[0].text = "Lote"
            inner.rows[0].cells[1].text = "Validade"
            inner.rows[1].cells[0].text = "L-99"
            inner.rows[1].cells[1].text = "01/06/2027"
            doc.save(str(path))

            outcome = _extract_docx(path, max_chars_total=500_000)
            self.assertTrue(outcome.ok, outcome.detail)
            self.assertIn("Reagente principal", outcome.text)
            self.assertIn("L-99", outcome.text)
            self.assertIn("01/06/2027", outcome.text)


if __name__ == "__main__":
    unittest.main()
