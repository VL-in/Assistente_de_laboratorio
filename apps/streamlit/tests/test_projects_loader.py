"""Testes do inventário de projetos."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects_loader import filter_scans_by_extensions, scan_project  # noqa: E402


class ProjectsLoaderTests(unittest.TestCase):
    def test_skips_office_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ProjA"
            root.mkdir()
            (root / "real.docx").write_bytes(b"not a real docx but listed")
            (root / "~$real.docx").write_bytes(b"temp")
            scan = scan_project(root, extensions=frozenset({".docx"}))
            names = [f.relative_path for f in scan.files]
            self.assertEqual(names, ["real.docx"])

    def test_filter_scans_by_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ProjA"
            root.mkdir()
            (root / "doc.docx").write_bytes(b"x")
            (root / "data.csv").write_bytes(b"a,b\n1,2")
            scan = scan_project(
                root,
                extensions=frozenset({".docx", ".csv"}),
                compute_hashes=False,
            )
            tabular = filter_scans_by_extensions([scan], frozenset({".csv"}))
            self.assertEqual(len(tabular), 1)
            self.assertEqual(len(tabular[0].files), 1)
            self.assertEqual(tabular[0].files[0].relative_path, "data.csv")


if __name__ == "__main__":
    unittest.main()
