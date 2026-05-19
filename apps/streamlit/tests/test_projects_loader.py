"""Testes do inventário de projetos."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects_loader import scan_project  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
