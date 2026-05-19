"""Testes de chunking (loops e limites)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.chunking import chunk_text  # noqa: E402
from rag.manifest import chunking_signature  # noqa: E402


class ChunkingTests(unittest.TestCase):
    def test_single_short_text(self) -> None:
        self.assertEqual(chunk_text("abc", max_chars=100, overlap=10), ["abc"])

    def test_overlap_advances(self) -> None:
        text = "a" * 200
        parts = chunk_text(text, max_chars=80, overlap=20)
        self.assertGreater(len(parts), 1)
        joined = "".join(parts)
        self.assertGreaterEqual(len(joined), 200)

    def test_no_infinite_loop_on_overlap_ge_max(self) -> None:
        text = "x" * 500
        parts = chunk_text(text, max_chars=100, overlap=150)
        self.assertGreater(len(parts), 0)
        self.assertLess(len(parts), 50)

    def test_chunking_signature_changes(self) -> None:
        a = chunking_signature(520, 80, 2_000_000)
        b = chunking_signature(600, 80, 2_000_000)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
