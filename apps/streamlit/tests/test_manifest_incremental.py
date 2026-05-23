"""Testes leves do manifesto e chaves de arquivo (sem txtai/GPU)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Permite importar módulos do app Streamlit
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects_loader import ScannedFile  # noqa: E402
from rag.manifest import (  # noqa: E402
    IndexManifest,
    file_index_key,
    load_manifest,
    manifest_path,
    save_manifest,
)


def _sample_file() -> ScannedFile:
    return ScannedFile(
        project_id="ELISA",
        project_root=Path("/data/projetos/ELISA"),
        absolute_path=Path("/data/projetos/ELISA/planning/doc.docx"),
        relative_path="planning/doc.docx",
        size_bytes=100,
        modified_epoch=1.0,
        content_hash_sha256="abc123",
    )


class ManifestTests(unittest.TestCase):
    def test_file_index_key(self) -> None:
        sf = _sample_file()
        self.assertEqual(file_index_key(sf), "ELISA/planning/doc.docx")

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import os

            prev = os.environ.get("ASSISTENTE_TXTAI_DIR")
            os.environ["ASSISTENTE_TXTAI_DIR"] = tmp
            try:
                m = IndexManifest(embedding_model="test-model")
                m.set_file("ELISA/a.docx", content_hash="hash1", chunk_ids=["id0", "id1"])
                save_manifest(m)
                loaded = load_manifest()
                self.assertEqual(loaded.embedding_model, "test-model")
                entry = loaded.get("ELISA/a.docx")
                self.assertIsNotNone(entry)
                assert entry is not None
                self.assertEqual(entry.content_hash_sha256, "hash1")
                self.assertEqual(entry.chunk_ids, ["id0", "id1"])
                self.assertTrue(manifest_path().is_file())
                raw = json.loads(manifest_path().read_text(encoding="utf-8"))
                self.assertEqual(raw["version"], 1)
            finally:
                if prev is None:
                    os.environ.pop("ASSISTENTE_TXTAI_DIR", None)
                else:
                    os.environ["ASSISTENTE_TXTAI_DIR"] = prev

    def test_remove_file_returns_chunk_ids(self) -> None:
        m = IndexManifest()
        m.set_file("k", content_hash="h", chunk_ids=["a", "b"])
        ids = m.remove_file("k")
        self.assertEqual(ids, ["a", "b"])
        self.assertIsNone(m.get("k"))


if __name__ == "__main__":
    unittest.main()
