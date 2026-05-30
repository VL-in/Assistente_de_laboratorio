"""Testes da busca híbrida RAG (BM25 + semântica via txtai)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.hybrid import (  # noqa: E402
    DEFAULT_HYBRID_WEIGHT,
    env_hybrid_enabled,
    hybrid_dense_weight,
)
from rag.index_txtai import (  # noqa: E402
    _annotate_hits,
    _backend_supports_hybrid,
    _search_parameters,
    embeddings_config,
    search_with_backend,
)
from rag.manifest import IndexManifest, load_manifest, save_manifest  # noqa: E402


class HybridConfigTests(unittest.TestCase):
    def test_hybrid_enabled_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertTrue(env_hybrid_enabled())

    @patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "0"}, clear=False)
    def test_hybrid_disabled_via_env(self) -> None:
        self.assertFalse(env_hybrid_enabled())

    def test_default_dense_weight(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(hybrid_dense_weight(), DEFAULT_HYBRID_WEIGHT)

    @patch.dict("os.environ", {"RAG_HYBRID_WEIGHT": "0.65"}, clear=False)
    def test_dense_weight_from_env(self) -> None:
        self.assertEqual(hybrid_dense_weight(), 0.65)

    def test_dense_weight_clamped(self) -> None:
        self.assertEqual(hybrid_dense_weight(1.5), 1.0)
        self.assertEqual(hybrid_dense_weight(-0.2), 0.0)

    @patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "0"}, clear=False)
    def test_embeddings_config_without_hybrid(self) -> None:
        cfg = embeddings_config()
        self.assertNotIn("hybrid", cfg)

    @patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "1"}, clear=False)
    def test_embeddings_config_with_hybrid(self) -> None:
        cfg = embeddings_config()
        self.assertTrue(cfg.get("hybrid"))

    def test_search_parameters_include_weight_when_hybrid(self) -> None:
        with patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "1"}, clear=False):
            params = _search_parameters("tampão de amostra")
        self.assertEqual(params["query"], "tampão de amostra")
        self.assertAlmostEqual(float(params["weight"]), DEFAULT_HYBRID_WEIGHT)


class HybridSearchBackendTests(unittest.TestCase):
    def test_backend_supports_hybrid_when_scoring_present(self) -> None:
        backend = MagicMock()
        backend.issparse.return_value = True
        with patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "1"}, clear=False):
            self.assertTrue(_backend_supports_hybrid(backend))

    def test_backend_without_bm25_is_semantic_only(self) -> None:
        backend = MagicMock()
        backend.issparse.return_value = False
        backend.scoring = None
        with patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "1"}, clear=False):
            self.assertFalse(_backend_supports_hybrid(backend))

    def test_annotate_hits_marks_hybrid_mode(self) -> None:
        backend = MagicMock()
        backend.issparse.return_value = True
        hits = [{"id": "1", "text": "a", "score": 0.9}]
        with patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "1"}, clear=False):
            out = _annotate_hits(hits, backend=backend, hybrid_weight=None)
        self.assertEqual(out[0]["search_mode"], "hybrid")
        self.assertIn("hybrid_dense_weight", out[0])

    @patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "1", "RAG_HYBRID_WEIGHT": "0.4"}, clear=False)
    def test_search_with_backend_passes_sql_weight(self) -> None:
        backend = MagicMock()
        backend.issparse.return_value = True
        backend.search.return_value = [{"id": "1", "text": "tampão", "score": 0.8}]

        hits = search_with_backend(backend, "tampão de amostra", 3)

        self.assertEqual(len(hits), 1)
        sql_arg = backend.search.call_args[0][0]
        self.assertIn("similar(:query, :weight)", sql_arg)
        params = backend.search.call_args[1]["parameters"]
        self.assertAlmostEqual(float(params["weight"]), 0.4)
        self.assertEqual(hits[0]["search_mode"], "hybrid")


class ManifestHybridTests(unittest.TestCase):
    def test_manifest_roundtrip_hybrid_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("rag.manifest.txtai_data_root", return_value=Path(tmp)):
                manifest = IndexManifest(hybrid_index=True)
                save_manifest(manifest)
                loaded = load_manifest()
        self.assertTrue(loaded.hybrid_index)


if __name__ == "__main__":
    unittest.main()
