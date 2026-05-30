"""Testes do cliente HTTP de embeddings (sem subir TEI)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.embedding_client import (  # noqa: E402
    EMBEDDING_MODEL_ID,
    EMBEDDING_TRANSFORM_PATH,
    embed_texts,
    embedding_http_batch_size,
    embedding_service_url,
    embedding_transform,
)
from rag.index_txtai import embeddings_config  # noqa: E402


class EmbeddingClientTests(unittest.TestCase):
    def test_model_id(self) -> None:
        self.assertEqual(EMBEDDING_MODEL_ID, "intfloat/multilingual-e5-small")

    def test_embedding_service_url_strips_trailing_slash(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_SERVICE_URL": "http://localhost:8080/"}):
            self.assertEqual(embedding_service_url(), "http://localhost:8080")

    def test_embed_texts_empty(self) -> None:
        arr = embed_texts([])
        self.assertEqual(arr.shape, (0, 0))

    def test_embedding_http_batch_size_env(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_HTTP_BATCH_SIZE": "8"}):
            self.assertEqual(embedding_http_batch_size(), 8)

    @patch("rag.embedding_client.requests.post")
    def test_embed_texts_calls_tei(self, mock_post: MagicMock) -> None:
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = [[0.1, 0.2], [0.3, 0.4]]

        result = embed_texts(["passage: a", "passage: b"])

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertTrue(args[0].endswith("/embed"))
        self.assertEqual(kwargs["json"], {"inputs": ["passage: a", "passage: b"]})
        np.testing.assert_array_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))

    @patch("rag.embedding_client.requests.post")
    def test_embed_texts_splits_large_batches(self, mock_post: MagicMock) -> None:
        mock_post.return_value.raise_for_status = MagicMock()
        counter = {"n": 0}

        def _fake_embed(url: str, **kwargs: object) -> MagicMock:
            inputs = kwargs["json"]["inputs"]  # type: ignore[index]
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = []
            for _ in inputs:
                resp.json.return_value.append([float(counter["n"])])
                counter["n"] += 1
            return resp

        mock_post.side_effect = _fake_embed

        with patch.dict("os.environ", {"EMBEDDING_HTTP_BATCH_SIZE": "2"}):
            texts = ["a", "b", "c", "d", "e"]
            result = embed_texts(texts)

        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(result.shape, (5, 1))
        np.testing.assert_array_equal(
            result.flatten(),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )

    def test_transform_delegates_to_embed_texts(self) -> None:
        with patch("rag.embedding_client.embed_texts", return_value=np.array([[1.0]], dtype=np.float32)) as mock_embed:
            out = embedding_transform(["x"])
            mock_embed.assert_called_once_with(["x"])
            np.testing.assert_array_equal(out, np.array([[1.0]], dtype=np.float32))

    @patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "1"}, clear=False)
    def test_embeddings_config_external_with_e5_prefixes(self) -> None:
        cfg = embeddings_config()
        self.assertEqual(cfg["path"], "external")
        self.assertEqual(cfg["method"], "external")
        self.assertTrue(cfg["content"])
        self.assertEqual(cfg["instructions"]["query"], "query: ")
        self.assertEqual(cfg["instructions"]["data"], "passage: ")
        self.assertEqual(cfg["transform"], EMBEDDING_TRANSFORM_PATH)
        self.assertTrue(cfg.get("hybrid"))


if __name__ == "__main__":
    unittest.main()
