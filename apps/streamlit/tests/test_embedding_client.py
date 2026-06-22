"""Testes do cliente de embeddings in-process (sem TEI, sem rede)."""

from __future__ import annotations

import os
import sys
import threading
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
    embedding_transform,
)
from rag.index_txtai import embeddings_config  # noqa: E402

DIM = 384  # multilingual-e5-small output dimension


def _fake_model(n: int = 1, dim: int = DIM) -> MagicMock:
    """Cria mock de SentenceTransformer cujo encode devolve matriz (n, dim)."""
    m = MagicMock()
    m.encode.return_value = np.zeros((n, dim), dtype=np.float32)
    return m


class EmbeddingClientTests(unittest.TestCase):
    def setUp(self) -> None:
        # Garante que o singleton seja limpo entre testes
        import rag.embedding_client as ec
        ec._model_instance = None

    def tearDown(self) -> None:
        import rag.embedding_client as ec
        ec._model_instance = None

    # ── Constantes de contrato ──────────────────────────────────────────────

    def test_model_id(self) -> None:
        self.assertEqual(EMBEDDING_MODEL_ID, "intfloat/multilingual-e5-small")

    def test_transform_path_importable_string(self) -> None:
        self.assertEqual(EMBEDDING_TRANSFORM_PATH, "rag.embedding_client.embedding_transform")

    # ── embed_texts: casos de borda ─────────────────────────────────────────

    def test_embed_texts_empty_returns_zero_shape(self) -> None:
        result = embed_texts([])
        self.assertEqual(result.shape, (0, 0))

    def test_embed_texts_single_text(self) -> None:
        fake = _fake_model(1)
        with patch("rag.embedding_client._get_model", return_value=fake):
            result = embed_texts(["passage: texto de teste"])
        self.assertEqual(result.shape, (1, DIM))
        self.assertEqual(result.dtype, np.float32)

    def test_embed_texts_batch(self) -> None:
        fake = _fake_model(3)
        with patch("rag.embedding_client._get_model", return_value=fake):
            result = embed_texts(["a", "b", "c"])
        self.assertEqual(result.shape, (3, DIM))
        fake.encode.assert_called_once()
        call_args = fake.encode.call_args
        self.assertEqual(call_args[0][0], ["a", "b", "c"])

    def test_embed_texts_returns_float32(self) -> None:
        fake = _fake_model(2)
        with patch("rag.embedding_client._get_model", return_value=fake):
            result = embed_texts(["x", "y"])
        self.assertEqual(result.dtype, np.float32)

    # ── embedding_transform: interface do txtai ─────────────────────────────

    def test_transform_list_input(self) -> None:
        fake = _fake_model(2)
        with patch("rag.embedding_client._get_model", return_value=fake):
            out = embedding_transform(["query: a", "query: b"])
        self.assertEqual(out.shape, (2, DIM))

    def test_transform_string_input_wrapped_as_list(self) -> None:
        fake = _fake_model(1)
        with patch("rag.embedding_client._get_model", return_value=fake):
            out = embedding_transform("query: texto")
        self.assertEqual(out.shape, (1, DIM))
        # Deve ter passado lista de 1 elemento para encode
        call_args = fake.encode.call_args
        self.assertEqual(call_args[0][0], ["query: texto"])

    # ── Singleton: modelo carregado apenas uma vez ──────────────────────────

    def test_model_loaded_once_on_repeated_calls(self) -> None:
        fake = _fake_model(1)
        with patch("rag.embedding_client._get_model", return_value=fake) as mock_get:
            embed_texts(["a"])
            embed_texts(["b"])
        # _get_model chamado duas vezes, mas o modelo em si é o mesmo objeto
        self.assertEqual(mock_get.call_count, 2)
        self.assertIs(mock_get.return_value, fake)

    def test_singleton_thread_safety(self) -> None:
        """Duas threads chamando _get_model simultaneamente retornam o mesmo objeto."""
        import rag.embedding_client as ec

        fake = _fake_model(1)
        # Injeta o singleton já populado — simula modelo carregado
        ec._model_instance = fake
        loaded: list[object] = []

        def _load() -> None:
            loaded.append(ec._get_model())

        t1 = threading.Thread(target=_load)
        t2 = threading.Thread(target=_load)
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertEqual(len(loaded), 2)
        self.assertIs(loaded[0], loaded[1])
        self.assertIs(loaded[0], fake)

    # ── embeddings_config: contrato com index_txtai ─────────────────────────

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

    @patch.dict("os.environ", {"RAG_HYBRID_ENABLED": "0"}, clear=False)
    def test_embeddings_config_no_hybrid(self) -> None:
        cfg = embeddings_config()
        self.assertNotIn("hybrid", cfg)

    # ── Variável de ambiente EMBEDDING_BATCH_SIZE ───────────────────────────

    def test_batch_size_env(self) -> None:
        import rag.embedding_client as ec
        with patch.dict("os.environ", {"EMBEDDING_BATCH_SIZE": "8"}):
            self.assertEqual(ec._batch_size(), 8)

    def test_batch_size_default(self) -> None:
        import rag.embedding_client as ec
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("EMBEDDING_BATCH_SIZE", None)
            self.assertEqual(ec._batch_size(), ec._DEFAULT_BATCH_SIZE)

    def test_batch_size_invalid_falls_back_to_default(self) -> None:
        import rag.embedding_client as ec
        with patch.dict("os.environ", {"EMBEDDING_BATCH_SIZE": "abc"}):
            self.assertEqual(ec._batch_size(), ec._DEFAULT_BATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
