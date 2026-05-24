"""Testes de embeddings ESM-2 + PCA (mock, sem download do modelo)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.sequence_embeddings import (  # noqa: E402
    SequenceEmbeddingConfig,
    SequenceEmbeddingTransformer,
    _disable_esm_token_dropout,
    _mean_pool_sequence_embeddings,
    clean_protein_sequence,
    extract_sequences_from_text,
)


class EsmPoolingTests(unittest.TestCase):
    def test_mean_pool_truncates_mask_to_hidden_length(self) -> None:
        import torch

        hidden = torch.ones(2, 5, 4)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
        pooled = _mean_pool_sequence_embeddings(hidden, mask)
        self.assertEqual(tuple(pooled.shape), (2, 4))


class DisableTokenDropoutTests(unittest.TestCase):
    def test_disables_on_config_and_embeddings(self) -> None:
        cfg = MagicMock()
        cfg.token_dropout = True
        emb = MagicMock()
        emb.token_dropout = True
        model = MagicMock()
        model.config = cfg
        model.embeddings = emb
        model.esm = None
        _disable_esm_token_dropout(model)
        self.assertFalse(cfg.token_dropout)
        self.assertFalse(emb.token_dropout)


class CleanSequenceTests(unittest.TestCase):
    def test_strips_fasta_header(self) -> None:
        raw = ">heavy\nACDEFGHIK"
        self.assertEqual(clean_protein_sequence(raw), "ACDEFGHIK")


class ExtractSequencesFromTextTests(unittest.TestCase):
    def test_assignment(self) -> None:
        msg = "Ag_seq = ACDEFGHIKLMNPQRSTVWYACDEFGHIK"
        found = extract_sequences_from_text(msg)
        self.assertIn("Ag_seq", found)
        self.assertTrue(len(found["Ag_seq"]) >= 10)


class SequenceEmbeddingTransformerTests(unittest.TestCase):
    def _fake_embedder(self, hidden: int = 8) -> MagicMock:
        mock = MagicMock()
        mock.hidden_size = hidden

        def embed_batch(seqs: list[str]) -> np.ndarray:
            out = np.zeros((len(seqs), hidden), dtype=np.float32)
            for i, s in enumerate(seqs):
                if s:
                    out[i, 0] = len(s)
            return out

        mock.embed_batch.side_effect = embed_batch
        return mock

    @patch("ml.sequence_embeddings._EsmEmbedder.shared")
    def test_fit_transform_pca_columns(self, mock_shared: MagicMock) -> None:
        mock_shared.return_value = self._fake_embedder()
        df = pd.DataFrame(
            {
                "Ab_heavy_chain_seq": ["ACDE", "ACDEF"],
                "Ab_light_chain_seq": ["GHIK", "GHIKL"],
                "Ag_seq": ["LMNP", "LMNPQ"],
            }
        )
        cfg = SequenceEmbeddingConfig(
            sequence_columns=("Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"),
            pca_max_components=4,
            pca_min_components=2,
            embed_batch_size=2,
        )
        tr = SequenceEmbeddingTransformer(cfg)
        tr.fit(df)
        out = tr.transform_pca_only(df)
        self.assertEqual(out.shape[0], 2)
        self.assertTrue(all(c.startswith("seq_pca_") for c in out.columns))
        self.assertGreaterEqual(len(tr.pca_columns_), 1)

    @patch("ml.sequence_embeddings._EsmEmbedder.shared")
    def test_fit_caches_train_pca_without_reembed(self, mock_shared: MagicMock) -> None:
        mock_shared.return_value = self._fake_embedder()
        df = pd.DataFrame(
            {
                "Ab_heavy_chain_seq": ["ACDE", "ACDEF"],
                "Ab_light_chain_seq": ["GHIK", "GHIKL"],
                "Ag_seq": ["LMNP", "LMNPQ"],
            }
        )
        tr = SequenceEmbeddingTransformer(
            SequenceEmbeddingConfig(pca_max_components=2, pca_min_components=1)
        )
        tr.fit(df)
        pca1 = tr.training_pca_frame()
        tr.clear_training_cache()
        with self.assertRaises(RuntimeError):
            tr.training_pca_frame()
        out = tr.transform_pca_only(df.iloc[[0]])
        self.assertEqual(len(out), 1)

    @patch("ml.sequence_embeddings._EsmEmbedder.shared")
    def test_impute_missing_sequence_at_transform(self, mock_shared: MagicMock) -> None:
        mock_shared.return_value = self._fake_embedder()
        df = pd.DataFrame(
            {
                "Ab_heavy_chain_seq": ["ACDE", None],
                "Ab_light_chain_seq": ["GHIK", "GHIK"],
                "Ag_seq": ["LMNP", "LMNP"],
            }
        )
        tr = SequenceEmbeddingTransformer()
        tr.fit(df.iloc[[0]])
        out = tr.transform_pca_only(df)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
