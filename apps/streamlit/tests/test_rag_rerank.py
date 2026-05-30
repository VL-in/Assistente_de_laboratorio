"""Testes do rerank RAG (cross-encoder) — sem baixar modelo real."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.rerank import (  # noqa: E402
    RERANKER_MODEL_ID,
    default_retrieve_k,
    env_rerank_enabled,
    hit_text_for_rerank,
    load_reranker_safe,
    rerank_hits,
)


class MockReranker:
    """Atribui score maior quando o texto contém 'relevante'."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for _, text in pairs:
            scores.append(10.0 if "relevante" in text.lower() else 1.0)
        return scores


class RerankHelpersTests(unittest.TestCase):
    def test_reranker_model_id_is_valid_slug(self) -> None:
        self.assertIn("L12-H384", RERANKER_MODEL_ID)
        self.assertNotIn("L6-H384", RERANKER_MODEL_ID)

    @patch("rag.rerank.load_reranker", side_effect=OSError("modelo ausente"))
    def test_load_reranker_safe_returns_error(self, _mock: object) -> None:
        result = load_reranker_safe()
        self.assertFalse(result.ok)
        self.assertIn("modelo ausente", result.error or "")

    def test_default_retrieve_k_heuristic(self) -> None:
        self.assertEqual(default_retrieve_k(6), 24)
        self.assertEqual(default_retrieve_k(6, override=30), 30)

    def test_default_retrieve_k_respects_top_k_floor(self) -> None:
        self.assertEqual(default_retrieve_k(6, override=3), 6)

    @patch.dict("os.environ", {"RAG_RERANK_RETRIEVE_K": "15"}, clear=False)
    def test_default_retrieve_k_env(self) -> None:
        self.assertEqual(default_retrieve_k(6), 15)

    @patch.dict("os.environ", {"RAG_RERANK_ENABLED": "0"}, clear=False)
    def test_env_rerank_disabled(self) -> None:
        self.assertFalse(env_rerank_enabled())

    def test_hit_text_prefers_cited(self) -> None:
        hit = {"cited": "texto citado", "text": "outro"}
        self.assertEqual(hit_text_for_rerank(hit), "texto citado")

    def test_hit_text_truncates_long_body(self) -> None:
        hit = {"text": "x" * 5000}
        self.assertEqual(len(hit_text_for_rerank(hit)), 2000)


class RerankHitsTests(unittest.TestCase):
    def test_promotes_relevant_chunk(self) -> None:
        hits = [
            {"id": "a", "text": "trecho genérico", "score": 0.9},
            {"id": "b", "text": "trecho relevante para a pergunta", "score": 0.5},
        ]
        out = rerank_hits(
            "pergunta",
            hits,
            reranker=MockReranker(),
            top_k=2,
        )
        self.assertEqual(out[0]["id"], "b")
        self.assertEqual(out[0]["retrieval_score"], 0.5)
        self.assertTrue(out[0].get("rerank_applied"))
        self.assertEqual(out[0].get("rerank_score"), out[0]["score"])
        self.assertGreater(out[0]["score"], out[1]["score"])

    def test_respects_top_k(self) -> None:
        hits = [{"id": str(i), "text": f"t{i}", "score": float(i)} for i in range(10)]
        out = rerank_hits("q", hits, reranker=MockReranker(), top_k=3)
        self.assertEqual(len(out), 3)

    def test_empty_query_returns_slice(self) -> None:
        hits = [{"id": "1", "text": "a", "score": 1.0}]
        out = rerank_hits("", hits, reranker=MockReranker(), top_k=1)
        self.assertEqual(out, hits)


class RagSearchToolRerankTests(unittest.TestCase):
    @patch("agents.tools.rerank_hits")
    @patch("agents.tools.search_with_backend")
    @patch("agents.tools.index_ready", return_value=True)
    def test_rag_tool_calls_rerank_when_enabled(
        self,
        _ready: MagicMock,
        mock_search: MagicMock,
        mock_rerank: MagicMock,
    ) -> None:
        from agents.tools import rag_search_tool

        mock_search.return_value = [{"id": "1", "text": "a", "score": 0.8}]
        mock_rerank.return_value = [{"id": "1", "text": "a", "score": 9.0}]

        result = rag_search_tool(
            "pergunta",
            backend=MagicMock(),
            top_k=4,
            reranker=MagicMock(),
            rerank_retrieve_k=20,
        )

        self.assertTrue(result.ok)
        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        self.assertEqual(kwargs.get("retrieve_limit"), 20)
        mock_rerank.assert_called_once()
        self.assertTrue(result.payload.get("rerank_enabled"))
        self.assertIn("rerank", result.summary)

    @patch("agents.tools.rerank_hits")
    @patch("agents.tools.search_with_backend")
    @patch("agents.tools.index_ready", return_value=True)
    def test_rag_tool_skips_rerank_without_model(
        self,
        _ready: MagicMock,
        mock_search: MagicMock,
        mock_rerank: MagicMock,
    ) -> None:
        from agents.tools import rag_search_tool

        mock_search.return_value = [{"id": "1", "text": "a", "score": 0.8}]

        rag_search_tool(
            "pergunta",
            backend=MagicMock(),
            top_k=4,
            reranker=None,
        )

        mock_rerank.assert_not_called()
        _, kwargs = mock_search.call_args
        self.assertIsNone(kwargs.get("retrieve_limit"))


if __name__ == "__main__":
    unittest.main()
