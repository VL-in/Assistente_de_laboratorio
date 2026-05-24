"""Testes do roteador de intenção do chat."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat_router import (  # noqa: E402
    ChatRouteDecision,
    _is_social_only,
    _parse_router_json,
    _rule_fallback_route,
    classify_chat_routes,
    resolve_chat_routes,
)


class SocialOnlyTests(unittest.TestCase):
    def test_greeting(self) -> None:
        self.assertTrue(_is_social_only("Olá!"))

    def test_lab_question_not_social(self) -> None:
        self.assertFalse(_is_social_only("Qual a validade do reagente X?"))


class ParseRouterJsonTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(
            _parse_router_json('{"documents": true, "spreadsheets": false, "ml_prediction": false}'),
            (True, False, False),
        )

    def test_with_markdown_fence(self) -> None:
        raw = '```json\n{"documents": false, "spreadsheets": false, "ml_prediction": true}\n```'
        self.assertEqual(_parse_router_json(raw), (False, False, True))

    def test_invalid(self) -> None:
        self.assertIsNone(_parse_router_json("não é json"))


class RuleFallbackTests(unittest.TestCase):
    def test_social(self) -> None:
        d = _rule_fallback_route("oi")
        self.assertEqual((d.use_documents, d.use_spreadsheets, d.use_ml), (False, False, False))

    def test_lab_keywords(self) -> None:
        d = _rule_fallback_route("Qual o lote do anticorpo?")
        self.assertTrue(d.use_documents)
        self.assertFalse(d.use_spreadsheets)
        self.assertFalse(d.use_ml)

    def test_tabular_keywords(self) -> None:
        d = _rule_fallback_route("Quantos registros por projeto na planilha?")
        self.assertTrue(d.use_spreadsheets)

    def test_ml_keywords(self) -> None:
        d = _rule_fallback_route("Faça a predição de log_Aff para este par Ab-Ag")
        self.assertTrue(d.use_ml)
        self.assertFalse(d.use_documents)


class ClassifyChatRoutesTests(unittest.TestCase):
    @patch("chat_router.router_enabled", return_value=True)
    def test_social_skips_llm(self, _enabled: MagicMock) -> None:
        client = MagicMock()
        d = classify_chat_routes(
            "Bom dia",
            client=client,
            model="test",
            documents_available=True,
            spreadsheets_available=True,
        )
        self.assertEqual(d.source, "rules")
        self.assertFalse(d.use_documents)
        client.chat.completions.create.assert_not_called()

    @patch("chat_router.router_enabled", return_value=True)
    @patch("chat_router.classify_with_llm")
    def test_llm_path(self, mock_llm: MagicMock, _enabled: MagicMock) -> None:
        mock_llm.return_value = ChatRouteDecision(True, False, False, "llm")
        d = classify_chat_routes(
            "Qual validade do lote 3?",
            client=MagicMock(),
            model="m",
            documents_available=True,
            spreadsheets_available=False,
            ml_available=True,
        )
        self.assertTrue(d.use_documents)
        self.assertFalse(d.use_spreadsheets)
        mock_llm.assert_called_once()

    @patch("chat_router.router_enabled", return_value=True)
    @patch("chat_router.classify_with_llm")
    def test_ml_only_when_available(self, mock_llm: MagicMock, _enabled: MagicMock) -> None:
        mock_llm.return_value = ChatRouteDecision(False, False, True, "llm")
        d = classify_chat_routes(
            "Rode o modelo ML",
            client=MagicMock(),
            model="m",
            ml_available=False,
        )
        self.assertFalse(d.use_ml)

    @patch("chat_router.router_enabled", return_value=True)
    @patch("chat_router.classify_with_llm")
    def test_ml_hint_overrides_documents(self, mock_llm: MagicMock, _enabled: MagicMock) -> None:
        mock_llm.return_value = ChatRouteDecision(True, False, False, "llm")
        d = classify_chat_routes(
            "Faça a predição de log_Aff com Agtype SARS-CoV-2",
            client=MagicMock(),
            model="m",
            ml_available=True,
        )
        self.assertTrue(d.use_ml)
        self.assertFalse(d.use_documents)

    @patch("chat_router.router_enabled", return_value=False)
    def test_disabled_router(self, _enabled: MagicMock) -> None:
        d = classify_chat_routes(
            "oi",
            documents_available=True,
            spreadsheets_available=True,
        )
        self.assertEqual(d.source, "disabled")
        self.assertTrue(d.use_documents)
        self.assertTrue(d.use_spreadsheets)


class ResolveChatRoutesTests(unittest.TestCase):
    def test_manual_override(self) -> None:
        d = resolve_chat_routes(
            "oi",
            documents_enabled=True,
            spreadsheets_enabled=False,
            ml_enabled=True,
            manual_override=True,
        )
        self.assertEqual(d.source, "manual")
        self.assertTrue(d.use_documents)
        self.assertFalse(d.use_spreadsheets)
        self.assertTrue(d.use_ml)


if __name__ == "__main__":
    unittest.main()
