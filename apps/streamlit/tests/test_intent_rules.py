"""Testes das regras puras de intenção compartilhadas pelos agentes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.intent_rules import (  # noqa: E402
    is_social_only,
    parse_router_json,
    rule_fallback,
)


class IsSocialOnlyTests(unittest.TestCase):
    def test_recognizes_greeting(self) -> None:
        self.assertTrue(is_social_only("Olá!"))
        self.assertTrue(is_social_only("Bom dia"))
        self.assertTrue(is_social_only("Obrigada!"))
        self.assertTrue(is_social_only("tchau"))

    def test_long_message_not_social(self) -> None:
        long_msg = "Olá, gostaria de saber qual o lote do anticorpo usado em 2024."
        self.assertFalse(is_social_only(long_msg))

    def test_lab_question_not_social(self) -> None:
        self.assertFalse(is_social_only("Qual a validade do reagente X?"))


class ParseRouterJsonTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(
            parse_router_json(
                '{"documents": true, "spreadsheets": false, "ml_prediction": false}'
            ),
            (True, False, False),
        )

    def test_with_markdown_fence(self) -> None:
        raw = (
            "```json\n"
            '{"documents": false, "spreadsheets": false, "ml_prediction": true}\n'
            "```"
        )
        self.assertEqual(parse_router_json(raw), (False, False, True))

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(parse_router_json("não é json"))

    def test_missing_field_returns_none(self) -> None:
        self.assertIsNone(parse_router_json('{"documents": true}'))

    def test_ml_defaults_to_false(self) -> None:
        # Quando o LLM esquece o campo ml_prediction, assumimos False.
        self.assertEqual(
            parse_router_json('{"documents": true, "spreadsheets": true}'),
            (True, True, False),
        )


class RuleFallbackTests(unittest.TestCase):
    def test_social_message(self) -> None:
        self.assertEqual(rule_fallback("oi"), (False, False, False))

    def test_lab_keywords(self) -> None:
        rag, olap, ml = rule_fallback("Qual o lote do anticorpo?")
        self.assertTrue(rag)
        self.assertFalse(olap)
        self.assertFalse(ml)

    def test_lab_keywords_plural(self) -> None:
        # Cenário real (BDD): com o LLM de triagem indisponível, o fallback
        # precisa reconhecer termos de laboratório no plural. "amostras",
        # "lotes", "reagentes" etc. casavam apenas no singular por causa do
        # ``\b`` de fechamento na regex — quebrava perguntas como esta.
        rag, olap, ml = rule_fallback(
            "Quais são as amostras positivas testadas no dia 09/02?"
        )
        self.assertTrue(rag)
        self.assertFalse(ml)

    def test_tabular_keywords(self) -> None:
        rag, olap, ml = rule_fallback(
            "Quantos registros por projeto na planilha?"
        )
        self.assertTrue(olap)
        self.assertFalse(ml)

    def test_ml_keywords_override_others(self) -> None:
        rag, olap, ml = rule_fallback(
            "Faça a predição de log_Aff para este par Ab-Ag usando a planilha"
        )
        self.assertTrue(ml)
        self.assertFalse(rag)
        self.assertFalse(olap)

    def test_short_message_without_hints_is_social(self) -> None:
        self.assertEqual(rule_fallback("não sei"), (False, False, False))


if __name__ == "__main__":
    unittest.main()
