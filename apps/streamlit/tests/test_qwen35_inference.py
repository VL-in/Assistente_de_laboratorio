"""Testes de parâmetros e pós-processamento Qwen3.5."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen35_inference import (  # noqa: E402
    PROFILE_CHAT_INSTRUCT,
    PROFILE_OLAP_SQL,
    build_completion_kwargs,
    is_qwen35_model,
    select_chat_profile,
    strip_thinking_blocks,
    strip_thinking_blocks_with_flag,
)


class Qwen35ModelDetectionTests(unittest.TestCase):
    def test_detects_mtp_id(self) -> None:
        self.assertTrue(is_qwen35_model("qwen3.5-4b-mtp"))

    def test_ignores_other_models(self) -> None:
        self.assertFalse(is_qwen35_model("llama-3-8b"))


class StripThinkingTests(unittest.TestCase):
    def test_removes_closed_block(self) -> None:
        raw = "<think>\nplan\n</think>\n\nResposta final."
        self.assertEqual(strip_thinking_blocks(raw), "Resposta final.")

    def test_truncated_open_block(self) -> None:
        raw = "<think>\nainda pensando"
        cleaned, truncated = strip_thinking_blocks_with_flag(raw)
        self.assertTrue(truncated)
        self.assertEqual(cleaned, "")


class CompletionKwargsTests(unittest.TestCase):
    def test_instruct_extra_body(self) -> None:
        kw = build_completion_kwargs(
            model="qwen3.5-4b-mtp",
            profile=PROFILE_CHAT_INSTRUCT,
            max_tokens=1024,
        )
        self.assertEqual(kw["temperature"], 0.7)
        self.assertFalse(kw["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_olap_sql_profile(self) -> None:
        kw = build_completion_kwargs(
            model="qwen3.5-4b-mtp",
            profile=PROFILE_OLAP_SQL,
            max_tokens=2048,
        )
        self.assertEqual(kw["temperature"], 0.6)
        self.assertEqual(kw["presence_penalty"], 0.0)

    def test_generic_model_no_extra_body(self) -> None:
        profile = select_chat_profile(model_id="other-model", use_thinking=False)
        kw = build_completion_kwargs(
            model="other-model",
            profile=profile,
            max_tokens=512,
        )
        self.assertNotIn("extra_body", kw)
