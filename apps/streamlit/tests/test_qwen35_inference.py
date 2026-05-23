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
        self.assertTrue(is_qwen35_model("qwen3.5-9b-mtp"))

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

    def test_sanitize_history_strips_any_model(self) -> None:
        from qwen35_inference import sanitize_history_message

        raw = "<think>x</think>\nok"
        self.assertEqual(
            sanitize_history_message("assistant", raw, model_id="llama-3"),
            "ok",
        )


class StreamAnswerTests(unittest.TestCase):
    def test_stream_hides_thinking_until_answer(self) -> None:
        from qwen35_inference import iter_stream_answer_text

        class _Delta:
            def __init__(self, content: str):
                self.content = content

        class _Choice:
            def __init__(self, content: str):
                self.delta = _Delta(content)

        class _Chunk:
            def __init__(self, content: str):
                self.choices = [_Choice(content)]

        stream = [
            _Chunk("<think>\n"),
            _Chunk("secret\n</think>\n\n"),
            _Chunk("Resposta."),
        ]
        parts = list(iter_stream_answer_text(stream, model_id="qwen3.5-9b-mtp"))
        self.assertEqual("".join(parts), "Resposta.")


class CompletionKwargsTests(unittest.TestCase):
    def test_instruct_extra_body(self) -> None:
        kw = build_completion_kwargs(
            model="qwen3.5-9b-mtp",
            profile=PROFILE_CHAT_INSTRUCT,
            max_tokens=1024,
        )
        self.assertEqual(kw["temperature"], 0.7)
        self.assertFalse(kw["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_olap_sql_profile(self) -> None:
        kw = build_completion_kwargs(
            model="qwen3.5-9b-mtp",
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
