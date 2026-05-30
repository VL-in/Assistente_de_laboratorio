"""Testes do helper Langfuse (sem rede)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from observability.langfuse_client import (
    crew_route_tags,
    crew_route_tags_from_execution,
    langfuse_enabled,
    langfuse_status,
    normalize_langfuse_env,
    update_chat_trace_route,
)


class TestLangfuseClient(unittest.TestCase):
    def test_disabled_without_keys(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "",
                "LANGFUSE_SECRET_KEY": "",
                "LANGFUSE_ENABLED": "1",
            },
            clear=False,
        ):
            self.assertFalse(langfuse_enabled())
            status = langfuse_status()
            self.assertFalse(status["enabled"])
            self.assertFalse(status["public_key_set"])

    def test_enabled_with_keys(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
                "LANGFUSE_ENABLED": "1",
            },
            clear=False,
        ):
            self.assertTrue(langfuse_enabled())

    def test_langfuse_enabled_flag_off(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
                "LANGFUSE_ENABLED": "0",
            },
            clear=False,
        ):
            self.assertFalse(langfuse_enabled())

    def test_crew_route_tags(self) -> None:
        tags = crew_route_tags(use_rag=True, use_olap=False, use_ml=False)
        self.assertIn("feature:chat", tags)
        self.assertIn("route:rag", tags)
        self.assertNotIn("route:olap", tags)

    def test_crew_route_tags_from_execution(self) -> None:
        tags = crew_route_tags_from_execution(
            tool_results={"rag": object(), "olap": object()},
        )
        self.assertIn("route:rag", tags)
        self.assertIn("route:olap", tags)
        self.assertNotIn("route:ml", tags)

        greet = crew_route_tags_from_execution(greeting=True)
        self.assertIn("route:greeter", greet)

        direct = crew_route_tags_from_execution(tool_results={})
        self.assertIn("route:direct", direct)

    def test_normalize_langfuse_env_host_to_base(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_HOST": "https://us.cloud.langfuse.com",
                "LANGFUSE_BASE_URL": "",
            },
            clear=False,
        ):
            normalize_langfuse_env()
            self.assertEqual(
                os.environ.get("LANGFUSE_BASE_URL"), "https://us.cloud.langfuse.com"
            )

    @patch("opentelemetry.trace.get_current_span")
    @patch("langfuse.get_client")
    @patch("observability.langfuse_client.langfuse_enabled", return_value=True)
    def test_update_chat_trace_route_v4(
        self,
        _enabled: object,
        mock_get_client: object,
        mock_get_current_span: object,
    ) -> None:
        mock_client = mock_get_client.return_value
        mock_span = mock_get_current_span.return_value
        mock_span.is_recording.return_value = True

        update_chat_trace_route(
            tool_results={"rag": object()},
        )

        mock_client.update_current_span.assert_called_once()
        metadata = mock_client.update_current_span.call_args.kwargs["metadata"]
        self.assertIn("route:rag", metadata["route_tags"])
        mock_span.set_attribute.assert_called_once_with(
            "langfuse.trace.tags",
            metadata["route_tags"],
        )


if __name__ == "__main__":
    unittest.main()
