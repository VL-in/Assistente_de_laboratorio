"""Testes do helper Langfuse (sem rede)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from observability.langfuse_client import (
    crew_route_tags,
    langfuse_enabled,
    langfuse_status,
    normalize_langfuse_env,
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


if __name__ == "__main__":
    unittest.main()
