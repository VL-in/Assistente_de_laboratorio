"""
Modelo juiz (LLM-as-judge) para metricas DeepEval.

Prioridade em modo ``auto``:
1. ``OPENROUTER_API_KEY`` -> ``OpenRouterModel`` (mesma stack do assistente)
2. ``OPENAI_API_KEY`` -> modelo OpenAI padrao do DeepEval
"""

from __future__ import annotations

import os
from typing import Literal

JudgeProvider = Literal["auto", "openrouter", "openai"]


def resolve_judge_provider(provider: JudgeProvider = "auto") -> JudgeProvider:
    """Resolve qual backend usar para o juiz das metricas."""
    env_provider = os.environ.get("EVAL_JUDGE_PROVIDER", "").strip().lower()
    if env_provider in ("openrouter", "openai"):
        return env_provider  # type: ignore[return-value]
    if provider in ("openrouter", "openai"):
        return provider

    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "openrouter"

    from llm_config import get_llm_api_key, get_llm_base_url_raw, is_openrouter_endpoint
    from llm_config import normalize_openai_base_url

    base = normalize_openai_base_url(get_llm_base_url_raw())
    if get_llm_api_key() and is_openrouter_endpoint(base):
        return "openrouter"

    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"

    return "openrouter"


def resolve_judge_model_name(explicit: str | None = None) -> str:
    """Nome do modelo juiz (slug OpenRouter ou OpenAI)."""
    if explicit and explicit.strip():
        return explicit.strip()
    for env_name in ("EVAL_JUDGE_MODEL", "OPENROUTER_MODEL_NAME", "LLM_MODEL", "OPENAI_MODEL_NAME"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            return raw
    return "openrouter/auto"


def build_judge_model(
    *,
    provider: JudgeProvider = "auto",
    model: str | None = None,
):
    """
    Instancia o juiz DeepEval.

    Retorna ``OpenRouterModel`` (OpenRouter) ou ``str`` (modelo OpenAI nativo).
    """
    resolved_provider = resolve_judge_provider(provider)
    model_name = resolve_judge_model_name(model)

    if resolved_provider == "openrouter":
        from deepeval.models import OpenRouterModel

        from llm_config import (
            get_llm_api_key,
            get_llm_base_url_raw,
            normalize_openai_base_url,
            openrouter_default_headers,
        )

        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or get_llm_api_key()
        if not api_key:
            raise RuntimeError(
                "Juiz OpenRouter requer OPENROUTER_API_KEY (ou chave em LLM_API_KEY)."
            )

        headers = openrouter_default_headers()
        client_kwargs: dict = {}
        if headers:
            client_kwargs["default_headers"] = headers

        return OpenRouterModel(
            model=model_name,
            api_key=api_key,
            base_url=normalize_openai_base_url(get_llm_base_url_raw()),
            **client_kwargs,
        )

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "Juiz OpenAI requer OPENAI_API_KEY. "
            "Use --judge-provider openrouter ou defina OPENROUTER_API_KEY."
        )
    return model_name


def judge_backend_label(provider: JudgeProvider = "auto") -> str:
    resolved = resolve_judge_provider(provider)
    model_name = resolve_judge_model_name(None)
    return f"{resolved} ({model_name})"
