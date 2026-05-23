"""
Parâmetros de inferência alinhados ao Qwen3.5 / Qwen3.5-MTP (Unsloth GGUF).

Referência: https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF
- Modo thinking (padrão do modelo): blocos ``<think>`` antes da resposta.
- Modo instruct: ``enable_thinking=False`` via ``chat_template_kwargs`` (recomendado para
  assistente documental e geração de SQL).
- Sampling: tabelas "Best Practices" e "Instruct mode" do model card Qwen3.5.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

# Resposta do chat — documentação sugere até 32k tokens; MVP usa default menor na UI.
DEFAULT_CHAT_MAX_TOKENS = 4096
DEFAULT_SQL_MAX_TOKENS = 2048

_THINK_CLOSED = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_REASONING_CLOSED = re.compile(
    r"<reasoning>.*?</reasoning>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN = re.compile(r"<think>|<reasoning>", re.IGNORECASE)


@dataclass(frozen=True)
class GenerationProfile:
    """Perfil de sampling + thinking para ``chat.completions.create``."""

    name: str
    temperature: float
    top_p: float
    presence_penalty: float
    enable_thinking: bool
    top_k: int = 20


# Instruct (non-thinking) — tarefas gerais do assistente documental.
PROFILE_CHAT_INSTRUCT = GenerationProfile(
    name="instruct_general",
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    enable_thinking=False,
)

# Thinking — raciocínio explícito (mais lento, mais tokens).
PROFILE_CHAT_THINKING = GenerationProfile(
    name="thinking_general",
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    enable_thinking=True,
)

# SQL / código preciso — doc: thinking mode coding OU instruct com temp baixa.
PROFILE_OLAP_SQL = GenerationProfile(
    name="instruct_sql",
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    enable_thinking=False,
)


def is_qwen35_model(model_id: str) -> bool:
    m = (model_id or "").lower()
    return "qwen3.5" in m or "qwen3_5" in m or "qwen35" in m or "qwen3-5" in m


def env_enable_thinking_default() -> bool:
    """``LLM_ENABLE_THINKING=1|true|yes`` liga thinking por padrão na UI."""
    return os.environ.get("LLM_ENABLE_THINKING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def strip_thinking_blocks_with_flag(text: str) -> tuple[str, bool]:
    """
    Retorna ``(texto_limpo, truncado)``.

    ``truncado`` quando há tag de thinking aberta sem resposta final (ex.: ``max_tokens``).
    """
    if not text:
        return "", False
    tmp = _THINK_CLOSED.sub("", text)
    tmp = _REASONING_CLOSED.sub("", tmp)
    truncated = _THINK_OPEN.search(tmp) is not None
    return strip_thinking_blocks(text), truncated


def strip_thinking_blocks(text: str) -> str:
    """
    Remove raciocínio embutido; mantém só a resposta final para histórico e UI.

    Qwen3.5: o histórico não deve incluir thinking (model card, seção Best Practices).
    """
    if not text:
        return ""
    cleaned = _THINK_CLOSED.sub("", text)
    cleaned = _REASONING_CLOSED.sub("", cleaned)
    open_match = _THINK_OPEN.search(cleaned)
    if open_match:
        cleaned = cleaned[: open_match.start()]
    return cleaned.strip()


def split_thinking_and_answer(text: str) -> tuple[str, str]:
    """Retorna ``(thinking, answer)`` para diagnóstico opcional."""
    if not text:
        return "", ""
    thinking_parts: list[str] = []
    for pat in (_THINK_CLOSED, _REASONING_CLOSED):
        for m in pat.finditer(text):
            thinking_parts.append(m.group(0))
    answer = strip_thinking_blocks(text)
    return "\n".join(thinking_parts).strip(), answer


def select_chat_profile(*, model_id: str, use_thinking: bool) -> GenerationProfile:
    if is_qwen35_model(model_id) and use_thinking:
        return PROFILE_CHAT_THINKING
    if is_qwen35_model(model_id):
        return PROFILE_CHAT_INSTRUCT
    # Outros modelos: parâmetros neutros, sem extra_body Qwen.
    return GenerationProfile(
        name="generic",
        temperature=0.7,
        top_p=0.9,
        presence_penalty=0.0,
        enable_thinking=False,
    )


def build_completion_kwargs(
    *,
    model: str,
    profile: GenerationProfile,
    max_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    """Kwargs para ``OpenAI.chat.completions.create`` (exceto ``messages``)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "stream": stream,
    }
    kwargs["presence_penalty"] = profile.presence_penalty
    if is_qwen35_model(model):
        kwargs["extra_body"] = {
            "top_k": profile.top_k,
            "chat_template_kwargs": {"enable_thinking": profile.enable_thinking},
        }
    return kwargs


def create_chat_completion(
    client: Any,
    *,
    messages: list[dict[str, str]],
    model: str,
    profile: GenerationProfile,
    max_tokens: int,
    stream: bool = False,
) -> Any:
    """
    Chama a API com perfil Qwen3.5; se ``extra_body`` falhar (LM Studio antigo),
    repete sem parâmetros específicos do template.
    """
    kwargs = build_completion_kwargs(
        model=model,
        profile=profile,
        max_tokens=max_tokens,
        stream=stream,
    )
    try:
        return client.chat.completions.create(messages=messages, **kwargs)
    except Exception:
        if "extra_body" not in kwargs:
            raise
        fallback = {k: v for k, v in kwargs.items() if k != "extra_body"}
        return client.chat.completions.create(messages=messages, **fallback)


def sanitize_history_message(role: str, content: str, *, model_id: str) -> str:
    """Histórico enviado ao modelo: assistant sem blocos de thinking."""
    if role == "assistant" and is_qwen35_model(model_id):
        return strip_thinking_blocks(content)
    return content
