"""
Perfis de inferência usados nas chamadas ao LLM remoto (OpenRouter).

O nome do módulo segue mantido porque os parâmetros originais foram derivados
do model card do Qwen3.5 — ainda úteis quando o ``LLM_MODEL`` aponta para
``qwen/qwen3.5-...`` no OpenRouter. Para outros modelos (Llama, Mistral,
Gemma, etc.), o ``select_chat_profile`` devolve um perfil neutro e os campos
específicos do Qwen (``extra_body.chat_template_kwargs``) **não** são enviados,
evitando erros em providers que não reconhecem o template.

Resumo dos perfis exportados:
- ``PROFILE_CHAT_INSTRUCT`` — chat geral (instruct, sem raciocínio).
- ``PROFILE_CHAT_THINKING`` — chat com raciocínio explícito (Qwen3.5 only).
- ``PROFILE_OLAP_SQL`` — geração de SQL DuckDB (temperatura baixa).
- ``PROFILE_CHAT_ROUTER`` — classificador JSON do roteador de intenção.

Referência Qwen3.5: https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF
Documentação OpenRouter: https://openrouter.ai/docs
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

# Resposta do chat — limites conservadores para conter custo no OpenRouter
# (prompt + saída). Predição ML: resposta curta (tabela já veio no system).
DEFAULT_CHAT_MAX_TOKENS = 2048
DEFAULT_CHAT_ML_MAX_TOKENS = 768
DEFAULT_CHAT_MAX_HISTORY_TURNS = 4
DEFAULT_CHAT_ML_MAX_HISTORY_TURNS = 2
DEFAULT_CHAT_HISTORY_CHARS = 400
DEFAULT_CHAT_ML_HISTORY_CHARS = 280
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

# Classificador de intenção do chat (JSON curto, sem thinking).
PROFILE_CHAT_ROUTER = GenerationProfile(
    name="instruct_router",
    temperature=0.2,
    top_p=0.9,
    presence_penalty=0.0,
    enable_thinking=False,
)

DEFAULT_ROUTER_MAX_TOKENS = 128


def _env_positive_int(
    name: str,
    default: int,
    *,
    min_val: int = 1,
    max_val: int = 32768,
) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_val, min(max_val, value))


def chat_max_tokens(*, ml_route: bool = False) -> int:
    """Limite de tokens de saída (``CHAT_MAX_TOKENS`` / ``CHAT_ML_MAX_TOKENS``)."""
    if ml_route:
        return _env_positive_int(
            "CHAT_ML_MAX_TOKENS", DEFAULT_CHAT_ML_MAX_TOKENS, min_val=64
        )
    return _env_positive_int("CHAT_MAX_TOKENS", DEFAULT_CHAT_MAX_TOKENS, min_val=256)


def chat_max_history_turns(*, ml_route: bool = False) -> int:
    """Pares user+assistant no histórico enviado ao modelo."""
    if ml_route:
        return _env_positive_int(
            "CHAT_ML_MAX_HISTORY_TURNS",
            DEFAULT_CHAT_ML_MAX_HISTORY_TURNS,
            min_val=0,
            max_val=30,
        )
    return _env_positive_int(
        "CHAT_MAX_HISTORY_TURNS",
        DEFAULT_CHAT_MAX_HISTORY_TURNS,
        min_val=0,
        max_val=30,
    )


def chat_history_chars_per_message(*, ml_route: bool = False) -> int:
    """Truncagem por mensagem no histórico (extrator, roteador e chat)."""
    if ml_route:
        return _env_positive_int(
            "CHAT_ML_HISTORY_CHARS",
            DEFAULT_CHAT_ML_HISTORY_CHARS,
            min_val=80,
            max_val=8000,
        )
    return _env_positive_int(
        "CHAT_HISTORY_CHARS", DEFAULT_CHAT_HISTORY_CHARS, min_val=80, max_val=8000
    )


def format_history_snippet(
    history: list[dict],
    *,
    max_turns: int = 3,
    max_chars_per_message: int | None = None,
) -> str:
    """Últimos turnos user/assistant para roteador, extrator ML e desambiguação."""
    if not history or max_turns <= 0:
        return "(sem histórico anterior)"
    tail = history[-(max_turns * 2) :]
    char_cap = max_chars_per_message if max_chars_per_message is not None else 400
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if char_cap > 0 and len(content) > char_cap:
            content = content[:char_cap] + "…"
        label = "Usuário" if role == "user" else "Assistente"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(sem histórico anterior)"


def effective_chat_limits(
    *,
    run_ml: bool,
    max_tokens: int,
    max_history_turns: int,
) -> tuple[int, int]:
    """Aplica tetos da rota ML sem alterar limites gerais quando ``run_ml`` é false."""
    if not run_ml:
        return max_tokens, max_history_turns
    return (
        min(max_tokens, chat_max_tokens(ml_route=True)),
        min(max_history_turns, chat_max_history_turns(ml_route=True)),
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
    Chama a API OpenAI-compatível (OpenRouter, OpenAI, vLLM, etc.).

    Quando o modelo é Qwen3.5, anexamos ``extra_body`` com
    ``chat_template_kwargs.enable_thinking`` — se o provider não aceitar (alguns
    serviços recusam o campo), repetimos a chamada sem ele.
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
    if role == "assistant":
        return strip_thinking_blocks(content)
    return content


def iter_stream_answer_text(stream: Any, *, model_id: str):
    """
    Itera tokens da resposta omitindo blocos de raciocínio durante o streaming.

    Para modelos Qwen3.5, acumula o texto bruto e só emite a parte já visível
    após ``strip_thinking_blocks``, evitando flash de ``<think>``
    na UI antes do pós-processamento.
    """
    if not is_qwen35_model(model_id):
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
        return

    raw = ""
    yielded_len = 0
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue
        raw += delta.content
        visible = strip_thinking_blocks(raw)
        if len(visible) > yielded_len:
            yield visible[yielded_len:]
            yielded_len = len(visible)
