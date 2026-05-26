"""
Configuração do LLM para o CrewAI apontando para o OpenRouter.

⚠️ STATUS — CÓDIGO PREPARATÓRIO, NÃO USADO EM RUNTIME (2026-05).
   O pipeline multiagente atual (``agents/crew.py``, ``agents/runner.py``)
   chama o cliente ``openai.OpenAI`` direto (via ``create_chat_completion``)
   para preservar streaming nativo do Streamlit e os perfis Qwen3.5
   (``PROFILE_CHAT_INSTRUCT`` etc.) — NÃO passa por ``crewai.LLM``.
   Este módulo fica aqui pronto para o dia em que adicionarmos um agente
   CrewAI "literal" com tool-calling autônomo (ex.: Auditor da Fase 4).
   Manter os builders facilita esse upgrade sem reinventar a roda.

O CrewAI usa LiteLLM por baixo dos panos. Para falar com o OpenRouter (API
OpenAI-compatível) o ``model`` precisa do prefixo ``openrouter/`` reconhecido
pelo LiteLLM, ou alternativamente ``openai/<slug>`` com ``base_url`` apontando
para o endpoint do OpenRouter. Optamos pela primeira forma porque o LiteLLM já
mapeia automaticamente para ``https://openrouter.ai/api/v1`` e propaga os
headers ``HTTP-Referer`` / ``X-Title``.

Os perfis Qwen3.5 já validados (``PROFILE_CHAT_INSTRUCT`` etc.) continuam usados
nas Tools que chamam o cliente OpenAI direto. Aqui só configuramos o LLM
"genérico" do Crew (Triage e Synthesizer) para quando ele for ativado.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from llm_config import (
    get_openrouter_app_title,
    get_openrouter_http_referer,
    is_openrouter_endpoint,
    llm_runtime_config,
)
from qwen35_inference import (
    PROFILE_CHAT_INSTRUCT,
    PROFILE_CHAT_ROUTER,
    PROFILE_CHAT_THINKING,
    GenerationProfile,
    is_qwen35_model,
)


def _crewai_llm_class() -> Any:
    """
    Importa ``crewai.LLM`` em tempo de uso.

    Mantemos o import preguiçoso para que ``apps/streamlit`` continue rodando
    mesmo sem o pacote CrewAI instalado (modo legado via ``USE_CREWAI=0``).
    """
    from crewai import LLM  # type: ignore[import-not-found]

    return LLM


def _model_with_provider(model_id: str, *, base_url: str) -> str:
    """
    Adiciona o prefixo de provider esperado pelo LiteLLM.

    - Quando o destino é o OpenRouter, usa o prefixo ``openrouter/`` (o LiteLLM
      reconhece e roteia direto, sem precisar de ``base_url`` extra).
    - Caso contrário (endpoint OpenAI-compatible genérico), mantém o prefixo
      ``openai/`` e o ``base_url`` é quem define o destino.
    - Se o ID já contém ``/`` (ex.: ``openrouter/free``, ``meta-llama/...``),
      assumimos que o prefixo já está correto e não duplicamos.
    """
    mid = (model_id or "").strip()
    if not mid:
        return mid
    if "/" in mid:
        if is_openrouter_endpoint(base_url) and not mid.startswith("openrouter/"):
            return f"openrouter/{mid}"
        return mid
    if is_openrouter_endpoint(base_url):
        return f"openrouter/{mid}"
    return f"openai/{mid}"


def _openrouter_headers() -> dict[str, str]:
    """Headers ``HTTP-Referer`` / ``X-Title`` propagados para o LiteLLM."""
    headers: dict[str, str] = {}
    referer = get_openrouter_http_referer()
    if referer:
        headers["HTTP-Referer"] = referer
    title = get_openrouter_app_title()
    if title:
        headers["X-Title"] = title
    return headers


def build_llm_for_profile(profile: GenerationProfile) -> Any:
    """
    Instancia um ``crewai.LLM`` com os parâmetros do perfil.

    Cada perfil cria sua própria instância (Triage e Synthesizer usam
    perfis diferentes). O CrewAI armazena o histórico no agente, então
    reutilizar a mesma instância entre agentes é seguro.

    O parâmetro ``extra_body`` (``chat_template_kwargs.enable_thinking``) é
    repassado apenas quando o modelo identificado é Qwen3.5 — caso contrário,
    fica fora dos kwargs para evitar incompatibilidade com modelos servidos via
    OpenRouter (Llama, Mistral, Gemma etc.).
    """
    LLM = _crewai_llm_class()
    base, model_id, api_key = llm_runtime_config()
    kwargs: dict[str, Any] = {
        "model": _model_with_provider(model_id, base_url=base),
        "base_url": base,
        "api_key": api_key,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
    }
    if profile.presence_penalty:
        kwargs["presence_penalty"] = profile.presence_penalty
    if is_openrouter_endpoint(base):
        headers = _openrouter_headers()
        if headers:
            kwargs["extra_headers"] = headers
    if is_qwen35_model(model_id):
        kwargs["extra_body"] = {
            "top_k": profile.top_k,
            "chat_template_kwargs": {"enable_thinking": profile.enable_thinking},
        }
    return LLM(**kwargs)


@lru_cache(maxsize=4)
def llm_for_triage() -> Any:
    """LLM determinístico para classificar intenção (saída JSON curta)."""
    return build_llm_for_profile(PROFILE_CHAT_ROUTER)


@lru_cache(maxsize=4)
def llm_for_synthesizer_instruct() -> Any:
    """LLM padrão do Synthesizer — instruct sem thinking."""
    return build_llm_for_profile(PROFILE_CHAT_INSTRUCT)


@lru_cache(maxsize=4)
def llm_for_synthesizer_thinking() -> Any:
    """LLM do Synthesizer quando o usuário liga raciocínio explícito."""
    return build_llm_for_profile(PROFILE_CHAT_THINKING)


def reset_llm_cache() -> None:
    """Limpa caches LRU; usar quando ``LLM_BASE_URL``/``LLM_MODEL`` mudar."""
    llm_for_triage.cache_clear()
    llm_for_synthesizer_instruct.cache_clear()
    llm_for_synthesizer_thinking.cache_clear()
