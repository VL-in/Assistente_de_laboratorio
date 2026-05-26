"""
Configuração do LLM para o CrewAI apontando para o LM Studio local.

O CrewAI usa LiteLLM por baixo dos panos. Para falar com APIs OpenAI-compatíveis
fora da OpenAI (LM Studio, Ollama, vLLM) o ``model`` precisa do prefixo
``openai/`` e ``base_url`` apontando para o servidor local.

Os perfis Qwen3.5 já validados (``PROFILE_CHAT_INSTRUCT`` etc.) continuam usados
nas Tools que chamam o cliente OpenAI direto. Aqui só configuramos o LLM
"genérico" do Crew (Triage e Synthesizer).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from llm_config import llm_runtime_config
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


def _model_with_provider(model_id: str) -> str:
    """Garante o prefixo ``openai/`` exigido pelo LiteLLM para LM Studio."""
    mid = (model_id or "").strip()
    if not mid:
        return mid
    if "/" in mid:
        return mid
    return f"openai/{mid}"


def build_llm_for_profile(profile: GenerationProfile) -> Any:
    """
    Instancia um ``crewai.LLM`` com os parâmetros do perfil Qwen3.5.

    Cada perfil cria sua própria instância (Triage e Synthesizer usam
    perfis diferentes). O CrewAI armazena o histórico no agente, então
    reutilizar a mesma instância entre agentes é seguro.

    O parâmetro ``extra_body`` (``chat_template_kwargs.enable_thinking``) é
    repassado quando o modelo é Qwen3.5 — caso contrário, fica fora dos kwargs
    para evitar incompatibilidade com servidores antigos.
    """
    LLM = _crewai_llm_class()
    base, model_id, api_key = llm_runtime_config()
    kwargs: dict[str, Any] = {
        "model": _model_with_provider(model_id),
        "base_url": base,
        "api_key": api_key,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
    }
    if profile.presence_penalty:
        kwargs["presence_penalty"] = profile.presence_penalty
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
