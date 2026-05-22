"""
Configuração central do LM Studio para o Streamlit.

Todas as chamadas ao LLM (Chat, OLAP em linguagem natural, Diagnóstico) devem
usar ``llm_runtime_config()`` ou os getters deste módulo — nunca ler
``LLM_MODEL`` / ``LLM_BASE_URL`` direto em outros arquivos.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Padrões do MVP — unsloth/Qwen3.5-9B-MTP-GGUF via LM Studio (API OpenAI-compatível).
# Parâmetros de sampling/thinking: ``qwen35_inference.py`` (model card Hugging Face).
DEFAULT_LLM_BASE_URL = "http://192.168.15.7:1234"
DEFAULT_LLM_MODEL = "qwen3.5-9b-mtp"
DEFAULT_LLM_API_KEY = "lm-studio"


def _bootstrap_dotenv() -> None:
    """Carrega ``.env`` da raiz do repositório quando existir (dev local)."""
    here = Path(__file__).resolve().parent
    for candidate in (here / ".env", here.parent.parent / ".env"):
        if candidate.is_file():
            load_dotenv(candidate)
            return
    load_dotenv()


_bootstrap_dotenv()


def normalize_openai_base_url(url: str) -> str:
    """Garante sufixo ``/v1`` exigido pelo SDK OpenAI."""
    u = url.strip().rstrip("/")
    if not u:
        return u
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def get_llm_base_url_raw() -> str:
    """URL base configurada (sem forçar ``/v1``)."""
    return os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL


def get_llm_model() -> str:
    """ID do modelo na API OpenAI-compatível (LM Studio)."""
    return os.environ.get("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL


def get_llm_api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
        or DEFAULT_LLM_API_KEY
    )


def llm_runtime_config() -> tuple[str, str, str]:
    """Retorna ``(base_url_com_v1, model_id, api_key)`` para o cliente OpenAI."""
    base = normalize_openai_base_url(get_llm_base_url_raw())
    return base, get_llm_model(), get_llm_api_key()
