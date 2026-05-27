"""
Configuração central do LLM remoto (OpenRouter) usado pelo Streamlit.

Todas as chamadas ao LLM (Chat, OLAP em linguagem natural, extração ML,
Diagnóstico) devem usar ``llm_runtime_config()`` ou os getters deste módulo —
nunca ler ``LLM_MODEL`` / ``LLM_BASE_URL`` direto em outros arquivos.

Por que OpenRouter?
-------------------
OpenRouter expõe uma API **compatível com OpenAI** em ``https://openrouter.ai/api/v1``,
o que nos permite seguir usando o mesmo cliente ``openai.OpenAI`` que o resto do
projeto já consome. O modelo padrão (``openrouter/free``) é um roteador que
sorteia entre dezenas de modelos gratuitos disponíveis na plataforma — útil para
contornar limitações de hardware local sem trocar a stack do projeto.

Documentação: https://openrouter.ai/openrouter/free/api
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Defaults do MVP — OpenRouter (API OpenAI-compatível, modelos free).
# A chave real fica no ``.env`` (variável ``OPENROUTER_API_KEY``); aqui só
# definimos placeholders para evitar erro de import quando ela não está setada.
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "openrouter/auto"

# Endpoint base sem ``/v1`` (alguns ambientes esperam só o host).
OPENROUTER_HOST = "https://openrouter.ai"

# Headers opcionais que aparecem nos rankings do OpenRouter.
# Não são obrigatórios para a API funcionar; quando vazios, o cliente os omite.
DEFAULT_OPENROUTER_TITLE = "Assistente de Lab"
DEFAULT_OPENROUTER_REFERER = "https://github.com/local/assistente-de-lab"


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
    """ID do modelo na API OpenAI-compatível (OpenRouter)."""
    return os.environ.get("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL


def get_llm_api_key() -> str:
    """
    Resolve a chave de API na ordem: ``OPENROUTER_API_KEY``
    → ``OPENAI_API_KEY`` → ``LLM_API_KEY``. Retorna string vazia quando
    nenhuma está configurada (o ``app.py`` mostra essa condição no Diagnóstico).
    """
    return (
        os.environ.get("OPENROUTER_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )


def get_openrouter_app_title() -> str:
    """Título exibido nos rankings do OpenRouter (opcional)."""
    return os.environ.get("OPENROUTER_APP_TITLE", "").strip() or DEFAULT_OPENROUTER_TITLE


def get_openrouter_http_referer() -> str:
    """URL do projeto exibida nos rankings do OpenRouter (opcional)."""
    return (
        os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
        or DEFAULT_OPENROUTER_REFERER
    )


def is_openrouter_endpoint(base_url: str) -> bool:
    """True quando a URL aponta para o OpenRouter (host openrouter.ai)."""
    return "openrouter.ai" in base_url.lower()


def openrouter_default_headers() -> dict[str, str]:
    """
    Cabeçalhos padrão para o cliente OpenAI quando o destino é o OpenRouter.

    ``HTTP-Referer`` e ``X-Title`` são opcionais; quando definidos, fazem o app
    aparecer nos rankings de uso (https://openrouter.ai/rankings).
    """
    headers: dict[str, str] = {}
    referer = get_openrouter_http_referer()
    if referer:
        headers["HTTP-Referer"] = referer
    title = get_openrouter_app_title()
    if title:
        headers["X-Title"] = title
    return headers


def llm_runtime_config() -> tuple[str, str, str]:
    """Retorna ``(base_url_com_v1, model_id, api_key)`` para o cliente OpenAI."""
    base = normalize_openai_base_url(get_llm_base_url_raw())
    return base, get_llm_model(), get_llm_api_key()
