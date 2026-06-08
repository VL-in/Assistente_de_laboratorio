"""
Modelo juiz (LLM-as-judge) para metricas DeepEval.

Usa exclusivamente o OpenRouter com o modelo gratuito ``openai/gpt-oss-20b:free``
via ``OpenRouterFreeJudgeModel`` — wrapper que:

- Desativa structured outputs (o tier free nao suporta).
- Extrai JSON manualmente da resposta em texto (regex robusto).
- Faz retry automatico em 429 com backoff exponencial.

Variaveis de ambiente relevantes
---------------------------------
EVAL_JUDGE_MODEL         : sobrepoe o slug do modelo (padrao: openai/gpt-oss-20b:free)
EVAL_JUDGE_RETRY_MAX     : tentativas em 429 (padrao: 5)
EVAL_JUDGE_RETRY_DELAY_S : delay inicial de backoff em segundos (padrao: 30)
OPENROUTER_API_KEY       : chave OpenRouter (obrigatoria)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import warnings
from typing import Any, Optional, Tuple, Union

_DEFAULT_FREE_JUDGE_MODEL = "openai/gpt-oss-20b:free"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Mantido por compatibilidade com run_assistente_eval.py
JudgeProvider = str

# DeepEval >= 4.x exige heranca de DeepEvalBaseLLM; versoes anteriores nao tinham a ABC.
try:
    from deepeval.models import DeepEvalBaseLLM as _DeepEvalBaseLLM  # noqa: PLC0415
    _JUDGE_BASE = _DeepEvalBaseLLM
except ImportError:
    _JUDGE_BASE = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers de JSON
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """Extrai o primeiro objeto JSON de uma string de texto livre."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidate = m.group(1).strip() if m else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    m2 = re.search(r"\{[\s\S]*\}", candidate)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Nao foi possivel extrair JSON valido da resposta: {text[:300]}")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


# ---------------------------------------------------------------------------
# OpenRouterFreeJudgeModel
# ---------------------------------------------------------------------------

class OpenRouterFreeJudgeModel(_JUDGE_BASE):
    """
    Wrapper DeepEval-compativel para modelos gratuitos do OpenRouter.

    Herda de DeepEvalBaseLLM quando disponivel (DeepEval >= 4.x exige),
    caindo de volta para object se a ABC nao existir.
    Suporta schema Pydantic via parsing de texto e retry automatico em 429.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = _OPENROUTER_BASE_URL,
        extra_headers: Optional[dict] = None,
        retry_max: int = 5,
        retry_delay_s: float = 30.0,
    ) -> None:
        # Seta atributos privados ANTES de chamar load_model (a ABC chama load_model
        # no seu __init__, mas nao chamamos super().__init__ para evitar esse fluxo).
        self.name = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._extra_headers = extra_headers or {}
        self._retry_max = retry_max
        self._retry_delay_s = retry_delay_s
        self._client: Any = None
        self._async_client: Any = None
        self.load_model()
        # Alias exigido pela ABC (self.model = instancia do cliente/modelo carregado)
        self.model = self._client

    # ------------------------------------------------------------------
    # DeepEvalBaseLLM interface
    # ------------------------------------------------------------------

    def load_model(self) -> "OpenRouterFreeJudgeModel":
        from openai import AsyncOpenAI, OpenAI  # noqa: PLC0415

        common: dict = dict(
            api_key=self._api_key,
            base_url=self._base_url,
            default_headers=self._extra_headers or None,
        )
        self._client = OpenAI(**common)
        self._async_client = AsyncOpenAI(**common)
        return self

    def get_model_name(self) -> str:
        return self.name

    def generate(self, prompt: str, schema: Any = None) -> Tuple[Any, float]:
        """Chamada sincrona com retry em 429."""
        return self._run_sync(prompt, schema)

    async def a_generate(self, prompt: str, schema: Any = None) -> Tuple[Any, float]:
        """Chamada assincrona com retry em 429."""
        return await self._run_async(prompt, schema)

    # Capacidades declaradas — evitam que o DeepEval tente structured outputs
    def supports_structured_outputs(self) -> bool:
        return False

    def supports_log_probs(self) -> bool:
        return False

    def supports_temperature(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chat(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    async def _achat(self, prompt: str) -> str:
        resp = await self._async_client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    def _run_sync(self, prompt: str, schema: Any) -> Tuple[Any, float]:
        delay = self._retry_delay_s
        last_exc: Exception = RuntimeError("retry loop nao executou")
        for attempt in range(1, self._retry_max + 1):
            try:
                text = self._chat(prompt)
                return self._finalize(text, schema), 0.0
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < self._retry_max:
                    warnings.warn(
                        f"[JudgeModel] 429 na tentativa {attempt}/{self._retry_max} "
                        f"— aguardando {delay:.0f}s antes de tentar novamente.",
                        UserWarning,
                        stacklevel=2,
                    )
                    time.sleep(delay)
                    delay *= 2
                    last_exc = exc
                else:
                    raise
        raise last_exc

    async def _run_async(self, prompt: str, schema: Any) -> Tuple[Any, float]:
        delay = self._retry_delay_s
        last_exc: Exception = RuntimeError("retry loop nao executou")
        for attempt in range(1, self._retry_max + 1):
            try:
                text = await self._achat(prompt)
                return self._finalize(text, schema), 0.0
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < self._retry_max:
                    warnings.warn(
                        f"[JudgeModel] 429 na tentativa {attempt}/{self._retry_max} "
                        f"— aguardando {delay:.0f}s antes de tentar novamente.",
                        UserWarning,
                        stacklevel=2,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                    last_exc = exc
                else:
                    raise
        raise last_exc

    @staticmethod
    def _finalize(text: str, schema: Any) -> Any:
        if schema is None:
            return text
        data = _extract_json(text)
        return schema.model_validate(data)


# ---------------------------------------------------------------------------
# Resolucao de provider / modelo
# ---------------------------------------------------------------------------

def resolve_judge_provider(*_: Any) -> str:
    return "openrouter"


def resolve_judge_model_name(explicit: Union[str, None] = None, *_: Any) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    raw = os.environ.get("EVAL_JUDGE_MODEL", "").strip()
    return raw if raw else _DEFAULT_FREE_JUDGE_MODEL


# ---------------------------------------------------------------------------
# Factory principal
# ---------------------------------------------------------------------------

def build_judge_model(
    *,
    provider: str = "auto",  # noqa: ARG001 — aceito para compatibilidade de CLI
    model: Union[str, None] = None,
) -> OpenRouterFreeJudgeModel:
    """
    Instancia ``OpenRouterFreeJudgeModel`` com ``openai/gpt-oss-20b:free``.

    Requer ``OPENROUTER_API_KEY`` no ambiente.
    O modelo pode ser sobreposto via ``EVAL_JUDGE_MODEL`` ou ``--judge-model``.
    """
    model_name = resolve_judge_model_name(model)

    try:
        from llm_config import (  # noqa: PLC0415
            get_llm_api_key,
            get_llm_base_url_raw,
            normalize_openai_base_url,
            openrouter_default_headers,
        )
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or get_llm_api_key()
        base_url = normalize_openai_base_url(get_llm_base_url_raw())
        extra_headers = openrouter_default_headers()
    except ImportError:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        base_url = _OPENROUTER_BASE_URL
        extra_headers = {}

    if not api_key:
        raise RuntimeError(
            "Juiz requer OPENROUTER_API_KEY no ambiente. "
            "Adicione ao .env: OPENROUTER_API_KEY=sk-or-v1-..."
        )

    retry_max = int(os.environ.get("EVAL_JUDGE_RETRY_MAX", "5"))
    retry_delay = float(os.environ.get("EVAL_JUDGE_RETRY_DELAY_S", "30"))

    return OpenRouterFreeJudgeModel(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        extra_headers=extra_headers,
        retry_max=retry_max,
        retry_delay_s=retry_delay,
    )


def judge_backend_label(*_: Any) -> str:
    model_name = resolve_judge_model_name(None)
    return f"openrouter ({model_name})"
