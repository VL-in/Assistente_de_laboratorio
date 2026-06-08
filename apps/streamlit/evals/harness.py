"""
Runtime e execução end-to-end do Assistente de Lab para avaliações DeepEval.

Carrega o mesmo pipeline do chat Streamlit (Greeter → Triage → Tools → Synthesizer),
sem depender de ``streamlit``. Use via ``run_assistente_eval.py`` ou testes pytest.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Evals: desliga Langfuse antes de importar llm_config (patch quebra em 429).
_EVALS_DIR = Path(__file__).resolve().parent
if str(_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALS_DIR))
from eval_bootstrap import configure_eval_env  # noqa: E402

configure_eval_env()

# Garante imports de ``agents``, ``rag``, ``ml``, etc.
_STREAMLIT_ROOT = Path(__file__).resolve().parent.parent
if str(_STREAMLIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_ROOT))

from openai import OpenAI  # noqa: E402

from agents.runner import CrewRunResult, run_crew_chat  # noqa: E402
from llm_config import (  # noqa: E402
    get_llm_api_key,
    get_llm_model,
    is_openrouter_endpoint,
    llm_runtime_config,
    openrouter_default_headers,
)
from ml.chat_infer import load_chat_model_bundle  # noqa: E402
from ml.paths import chat_ml_model_available, chat_ml_model_path  # noqa: E402
from ml.training import ModelBundle  # noqa: E402
from olap.ingest import has_ingested_tables  # noqa: E402
from qwen35_inference import (  # noqa: E402
    chat_history_chars_per_message,
    chat_max_history_turns,
    chat_max_tokens,
    create_chat_completion,
    effective_chat_limits,
    select_chat_profile,
    strip_thinking_blocks,
)
from rag.index_txtai import embeddings_config, index_mtime, index_ready  # noqa: E402
from rag.paths import txtai_index_path  # noqa: E402
from rag.rerank import env_rerank_enabled, load_reranker_safe  # noqa: E402


@dataclass
class EvalRuntime:
    """Recursos compartilhados entre vários casos de teste."""

    client: OpenAI
    model: str
    rag_backend: object | None
    rag_reranker: object | None
    ml_bundle: ModelBundle | None
    ml_model_path: Path | None
    documents_available: bool
    spreadsheets_available: bool
    ml_available: bool
    rag_top_k: int
    max_tokens: int
    max_history_turns: int


@dataclass
class TurnResult:
    """Saída observável de uma pergunta ao assistente."""

    actual_output: str
    retrieval_context: list[str]
    crew_result: CrewRunResult


def _openai_client() -> OpenAI:
    base, _, api_key = llm_runtime_config()
    timeout_s = float(os.environ.get("LLM_TIMEOUT_S", "120"))
    headers = openrouter_default_headers() if is_openrouter_endpoint(base) else {}
    return OpenAI(
        base_url=base,
        api_key=api_key,
        timeout=timeout_s,
        default_headers=headers or None,
    )


def _load_rag_backend() -> object | None:
    if not index_ready():
        return None
    if index_mtime() <= 0:
        return None
    from txtai import Embeddings

    emb = Embeddings(embeddings_config())
    emb.load(str(txtai_index_path()))
    return emb


def _load_ml_bundle() -> tuple[ModelBundle | None, Path | None]:
    path = chat_ml_model_path()
    if not path.is_file():
        return None, None
    return load_chat_model_bundle(path), path


def build_eval_runtime() -> EvalRuntime:
    """Monta cliente LLM, índice RAG, modelo ML e flags de disponibilidade."""
    documents_available = index_ready()
    spreadsheets_available = has_ingested_tables()
    ml_available = chat_ml_model_available()

    rag_backend = _load_rag_backend() if documents_available else None
    rag_reranker = None
    if documents_available and env_rerank_enabled():
        rag_reranker = load_reranker_safe().model

    ml_bundle, ml_path = _load_ml_bundle() if ml_available else (None, None)

    return EvalRuntime(
        client=_openai_client(),
        model=get_llm_model(),
        rag_backend=rag_backend,
        rag_reranker=rag_reranker,
        ml_bundle=ml_bundle,
        ml_model_path=ml_path,
        documents_available=documents_available,
        spreadsheets_available=spreadsheets_available,
        ml_available=ml_available,
        rag_top_k=int(os.environ.get("EVAL_RAG_TOP_K", "6")),
        max_tokens=chat_max_tokens(ml_route=False),
        max_history_turns=chat_max_history_turns(ml_route=False),
    )


def runtime_status(runtime: EvalRuntime) -> dict[str, bool]:
    """Resumo legível do que está disponível para evals."""
    return {
        "documents_index": runtime.documents_available,
        "olap_tables": runtime.spreadsheets_available,
        "ml_model": runtime.ml_available,
        "openrouter_key_set": bool(get_llm_api_key()),
    }


def runtime_paths() -> dict[str, str]:
    """Caminhos resolvidos dos dados persistentes (para diagnóstico)."""
    from ml.paths import chat_ml_model_path, ml_models_root
    from olap.paths import duckdb_data_root

    return {
        "txtai_index": str(txtai_index_path()),
        "duckdb": str(duckdb_data_root()),
        "ml_model": str(chat_ml_model_path()),
        "ml_models_root": str(ml_models_root()),
    }


def _golden_meta(golden: Any) -> dict[str, Any]:
    meta = getattr(golden, "additional_metadata", None) or {}
    return meta if isinstance(meta, dict) else {}


def golden_requires(golden: Any) -> dict[str, bool]:
    """Flags ``requires_*`` anotadas no golden."""
    meta = _golden_meta(golden)
    return {
        "index": bool(meta.get("requires_index", True)),
        "olap": bool(meta.get("requires_olap", False)),
        "ml": bool(meta.get("requires_ml_model", False)),
    }


def golden_runnable(golden: Any, runtime: EvalRuntime) -> tuple[bool, str | None]:
    """
    True quando o runtime atual atende aos pré-requisitos do golden.

    Retorna também um motivo legível quando False.
    """
    req = golden_requires(golden)
    missing: list[str] = []
    if req["index"] and not runtime.documents_available:
        missing.append("indice RAG")
    if req["olap"] and not runtime.spreadsheets_available:
        missing.append("planilhas OLAP")
    if req["ml"] and not runtime.ml_available:
        missing.append("modelo ML")
    if missing:
        return False, ", ".join(missing)
    return True, None


def filter_runnable_goldens(goldens: Iterable[Any], runtime: EvalRuntime) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Separa goldens executáveis dos bloqueados por falta de infra."""
    ok: list[Any] = []
    skipped: list[tuple[Any, str]] = []
    for golden in goldens:
        runnable, reason = golden_runnable(golden, runtime)
        if runnable:
            ok.append(golden)
        elif reason:
            skipped.append((golden, reason))
    return ok, skipped


def extract_tools_called(crew_result: CrewRunResult) -> list[Any]:
    """Constrói lista de ToolCall DeepEval a partir do CrewRunResult."""
    from deepeval.test_case import ToolCall

    tools: list[ToolCall] = []

    triage = crew_result.triage
    reasoning = getattr(triage, "reason", None) if triage else None

    for name, result in crew_result.tool_results.items():
        payload = result.payload or {}
        output: Any
        if name == "rag":
            hits = payload.get("hits") or []
            output = [
                (h.get("cited") or h.get("text") or "")[:200]
                for h in hits
                if isinstance(h, dict)
            ]
        elif name == "olap":
            sql = payload.get("sql")
            df = payload.get("dataframe")
            rows = int(len(df)) if df is not None else 0
            output = {"sql": sql, "rows": rows}
        elif name == "ml":
            preds = payload.get("predictions")
            output = preds[:3] if isinstance(preds, list) else preds
        else:
            output = str(payload)[:200] if payload else None

        tools.append(
            ToolCall(
                name=name,
                reasoning=reasoning,
                output=output if result.ok else f"erro: {result.error}",
            )
        )

    return tools


def _extract_retrieval_context(crew_result: CrewRunResult) -> list[str]:
    rag = crew_result.tool_results.get("rag")
    if rag is None or not rag.ok:
        return []
    chunks: list[str] = []
    for hit in rag.payload.get("hits") or []:
        if not isinstance(hit, dict):
            continue
        body = (hit.get("cited") or hit.get("text") or "").strip()
        if body:
            chunks.append(body)
    return chunks


def _completion_text(completion: Any) -> str:
    """Extrai texto da resposta OpenAI-compatível com mensagem clara se vier vazia."""
    choices = getattr(completion, "choices", None)
    if not choices:
        raise RuntimeError(
            "Resposta LLM vazia ou malformada (choices ausente). "
            "Comum em rate limit do OpenRouter free — aumente LLM_MIN_REQUEST_INTERVAL_S "
            "ou use --request-interval."
        )
    message = choices[0].message
    return (getattr(message, "content", None) or "").strip()


def resolve_retrieval_context(turn: TurnResult, golden: Any) -> list[str] | None:
    """
    Contexto para metricas RAG (Faithfulness).

    Prioridade: chunks recuperados pelo RAG -> ``context`` anotado no golden.
    """
    if turn.retrieval_context:
        return turn.retrieval_context
    golden_ctx = getattr(golden, "context", None)
    if golden_ctx:
        return [str(c) for c in golden_ctx if str(c).strip()]
    return None


def run_assistente_turn(
    user_message: str,
    *,
    runtime: EvalRuntime,
    project_ids: set[str] | None = None,
    history: list[dict] | None = None,
) -> TurnResult:
    """
    Executa uma pergunta end-to-end e devolve a resposta final do assistente.

    Inclui saudações (Greeter) e respostas completas do Synthesizer (sem stream).
    """
    history = history or []
    pre_max_tokens, pre_hist_turns = effective_chat_limits(
        run_ml=False,
        max_tokens=runtime.max_tokens,
        max_history_turns=runtime.max_history_turns,
    )
    pre_hist_chars = chat_history_chars_per_message(ml_route=False)

    crew_result = run_crew_chat(
        user_message,
        history=history,
        client=runtime.client,
        model=runtime.model,
        rag_backend=runtime.rag_backend,
        rag_top_k=runtime.rag_top_k,
        rag_project_ids=project_ids,
        rag_reranker=runtime.rag_reranker,
        documents_available=runtime.documents_available,
        spreadsheets_available=runtime.spreadsheets_available,
        ml_available=runtime.ml_available,
        ml_bundle=runtime.ml_bundle,
        ml_model_path=runtime.ml_model_path,
        max_history_turns=int(pre_hist_turns),
        max_chars_per_message=int(pre_hist_chars),
    )

    if crew_result.greeting_response is not None:
        return TurnResult(
            actual_output=crew_result.greeting_response,
            retrieval_context=[],
            crew_result=crew_result,
        )

    if crew_result.synth is None:
        raise RuntimeError("Crew não preparou o prompt do Synthesizer.")

    used_ml = crew_result.synth.used_ml
    reply_max_tokens, history_turns = effective_chat_limits(
        run_ml=used_ml,
        max_tokens=runtime.max_tokens,
        max_history_turns=runtime.max_history_turns,
    )
    history_chars = chat_history_chars_per_message(ml_route=used_ml)

    if used_ml and (
        history_turns != pre_hist_turns or history_chars != pre_hist_chars
    ):
        from agents.synthesizer import build_messages as rebuild_messages

        history_with_user = history + [{"role": "user", "content": user_message}]
        crew_result.synth = rebuild_messages(
            user_message=user_message,
            history=history_with_user,
            tool_results=crew_result.tool_results,
            model_id=runtime.model,
            max_history_turns=int(history_turns),
            max_chars_per_message=int(history_chars),
        )

    chat_profile = select_chat_profile(model_id=runtime.model, use_thinking=False)
    completion = create_chat_completion(
        runtime.client,
        messages=crew_result.synth.messages,
        model=runtime.model,
        profile=chat_profile,
        max_tokens=int(reply_max_tokens),
        stream=False,
        generation_name="eval-synthesizer",
    )
    raw = _completion_text(completion)
    actual_output = strip_thinking_blocks(raw)

    return TurnResult(
        actual_output=actual_output,
        retrieval_context=_extract_retrieval_context(crew_result),
        crew_result=crew_result,
    )


def golden_project_ids(golden: Any) -> set[str] | None:
    """Extrai ``project_ids`` de um ``Golden`` DeepEval ou dict exportado."""
    meta = getattr(golden, "additional_metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    raw = meta.get("project_ids")
    if not raw:
        return None
    return {str(x) for x in raw}


def golden_category(golden: Any) -> str | None:
    meta = getattr(golden, "additional_metadata", None) or {}
    if isinstance(meta, dict):
        return meta.get("category")
    return None


def filter_goldens(
    goldens: Iterable[Any],
    *,
    category: str | None = None,
    limit: int | None = None,
) -> list[Any]:
    items = list(goldens)
    if category:
        cat = category.strip().lower()
        items = [g for g in items if (golden_category(g) or "").lower() == cat]
    if limit is not None and limit > 0:
        items = items[:limit]
    return items
