"""
Orquestração do Crew — Triage → Dispatcher → Tools → Synthesizer.

Este módulo é o "maestro" do sistema multiagentes. Não usa o
``crewai.Crew`` literal porque:

1. As Tools (RAG/OLAP/ML) são determinísticas e já fazem suas próprias
   chamadas LLM internas com perfis Qwen3.5 que não queremos perder.
2. O Streamlit pede streaming nativo no Synthesizer
   (``st.write_stream + iter_stream_answer_text``); o ``Crew.kickoff()`` do
   CrewAI ainda mistura streaming de várias tasks de forma incompatível.
3. Queremos paralelismo controlado entre Tools (``ThreadPoolExecutor``).

Para manter compatibilidade com o ecossistema CrewAI, expomos os mesmos
papéis (Triage Agent, Synthesizer Agent) e respeitamos a estrutura de Tasks.
A ``crewai.LLM`` em ``agents.llm`` está pronta para o dia em que precisarmos
adicionar agentes que **realmente** façam tool-calling autônomo (ex.: novo
agente "Auditor" da Fase 4).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from agents.handoff import HandoffTrace
from agents.tools import (
    ToolResult,
    duckdb_query_tool,
    ml_predict_tool,
    rag_search_tool,
)
from agents.triage import TriageDecision, classify_intent
from ml.training import ModelBundle


@dataclass
class TriageOutput:
    """Resultado do agente de triagem com trace anexado."""

    decision: TriageDecision

    @property
    def use_rag(self) -> bool:
        return self.decision.use_rag

    @property
    def use_olap(self) -> bool:
        return self.decision.use_olap

    @property
    def use_ml(self) -> bool:
        return self.decision.use_ml


@dataclass
class CrewContext:
    """Tudo que as etapas do Crew precisam para rodar — passado por valor."""

    user_message: str
    history: list[dict] = field(default_factory=list)
    client: OpenAI | None = None
    model: str = ""
    rag_backend: object | None = None
    rag_top_k: int = 6
    rag_project_ids: set[str] | None = None
    ml_bundle: ModelBundle | None = None
    ml_model_path: Path | None = None
    documents_available: bool = False
    spreadsheets_available: bool = False
    ml_available: bool = False
    parallel_tools: bool = True


def run_triage(ctx: CrewContext, *, trace: HandoffTrace) -> TriageOutput:
    """Roda o Triage Agent e registra o handoff."""
    with trace.start("Triage", input_summary=ctx.user_message) as h:
        decision = classify_intent(
            ctx.user_message,
            history=ctx.history,
            client=ctx.client,
            model=ctx.model,
            documents_available=ctx.documents_available,
            spreadsheets_available=ctx.spreadsheets_available,
            ml_available=ctx.ml_available,
        )
        h.set_output(
            f"rag={decision.use_rag} · olap={decision.use_olap} · "
            f"ml={decision.use_ml} ({decision.source})"
        )
        h.set_metadata(**decision.to_dict())
    return TriageOutput(decision=decision)


def dispatch_specialists(
    ctx: CrewContext,
    triage: TriageOutput,
    *,
    trace: HandoffTrace,
) -> dict[str, ToolResult]:
    """
    Executa as Tools selecionadas pelo Triage.

    Quando o Triage seleciona ML, RAG/OLAP são forçadas a ``False`` (predição é
    auto-contida) — mesma regra do ``ML_HINT`` em ``agents.intent_rules``.

    Quando ``ctx.parallel_tools`` é true e há ≥2 tools selecionadas, executa em
    paralelo via ``ThreadPoolExecutor``. Caso contrário, executa em série
    (debug mais previsível).
    """
    decision = triage.decision

    if decision.use_ml:
        run_rag = False
        run_olap = False
    else:
        run_rag = decision.use_rag
        run_olap = decision.use_olap
    run_ml = decision.use_ml

    if not (run_rag or run_olap or run_ml):
        return {}

    jobs: list[tuple[str, callable]] = []
    if run_rag:
        jobs.append(("rag", lambda: _run_rag(ctx, trace)))
    if run_olap:
        jobs.append(("olap", lambda: _run_olap(ctx, trace)))
    if run_ml:
        jobs.append(("ml", lambda: _run_ml(ctx, trace)))

    results: dict[str, ToolResult] = {}
    if ctx.parallel_tools and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = {ex.submit(fn): name for name, fn in jobs}
            for fut in futures:
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[name] = ToolResult(
                        name=name,
                        ok=False,
                        error=f"Tool {name} levantou {type(exc).__name__}: {exc}",
                        summary=f"crash: {type(exc).__name__}",
                    )
    else:
        for name, fn in jobs:
            try:
                results[name] = fn()
            except Exception as exc:  # noqa: BLE001
                results[name] = ToolResult(
                    name=name,
                    ok=False,
                    error=f"Tool {name} levantou {type(exc).__name__}: {exc}",
                    summary=f"crash: {type(exc).__name__}",
                )
    return results


def _run_rag(ctx: CrewContext, trace: HandoffTrace) -> ToolResult:
    with trace.start("RAG Tool", input_summary=ctx.user_message) as h:
        result = rag_search_tool(
            ctx.user_message,
            backend=ctx.rag_backend,
            top_k=ctx.rag_top_k,
            project_ids=ctx.rag_project_ids,
        )
        h.set_output(result.summary or ("ok" if result.ok else (result.error or "")))
        hits = result.payload.get("hits") or []
        evidence_files = [
            {
                "n": i,
                "project_id": h_.get("project_id"),
                "arquivo": h_.get("relative_path"),
                "chunk": h_.get("chunk_index"),
                "score": h_.get("score"),
            }
            for i, h_ in enumerate(hits, start=1)
        ]
        h.set_metadata(
            ok=result.ok,
            top_k=ctx.rag_top_k,
            evidence_count=len(hits),
            project_ids=sorted(ctx.rag_project_ids) if ctx.rag_project_ids else None,
            evidence_files=evidence_files,
        )
        if not result.ok:
            h.mark_failed(result.error or "rag_tool falhou")
    return result


def _run_olap(ctx: CrewContext, trace: HandoffTrace) -> ToolResult:
    with trace.start("OLAP Tool", input_summary=ctx.user_message) as h:
        if ctx.client is None or not ctx.model:
            result = ToolResult(
                name="olap",
                ok=False,
                error="Cliente OpenAI ausente para gerar SQL.",
                summary="cliente ausente",
            )
        else:
            result = duckdb_query_tool(
                ctx.user_message,
                client=ctx.client,
                model=ctx.model,
            )
        h.set_output(result.summary or ("ok" if result.ok else (result.error or "")))
        df = result.payload.get("dataframe")
        rows = int(len(df)) if df is not None else 0
        h.set_metadata(
            ok=result.ok,
            sql=result.payload.get("sql"),
            rows=rows,
        )
        if not result.ok:
            h.mark_failed(result.error or "duckdb_tool falhou")
    return result


def _run_ml(ctx: CrewContext, trace: HandoffTrace) -> ToolResult:
    with trace.start("ML Tool", input_summary=ctx.user_message) as h:
        if ctx.client is None or not ctx.model:
            result = ToolResult(
                name="ml",
                ok=False,
                error="Cliente OpenAI ausente para extrair features.",
                summary="cliente ausente",
            )
        else:
            result = ml_predict_tool(
                ctx.user_message,
                client=ctx.client,
                model=ctx.model,
                history=ctx.history,
                bundle=ctx.ml_bundle,
                model_path=ctx.ml_model_path,
            )
        h.set_output(result.summary or ("ok" if result.ok else (result.error or "")))
        df = result.payload.get("predictions")
        rows = int(len(df)) if df is not None else 0
        h.set_metadata(
            ok=result.ok,
            predictions=rows,
            model_path=str(ctx.ml_model_path) if ctx.ml_model_path else None,
        )
        if not result.ok:
            h.mark_failed(result.error or "ml_tool falhou")
    return result
