"""
Runner de alto nível do Crew — ponto único chamado pelo ``app.py``.

Responsabilidades:
1. Curto-circuito do Greeter (saudação rule-based) — zero tokens.
2. Triagem (LLM curto JSON).
3. Despacho paralelo de Tools (RAG, OLAP, ML).
4. Construção do prompt do Synthesizer.
5. Devolve um ``CrewRunResult`` com tudo que o Streamlit precisa para gerar
   o stream da resposta final e exibir a trilha.

Não chama o Synthesizer aqui — quem dispara ``create_chat_completion(stream=True)``
é o ``app.py`` (precisa do ``st.write_stream``). Este módulo só prepara o prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from langfuse import observe
from openai import OpenAI

from agents.crew import CrewContext, dispatch_specialists, run_triage
from observability.langfuse_client import langfuse_enabled, langfuse_span
from agents.greeter import handle_greeting
from agents.handoff import HandoffTrace
from agents.synthesizer import SynthesizerInput, build_messages
from agents.tools import ToolResult
from agents.triage import TriageDecision
from ml.training import ModelBundle


def trace_handoff_enabled() -> bool:
    """``CREW_TRACE_HANDOFF=0|false|no|off`` desativa a coleta de trilha."""
    return os.environ.get("CREW_TRACE_HANDOFF", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def parallel_tools_enabled() -> bool:
    """``CREW_PARALLEL_TOOLS=0|false|no|off`` força execução serial das Tools."""
    return os.environ.get("CREW_PARALLEL_TOOLS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@dataclass
class CrewRunResult:
    """O que o ``app.py`` recebe após orquestrar o Crew."""

    greeting_response: str | None = None
    triage: TriageDecision | None = None
    tool_results: dict[str, ToolResult] = field(default_factory=dict)
    synth: SynthesizerInput | None = None
    trace: HandoffTrace = field(default_factory=HandoffTrace)

    @property
    def used_tools(self) -> list[str]:
        return [name for name, r in self.tool_results.items() if r.ok]

    @property
    def ml_predictions(self):
        ml = self.tool_results.get("ml")
        if ml is None:
            return None
        return ml.payload.get("predictions")

    @property
    def olap_dataframe(self):
        olap = self.tool_results.get("olap")
        if olap is None:
            return None
        return olap.payload.get("dataframe")

    @property
    def olap_sql(self) -> str | None:
        olap = self.tool_results.get("olap")
        if olap is None:
            return None
        return olap.payload.get("sql")


@observe(name="crew-pipeline", capture_input=False, capture_output=False)
def run_crew_chat(
    user_message: str,
    *,
    history: list[dict],
    client: OpenAI,
    model: str,
    rag_backend: object | None,
    rag_top_k: int = 6,
    rag_project_ids: set[str] | None = None,
    rag_reranker: object | None = None,
    rag_rerank_retrieve_k: int | None = None,
    force_routes: bool = False,
    force_use_rag: bool = False,
    force_use_olap: bool = False,
    force_use_ml: bool = False,
    documents_available: bool = False,
    spreadsheets_available: bool = False,
    ml_available: bool = False,
    ml_bundle: ModelBundle | None = None,
    ml_model_path: Path | None = None,
    parallel_tools: bool | None = None,
    max_history_turns: int | None = None,
    max_chars_per_message: int | None = None,
) -> CrewRunResult:
    """
    Executa o pipeline multiagente até a fronteira do Synthesizer (sem stream).

    O ``app.py`` deve checar ``result.greeting_response`` antes de qualquer
    outra coisa: se for não-``None``, ignora ``synth`` e exibe a resposta curta
    direto. Senão, usa ``result.synth.messages`` (já truncado conforme
    ``max_history_turns``/``max_chars_per_message``) para chamar
    ``create_chat_completion(stream=True)`` e ``iter_stream_answer_text``.
    """
    if langfuse_enabled():
        from langfuse import get_client

        get_client().update_current_span(
            input={"user_message": user_message.strip()[:2000]},
            metadata={"model": model},
        )

    trace = HandoffTrace()

    with trace.start("Greeter", input_summary=user_message) as h, langfuse_span(
        "Greeter",
        input_data={"message": user_message[:500]},
    ):
        greeting = handle_greeting(user_message)
        if greeting is not None:
            h.set_output(greeting)
            h.set_metadata(short_circuit=True)
            return CrewRunResult(greeting_response=greeting, trace=trace)
        h.set_output("não-saudação · prossegue para Triage")
        h.set_metadata(short_circuit=False)

    use_parallel = parallel_tools_enabled() if parallel_tools is None else parallel_tools

    ctx = CrewContext(
        user_message=user_message,
        history=history or [],
        client=client,
        model=model,
        rag_backend=rag_backend,
        rag_top_k=int(rag_top_k),
        rag_project_ids=rag_project_ids,
        rag_reranker=rag_reranker,
        rag_rerank_retrieve_k=rag_rerank_retrieve_k,
        ml_bundle=ml_bundle,
        ml_model_path=ml_model_path,
        documents_available=bool(documents_available),
        spreadsheets_available=bool(spreadsheets_available),
        ml_available=bool(ml_available),
        parallel_tools=bool(use_parallel),
    )

    triage = run_triage(ctx, trace=trace)
    if force_routes:
        triage.decision = TriageDecision(
            use_rag=bool(force_use_rag and documents_available),
            use_olap=bool(force_use_olap and spreadsheets_available),
            use_ml=bool(force_use_ml and ml_available),
            source="dev_force",
            reason="configuração manual do chat (dev)",
        )
    tool_results = dispatch_specialists(ctx, triage, trace=trace)

    history_with_user = list(history or []) + [{"role": "user", "content": user_message}]
    with trace.start("Synthesizer (build prompt)", input_summary=user_message) as h, langfuse_span(
        "Synthesizer (build prompt)",
        input_data={"user_message": user_message[:500]},
    ):
        synth = build_messages(
            user_message=user_message,
            history=history_with_user,
            tool_results=tool_results,
            model_id=model,
            triage_decision=triage.decision,
            max_history_turns=max_history_turns,
            max_chars_per_message=max_chars_per_message,
        )
        h.set_output(
            f"system={len(synth.system_prompt)} chars · "
            f"messages={len(synth.messages)}"
        )
        h.set_metadata(
            used_ml=synth.used_ml,
            tools_ok=[name for name, r in tool_results.items() if r.ok],
            tools_failed=[name for name, r in tool_results.items() if not r.ok],
        )

    return CrewRunResult(
        greeting_response=None,
        triage=triage.decision,
        tool_results=tool_results,
        synth=synth,
        trace=trace,
    )
