"""
Tools determinísticas do Crew — wrappers finos sobre RAG, OLAP e ML.

Cada Tool é uma função Python pura que:
- recebe argumentos tipados,
- chama o subsistema já validado (``rag_semantic_search``, ``run_nl_olap_query``,
  ``run_chat_ml_inference``),
- devolve um ``ToolResult`` uniforme com:
    * ``ok``: bool
    * ``context_for_llm``: texto pronto para injetar no system prompt do
      Synthesizer,
    * ``payload``: dados estruturados (DataFrames, hits, predictions),
    * ``error``: mensagem amigável quando algo falhou.

Manter o LLM call **dentro** das Tools (em vez de em agentes separados) evita
duplicar chamadas e preserva os perfis Qwen3.5 já testados (``PROFILE_OLAP_SQL``,
``PROFILE_CHAT_ROUTER``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from ml.chat_infer import MlInferResult, run_chat_ml_inference
from ml.paths import chat_ml_model_available, chat_ml_model_path
from ml.training import ModelBundle
from olap import OlapQueryResult, has_ingested_tables, run_nl_olap_query
from rag import (
    default_retrieve_k,
    format_context_for_llm,
    index_ready,
    rerank_hits,
    search_with_backend,
)


@dataclass
class ToolResult:
    """Saída uniforme de uma Tool, consumida pelo Dispatcher e pelo Synthesizer."""

    name: str
    ok: bool
    context_for_llm: str = ""
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "ok": self.ok,
            "context_for_llm": self.context_for_llm,
            "error": self.error,
            "summary": self.summary,
        }
        return out


# ── RAG ─────────────────────────────────────────────────────────────────────


def rag_search_tool(
    query: str,
    *,
    backend: object | None,
    top_k: int = 6,
    project_ids: set[str] | None = None,
    reranker: object | None = None,
    rerank_enabled: bool = True,
    rerank_retrieve_k: int | None = None,
) -> ToolResult:
    """
    Busca semântica no índice txtai e formata contexto citável.

    Parameters
    ----------
    query:
        Texto da pergunta do usuário (será embeddado).
    backend:
        Instância ``Embeddings`` já carregada — vinda do cache do Streamlit
        (``_txtai_backend_cached``). Quando ``None``, retorna ``ok=False``.
    top_k:
        Quantos chunks entregar ao LLM após busca (e rerank, se ativo).
    project_ids:
        Restringe a busca a um subconjunto de ``project_id``.
    reranker:
        Instância ``CrossEncoder`` em cache. Quando ``None`` ou
        ``rerank_enabled=False``, usa só a ordem da busca vetorial.
    rerank_enabled:
        Liga/desliga o rerank sem descartar o modelo em cache.
    rerank_retrieve_k:
        Candidatos antes do rerank. ``None`` ou ``0`` usa heurística
        ``max(top_k * 4, 20)``.
    """
    if not index_ready():
        return ToolResult(
            name="rag",
            ok=False,
            error="Índice RAG ainda não foi construído. Use a aba **Documentos**.",
            summary="índice ausente",
        )
    if backend is None:
        return ToolResult(
            name="rag",
            ok=False,
            error="Backend txtai indisponível (cache vazio).",
            summary="backend indisponível",
        )

    q = (query or "").strip()
    if not q:
        return ToolResult(name="rag", ok=False, error="Consulta vazia.", summary="vazio")

    tk = int(top_k)
    use_rerank = bool(rerank_enabled and reranker is not None)
    retrieve_k = default_retrieve_k(tk, rerank_retrieve_k) if use_rerank else tk

    hits: list[dict] = search_with_backend(
        backend,
        q,
        tk,
        project_ids=project_ids,
        retrieve_limit=retrieve_k if use_rerank else None,
    )
    if use_rerank and hits:
        hits = rerank_hits(q, hits, reranker=reranker, top_k=tk)

    if not hits:
        return ToolResult(
            name="rag",
            ok=True,
            context_for_llm="",
            payload={"hits": []},
            summary="0 evidências",
        )

    ctx = format_context_for_llm(hits)
    instructions = (
        "\n\nBaseie respostas sobre ensaios neste contexto quando for relevante; "
        "cite projeto e arquivo como nos cabeçalhos [n]. Se o contexto não ajudar, "
        "diga claramente."
    )
    return ToolResult(
        name="rag",
        ok=True,
        context_for_llm=(
            "### Contexto recuperado dos documentos do laboratório\n" + ctx + instructions
        ),
        payload={
            "hits": hits,
            "top_k": tk,
            "rerank_enabled": use_rerank,
            "retrieve_k": retrieve_k if use_rerank else tk,
        },
        summary=f"{len(hits)} evidência(s)" + (" · rerank" if use_rerank else ""),
    )


# ── OLAP (DuckDB) ───────────────────────────────────────────────────────────


def duckdb_query_tool(
    question: str,
    *,
    client: OpenAI,
    model: str,
) -> ToolResult:
    """
    Executa a pergunta do usuário em SQL DuckDB read-only.

    Internamente chama ``olap.run_nl_olap_query`` que:
    1. Pede ao OpenRouter um SELECT (perfil ``PROFILE_OLAP_SQL``).
    2. Valida que o SQL é só leitura (`validate_readonly_sql`).
    3. Executa no DuckDB e devolve o DataFrame + texto formatado.
    """
    if not has_ingested_tables():
        return ToolResult(
            name="olap",
            ok=False,
            error="Nenhuma planilha ingerida no DuckDB. Escaneie as pastas primeiro.",
            summary="sem tabelas",
        )

    result: OlapQueryResult = run_nl_olap_query(
        question,
        client=client,
        model=model,
    )

    if not result.ok:
        return ToolResult(
            name="olap",
            ok=False,
            error=result.error,
            payload={
                "sql": result.sql,
                "raw_llm_response": result.raw_llm_response,
            },
            summary=f"erro: {result.error or 'desconhecido'}"[:120],
        )

    rows = 0
    if result.dataframe is not None:
        rows = int(len(result.dataframe))
    return ToolResult(
        name="olap",
        ok=True,
        context_for_llm=(
            result.context_for_llm
            + "\n\nAo responder com dados tabulares acima, cite projeto (_project_id) "
            "e arquivo (_source_file). Não invente valores fora do resultado SQL."
        ),
        payload={
            "sql": result.sql,
            "dataframe": result.dataframe,
            "raw_llm_response": result.raw_llm_response,
        },
        summary=f"{rows} linha(s) · SQL gerado",
    )


# ── ML predição ─────────────────────────────────────────────────────────────


def ml_predict_tool(
    message: str,
    *,
    client: OpenAI,
    model: str,
    history: list[dict] | None = None,
    bundle: ModelBundle | None = None,
    model_path: Path | None = None,
) -> ToolResult:
    """
    Roda inferência ML quando o usuário pede predição explicitamente.

    Não tenta predizer "às escuras" se o modelo não estiver disponível: retorna
    ``ok=False`` com mensagem clara — o Synthesizer explica ao usuário.
    """
    if not chat_ml_model_available():
        path = chat_ml_model_path()
        return ToolResult(
            name="ml",
            ok=False,
            error=f"Modelo ML não encontrado em `{path}`. Treine ou copie o .pkl.",
            summary="modelo ausente",
        )

    result: MlInferResult = run_chat_ml_inference(
        message,
        history=history or [],
        client=client,
        model=model,
        bundle=bundle,
        model_path=model_path,
    )

    if not result.ok:
        return ToolResult(
            name="ml",
            ok=False,
            error=result.error,
            context_for_llm=result.context_for_llm,
            payload={
                "raw_llm_response": result.raw_llm_response,
            },
            summary=f"erro: {result.error or 'desconhecido'}"[:120],
        )

    rows = 0
    if result.predictions is not None:
        rows = int(len(result.predictions))
    return ToolResult(
        name="ml",
        ok=True,
        context_for_llm=result.context_for_llm,
        payload={
            "predictions": result.predictions,
            "raw_llm_response": result.raw_llm_response,
            "model_path": result.model_path,
        },
        summary=f"{rows} predição(ões)",
    )
