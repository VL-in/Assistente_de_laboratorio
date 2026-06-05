"""
Template para construir o golden dataset do Assistente de Lab (DeepEval — end-to-end).

Este arquivo **não exige** ``deepeval`` instalado. Use-o para curar casos de teste
com ``input`` (pergunta do usuário) e ``expected_output`` (resposta ideal), além de
metadados que descrevem qual rota do crew deve ser acionada.

Fluxo recomendado
-----------------
1. Edite ``build_golden_dataset()`` — substitua os exemplos placeholder pelos seus
   casos reais, com base nos documentos/planilhas/modelo ML do seu ambiente.
2. Execute este módulo para exportar JSON/JSONL:

       python apps/streamlit/evals/golden_dataset_template.py

3. Depois de ``pip install -U deepeval``, carregue o dataset e rode evals end-to-end
   (ver seção ``PRÓXIMOS PASSOS`` no final do arquivo).

Referência DeepEval: https://deepeval.com/docs/evaluation-datasets
Modo end-to-end: https://deepeval.com/docs/evaluation-end-to-end-llm-evals
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal

# Diretório padrão de saída (relativo a este arquivo).
EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = EVALS_DIR / "datasets"


class EvalCategory(str, Enum):
    """Tipo de interação esperada no chat — ajuda a organizar cobertura de testes."""

    GREETING = "greeting"  # Greeter rule-based (zero LLM)
    RAG = "rag"  # Documentos (txtai + rerank)
    OLAP = "olap"  # Planilhas / DuckDB NL→SQL
    ML = "ml"  # Predição AbRank (sequências Ab/Ag)
    COMBINED = "combined"  # Mais de uma tool (ex.: docs + planilhas)
    OUT_OF_SCOPE = "out_of_scope"  # Pergunta fora da base — resposta honesta


@dataclass
class ExpectedRoutes:
    """Quais especialistas o Triage deveria acionar (referência para anotação humana)."""

    documents: bool = False
    spreadsheets: bool = False
    ml_prediction: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "documents": self.documents,
            "spreadsheets": self.spreadsheets,
            "ml_prediction": self.ml_prediction,
        }


@dataclass
class ChatGolden:
    """
    Espelha o modelo ``Golden`` do DeepEval (single-turn, end-to-end).

    Campos principais
    -----------------
    input:
        Mensagem do usuário (como na aba Conversa).
    expected_output:
        Resposta ideal que o assistente deveria produzir. Seja específico o
        suficiente para métricas como GEval ou AnswerRelevancy, mas evite
        frases literais demais se o LLM pode variar a redação.
    context:
        Trechos de referência dos seus documentos (ground truth para RAG).
        Opcional para saudações e perguntas puramente tabulares.
    additional_metadata:
        Metadados do projeto — categoria, rotas esperadas, pré-requisitos de
        índice/planilha/ML, IDs de projeto, etc.

    Campos que **não** preencha aqui (são gerados na hora da avaliação):
    actual_output, retrieval_context, tools_called
    """

    input: str
    expected_output: str | None = None
    context: list[str] | None = None
    category: EvalCategory = EvalCategory.RAG
    expected_routes: ExpectedRoutes | None = None
    project_ids: list[str] | None = None
    requires_index: bool = True
    requires_olap: bool = False
    requires_ml_model: bool = False
    comments: str | None = None
    tags: list[str] = field(default_factory=list)
    golden_id: str | None = None

    def to_deepeval_dict(self) -> dict[str, Any]:
        """Formato compatível com ``EvaluationDataset.add_goldens_from_json_file``."""
        additional_metadata: dict[str, Any] = {
            "category": self.category.value,
            "requires_index": self.requires_index,
            "requires_olap": self.requires_olap,
            "requires_ml_model": self.requires_ml_model,
            "tags": self.tags,
        }
        if self.golden_id:
            additional_metadata["golden_id"] = self.golden_id
        if self.expected_routes is not None:
            additional_metadata["expected_routes"] = self.expected_routes.to_dict()
        if self.project_ids:
            additional_metadata["project_ids"] = self.project_ids

        row: dict[str, Any] = {"input": self.input}
        if self.expected_output is not None:
            row["expected_output"] = self.expected_output
        if self.context:
            row["context"] = self.context
        if self.comments:
            row["comments"] = self.comments
        row["additional_metadata"] = additional_metadata
        return row


def _validate_golden(golden: ChatGolden, index: int) -> list[str]:
    """Validações leves antes de exportar — retorna lista de avisos/erros."""
    issues: list[str] = []
    prefix = f"golden[{index}]"

    if not golden.input.strip():
        issues.append(f"{prefix}: ``input`` vazio.")

    if golden.category == EvalCategory.GREETING:
        if golden.expected_routes and any(
            (
                golden.expected_routes.documents,
                golden.expected_routes.spreadsheets,
                golden.expected_routes.ml_prediction,
            )
        ):
            issues.append(f"{prefix}: saudação não deve acionar tools.")
        if golden.requires_index or golden.requires_olap or golden.requires_ml_model:
            issues.append(f"{prefix}: saudação não exige índice/OLAP/ML.")

    if golden.category == EvalCategory.ML:
        if golden.expected_routes and not golden.expected_routes.ml_prediction:
            issues.append(f"{prefix}: categoria ML deveria ter ml_prediction=true.")
        if not golden.requires_ml_model:
            issues.append(f"{prefix}: categoria ML deveria ter requires_ml_model=true.")

    if golden.category in (EvalCategory.RAG, EvalCategory.COMBINED):
        if golden.expected_output and not golden.context and golden.requires_index:
            issues.append(
                f"{prefix}: considere adicionar ``context`` (trechos dos docs) "
                "para avaliar qualidade RAG end-to-end."
            )

    return issues


def validate_dataset(goldens: Iterable[ChatGolden]) -> list[str]:
    """Valida todos os goldens; levanta ``ValueError`` se houver erros bloqueantes."""
    all_issues: list[str] = []
    for i, g in enumerate(goldens):
        all_issues.extend(_validate_golden(g, i))

    blocking = [m for m in all_issues if "vazio" in m]
    if blocking:
        raise ValueError("Dataset inválido:\n- " + "\n- ".join(blocking))
    return all_issues


def export_json(goldens: list[ChatGolden], path: Path) -> Path:
    """Exporta lista de goldens para JSON (array de objetos)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [g.to_deepeval_dict() for g in goldens]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_jsonl(goldens: list[ChatGolden], path: Path) -> Path:
    """Exporta um golden por linha (JSONL) — formato preferido para datasets grandes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(g.to_deepeval_dict(), ensure_ascii=False) for g in goldens]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_json(path: Path) -> list[ChatGolden]:
    """Carrega goldens exportados (útil para editar e re-exportar)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [_golden_from_dict(row) for row in raw]


def _golden_from_dict(row: dict[str, Any]) -> ChatGolden:
    meta = row.get("additional_metadata") or {}
    routes_raw = meta.get("expected_routes")
    expected_routes = None
    if isinstance(routes_raw, dict):
        expected_routes = ExpectedRoutes(
            documents=bool(routes_raw.get("documents")),
            spreadsheets=bool(routes_raw.get("spreadsheets")),
            ml_prediction=bool(routes_raw.get("ml_prediction")),
        )
    category_raw = meta.get("category", EvalCategory.RAG.value)
    return ChatGolden(
        input=str(row["input"]),
        expected_output=row.get("expected_output"),
        context=row.get("context"),
        category=EvalCategory(category_raw),
        expected_routes=expected_routes,
        project_ids=meta.get("project_ids"),
        requires_index=bool(meta.get("requires_index", True)),
        requires_olap=bool(meta.get("requires_olap", False)),
        requires_ml_model=bool(meta.get("requires_ml_model", False)),
        comments=row.get("comments"),
        tags=list(meta.get("tags") or []),
        golden_id=meta.get("golden_id"),
    )




def build_golden_dataset() -> list[ChatGolden]:
    """
    Retorna a lista de goldens do projeto.

    Por padrão carrega os 40 casos curados dos projetos 252/253
    (``goldens_projetos_252_253.build_projetos_goldens``).
    """
    from goldens_projetos_252_253 import build_projetos_goldens

    return build_projetos_goldens()


def export_dataset(
    goldens: list[ChatGolden] | None = None,
    *,
    export_dir: Path = DEFAULT_EXPORT_DIR,
    formats: tuple[Literal["json", "jsonl"], ...] = ("json", "jsonl"),
) -> dict[str, Path]:
    """Valida, exporta e retorna caminhos dos arquivos gerados."""
    items = list(goldens if goldens is not None else build_golden_dataset())
    warnings = validate_dataset(items)
    for w in warnings:
        print(f"[aviso] {w}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}
    if "json" in formats:
        paths["json"] = export_json(items, export_dir / f"assistente_lab_goldens_{stamp}.json")
    if "jsonl" in formats:
        paths["jsonl"] = export_jsonl(items, export_dir / f"assistente_lab_goldens_{stamp}.jsonl")
    return paths


# ---------------------------------------------------------------------------
# PRÓXIMOS PASSOS (após pip install -U deepeval)
# ---------------------------------------------------------------------------
#
# 1. Exportar goldens (se ainda não fez):
#
#        python apps/streamlit/evals/golden_dataset_template.py
#
# 2. Instalar deps de eval (venv separado — deepeval conflita com crewai):
#
#        python -m venv .venv-evals
#        .venv-evals\Scripts\activate
#        pip install -r apps/streamlit/requirements-evals.txt
#
# 3. Rodar no Docker (recomendado — usa volumes /data/txtai, /data/duckdb, /data/ml):
#
#        docker compose build streamlit
#        docker compose exec streamlit python evals/run_assistente_eval.py --require-ready --limit 3
#
#    Ou: .\scripts\run_evals_docker.ps1 --limit 5
#
# 4. Rodar avaliacao local (venv evals, sem crewai):
#
#        python apps/streamlit/evals/run_assistente_eval.py --limit 5
#
# 5. Pre-requisitos de ambiente:
#    - OPENROUTER_API_KEY — assistente e juiz LLM-as-judge (padrao)
#    - OPENAI_API_KEY — so se usar --judge-provider openai
#    - Indice RAG, planilhas OLAP e modelo ML conforme cada golden
#
#    Opcional: EVAL_JUDGE_MODEL=openrouter/auto (modelo só para métricas)
#
#    Alternativa via CLI DeepEval:
#        deepeval set-openrouter -m openrouter/auto --save=dotenv
#
# Documentação: https://deepeval.com/docs/getting-started
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    out = export_dataset()
    print(f"Exportados {len(build_golden_dataset())} goldens:")
    for fmt, p in out.items():
        print(f"  {fmt}: {p}")
