#!/usr/bin/env python
"""
Avaliação end-to-end do Assistente de Lab com DeepEval.

Uso básico (a partir da raiz do repositório)::

    # venv dedicado (sem crewai — evita conflito posthog deepeval x chromadb)
    python -m venv .venv-evals
    .venv-evals\\Scripts\\activate
    pip install -r apps/streamlit/requirements-evals.txt

    python apps/streamlit/evals/run_assistente_eval.py

Com dataset exportado e limite de casos::

    python apps/streamlit/evals/run_assistente_eval.py \\
        --dataset apps/streamlit/evals/datasets/assistente_lab_goldens_....json \\
        --limit 5 \\
        --category rag

Somente gerar respostas (sem métricas LLM-as-judge)::

    python apps/streamlit/evals/run_assistente_eval.py --skip-metrics

Via Docker (RAG/OLAP/ML nos volumes — recomendado)::

    docker compose build streamlit
    docker compose up -d streamlit

    # Fase 1: só respostas (recomendado no tier free — ~80+ chamadas LLM nos 40 casos)
    docker compose exec -e LANGFUSE_ENABLED=0 streamlit \\
        python evals/run_assistente_eval.py --require-ready --skip-metrics

    # Fase 2: métricas no JSON salvo (serial + throttle; não estoura 429 em rajada)
    docker compose exec -e LANGFUSE_ENABLED=0 streamlit \\
        python evals/run_assistente_eval.py --resume-metrics evals/results/eval_test_cases_....json

    # ou: .\\scripts\\run_evals_docker.ps1 --require-ready --skip-metrics

Via pytest / DeepEval CLI::

    deepeval test run apps/streamlit/evals/test_assistente_e2e.py

Pré-requisitos
--------------
- ``OPENROUTER_API_KEY`` (ou ``LLM_*``) para o assistente responder.
- Juiz LLM-as-judge: ``OPENROUTER_API_KEY`` (padrão) ou ``OPENAI_API_KEY``.
- Índice RAG, planilhas OLAP e modelo ML conforme os casos do dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS_DIR = EVALS_DIR / "datasets"
DEFAULT_RESULTS_DIR = EVALS_DIR / "results"

if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))
from eval_bootstrap import (
    configure_eval_env,
    eval_case_interval_s,
    eval_metrics_max_concurrent,
    eval_metrics_throttle_s,
)  # noqa: E402

configure_eval_env()


def _ensure_import_paths() -> None:
    streamlit_root = EVALS_DIR.parent
    for path in (streamlit_root, EVALS_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _latest_dataset_path() -> Path | None:
    candidates: list[Path] = []
    for pattern in ("assistente_lab_goldens_*.json", "assistente_lab_goldens_*.jsonl"):
        candidates.extend(DEFAULT_DATASETS_DIR.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _golden_meta_id(golden) -> str:
    meta = getattr(golden, "additional_metadata", None) or {}
    if isinstance(meta, dict) and meta.get("golden_id"):
        return str(meta["golden_id"])
    return "case"


def _configure_eval_rate_limits(*, case_interval_s: float | None = None) -> None:
    """Reaplica throttle (CLI pode passar --request-interval)."""
    configure_eval_env(case_interval_s=case_interval_s)


def _print_rate_limit_config() -> None:
    from qwen35_inference import (
        llm_min_request_interval_s,
        llm_retry_base_delay_s,
        llm_retry_max_attempts,
    )

    case_interval = eval_case_interval_s()
    print(
        "Limites LLM (eval): "
        f"intervalo minimo {llm_min_request_interval_s():.0f}s entre chamadas; "
        f"retry ate {llm_retry_max_attempts()}x (backoff {llm_retry_base_delay_s():.0f}s)"
        + (f"; pausa extra {case_interval:.0f}s entre casos" if case_interval > 0 else "")
    )


def _load_goldens_from_rows(rows: list[dict]) -> list:
    from deepeval.dataset import Golden

    return [
        Golden(
            input=str(row["input"]),
            expected_output=row.get("expected_output"),
            context=row.get("context"),
            additional_metadata=row.get("additional_metadata"),
            comments=row.get("comments"),
        )
        for row in rows
    ]


def _load_goldens(dataset_path: Path | None):
    from deepeval.dataset import EvaluationDataset, Golden

    if dataset_path is not None:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset nao encontrado: {dataset_path}")

        if dataset_path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return EvaluationDataset(goldens=_load_goldens_from_rows(rows))

        dataset = EvaluationDataset()
        dataset.add_goldens_from_json_file(file_path=str(dataset_path))
        return dataset

    from golden_dataset_template import build_golden_dataset

    goldens = [
        Golden(
            input=g.input,
            expected_output=g.expected_output,
            context=g.context,
            additional_metadata=g.to_deepeval_dict()["additional_metadata"],
            comments=g.comments,
        )
        for g in build_golden_dataset()
    ]
    return EvaluationDataset(goldens=goldens)


def _has_retrieval_context(test_case) -> bool:
    rc = getattr(test_case, "retrieval_context", None)
    return isinstance(rc, list) and len(rc) > 0


def _load_test_cases_from_results(path: Path) -> list:
    """Carrega test cases de um JSON gerado por execução anterior (--skip-metrics)."""
    from deepeval.test_case import LLMTestCase

    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de test cases nao encontrado: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"JSON invalido (esperado lista): {path}")
    allowed = set(LLMTestCase.model_fields.keys())
    cases: list[LLMTestCase] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Entrada {i} invalida em {path}")
        filtered = {k: v for k, v in row.items() if k in allowed}
        cases.append(LLMTestCase(**filtered))
    return cases


def _build_metrics(
    *,
    threshold: float,
    judge_model,
    test_cases: list,
    metrics_async: bool = False,
) -> list:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import SingleTurnParams

    kwargs: dict = {
        "threshold": threshold,
        "model": judge_model,
        "async_mode": metrics_async,
    }

    metrics = [
        GEval(
            name="Correctness",
            criteria=(
                "Determine se a resposta do assistente ('actual output') está correta "
                "e completa em relação à resposta esperada ('expected output'). "
                "Aceite redação diferente se os fatos principais coincidirem."
            ),
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            **kwargs,
        ),
        AnswerRelevancyMetric(**kwargs),
    ]

    if all(_has_retrieval_context(tc) for tc in test_cases):
        metrics.append(FaithfulnessMetric(**kwargs))
    else:
        missing = sum(1 for tc in test_cases if not _has_retrieval_context(tc))
        print(
            f"\nAviso: Faithfulness omitida — {missing} caso(s) sem retrieval_context."
        )

    return metrics


def _build_deepeval_async_config(
    *,
    max_concurrent: int | None = None,
    throttle_s: float | None = None,
):
    from deepeval.evaluate.configs import AsyncConfig

    mc = max_concurrent if max_concurrent is not None else eval_metrics_max_concurrent()
    tv = throttle_s if throttle_s is not None else eval_metrics_throttle_s()
    return AsyncConfig(run_async=True, max_concurrent=mc, throttle_value=tv)


def _warn_judge_rate_limits(judge_model_name: str) -> None:
    name = (judge_model_name or "").lower()
    if "free" in name or name in ("openrouter/auto", "openrouter/free"):
        print(
            "\nAviso: juiz no tier gratuito OpenRouter — limite ~50 req/dia e ~20/min.\n"
            "  Fase de métricas roda serial (max_concurrent=1) com pausa entre casos.\n"
            "  Se falhar com 429: use --skip-metrics, depois --resume-metrics no JSON salvo,\n"
            "  ou defina EVAL_JUDGE_MODEL para um modelo pago (ex.: openai/gpt-4o-mini).\n"
        )


def _print_runtime_status(runtime) -> None:
    from harness import runtime_paths, runtime_status

    status = runtime_status(runtime)
    paths = runtime_paths()
    print("Recursos disponiveis:")
    for key, ok in status.items():
        mark = "ok" if ok else "indisponivel"
        print(f"  - {key}: {mark}")
    print("Caminhos esperados (local/Docker):")
    for key, path in paths.items():
        print(f"  - {key}: {path}")


def run_evaluation(
    *,
    dataset_path: Path | None = None,
    limit: int | None = None,
    category: str | None = None,
    threshold: float = 0.5,
    judge_model: str | None = None,
    judge_provider: str = "auto",
    skip_metrics: bool = False,
    skip_unavailable: bool = False,
    require_ready: bool = False,
    force: bool = False,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    request_interval: float | None = None,
    resume_metrics: Path | None = None,
    metrics_max_concurrent: int | None = None,
    metrics_throttle: float | None = None,
) -> int:
    _ensure_import_paths()
    _configure_eval_rate_limits(case_interval_s=request_interval)

    from deepeval.test_case import LLMTestCase

    if resume_metrics is not None:
        print("Assistente de Lab — avaliação DeepEval (somente métricas)")
        print(f"Test cases: {resume_metrics}")
        try:
            test_cases = _load_test_cases_from_results(resume_metrics)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not test_cases:
            print("Nenhum test case no arquivo.", file=sys.stderr)
            return 1
        print(f"Carregados {len(test_cases)} test case(s).\n")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if skip_metrics:
            print("Nada a fazer: --resume-metrics ja contem respostas; omita --skip-metrics.", file=sys.stderr)
            return 1
    else:
        from harness import (
            build_eval_runtime,
            filter_goldens,
            filter_runnable_goldens,
            golden_project_ids,
            resolve_retrieval_context,
            run_assistente_turn,
            runtime_status,
        )

        if dataset_path is None:
            dataset_path = _latest_dataset_path()

        print("Assistente de Lab — avaliação DeepEval (end-to-end)")
        if dataset_path:
            print(f"Dataset: {dataset_path}")
        else:
            print("Dataset: goldens em memória (build_golden_dataset)")

        try:
            dataset = _load_goldens(dataset_path)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        goldens = filter_goldens(dataset.goldens, category=category, limit=limit)
        if not goldens:
            print("Nenhum golden encontrado com os filtros informados.", file=sys.stderr)
            return 1

        runtime = build_eval_runtime()
        _print_runtime_status(runtime)

        status = runtime_status(runtime)
        core_ready = all(
            status[k] for k in ("documents_index", "olap_tables", "ml_model")
        )
        if require_ready and not core_ready and not force:
            print(
                "\nAbortado: indice RAG, OLAP ou ML indisponivel.\n"
                "Prepare a base no Streamlit (Documentos) ou rode dentro do Docker:\n"
                "  docker compose build streamlit\n"
                "  docker compose up -d streamlit\n"
                "  docker compose exec streamlit python evals/run_assistente_eval.py --limit 3\n"
                "Use --force para rodar mesmo assim (resultados tendem a ser invalidos).",
                file=sys.stderr,
            )
            return 1

        skipped: list[tuple] = []
        if skip_unavailable:
            goldens, skipped = filter_runnable_goldens(goldens, runtime)
            if skipped:
                print(f"\nIgnorados {len(skipped)} golden(s) por falta de infra:")
                for g, reason in skipped[:5]:
                    gid = _golden_meta_id(g)
                    print(f"  - {gid}: falta {reason}")
                if len(skipped) > 5:
                    print(f"  ... e mais {len(skipped) - 5}")
            if not goldens:
                print("\nNenhum golden executavel com a infra atual.", file=sys.stderr)
                return 1
        elif not core_ready and not force:
            print(
                "\nAviso: infra incompleta — respostas provavelmente dirao "
                "'nao encontrei nos documentos' ou alucinarao tool calls.\n"
                "Use --skip-unavailable, --require-ready ou rode no Docker.\n"
            )

        print(f"\nExecutando {len(goldens)} caso(s)...\n")
        _print_rate_limit_config()
        print()

        case_interval = eval_case_interval_s()
        test_cases: list[LLMTestCase] = []
        for i, golden in enumerate(goldens, start=1):
            golden_id = _golden_meta_id(golden)
            print(f"[{i}/{len(goldens)}] {golden_id}: {golden.input[:80]}...")
            turn = run_assistente_turn(
                golden.input,
                runtime=runtime,
                project_ids=golden_project_ids(golden),
            )
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=turn.actual_output,
                expected_output=getattr(golden, "expected_output", None),
                context=getattr(golden, "context", None),
                retrieval_context=resolve_retrieval_context(turn, golden),
            )
            test_cases.append(test_case)
            preview = turn.actual_output.replace("\n", " ")[:120]
            print(f"    -> {preview}{'...' if len(turn.actual_output) > 120 else ''}")
            if case_interval > 0 and i < len(goldens):
                time.sleep(case_interval)

        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cases_path = output_dir / f"eval_test_cases_{stamp}.json"
        cases_path.write_text(
            json.dumps([tc.model_dump() for tc in test_cases], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nTest cases salvos em: {cases_path}")

        if skip_metrics:
            print("\nMétricas ignoradas (--skip-metrics).")
            print(
                "Para rodar só o juiz depois:\n"
                f"  python evals/run_assistente_eval.py --resume-metrics {cases_path}"
            )
            return 0

    if skip_metrics:
        print("\nMétricas ignoradas (--skip-metrics).")
        return 0

    from judge_model import build_judge_model, judge_backend_label, resolve_judge_model_name

    try:
        judge = build_judge_model(provider=judge_provider, model=judge_model)  # type: ignore[arg-type]
    except RuntimeError as exc:
        print(f"\nErro ao configurar juiz LLM: {exc}", file=sys.stderr)
        return 1

    print(f"\nJuiz LLM-as-judge: {judge_backend_label(judge_provider)}")  # type: ignore[arg-type]
    _warn_judge_rate_limits(resolve_judge_model_name(judge_model))

    mc = metrics_max_concurrent if metrics_max_concurrent is not None else eval_metrics_max_concurrent()
    tv = metrics_throttle if metrics_throttle is not None else eval_metrics_throttle_s()
    print(
        f"Metricas DeepEval: max_concurrent={mc}, throttle={tv:.0f}s entre casos "
        "(env: EVAL_METRICS_MAX_CONCURRENT, EVAL_METRICS_THROTTLE_S)\n"
    )

    from deepeval import evaluate

    metrics = _build_metrics(
        threshold=threshold,
        judge_model=judge,
        test_cases=test_cases,
        metrics_async=False,
    )
    print(f"Rodando {len(metrics)} metrica(s) DeepEval...\n")
    try:
        result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            identifier=f"assistente-lab-{stamp}",
            async_config=_build_deepeval_async_config(
                max_concurrent=mc,
                throttle_s=tv,
            ),
        )
    except Exception as exc:
        err = str(exc).lower()
        if "429" in err or "rate limit" in err or "retryerror" in err:
            print(
                "\nErro: limite OpenRouter na fase de metricas (juiz LLM).\n"
                "  Opcao A: aguarde reset diario ou adicione creditos no OpenRouter.\n"
                "  Opcao B: rode só metricas depois no JSON ja salvo:\n"
                "    python evals/run_assistente_eval.py --resume-metrics evals/results/eval_test_cases_....json\n"
                "  Opcao C: --skip-metrics na proxima geracao de respostas.\n",
                file=sys.stderr,
            )
        raise

    passed = sum(1 for tr in result.test_results if tr.success)
    total = len(result.test_results)
    print(f"\nResumo: {passed}/{total} test case(s) passaram.")
    if result.confident_link:
        print(f"Confident AI: {result.confident_link}")

    return 0 if passed == total else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Avaliação end-to-end do Assistente de Lab com DeepEval.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="JSON de goldens exportado (padrão: arquivo mais recente em datasets/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Máximo de casos a avaliar (útil para smoke test).",
    )
    parser.add_argument(
        "--category",
        choices=["greeting", "rag", "olap", "ml", "combined", "out_of_scope"],
        default=None,
        help="Filtra por categoria anotada no golden.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Limiar das métricas DeepEval (0–1).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Modelo juiz (ex.: openrouter/auto, openai/gpt-4o-mini). "
            "Padrao: EVAL_JUDGE_MODEL ou LLM_MODEL."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        choices=["auto", "openrouter", "openai"],
        default="auto",
        help="Backend do juiz: auto detecta OPENROUTER_API_KEY (padrao).",
    )
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="Ignora goldens cujo indice/OLAP/ML exigido nao esta pronto.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Aborta se RAG, OLAP ou ML nao estiverem todos disponiveis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Roda mesmo com infra incompleta (nao recomendado).",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="So gera actual_output; nao chama metricas LLM-as-judge.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Pasta para salvar test cases gerados.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Pausa extra (s) entre cada golden, além de LLM_MIN_REQUEST_INTERVAL_S. "
            "Útil no tier gratuito do OpenRouter. "
            "Env: EVAL_CASE_INTERVAL_S."
        ),
    )
    parser.add_argument(
        "--resume-metrics",
        type=Path,
        default=None,
        metavar="JSON",
        help=(
            "Pula geracao de respostas; roda apenas metricas DeepEval no JSON "
            "salvo (eval_test_cases_*.json)."
        ),
    )
    parser.add_argument(
        "--metrics-max-concurrent",
        type=int,
        default=None,
        help="Casos em paralelo na fase de metricas (padrao: 1). Env: EVAL_METRICS_MAX_CONCURRENT.",
    )
    parser.add_argument(
        "--metrics-throttle",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Pausa entre casos nas metricas (padrao: 15s). Env: EVAL_METRICS_THROTTLE_S.",
    )
    args = parser.parse_args(argv)

    return run_evaluation(
        dataset_path=args.dataset,
        limit=args.limit,
        category=args.category,
        threshold=args.threshold,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        skip_metrics=args.skip_metrics,
        skip_unavailable=args.skip_unavailable,
        require_ready=args.require_ready,
        force=args.force,
        output_dir=args.output_dir,
        request_interval=args.request_interval,
        resume_metrics=args.resume_metrics,
        metrics_max_concurrent=args.metrics_max_concurrent,
        metrics_throttle=args.metrics_throttle,
    )


if __name__ == "__main__":
    raise SystemExit(main())
