"""
Predição ML acionada pelo chat — extração de features via LLM e inferência local.

Só deve ser chamada quando o roteador de intenção marca ``use_ml`` (pedido explícito).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openai import OpenAI

from ml.dictionary import load_dataset_catalog
from ml.paths import chat_ml_model_available, chat_ml_model_path
from ml.predict import predict_from_bundle
from ml.training import ModelBundle, load_model_bundle
from qwen35_inference import (
    PROFILE_CHAT_ROUTER,
    create_chat_completion,
    strip_thinking_blocks,
)

_JSON_OBJECT = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_EXTRACT_MAX_TOKENS = 768

_EXTRACT_SYSTEM = """Você extrai valores de features para predição com um modelo ML de laboratório.

Responda APENAS com JSON válido, sem markdown:
{"rows": [ { "nome_exato_da_coluna": valor, ... }, ... ], "error": null ou "motivo"}

Regras:
- Use EXATAMENTE os nomes de coluna listados no esquema (incluindo espaços e colchetes).
- ``rows`` pode ter uma ou mais linhas (uma predição por objeto).
- Valores numéricos como número JSON; categorias como string.
- Se o usuário não forneceu dados suficientes para preencher as features obrigatórias, retorne ``rows``: [] e ``error`` explicando o que falta.
- Não invente valores que o usuário não mencionou; campos ausentes podem ser omitidos (o pipeline imputa medianas).
- Não inclua a coluna-alvo (target) nas linhas."""

_EXTRACT_USER_TEMPLATE = """Esquema de features do modelo (colunas obrigatórias para inferência):
{schema}

Histórico recente:
{history}

Mensagem atual do usuário:
{message}

Extraia as linhas para predição."""


@dataclass
class MlInferResult:
    """Resultado da inferência ML para injetar no contexto do chat."""

    ok: bool
    model_path: str = ""
    error: str | None = None
    context_for_llm: str = ""
    predictions: pd.DataFrame | None = None
    raw_llm_response: str = ""


def resolve_chat_model_path() -> Path:
    return chat_ml_model_path()


def load_chat_model_bundle(path: Path | None = None) -> ModelBundle:
    target = (path or chat_ml_model_path()).expanduser().resolve()
    return load_model_bundle(target)


def _format_history_snippet(history: list[dict], *, max_turns: int = 3) -> str:
    if not history:
        return "(sem histórico anterior)"
    tail = history[-(max_turns * 2) :]
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "?")
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        label = "Usuário" if role == "user" else "Assistente"
        lines.append(f"{label}: {content[:400]}")
    return "\n".join(lines) if lines else "(sem histórico anterior)"


def _build_feature_schema_text(bundle: ModelBundle) -> str:
    try:
        catalog = load_dataset_catalog(bundle.catalog_id)
    except FileNotFoundError:
        catalog = None
    lines: list[str] = []
    for col in bundle.feature_columns:
        desc = ""
        if catalog is not None:
            desc = catalog.description_for(col)
        line = f"- `{col}`"
        if desc:
            line += f": {desc}"
        lines.append(line)
    task = getattr(bundle, "task", "classification")
    target = bundle.target_column
    lines.append(f"\nTarefa: **{task}** · coluna-alvo no treino: `{target}`")
    if task == "regression":
        lines.append(
            "A saída da predição é `predicao_log_aff` (log de afinidade Ab–Ag, benchmark AbRank)."
        )
    return "\n".join(lines)


def _parse_extract_json(raw: str) -> tuple[list[dict], str | None]:
    text = strip_thinking_blocks((raw or "").strip())
    if not text:
        return [], "Resposta vazia do extrator de features."
    data: dict | None = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(data, dict):
        return [], "Não foi possível interpretar o JSON de features."
    rows = data.get("rows")
    err = data.get("error")
    if err and str(err).strip().lower() not in ("null", "none", ""):
        return [], str(err).strip()
    if not isinstance(rows, list):
        return [], "JSON sem lista `rows`."
    clean: list[dict] = []
    for item in rows:
        if isinstance(item, dict) and item:
            clean.append(item)
    return clean, None


def extract_prediction_rows(
    message: str,
    *,
    history: list[dict],
    client: OpenAI,
    model: str,
    bundle: ModelBundle,
) -> tuple[list[dict], str, str | None]:
    """Usa o LLM para montar linhas de entrada a partir da mensagem do usuário."""
    schema = _build_feature_schema_text(bundle)
    user_block = _EXTRACT_USER_TEMPLATE.format(
        schema=schema,
        history=_format_history_snippet(history),
        message=message.strip(),
    )
    try:
        completion = create_chat_completion(
            client,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user_block},
            ],
            model=model,
            profile=PROFILE_CHAT_ROUTER,
            max_tokens=_EXTRACT_MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        return [], "", f"Falha ao extrair features: {exc}"

    rows, parse_err = _parse_extract_json(raw)
    if parse_err:
        return [], raw, parse_err
    if not rows:
        return [], raw, "Nenhuma linha de features foi extraída da mensagem."
    return rows, raw, None


def _rows_to_dataframe(rows: list[dict], bundle: ModelBundle) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in bundle.feature_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _format_predictions_for_llm(df: pd.DataFrame, bundle: ModelBundle) -> str:
    pred_col = (
        "predicao_log_aff"
        if getattr(bundle, "task", "classification") == "regression"
        else "predicao"
    )
    preview_cols = [c for c in df.columns if c in bundle.feature_columns or c == pred_col]
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    preview_cols.extend(prob_cols)
    slim = df[preview_cols] if preview_cols else df
    try:
        table = slim.to_markdown(index=False)
    except ImportError:
        table = "```\n" + slim.to_string(index=False) + "\n```"
    lines = [
        "### Resultado da inferência ML (modelo treinado)",
        f"Modelo: `{bundle.catalog_id}` · features: {len(bundle.feature_columns)}",
        "",
        table,
        "",
        "Explique o resultado ao usuário em linguagem clara. "
        "Não invente valores além da tabela acima.",
    ]
    if getattr(bundle, "task", "classification") == "regression":
        lines.insert(
            3,
            "Coluna `predicao_log_aff`: estimativa de log de afinidade (quanto maior, em geral, maior afinidade no AbRank).",
        )
    return "\n".join(lines)


def run_chat_ml_inference(
    message: str,
    *,
    history: list[dict] | None = None,
    client: OpenAI,
    model: str,
    bundle: ModelBundle | None = None,
    model_path: Path | None = None,
) -> MlInferResult:
    """
    Extrai features da mensagem, executa ``predict_from_bundle`` e formata contexto.

    Parameters
    ----------
    message:
        Pergunta atual do usuário (deve pedir predição explicitamente — roteador).
    history:
        Mensagens anteriores para desambiguar valores citados antes.
    client, model:
        Cliente OpenAI (LM Studio) para extração estruturada.
    bundle:
        Modelo já carregado (opcional; evita releitura do .pkl).
    model_path:
        Caminho do .pkl quando ``bundle`` não é passado.
    """
    path = (model_path or chat_ml_model_path()).expanduser().resolve()
    if not path.is_file():
        return MlInferResult(
            ok=False,
            model_path=str(path),
            error=f"Modelo ML não encontrado em `{path}`. Treine ou copie o .pkl para esse caminho.",
        )

    try:
        loaded = bundle if bundle is not None else load_model_bundle(path)
    except Exception as exc:
        return MlInferResult(
            ok=False,
            model_path=str(path),
            error=f"Não foi possível carregar o modelo: {exc}",
        )

    rows, raw, extract_err = extract_prediction_rows(
        message,
        history=history or [],
        client=client,
        model=model,
        bundle=loaded,
    )
    if extract_err:
        return MlInferResult(
            ok=False,
            model_path=str(path),
            error=extract_err,
            raw_llm_response=raw,
            context_for_llm=(
                "### Predição ML (dados insuficientes)\n"
                f"{extract_err}\n\n"
                "Peça ao usuário os valores das features listadas no esquema, "
                "sem inventar números."
            ),
        )

    try:
        input_df = _rows_to_dataframe(rows, loaded)
        pred_df = predict_from_bundle(loaded, input_df)
    except Exception as exc:
        return MlInferResult(
            ok=False,
            model_path=str(path),
            error=f"Erro na inferência: {exc}",
            raw_llm_response=raw,
        )

    ctx = _format_predictions_for_llm(pred_df, loaded)
    return MlInferResult(
        ok=True,
        model_path=str(path),
        context_for_llm=ctx,
        predictions=pred_df,
        raw_llm_response=raw,
    )


def ml_inference_status_message() -> str:
    """Texto curto para a UI (aba Conversa)."""
    path = chat_ml_model_path()
    if chat_ml_model_available():
        return f"Modelo ML pronto (`{path.name}`)"
    return f"Modelo ML ausente em `{path}`"
