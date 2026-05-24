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

from ml.datasets import normalize_column_name
from ml.dictionary import load_dataset_catalog
from ml.paths import chat_ml_model_available, chat_ml_model_path
from ml.predict import predict_from_bundle
from ml.sequence_embeddings import extract_sequences_from_text, sequence_column_names
from ml.training import ModelBundle, load_model_bundle
from qwen35_inference import (
    PROFILE_CHAT_ROUTER,
    chat_history_chars_per_message,
    chat_max_history_turns,
    chat_max_tokens,
    create_chat_completion,
    format_history_snippet,
    strip_thinking_blocks,
)

_JSON_OBJECT = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_EXTRACT_MAX_TOKENS = min(512, chat_max_tokens(ml_route=True))

_EXTRACT_SYSTEM = """Você extrai valores de features para predição com um modelo ML de laboratório.

Responda APENAS com JSON válido, sem markdown:
{"rows": [ { "nome_exato_da_coluna": valor, ... }, ... ], "error": null ou "motivo"}

Regras:
- Use EXATAMENTE os nomes de coluna listados no esquema (incluindo espaços e colchetes).
- ``rows`` pode ter uma ou mais linhas (uma predição por objeto).
- Valores numéricos como número JSON; categorias como string.
- Se o usuário não forneceu dados suficientes para preencher as features obrigatórias, retorne ``rows``: [] e ``error`` explicando o que falta.
- Não invente valores que o usuário não mencionou; campos ausentes podem ser omitidos (o pipeline imputa medianas).
- Não inclua a coluna-alvo (target) nas linhas.
- Colunas de sequência (Ab_heavy_chain_seq, Ab_light_chain_seq, Ag_seq) podem ser strings longas com aminoácidos (A,C,D,E,...)."""

_EXTRACT_USER_TEMPLATE = """Esquema de features do modelo (colunas obrigatórias para inferência):
{schema}

Histórico recente:
{history}

Mensagem atual do usuário:
{message}

Extraia as linhas para predição."""

_ASSIGNMENT_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_\[\] /µ\-\.]*?)\s*=\s*(?P<val>[^,;\n]+?)"
    r"(?=\s*(?:,|\.|$|\n|\s+e\s+))",
    re.IGNORECASE,
)
_AGTYPE_INLINE_RE = re.compile(
    r"\bAgtype\s*(?:[:=]\s*|\s+)(?P<val>[A-Za-z0-9][A-Za-z0-9_\-\.]*)",
    re.IGNORECASE,
)


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


def _build_feature_schema_text(bundle: ModelBundle) -> str:
    try:
        catalog = load_dataset_catalog(bundle.catalog_id)
    except FileNotFoundError:
        catalog = None
    lines: list[str] = []

    if catalog is not None:
        lines.append("#### Colunas disponíveis no dataset AbRank (entrada possível)")
        for col_name in catalog.input_feature_column_names():
            desc = catalog.description_for(col_name)
            line = f"- `{col_name}`"
            if desc:
                line += f": {desc}"
            lines.append(line)
        lines.append("")

    lines.append("#### Colunas que ESTE arquivo .pkl usa na inferência (obrigatórias no JSON)")
    seq_pca_cols = {c for c in bundle.feature_columns if c.startswith("seq_pca_")}
    tabular_cols = [c for c in bundle.feature_columns if c not in seq_pca_cols]
    for col in tabular_cols:
        desc = catalog.description_for(col) if catalog else ""
        line = f"- `{col}`"
        if desc:
            line += f": {desc}"
        lines.append(line)

    transformer = getattr(bundle, "sequence_transformer", None)
    if transformer is not None:
        lines.append("")
        lines.append("#### Sequências (pré-processadas com ESM-2 + PCA antes da predição)")
        for col in transformer.config.sequence_columns:
            desc = catalog.description_for(col) if catalog else ""
            line = f"- `{col}` (texto aminoácido; gera {len(transformer.pca_columns_)} colunas ``seq_pca_*``)"
            if desc:
                line += f": {desc}"
            lines.append(line)
        lines.append(
            "_Informe pelo menos uma sequência quando o modelo foi treinado com embeddings._"
        )
    elif seq_pca_cols:
        for col in sorted(seq_pca_cols):
            lines.append(f"- `{col}` (derivada de sequência no treino)")

    if catalog is not None:
        missing_in_model = [
            c for c in catalog.input_feature_column_names() if c not in bundle.feature_columns
        ]
        if missing_in_model:
            lines.append(
                "\n_Outras colunas do dataset só entram na predição após **retreinar** o modelo "
                f"incluindo-as nas features (ex.: {', '.join(f'`{c}`' for c in missing_in_model[:6])}"
                f"{'…' if len(missing_in_model) > 6 else ''})._"
            )

    task = getattr(bundle, "task", "classification")
    target = bundle.target_column
    lines.append(f"\nTarefa: **{task}** · coluna-alvo no treino: `{target}`")
    if task == "regression":
        lines.append(
            "A saída da predição é `predicao_log_aff` (log de afinidade Ab–Ag, benchmark AbRank)."
        )
    return "\n".join(lines)


def _coerce_feature_value(raw: str, column: str) -> object:
    text = str(raw).strip().strip("\"'")
    if not text:
        return pd.NA
    lower_col = column.lower()
    if "agtype" in lower_col or column in {"Agtype"}:
        return text
    try:
        if "." in text or "e" in text.lower():
            return float(text.replace(",", "."))
        return int(text)
    except ValueError:
        return text


def _column_name_map(bundle: ModelBundle) -> dict[str, str]:
    cols = list(_tabular_feature_columns(bundle))
    transformer = getattr(bundle, "sequence_transformer", None)
    if transformer is not None:
        cols.extend(list(transformer.config.sequence_columns))
    return {normalize_column_name(c): c for c in cols}


def extract_features_rule_based(message: str, bundle: ModelBundle) -> list[dict]:
    """
    Extrai features da mensagem com regex (sem LLM).

    Cobre o formato comum no chat: ``Agtype X`` e ``coluna = valor`` separados por vírgula ou `` e ``.
    """
    col_map = _column_name_map(bundle)
    row: dict[str, object] = {}

    param_section = message
    parts = re.split(r"parâmetros\s*:", message, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) > 1:
        param_section = parts[1]

    for chunk in re.split(r",|\s+e\s+", param_section, flags=re.IGNORECASE):
        chunk = chunk.strip().strip(".")
        if not chunk or "=" not in chunk:
            continue
        key_part, _, val_part = chunk.partition("=")
        key_norm = normalize_column_name(key_part)
        canonical = col_map.get(key_norm)
        if canonical:
            row[canonical] = _coerce_feature_value(val_part, canonical)

    for match in _ASSIGNMENT_RE.finditer(message):
        key_norm = normalize_column_name(match.group("key"))
        canonical = col_map.get(key_norm)
        if canonical and canonical not in row:
            row[canonical] = _coerce_feature_value(match.group("val"), canonical)

    agtype_match = _AGTYPE_INLINE_RE.search(message)
    if agtype_match and "agtype" in col_map:
        canonical = col_map["agtype"]
        row[canonical] = agtype_match.group("val")

    for col in _tabular_feature_columns(bundle):
        if col in row:
            continue
        escaped = re.escape(col)
        direct = re.search(
            rf"{escaped}\s*[:=]\s*([^,;\n]+)",
            message,
            re.IGNORECASE,
        )
        if direct:
            row[col] = _coerce_feature_value(direct.group(1), col)

    if not row:
        return []
    return [row]


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
        history=format_history_snippet(
            history,
            max_turns=chat_max_history_turns(ml_route=True),
            max_chars_per_message=chat_history_chars_per_message(ml_route=True),
        ),
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


def _tabular_feature_columns(bundle: ModelBundle) -> list[str]:
    return [c for c in bundle.feature_columns if not c.startswith("seq_pca_")]


def _merge_sequence_fields(rows: list[dict], message: str) -> list[dict]:
    """Combina sequências extraídas da mensagem (regex/FASTA) com linhas do LLM/regras."""
    seqs = extract_sequences_from_text(message)
    if not rows and seqs:
        return [dict(seqs)]
    if not seqs:
        return rows
    merged: list[dict] = []
    for row in rows:
        item = dict(row)
        for col, seq in seqs.items():
            existing = str(item.get(col, "") or "").strip()
            if len(existing) < 10:
                item[col] = seq
        merged.append(item)
    return merged


def _rows_to_dataframe(rows: list[dict], bundle: ModelBundle) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in _tabular_feature_columns(bundle):
        if col not in df.columns:
            df[col] = pd.NA
    transformer = getattr(bundle, "sequence_transformer", None)
    if transformer is not None:
        for col in transformer.config.sequence_columns:
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
    if extract_err or not rows:
        rule_rows = extract_features_rule_based(message, loaded)
        if rule_rows:
            rows = rule_rows
            extract_err = None
            raw = raw or json.dumps({"rows": rows, "source": "rule_based"}, ensure_ascii=False)
    rows = _merge_sequence_fields(rows, message)
    if not rows:
        err = extract_err or "Nenhuma feature ou sequência foi extraída da mensagem."
        return MlInferResult(
            ok=False,
            model_path=str(path),
            error=err,
            raw_llm_response=raw,
            context_for_llm=(
                "### Predição ML (dados insuficientes)\n"
                f"{err}\n\nInforme features tabulares e/ou sequências de aminoácidos."
            ),
        )

    transformer = getattr(loaded, "sequence_transformer", None)
    if transformer is not None:
        from ml.sequence_embeddings import clean_protein_sequence

        has_seq = any(
            clean_protein_sequence(rows[0].get(col))
            for col in transformer.config.sequence_columns
        )
        if not has_seq:
            err = (
                "Este modelo usa embeddings ESM-2. Informe pelo menos uma sequência: "
                "Ab_heavy_chain_seq, Ab_light_chain_seq ou Ag_seq."
            )
            return MlInferResult(
                ok=False,
                model_path=str(path),
                error=err,
                raw_llm_response=raw,
                context_for_llm=f"### Predição ML (dados insuficientes)\n{err}",
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
