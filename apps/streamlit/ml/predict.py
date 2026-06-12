"""
Predição com modelos serializados (.pkl) em dados novos.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from ml.datasets import coerce_abs_columns
from ml.sequence_embeddings import clean_protein_sequence

DEFAULT_MERGE_KEY = "AbAgID"
from ml.training import ModelBundle


def read_prediction_table(uploaded) -> pd.DataFrame:
    """Lê CSV ou Excel enviado pela UI Streamlit."""
    name = (getattr(uploaded, "name", "") or "").lower()
    raw = uploaded.getvalue()
    if name.endswith(".csv") or name.endswith(".tsv"):
        sep = "\t" if name.endswith(".tsv") else ","
        return pd.read_csv(BytesIO(raw), sep=sep)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(BytesIO(raw))
    raise ValueError("Formato não suportado. Use CSV ou Excel (.xlsx).")


def _has_sequence_input(df: pd.DataFrame, sequence_columns: tuple[str, ...]) -> bool:
    for col in sequence_columns:
        if col not in df.columns:
            continue
        for raw in df[col].dropna().astype(str):
            if clean_protein_sequence(raw):
                return True
    return False


def validate_prediction_columns(bundle: ModelBundle, df: pd.DataFrame) -> list[str]:
    """Lista colunas tabulares ausentes; ``seq_pca_*`` são derivadas do ESM no bundle."""
    seq_pca = {c for c in bundle.feature_columns if c.startswith("seq_pca_")}
    tabular_needed = [c for c in bundle.feature_columns if c not in seq_pca]
    missing = [c for c in tabular_needed if c not in df.columns]
    transformer = getattr(bundle, "sequence_transformer", None)
    if transformer is not None and seq_pca:
        if not _has_sequence_input(df, transformer.config.sequence_columns):
            missing.append(
                "sequências ESM (informe Ab_heavy_chain_seq, Ab_light_chain_seq e/ou Ag_seq)"
            )
    return missing


def predict_from_bundle(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """
    Anexa colunas de predição ao DataFrame de entrada.

    Preserva todas as colunas originais e adiciona ``predicao`` e, se possível,
    probabilidades por classe.
    """
    missing = validate_prediction_columns(bundle, df)
    if missing:
        raise ValueError(f"Colunas ausentes para predição: {missing}")
    work = coerce_abs_columns(df.copy())
    out = work.copy()
    preds = bundle.predict_labels(work)
    out_col = "predicao" if getattr(bundle, "task", "classification") != "regression" else "predicao_log_aff"
    out[out_col] = preds
    if getattr(bundle, "task", "classification") == "regression":
        return out
    if hasattr(bundle.pipeline, "predict_proba"):
        # Probabilidades são opcionais ("se possível"): uma falha aqui (estimador
        # sem suporte real, problema numérico, classe ausente) não deve invalidar
        # a predição de label já calculada com sucesso acima.
        try:
            proba = bundle.predict_proba(work)
            for cls in proba.columns:
                out[f"prob_{cls}"] = proba[cls].values
        except (AttributeError, ValueError):
            pass
    return out


def merge_with_amostras_if_needed(
    df: pd.DataFrame,
    amostras: pd.DataFrame | None,
    *,
    merge_key: str = DEFAULT_MERGE_KEY,
) -> pd.DataFrame:
    """
    Opcional: enriquece predição apenas com metadados de amostras (por ID).

    Útil quando o lote novo traz só leituras ABS + ID amostra.
    """
    if (
        amostras is None
        or merge_key not in df.columns
        or merge_key not in amostras.columns
    ):
        return df
    meta_cols = [c for c in amostras.columns if c not in df.columns or c == merge_key]
    slim = amostras[meta_cols].drop_duplicates(subset=[merge_key])
    return df.merge(slim, on=merge_key, how="left")
