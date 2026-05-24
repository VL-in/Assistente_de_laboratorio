"""
Predição com modelos serializados (.pkl) em dados novos.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from ml.datasets import MERGE_KEY, coerce_abs_columns
from ml.training import ModelBundle, load_model_bundle


def read_prediction_table(uploaded) -> pd.DataFrame:
    """Lê CSV ou Excel enviado pela UI Streamlit."""
    name = (getattr(uploaded, "name", "") or "").lower()
    raw = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(raw))
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(BytesIO(raw))
    raise ValueError("Formato não suportado. Use CSV ou Excel (.xlsx).")


def predict_from_bundle(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """
    Anexa colunas de predição ao DataFrame de entrada.

    Preserva todas as colunas originais e adiciona ``predicao`` e, se possível,
    probabilidades por classe.
    """
    work = coerce_abs_columns(df.copy())
    out = work.copy()
    out["predicao"] = bundle.predict_labels(work)
    if hasattr(bundle.pipeline, "predict_proba"):
        proba = bundle.predict_proba(work)
        for cls in proba.columns:
            out[f"prob_{cls}"] = proba[cls].values
    return out


def validate_prediction_columns(bundle: ModelBundle, df: pd.DataFrame) -> list[str]:
    """Lista colunas exigidas pelo modelo que faltam no lote novo."""
    return [c for c in bundle.feature_columns if c not in df.columns]


def load_bundle_from_path(path: str | Path) -> ModelBundle:
    return load_model_bundle(Path(path))


def merge_with_amostras_if_needed(
    df: pd.DataFrame,
    amostras: pd.DataFrame | None,
    *,
    merge_key: str = MERGE_KEY,
) -> pd.DataFrame:
    """
    Opcional: enriquece predição apenas com metadados de amostras (por ID).

    Útil quando o lote novo traz só leituras ABS + ID amostra.
    """
    if amostras is None or merge_key not in df.columns or merge_key not in amostras.columns:
        return df
    meta_cols = [c for c in amostras.columns if c not in df.columns or c == merge_key]
    slim = amostras[meta_cols].drop_duplicates(subset=[merge_key])
    return df.merge(slim, on=merge_key, how="left")
