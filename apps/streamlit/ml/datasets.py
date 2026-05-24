"""
Carregamento e preparação de datasets tabulares para ML tradicional.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from ml.dictionary import DatasetCatalog, load_dataset_catalog
from ml.kaggle_sources import load_kaggle_csv


def normalize_column_name(name: str) -> str:
    """Normaliza nomes de coluna para comparação (acentos, espaços, case)."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataset_from_catalog(
    catalog: DatasetCatalog | None = None,
    *,
    catalog_id: str | None = None,
    max_rows: int | None = None,
    force_download: bool = False,
) -> tuple[pd.DataFrame, DatasetCatalog]:
    """
    Carrega o dataset descrito no catálogo YAML (fonte Kaggle).
    """
    catalog = catalog or load_dataset_catalog(catalog_id)
    if not catalog.is_kaggle:
        raise ValueError(
            f"Catálogo `{catalog.dataset_id}` sem fonte Kaggle. "
            "O pipeline ML suporta apenas datasets Kaggle (ex.: AbRank)."
        )
    if not catalog.kaggle_split_file:
        raise ValueError(f"Catálogo `{catalog.dataset_id}` sem `kaggle_split_file`.")
    df = load_kaggle_csv(
        catalog.kaggle_handle,
        catalog.kaggle_split_file,
        separator=catalog.csv_separator,
        max_rows=max_rows,
        force_download=force_download,
    )
    return df, catalog


def coerce_abs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas ABS para numérico (planilha pode trazer texto ou vírgula)."""
    out = df.copy()
    for col in out.columns:
        if "ABS" in str(col).upper() or col.startswith("ABS "):
            series = out[col]
            if series.dtype == object:
                series = series.astype(str).str.replace(",", ".", regex=False)
            out[col] = pd.to_numeric(series, errors="coerce")
    return out


def prepare_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    drop_na_target: bool = True,
    regression_target: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X/y e remove linhas sem alvo quando solicitado."""
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Colunas de feature ausentes: {missing_features}")
    if target_column not in df.columns:
        raise ValueError(f"Coluna-alvo ausente: {target_column}")

    work = df.copy()
    work = coerce_abs_columns(work)
    y = work[target_column]
    if drop_na_target:
        mask = y.notna() & (y.astype(str).str.strip() != "")
        work = work.loc[mask]
        y = work[target_column]
        if regression_target:
            numeric_target = pd.to_numeric(y, errors="coerce")
            if numeric_target.notna().sum() > len(y) * 0.5:
                valid = numeric_target.notna()
                work = work.loc[valid]
                y = numeric_target.loc[valid]

    x = work[feature_columns].copy()
    return x, y


def default_feature_columns(
    df: pd.DataFrame,
    catalog: DatasetCatalog,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """Sugere features numéricas excluindo colunas do catálogo e alvo."""
    exclude = exclude or set()
    exclude |= catalog.columns_excluded_from_features()

    numeric_cols: list[str] = []
    hinted_numeric = set(catalog.feature_hints.get("numeric") or [])
    hinted_categorical = set(catalog.feature_hints.get("categorical") or [])

    for col in df.columns:
        if col in exclude:
            continue
        if catalog.merge_key and col == catalog.merge_key:
            continue
        series = df[col]
        if col in hinted_numeric and col not in exclude:
            if series.notna().sum() > 0:
                numeric_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().sum() == 0:
                continue
            numeric_cols.append(col)
            continue
        if "ABS" in str(col).upper():
            numeric_cols.append(col)
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.dropna().astype(str).head(200)
            if sample.empty:
                continue
            if sample.str.len().max() > 64:
                continue
            if col in hinted_categorical:
                numeric_cols.append(col)
    return sorted(set(numeric_cols))
