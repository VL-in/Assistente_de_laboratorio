"""
Carregamento e preparação de datasets tabulares para ML tradicional.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from ml.dictionary import DatasetCatalog, load_dataset_catalog
from ml.paths import resolve_dengue_results_dir

AMOSTRAS_FILE = "amostras_dengue.xlsx"
AMOSTRAS_SHEET = "Amostras"
COMPILADO_FILE = "Compilado_resultado_otimizacao.xlsx"
COMPILADO_SHEET = "Planilha1"
MERGE_KEY = "ID amostra"


def normalize_column_name(name: str) -> str:
    """Normaliza nomes de coluna para comparação (acentos, espaços, case)."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def align_catalog_columns(df: pd.DataFrame, catalog: DatasetCatalog) -> dict[str, str]:
    """
    Mapeia nomes canônicos do catálogo → nomes reais no DataFrame.

    Útil quando o Excel grava acentos de forma inconsistente entre máquinas.
    """
    index = {normalize_column_name(col): col for col in df.columns}
    mapping: dict[str, str] = {}
    for spec in catalog.columns:
        canonical = spec.name
        if canonical in df.columns:
            mapping[canonical] = canonical
            continue
        hit = index.get(normalize_column_name(canonical))
        if hit:
            mapping[canonical] = hit
    return mapping


def load_dengue_elisa_253(
    results_dir: Path | None = None,
    *,
    catalog: DatasetCatalog | None = None,
) -> tuple[pd.DataFrame, DatasetCatalog]:
    """
    Carrega e junta as planilhas do projeto 253 (amostras + compilado ELISA).

    Retorna o DataFrame merged e o catálogo de colunas.
    """
    catalog = catalog or load_dataset_catalog()
    base = (results_dir or resolve_dengue_results_dir()).resolve()
    amostras_path = base / AMOSTRAS_FILE
    compilado_path = base / COMPILADO_FILE
    missing = [p.name for p in (amostras_path, compilado_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Arquivo(s) ausente(s) em `{base}`: {', '.join(missing)}. "
            "Configure ASSISTENTE_ML_DENGUE_RESULTS ou monte a pasta Projetos no Docker."
        )

    amostras = pd.read_excel(amostras_path, sheet_name=AMOSTRAS_SHEET)
    compilado = pd.read_excel(compilado_path, sheet_name=COMPILADO_SHEET)
    key = catalog.merge_key if catalog.merge_key in amostras.columns else MERGE_KEY
    if key not in amostras.columns or key not in compilado.columns:
        raise ValueError(f"Coluna de junção `{key}` não encontrada nas planilhas.")

    merged = amostras.merge(compilado, on=key, how=catalog.merge_how)
    return merged, catalog


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
    exclude |= set(catalog.suggested_drop)
    exclude.add(catalog.default_target)

    numeric_cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if col == catalog.merge_key:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        if "ABS" in str(col).upper():
            numeric_cols.append(col)
    return sorted(set(numeric_cols))
