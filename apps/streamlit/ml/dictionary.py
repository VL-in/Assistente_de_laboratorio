"""
Catálogo de colunas (dicionário de dados) para datasets de ML tradicional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CATALOGS_DIR = Path(__file__).resolve().parent / "catalogs"
DEFAULT_CATALOG_ID = "abrank_kaggle"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    role: str
    dtype: str
    description: str
    ml_use: str
    notes: str = ""


@dataclass(frozen=True)
class DatasetCatalog:
    dataset_id: str
    display_name: str
    project_folder: str = ""
    results_subdir: str = ""
    source_files: list[dict[str, str]] = field(default_factory=list)
    merge_key: str = ""
    merge_how: str = "inner"
    kaggle_handle: str = ""
    kaggle_split_file: str = ""
    csv_separator: str = ","
    ml_task: str = "classification"
    default_metric: str = ""
    default_target: str = ""
    default_positive_label: str = "Positivo"
    suggested_drop: list[str] = field(default_factory=list)
    feature_hints: dict[str, list[str]] = field(default_factory=dict)
    columns: list[ColumnSpec] = field(default_factory=list)

    @property
    def is_kaggle(self) -> bool:
        return bool(self.kaggle_handle.strip())

    def column_dict_rows(self) -> list[dict[str, str]]:
        return [
            {
                "coluna": c.name,
                "tipo": c.dtype,
                "papel": c.role,
                "uso_ml": c.ml_use,
                "descrição": c.description,
                "notas": c.notes,
            }
            for c in self.columns
        ]

    def description_for(self, column_name: str) -> str:
        for col in self.columns:
            if col.name == column_name:
                return col.description
        return ""

    def columns_excluded_from_features(self) -> set[str]:
        """Colunas que nunca entram como feature (IDs, alvo, sequências longas, etc.)."""
        excluded = set(self.suggested_drop)
        if self.default_target:
            excluded.add(self.default_target)
        for col in self.columns:
            if col.ml_use in {
                "exclude",
                "identifier",
                "metadata",
                "target",
                "sequence",
                "auxiliary",
            }:
                excluded.add(col.name)
        return excluded

    def input_feature_column_names(self) -> list[str]:
        """Colunas do catálogo elegíveis como entrada do modelo (feature + optional_feature)."""
        excluded = self.columns_excluded_from_features()
        names: list[str] = []
        for col in self.columns:
            if col.name in excluded:
                continue
            if col.ml_use in {"feature", "optional_feature"}:
                names.append(col.name)
        return names


def _catalog_path(catalog_id: str) -> Path:
    path = CATALOGS_DIR / f"{catalog_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Catálogo não encontrado: {path}")
    return path


def list_catalog_ids() -> list[str]:
    paths = sorted(CATALOGS_DIR.glob("*.yaml"))
    return [p.stem for p in paths]


def load_dataset_catalog(catalog_id: str | None = None) -> DatasetCatalog:
    cid = (catalog_id or DEFAULT_CATALOG_ID).strip()
    raw: dict[str, Any] = yaml.safe_load(_catalog_path(cid).read_text(encoding="utf-8"))
    merge = raw.get("merge") or {}
    columns: list[ColumnSpec] = []
    for item in raw.get("columns") or []:
        columns.append(
            ColumnSpec(
                name=str(item["name"]),
                role=str(item.get("role", "")),
                dtype=str(item.get("dtype", "")),
                description=str(item.get("description", "")).strip(),
                ml_use=str(item.get("ml_use", "")),
                notes=str(item.get("notes", "")).strip(),
            )
        )
    sep = str(raw.get("csv_separator", ","))
    if sep.lower() in ("\\t", "tab"):
        sep = "\t"

    return DatasetCatalog(
        dataset_id=str(raw["dataset_id"]),
        display_name=str(raw["display_name"]),
        project_folder=str(raw.get("project_folder", "")),
        results_subdir=str(raw.get("results_subdir", "")),
        source_files=list(raw.get("source_files") or []),
        merge_key=str(merge.get("key", "")),
        merge_how=str(merge.get("how", "inner")),
        kaggle_handle=str(raw.get("kaggle_handle", "")),
        kaggle_split_file=str(raw.get("kaggle_split_file", "")),
        csv_separator=sep,
        ml_task=str(raw.get("ml_task", "classification")),
        default_metric=str(raw.get("default_metric", "")),
        default_target=str(raw.get("default_target", "")),
        default_positive_label=str(raw.get("default_positive_label", "Positivo")),
        suggested_drop=[str(x) for x in raw.get("suggested_drop") or []],
        feature_hints=dict(raw.get("feature_hints") or {}),
        columns=columns,
    )
