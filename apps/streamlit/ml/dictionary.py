"""
Catálogo de colunas (dicionário de dados) para datasets de ML tradicional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CATALOGS_DIR = Path(__file__).resolve().parent / "catalogs"
DEFAULT_CATALOG_ID = "dengue_elisa_253"


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
    project_folder: str
    results_subdir: str
    source_files: list[dict[str, str]]
    merge_key: str
    merge_how: str
    default_target: str
    default_positive_label: str
    suggested_drop: list[str]
    feature_hints: dict[str, list[str]]
    columns: list[ColumnSpec] = field(default_factory=list)

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


def _catalog_path(catalog_id: str) -> Path:
    path = CATALOGS_DIR / f"{catalog_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Catálogo não encontrado: {path}")
    return path


def load_dataset_catalog(catalog_id: str = DEFAULT_CATALOG_ID) -> DatasetCatalog:
    raw: dict[str, Any] = yaml.safe_load(_catalog_path(catalog_id).read_text(encoding="utf-8"))
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
    return DatasetCatalog(
        dataset_id=str(raw["dataset_id"]),
        display_name=str(raw["display_name"]),
        project_folder=str(raw["project_folder"]),
        results_subdir=str(raw["results_subdir"]),
        source_files=list(raw.get("source_files") or []),
        merge_key=str(merge.get("key", "ID amostra")),
        merge_how=str(merge.get("how", "inner")),
        default_target=str(raw.get("default_target", "")),
        default_positive_label=str(raw.get("default_positive_label", "Positivo")),
        suggested_drop=[str(x) for x in raw.get("suggested_drop") or []],
        feature_hints=dict(raw.get("feature_hints") or {}),
        columns=columns,
    )
