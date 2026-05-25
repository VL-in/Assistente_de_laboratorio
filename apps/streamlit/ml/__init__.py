"""ML tradicional (AutoML com FLAML) — AbRank (Kaggle), treino e predição."""

from ml.datasets import load_dataset_from_catalog
from ml.dictionary import load_dataset_catalog
from ml.kaggle_sources import load_abrank_split
from ml.paths import ml_models_root

__all__ = [
    "load_abrank_split",
    "load_dataset_catalog",
    "load_dataset_from_catalog",
    "ml_models_root",
]
