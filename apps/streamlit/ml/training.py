"""
Treino AutoML com FLAML (instalação mínima: estimadores baseados em scikit-learn).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ml.datasets import coerce_abs_columns, prepare_feature_matrix

# Estimadores disponíveis sem extras pip (lightgbm/xgboost/catboost).
MINIMAL_ESTIMATOR_LIST = ("rf", "extra_tree", "lrl1")


@dataclass
class FlamlTrainConfig:
    time_budget: int = 120
    metric: str = "f1"
    n_splits: int = 5
    estimator_list: tuple[str, ...] = MINIMAL_ESTIMATOR_LIST
    eval_method: str = "cv"
    seed: int = 42
    test_size: float = 0.2
    early_stop: bool = True
    verbose: int = 0
    custom_hp: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainReport:
    metric: str
    best_estimator: str
    best_config: dict[str, Any]
    train_seconds: float
    accuracy: float
    f1: float
    roc_auc: float | None
    confusion: list[list[int]]
    labels: list[str]
    classification_report: str
    n_train: int
    n_test: int
    feature_columns: list[str]
    target_column: str


@dataclass
class ModelBundle:
    pipeline: Any
    label_encoder: LabelEncoder
    feature_columns: list[str]
    target_column: str
    dataset_id: str
    flaml_config: dict[str, Any]
    train_report: dict[str, Any]
    trained_at: str
    catalog_id: str = "dengue_elisa_253"

    def predict_labels(self, df: pd.DataFrame) -> np.ndarray:
        x = coerce_abs_columns(df)
        missing = [c for c in self.feature_columns if c not in x.columns]
        if missing:
            raise ValueError(f"Colunas ausentes para predição: {missing}")
        proba_or_pred = self.pipeline.predict(x[self.feature_columns])
        return self.label_encoder.inverse_transform(proba_or_pred)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        x = coerce_abs_columns(df)
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("O estimador treinado não expõe predict_proba.")
        probs = self.pipeline.predict_proba(x[self.feature_columns])
        return pd.DataFrame(probs, columns=list(self.label_encoder.classes_))


def flaml_available() -> tuple[bool, str]:
    try:
        import flaml

        if not getattr(flaml, "has_automl", False):
            return (
                False,
                "AutoML do FLAML indisponível. Instale: pip install \"flaml[automl]\" scikit-learn joblib",
            )
        from flaml import AutoML  # noqa: F401

        return True, ""
    except OSError as exc:
        if "libgomp" in str(exc).lower():
            return (
                False,
                "Biblioteca do sistema **libgomp** ausente (comum em imagens Docker slim). "
                "Reconstrua a imagem: `docker compose build --no-cache` "
                "(o Dockerfile já instala `libgomp1`).",
            )
        return False, f"FLAML indisponível (erro de biblioteca nativa): {exc}"
    except ImportError:
        return False, "Pacote flaml não instalado. Rode: pip install \"flaml[automl]\" scikit-learn joblib"
    except Exception as exc:
        return False, f"FLAML indisponível: {exc}"


def _build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("Nenhuma coluna de feature disponível após a seleção.")
    return ColumnTransformer(transformers=transformers)


def train_flaml_classifier(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    config: FlamlTrainConfig,
    dataset_id: str = "dengue_elisa_253",
) -> tuple[ModelBundle, TrainReport]:
    ok, msg = flaml_available()
    if not ok:
        raise RuntimeError(msg)

    from flaml import AutoML

    x, y_raw = prepare_feature_matrix(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    preprocessor = _build_preprocessor(x_train)
    automl = AutoML()

    fit_kwargs: dict[str, Any] = {
        "X_train": x_train,
        "y_train": y_train,
        "task": "classification",
        "metric": config.metric,
        "time_budget": config.time_budget,
        "estimator_list": list(config.estimator_list),
        "eval_method": config.eval_method,
        "n_splits": config.n_splits,
        "seed": config.seed,
        "verbose": config.verbose,
        "custom_hp": config.custom_hp or None,
    }
    if config.early_stop:
        fit_kwargs["early_stop"] = True

    automl.fit(**fit_kwargs)

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", automl.model.estimator),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    labels = list(label_encoder.classes_)
    report = TrainReport(
        metric=config.metric,
        best_estimator=str(automl.best_estimator),
        best_config=dict(automl.best_config or {}),
        train_seconds=float(getattr(automl, "best_config_train_time", 0) or config.time_budget),
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, average="binary" if len(labels) == 2 else "weighted")),
        roc_auc=_safe_roc_auc(y_test, pipeline, x_test, len(labels)),
        confusion=confusion_matrix(y_test, y_pred).tolist(),
        labels=labels,
        classification_report=classification_report(y_test, y_pred, target_names=labels),
        n_train=len(x_train),
        n_test=len(x_test),
        feature_columns=list(feature_columns),
        target_column=target_column,
    )

    bundle = ModelBundle(
        pipeline=pipeline,
        label_encoder=label_encoder,
        feature_columns=list(feature_columns),
        target_column=target_column,
        dataset_id=dataset_id,
        flaml_config=asdict(config),
        train_report=asdict(report),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )
    return bundle, report


def _safe_roc_auc(y_test: np.ndarray, pipeline: Pipeline, x_test: pd.DataFrame, n_classes: int) -> float | None:
    if n_classes != 2 or not hasattr(pipeline, "predict_proba"):
        return None
    try:
        prob = pipeline.predict_proba(x_test)[:, 1]
        return float(roc_auc_score(y_test, prob))
    except ValueError:
        return None


def save_model_bundle(bundle: ModelBundle, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: Path) -> ModelBundle:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Arquivo .pkl não contém um ModelBundle válido.")
    return bundle


def bundle_metadata_json(bundle: ModelBundle) -> str:
    meta = {
        "dataset_id": bundle.dataset_id,
        "catalog_id": bundle.catalog_id,
        "target_column": bundle.target_column,
        "feature_columns": bundle.feature_columns,
        "trained_at": bundle.trained_at,
        "train_report": bundle.train_report,
    }
    return json.dumps(meta, ensure_ascii=False, indent=2)
