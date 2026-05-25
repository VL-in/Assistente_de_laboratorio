"""
Treino AutoML com FLAML (instalação mínima: estimadores baseados em scikit-learn).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

TrainProgressFn = Callable[[str, float | None], None]

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
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ml.datasets import coerce_abs_columns, prepare_feature_matrix
from ml.sequence_embeddings import (
    SequenceEmbeddingConfig,
    SequenceEmbeddingTransformer,
    esm_available,
    sequence_column_names,
)

# Estimadores disponíveis sem extras pip (lightgbm/xgboost/catboost).
MINIMAL_ESTIMATOR_LIST = ("rf", "extra_tree", "lrl1")

ESTIMATOR_LABELS: dict[str, str] = {
    "rf": "Random Forest (rf)",
    "extra_tree": "Extra Trees (extra_tree)",
    "lrl1": "Regressão logística L1 (lrl1)",
}

# Métricas em que o FLAML minimiza (1 - score); exibir como score “maior é melhor”.
_HIGHER_IS_BETTER_METRICS = frozenset(
    {
        "r2",
        "accuracy",
        "roc_auc",
        "roc_auc_ovr",
        "roc_auc_ovo",
        "roc_auc_weighted",
        "roc_auc_ovo_weighted",
        "roc_auc_ovr_weighted",
        "f1",
        "micro_f1",
        "macro_f1",
        "ap",
    }
)


@dataclass
class FlamlTrainConfig:
    task: str = "classification"
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

    def resolved_metric(self) -> str:
        if self.metric:
            return self.metric
        return "r2" if self.task == "regression" else "f1"


@dataclass
class EstimatorSummary:
    """Resumo de um estimador avaliado pelo FLAML na busca AutoML."""

    estimator: str
    label: str
    n_trials: int
    n_samples: int
    best_val_loss: float | None
    cv_score: float | None
    is_best: bool


@dataclass
class TrainReport:
    task: str
    metric: str
    best_estimator: str
    best_config: dict[str, Any]
    train_seconds: float
    accuracy: float | None
    f1: float | None
    r2: float | None
    rmse: float | None
    roc_auc: float | None
    confusion: list[list[int]] | None
    labels: list[str]
    classification_report: str
    n_total: int
    n_train: int
    n_test: int
    feature_columns: list[str]
    target_column: str
    estimator_summaries: list[EstimatorSummary] = field(default_factory=list)
    best_cv_score: float | None = None


@dataclass
class ModelBundle:
    pipeline: Any
    label_encoder: LabelEncoder | None
    feature_columns: list[str]
    target_column: str
    dataset_id: str
    task: str
    flaml_config: dict[str, Any]
    train_report: dict[str, Any]
    trained_at: str
    catalog_id: str = "abrank_kaggle"
    sequence_transformer: SequenceEmbeddingTransformer | None = None

    def _prepare_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica embedding+PCA de sequências quando o bundle foi treinado com ESM-2."""
        work = coerce_abs_columns(df.copy())
        if self.sequence_transformer is None:
            return work
        for col in self.sequence_transformer.config.sequence_columns:
            if col not in work.columns:
                work[col] = pd.NA
        return self.sequence_transformer.transform(work)

    def predict_labels(self, df: pd.DataFrame) -> np.ndarray:
        work = self._prepare_feature_frame(df)
        missing = [c for c in self.feature_columns if c not in work.columns]
        if missing:
            raise ValueError(f"Colunas ausentes para predição: {missing}")
        preds = self.pipeline.predict(work[self.feature_columns])
        if self.task == "regression" or self.label_encoder is None:
            return np.asarray(preds)
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.task == "regression":
            raise AttributeError("Predição de probabilidade não se aplica a regressão.")
        work = self._prepare_feature_frame(df)
        if self.label_encoder is None:
            raise AttributeError("Modelo sem codificador de classes.")
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("O estimador treinado não expõe predict_proba.")
        probs = self.pipeline.predict_proba(work[self.feature_columns])
        return pd.DataFrame(probs, columns=list(self.label_encoder.classes_))


def flaml_loss_to_metric_value(metric: str, loss: float) -> float:
    """Converte a perda de validação do FLAML para o valor da métrica escolhida."""
    metric_key = (metric or "").lower()
    if metric_key in _HIGHER_IS_BETTER_METRICS:
        return 1.0 - float(loss)
    return float(loss)


def estimator_display_label(estimator: str) -> str:
    return ESTIMATOR_LABELS.get(estimator, estimator)


def build_estimator_summaries(
    automl: Any,
    *,
    metric: str,
    requested_estimators: tuple[str, ...] | list[str],
    best_estimator: str,
    default_n_samples: int,
) -> list[EstimatorSummary]:
    """Agrega tentativas, amostras e melhor score de validação por estimador FLAML."""
    losses: dict[str, float] = dict(getattr(automl, "best_loss_per_estimator", {}) or {})
    trials: dict[str, int] = dict(getattr(automl, "_iter_per_learner", {}) or {})
    search_states: dict[str, Any] = dict(getattr(automl, "_search_states", {}) or {})

    summaries: list[EstimatorSummary] = []
    for estimator in requested_estimators:
        state = search_states.get(estimator)
        n_samples = default_n_samples
        if state is not None:
            n_samples = int(
                getattr(state, "best_config_sample_size", None)
                or getattr(state, "sample_size", None)
                or default_n_samples
            )
        loss = losses.get(estimator)
        cv_score = flaml_loss_to_metric_value(metric, loss) if loss is not None else None
        summaries.append(
            EstimatorSummary(
                estimator=estimator,
                label=estimator_display_label(estimator),
                n_trials=int(trials.get(estimator, 0)),
                n_samples=n_samples,
                best_val_loss=float(loss) if loss is not None else None,
                cv_score=cv_score,
                is_best=estimator == best_estimator,
            )
        )

    summaries.sort(
        key=lambda row: (
            row.cv_score is None,
            -(row.cv_score if row.cv_score is not None else float("-inf")),
        )
    )
    return summaries


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


def train_flaml_model(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    config: FlamlTrainConfig,
    dataset_id: str = "abrank_kaggle",
    catalog_id: str = "abrank_kaggle",
    use_sequence_embeddings: bool = False,
    embedding_config: SequenceEmbeddingConfig | None = None,
    progress: TrainProgressFn | None = None,
) -> tuple[ModelBundle, TrainReport]:
    """Treina um modelo FLAML (classificação ou regressão conforme ``config.task``)."""
    ok, msg = flaml_available()
    if not ok:
        raise RuntimeError(msg)

    if progress is not None:
        progress("Preparando dados para o treino…", 0.03)

    from flaml import AutoML

    task = config.task if config.task in ("classification", "regression") else "classification"
    metric = config.resolved_metric()

    seq_cols = set(sequence_column_names())
    tabular_features = [c for c in feature_columns if c not in seq_cols]
    sequence_transformer: SequenceEmbeddingTransformer | None = None

    x, y_raw = prepare_feature_matrix(
        df,
        feature_columns=tabular_features,
        target_column=target_column,
        regression_target=(task == "regression"),
    )

    if use_sequence_embeddings:
        present_seq = [c for c in sequence_column_names() if c in df.columns]
        if not present_seq:
            raise ValueError(
                "Embeddings de sequência solicitados, mas o dataset não contém "
                f"{list(sequence_column_names())}."
            )
        ok_esm, esm_msg = esm_available()
        if not ok_esm:
            raise RuntimeError(esm_msg)

    label_encoder: LabelEncoder | None = None
    if task == "regression":
        y = pd.to_numeric(y_raw, errors="coerce")
        mask = y.notna()
        x = x.loc[mask]
        y = y.loc[mask].astype(float)
        stratify = None
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        stratify = y if len(np.unique(y)) > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=stratify,
    )

    final_feature_columns = list(tabular_features)

    if use_sequence_embeddings:
        if progress is not None:
            progress(
                "ESM-2: carregando modelo (primeira execução pode baixar ~30 MB)…",
                0.02,
            )
        sequence_transformer = SequenceEmbeddingTransformer(embedding_config)
        train_df = df.loc[x_train.index]
        sequence_transformer.fit(train_df, progress=progress)
        pca_train = sequence_transformer.training_pca_frame()
        sequence_transformer.clear_training_cache()
        pca_test = sequence_transformer.transform_pca_only(
            df.loc[x_test.index],
            progress=progress,
        )
        if progress is not None:
            progress("ESM-2: embeddings concluídos. Iniciando FLAML…", 0.62)
        x_train = pd.concat([x_train.reset_index(drop=True), pca_train.reset_index(drop=True)], axis=1)
        x_test = pd.concat([x_test.reset_index(drop=True), pca_test.reset_index(drop=True)], axis=1)
        final_feature_columns = tabular_features + sequence_transformer.pca_columns_

    if progress is not None:
        progress(
            f"FLAML: buscando melhor modelo (até {config.time_budget}s, métrica={metric})…",
            0.68,
        )
    preprocessor = _build_preprocessor(x_train)
    x_train_pp = preprocessor.fit_transform(x_train)
    x_test_pp = preprocessor.transform(x_test)

    automl = AutoML()

    fit_kwargs: dict[str, Any] = {
        "X_train": x_train_pp,
        "y_train": y_train,
        "task": task,
        "metric": metric,
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
    fit_kwargs["keep_search_state"] = True

    automl.fit(**fit_kwargs)

    if progress is not None:
        progress("FLAML: montando pipeline final e avaliando no teste…", 0.92)

    best_estimator = str(automl.best_estimator)
    estimator_summaries = build_estimator_summaries(
        automl,
        metric=metric,
        requested_estimators=config.estimator_list,
        best_estimator=best_estimator,
        default_n_samples=len(x_train),
    )
    best_cv_score: float | None = None
    for row in estimator_summaries:
        if row.is_best:
            best_cv_score = row.cv_score
            break

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", automl.model.estimator),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    if task == "regression":
        report = TrainReport(
            task=task,
            metric=metric,
            best_estimator=best_estimator,
            best_config=dict(automl.best_config or {}),
            train_seconds=float(getattr(automl, "best_config_train_time", 0) or config.time_budget),
            accuracy=None,
            f1=None,
            r2=float(r2_score(y_test, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            roc_auc=None,
            confusion=None,
            labels=[],
            classification_report="",
            n_total=len(x),
            n_train=len(x_train),
            n_test=len(x_test),
            feature_columns=list(final_feature_columns),
            target_column=target_column,
            estimator_summaries=estimator_summaries,
            best_cv_score=best_cv_score,
        )
    else:
        labels = list(label_encoder.classes_) if label_encoder else []
        report = TrainReport(
            task=task,
            metric=metric,
            best_estimator=best_estimator,
            best_config=dict(automl.best_config or {}),
            train_seconds=float(getattr(automl, "best_config_train_time", 0) or config.time_budget),
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1=float(
                f1_score(y_test, y_pred, average="binary" if len(labels) == 2 else "weighted")
            ),
            r2=None,
            rmse=None,
            roc_auc=_safe_roc_auc(y_test, pipeline, x_test, len(labels)),
            confusion=confusion_matrix(y_test, y_pred).tolist(),
            labels=labels,
            classification_report=classification_report(
                y_test, y_pred, target_names=labels
            ),
            n_total=len(x),
            n_train=len(x_train),
            n_test=len(x_test),
            feature_columns=list(final_feature_columns),
            target_column=target_column,
            estimator_summaries=estimator_summaries,
            best_cv_score=best_cv_score,
        )

    if sequence_transformer is not None:
        sequence_transformer.clear_training_cache()

    bundle = ModelBundle(
        pipeline=pipeline,
        label_encoder=label_encoder,
        feature_columns=list(final_feature_columns),
        target_column=target_column,
        dataset_id=dataset_id,
        task=task,
        flaml_config=asdict(config),
        train_report=asdict(report),
        trained_at=datetime.now(timezone.utc).isoformat(),
        catalog_id=catalog_id,
        sequence_transformer=sequence_transformer,
    )
    if progress is not None:
        progress("Treino concluído.", 1.0)
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
    if not getattr(bundle, "task", None):
        bundle.task = "classification"
    if not getattr(bundle, "catalog_id", None):
        bundle.catalog_id = bundle.dataset_id
    if not hasattr(bundle, "sequence_transformer"):
        bundle.sequence_transformer = None
    return bundle


def bundle_metadata_json(bundle: ModelBundle) -> str:
    meta = {
        "dataset_id": bundle.dataset_id,
        "catalog_id": bundle.catalog_id,
        "target_column": bundle.target_column,
        "feature_columns": bundle.feature_columns,
        "trained_at": bundle.trained_at,
        "train_report": bundle.train_report,
        "sequence_embeddings": bundle.sequence_transformer is not None,
        "sequence_columns": (
            list(bundle.sequence_transformer.config.sequence_columns)
            if bundle.sequence_transformer is not None
            else []
        ),
        "seq_pca_columns": (
            list(bundle.sequence_transformer.pca_columns_)
            if bundle.sequence_transformer is not None
            else []
        ),
    }
    return json.dumps(meta, ensure_ascii=False, indent=2)
