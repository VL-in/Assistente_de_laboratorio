"""Testes do módulo ML tradicional (AbRank Kaggle + catálogo)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ml.datasets import (
    default_feature_columns,
    load_dataset_from_catalog,
    normalize_column_name,
    prepare_feature_matrix,
)
from ml.dictionary import load_dataset_catalog, list_catalog_ids
from ml.kaggle_sources import KAGGLE_ABRANK_HANDLE, load_abrank_split
from ml.training import (
    FlamlTrainConfig,
    ModelBundle,
    build_estimator_summaries,
    flaml_available,
    flaml_loss_to_metric_value,
    load_model_bundle,
    save_model_bundle,
    train_flaml_model,
)


class _FakeSearchState:
    def __init__(self, best_loss: float, sample_size: int) -> None:
        self.best_loss = best_loss
        self.sample_size = sample_size
        self.best_config_sample_size = sample_size


class _FakeAutoML:
    def __init__(self) -> None:
        self.best_estimator = "rf"
        self.best_loss_per_estimator = {"rf": 0.1, "extra_tree": 0.25}
        self._iter_per_learner = {"rf": 12, "extra_tree": 5}
        self._search_states = {
            "rf": _FakeSearchState(0.1, 800),
            "extra_tree": _FakeSearchState(0.25, 800),
        }


class TestFlamlEstimatorSummaries(unittest.TestCase):
    def test_loss_to_r2(self) -> None:
        self.assertAlmostEqual(flaml_loss_to_metric_value("r2", 0.2), 0.8)

    def test_build_summaries_marks_best(self) -> None:
        rows = build_estimator_summaries(
            _FakeAutoML(),
            metric="r2",
            requested_estimators=("rf", "extra_tree"),
            best_estimator="rf",
            default_n_samples=800,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].is_best)
        self.assertEqual(rows[0].estimator, "rf")
        self.assertAlmostEqual(rows[0].cv_score, 0.9)
        self.assertEqual(rows[0].n_trials, 12)
        self.assertEqual(rows[0].n_samples, 800)


class TestMlCatalog(unittest.TestCase):
    def test_list_catalogs_abrank_only(self) -> None:
        self.assertEqual(list_catalog_ids(), ["abrank_kaggle"])

    def test_abrank_catalog_defaults(self) -> None:
        catalog = load_dataset_catalog("abrank_kaggle")
        self.assertEqual(catalog.kaggle_handle, KAGGLE_ABRANK_HANDLE)
        self.assertEqual(catalog.ml_task, "regression")
        self.assertEqual(catalog.default_target, "log_Aff")
        self.assertIn("Ab_name", catalog.suggested_drop)

    def test_normalize_column_name(self) -> None:
        a = normalize_column_name("log_Aff")
        b = normalize_column_name("log_aff")
        self.assertEqual(a, b)

    def test_flaml_libgomp_oserror_message(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "flaml" or name.startswith("flaml."):
                raise OSError("libgomp.so.1: cannot open shared object file")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            ok, msg = flaml_available()
        self.assertFalse(ok)
        self.assertIn("libgomp", msg.lower())


class TestMlAbrankDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.df = load_abrank_split(max_rows=2000)
        except Exception as exc:
            raise unittest.SkipTest(f"AbRank indisponível (Kaggle/cache): {exc}") from exc
        cls.catalog = load_dataset_catalog("abrank_kaggle")

    def test_load_via_catalog(self) -> None:
        df, catalog = load_dataset_from_catalog(self.catalog, max_rows=500)
        self.assertGreater(len(df), 100)
        self.assertIn(catalog.default_target, df.columns)

    def test_default_features_exclude_names(self) -> None:
        features = default_feature_columns(self.df, self.catalog)
        self.assertNotIn("Ab_name", features)
        self.assertNotIn(self.catalog.default_target, features)
        self.assertNotIn("Ag_epitope_restrictions", features)
        self.assertNotIn("IC50 [ug/mL]", features)
        self.assertIn("Agtype", features)

    def test_catalog_excluded_columns(self) -> None:
        excluded = self.catalog.columns_excluded_from_features()
        self.assertIn("Ab_name", excluded)
        self.assertIn("log_Aff", excluded)

    def test_prepare_regression_matrix(self) -> None:
        features = default_feature_columns(self.df, self.catalog)[:4]
        x, y = prepare_feature_matrix(
            self.df,
            feature_columns=features,
            target_column="log_Aff",
        )
        self.assertEqual(len(x), len(y))
        self.assertTrue(pd.api.types.is_numeric_dtype(y) or y.dtype == float)

    def test_prepare_drops_nan_numeric_target(self) -> None:
        df = self.df.head(100).copy()
        df.loc[df.index[:5], "log_Aff"] = float("nan")
        features = default_feature_columns(df, self.catalog)[:3]
        x, y = prepare_feature_matrix(
            df,
            feature_columns=features,
            target_column="log_Aff",
        )
        self.assertFalse(y.isna().any())
        self.assertLess(len(x), 100)

    def test_prepare_keeps_classification_labels(self) -> None:
        df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, 3.0, 4.0],
                "label": ["Positivo", "Negativo", "Positivo", "Negativo"],
            }
        )
        x, y = prepare_feature_matrix(
            df,
            feature_columns=["feat_a"],
            target_column="label",
            regression_target=False,
        )
        self.assertEqual(len(x), 4)
        self.assertEqual(list(y.astype(str)), ["Positivo", "Negativo", "Positivo", "Negativo"])


@unittest.skipUnless(flaml_available()[0], "flaml não instalado")
class TestMlTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.df = load_abrank_split(max_rows=3000)
        except Exception as exc:
            raise unittest.SkipTest(f"AbRank indisponível: {exc}") from exc
        cls.catalog = load_dataset_catalog("abrank_kaggle")

    def test_regression_train_and_pkl(self) -> None:
        features = default_feature_columns(self.df, self.catalog)[:5]
        config = FlamlTrainConfig(
            task="regression",
            metric="r2",
            time_budget=5,
            n_splits=2,
            estimator_list=("rf",),
        )
        bundle, report = train_flaml_model(
            self.df,
            feature_columns=features,
            target_column="log_Aff",
            config=config,
            dataset_id="abrank_kaggle",
        )
        self.assertEqual(report.task, "regression")
        self.assertIsNotNone(report.r2)
        self.assertGreater(report.n_total, report.n_train)
        self.assertTrue(report.estimator_summaries)
        self.assertTrue(any(s.is_best for s in report.estimator_summaries))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "abrank_model.pkl"
            save_model_bundle(bundle, path)
            loaded = load_model_bundle(path)
            self.assertEqual(loaded.task, "regression")
            preds = loaded.predict_labels(self.df.head(5))
            self.assertEqual(len(preds), 5)


if __name__ == "__main__":
    unittest.main()
