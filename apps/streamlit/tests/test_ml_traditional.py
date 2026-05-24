"""Testes do módulo ML tradicional (datasets e catálogo)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.datasets import (
    default_feature_columns,
    load_dengue_elisa_253,
    normalize_column_name,
    prepare_feature_matrix,
)
from ml.dictionary import load_dataset_catalog
from ml.paths import resolve_dengue_results_dir
from ml.training import (
    FlamlTrainConfig,
    ModelBundle,
    flaml_available,
    save_model_bundle,
    load_model_bundle,
)


class TestMlCatalog(unittest.TestCase):
    def test_load_catalog_has_target_and_drops(self) -> None:
        catalog = load_dataset_catalog()
        self.assertEqual(catalog.dataset_id, "dengue_elisa_253")
        self.assertIn("Nome completo", catalog.suggested_drop)
        self.assertTrue(catalog.default_target)

    def test_normalize_column_name(self) -> None:
        a = normalize_column_name("ABS Tempo sensibilização 1h")
        b = normalize_column_name("ABS Tempo sensibilizacao 1h")
        self.assertEqual(a, b)

    def test_flaml_libgomp_oserror_message(self) -> None:
        import builtins
        from unittest.mock import patch

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "flaml" or name.startswith("flaml."):
                raise OSError("libgomp.so.1: cannot open shared object file")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            ok, msg = flaml_available()
        self.assertFalse(ok)
        self.assertIn("libgomp", msg.lower())


class TestMlDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results_dir = resolve_dengue_results_dir()
        if not (cls.results_dir / "amostras_dengue.xlsx").is_file():
            raise unittest.SkipTest(f"Dados Dengue não encontrados em {cls.results_dir}")

    def test_load_merged_shape(self) -> None:
        df, catalog = load_dengue_elisa_253(self.results_dir)
        self.assertEqual(len(df), 320)
        self.assertIn(catalog.default_target, df.columns)
        self.assertIn(catalog.merge_key, df.columns)

    def test_default_features_exclude_pii(self) -> None:
        df, catalog = load_dengue_elisa_253(self.results_dir)
        features = default_feature_columns(df, catalog)
        self.assertNotIn("Nome completo", features)
        self.assertNotIn(catalog.default_target, features)

    def test_prepare_feature_matrix(self) -> None:
        df, catalog = load_dengue_elisa_253(self.results_dir)
        features = default_feature_columns(df, catalog)[:3]
        x, y = prepare_feature_matrix(
            df,
            feature_columns=features,
            target_column=catalog.default_target,
        )
        self.assertEqual(len(x), len(y))
        self.assertEqual(list(x.columns), features)


@unittest.skipUnless(flaml_available()[0], "flaml não instalado")
class TestMlTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results_dir = resolve_dengue_results_dir()
        if not (cls.results_dir / "amostras_dengue.xlsx").is_file():
            raise unittest.SkipTest(f"Dados Dengue não encontrados em {cls.results_dir}")

    def test_short_train_and_pkl_roundtrip(self) -> None:
        df, catalog = load_dengue_elisa_253(self.results_dir)
        features = default_feature_columns(df, catalog)[:4]
        config = FlamlTrainConfig(time_budget=5, n_splits=2, estimator_list=("rf",))
        bundle, report = __import__(
            "ml.training", fromlist=["train_flaml_classifier"]
        ).train_flaml_classifier(
            df,
            feature_columns=features,
            target_column=catalog.default_target,
            config=config,
        )
        self.assertGreater(report.f1, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_model.pkl"
            save_model_bundle(bundle, path)
            loaded = load_model_bundle(path)
            self.assertIsInstance(loaded, ModelBundle)
            preds = loaded.predict_labels(df.head(5))
            self.assertEqual(len(preds), 5)


if __name__ == "__main__":
    unittest.main()
