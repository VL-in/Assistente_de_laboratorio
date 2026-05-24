"""Testes da inferência ML via chat."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.chat_infer import (  # noqa: E402
    MlInferResult,
    _parse_extract_json,
    _rows_to_dataframe,
    run_chat_ml_inference,
)
from ml.training import ModelBundle, save_model_bundle  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


class ParseExtractJsonTests(unittest.TestCase):
    def test_valid_rows(self) -> None:
        raw = '{"rows": [{"Agtype": "HER2", "escape": 0.1}], "error": null}'
        rows, err = _parse_extract_json(raw)
        self.assertIsNone(err)
        self.assertEqual(len(rows), 1)

    def test_error_field(self) -> None:
        raw = '{"rows": [], "error": "faltam features"}'
        rows, err = _parse_extract_json(raw)
        self.assertEqual(err, "faltam features")
        self.assertEqual(rows, [])


class RunChatMlInferenceTests(unittest.TestCase):
    def _minimal_bundle(self) -> ModelBundle:
        pipe = Pipeline([("model", DummyRegressor(strategy="constant", constant=1.5))])
        pipe.fit(pd.DataFrame({"escape": [0.0, 1.0]}), [1.0, 2.0])
        return ModelBundle(
            pipeline=pipe,
            label_encoder=None,
            feature_columns=["escape"],
            target_column="log_Aff",
            dataset_id="abrank_kaggle",
            task="regression",
            flaml_config={},
            train_report={},
            trained_at="2026-05-24T00:00:00Z",
            catalog_id="abrank_kaggle",
        )

    def test_rows_to_dataframe_fills_missing_features(self) -> None:
        bundle = self._minimal_bundle()
        df = _rows_to_dataframe([{"escape": 0.1}], bundle)
        self.assertIn("escape", df.columns)

    @patch("ml.chat_infer.extract_prediction_rows")
    def test_run_inference_success(self, mock_extract: MagicMock) -> None:
        bundle = self._minimal_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "modelo.pkl"
            save_model_bundle(bundle, path)
            mock_extract.return_value = (
                [{"escape": 0.0}],
                "{}",
                None,
            )
            result = run_chat_ml_inference(
                "Preveja a afinidade",
                client=MagicMock(),
                model="test",
                bundle=bundle,
                model_path=path,
            )
        self.assertIsInstance(result, MlInferResult)
        self.assertTrue(result.ok)
        self.assertIn("predicao_log_aff", result.predictions.columns)


if __name__ == "__main__":
    unittest.main()
