"""
tests/test_models.py
────────────────────
Unit tests for models, evaluation metrics, and backtesting framework.
Runs with standard unittest and pytest.
"""

import unittest
import numpy as np
import pandas as pd

from src.models.baseline import NaiveM0, ClimatologicalBaseline
from src.models.glm import NegativeBinomialM1
from src.models.gbm import LightGBMM2
from src.models.evaluation import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    brier_decomposition,
    diebold_mariano_test,
)


class TestModelsAndEvaluation(unittest.TestCase):

    def test_brier_and_decomposition(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7])

        bs = brier_score(y_true, y_prob)
        self.assertTrue(0.0 <= bs <= 1.0)

        bss = brier_skill_score(y_true, y_prob)
        self.assertTrue(bss > 0.0)

        decomp = brier_decomposition(y_true, y_prob)
        self.assertIn("reliability", decomp)
        self.assertIn("resolution", decomp)
        self.assertIn("uncertainty", decomp)
        calculated_bs = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
        self.assertTrue(np.isclose(decomp["brier"], calculated_bs, atol=0.05))

    def test_expected_calibration_error(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        self.assertTrue(np.isclose(ece, 0.0))

        y_prob_bad = np.array([0.0, 0.0, 1.0, 1.0])
        ece_bad = expected_calibration_error(y_true, y_prob_bad, n_bins=5)
        self.assertTrue(ece_bad > 0.5)

    def test_diebold_mariano_test(self):
        np.random.seed(42)
        loss_m1 = np.random.normal(loc=0.35, scale=0.02, size=50)
        loss_m2 = np.random.normal(loc=0.18, scale=0.02, size=50)
        d_seq = loss_m1 - loss_m2

        res = diebold_mariano_test(d_seq, h=1)
        self.assertTrue(res["significant"])
        self.assertEqual(res["superior_model"], "model_2")
        self.assertLess(res["p_value"], 0.001)

        res_null = diebold_mariano_test(np.zeros(50), h=1)
        self.assertFalse(res_null["significant"])
        self.assertEqual(res_null["superior_model"], "identical")

    def test_naive_m0(self):
        X = pd.DataFrame({"fatal_event": [1, 0, 1, 0]})
        y = np.array([1, 0, 1, 0])
        m0 = NaiveM0()
        m0.fit(X, y)
        probs = m0.predict_proba(X)
        self.assertTrue(np.array_equal(probs, np.array([1.0, 0.0, 1.0, 0.0])))
        preds = m0.predict(X)
        self.assertTrue(np.array_equal(preds, y))

    def test_negative_binomial_m1(self):
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            "fatalities_lag1": np.random.poisson(lam=5, size=n),
            "fatalities_roll3m_mean": np.random.uniform(0, 10, size=n),
            "fatalities_roll6m_mean": np.random.uniform(0, 10, size=n),
            "spatial_fatalities_lag1": np.random.uniform(0, 10, size=n),
            "coverage_lag1": np.random.uniform(0.3, 0.9, size=n),
            "month_sin": np.sin(np.linspace(0, 2*np.pi, n)),
            "month_cos": np.cos(np.linspace(0, 2*np.pi, n)),
        })
        y_cnt = np.random.negative_binomial(n=2, p=0.3, size=n)

        m1 = NegativeBinomialM1()
        m1.fit(X, y_cnt)

        probs = m1.predict_proba(X)
        self.assertEqual(len(probs), n)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        counts = m1.predict_count(X)
        self.assertEqual(len(counts), n)
        self.assertTrue(np.all(counts >= 0.0))

    def test_lightgbm_m2(self):
        np.random.seed(42)
        n = 100
        features = [f"feat_{i}" for i in range(5)]
        X = pd.DataFrame(np.random.randn(n, 5), columns=features)
        y_bin = (X["feat_0"] + X["feat_1"] > 0).astype(int)

        m2 = LightGBMM2(features=features, n_estimators=10, max_depth=3)
        m2.fit(X, y_bin)

        probs = m2.predict_proba(X)
        self.assertEqual(len(probs), n)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        imp_df = m2.get_feature_importance_df()
        self.assertEqual(len(imp_df), 5)
        self.assertIn("feature", imp_df.columns)
        self.assertIn("importance", imp_df.columns)


if __name__ == "__main__":
    unittest.main()
