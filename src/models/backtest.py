"""
src/models/backtest.py
──────────────────────
Expanding-window temporal backtester for conflict early warning.

Guarantees:
  1. Strict temporal hygiene: At fold month T, models train ONLY on data
     observed prior to T. No data from month T or future months is ever seen.
  2. Burn-in enforcement: First 12 months (needed for 12-month rolling lags)
     are excluded from training if lags require it.
  3. Consistent multi-model comparison on identical out-of-sample evaluation sets.
  4. Granular risk breakdowns: Northwest pilot states vs national, and
     high-risk coverage bias cells vs normal reporting cells.
  5. Statistical validation: Diebold-Mariano tests on fold loss differentials.
"""

import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from src.models.baseline import NaiveM0, ClimatologicalBaseline
from src.models.glm import NegativeBinomialM1, DEFAULT_GLM_FEATURES
from src.models.gbm import LightGBMM2, DEFAULT_GBM_FEATURES
from src.models.evaluation import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    brier_decomposition,
    diebold_mariano_test,
)

logger = logging.getLogger(__name__)

NORTHWEST_STATES = {"Zamfara", "Katsina", "Sokoto", "Kaduna"}


class ExpandingWindowBacktester:
    """
    Orchestrates expanding-window evaluation across the monthly panel.
    """

    def __init__(
        self,
        min_train_months: int = 36,
        burn_in_months: int = 12,
        glm_features: list[str] | None = None,
        gbm_features: list[str] | None = None,
    ):
        self.min_train_months = min_train_months
        self.burn_in_months = burn_in_months
        self.glm_features = glm_features or DEFAULT_GLM_FEATURES
        self.gbm_features = gbm_features or DEFAULT_GBM_FEATURES

        self.predictions_df_: pd.DataFrame | None = None
        self.metrics_: dict[str, Any] = {}

    def run(
        self,
        feature_matrix: pd.DataFrame,
        target_bin_col: str = "fatal_event_next",
        target_cnt_col: str = "fatalities_next",
    ) -> pd.DataFrame:
        """
        Execute expanding-window backtesting.

        Parameters
        ----------
        feature_matrix : Pre-built feature matrix from Phase 3.
        target_bin_col : Next-month binary fatal event column.
        target_cnt_col : Next-month fatalities count column (created if missing).

        Returns
        -------
        pd.DataFrame containing all out-of-sample predictions, actuals, and metadata.
        """
        df = feature_matrix.copy()
        df = df.sort_values(["admin1", "month"]).reset_index(drop=True)

        # Create count target if not present
        if target_cnt_col not in df.columns and "fatalities" in df.columns:
            df[target_cnt_col] = df.groupby("admin1")["fatalities"].shift(-1)

        # Filter out rows where target is missing (e.g. the very last month in the panel)
        valid = df.dropna(subset=[target_bin_col]).copy()
        all_months = sorted(valid["month"].unique())
        n_months = len(all_months)

        if n_months <= self.min_train_months:
            raise ValueError(
                f"Dataset has {n_months} months, but min_train_months={self.min_train_months}."
            )

        test_months = all_months[self.min_train_months:]
        burn_in_threshold = all_months[self.burn_in_months] if self.burn_in_months < n_months else all_months[0]

        logger.info(
            f"Starting expanding-window backtest: {len(test_months)} folds, "
            f"min_train={self.min_train_months}m, burn_in={self.burn_in_months}m."
        )

        fold_records = []

        for fold_idx, t_month in enumerate(test_months):
            # Training set: months between burn-in and prior to t_month
            train_mask = (valid["month"] < t_month) & (valid["month"] >= burn_in_threshold)
            train_df = valid[train_mask].copy()

            # Test set: records at month t_month predicting outcome at t_month + 1
            test_df = valid[valid["month"] == t_month].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            y_train_bin = train_df[target_bin_col].astype(int).values
            y_train_cnt = train_df[target_cnt_col].astype(float).values
            y_test_bin = test_df[target_bin_col].astype(int).values

            # ── 1. Climatological Baseline ─────────────────────────────────────
            clim_mod = ClimatologicalBaseline().fit(train_df, y_train_bin)
            p_clim = clim_mod.predict_proba(test_df)

            # ── 2. M0 Naive Persistence ────────────────────────────────────────
            m0_mod = NaiveM0().fit(train_df, y_train_bin)
            p_m0 = m0_mod.predict_proba(test_df)

            # ── 3. M1 Negative-Binomial GLM ───────────────────────────────────
            m1_mod = NegativeBinomialM1(features=self.glm_features, alpha=1.0)
            m1_mod.fit(train_df, y_train_cnt)
            p_m1 = m1_mod.predict_proba(test_df)

            # ── 4. M2 LightGBM Classifier ─────────────────────────────────────
            m2_mod = LightGBMM2(
                features=self.gbm_features,
                n_estimators=100,
                max_depth=4,
                num_leaves=12,
                learning_rate=0.03,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            m2_mod.fit(train_df, y_train_bin)
            p_m2 = m2_mod.predict_proba(test_df)

            # Store predictions for this fold
            for i, (_, row) in enumerate(test_df.iterrows()):
                fold_records.append({
                    "fold": fold_idx,
                    "forecast_month": str(t_month),
                    "admin1": row["admin1"],
                    "is_northwest": bool(row["admin1"] in NORTHWEST_STATES),
                    "low_coverage_flag": int(row.get("low_coverage_flag", 0)),
                    "reliability_score": float(row.get("reliability_score", 1.0)),
                    "actual": int(y_test_bin[i]),
                    "p_clim": float(p_clim[i]),
                    "p_m0": float(p_m0[i]),
                    "p_m1": float(p_m1[i]),
                    "p_m2": float(p_m2[i]),
                })

        pred_df = pd.DataFrame(fold_records)
        self.predictions_df_ = pred_df

        # Compute full evaluation suite
        self.metrics_ = self._compute_summary_metrics(pred_df)
        logger.info("Expanding-window backtest complete.")
        return pred_df

    def _compute_summary_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute aggregated metrics, Diebold-Mariano tests, and subgroup scores."""
        y_true = df["actual"].values
        p_clim = df["p_clim"].values
        p_m0   = df["p_m0"].values
        p_m1   = df["p_m1"].values
        p_m2   = df["p_m2"].values

        brier_clim = brier_score(y_true, p_clim)
        brier_m0   = brier_score(y_true, p_m0)
        brier_m1   = brier_score(y_true, p_m1)
        brier_m2   = brier_score(y_true, p_m2)

        bss_m0 = brier_skill_score(y_true, p_m0, brier_clim)
        bss_m1 = brier_skill_score(y_true, p_m1, brier_clim)
        bss_m2 = brier_skill_score(y_true, p_m2, brier_clim)

        auc_m0 = float(roc_auc_score(y_true, p_m0))
        auc_m1 = float(roc_auc_score(y_true, p_m1))
        auc_m2 = float(roc_auc_score(y_true, p_m2))

        ece_m0 = expected_calibration_error(y_true, p_m0, n_bins=10)
        ece_m1 = expected_calibration_error(y_true, p_m1, n_bins=10)
        ece_m2 = expected_calibration_error(y_true, p_m2, n_bins=10)

        decomp_m0 = brier_decomposition(y_true, p_m0, n_bins=10)
        decomp_m1 = brier_decomposition(y_true, p_m1, n_bins=10)
        decomp_m2 = brier_decomposition(y_true, p_m2, n_bins=10)

        # Diebold-Mariano tests aggregated by forecast fold (month)
        monthly = df.groupby("forecast_month").apply(
            lambda g: pd.Series({
                "loss_m0": np.mean((g["actual"] - g["p_m0"]) ** 2),
                "loss_m1": np.mean((g["actual"] - g["p_m1"]) ** 2),
                "loss_m2": np.mean((g["actual"] - g["p_m2"]) ** 2),
            })
        ).reset_index()

        dm_m1_vs_m0 = diebold_mariano_test(monthly["loss_m0"].values - monthly["loss_m1"].values, h=1)
        dm_m2_vs_m0 = diebold_mariano_test(monthly["loss_m0"].values - monthly["loss_m2"].values, h=1)
        dm_m2_vs_m1 = diebold_mariano_test(monthly["loss_m1"].values - monthly["loss_m2"].values, h=1)

        # Subgroup: Northwest Pilot States
        nw = df[df["is_northwest"]]
        nw_metrics = {
            "n_samples": len(nw),
            "prevalence": float(nw["actual"].mean()),
            "brier_clim": round(brier_score(nw["actual"], nw["p_clim"]), 4),
            "brier_m0": round(brier_score(nw["actual"], nw["p_m0"]), 4),
            "brier_m1": round(brier_score(nw["actual"], nw["p_m1"]), 4),
            "brier_m2": round(brier_score(nw["actual"], nw["p_m2"]), 4),
            "auc_m1": round(float(roc_auc_score(nw["actual"], nw["p_m1"])), 4),
            "auc_m2": round(float(roc_auc_score(nw["actual"], nw["p_m2"])), 4),
        }

        # Subgroup: High-Risk Coverage Bias Cells vs Normal
        hr = df[df["low_coverage_flag"] == 1]
        norm = df[df["low_coverage_flag"] == 0]

        coverage_breakdown = {
            "high_risk_cells": {
                "n_samples": len(hr),
                "brier_m0": round(brier_score(hr["actual"], hr["p_m0"]), 4) if len(hr) > 0 else None,
                "brier_m1": round(brier_score(hr["actual"], hr["p_m1"]), 4) if len(hr) > 0 else None,
                "brier_m2": round(brier_score(hr["actual"], hr["p_m2"]), 4) if len(hr) > 0 else None,
            },
            "normal_cells": {
                "n_samples": len(norm),
                "brier_m0": round(brier_score(norm["actual"], norm["p_m0"]), 4),
                "brier_m1": round(brier_score(norm["actual"], norm["p_m1"]), 4),
                "brier_m2": round(brier_score(norm["actual"], norm["p_m2"]), 4),
            }
        }

        return {
            "folds_evaluated": int(df["forecast_month"].nunique()),
            "total_predictions": len(df),
            "target_prevalence": round(float(np.mean(y_true)), 4),
            "models": {
                "climatology": {
                    "brier_score": round(brier_clim, 4),
                },
                "M0_naive": {
                    "brier_score": round(brier_m0, 4),
                    "brier_skill_score": round(bss_m0, 4),
                    "auc_roc": round(auc_m0, 4),
                    "ece": round(ece_m0, 4),
                    "decomposition": decomp_m0,
                },
                "M1_negbin_glm": {
                    "brier_score": round(brier_m1, 4),
                    "brier_skill_score": round(bss_m1, 4),
                    "auc_roc": round(auc_m1, 4),
                    "ece": round(ece_m1, 4),
                    "decomposition": decomp_m1,
                    "dm_test_vs_m0": dm_m1_vs_m0,
                },
                "M2_lightgbm": {
                    "brier_score": round(brier_m2, 4),
                    "brier_skill_score": round(bss_m2, 4),
                    "auc_roc": round(auc_m2, 4),
                    "ece": round(ece_m2, 4),
                    "decomposition": decomp_m2,
                    "dm_test_vs_m0": dm_m2_vs_m0,
                    "dm_test_vs_m1": dm_m2_vs_m1,
                },
            },
            "subgroups": {
                "northwest_pilot": nw_metrics,
                "coverage_bias": coverage_breakdown,
            }
        }
