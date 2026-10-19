"""
src/models/gbm.py
─────────────────
M2: Gradient Boosted Decision Tree (LightGBM) for conflict early warning.

Advantages:
  - Captures complex non-linear relationships and multi-way interaction effects
    (e.g. low coverage + high spatial lag + dry season).
  - Handles correlated features and missing values natively.
  - Produces interpretable feature importances (gain & split).
  - Can be post-calibrated with Platt scaling or Isotonic regression if required.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

logger = logging.getLogger(__name__)

# Full feature set used by LightGBM
DEFAULT_GBM_FEATURES = [
    # Group A: Temporal lags
    "events_lag1", "events_lag2", "events_lag3", "events_lag6", "events_lag12",
    "fatalities_lag1", "fatalities_lag2", "fatalities_lag3", "fatalities_lag6", "fatalities_lag12",
    "fatal_event_lag1", "fatal_event_lag2", "fatal_event_lag3", "fatal_event_lag6", "fatal_event_lag12",
    # Group B: Rolling windows
    "events_roll3m_sum", "events_roll3m_mean", "events_roll6m_sum", "events_roll6m_mean",
    "events_roll12m_sum", "events_roll12m_mean",
    "fatalities_roll3m_sum", "fatalities_roll3m_mean", "fatalities_roll6m_sum", "fatalities_roll6m_mean",
    "fatalities_roll12m_sum", "fatalities_roll12m_mean",
    # Group C: Spatial lags
    "spatial_events_lag1", "spatial_fatalities_lag1",
    # Group D: Seasonality & calendar
    "month_sin", "month_cos", "month_of_year", "year",
    "is_harmattan", "is_farming", "is_post_harvest",
    # Group E: Islamic calendar
    "is_ramadan", "is_post_ramadan",
    # Group F: Coverage proxy & reliability
    "reliability_score", "log_reliability", "low_coverage_flag", "coverage_lag1",
    # Group G: Source disagreement
    "source_disagree_lag1", "abs_diff_fat_lag1",
    # Geographic location
    "latitude", "longitude"
]


class LightGBMM2(BaseEstimator, ClassifierMixin):
    """
    LightGBM Gradient Boosted Decision Tree (M2).
    """

    def __init__(
        self,
        features: list[str] | None = None,
        n_estimators: int = 100,
        learning_rate: float = 0.03,
        num_leaves: int = 12,
        max_depth: int = 4,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        calibrate: bool = False,
    ):
        self.features = features or DEFAULT_GBM_FEATURES
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.calibrate = calibrate

        self.clf_ = None
        self.feature_names_in_ = None
        self.feature_importances_ = None
        self.baseline_prevalence_ = 0.5

    def _prepare_X(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.features if c in X.columns]
            if len(cols) == 0:
                raise ValueError("None of the specified features found in DataFrame.")
            return X[cols].fillna(0.0).values.astype(float), cols
        else:
            X_arr = np.asarray(X, dtype=float)
            cols = [f"f_{i}" for i in range(X_arr.shape[1] if X_arr.ndim > 1 else 1)]
            return np.nan_to_num(X_arr, nan=0.0), cols

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series):
        """Fit the gradient boosting model."""
        y_arr = np.asarray(y, dtype=int)
        self.baseline_prevalence_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5

        X_mat, cols = self._prepare_X(X)
        self.feature_names_in_ = cols

        base_lgb = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True
        )

        if self.calibrate:
            self.clf_ = CalibratedClassifierCV(base_lgb, cv=3, method="sigmoid")
            self.clf_.fit(X_mat, y_arr)
            # Retrieve importance from underlying estimator
            try:
                raw_lgb = self.clf_.calibrated_classifiers_[0].estimator
                self.feature_importances_ = raw_lgb.feature_importances_
            except Exception:
                self.feature_importances_ = np.ones(len(cols)) / len(cols)
        else:
            self.clf_ = base_lgb
            self.clf_.fit(X_mat, y_arr)
            self.feature_importances_ = self.clf_.feature_importances_

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return 1D array of probabilities P(fatal_event_{T+1} >= 1)."""
        if self.clf_ is None:
            return np.full(len(X), fill_value=self.baseline_prevalence_)

        X_mat, _ = self._prepare_X(X)
        probs_2d = self.clf_.predict_proba(X_mat)
        if probs_2d.shape[1] > 1:
            return probs_2d[:, 1]
        return probs_2d[:, 0]

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return binary prediction at 0.5 threshold."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_feature_importance_df(self) -> pd.DataFrame:
        """Return sorted DataFrame of feature importances."""
        if self.feature_importances_ is None or self.feature_names_in_ is None:
            return pd.DataFrame()
        df = pd.DataFrame({
            "feature": self.feature_names_in_,
            "importance": self.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df
