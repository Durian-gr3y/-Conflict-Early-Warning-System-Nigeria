"""
src/models/baseline.py
───────────────────────
Baseline forecast models for Conflict Early Warning Nigeria:
  - M0: Naive persistence baseline (last observed month = forecast for next month)
  - Climatological baseline: predicts empirical prevalence (long-run mean)
"""

import numpy as np
import pandas as pd


class NaiveM0:
    """
    Naive Persistence Model (M0).

    Operational logic:
      "What happened last month will happen next month."
      P(fatal_event_{T+1} >= 1 | State s) = 1.0 if fatal_event_{T} >= 1 else 0.0.

    When soft probabilities are needed (e.g. for log-loss), an optional smoothing
    parameter epsilon can be provided (default: 0.0, strict 0/1 persistence).
    """

    def __init__(self, smoothing: float = 0.0):
        self.smoothing = smoothing
        self.is_fitted = False
        self.default_prob_ = 0.5

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series):
        """Fit records historical target prevalence as fallback."""
        y_arr = np.asarray(y, dtype=float)
        self.default_prob_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict probability of fatal event in month T+1.

        If X is a DataFrame containing 'fatal_event' or 'fatal_event_lag1',
        that column is used directly. Otherwise, expects a 1D or 2D array where
        the first feature is the current fatal_event indicator.
        """
        if isinstance(X, pd.DataFrame):
            if "fatal_event" in X.columns:
                vals = X["fatal_event"].astype(float).values
            elif "fatal_event_lag1" in X.columns:
                vals = X["fatal_event_lag1"].astype(float).values
            else:
                vals = X.iloc[:, 0].astype(float).values
        else:
            X_arr = np.asarray(X)
            vals = X_arr[:, 0] if X_arr.ndim > 1 else X_arr

        # Handle NaNs
        vals = np.where(np.isnan(vals), self.default_prob_, vals)

        if self.smoothing > 0.0:
            probs = np.where(vals >= 0.5, 1.0 - self.smoothing, self.smoothing)
        else:
            probs = np.where(vals >= 0.5, 1.0, 0.0)

        # Return Nx2 array or 1D array depending on standard usage
        return probs

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Binary 0/1 prediction."""
        return (self.predict_proba(X) >= 0.5).astype(int)


class ClimatologicalBaseline:
    """
    Climatological prevalence baseline.
    Predicts constant sample prevalence: p_i = sum(y_train) / N_train.
    Serves as the benchmark for Brier Skill Score calculation.
    """

    def __init__(self):
        self.prevalence_ = 0.5
        self.is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series):
        y_arr = np.asarray(y, dtype=float)
        self.prevalence_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        n = len(X)
        return np.full(n, fill_value=self.prevalence_, dtype=float)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
