"""
src/models/glm.py
─────────────────
M1: Negative-Binomial Generalized Linear Model (GLM) for conflict fatalities.

Rationale:
  Conflict event counts and fatalities are heavily overdispersed (variance >> mean).
  Standard Poisson regression underestimates variance and produces overly narrow
  confidence intervals and uncalibrated risk probabilities.
  The Negative Binomial GLM parameterises the mean mu = exp(X beta) with dispersion alpha:
      Var(Y) = mu + alpha * mu^2
  
Probability derivation:
  For NB2 parameterisation, the zero-event probability is:
      P(Y = 0) = (1 + alpha * mu)^(-1 / alpha)
  Thus the probability of >= 1 fatal event in month T+1 is:
      P(fatal_event_{T+1} >= 1) = 1 - (1 + alpha * mu)^(-1 / alpha)
  In the Poisson limit (alpha -> 0):
      P(fatal_event_{T+1} >= 1) = 1 - exp(-mu)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# Default recommended features for M1 GLM (parsimonious set to prevent multicollinearity)
DEFAULT_GLM_FEATURES = [
    "fatalities_lag1",
    "fatalities_roll3m_mean",
    "fatalities_roll6m_mean",
    "spatial_fatalities_lag1",
    "coverage_lag1",
    "month_sin",
    "month_cos",
]


class NegativeBinomialM1(BaseEstimator, ClassifierMixin):
    """
    Negative-Binomial GLM (M1).
    Fits count regression on fatalities and transforms expected counts into
    calibrated exceedance probabilities: P(Y >= 1).
    """

    def __init__(
        self,
        features: list[str] | None = None,
        alpha: float = 1.0,
        clip_prob_min: float = 1e-4,
        clip_prob_max: float = 1.0 - 1e-4,
        maxiter: int = 100,
    ):
        self.features = features or DEFAULT_GLM_FEATURES
        self.alpha = float(alpha)
        self.clip_prob_min = clip_prob_min
        self.clip_prob_max = clip_prob_max
        self.maxiter = maxiter

        self.scaler_ = None
        self.model_res_ = None
        self.feature_names_in_ = None
        self.baseline_prevalence_ = 0.5
        self.coefficients_ = {}

    def _prepare_X(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            available = [f for f in self.features if f in X.columns]
            if len(available) == 0:
                raise ValueError(f"None of GLM features {self.features} found in DataFrame.")
            X_df = X[available].copy().fillna(0.0)
            return X_df.values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)
            return np.nan_to_num(X_arr, nan=0.0)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series):
        """
        Fit Negative Binomial GLM.

        Parameters
        ----------
        X : Features (DataFrame or array)
        y : Count of fatalities (or events) for month T+1.
            If binary 0/1 is passed, fits Binomial GLM / Logistic fallback.
        """
        y_arr = np.asarray(y, dtype=float)
        # Record prevalence for fallbacks
        self.baseline_prevalence_ = float(np.mean(y_arr > 0)) if len(y_arr) > 0 else 0.5

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [f for f in self.features if f in X.columns]
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1] if X.ndim > 1 else 1)]

        X_mat = self._prepare_X(X)

        # Scale features to stabilise gradient descent
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_mat)
        X_design = sm.add_constant(X_scaled, has_constant="add")

        try:
            # Check if y is count or binary
            is_count = np.any(y_arr > 1.0)
            if is_count:
                family = sm.families.NegativeBinomial(alpha=self.alpha)
            else:
                family = sm.families.Binomial()

            glm_model = sm.GLM(y_arr, X_design, family=family)
            self.model_res_ = glm_model.fit(maxiter=self.maxiter, disp=False)

            # Store coefficients
            col_names = ["const"] + self.feature_names_in_
            if len(self.model_res_.params) == len(col_names):
                self.coefficients_ = dict(zip(col_names, self.model_res_.params))

        except Exception as e:
            logger.warning(f"[NegativeBinomialM1] Fit failed ({e}). Using baseline fallback.")
            self.model_res_ = None

        return self

    def predict_count(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict expected fatalities count mu = E[Y]."""
        if self.model_res_ is None:
            return np.full(len(X), fill_value=self.baseline_prevalence_)

        X_mat = self._prepare_X(X)
        X_scaled = self.scaler_.transform(X_mat)
        X_design = sm.add_constant(X_scaled, has_constant="add")
        mu = self.model_res_.predict(X_design)
        return np.asarray(mu, dtype=float)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict probability P(fatal_event_{T+1} >= 1).

        Converts expected count mu to exceedance probability:
            P(Y >= 1) = 1 - (1 + alpha * mu)^(-1 / alpha)
        """
        if self.model_res_ is None:
            n = len(X)
            return np.full(n, fill_value=self.baseline_prevalence_, dtype=float)

        X_mat = self._prepare_X(X)
        X_scaled = self.scaler_.transform(X_mat)
        X_design = sm.add_constant(X_scaled, has_constant="add")

        if isinstance(self.model_res_.family, sm.families.Binomial):
            probs = self.model_res_.predict(X_design)
        else:
            # Negative Binomial
            mu = self.model_res_.predict(X_design)
            mu = np.clip(mu, 1e-6, 1e6)
            if self.alpha <= 1e-5:
                # Poisson limit
                probs = 1.0 - np.exp(-mu)
            else:
                p_zero = (1.0 + self.alpha * mu) ** (-1.0 / self.alpha)
                probs = 1.0 - p_zero

        probs = np.clip(probs, self.clip_prob_min, self.clip_prob_max)
        return np.asarray(probs, dtype=float)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Binary 0/1 prediction at 0.5 threshold."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def summary(self) -> str:
        """Return model stats summary."""
        if self.model_res_ is not None:
            return str(self.model_res_.summary())
        return "Model not fitted or fitted with baseline fallback."
