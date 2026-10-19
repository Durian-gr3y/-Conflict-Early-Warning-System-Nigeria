"""
src/models package
──────────────────
Models, backtesting framework, and evaluation metrics for Conflict Early Warning Nigeria.
"""

from src.models.baseline import NaiveM0, ClimatologicalBaseline
from src.models.glm import NegativeBinomialM1, DEFAULT_GLM_FEATURES
from src.models.gbm import LightGBMM2, DEFAULT_GBM_FEATURES
from src.models.backtest import ExpandingWindowBacktester
from src.models.evaluation import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    brier_decomposition,
    diebold_mariano_test,
)

__all__ = [
    "NaiveM0",
    "ClimatologicalBaseline",
    "NegativeBinomialM1",
    "DEFAULT_GLM_FEATURES",
    "LightGBMM2",
    "DEFAULT_GBM_FEATURES",
    "ExpandingWindowBacktester",
    "brier_score",
    "brier_skill_score",
    "expected_calibration_error",
    "brier_decomposition",
    "diebold_mariano_test",
]
