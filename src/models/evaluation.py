"""
src/models/evaluation.py
─────────────────────────
Evaluation metrics and statistical tests for conflict early warning forecasts.

Includes:
  - Brier score: MSE of probabilistic forecast (0 = perfect, lower is better)
  - Brier Skill Score (BSS): skill relative to reference baseline (>0 = beats baseline)
  - Murphy Brier Decomposition: Reliability, Resolution, Uncertainty
  - Expected Calibration Error (ECE): weighted average bin calibration error
  - AUC-ROC: discrimination ability
  - Diebold-Mariano test (1995, with Harvey-Leybourne-Newbold 1997 correction):
    formal test of whether one forecasting model statistically significantly
    outperforms another on time-series forecast error differentials.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score: MSE of predicted probabilities vs binary outcomes."""
    return float(brier_score_loss(y_true, y_prob))


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline_brier: float | None = None
) -> float:
    """
    Compute Brier Skill Score (BSS) relative to reference baseline:
        BSS = 1 - (BS_model / BS_ref)
    If baseline_brier is None, uses the climatological prevalence (sample mean) as reference.
    BSS > 0 means the model beats the reference; BSS = 1 is perfect.
    """
    bs = brier_score(y_true, y_prob)
    if baseline_brier is None:
        p_ref = np.full_like(y_true, fill_value=np.mean(y_true), dtype=float)
        baseline_brier = brier_score(y_true, p_ref)
    if baseline_brier == 0:
        return 1.0 if bs == 0 else -np.inf
    return float(1.0 - (bs / baseline_brier))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Divides predictions into bins and computes sum_k (|B_k| / N) * |acc(B_k) - conf(B_k)|.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[0] = 0.0
        bins[-1] = 1.0 + 1e-6
        bin_ids = np.digitize(y_prob, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ece = 0.0
    for b in range(n_bins):
        mask = (bin_ids == b)
        count = np.sum(mask)
        if count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def brier_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> dict[str, float]:
    """
    Murphy (1973) Brier Score Decomposition:
        BS = Reliability - Resolution + Uncertainty
    where:
        Uncertainty = c * (1 - c) with c = base rate
        Reliability = sum_k (N_k / N) * (p_k - o_k)^2 (0 is perfect calibration)
        Resolution  = sum_k (N_k / N) * (o_k - c)^2 (higher means better discrimination)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"brier": 0.0, "reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0}

    c = np.mean(y_true)
    uncertainty = c * (1.0 - c)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0

    for b in range(n_bins):
        mask = (bin_ids == b)
        count = np.sum(mask)
        if count > 0:
            p_k = np.mean(y_prob[mask])
            o_k = np.mean(y_true[mask])
            weight = count / n
            reliability += weight * ((p_k - o_k) ** 2)
            resolution += weight * ((o_k - c) ** 2)

    bs = float(brier_score(y_true, y_prob))
    return {
        "brier": round(bs, 4),
        "reliability": round(float(reliability), 4),
        "resolution": round(float(resolution), 4),
        "uncertainty": round(float(uncertainty), 4),
    }


def diebold_mariano_test(
    d_seq: np.ndarray,
    h: int = 1
) -> dict[str, float | str | bool]:
    """
    Diebold-Mariano test (1995) with Harvey, Leybourne, and Newbold (1997) correction.

    Parameters
    ----------
    d_seq : 1D array of loss differentials across time steps:
            d_t = L(y_t, p1_t) - L(y_t, p2_t).
            If d_t > 0, model 2 has smaller loss than model 1 at time t.
    h : Forecast horizon (default 1 for 1-month-ahead forecast).

    Returns
    -------
    dict with:
      - mean_diff: mean of d_seq (positive means model 2 has lower loss)
      - dm_stat: DM test statistic
      - p_value: 2-sided p-value
      - significant: bool (p < 0.05)
      - superior_model: 'model_2' if d > 0 and sig, 'model_1' if d < 0 and sig, else 'no_difference'
    """
    d_seq = np.asarray(d_seq, dtype=float)
    T = len(d_seq)
    if T < 3:
        return {
            "mean_diff": float(np.mean(d_seq)) if T > 0 else 0.0,
            "dm_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "superior_model": "insufficient_data"
        }

    d_mean = np.mean(d_seq)
    
    # Autocovariance estimation up to lag h-1
    # For h=1, this is simply the variance of d_seq
    gamma_0 = np.var(d_seq, ddof=0)
    var_d = gamma_0
    
    for k in range(1, h):
        gamma_k = np.mean((d_seq[k:] - d_mean) * (d_seq[:-k] - d_mean))
        var_d += 2.0 * gamma_k

    if var_d <= 1e-12:
        return {
            "mean_diff": round(float(d_mean), 6),
            "dm_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "superior_model": "identical"
        }

    # Standard Diebold-Mariano statistic
    dm_raw = d_mean / np.sqrt(var_d / T)

    # Harvey, Leybourne, Newbold (1997) small-sample correction factor
    hl_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    dm_stat = dm_raw * hl_factor

    # Student's t distribution with (T - 1) degrees of freedom
    df = T - 1
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=df)))
    significant = bool(p_value < 0.05)

    if significant:
        superior = "model_2" if d_mean > 0 else "model_1"
    else:
        superior = "no_difference"

    return {
        "mean_diff": round(float(d_mean), 6),
        "dm_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 5),
        "significant": significant,
        "superior_model": superior,
        "degrees_of_freedom": df
    }
