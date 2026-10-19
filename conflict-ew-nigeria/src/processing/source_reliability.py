"""
src/processing/source_reliability.py
─────────────────────────────────────
Source-reliability scoring for each data source × state × month cell.

WHAT IS SOURCE RELIABILITY?
    In conflict data, not all events get recorded. Whether an event appears
    in a database depends on:
      1. Whether anyone witnessed it and reported it.
      2. Whether a coder reviewed and included that report.
      3. How precise the location and date information are.

    A 'source-reliability score' (0–1) summarises how complete and accurate
    we believe each source is for a given state-month. It is NOT a measure
    of source integrity (UCDP and ACLED are both reputable); it is a measure
    of *coverage* — how much of ground truth is likely captured.

SCORING METHOD (transparent and reproducible)
    Each source contributes sub-scores on these dimensions:

    For ACLED weekly aggregated data (state-level only):
        S1_weeks : fraction of weeks in the month that have at least one
                   row for the state (0 to 1)
        S2_geo   : fixed penalty because admin1 data uses state centroids,
                   not event GPS points (constant 0.6 for all cells)

    For ACLED event-level (when key is received):
        S1_weeks : as above (should be 1.0 once API is live)
        S2_geo   : fraction of events with geo_precision <= 2 (town level)

    For UCDP GED:
        S1_coverage : fraction of state-months with at least 1 event,
                      compared to ACLED for the same period
        S2_geo      : 1 - (mean_where_prec - 1) / 6  [maps 1→1.0, 7→0.0]
        S3_clarity  : pct_clear_events (event_clarity == 1)

    Final score = geometric mean of sub-scores:
        score = (S1 * S2 * ... * Sk) ^ (1/k)

    Geometric mean is used rather than arithmetic mean so that a score of 0
    on any single dimension sets the final score to 0.
    This is conservative: a perfectly geo-precise dataset with zero coverage
    is useless.

    All sub-scores and the composite are written to the output, so anyone
    can audit why a particular cell scored the way it did.

CALIBRATION NOTE
    These scores are expert-judgement weights, not statistically calibrated
    probabilities of completeness. They should be labelled "coverage quality
    index" in user-facing materials, not "% events captured."
    A Phase 3 improvement would regress these scores against local media
    density (GDELT article counts) to calibrate them empirically.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Penalty applied to ALL ACLED weekly aggregated cells because the data
# uses state centroids (not event GPS). Set to 0.6 to reflect substantial
# but not total information loss.
ACLED_AGG_GEO_PENALTY = 0.6

# UCDP typically captures a lower fraction of events than ACLED in Nigeria
# because of its stricter inclusion threshold (must have organised armed actor
# and documented deaths). This prior shrinks UCDP scores relative to ACLED.
UCDP_COVERAGE_PRIOR = 0.7   # "expect UCDP to capture ~70% of ACLED events"


# ─── ACLED weekly aggregate reliability ───────────────────────────────────────

def score_acled_agg(
    monthly_panel: pd.DataFrame,
    weeks_per_month: int = 4,
) -> pd.DataFrame:
    """
    Compute reliability scores for the ACLED weekly aggregated panel.

    Parameters
    ----------
    monthly_panel : pd.DataFrame
        The monthly admin1 panel produced by Phase 1 (run_phase1.py).
        Must contain: month, admin1, weeks_observed.
    weeks_per_month : int
        Expected number of weeks per month (4 or 5). Used to normalise
        the coverage score. Default 4 is conservative.

    Returns
    -------
    pd.DataFrame with columns:
        month, admin1,
        s1_temporal_coverage,   S1 sub-score
        s2_geo_precision,       S2 sub-score (constant for weekly agg)
        reliability_acled_agg,  composite score 0–1
        reliability_label       human-readable tier
    """
    req = {"month", "admin1", "weeks_observed"}
    missing = req - set(monthly_panel.columns)
    if missing:
        raise ValueError(f"monthly_panel missing columns: {missing}")

    out = monthly_panel[["month", "admin1", "weeks_observed"]].copy()

    # S1: temporal coverage
    out["s1_temporal_coverage"] = (
        out["weeks_observed"].clip(upper=weeks_per_month) / weeks_per_month
    ).clip(0, 1)

    # S2: geo precision (constant — state centroid penalty)
    out["s2_geo_precision"] = ACLED_AGG_GEO_PENALTY

    # Composite: geometric mean of S1, S2
    out["reliability_acled_agg"] = np.sqrt(
        out["s1_temporal_coverage"] * out["s2_geo_precision"]
    ).round(4)

    out["reliability_label"] = pd.cut(
        out["reliability_acled_agg"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Medium", "High", "Very High"],
    )
    out["source"] = "ACLED_agg"

    logger.info(
        f"ACLED_agg reliability scored: {len(out):,} cells  |  "
        f"Mean score: {out['reliability_acled_agg'].mean():.3f}"
    )
    return out


# ─── UCDP reliability ─────────────────────────────────────────────────────────

def score_ucdp(
    ucdp_monthly: pd.DataFrame,
    acled_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute reliability scores for UCDP monthly panel cells.

    Uses ACLED as a reference to compute a cross-source coverage score:
    if UCDP shows events in fewer months than ACLED for a state, the
    UCDP coverage score is penalised.

    Parameters
    ----------
    ucdp_monthly : pd.DataFrame
        Output of src/ingestion/ucdp.load_ucdp_nigeria() — must contain:
        month, admin1, mean_geo_prec, pct_clear_events, events_ucdp.
    acled_monthly : pd.DataFrame
        Monthly ACLED panel — must contain: month, admin1, events.

    Returns
    -------
    pd.DataFrame with columns:
        month, admin1,
        s1_cross_source_coverage,
        s2_geo_precision,
        s3_event_clarity,
        reliability_ucdp,
        reliability_label,
        source
    """
    req_ucdp = {"month", "admin1", "mean_geo_prec", "pct_clear_events", "events_ucdp"}
    req_acled = {"month", "admin1", "events"}
    for req, df, name in [
        (req_ucdp, ucdp_monthly, "ucdp_monthly"),
        (req_acled, acled_monthly, "acled_monthly"),
    ]:
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    out = ucdp_monthly[
        ["month", "admin1", "mean_geo_prec", "pct_clear_events", "events_ucdp"]
    ].copy()

    # ── S1: Cross-source coverage ─────────────────────────────────────────────
    # For each state, what fraction of ACLED-active months does UCDP also see?
    # "ACLED-active" = months where ACLED records at least 1 event.
    acled_active = (
        acled_monthly[acled_monthly["events"] > 0]
        .groupby("admin1")["month"]
        .count()
        .rename("acled_active_months")
    )
    ucdp_active = (
        out[out["events_ucdp"] > 0]
        .groupby("admin1")["month"]
        .count()
        .rename("ucdp_active_months")
    )
    coverage_ratio = (
        pd.concat([acled_active, ucdp_active], axis=1)
        .fillna(0)
        .assign(coverage_ratio=lambda d:
            (d["ucdp_active_months"] / d["acled_active_months"].clip(lower=1))
            .clip(upper=1)
            * UCDP_COVERAGE_PRIOR
        )["coverage_ratio"]
    )
    # Join coverage ratio back to cell-level
    out = out.merge(
        coverage_ratio.reset_index().rename(columns={"coverage_ratio": "s1_cross_source_coverage"}),
        on="admin1", how="left"
    )
    out["s1_cross_source_coverage"] = out["s1_cross_source_coverage"].fillna(
        UCDP_COVERAGE_PRIOR * 0.5   # conservative prior if no ACLED reference
    ).clip(0, 1)

    # ── S2: Geo precision ─────────────────────────────────────────────────────
    # where_prec: 1 (exact) → 1.0 score; 7 (country centroid) → 0.0 score
    out["s2_geo_precision"] = (
        1 - (out["mean_geo_prec"].clip(1, 7) - 1) / 6
    ).clip(0, 1)

    # ── S3: Event clarity ─────────────────────────────────────────────────────
    out["s3_event_clarity"] = out["pct_clear_events"].clip(0, 1)

    # ── Composite: geometric mean of S1, S2, S3 ───────────────────────────────
    out["reliability_ucdp"] = (
        out["s1_cross_source_coverage"]
        * out["s2_geo_precision"]
        * out["s3_event_clarity"]
    ).pow(1/3).round(4)

    out["reliability_label"] = pd.cut(
        out["reliability_ucdp"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Medium", "High", "Very High"],
    )
    out["source"] = "UCDP"

    logger.info(
        f"UCDP reliability scored: {len(out):,} cells  |  "
        f"Mean score: {out['reliability_ucdp'].mean():.3f}"
    )
    return out


# ─── Combined reliability table ───────────────────────────────────────────────

def build_reliability_table(
    acled_scores: pd.DataFrame,
    ucdp_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combine per-source scores into a single wide reliability table.

    Each row is a (month, admin1) cell with reliability scores from
    all available sources. Cells present in one source but not another
    are filled with NaN (not 0 — to distinguish 'no score' from 'low score').

    Also computes:
        source_agreement  : 1 - abs(fatal_event_acled - fatal_event_ucdp)
                            Only meaningful when both sources are present.
        combined_reliability : mean of available source scores

    Parameters
    ----------
    acled_scores : pd.DataFrame  Output of score_acled_agg()
    ucdp_scores  : pd.DataFrame  Output of score_ucdp()  (or None if not yet available)

    Returns
    -------
    pd.DataFrame  Wide table keyed by (month, admin1)
    """
    # ACLED side
    acled_wide = acled_scores[
        ["month", "admin1", "reliability_acled_agg",
         "s1_temporal_coverage", "s2_geo_precision"]
    ].rename(columns={
        "s1_temporal_coverage": "acled_s1_temporal",
        "s2_geo_precision": "acled_s2_geo",
    })

    if ucdp_scores is None:
        acled_wide["combined_reliability"] = acled_wide["reliability_acled_agg"]
        acled_wide["sources_available"] = "ACLED_agg"
        return acled_wide

    # UCDP side
    ucdp_wide = ucdp_scores[
        ["month", "admin1", "reliability_ucdp",
         "s1_cross_source_coverage", "s2_geo_precision", "s3_event_clarity"]
    ].rename(columns={
        "s1_cross_source_coverage": "ucdp_s1_coverage",
        "s2_geo_precision": "ucdp_s2_geo",
        "s3_event_clarity": "ucdp_s3_clarity",
    })

    merged = acled_wide.merge(ucdp_wide, on=["month", "admin1"], how="outer")

    # Combined reliability: mean of available sources
    score_cols = ["reliability_acled_agg", "reliability_ucdp"]
    merged["combined_reliability"] = merged[score_cols].mean(axis=1, skipna=True).round(4)

    # Which sources are available?
    def sources_str(row):
        srcs = []
        if pd.notna(row["reliability_acled_agg"]):
            srcs.append("ACLED_agg")
        if pd.notna(row["reliability_ucdp"]):
            srcs.append("UCDP")
        return "+".join(srcs) if srcs else "none"

    merged["sources_available"] = merged.apply(sources_str, axis=1)

    logger.info(
        f"Combined reliability table: {len(merged):,} cells  |  "
        f"Both sources: {(merged['sources_available'] == 'ACLED_agg+UCDP').sum()}"
    )
    return merged


# ─── Source-disagreement metrics ─────────────────────────────────────────────

def compute_source_disagreement(
    acled_monthly: pd.DataFrame,
    ucdp_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute month × state disagreement between ACLED and UCDP.

    Disagreement metrics:
        agree_on_presence : both agree on whether >=1 fatal event occurred
        fatality_ratio    : UCDP best / ACLED fatalities (>1 = UCDP higher)
        abs_diff_fatalities : |ACLED - UCDP|
        disagreement_flag : 1 if sources disagree on presence of fatal event

    This table is used to:
        a) flag cells where forecast uncertainty should be widened
        b) investigate systematic ACLED / UCDP coverage differences

    NOTE: UCDP lags ACLED by ~12 months for the annual release.
    Only compare the overlapping time period.
    """
    # Standardise merge keys
    acled = acled_monthly[
        ["month", "admin1", "fatal_event", "fatalities"]
    ].rename(columns={
        "fatal_event": "fatal_acled",
        "fatalities": "fat_acled",
    })
    ucdp = ucdp_monthly[
        ["month", "admin1", "fatal_event_ucdp", "fatalities_ucdp"]
    ].rename(columns={
        "fatal_event_ucdp": "fatal_ucdp",
        "fatalities_ucdp": "fat_ucdp",
    })

    merged = acled.merge(ucdp, on=["month", "admin1"], how="inner")

    merged["agree_on_presence"] = (
        merged["fatal_acled"] == merged["fatal_ucdp"]
    ).astype(int)
    merged["disagreement_flag"] = 1 - merged["agree_on_presence"]
    merged["abs_diff_fatalities"] = (
        merged["fat_acled"] - merged["fat_ucdp"]
    ).abs()
    merged["fatality_ratio"] = (
        merged["fat_ucdp"] / merged["fat_acled"].clip(lower=1)
    ).round(3)

    pct_agree = merged["agree_on_presence"].mean()
    logger.info(
        f"Source disagreement: {len(merged):,} overlapping cells  |  "
        f"Agreement on fatal event presence: {pct_agree:.1%}"
    )
    return merged
