"""
src/features/feature_engineering.py
─────────────────────────────────────
Feature matrix construction for the Conflict Early Warning Nigeria model.

DESIGN PRINCIPLES
    1. Every feature is computed from data strictly BEFORE the target month.
       No future leakage is possible by design — each function receives only
       the panel up to month T and produces a value for T+1.
    2. Every feature has a documented name, formula, and rationale.
    3. Stub features (environmental, economic) produce zeros with a clear
       STUB marker so they are easy to replace when real data arrives.
    4. The leakage_check() function is the final gate before saving.

TARGET VARIABLE (reminder)
    fatal_event : 1 if fatalities > 0 in state S in month T+1, else 0
    Forecast horizon: 1 month ahead

FEATURE GROUPS
    A. Temporal lags          — past outcomes, 1–12 months back
    B. Rolling windows        — short/medium/long-run trend summary
    C. Spatial lags           — IDW-weighted neighbour-state activity
    D. Seasonality            — month cyclical encoding, agricultural/pastoral calendar
    E. Islamic calendar       — Ramadan proximity (conflict pattern modulator in NW Nigeria)
    F. Coverage proxy         — reliability score as explicit model covariate
    G. Source disagreement    — ACLED/UCDP divergence as uncertainty signal
    H. Stubs: environmental   — NDVI anomaly, rainfall deficit (zeros until GEE data arrives)
    I. Stubs: economic        — food price deviation (zeros until WFP data arrives)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Nigerian state centroid table ────────────────────────────────────────────
# Used for spatial-lag calculation.
# Source: approximate centroids; will be refined once event-level ACLED arrives.
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "Zamfara":  (12.17,  6.66), "Katsina":  (12.99,  7.60),
    "Sokoto":   (13.06,  5.24), "Kaduna":   (10.52,  7.44),
    "Borno":    (11.85, 13.16), "Yobe":     (12.29, 11.44),
    "Adamawa":  ( 9.33, 12.39), "Lagos":    ( 6.52,  3.38),
    "Ogun":     ( 6.99,  3.47), "Oyo":      ( 7.85,  3.93),
    "Rivers":   ( 4.85,  7.04), "Delta":    ( 5.52,  5.90),
    "Bayelsa":  ( 4.77,  6.07), "Plateau":  ( 9.22,  9.52),
    "Benue":    ( 7.34,  8.74), "Nasarawa": ( 8.49,  8.52),
    "Kano":     (12.00,  8.52), "Jigawa":   (12.21,  9.56),
    "Kebbi":    (11.49,  4.20), "Niger":    (10.00,  5.60),
    "Kwara":    ( 8.49,  4.55), "FCT":      ( 9.07,  7.40),
    "Fct":      ( 9.07,  7.40),   # alias: title-cased by demo data normalisation

    "Gombe":    (10.29, 11.17), "Taraba":   ( 8.00, 11.50),
    "Bauchi":   (10.31,  9.84), "Kogi":     ( 7.73,  6.69),
    "Ondo":     ( 6.91,  5.20), "Osun":     ( 7.56,  4.56),
    "Ekiti":    ( 7.72,  5.30), "Edo":      ( 6.34,  5.62),
    "Anambra":  ( 6.21,  7.07), "Enugu":    ( 6.46,  7.55),
    "Imo":      ( 5.57,  7.06), "Abia":     ( 5.45,  7.52),
    "Ebonyi":   ( 6.26,  8.01), "Cross River": ( 5.87, 8.60),
    "Akwa Ibom":( 5.05,  7.93),
}

# Approximate Ramadan start months (Gregorian) per year.
# Ramadan lasts ~29-30 days. We flag the month in which Ramadan falls.
# Source: Islamic Society of North America (ISNA) calendar.
RAMADAN_MONTHS: dict[int, int] = {
    2015: 7,   # July 2015 (partial — started 18 June, ended 16 July)
    2016: 6,   # June 2016
    2017: 5,   # May 2017 (started 27 May)
    2018: 5,   # May 2018 (started 16 May)
    2019: 5,   # May 2019 (started 5 May)
    2020: 4,   # April 2020 (started 23 April)
    2021: 4,   # April 2021 (started 13 April)
    2022: 4,   # April 2022 (started 2 April)
    2023: 3,   # March 2023 (started 23 March)
    2024: 3,   # March 2024 (started 11 March)
    2025: 3,   # March 2025 (started 1 March)
}


# ─── Haversine distance ───────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in kilometres."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_distance_matrix(states: list[str]) -> pd.DataFrame:
    """
    Build pairwise distance matrix (km) between Nigerian states.

    States not in STATE_CENTROIDS get distance = NaN (effectively excluded
    from spatial lag computation). A warning is logged for each such state.

    Returns
    -------
    pd.DataFrame  shape (n_states, n_states), indexed and columned by state name.
                  Diagonal = 0. Upper and lower triangles are symmetric.
    """
    n = len(states)
    dist = np.zeros((n, n))
    missing = [s for s in states if s not in STATE_CENTROIDS]
    if missing:
        logger.warning(f"[SPATIAL] States missing from centroid table: {missing}")

    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if i == j:
                continue
            if s1 not in STATE_CENTROIDS or s2 not in STATE_CENTROIDS:
                dist[i, j] = np.nan
                continue
            lat1, lon1 = STATE_CENTROIDS[s1]
            lat2, lon2 = STATE_CENTROIDS[s2]
            dist[i, j] = haversine_km(lat1, lon1, lat2, lon2)

    return pd.DataFrame(dist, index=states, columns=states)


def idw_weights(dist_matrix: pd.DataFrame, power: float = 2.0) -> pd.DataFrame:
    """
    Compute inverse-distance weights from a distance matrix.

    Formula: w_ij = (1 / d_ij^power) / sum_k(1 / d_ik^power)
    where the sum excludes the diagonal (self).

    States with NaN distances receive weight 0 (excluded from sum).

    Parameters
    ----------
    dist_matrix : pd.DataFrame  Output of build_distance_matrix().
    power       : float         Distance decay exponent. Default 2.0 (standard IDW).
                                Higher values give more weight to nearby states.
    """
    raw_weights = 1.0 / dist_matrix.replace(0, np.nan).pow(power)
    row_sums = raw_weights.sum(axis=1, skipna=True)
    return raw_weights.div(row_sums, axis=0).fillna(0)


# ─── Feature group A: Temporal lags ──────────────────────────────────────────

def add_temporal_lags(
    panel: pd.DataFrame,
    target_cols: list[str] = ("events", "fatalities", "fatal_event"),
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
) -> pd.DataFrame:
    """
    Add lagged versions of target columns, grouped by state.

    LEAKAGE SAFETY: shift(n) for n>=1 means the value at row T uses data
    from row T-n. Since our target is T+1, any lag >= 1 is safe.

    Parameters
    ----------
    panel       : monthly admin1 panel sorted by (admin1, month)
    target_cols : columns to lag
    lags        : how many months back

    New columns added: {col}_lag{n}  e.g. events_lag1, fatalities_lag3
    """
    panel = panel.sort_values(["admin1", "month"]).copy()
    for col in target_cols:
        if col not in panel.columns:
            logger.warning(f"[LAGS] Column '{col}' not in panel — skipping.")
            continue
        for n in lags:
            panel[f"{col}_lag{n}"] = (
                panel.groupby("admin1")[col].shift(n)
            )
    logger.info(f"Temporal lags added: {[f'{c}_lag{n}' for c in target_cols for n in lags]}")
    return panel


# ─── Feature group B: Rolling windows ────────────────────────────────────────

def add_rolling_windows(
    panel: pd.DataFrame,
    target_cols: list[str] = ("events", "fatalities"),
    windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Add backward-looking rolling sums and means.

    Uses shift(1) before rolling so that window [T-1, T-2, ..., T-w] is used
    to predict T+1. This is safe: no future information.

    New columns: {col}_roll{w}m_sum, {col}_roll{w}m_mean
    """
    panel = panel.sort_values(["admin1", "month"]).copy()
    for col in target_cols:
        if col not in panel.columns:
            logger.warning(f"[ROLLING] Column '{col}' not in panel — skipping.")
            continue
        # shift(1) so window ends at T-1
        shifted = panel.groupby("admin1")[col].shift(1)
        for w in windows:
            rolling = shifted.groupby(panel["admin1"]).transform(
                lambda x: x.rolling(window=w, min_periods=1).sum()
            )
            panel[f"{col}_roll{w}m_sum"] = rolling

            rolling_mean = shifted.groupby(panel["admin1"]).transform(
                lambda x: x.rolling(window=w, min_periods=1).mean()
            )
            panel[f"{col}_roll{w}m_mean"] = rolling_mean

    logger.info(f"Rolling windows added (sum+mean): windows={windows}")
    return panel


# ─── Feature group C: Spatial lags ───────────────────────────────────────────

def add_spatial_lags(
    panel: pd.DataFrame,
    weight_matrix: pd.DataFrame,
    source_cols: list[str] = ("events_lag1", "fatalities_lag1"),
) -> pd.DataFrame:
    """
    Add inverse-distance-weighted spatial lags.

    For state S in month T, the spatial lag for events is:
        spatial_lag_events[S, T] = sum_j w[S,j] * events_lag1[j, T]

    Because we use the lagged values (already computed in group A),
    there is no leakage: the lagged values at time T reference month T-1.

    New columns: spatial_{col}
    """
    if weight_matrix is None or weight_matrix.empty:
        logger.warning("[SPATIAL] No weight matrix — skipping spatial lags.")
        return panel

    panel = panel.sort_values(["admin1", "month"]).copy()
    states_in_panel = panel["admin1"].unique()
    states_in_wmat  = weight_matrix.index

    # Only use states present in both panel and weight matrix
    common_states = [s for s in states_in_panel if s in states_in_wmat]
    if len(common_states) < len(states_in_panel):
        missing = set(states_in_panel) - set(common_states)
        logger.warning(f"[SPATIAL] States missing from weight matrix: {missing}")

    months = sorted(panel["month"].unique())
    spatial_records = []

    for month in months:
        month_data = panel[panel["month"] == month].set_index("admin1")
        for state in common_states:
            row = {"month": month, "admin1": state}
            for col in source_cols:
                if col not in month_data.columns:
                    row[f"spatial_{col}"] = np.nan
                    continue
                # Get weight vector for this state
                w = weight_matrix.loc[state, common_states]
                # Get source values for all states in this month
                # Cast to float64 to avoid pandas Int64 masked-array errors
                vals = (
                    month_data.loc[
                        month_data.index.isin(common_states), col
                    ]
                    .reindex(common_states)
                    .fillna(0)
                    .astype(float)          # <-- critical: strip nullable Int64
                )
                w_arr = w.values.astype(float)
                # Weighted sum
                row[f"spatial_{col}"] = float(w_arr @ vals.values)
            spatial_records.append(row)

    if not spatial_records:
        return panel

    spatial_df = pd.DataFrame(spatial_records)
    new_cols = [c for c in spatial_df.columns if c.startswith("spatial_")]
    panel = panel.merge(spatial_df[["month", "admin1"] + new_cols],
                        on=["month", "admin1"], how="left")

    logger.info(f"Spatial lags added: {new_cols}")
    return panel


# ─── Feature group D: Seasonality ────────────────────────────────────────────

def add_seasonality(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based seasonality features.

    Cyclical encoding (sin/cos) is preferred over month dummies because:
    - It preserves the circular nature of months (Dec is close to Jan)
    - It reduces the number of features from 11 dummies to 2
    - Gradient boosting can handle either; linear models prefer cyclical

    Conflict calendar for Nigeria:
    - Harmattan (Nov-Mar): dry season; reduced visibility; cattle driven south
      → peak Fulani-farmer conflict in Middle Belt
    - Farming season (Apr-Oct): crops planted; displacement; resource competition
    - Post-harvest (Nov-Jan): food availability; sometimes calming effect
    """
    panel = panel.copy()

    # Extract month number from Period
    if hasattr(panel["month"].dtype, "freq"):
        month_num = panel["month"].dt.month
        year_num  = panel["month"].dt.year
    else:
        month_num = panel["month"].astype(str).str[-2:].astype(int)
        year_num  = panel["month"].astype(str).str[:4].astype(int)

    # Cyclical encoding
    panel["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    panel["month_cos"] = np.cos(2 * np.pi * month_num / 12)
    panel["month_of_year"] = month_num      # raw month, for interpretability
    panel["year"] = year_num

    # Conflict-calendar flags (binary)
    panel["is_harmattan"]    = month_num.isin([11, 12, 1, 2, 3]).astype(int)
    panel["is_farming"]      = month_num.isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
    panel["is_post_harvest"] = month_num.isin([11, 12, 1]).astype(int)

    logger.info("Seasonality features added.")
    return panel


# ─── Feature group E: Islamic calendar ───────────────────────────────────────

def add_ramadan_flag(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Flag months that coincide with Ramadan.

    Ramadan effect in Northwest Nigeria literature:
    - Some studies find reduced inter-group violence during Ramadan
    - Others find increased tension immediately after (Eid)
    - Effect is contested; we include it as a feature and let the model decide

    We flag both the Ramadan month itself and the following month (Eid effect).
    """
    panel = panel.copy()

    if hasattr(panel["month"].dtype, "freq"):
        month_num = panel["month"].dt.month
        year_num  = panel["month"].dt.year
    else:
        month_num = panel["month"].astype(str).str[-2:].astype(int)
        year_num  = panel["month"].astype(str).str[:4].astype(int)

    def is_ramadan(year, month):
        ram_month = RAMADAN_MONTHS.get(int(year))
        if ram_month is None:
            return 0
        return 1 if int(month) == ram_month else 0

    def is_post_ramadan(year, month):
        ram_month = RAMADAN_MONTHS.get(int(year))
        if ram_month is None:
            return 0
        eid_month = (ram_month % 12) + 1   # next month (wraps Dec → Jan)
        return 1 if int(month) == eid_month else 0

    panel["is_ramadan"]      = [is_ramadan(y, m) for y, m
                                in zip(year_num, month_num)]
    panel["is_post_ramadan"] = [is_post_ramadan(y, m) for y, m
                                in zip(year_num, month_num)]

    logger.info("Ramadan flags added.")
    return panel


# ─── Feature group F: Coverage proxy ─────────────────────────────────────────

def add_coverage_proxy(
    panel: pd.DataFrame,
    reliability_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add the source-reliability score as an explicit model covariate.

    WHY THIS MATTERS:
    The model should know how confident we are in the absence of events.
    A cell with reliability=0.3 and events=0 should produce a higher
    forecast than a cell with reliability=0.9 and events=0, because
    in the low-reliability cell 'quiet' may mean 'un-observed'.

    Features added:
        reliability_score   : combined_reliability from Phase 2 (0–1)
        log_reliability     : log(reliability + 0.01) — makes effect linear
        low_coverage_flag   : 1 if reliability < 0.4
        coverage_lag1       : coverage score from previous month
    """
    panel = panel.copy()

    if reliability_table is not None and len(reliability_table) > 0:
        # Harmonise month type
        rel = reliability_table.copy()
        rel["month"] = rel["month"].astype(str).apply(
            lambda x: pd.Period(x, freq="M")
        )
        merge_cols = ["month", "admin1", "combined_reliability"]
        avail = [c for c in merge_cols if c in rel.columns]
        if "combined_reliability" in rel.columns:
            panel = panel.merge(
                rel[avail].rename(columns={"combined_reliability": "reliability_score"}),
                on=["month", "admin1"],
                how="left",
            )
        else:
            panel["reliability_score"] = np.nan
    else:
        # Fall back to the low_coverage column from Phase 1
        if "low_coverage" in panel.columns:
            panel["reliability_score"] = 1 - panel["low_coverage"].astype(float) * 0.5
        else:
            panel["reliability_score"] = 0.5   # neutral prior

    panel["reliability_score"] = panel["reliability_score"].fillna(0.5).clip(0, 1)
    panel["log_reliability"]   = np.log(panel["reliability_score"] + 0.01)
    panel["low_coverage_flag"] = (panel["reliability_score"] < 0.4).astype(int)

    # Lag of coverage (coverage in previous month)
    panel = panel.sort_values(["admin1", "month"])
    panel["coverage_lag1"] = panel.groupby("admin1")["reliability_score"].shift(1)

    logger.info("Coverage proxy features added.")
    return panel


# ─── Feature group G: Source disagreement ────────────────────────────────────

def add_source_disagreement(
    panel: pd.DataFrame,
    disagree_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add the ACLED/UCDP disagreement flag as a model covariate.

    When two independent sources disagree on whether a fatal event occurred,
    the model's output uncertainty should be wider.
    We include the disagreement flag as an input feature so the model can
    learn to be less confident in such cells.

    Features added:
        source_disagree_lag1 : disagreement flag from month T-1
        abs_diff_fat_lag1    : absolute fatality difference, month T-1
    """
    panel = panel.copy()

    if disagree_table is not None and len(disagree_table) > 0:
        dis = disagree_table.copy()
        dis["month"] = dis["month"].astype(str).apply(
            lambda x: pd.Period(x, freq="M")
        )
        cols = ["month", "admin1", "disagreement_flag", "abs_diff_fatalities"]
        avail = [c for c in cols if c in dis.columns]
        panel = panel.merge(dis[avail], on=["month", "admin1"], how="left")
        panel["disagreement_flag"] = panel["disagreement_flag"].fillna(0).astype(int)
        if "abs_diff_fatalities" in panel.columns:
            panel["abs_diff_fatalities"] = panel["abs_diff_fatalities"].fillna(0)
    else:
        panel["disagreement_flag"]  = 0
        panel["abs_diff_fatalities"] = 0.0

    # Lag the disagreement (to predict T+1, use T's disagreement)
    panel = panel.sort_values(["admin1", "month"])
    panel["source_disagree_lag1"] = (
        panel.groupby("admin1")["disagreement_flag"].shift(1).fillna(0)
    )
    if "abs_diff_fatalities" in panel.columns:
        panel["abs_diff_fat_lag1"] = (
            panel.groupby("admin1")["abs_diff_fatalities"].shift(1).fillna(0)
        )

    logger.info("Source disagreement features added.")
    return panel


# ─── Feature groups H & I: Environmental and economic stubs ──────────────────

def add_environmental_stubs(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add placeholder environmental features.

    STUBS — replace with real data from Google Earth Engine in Phase 3b:
        ndvi_anomaly      : NDVI deviation from 10-year monthly mean (current: 0)
        rainfall_deficit  : mm below monthly normal from CHIRPS (current: 0)
        enso_phase        : El Nino/La Nina index (current: 0)

    To replace a stub:
        1. Download CHIRPS Nigeria data via GEE or direct API
        2. Compute monthly admin1 averages
        3. Join on (month, admin1) and overwrite the column
        4. Remove the [STUB] prefix from the column documentation

    NOTE: NDVI is a proxy for drought stress and pastoral pressure.
    Rainfall deficit is the primary driver of farmer-herder conflict
    via reduced grazing land availability.
    """
    panel = panel.copy()
    panel["ndvi_anomaly"]     = 0.0   # [STUB] replace with MODIS/NDVI
    panel["rainfall_deficit"] = 0.0   # [STUB] replace with CHIRPS
    panel["enso_phase"]       = 0.0   # [STUB] replace with NOAA ONI index
    logger.info("[STUB] Environmental features added (zeros — replace with real data).")
    return panel


def add_economic_stubs(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add placeholder economic features.

    STUBS — replace with WFP VAM / FAO GIEWS data:
        food_price_dev   : local food price index deviation from 12-month mean
        fuel_price_dev   : fuel price deviation (proxy for transport/security costs)

    Food price spikes correlate with civil unrest across sub-Saharan Africa
    (Burke et al. 2015; Hendrix & Haggard 2015).
    """
    panel = panel.copy()
    panel["food_price_dev"] = 0.0   # [STUB] replace with WFP VAM
    panel["fuel_price_dev"] = 0.0   # [STUB] replace with DPR Nigeria
    logger.info("[STUB] Economic features added (zeros — replace with real data).")
    return panel


# ─── Master builder ───────────────────────────────────────────────────────────

def build_feature_matrix(
    monthly_panel: pd.DataFrame,
    reliability_table: pd.DataFrame | None = None,
    disagree_table: pd.DataFrame | None = None,
    include_spatial: bool = True,
    idw_power: float = 2.0,
) -> pd.DataFrame:
    """
    Build the complete feature matrix for modelling.

    Parameters
    ----------
    monthly_panel     : ACLED monthly admin1 panel (Phase 1 output).
    reliability_table : Phase 2 combined reliability table (or None).
    disagree_table    : Phase 2 source-disagreement table (or None).
    include_spatial   : Whether to compute spatial lags (slower).
    idw_power         : Distance decay exponent for IDW (default 2).

    Returns
    -------
    pd.DataFrame  Feature matrix, one row per (month, admin1).
                  All features reference data from month T or earlier.
                  Target variable: fatal_event (to predict T+1, shift this by -1
                  in the modelling script, or use fatal_event_lag1 as the target).

    Leakage safety
    --------------
    Run leakage_check(feature_matrix) before training any model.
    """
    panel = monthly_panel.copy()

    # Ensure sorted
    panel = panel.sort_values(["admin1", "month"]).reset_index(drop=True)

    # A. Temporal lags
    panel = add_temporal_lags(
        panel,
        target_cols=["events", "fatalities", "fatal_event"],
        lags=(1, 2, 3, 6, 12),
    )

    # B. Rolling windows
    panel = add_rolling_windows(
        panel,
        target_cols=["events", "fatalities"],
        windows=(3, 6, 12),
    )

    # C. Spatial lags
    if include_spatial:
        states = sorted(panel["admin1"].unique())
        dist_matrix = build_distance_matrix(states)
        weights     = idw_weights(dist_matrix, power=idw_power)
        panel = add_spatial_lags(
            panel, weights,
            source_cols=["events_lag1", "fatalities_lag1"],
        )

    # D. Seasonality
    panel = add_seasonality(panel)

    # E. Islamic calendar
    panel = add_ramadan_flag(panel)

    # F. Coverage proxy
    panel = add_coverage_proxy(panel, reliability_table)

    # G. Source disagreement
    panel = add_source_disagreement(panel, disagree_table)

    # H. Environmental stubs
    panel = add_environmental_stubs(panel)

    # I. Economic stubs
    panel = add_economic_stubs(panel)

    # ── Create the MODEL-READY target ──────────────────────────────────────────
    # fatal_event_next : fatal_event shifted BACK by 1 (i.e., next month's outcome)
    # This is what we actually predict. We include it in the matrix for
    # bookkeeping; it must NEVER appear as an input feature in any model.
    panel["fatal_event_next"] = (
        panel.sort_values(["admin1", "month"])
        .groupby("admin1")["fatal_event"]
        .shift(-1)
    )

    logger.info(
        f"Feature matrix built: {len(panel):,} rows x {panel.shape[1]} columns"
    )
    return panel


# ─── Leakage check ────────────────────────────────────────────────────────────

def leakage_check(feature_matrix: pd.DataFrame) -> dict:
    """
    Verify that no feature uses future information.

    WHAT WE CHECK:
    1. Columns named *_lag{n} for n < 1 would be dangerous — we assert none exist.
    2. Columns named *_roll*  should all be built from shift(1); we cannot verify
       this from column names alone, but we log the column names for manual review.
    3. fatal_event_next (the target) must NOT appear among the feature columns
       we intend to use for training. We flag it explicitly.
    4. Temporal consistency: for each (admin1, col), if col uses lag-1 data,
       the value at month T should equal the value at month T-1 (for raw cols).
       We spot-check this for events_lag1.

    Returns
    -------
    dict  with keys: passed (bool), warnings (list[str]), errors (list[str])
    """
    errors   = []
    warnings = []

    # Check 1: No lag-0 columns (which would be contemporaneous)
    lag0_cols = [c for c in feature_matrix.columns
                 if "_lag0" in c.lower()]
    if lag0_cols:
        errors.append(f"Lag-0 columns found (contemporaneous leakage!): {lag0_cols}")

    # Check 2: fatal_event_next must not be listed as a predictor
    if "fatal_event_next" not in feature_matrix.columns:
        warnings.append("fatal_event_next not found — ensure target is defined.")
    else:
        warnings.append(
            "REMINDER: 'fatal_event_next' is the TARGET. "
            "Never include it in the X matrix when training."
        )

    # Check 3: Stub columns
    stub_cols = [c for c in feature_matrix.columns
                 if feature_matrix[c].eq(0).all() and c not in
                 ("disagreement_flag", "source_disagree_lag1")]
    if stub_cols:
        warnings.append(
            f"All-zero columns (likely stubs not yet replaced): {stub_cols}"
        )

    # Check 4: Spot-check events_lag1
    if "events_lag1" in feature_matrix.columns and "events" in feature_matrix.columns:
        fm = feature_matrix.sort_values(["admin1", "month"])
        check = fm.groupby("admin1").apply(
            lambda g: (g["events_lag1"] == g["events"].shift(1)).all()
        )
        if not check.all():
            bad = check[~check].index.tolist()
            errors.append(
                f"events_lag1 != events.shift(1) for states: {bad} — "
                f"possible leakage or sort order issue."
            )

    passed = len(errors) == 0
    result = {"passed": passed, "errors": errors, "warnings": warnings}

    if passed:
        logger.info("[LEAKAGE CHECK] PASSED -- no leakage detected.")
    else:
        logger.error(f"[LEAKAGE CHECK] FAILED -- {len(errors)} error(s): {errors}")

    for w in warnings:
        logger.warning(f"[LEAKAGE CHECK] {w}")

    return result


# ─── Feature catalogue ────────────────────────────────────────────────────────

def feature_catalogue(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame documenting every feature: name, type, null count,
    mean, std, and the group it belongs to.
    Useful for the Phase 3 report and for model explainability.
    """
    GROUPS = {
        "lag":      "A_temporal_lags",
        "roll":     "B_rolling_windows",
        "spatial":  "C_spatial_lags",
        "month":    "D_seasonality",
        "year":     "D_seasonality",
        "is_harm":  "D_seasonality",
        "is_farm":  "D_seasonality",
        "is_post":  "D_seasonality",
        "is_ramadan":     "E_islamic_calendar",
        "is_post_ramadan":"E_islamic_calendar",
        "reliability":    "F_coverage_proxy",
        "log_rel":        "F_coverage_proxy",
        "low_coverage":   "F_coverage_proxy",
        "coverage_lag":   "F_coverage_proxy",
        "disagree":       "G_source_disagreement",
        "abs_diff":       "G_source_disagreement",
        "ndvi":           "H_environmental_stub",
        "rainfall":       "H_environmental_stub",
        "enso":           "H_environmental_stub",
        "food":           "I_economic_stub",
        "fuel":           "I_economic_stub",
    }

    def assign_group(col):
        col_lower = col.lower()
        for key, group in GROUPS.items():
            if key in col_lower:
                return group
        return "Z_other"

    records = []
    for col in feature_matrix.columns:
        s = feature_matrix[col]
        def _safe(series, fn):
            try:
                return round(float(fn(series)), 4)
            except (TypeError, AttributeError, NotImplementedError):
                return None

        records.append({
            "feature":    col,
            "group":      assign_group(col),
            "dtype":      str(s.dtype),
            "null_count": int(s.isna().sum()),
            "null_pct":   round(float(s.isna().mean()) * 100, 1),
            "mean":       _safe(s, lambda x: x.mean(skipna=True)),
            "std":        _safe(s, lambda x: x.std(skipna=True)),
            "is_stub":    (bool(s.eq(0).all()) if s.dtype not in (object,) else False),
        })
    return pd.DataFrame(records).sort_values(["group", "feature"])

