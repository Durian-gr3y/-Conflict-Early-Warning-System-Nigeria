# Phase 3 -- Feature Engineering & Coverage-Bias Report
**Generated:** 2026-10-19 00:02

## Feature matrix summary

| Metric | Value |
|--------|-------|
| Rows | 2,381 |
| Total columns | 61 |
| Predictor columns | 51 |
| Stub features (all-zero) | 5 |
| Leakage check | PASSED |

## Feature groups

| Group | Features | Stubs | Avg null% |
|-------|----------|-------|-----------|
| A_temporal_lags | 22 | 0 | 3.1% |
| B_rolling_windows | 12 | 0 | 0.9% |
| D_seasonality | 9 | 0 | 0.0% |
| E_islamic_calendar | 1 | 0 | 0.0% |
| F_coverage_proxy | 3 | 0 | 0.0% |
| G_source_disagreement | 1 | 0 | 0.0% |
| H_environmental_stub | 3 | 3 | 0.0% |
| I_economic_stub | 2 | 2 | 0.0% |
| Z_other | 8 | 0 | 0.1% |

## Autocorrelation of fatal_event

| Lag | Pearson r |
|-----|-----------|
| 1 month | {lag1_autocorr:.4f} |
| 3 months | {lag3_autocorr:.4f} |

### By pilot state (lag-1)

| State | Autocorrelation lag-1 |
|-------|----------------------|
{pilot_ac_md}

> Autocorrelation < 0.3 means the naive "last month = this month" model is weak.
> This is consistent with Phase 1 finding (BSS < 0 for naive model).
> The gradient-boosting model can combine spatial + rolling features to compensate.

## Coverage-bias risk cells

**Definition:** A state-month is HIGH RISK if:
1. `reliability_score` < 0.4 (low source coverage quality), AND
2. `fatal_event == 0` (recorded as quiet), AND
3. `spatial_events_lag1 > 0` (neighbouring state had events last month)

These cells most likely represent under-reporting rather than genuine peace.
Every forecast for these cells must carry a widened uncertainty band.

**Total high-risk state-months:** {len(high_risk):,}

### Top states by coverage-risk fraction

{top5_md}

### Interpretation for Northwest pilot states

States with `spatial_events_lag1 > 0` but `fatal_event == 0` and low reliability
are the most dangerous cells to treat as 'peaceful'. The model will include
`reliability_score` and `low_coverage_flag` as explicit features so it can
produce higher uncertainty estimates for these cells.

## Stub features to replace in Phase 3b

| Feature | Replace with | Source |
|---------|-------------|--------|
| `ndvi_anomaly` | MODIS NDVI via GEE | `src/ingestion/gee.py` (Phase 3b) |
| `rainfall_deficit` | CHIRPS monthly via GEE | `src/ingestion/gee.py` (Phase 3b) |
| `enso_phase` | NOAA ONI index | Public API |
| `food_price_dev` | WFP VAM API | `src/ingestion/wfp.py` (Phase 3b) |
| `fuel_price_dev` | DPR Nigeria / NNPC | Manual monthly |

## Files written
- `data/features/feature_matrix.parquet`
- `metrics/phase3_metrics.json`
- `reports/03_feature_catalogue.csv`
- `reports/fig10_spatial_idw_weights.png`
- `reports/fig11_coverage_risk_by_state.png`
- `reports/fig12_autocorrelation.png`
- `reports/fig13_feature_nulls_by_group.png`

## Next step (Phase 4)
Fit models M1 (negative-binomial GLM) and M2 (LightGBM) on this
feature matrix using an expanding-window backtest. Every model must
beat the naive baseline Brier score of **0.3331** (from Phase 1).
