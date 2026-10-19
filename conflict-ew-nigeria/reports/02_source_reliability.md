# Phase 2 -- Source Ingestion & Reliability Report
**Generated:** 2026-10-18 23:50
**UCDP mode:** DEMO (synthetic)

## Data sources ingested

| Source | Rows | States | Date range |
|--------|------|--------|------------|
| ACLED (weekly agg) | 2,381 | 22 | 2015-01 -- 2024-12 |
| UCDP GED | 467 | 22 | 2015-01 -- 2024-12 |

## Source-reliability scoring method

### ACLED weekly aggregate
Geometric mean of two sub-scores:

| Sub-score | Formula | Notes |
|-----------|---------|-------|
| S1 temporal | weeks_observed / 4 | 4 expected weeks/month |
| S2 geo | Fixed 0.60 | State centroid penalty; loses sub-state precision |

**Mean reliability: 0.562**

### UCDP GED
Geometric mean of three sub-scores:

| Sub-score | Formula | Notes |
|-----------|---------|-------|
| S1 cross-source | UCDP_active / ACLED_active x 0.70 prior | Adjusted for UCDP's stricter threshold |
| S2 geo | 1 - (mean_where_prec - 1)/6 | Maps prec 1->1.0, prec 7->0.0 |
| S3 clarity | pct events with clarity==1 | Fraction with clearly documented events |

**Mean reliability: 0.422**

> **Calibration note:** These scores are expert-judgement indices, not statistically
> calibrated probabilities of completeness. Label them 'coverage quality index'
> in user-facing materials. Phase 3 will regress them against GDELT media density.

## Source disagreement (ACLED vs UCDP)

| Metric | Value |
|--------|-------|
| Overlapping state-months | 430 |
| Agreement on fatal event (>=1 death) | 55.1% |
| Mean absolute fatality difference | 10.6 |
| Cells flagged as disagreeing | 44.9% |

## Interpretation

- **Agreement < 80%** signals meaningful source disagreement -- forecasts for
  those cells should carry widened uncertainty bands.
- **UCDP consistently lower than ACLED** is expected (stricter inclusion) but
  if UCDP records *more* fatalities in a cell, investigate: may indicate
  ACLED missed a large battle.
- **Low ACLED reliability + low UCDP reliability in the same cell** = highest
  uncertainty. Flag these as 'interpret with caution' in all outputs.

## Files written
- `data/processed/ucdp_nigeria_monthly.parquet`
- `data/processed/reliability_table.parquet`
- `metrics/phase2_metrics.json`
- `reports/fig07_acled_reliability_heatmap.png`
- `reports/fig08_source_disagreement.png`
- `reports/fig09_reliability_distributions.png`

## Limitations
1. UCDP annual release lags ACLED by ~12 months -- disagreement for recent
   months cannot be computed until the next UCDP release.
2. The UCDP_COVERAGE_PRIOR (0.70) is a fixed assumption; it should be
   estimated empirically in Phase 3 using news article density.
3. Admin1 name normalisation is imperfect -- check 'unmapped' warnings in
   the log for any UCDP states that didn't match the canonical table.

## Next step (Phase 3)
Build the feature matrix: temporal lags, spatial lags, environmental
variables, and the coverage proxy from GDELT media density.
