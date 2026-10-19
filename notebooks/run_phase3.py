"""
notebooks/run_phase3.py
───────────────────────
Phase 3 -- Feature Engineering & Coverage-Bias Modelling

What this script does:
  1. Load Phase 1 (ACLED monthly panel) and Phase 2 (reliability, disagreement)
  2. Build the full feature matrix via src/features/feature_engineering.py
  3. Run the leakage check -- hard stop if any errors
  4. Produce the feature catalogue (name, group, nulls, stats)
  5. Identify coverage-bias risk cells:
       states x periods where both reliability and UCDP coverage are low
  6. Visualise feature distributions and spatial lag weights
  7. Save feature matrix to data/features/feature_matrix.parquet
  8. Write reports/03_coverage_bias.md

Done when:
  - Leakage check passes (no errors)
  - data/features/feature_matrix.parquet exists
  - reports/03_coverage_bias.md names specific state-period combinations
  - Feature catalogue is complete for all columns
"""

import json
import logging
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.feature_engineering import (
    build_feature_matrix,
    build_distance_matrix,
    idw_weights,
    leakage_check,
    feature_catalogue,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROC_DIR    = ROOT / "data" / "processed"
FEAT_DIR    = ROOT / "data" / "features"
METRICS_DIR = ROOT / "metrics"
REPORTS_DIR = ROOT / "reports"
for d in [FEAT_DIR, METRICS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("Conflict Early Warning Nigeria -- Phase 3")
print("=" * 65)


# =============================================================================
# 1 · Load panels from Phases 1 and 2
# =============================================================================
print("\n--- Loading Phase 1 + Phase 2 outputs ---")

def load_parquet_or_csv(stem: str) -> pd.DataFrame:
    for ext in [".parquet", ".csv"]:
        p = PROC_DIR / (stem + ext)
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(
        f"Neither {stem}.parquet nor {stem}.csv found in {PROC_DIR}.\n"
        "Run Phase 1 and Phase 2 first."
    )

acled = load_parquet_or_csv("monthly_admin1_panel")
print(f"ACLED panel      : {len(acled):,} rows")

try:
    rel_table = load_parquet_or_csv("reliability_table")
    print(f"Reliability table: {len(rel_table):,} rows")
except FileNotFoundError:
    rel_table = None
    print("[WARN] Reliability table not found -- coverage proxy will use fallback.")

# Source disagreement table is inside the Phase 2 outputs;
# re-derive it from the UCDP and ACLED panels if not cached separately.
try:
    ucdp = load_parquet_or_csv("ucdp_nigeria_monthly")
    from src.processing.source_reliability import compute_source_disagreement

    def normalise_period(df):
        df = df.copy()
        df["month"] = df["month"].astype(str).apply(
            lambda x: pd.Period(x, freq="M"))
        return df

    disagree_table = compute_source_disagreement(
        normalise_period(acled), normalise_period(ucdp)
    )
    print(f"Disagreement table: {len(disagree_table):,} rows  |  "
          f"Agreement: {disagree_table['agree_on_presence'].mean():.1%}")
except Exception as e:
    disagree_table = None
    print(f"[WARN] Could not build disagreement table: {e}")

# Ensure ACLED month is Period
acled["month"] = acled["month"].astype(str).apply(
    lambda x: pd.Period(x, freq="M"))


# =============================================================================
# 2 · Build feature matrix
# =============================================================================
print("\n--- Building feature matrix ---")
print("    (spatial lags require pairwise distance computation -- ~30s)")

fm = build_feature_matrix(
    monthly_panel=acled,
    reliability_table=rel_table,
    disagree_table=disagree_table,
    include_spatial=True,
    idw_power=2.0,
)

print(f"\nFeature matrix: {fm.shape[0]:,} rows x {fm.shape[1]} columns")


# =============================================================================
# 3 · Leakage check -- hard stop on errors
# =============================================================================
print("\n--- Leakage check ---")
check = leakage_check(fm)

print(f"Passed : {check['passed']}")
if check["errors"]:
    print("\n[CRITICAL] LEAKAGE ERRORS:")
    for err in check["errors"]:
        print(f"  - {err}")
    raise SystemExit(
        "Leakage check FAILED. Fix errors before proceeding to modelling."
    )
if check["warnings"]:
    print("Warnings:")
    for w in check["warnings"]:
        print(f"  [WARN] {w}")

print("[OK] Leakage check passed -- no future information detected.")


# =============================================================================
# 4 · Feature catalogue
# =============================================================================
print("\n--- Feature catalogue ---")
cat = feature_catalogue(fm)
cat_path = REPORTS_DIR / "03_feature_catalogue.csv"
cat.to_csv(cat_path, index=False)
print(f"Saved: {cat_path}  ({len(cat)} features)")

# Print group summary
group_summary = cat.groupby("group").agg(
    n_features=("feature", "count"),
    n_stubs=("is_stub", "sum"),
    avg_null_pct=("null_pct", "mean"),
).round(1)
print("\nFeature groups:")
print(group_summary.to_string())

# Identify model-ready predictors (not target, not raw outcomes)
FORBIDDEN = {"fatal_event_next", "fatal_event", "events", "fatalities",
             "weeks_observed", "low_coverage", "latitude", "longitude",
             "month", "admin1", "source"}
PREDICTOR_COLS = [c for c in fm.columns if c not in FORBIDDEN
                  and not c.startswith("fatalities_low")
                  and not c.startswith("fatalities_high")]
print(f"\nPotential predictor columns: {len(PREDICTOR_COLS)}")
print(f"  Target column             : fatal_event_next")
print(f"  Stub features (all-zero)  : "
      f"{cat[cat['is_stub']]['feature'].tolist()}")


# =============================================================================
# 5 · Coverage-bias risk identification
# =============================================================================
print("\n--- Coverage-bias risk analysis ---")

# A cell is HIGH RISK if:
#   - reliability_score < 0.4       (low source coverage quality)
#   - fatal_event == 0              (recorded as quiet)
#   - At least 1 neighbouring state had a fatal event in the same month
#     (spatial context suggests conflict was nearby)

# Condition 1 + 2
low_rel_quiet = fm[
    (fm["reliability_score"] < 0.4) &
    (fm["fatal_event"] == 0)
][["month", "admin1", "reliability_score", "fatal_event",
   "spatial_events_lag1"]].copy()

# Condition 3: non-zero spatial activity (neighbours were active)
high_risk = low_rel_quiet[
    low_rel_quiet["spatial_events_lag1"].fillna(0) > 0
].copy()
high_risk["risk_label"] = "HIGH: low coverage + quiet + active neighbours"

# Count by state
risk_by_state = (
    high_risk.groupby("admin1")
    .size()
    .sort_values(ascending=False)
    .rename("high_risk_months")
)
total_months = fm.groupby("admin1").size().rename("total_months")
risk_pct = (risk_by_state / total_months * 100).round(1).rename("risk_pct")
risk_summary = pd.concat([risk_by_state, total_months, risk_pct], axis=1).dropna()

print("\nHigh-coverage-risk state-months (quiet + low reliability + active neighbours):")
print(risk_summary.sort_values("risk_pct", ascending=False).to_string())


# =============================================================================
# 6 · Visualisations
# =============================================================================
print("\n--- Generating figures ---")

PILOT_STATES = ["Zamfara", "Katsina", "Sokoto", "Kaduna"]

# Fig 10 -- Spatial IDW weight heatmap
states_all = sorted(fm["admin1"].unique())
dist_mat = build_distance_matrix(states_all)
w_mat    = idw_weights(dist_mat)

# Show subset for readability
pilot_weights = w_mat.loc[PILOT_STATES, :].round(3)
fig, ax = plt.subplots(figsize=(16, 3))
sns.heatmap(
    pilot_weights,
    cmap="Blues",
    linewidths=0.3,
    ax=ax,
    cbar_kws={"label": "IDW weight"},
    xticklabels=True,
    yticklabels=True,
)
ax.set_title("Spatial IDW weights from Northwest pilot states to all others\n"
             "(higher = more influence on spatial lag feature)")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.tight_layout()
fig.savefig(REPORTS_DIR / "fig10_spatial_idw_weights.png", dpi=150)
plt.close()
print("Saved: fig10_spatial_idw_weights.png")

# Fig 11 -- Coverage risk map (bar chart by state)
if len(risk_summary) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    risk_summary["risk_pct"].sort_values().plot.barh(ax=ax, color="#c0392b")
    ax.set_xlabel("% of months classified as high coverage-risk")
    ax.set_title("Coverage-bias risk by state\n"
                 "(quiet + low reliability + active neighbours)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "fig11_coverage_risk_by_state.png", dpi=150)
    plt.close()
    print("Saved: fig11_coverage_risk_by_state.png")

# Fig 12 -- Temporal lag autocorrelation (pilot states)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, state in enumerate(PILOT_STATES):
    ax = axes[i // 2, i % 2]
    d = fm[fm["admin1"] == state].sort_values("month")
    lag_cols = [f"fatal_event_lag{n}" for n in [1, 2, 3, 6, 12]
                if f"fatal_event_lag{n}" in d.columns]
    corrs = [d["fatal_event"].corr(d[col]) for col in lag_cols]
    lags  = [int(c.split("lag")[1]) for c in lag_cols]
    ax.bar(lags, corrs, color="#2980b9", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{state} — autocorrelation of fatal_event")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Pearson r")
    ax.set_xticks(lags)
plt.suptitle("Temporal autocorrelation by pilot state\n"
             "Bars show corr(fatal_event_T, fatal_event_T-n)",
             fontsize=11)
plt.tight_layout()
fig.savefig(REPORTS_DIR / "fig12_autocorrelation.png", dpi=150)
plt.close()
print("Saved: fig12_autocorrelation.png")

# Fig 13 -- Feature null % by group
cat_no_stub = cat[~cat["is_stub"]]
null_by_group = cat_no_stub.groupby("group")["null_pct"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
null_by_group.plot.barh(ax=ax, color="#8e44ad")
ax.set_xlabel("Mean null % across features in group")
ax.set_title("Average missingness by feature group\n(stubs excluded)")
ax.axvline(20, color="red", linestyle="--", label="20% threshold")
ax.legend()
plt.tight_layout()
fig.savefig(REPORTS_DIR / "fig13_feature_nulls_by_group.png", dpi=150)
plt.close()
print("Saved: fig13_feature_nulls_by_group.png")


# =============================================================================
# 7 · Save feature matrix
# =============================================================================
fm_path = FEAT_DIR / "feature_matrix.parquet"
try:
    # Convert Period to string for Parquet compatibility
    fm_save = fm.copy()
    fm_save["month"] = fm_save["month"].astype(str)
    fm_save.to_parquet(fm_path, index=False)
except Exception as e:
    fm_path = FEAT_DIR / "feature_matrix.csv"
    fm.copy().assign(month=fm["month"].astype(str)).to_csv(fm_path, index=False)
    print(f"  (Parquet failed: {e} — saved as CSV)")

print(f"\nSaved: {fm_path}")


# =============================================================================
# 8 · Metrics + coverage-bias report
# =============================================================================
lag1_autocorr = float(fm["fatal_event"].corr(fm["fatal_event_lag1"]))
lag3_autocorr = float(fm["fatal_event"].corr(fm["fatal_event_lag3"]))

metrics = {
    "phase": 3,
    "run_date": datetime.now().isoformat(),
    "feature_matrix": {
        "rows": int(fm.shape[0]),
        "columns": int(fm.shape[1]),
        "predictor_cols": len(PREDICTOR_COLS),
        "stub_cols": int(cat["is_stub"].sum()),
    },
    "leakage_check": {
        "passed": check["passed"],
        "errors": check["errors"],
    },
    "autocorrelation": {
        "fatal_event_lag1": round(lag1_autocorr, 4),
        "fatal_event_lag3": round(lag3_autocorr, 4),
    },
    "coverage_bias": {
        "high_risk_cells_total": int(len(high_risk)),
        "states_at_risk": risk_summary.index.tolist(),
        "top_risk_state": risk_summary["risk_pct"].idxmax()
                          if len(risk_summary) > 0 else "none",
        "top_risk_pct": float(risk_summary["risk_pct"].max())
                        if len(risk_summary) > 0 else 0.0,
    },
}
metrics_path = METRICS_DIR / "phase3_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: {metrics_path}")
print(json.dumps(metrics, indent=2))

# ── Markdown report ────────────────────────────────────────────────────────────
# Top risk states for report
top5 = (risk_summary.sort_values("risk_pct", ascending=False)
        .head(5)
        .reset_index())
top5_md = "\n".join(
    f"- **{r.admin1}**: {r.risk_pct:.0f}% of months ({r.high_risk_months:.0f}/{r.total_months:.0f}) — "
    f"treat all 'quiet' forecasts here with extra caution"
    for _, r in top5.iterrows()
) if len(top5) > 0 else "- None identified"

# Pilot state lag1 autocorrelations
pilot_ac = {}
for state in PILOT_STATES:
    d = fm[fm["admin1"] == state]
    if "fatal_event_lag1" in d.columns:
        pilot_ac[state] = round(float(d["fatal_event"].corr(d["fatal_event_lag1"])), 3)

pilot_ac_md = "\n".join(
    f"| {s} | {v} |" for s, v in pilot_ac.items()
)

report = f"""# Phase 3 -- Feature Engineering & Coverage-Bias Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Feature matrix summary

| Metric | Value |
|--------|-------|
| Rows | {fm.shape[0]:,} |
| Total columns | {fm.shape[1]} |
| Predictor columns | {len(PREDICTOR_COLS)} |
| Stub features (all-zero) | {int(cat['is_stub'].sum())} |
| Leakage check | {"PASSED" if check['passed'] else "FAILED"} |

## Feature groups

| Group | Features | Stubs | Avg null% |
|-------|----------|-------|-----------|
""" + "\n".join(
    f"| {idx} | {int(r.n_features)} | {int(r.n_stubs)} | {r.avg_null_pct:.1f}% |"
    for idx, r in group_summary.iterrows()
) + """

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
"""

report_path = REPORTS_DIR / "03_coverage_bias.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Saved: {report_path}")

print("\n" + "=" * 65)
print("Phase 3 COMPLETE")
print("=" * 65)
print("\n[OK] What we learned:")
print(f"  1. {len(PREDICTOR_COLS)} predictor features built, leakage check passed")
print(f"  2. fatal_event lag-1 autocorrelation: {lag1_autocorr:.3f} "
      f"(low -- spatial + rolling features add value)")
print(f"  3. {len(high_risk):,} state-months flagged as coverage-bias HIGH RISK")
print(f"  4. {int(cat['is_stub'].sum())} stub features await real environmental/economic data")
print(f"\n[NEXT] Phase 4 -- Models M1 (NB-GLM) + M2 (LightGBM) + expanding-window backtest")
print("       Run: python notebooks/run_phase4.py")
