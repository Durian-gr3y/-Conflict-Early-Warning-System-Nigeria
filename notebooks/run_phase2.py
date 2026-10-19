"""
notebooks/run_phase2.py
───────────────────────
Phase 2 — Multi-Source Ingestion & Processing Pipeline

What this script does (in order):
  1. Download UCDP GED for Nigeria  (or use a synthetic stub if offline)
  2. Parse and filter UCDP to Nigeria events
  3. Aggregate UCDP to monthly admin1 panel
  4. Load the ACLED monthly panel from Phase 1
  5. Compute source-reliability scores for each source × state × month
  6. Compute source-disagreement metrics (ACLED vs UCDP)
  7. Build the combined reliability table
  8. Save all outputs and write the Phase 2 report
  9. Run ACLED API dry-run demo (no key needed)

Run from project root:
    python notebooks/run_phase2.py
    python notebooks/run_phase2.py --download   # actually fetch UCDP

Done when:
  - data/processed/ucdp_nigeria_monthly.parquet exists
  - data/processed/reliability_table.parquet exists
  - metrics/phase2_metrics.json exists
  - reports/02_source_reliability.md exists
  - No assertion errors in the schema checks
"""

import argparse
import json
import logging
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Make src importable from the notebooks/ directory ─────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.ucdp import (
    download_ucdp_ged,
    load_ucdp_nigeria,
)
from src.ingestion.acled_api import ACLEDClient
from src.processing.source_reliability import (
    score_acled_agg,
    score_ucdp,
    build_reliability_table,
    compute_source_disagreement,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR     = ROOT / "data" / "raw"
PROC_DIR    = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "metrics"
REPORTS_DIR = ROOT / "reports"
for d in [RAW_DIR, PROC_DIR, METRICS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 0 · Argument parsing
# =============================================================================
parser = argparse.ArgumentParser(description="Phase 2 ingestion pipeline")
parser.add_argument(
    "--download", action="store_true",
    help="Actually download UCDP GED (requires internet). "
         "Without this flag the script runs with synthetic UCDP data."
)
parser.add_argument(
    "--ucdp-version", default="24.1",
    help="UCDP GED version to download (default: 24.1)"
)
parser.add_argument(
    "--force", action="store_true",
    help="Re-download even if files exist"
)
args = parser.parse_args()

DEMO_UCDP = not args.download
print("=" * 65)
print("Conflict Early Warning Nigeria -- Phase 2")
print("=" * 65)
if DEMO_UCDP:
    print("\n[WARN] Running in DEMO MODE for UCDP (synthetic data).")
    print("       Run with --download to fetch the real UCDP GED.\n")


# =============================================================================
# 1 · Load ACLED monthly panel from Phase 1
# =============================================================================
print("\n--- Loading ACLED panel from Phase 1 ---")

acled_path = PROC_DIR / "monthly_admin1_panel.parquet"
if not acled_path.exists():
    # CSV fallback
    acled_path = PROC_DIR / "monthly_admin1_panel.csv"
if not acled_path.exists():
    raise FileNotFoundError(
        f"Phase 1 output not found at {PROC_DIR}.\n"
        "Run 'python notebooks/run_phase1.py' first."
    )

if acled_path.suffix == ".parquet":
    acled = pd.read_parquet(acled_path)
else:
    acled = pd.read_csv(acled_path)
    acled["month"] = acled["month"].astype("period[M]") if "month" in acled.columns else acled["month"]

# Ensure month is Period
if "month" in acled.columns and not hasattr(acled["month"].dtype, "freq"):
    acled["month"] = acled["month"].astype(str).apply(
        lambda x: pd.Period(x, freq="M")
    )

print(f"ACLED panel: {len(acled):,} rows  |  "
      f"States: {acled['admin1'].nunique()}  |  "
      f"Months: {acled['month'].min()} --> {acled['month'].max()}")


# =============================================================================
# 2 · Load / synthesise UCDP data
# =============================================================================
print("\n--- UCDP GED ---")

ucdp_out_path = PROC_DIR / "ucdp_nigeria_monthly.parquet"

if DEMO_UCDP:
    # Synthesise a realistic UCDP Nigeria monthly panel
    # UCDP is narrower than ACLED: fewer events, higher fatality threshold,
    # roughly 1989-onward but we align to the ACLED date range
    rng = np.random.default_rng(99)

    months_acled = sorted(acled["month"].unique())
    states_acled = acled["admin1"].unique()

    HIGH_CONFLICT_STATES = [
        "Zamfara", "Katsina", "Sokoto", "Kaduna",
        "Borno", "Yobe", "Adamawa",
    ]

    rows = []
    for month in months_acled:
        for state in states_acled:
            # UCDP captures ~40-70% of ACLED events
            intensity = 0.65 if state in HIGH_CONFLICT_STATES else 0.25
            n = int(rng.negative_binomial(1, 1 / (1 + intensity * 0.6)))
            if n == 0:
                continue
            fat_best = int(rng.negative_binomial(n, 1 / (1 + intensity)))
            rows.append({
                "month": month,
                "admin1": state,
                "events_ucdp": n,
                "fatalities_ucdp": fat_best,
                "fatalities_low": max(0, fat_best - rng.integers(0, 3)),
                "fatalities_high": fat_best + rng.integers(0, 5),
                # where_prec 1-5; Northwest events tend to be less precise
                "mean_geo_prec": float(rng.integers(2, 6)
                                       if state in HIGH_CONFLICT_STATES
                                       else rng.integers(1, 4)),
                "pct_clear_events": float(rng.uniform(0.5, 1.0)),
                "source": "UCDP",
            })
    ucdp = pd.DataFrame(rows)
    ucdp["fatal_event_ucdp"] = (ucdp["fatalities_ucdp"] > 0).astype(int)
    print(f"Synthetic UCDP panel: {len(ucdp):,} rows")

else:
    # Real download path
    csv_path = download_ucdp_ged(
        version=args.ucdp_version,
        dest_dir=RAW_DIR,
        force=args.force,
    )
    ucdp = load_ucdp_nigeria(
        raw_dir=RAW_DIR,
        version=args.ucdp_version,
        include_candidate=False,
    )
    print(f"Real UCDP panel: {len(ucdp):,} rows")

# Save UCDP panel
try:
    ucdp.to_parquet(ucdp_out_path, index=False)
except Exception:
    ucdp_out_path = PROC_DIR / "ucdp_nigeria_monthly.csv"
    ucdp.to_csv(ucdp_out_path, index=False)
print(f"Saved: {ucdp_out_path}")


# =============================================================================
# 3 · Source-reliability scoring
# =============================================================================
print("\n--- Source-reliability scoring ---")

# 3a. ACLED weekly aggregate
acled_scores = score_acled_agg(acled, weeks_per_month=4)
print(f"ACLED scores computed: {len(acled_scores):,} cells")
print(f"  Mean reliability: {acled_scores['reliability_acled_agg'].mean():.3f}")
print(f"  Score distribution:\n{acled_scores['reliability_label'].value_counts().sort_index().to_string()}")

# 3b. UCDP
ucdp_scores = score_ucdp(ucdp, acled)
print(f"\nUCDP scores computed: {len(ucdp_scores):,} cells")
print(f"  Mean reliability: {ucdp_scores['reliability_ucdp'].mean():.3f}")
print(f"  Score distribution:\n{ucdp_scores['reliability_label'].value_counts().sort_index().to_string()}")

# 3c. Combined table
rel_table = build_reliability_table(acled_scores, ucdp_scores)
print(f"\nCombined reliability table: {len(rel_table):,} cells")
print(f"  Source coverage breakdown:\n{rel_table['sources_available'].value_counts().to_string()}")

# Save
rel_path = PROC_DIR / "reliability_table.parquet"
try:
    rel_table.to_parquet(rel_path, index=False)
except Exception:
    rel_path = PROC_DIR / "reliability_table.csv"
    rel_table.to_csv(rel_path, index=False)
print(f"Saved: {rel_path}")


# =============================================================================
# 4 · Source-disagreement analysis
# =============================================================================
print("\n--- Source disagreement (ACLED vs UCDP) ---")

# Align month types for merge
def normalise_month(df):
    """Ensure 'month' column is Period[M] for joining."""
    df = df.copy()
    if "month" in df.columns:
        df["month"] = df["month"].astype(str).apply(
            lambda x: pd.Period(x, freq="M")
        )
    return df

disagree = compute_source_disagreement(
    normalise_month(acled),
    normalise_month(ucdp),
)

print(f"Overlapping state-months: {len(disagree):,}")
print(f"Agreement on fatal event presence: {disagree['agree_on_presence'].mean():.1%}")
print(f"Mean absolute fatality difference:  {disagree['abs_diff_fatalities'].mean():.1f}")
print(f"Disagreement flags raised:          {disagree['disagreement_flag'].sum():,} "
      f"({disagree['disagreement_flag'].mean():.1%} of cells)")

# Top-10 most disagreed cells
top_disagree = (
    disagree.sort_values("abs_diff_fatalities", ascending=False)
    .head(10)[["month", "admin1", "fat_acled", "fat_ucdp",
               "abs_diff_fatalities", "disagreement_flag"]]
)
print("\nTop-10 largest fatality disagreements:")
print(top_disagree.to_string(index=False))


# =============================================================================
# 5 · Visualisations
# =============================================================================
print("\n--- Generating figures ---")

# Fig 07 — ACLED reliability heatmap (pilot states)
PILOT_STATES = ["Zamfara", "Katsina", "Sokoto", "Kaduna"]
pilot_rel = (
    acled_scores[acled_scores["admin1"].isin(PILOT_STATES)]
    .copy()
)
# Convert Period to string for pivot
pilot_rel["month_str"] = pilot_rel["month"].astype(str)
pivot = (
    pilot_rel.pivot_table(
        index="admin1", columns="month_str",
        values="reliability_acled_agg", aggfunc="mean"
    )
)
fig, ax = plt.subplots(figsize=(18, 3))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.8)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
# Show every 12th month on x-axis to avoid crowding
xtick_idx = list(range(0, len(pivot.columns), 12))
ax.set_xticks(xtick_idx)
ax.set_xticklabels([pivot.columns[i] for i in xtick_idx], rotation=45, ha="right")
ax.set_title("ACLED weekly-agg reliability score — Northwest pilot states\n"
             "Green=high coverage quality, Red=low")
plt.colorbar(im, ax=ax, fraction=0.015, label="Reliability (0-1)")
plt.tight_layout()
fig.savefig(REPORTS_DIR / "fig07_acled_reliability_heatmap.png", dpi=150)
plt.close()
print("Saved: fig07_acled_reliability_heatmap.png")

# Fig 08 — Source disagreement by state (bar chart)
state_disagree = (
    disagree.groupby("admin1")
    .agg(
        pct_disagree=("disagreement_flag", "mean"),
        mean_abs_diff=("abs_diff_fatalities", "mean"),
    )
    .sort_values("pct_disagree", ascending=False)
    .head(15)
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
state_disagree["pct_disagree"].sort_values().plot.barh(
    ax=axes[0], color="#e74c3c")
axes[0].set_xlabel("Fraction of months with disagreement")
axes[0].set_title("ACLED vs UCDP: % months with fatal-event disagreement")
axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

state_disagree["mean_abs_diff"].sort_values().plot.barh(
    ax=axes[1], color="#e67e22")
axes[1].set_xlabel("Mean absolute fatality difference")
axes[1].set_title("ACLED vs UCDP: mean |fatalities_ACLED - fatalities_UCDP|")
plt.suptitle("Source disagreement by state", fontsize=12)
plt.tight_layout()
fig.savefig(REPORTS_DIR / "fig08_source_disagreement.png", dpi=150)
plt.close()
print("Saved: fig08_source_disagreement.png")

# Fig 09 — Reliability distribution, both sources
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
acled_scores["reliability_acled_agg"].hist(ax=axes[0], bins=30, color="#2980b9")
axes[0].set_title("ACLED reliability score distribution")
axes[0].set_xlabel("Score (0-1)")
ucdp_scores["reliability_ucdp"].hist(ax=axes[1], bins=30, color="#27ae60")
axes[1].set_title("UCDP reliability score distribution")
axes[1].set_xlabel("Score (0-1)")
plt.tight_layout()
fig.savefig(REPORTS_DIR / "fig09_reliability_distributions.png", dpi=150)
plt.close()
print("Saved: fig09_reliability_distributions.png")


# =============================================================================
# 6 · ACLED API dry-run demo
# =============================================================================
print("\n--- ACLED API client demo (dry run) ---")
client = ACLEDClient(dry_run=True)
client.fetch_nigeria_events(
    start_date="2024-01-01",
    end_date="2024-12-31",
    event_types=["Battles", "Violence against civilians"],
)
print("  (No key needed for dry run — shows what request would be sent)")


# =============================================================================
# 7 · Save metrics + audit report
# =============================================================================
print("\n--- Saving metrics and report ---")

metrics = {
    "phase": 2,
    "run_date": datetime.now().isoformat(),
    "demo_ucdp": DEMO_UCDP,
    "acled_panel": {
        "rows": int(len(acled)),
        "states": int(acled["admin1"].nunique()),
        "months": f"{acled['month'].min()} --> {acled['month'].max()}",
    },
    "ucdp_panel": {
        "rows": int(len(ucdp)),
        "states": int(ucdp["admin1"].nunique()),
        "months": f"{ucdp['month'].min()} --> {ucdp['month'].max()}",
    },
    "reliability_acled_agg": {
        "mean_score": round(float(acled_scores["reliability_acled_agg"].mean()), 4),
        "pct_low":    round(float((acled_scores["reliability_label"] == "Low").mean()), 4),
        "pct_high":   round(float((acled_scores["reliability_label"].isin(
                            ["High", "Very High"])).mean()), 4),
    },
    "reliability_ucdp": {
        "mean_score": round(float(ucdp_scores["reliability_ucdp"].mean()), 4),
        "pct_low":    round(float((ucdp_scores["reliability_label"] == "Low").mean()), 4),
        "pct_high":   round(float((ucdp_scores["reliability_label"].isin(
                            ["High", "Very High"])).mean()), 4),
    },
    "source_disagreement": {
        "overlapping_cells": int(len(disagree)),
        "pct_agree_on_presence": round(float(disagree["agree_on_presence"].mean()), 4),
        "mean_abs_diff_fatalities": round(float(disagree["abs_diff_fatalities"].mean()), 3),
        "pct_disagreement_flagged": round(float(disagree["disagreement_flag"].mean()), 4),
    },
}

metrics_path = METRICS_DIR / "phase2_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: {metrics_path}")
print(json.dumps(metrics, indent=2))

# ── Markdown report ────────────────────────────────────────────────────────────
report = f"""# Phase 2 -- Source Ingestion & Reliability Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**UCDP mode:** {"DEMO (synthetic)" if DEMO_UCDP else "REAL DATA"}

## Data sources ingested

| Source | Rows | States | Date range |
|--------|------|--------|------------|
| ACLED (weekly agg) | {len(acled):,} | {acled['admin1'].nunique()} | {acled['month'].min()} -- {acled['month'].max()} |
| UCDP GED | {len(ucdp):,} | {ucdp['admin1'].nunique()} | {ucdp['month'].min()} -- {ucdp['month'].max()} |

## Source-reliability scoring method

### ACLED weekly aggregate
Geometric mean of two sub-scores:

| Sub-score | Formula | Notes |
|-----------|---------|-------|
| S1 temporal | weeks_observed / 4 | 4 expected weeks/month |
| S2 geo | Fixed 0.60 | State centroid penalty; loses sub-state precision |

**Mean reliability: {metrics['reliability_acled_agg']['mean_score']:.3f}**

### UCDP GED
Geometric mean of three sub-scores:

| Sub-score | Formula | Notes |
|-----------|---------|-------|
| S1 cross-source | UCDP_active / ACLED_active x 0.70 prior | Adjusted for UCDP's stricter threshold |
| S2 geo | 1 - (mean_where_prec - 1)/6 | Maps prec 1->1.0, prec 7->0.0 |
| S3 clarity | pct events with clarity==1 | Fraction with clearly documented events |

**Mean reliability: {metrics['reliability_ucdp']['mean_score']:.3f}**

> **Calibration note:** These scores are expert-judgement indices, not statistically
> calibrated probabilities of completeness. Label them 'coverage quality index'
> in user-facing materials. Phase 3 will regress them against GDELT media density.

## Source disagreement (ACLED vs UCDP)

| Metric | Value |
|--------|-------|
| Overlapping state-months | {metrics['source_disagreement']['overlapping_cells']:,} |
| Agreement on fatal event (>=1 death) | {metrics['source_disagreement']['pct_agree_on_presence']:.1%} |
| Mean absolute fatality difference | {metrics['source_disagreement']['mean_abs_diff_fatalities']:.1f} |
| Cells flagged as disagreeing | {metrics['source_disagreement']['pct_disagreement_flagged']:.1%} |

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
"""

report_path = REPORTS_DIR / "02_source_reliability.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Saved: {report_path}")

print("\n" + "=" * 65)
print("Phase 2 COMPLETE")
print("=" * 65)
print("\n[OK] What we learned:")
print("  1. UCDP covers fewer state-months than ACLED (expected)")
print("  2. Reliability scores are now attached to every cell")
print("  3. Disagreement flags mark cells needing wider uncertainty bands")
print(f"  4. ACLED-UCDP agreement on fatal events: "
      f"{metrics['source_disagreement']['pct_agree_on_presence']:.1%}")
print("\n[NEXT] Phase 3 -- Feature engineering + coverage-bias model")
print("       Run: python notebooks/run_phase3.py  (coming next)")
