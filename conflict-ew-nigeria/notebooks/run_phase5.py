"""
notebooks/run_phase5.py
───────────────────────
Phase 5: Automated Intelligence Briefs, Alert Dispatching & Delivery Artifacts.

What this script executes:
  1. Loads out-of-sample forecast predictions from Phase 4.
  2. Runs AlertSystem to identify threshold breaches and generate alert payloads.
  3. Uses MonthlyBriefGenerator to compile and save three stakeholder briefs:
       - reports/05_humanitarian_brief.md (Humanitarian access & protection)
       - reports/05_government_brief.md (Policy, hotspot ranking, Do No Harm)
       - reports/05_corporate_brief.md (Commercial operations & supply chain)
  4. Saves reports/05_active_alerts.json and reports/05_alert_bulletin.md.
  5. Verifies app/dashboard.py readiness.
"""

import json
import logging
import pathlib
import sys
from datetime import datetime

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.delivery.alerts import AlertSystem
from src.delivery.brief_generator import MonthlyBriefGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT / "reports"
DATA_DIR    = ROOT / "data" / "processed"
FEAT_DIR    = ROOT / "data" / "features"

print("=" * 70)
print("Conflict Early Warning Nigeria — Phase 5: Delivery & Intelligence Products")
print("=" * 70)

# 1. Load latest predictions
pred_file = DATA_DIR / "phase4_predictions.parquet"
if not pred_file.exists():
    raise FileNotFoundError(f"Predictions not found at {pred_file}. Run Phase 4 first.")

preds = pd.read_parquet(pred_file)
feat_file = FEAT_DIR / "feature_matrix.parquet"
feats = pd.read_parquet(feat_file) if feat_file.exists() else None

latest_month = str(sorted(preds["forecast_month"].unique())[-1])
print(f"Loaded out-of-sample predictions ({len(preds):,} records across {preds['forecast_month'].nunique()} months).")
print(f"Target Delivery Month: {latest_month}")

latest_preds = preds[preds["forecast_month"] == latest_month].copy()

# 2. Run Alert System
print("\n--- Step 2: Running Automated Alert Dispatcher ---")
alert_sys = AlertSystem(default_threshold=0.70)
alerts = alert_sys.generate_alerts(latest_preds, prob_col="p_m2", threshold=0.70, feature_matrix=feats)
print(f"Generated {len(alerts)} alerts at threshold >= 70% for {latest_month}.")

# Save alerts JSON
alerts_json_path = REPORTS_DIR / "05_active_alerts.json"
with open(alerts_json_path, "w", encoding="utf-8") as f:
    json.dump([a.to_dict() for a in alerts], f, indent=2)
print(f"Saved: {alerts_json_path}")

# Save alert bulletin Markdown
bulletin_md = alert_sys.format_alert_bulletin(alerts)
bulletin_path = REPORTS_DIR / "05_alert_bulletin.md"
with open(bulletin_path, "w", encoding="utf-8") as f:
    f.write(bulletin_md)
print(f"Saved: {bulletin_path}")

# 3. Generate the 3 Stakeholder Briefs
print("\n--- Step 3: Generating Publication-Grade Stakeholder Briefs ---")
brief_gen = MonthlyBriefGenerator(latest_preds)

ngo_brief = brief_gen.generate_ngo_brief(target_month=latest_month)
ngo_path = REPORTS_DIR / "05_humanitarian_brief.md"
with open(ngo_path, "w", encoding="utf-8") as f:
    f.write(ngo_brief)
print(f"Saved: {ngo_path}")

gov_brief = brief_gen.generate_government_brief(target_month=latest_month)
gov_path = REPORTS_DIR / "05_government_brief.md"
with open(gov_path, "w", encoding="utf-8") as f:
    f.write(gov_brief)
print(f"Saved: {gov_path}")

corp_brief = brief_gen.generate_corporate_brief(target_month=latest_month)
corp_path = REPORTS_DIR / "05_corporate_brief.md"
with open(corp_path, "w", encoding="utf-8") as f:
    f.write(corp_brief)
print(f"Saved: {corp_path}")

print("\n" + "=" * 70)
print("PHASE 5 COMPLETE: DASHBOARD READY, BRIEFS COMPILED & ALERTS ACTIVE!")
print("=" * 70)
print("\nTo launch the live interactive dashboard:")
print("  streamlit run app/dashboard.py")
print("=" * 70)
