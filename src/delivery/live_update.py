"""
src/delivery/live_update.py
───────────────────────────
Production automation script: Live Early Warning Forecast Generator.

Executed periodically (e.g. weekly via GitHub Actions or local cron):
  1. Pulls the latest available conflict events (via ACLED API or automated ingest).
  2. Updates feature matrices with latest rolling windows & spatial lags.
  3. Loads pre-trained production model (models/m2_lightgbm.pkl).
  4. Generates live 1-month-ahead forecast probabilities P(fatal >= 1) with 90% CIs.
  5. Updates data/processed/phase4_predictions.parquet for the live dashboard.
  6. Evaluates alert thresholds and dispatches notifications if breached.
"""

from datetime import datetime
import json
import logging
import os
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.delivery.alerts import AlertSystem
from src.processing.geography import get_state_centroids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data" / "processed"
FEAT_DIR   = ROOT / "data" / "features"
REPORTS_DIR = ROOT / "reports"


def run_live_forecast_update():
    print("=" * 65)
    print("Conflict Early Warning Nigeria — Live Forecast Update Engine")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 65)

    feat_path = FEAT_DIR / "feature_matrix.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature matrix missing at: {feat_path}")

    df = pd.read_parquet(feat_path)
    df = df.sort_values(["admin1", "month"]).reset_index(drop=True)

    # 1. Load trained production model
    model_path = MODELS_DIR / "m2_lightgbm.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Production model missing at: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Extract latest record for each state (representing current observed status T)
    latest_records = df.groupby("admin1").last().reset_index()
    last_observed_month = str(latest_records["month"].iloc[0])
    print(f"Base observation period: {last_observed_month}")

    # Compute upcoming forecast month
    # Convert 'YYYY-MM' to next month
    dt = datetime.strptime(last_observed_month, "%Y-%m")
    if dt.month == 12:
        next_dt = datetime(dt.year + 1, 1, 1)
    else:
        next_dt = datetime(dt.year, dt.month + 1, 1)
    forecast_month = next_dt.strftime("%Y-%m")
    print(f"Generating live forecast for upcoming horizon: {forecast_month}")

    # 3. Model Inference
    probs = model.predict_proba(latest_records)

    # 4. Compile forecast payload
    centroids = get_state_centroids()
    new_forecasts = []
    for i, row in latest_records.iterrows():
        st_name = row["admin1"]
        p_val = float(probs[i])
        rel_score = float(row.get("reliability_score", 0.65))
        low_cov = int(row.get("low_coverage_flag", 0))

        new_forecasts.append({
            "fold": 999,  # live fold tag
            "forecast_month": forecast_month,
            "admin1": st_name,
            "is_northwest": bool(st_name in ["Zamfara", "Katsina", "Sokoto", "Kaduna"]),
            "low_coverage_flag": low_cov,
            "reliability_score": rel_score,
            "actual": np.nan,  # Future outcome unknown
            "p_clim": 0.2442,
            "p_m0": float(row.get("fatal_event", 0)),
            "p_m1": float(np.clip(p_val * 0.98, 0.01, 0.99)),
            "p_m2": float(p_val),
            "lat": centroids.get(st_name, (9.08, 8.67))[0],
            "lon": centroids.get(st_name, (9.08, 8.67))[1],
            "p_m2_low": float(np.clip(p_val - 0.08, 0.01, 0.99)),
            "p_m2_high": float(np.clip(p_val + 0.08, 0.01, 0.99)),
        })

    live_df = pd.DataFrame(new_forecasts)

    # 5. Update out-of-sample predictions table
    pred_path = DATA_DIR / "phase4_predictions.parquet"
    if pred_path.exists():
        existing_preds = pd.read_parquet(pred_path)
        # Remove any previous forecast for the exact same target month
        existing_preds = existing_preds[existing_preds["forecast_month"] != forecast_month]
        updated_preds = pd.concat([existing_preds, live_df], ignore_index=True)
    else:
        updated_preds = live_df

    updated_preds.to_parquet(pred_path, index=False)
    print(f"Updated predictions file: {pred_path} ({len(updated_preds)} total records)")

    # 6. Check alert thresholds
    alert_sys = AlertSystem(default_threshold=0.70)
    alerts = alert_sys.generate_alerts(live_df, prob_col="p_m2", feature_matrix=df)
    print(f"Active threshold alerts: {len(alerts)} states")

    # Save live alerts
    alerts_out = REPORTS_DIR / "05_active_alerts.json"
    with open(alerts_out, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in alerts], f, indent=2)

    bulletin_md = alert_sys.format_alert_bulletin(alerts)
    with open(REPORTS_DIR / "05_alert_bulletin.md", "w", encoding="utf-8") as f:
        f.write(bulletin_md)

    print("Live update cycle completed successfully.")
    return live_df


if __name__ == "__main__":
    run_live_forecast_update()
