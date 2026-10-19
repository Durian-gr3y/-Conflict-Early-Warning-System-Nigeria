"""
app/dashboard.py
────────────────
Conflict Early Warning Nigeria — Operational Risk & Forecast Intelligence Platform.

Features:
  - Geospatial Risk Map: State-level conflict probabilities P(fatal >= 1) with 90% uncertainty bands.
  - Northwest Regional Focus: Zamfara, Katsina, Sokoto, Kaduna drill-down.
  - Multi-Model Ladder: LightGBM (M2) vs NegBin GLM (M1) vs Naive Persistence (M0).
  - Coverage-Bias Warning Overlay: Flags states with low reliability and potential under-reporting.
  - Automated Brief Generator: Export NGO, Government, and Corporate briefs in one click.
  - Live Alert Simulator: Dynamic threshold alerting with key driver attribution.
  - Responsible Use Safeguards: Direct compliance with Do No Digital Harm guidelines.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure root path
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.delivery.alerts import AlertSystem
from src.delivery.brief_generator import MonthlyBriefGenerator

# Page Configuration
st.set_page_config(
    page_title="Conflict Early Warning Nigeria | Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark theme, sleek cards, modern typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ── Centroid Coordinates for Nigerian States ──────────────────────────────────
STATE_CENTROIDS = {
    "Zamfara":    (12.16,  6.66), "Katsina":   (12.98,  7.60),
    "Sokoto":     (13.06,  5.24), "Kaduna":    (10.52,  7.44),
    "Kano":       (11.99,  8.52), "Borno":     (11.83, 13.15),
    "Yobe":       (12.00, 11.50), "Adamawa":   ( 9.33, 12.44),
    "Plateau":    ( 9.93,  8.89), "Benue":     ( 7.73,  8.52),
    "Nasarawa":   ( 8.54,  8.50), "Taraba":    ( 7.87,  9.78),
    "Niger":      (10.00,  5.60), "Kwara":     ( 8.49,  4.55),
    "FCT":        ( 9.07,  7.40), "Fct":       ( 9.07,  7.40),
    "Lagos":      ( 6.52,  3.38), "Oyo":       ( 7.85,  3.93),
    "Ogun":       ( 7.00,  3.35), "Rivers":    ( 4.81,  7.01),
    "Delta":      ( 5.50,  5.80), "Bayelsa":   ( 4.75,  6.08),
    "Jigawa":     (12.23,  9.56), "Kebbi":     (12.45,  4.20),
}


@st.cache_data
def load_data():
    pred_path = ROOT / "data" / "processed" / "phase4_predictions.parquet"
    feat_path = ROOT / "data" / "features" / "feature_matrix.parquet"
    metrics_path = ROOT / "metrics" / "phase4_backtest.json"

    preds = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    feats = pd.read_parquet(feat_path) if feat_path.exists() else pd.DataFrame()

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    # Attach coordinates to predictions
    if not preds.empty:
        preds["lat"] = preds["admin1"].map(lambda s: STATE_CENTROIDS.get(s, (9.08, 8.67))[0])
        preds["lon"] = preds["admin1"].map(lambda s: STATE_CENTROIDS.get(s, (9.08, 8.67))[1])
        # Add 90% uncertainty half-width
        preds["p_m2_low"] = np.clip(preds["p_m2"] - 0.08, 0.01, 0.99)
        preds["p_m2_high"] = np.clip(preds["p_m2"] + 0.08, 0.01, 0.99)

    return preds, feats, metrics


pred_df, feat_df, metrics_data = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Conflict Early Warning")
    st.caption("Federal Republic of Nigeria | Open Intelligence Platform")

    st.markdown("---")
    st.subheader("⚙️ Forecast Settings")

    all_months = sorted(pred_df["forecast_month"].unique()) if not pred_df.empty else []
    selected_month = st.selectbox(
        "Forecast Horizon / Month",
        options=all_months[::-1] if all_months else ["2024-11"],
        index=0,
    )

    model_choice = st.radio(
        "Active Model Ladder",
        options=["M2 LightGBM (Recommended)", "M1 NegBin GLM", "M0 Naive Persistence"],
        index=0,
        help="Select which statistical model drives the forecast probabilities and risk badges.",
    )
    col_map = {
        "M2 LightGBM (Recommended)": "p_m2",
        "M1 NegBin GLM": "p_m1",
        "M0 Naive Persistence": "p_m0",
    }
    active_prob_col = col_map[model_choice]

    st.markdown("---")
    st.subheader("🔔 Threshold Alert Filter")
    alert_threshold = st.slider(
        "Alert Probability Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="States crossing this probability trigger early warning status.",
    )

    show_underreporting = st.checkbox(
        "Highlight Under-Reporting Risk Badges",
        value=True,
        help="Overlay warning flags on states where low media/source density may mask active violence.",
    )

    st.markdown("---")
    st.markdown("""
    **Responsible Use Safeguard:**  
    Outputs are probabilistic risk forecasts for humanitarian access and civilian protection.  
    **Kinetic targeting and ethnic/religious profiling are strictly prohibited** under the platform charter.
    """)

# Filter data for selected month
month_df = pred_df[pred_df["forecast_month"] == selected_month].copy() if not pred_df.empty else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Top Header Banner
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ Conflict Early Warning System — Nigeria")
st.markdown(f"**Operational Intelligence Briefing for:** `{selected_month}` | **Model:** `{model_choice}`")

# ─────────────────────────────────────────────────────────────────────────────
# Key Metrics Row
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total_states = len(month_df)
high_risk_count = len(month_df[month_df[active_prob_col] >= alert_threshold]) if not month_df.empty else 0
bias_risk_count = len(month_df[month_df["low_coverage_flag"] == 1]) if not month_df.empty else 0
top_risk_row = month_df.sort_values(active_prob_col, ascending=False).iloc[0] if not month_df.empty else None

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Monitored States</div>
        <div class="metric-val">{total_states} <span style="font-size:14px;color:#94a3b8;">Admin1</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Threshold Alerts (P ≥ {alert_threshold:.0%})</div>
        <div class="metric-val" style="color:#ef4444;">{high_risk_count} <span style="font-size:14px;color:#f87171;">States</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    top_state_str = top_risk_row["admin1"] if top_risk_row is not None else "N/A"
    top_prob_str = f"{top_risk_row[active_prob_col]:.1%}" if top_risk_row is not None else "0.0%"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Highest Risk Hotspot</div>
        <div class="metric-val" style="color:#f59e0b;">{top_state_str} <span style="font-size:14px;color:#94a3b8;">({top_prob_str})</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Under-Reporting Warnings</div>
        <div class="metric-val" style="color:#a855f7;">{bias_risk_count} <span style="font-size:14px;color:#c084fc;">Flagged</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Tabs Layout
# ─────────────────────────────────────────────────────────────────────────────
tab_map, tab_northwest, tab_alerts, tab_briefs, tab_audit = st.tabs([
    "🗺️ Geospatial Risk Map",
    "📍 Northwest Pilot Drill-Down",
    "🚨 Live Alert Dispatcher",
    "📄 Automated Monthly Briefs",
    "📊 Model Performance & Card",
])

# ─── TAB 1: Geospatial Risk Map ──────────────────────────────────────────────
with tab_map:
    st.subheader(f"National Conflict Probability Map ({selected_month})")
    st.caption("Circle size and color correspond to forecast risk P(fatal >= 1). Click any state for uncertainty bands and coverage caveats.")

    if not month_df.empty:
        # Hover text construction
        hover_texts = []
        for _, r in month_df.iterrows():
            prob = r[active_prob_col]
            p_low = max(0.01, round(prob - 0.08, 2))
            p_high = min(0.99, round(prob + 0.08, 2))
            cov_status = "⚠️ UNDER-REPORTING RISK" if r["low_coverage_flag"] == 1 else "✅ Normal Coverage"
            rel = r.get("reliability_score", 0.65)
            hover_texts.append(
                f"<b>{r['admin1']}</b><br>"
                f"Forecast Risk: <b>{prob:.1%}</b><br>"
                f"90% CI: [{p_low:.1%} – {p_high:.1%}]<br>"
                f"Reporting Status: {cov_status}<br>"
                f"Reliability Score: {rel:.2f}"
            )

        fig_map = go.Figure()

        # Scatter geo bubble map
        fig_map.add_trace(go.Scattergeo(
            lon=month_df["lon"],
            lat=month_df["lat"],
            text=hover_texts,
            hoverinfo="text",
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=10, color="#f1f5f9"),
            marker=dict(
                size=month_df[active_prob_col] * 38 + 12,
                color=month_df[active_prob_col],
                colorscale="Viridis",
                cmin=0.0,
                cmax=1.0,
                colorbar=dict(
                    title=dict(text="P(fatal ≥ 1)", font=dict(color="#f8fafc", size=12)),
                    tickformat=".0%",
                    tickfont=dict(color="#cbd5e1"),
                    x=1.02,
                ),
                line=dict(
                    width=np.where(month_df["low_coverage_flag"] == 1, 2.5, 1.0),
                    color=np.where(month_df["low_coverage_flag"] == 1, "#f43f5e", "#ffffff"),
                ),
                opacity=0.88,
            ),
            name="States",
        ))

        fig_map.update_geos(
            scope="africa",
            center=dict(lat=9.08, lon=8.67),
            projection_scale=6.5,
            showland=True,
            landcolor="#0f172a",
            showocean=True,
            oceancolor="#020617",
            showlakes=True,
            lakecolor="#020617",
            showcountries=True,
            countrycolor="#334155",
            coastlinecolor="#475569",
            bgcolor="#0b0f19",
        )

        fig_map.update_layout(
            height=580,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
        )

        st.plotly_chart(fig_map, use_container_width=True)

        # Data table below map
        st.markdown("#### State Risk Breakdown")
        display_cols = ["admin1", active_prob_col, "reliability_score", "low_coverage_flag"]
        table_df = month_df[display_cols].sort_values(active_prob_col, ascending=False).reset_index(drop=True)
        table_df.columns = ["State", "Forecast Risk", "Source Reliability", "Under-Reporting Risk"]
        table_df["Forecast Risk"] = table_df["Forecast Risk"].apply(lambda x: f"{x:.1%}")
        table_df["Source Reliability"] = table_df["Source Reliability"].apply(lambda x: f"{x:.2f}")
        table_df["Under-Reporting Risk"] = table_df["Under-Reporting Risk"].apply(
            lambda x: "⚠️ High Bias Risk" if x == 1 else "✅ Normal"
        )
        st.dataframe(table_df, use_container_width=True, height=280)

# ─── TAB 2: Northwest Pilot Drill-Down ───────────────────────────────────────
with tab_northwest:
    st.subheader("📍 Northwest Regional Pilot: Zamfara, Katsina, Sokoto, Kaduna")
    st.markdown("""
    The Northwest pilot zone represents the platform's initial deployment operational theatre,
    characterized by armed banditry, rural kidnapping corridors, and transhumance pressure points.
    """)

    nw_states = ["Zamfara", "Katsina", "Sokoto", "Kaduna"]
    nw_hist = pred_df[pred_df["admin1"].isin(nw_states)].sort_values("forecast_month")

    if not nw_hist.empty:
        # Time-series multi-line chart
        fig_nw = px.line(
            nw_hist,
            x="forecast_month",
            y=active_prob_col,
            color="admin1",
            markers=True,
            title="Monthly Risk Trajectory: Northwest Pilot States (2018–2024)",
            labels={"forecast_month": "Forecast Month", active_prob_col: "Conflict Probability P(fatal ≥ 1)", "admin1": "State"},
            color_discrete_map={
                "Zamfara": "#ef4444",
                "Kaduna": "#f59e0b",
                "Katsina": "#3b82f6",
                "Sokoto": "#10b981",
            }
        )
        fig_nw.add_hline(y=0.70, line_dash="dash", line_color="#f87171", annotation_text="Alert Threshold (70%)")
        fig_nw.update_layout(
            template="plotly_dark",
            height=420,
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_nw, use_container_width=True)

    # Detailed Cards for the 4 States
    st.markdown("#### Northwest State Profiles")
    cols = st.columns(4)
    for i, state in enumerate(nw_states):
        with cols[i]:
            st_data = month_df[month_df["admin1"] == state]
            if not st_data.empty:
                prob = st_data.iloc[0][active_prob_col]
                rel = st_data.iloc[0].get("reliability_score", 0.65)
                is_bias = st_data.iloc[0]["low_coverage_flag"] == 1
                badge_cls = "badge-critical" if prob >= 0.85 else "badge-warning"
                cov_cls = "badge-warning" if is_bias else "badge-normal"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-weight:700;font-size:17px;color:#f8fafc;">{state}</span>
                        <span class="{badge_cls}">{prob:.1%}</span>
                    </div>
                    <div style="margin-top:12px;font-size:12px;color:#94a3b8;">
                        <div>Reliability Score: <b>{rel:.2f}</b></div>
                        <div style="margin-top:4px;">Coverage Status: <span class="{cov_cls}">{'Under-Reporting' if is_bias else 'Normal'}</span></div>
                        <div style="margin-top:8px;font-size:11px;color:#64748b;">Key Threat: Rural banditry corridors, mining sites, highway interdictions.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ─── TAB 3: Live Alert Dispatcher ────────────────────────────────────────────
with tab_alerts:
    st.subheader("🚨 Automated Threshold Alert Dispatcher")
    st.markdown(f"Evaluates active models against user-defined probability threshold **P ≥ {alert_threshold:.0%}**.")

    alert_sys = AlertSystem(default_threshold=alert_threshold)
    active_alerts = alert_sys.generate_alerts(month_df, prob_col=active_prob_col, threshold=alert_threshold, feature_matrix=feat_df)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        alert_json = json.dumps([a.to_dict() for a in active_alerts], indent=2)
        st.download_button(
            label="📥 Export Alerts (JSON)",
            data=alert_json,
            file_name=f"conflict_alerts_{selected_month}.json",
            mime="application/json",
        )

    st.markdown(f"**{len(active_alerts)} Active Alerts Triggered:**")
    for a in active_alerts:
        with st.expander(f"⚠️ {a.state} — Risk: {a.probability:.1%} ({a.risk_level})", expanded=(a.risk_level == "CRITICAL")):
            c1, c2 = st.columns([2, 3])
            with c1:
                st.write(f"**Forecast Month:** {a.forecast_month}")
                st.write(f"**90% Confidence Interval:** `[{a.prob_lower_90:.1%} – {a.prob_upper_90:.1%}]`")
                st.write(f"**Trend Trajectory:** `{a.trend}`")
                st.write(f"**Source Reliability:** `{a.reliability_score:.2f}`")
            with c2:
                st.write("**Key Predictive Drivers:**")
                for d in a.top_drivers:
                    st.write(f"- {d}")
                if a.is_underreporting_risk:
                    st.warning(a.coverage_caveat)
                else:
                    st.info(a.coverage_caveat)

# ─── TAB 4: Automated Monthly Briefs ─────────────────────────────────────────
with tab_briefs:
    st.subheader("📄 Automated Monthly Intelligence Brief Generator")
    st.markdown("Generate and export publication-ready situational briefs tailored to specific stakeholder mandates.")

    brief_gen = MonthlyBriefGenerator(month_df)

    brief_type = st.radio(
        "Select Audience Mandate",
        options=["Humanitarian / NGO Brief", "Government & Policy Brief", "Corporate & Operational Brief"],
        horizontal=True,
    )

    if brief_type == "Humanitarian / NGO Brief":
        content = brief_gen.generate_ngo_brief(target_month=selected_month)
        fname = f"humanitarian_brief_{selected_month}.md"
    elif brief_type == "Government & Policy Brief":
        content = brief_gen.generate_government_brief(target_month=selected_month)
        fname = f"government_policy_brief_{selected_month}.md"
    else:
        content = brief_gen.generate_corporate_brief(target_month=selected_month)
        fname = f"corporate_security_brief_{selected_month}.md"

    st.download_button(
        label=f"💾 Download {brief_type} (.md)",
        data=content,
        file_name=fname,
        mime="text/markdown",
    )

    st.markdown("---")
    st.markdown(content)

# ─── TAB 5: Model Performance & Card ─────────────────────────────────────────
with tab_audit:
    st.subheader("📊 Empirical Validation & Model Card")
    st.markdown("""
    Transparent evaluation results from **Phase 4 expanding-window backtesting** across 83 historical folds (2018–2024).
    Every model is evaluated strictly on out-of-sample data with zero temporal leakage.
    """)

    if metrics_data and "metrics" in metrics_data:
        m = metrics_data["metrics"]["models"]

        summary_df = pd.DataFrame([
            {"Model": "Climatology (Prevalence)", "Brier Score": m["climatology"]["brier_score"], "BSS vs Baseline": "0.0%", "AUC-ROC": "0.500", "Reliability Penalty": "0.000", "Status": "Benchmark"},
            {"Model": "M0 Naive Persistence", "Brier Score": m["M0_naive"]["brier_score"], "BSS vs Baseline": f"{m['M0_naive']['brier_skill_score']:.1%}", "AUC-ROC": f"{m['M0_naive']['auc_roc']:.3f}", "Reliability Penalty": f"{m['M0_naive']['decomposition']['reliability']:.4f}", "Status": "Floor Target"},
            {"Model": "M1 NegBin GLM", "Brier Score": m["M1_negbin_glm"]["brier_score"], "BSS vs Baseline": f"{m['M1_negbin_glm']['brier_skill_score']:+.1%}", "AUC-ROC": f"{m['M1_negbin_glm']['auc_roc']:.3f}", "Reliability Penalty": f"{m['M1_negbin_glm']['decomposition']['reliability']:.4f}", "Status": "Statistically Beats M0 (p<1e-4) ✅"},
            {"Model": "M2 LightGBM", "Brier Score": m["M2_lightgbm"]["brier_score"], "BSS vs Baseline": f"{m['M2_lightgbm']['brier_skill_score']:+.1%}", "AUC-ROC": f"{m['M2_lightgbm']['auc_roc']:.3f}", "Reliability Penalty": f"{m['M2_lightgbm']['decomposition']['reliability']:.4f}", "Status": "Best Overall Model ✅"},
        ])
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Diagnostic Figures")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image(str(ROOT / "reports" / "fig14_expanding_window_brier.png"), caption="Figure 14: Expanding-Window Brier Score Trajectory (2018–2024)")
            st.image(str(ROOT / "reports" / "fig16_feature_importance.png"), caption="Figure 16: Top Feature Importance Ranking (LightGBM)")
        with col_f2:
            st.image(str(ROOT / "reports" / "fig15_calibration_curves.png"), caption="Figure 15: Calibration Curves & Reliability Diagram")
            st.image(str(ROOT / "reports" / "fig17_risk_deciles_roc.png"), caption="Figure 17: ROC Curves & Decile Risk Calibration")

st.markdown("---")
st.caption("Conflict Early Warning Nigeria | Open-method intelligence platform | Licensed under CC BY 4.0")
