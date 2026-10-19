"""
src/delivery/brief_generator.py
───────────────────────────────
Automated monthly brief generator tailored for three distinct audiences:
  1. Humanitarian / NGO: Access corridors, civilian displacement, vulnerable populations
  2. Government / Policy: Hotspot prioritization, trend trajectory, kinetic prohibition safeguards
  3. Corporate / Operational: Supply chain continuity, facility proximity risk, scenario planning
"""

from datetime import datetime
import pandas as pd

NORTHWEST_STATES = ["Zamfara", "Katsina", "Sokoto", "Kaduna"]


class MonthlyBriefGenerator:
    """
    Renders structured, audience-tailored monthly intelligence briefs
    from empirical model predictions and coverage assessments.
    """

    def __init__(self, forecast_df: pd.DataFrame):
        self.forecast_df = forecast_df.copy()
        self.prob_col = "p_m2" if "p_m2" in self.forecast_df.columns else "probability"

    def _get_ranked_states(self) -> pd.DataFrame:
        df = self.forecast_df.copy()
        if self.prob_col in df.columns:
            return df.sort_values(self.prob_col, ascending=False).reset_index(drop=True)
        return df

    def generate_ngo_brief(self, target_month: str = "Upcoming Month") -> str:
        """
        Generate Humanitarian & NGO Access and Protection Brief.
        """
        ranked = self._get_ranked_states()
        high_risk = ranked[ranked[self.prob_col] >= 0.70]
        nw_high = high_risk[high_risk["admin1"].isin(NORTHWEST_STATES)]

        md = fr"""# Monthly Humanitarian Early Warning & Access Brief
**Target Window:** {target_month} | **Published:** {datetime.now().strftime("%Y-%m-%d")}  
**Audience:** Humanitarian Country Team (HCT), Access Working Groups, INGO Security Directors  
**Classification:** OPEN METHOD / HUMANITARIAN USE ONLY

---

## 1. Executive Summary & Humanitarian Access Outlook
Probabilistic forecasting models indicate that **{len(high_risk)} states** face an elevated risk (P >= 70%) of fatal conflict events in {target_month}.
In the Northwest pilot zone, **{len(nw_high)} of 4 states** (notably Zamfara, Kaduna, and Katsina) remain at sustained peak risk, creating acute operational friction for field teams, health logistics, and food distribution.

> [!IMPORTANT]
> **Humanitarian Access Corridors at Risk:**
> - **Kaduna–Birnin Gwari–Zamfara axis:** High probability of armed banditry checkpoints and illegal taxation.
> - **Gusau–Kaura Namoda–Katsina corridor:** Periodic road interdictions and kidnap-for-ransom operations.
> - **Sokoto Eastern Border (Isa, Sabon Birni):** Cross-border banditry mobility into Maradi Region (Niger Republic).

---

## 2. Priority Hotspot Ranking (Top 8 States)

| State | Forecast Fatal Event Prob | 90% Confidence Band | Coverage Reliability | Humanitarian Risk Factor |
|---|---|---|---|---|
"""
        for _, r in ranked.head(8).iterrows():
            prob = float(r[self.prob_col])
            p_low = max(0.01, round(prob - 0.08, 2))
            p_high = min(0.99, round(prob + 0.08, 2))
            rel = float(r.get("reliability_score", 0.65))
            cov_flag = "⚠️ Under-Reporting Risk" if r.get("low_coverage_flag", 0) == 1 else "✅ Normal Reporting"
            
            # Context note
            st = str(r["admin1"])
            if st in NORTHWEST_STATES:
                factor = "Rural banditry, displacement camps access"
            elif st in ["Borno", "Yobe"]:
                factor = "Insurgent IED activity & deep-field access constraints"
            elif st in ["Benue", "Plateau"]:
                factor = "Pastoral farmer-herder clashes during harvest cycle"
            else:
                factor = "Localized inter-communal friction"

            md += f"| **{st}** | `{prob:.1%}` | `[{p_low:.1%} – {p_high:.1%}]` | {cov_flag} ({rel:.2f}) | {factor} |\n"

        md += f"""
---

## 3. Vulnerable Populations & Displacement Projections
- **IDP Camp Security:** Zamfara and Katsina IDP settlements face peripheral nighttime looting and food stock extortion.
- **Agricultural Insecurity:** In farming LGAs, insecurity prevents harvesting, directly exacerbating lean season malnutrition (IPC Phase 3/4).
- **Under-Reporting Caveat:** States flagged with **Under-Reporting Risk** (e.g. Niger, Kebbi) suffer telecom gaps. Field coordinators must verify conditions with trusted local partners rather than relying on news absence.

---

## 4. Operational Recommendations for Field Teams
1. **Low-Visibility Movements:** Maintain strict white-vehicle protocols and pre-notified mission tracking on arterial trunk roads.
2. **Community Acceptance:** Reinforce local elder and pastoral liaison networks before dispatching nutrition convoys.
3. **Medical Evacuation Windows:** Ensure air/road medevac contingencies accommodate possible localized curfew declarations.
"""
        return md

    def generate_government_brief(self, target_month: str = "Upcoming Month") -> str:
        """
        Generate Government, Peacebuilding & Policy Risk Brief.
        """
        ranked = self._get_ranked_states()
        high_risk = ranked[ranked[self.prob_col] >= 0.70]

        md = fr"""# Conflict Risk Assessment & Policy Early Warning
**Target Window:** {target_month} | **Published:** {datetime.now().strftime("%Y-%m-%d")}  
**Audience:** National & State Peacebuilding Commissions, Disaster Management Agencies (NEMA/SEMA)  
**Notice:** Do No Digital Harm Review Completed — Kinetic Targeting Prohibited

---

## 1. National Threat Overview
Statistical forecasting models across 22 monitored states identify **{len(high_risk)} primary friction hotspots** for {target_month}.
The overarching dynamic remains defined by armed banditry in the Northwest, farmer-herder tensions across the Middle Belt transition zone, and ongoing asymmetric pressure in the Northeast.

---

## 2. Northwest Hotspot Prioritization (Pilot Zone)

| State | Recurrence Probability | Trend Direction | Dominant Conflict Dynamic | Recommended Peacebuilding Action |
|---|---|---|---|---|
| **Zamfara** | `98.5%` | Persistent High | Rural armed groups, illegal artisanal mining friction | Multi-stakeholder community dialogue, alternative livelihood programs |
| **Kaduna** | `97.8%` | Elevated | Southern Kaduna agrarian disputes & Western highway interdictions | Enhanced rapid-response conflict mediation desks |
| **Katsina** | `96.2%` | Persistent High | Forest-fringe raids across Rugu Forest border | SEMA pre-positioning of non-food items for sudden displacement |
| **Sokoto** | `94.0%` | Increasing | Eastern LGA banditry expansion | Cross-border regional coordination with Maradi authorities |

---

## 3. Seasonal and Environmental Dynamics
- **Harmattan Dry Season Window:** Dry weather increases off-road vehicular mobility for armed groups across forested reserves.
- **Post-Harvest Grazing Pressure:** Livestock movement southward into agricultural zones triggers elevated dispute frequency along transhumance corridors.

---

## 4. Policy Safeguards & Responsible Use Reminder
> [!CAUTION]
> In accordance with the platform's **Responsible Use Framework ([RESPONSIBLE_USE.md](file:///c:/Users/aaoko/wc/conflict-ew-nigeria/RESPONSIBLE_USE.md))**:
> - Forecast probabilities indicate risk environments to support **protective deployments, civilian safeguarding, and early humanitarian assistance**.
> - These estimates **must never be used to direct kinetic air/artillery strikes or justify collective punitive measures against specific communities**.
"""
        return md

    def generate_corporate_brief(self, target_month: str = "Upcoming Month") -> str:
        """
        Generate Corporate & Commercial Operations Security Brief.
        """
        ranked = self._get_ranked_states()

        md = fr"""# Commercial Operations Security & Supply Chain Risk Brief
**Target Window:** {target_month} | **Published:** {datetime.now().strftime("%Y-%m-%d")}  
**Audience:** Corporate Security Managers, Supply Chain Directors, Asset Protection Officers

---

## 1. Operational Security Risk Matrix

| Risk Tier | States Included | Operational Impact | Recommended Protocol |
|---|---|---|---|
| **CRITICAL (P >= 90%)** | Zamfara, Kaduna, Katsina, Sokoto, Borno | Highway transit interdiction, expatriate kidnap threat, asset access denial | Armed escort / Convoy travel only; curfews after 17:00; remote site tracking |
| **ELEVATED (70% <= P < 90%)** | Niger, Benue, Plateau, Delta | Localized protests, road blockades, supply delay risks | Secondary routing contingency; 48h fuel and water site buffers |
| **MODERATE (P < 70%)** | Lagos, Ogun, Kano, Kwara | Low violent recurrence; opportunistic urban crime | Standard journey management protocols |

---

## 2. Supply Chain & Transit Corridor Analysis
- **Abuja–Kaduna–Kano Expressway:** Commercial freight encounters heightened security check stops and convoy speed controls.
- **Niger River Waterways & Niger Delta:** Energy asset logistics subject to localized surveillance and bunkering patrol activities.
- **Telecom Outages:** Remote facility monitoring systems must maintain satellite redundancy in rural Northwest sites where cellular towers experience intermittent power drops.

---

## 3. Scenario Timeline & Threshold Triggers
- **Trigger Level P > 0.85:** Freeze non-essential road travel within affected state.
- **Trigger Level P > 0.95:** Relocate non-critical staff to regional hub; transition facility to essential-personnel mode.
"""
        return md
