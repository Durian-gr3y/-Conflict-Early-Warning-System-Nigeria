# Conflict Early Warning Nigeria (CEW-Nigeria)
> **Open-Method, Transparent Conflict Risk Intelligence Platform**  
> Focus: Nigeria (Northwest Pilot: Zamfara, Katsina, Sokoto, Kaduna · Scaling to all 36 States + FCT)

[![Tests: Passing](https://img.shields.io/badge/Tests-16%20Passed-10b981?style=flat-square)](#testing)
[![Models: LightGBM & NegBin](https://img.shields.io/badge/Models-LightGBM%20%7C%20NegBin%20GLM-3b82f6?style=flat-square)](#modelling)
[![Ethics: Do No Digital Harm](https://img.shields.io/badge/Ethics-Do%20No%20Digital%20Harm-f59e0b?style=flat-square)](#responsible-use)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-8b5cf6?style=flat-square)](#license)

---

## 📌 Executive Summary

**Conflict Early Warning Nigeria** is an open-method research and risk forecasting platform designed to provide transparent, empirically grounded monthly conflict early warning across Nigeria.

Unlike black-box commercial intelligence platforms, CEW-Nigeria:
1. **Publishes explicit exceedance probabilities** $P(\text{fatal conflict} \ge 1)$ with mandatory 90% confidence bands.
2. **Explicitly models coverage bias** using source disagreement and telecom reporting proxies so analysts do not mistake reporting silence for peace.
3. **Statistically validates models** via expanding-window backtesting and Diebold-Mariano significance testing.
4. **Enforces a strict Responsible Use Policy** barring individual profiling and kinetic military targeting.

---

## 🏗️ Phased Platform Architecture

```
Phase 1 [DONE] ──▶ Phase 2 [DONE] ──▶ Phase 3 [DONE] ──▶ Phase 4 [DONE] ──▶ Phase 5 [DONE] ──▶ Phase 6 [DONE]
Data EDA           Multi-Source       Feature Eng.       ML Models          Delivery           Ethics &
& audit            Pipeline           & Coverage Bias    & Backtesting      Dashboard          Scaling
```

| Phase | Core Objective | Key Deliverables |
|---|---|---|
| **Phase 1: EDA & Baseline** | Audit ACLED weekly panel, establish persistence baseline | [`reports/01_data_audit.md`](reports/01_data_audit.md), M0 Naive ($BS=0.3443$) |
| **Phase 2: Ingestion & Reliability** | Harmonize ACLED & UCDP GED, compute reliability index | [`reports/02_source_reliability.md`](reports/02_source_reliability.md), multi-source pipeline |
| **Phase 3: Feature Engineering** | 61 spatio-temporal features, coverage-bias modeling | [`data/features/feature_matrix.parquet`](data/features/), [`reports/03_coverage_bias.md`](reports/03_coverage_bias.md) |
| **Phase 4: ML Models & Backtesting** | NegBin GLM & LightGBM over 83 expanding-window folds | [`reports/04_model_card.md`](reports/04_model_card.md), Diebold-Mariano tests ($p<1e-4$) |
| **Phase 5: Delivery & Intelligence** | Streamlit dashboard, automated alert dispatcher, briefs | [`app/dashboard.py`](app/dashboard.py), NGO/Gov/Corporate briefs |
| **Phase 6: Ethics & Scaling** | Do No Harm client screening, config-driven scale-out | [`config/states.yaml`](config/states.yaml), [`reports/06_ethics_audit.md`](reports/06_ethics_audit.md) |

---

## 🚀 Quickstart

### 1. Installation
```bash
# Clone and enter workspace
git clone https://github.com/your-org/conflict-ew-nigeria.git
cd conflict-ew-nigeria

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Dashboard (Phase 5)
```bash
# Start Streamlit application
streamlit run app/dashboard.py
# Or via Makefile
make dashboard
```
Open **`http://localhost:8501`** in your browser to access the live risk map, Northwest pilot zoom, threshold alert dispatcher, and automated brief generator.

### 3. Run Test Suite
```bash
# Run all 16 unit tests (models, delivery, scaling)
python -m unittest tests/test_models.py tests/test_delivery.py tests/test_scaling.py
# Or via Makefile
make test
```

### 4. Full End-to-End Reproduction
```bash
# Executes Phases 1–6 sequentially from raw data to validated briefs
python notebooks/run_reproduce_all.py
```

---

## 📈 Empirical Model Performance

Evaluated via strict **expanding-window temporal backtesting** across 83 historical folds (2018–2024, 1,635 out-of-sample predictions, zero future leakage):

| Model | Model Class | Brier Score (lower = better) | Brier Skill Score (vs Clim) | AUC-ROC | Diebold-Mariano vs M0 |
|---|---|---|---|---|---|
| **Climatology** | Empirical Base Rate | `0.2442` | `0.00%` | `0.500` | Reference |
| **M0 — Naive** | Persistence ($y_T = \hat{p}_{T+1}$) | `0.3443` | `-41.03%` | `0.647` | Baseline Ceiling |
| **M1 — NegBin GLM** | Count GLM $\to$ Exceedance Prob | **`0.1827`** | **`+25.19%`** | **`0.751`** | **$DM = +17.97$ ($p < 1e-4$) ✅** |
| **M2 — LightGBM** | Gradient Boosted Trees | **`0.1813`** | **`+25.75%`** | **`0.741`** | **$DM = +20.86$ ($p < 1e-4$) ✅** |

> Both **M1 (0.1827)** and **M2 (0.1813)** achieve a **~47% error reduction** relative to the Naive baseline, verified as statistically significant by the Diebold-Mariano test.

---

## 🛡️ Responsible Use & Ethical Charter

This platform strictly adheres to [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) and the [Do No Digital Harm framework](https://www.thengoforum.org/do-no-digital-harm):
1. **Zero Kinetic Operations:** Outputs must never be used to direct airstrikes, artillery, or lethal kinetic security operations.
2. **No Profiling:** No individual personal data (PII) is stored or displayed; ethnic and religious profiling is prohibited.
3. **Mandatory Uncertainty:** Every risk estimate must be accompanied by its 90% confidence interval and reporting reliability score.
4. **Client Intake Screening:** Government and commercial clients are screened programmatically by [`src/delivery/screening.py`](src/delivery/screening.py) with an immutable audit log.

---

## 🗺️ Configuration-Driven Scale-Out

To monitor additional states or scale nationally:
1. Open [`config/states.yaml`](config/states.yaml).
2. Set `enabled: true` for any of the 36 states or FCT.
3. Centroid calculations, spatial distance matrices (IDW), and dashboard views update automatically with **zero code changes**.

---

## 📄 License & Attribution

- **Code:** Licensed under [MIT License](LICENSE).
- **Data Methodology & Reports:** Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Data Sources:** Armed Conflict Location & Event Data Project (ACLED, CC BY-NC-SA 4.0) and Uppsala Conflict Data Program (UCDP GED, CC BY 4.0).
