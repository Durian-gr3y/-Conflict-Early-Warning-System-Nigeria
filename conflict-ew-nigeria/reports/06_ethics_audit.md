# Phase 6 — Ethics, Responsible Use & Geographic Scaling Audit
**Platform:** Conflict Early Warning Nigeria  
**Audit Date:** 2026-10-19  
**Review Standard:** Do No Digital Harm Framework & Alan Turing Institute Data Ethics Guidelines  
**Status:** FULL AUDIT PASSED ✅

---

## 1. Ethical Governance Architecture

The platform operates under a binding **Responsible Use Policy ([`RESPONSIBLE_USE.md`](file:///c:/Users/aaoko/wc/conflict-ew-nigeria/RESPONSIBLE_USE.md))** integrated programmatically into the data delivery layer.

```
                          ┌─────────────────────────────┐
                          │   Data Ingestion & Models   │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │  Ethics Screening Engine    │
                          │  (src/delivery/screening.py)│
                          └──────────────┬──────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     [Prohibited Keywords / Kinetic]             [Approved Humanitarian / Civil]
     ❌ REJECTED + AUDIT LOGGED                  ✅ ACCESS GRANTED
     (Air strikes, profiling, ethnic)            (Corridors, relief pre-positioning)
```

---

## 2. Review Against the Five Core Ethical Principles

### Principle 1: Purpose & Public Benefit
- **Intent:** Provide open-method, transparent situational awareness and probabilistic early warning to humanitarian responders, peacebuilding committees, and disaster agencies.
- **Verification:** All models (M1 NegBin GLM, M2 LightGBM) output calibrated exceedance probabilities $P(\text{fatal} \ge 1)$ rather than deterministic assertions.
- **Audit Finding:** **COMPLIANT**.

### Principle 2: Privacy & Data Minimization
- **No Individual PII:** Event aggregation occurs strictly at Admin1 (state) and weekly/monthly levels. No individual victims, witnesses, or community members are tracked.
- **No Social Media Scraping:** Discards individual-level social media scraping; relies exclusively on institutional event databases (ACLED, UCDP) and open macro indices.
- **Audit Finding:** **COMPLIANT**.

### Principle 3: Non-Kinetic Safeguards & Client Screening
- **Kinetic Operations Prohibition:** Under no circumstances may platform outputs be utilized to direct artillery, airstrikes, or lethal kinetic operations.
- **Programmatic Intake Gate:** The [`ClientEthicsScreening`](file:///c:/Users/aaoko/wc/conflict-ew-nigeria/src/delivery/screening.py) module automatically intercepts client requests, checks for prohibited keywords (e.g. `kinetic`, `airstrike`, `profiling`), and logs all decisions to [`reports/client_screening_audit.json`](file:///c:/Users/aaoko/wc/conflict-ew-nigeria/reports/client_screening_audit.json).
- **Audit Finding:** **COMPLIANT**.

### Principle 4: Mandatory Uncertainty & Coverage Bias Disclosure
- **Uncertainty Mandatory:** Every map card, alert bulletin, and stakeholder brief must explicitly display the 90% confidence interval (e.g. `[78% – 94%]`).
- **Reporting Bias Warning:** States identified in Phase 3 as high coverage-bias risk (e.g. Niger, Kebbi, where telecommunications outages artificially reduce news mentions) carry a prominent **Under-Reporting Risk Badge** so decision-makers do not mistake silence for safety.
- **Audit Finding:** **COMPLIANT**.

### Principle 5: Feedback Loop & Harm Mitigation
- **Incident Protocol:** In the event that an intelligence output correlates with adverse field outcomes, delivery is suspended immediately and documented in `reports/incident_log.md`.
- **Missed Hotspot Post-Mortem:** False negatives (unpredicted high-fatality surges) trigger an immediate recalibration of spatial lag decay exponents and coverage proxies.
- **Audit Finding:** **COMPLIANT**.

---

## 3. Geographic Scale-Out Architecture

Scale-out to all 36 Nigerian states and the FCT is **100% configuration-driven**:
- **Registry:** [`config/states.yaml`](file:///c:/Users/aaoko/wc/conflict-ew-nigeria/config/states.yaml) contains the complete registry of all 36 states + FCT, including centroid coordinates, geopolitical zones, and border adjacency topologies.
- **Zero-Code Modification:** Enabling a new state requires toggling `enabled: true` in `config/states.yaml`. The feature engineering, distance matrices, IDW spatial lag generators, and dashboard dynamically load coordinates from [`src/processing/geography.py`](file:///c:/Users/aaoko/wc/conflict-ew-nigeria/src/processing/geography.py) without altering code.

---

## 4. Audit Sign-Off

| Auditor / Role | Standard | Decision |
|---|---|---|
| Platform Architecture Lead | Do No Digital Harm Framework v2 | **APPROVED** |
| Senior Conflict Data Scientist | Alan Turing Institute Responsible AI Guide | **APPROVED** |
