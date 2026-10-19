# Phase 1 — Data Audit Report
**Generated:** 2026-10-18 23:38
**Mode:** DEMO (synthetic data)

## Dataset summary
| Field | Value |
|-------|-------|
| Raw rows | 7,573 |
| Monthly panel rows | 2,381 |
| States | 22 |
| Date range | 2015-01 → 2024-12 |
| Target prevalence | 56.3% |
| Low-coverage cells | 62.2% |

## Naive baseline metrics (M0)
| Metric | Value | Notes |
|--------|-------|-------|
| Brier score | 0.3383 | Lower is better; 0.25 = coin flip |
| Brier Skill Score | -0.3752 | >0 = beats climatology |
| AUC-ROC | 0.6563 | 0.5=random; 1.0=perfect |
| Expanding-window Brier | 0.3331 | PRIMARY benchmark for Phase 4 |

> **All Phase 4 models must achieve a lower expanding-window Brier score than the value above.**

## Coverage-bias flags

### ⚠️ Flag 1 — Silent week fraction > 30%
- `Nasarawa`: 64% silent weeks
- `Kebbi`: 64% silent weeks
- `Adamawa`: 63% silent weeks
- `Oyo`: 62% silent weeks
- `Rivers`: 62% silent weeks
- `Fct`: 62% silent weeks
- `Benue`: 62% silent weeks
- `Niger`: 62% silent weeks
- `Kano`: 62% silent weeks
- `Ogun`: 62% silent weeks
- `Lagos`: 62% silent weeks
- `Delta`: 61% silent weeks
- `Kwara`: 61% silent weeks
- `Jigawa`: 60% silent weeks
- `Bayelsa`: 59% silent weeks
- `Plateau`: 59% silent weeks
- `Sokoto`: 34% silent weeks
- `Zamfara`: 34% silent weeks
- `Kaduna`: 33% silent weeks
- `Katsina`: 32% silent weeks
- `Borno`: 31% silent weeks

### ⚠️ Flag 2 — Long zero-fatality streak near a hotspot
- None identified

### ⚠️ Flag 3 — Pilot state months with < 3 weeks observed
- `Kaduna`: 39 low-coverage months
- `Katsina`: 37 low-coverage months
- `Sokoto`: 37 low-coverage months
- `Zamfara`: 46 low-coverage months

## Overdispersion test
See console output. If var/mean > 3 for fatalities, Poisson is ruled out — use negative binomial.

## Limitations (carry into Phase 2)
1. Admin1 centroids are state centroids, not event locations — sub-state mapping is misleading.
2. "Quiet" states near active conflict zones may be under-reported, not peaceful.
3. Weekly aggregation loses within-week sequencing; event ordering unknown.
4. ACLED may have sourcing gaps for rural Zamfara and Sokoto — UCDP cross-check needed.
