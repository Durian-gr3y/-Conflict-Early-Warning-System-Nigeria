# Conflict Early Warning Nigeria — Responsible Use Policy v1.0

**Effective date:** 2026-10-18  
**Review date:** 2027-04-18 (or after any significant model change)

---

## 1. What this platform does

This platform aggregates publicly available conflict event data, environmental
indicators, and humanitarian reports to produce probabilistic forecasts of
violent events at the Nigerian state level. All outputs are statistical
estimates with documented uncertainty, not intelligence products or verified
ground truth.

---

## 2. Prohibited uses

| Prohibited use | Reason |
|---|---|
| Profiling individuals or named groups | Violates privacy rights; enables targeting |
| Targeting any ethnic, religious, or political community | Risk of harm amplification |
| Informing kinetic (lethal) military operations | Humanitarian law obligations |
| Publishing forecasts without uncertainty bands | Misrepresents confidence |
| Reselling raw ACLED or UCDP data | Licence violation (CC BY-NC-SA / CC BY) |

---

## 3. Government client screening

Before any forecast output is shared with a government security actor, the
operator **must** complete the following checklist:

- [ ] Written use-case statement from the client
- [ ] Review against the [Do No Digital Harm framework](https://www.thengoforum.org/do-no-digital-harm)
- [ ] Confirmation that outputs will **not** be used to direct security-force operations against civilians
- [ ] Approval by the platform operator's ethics reviewer (or an equivalent independent body)

If any item above cannot be confirmed, data sharing must not proceed.

---

## 4. Individual privacy

The platform:
- **Does not** store personal identifiers
- **Does not** process social-media posts linked to individuals
- **Does not** produce outputs at a resolution finer than admin1 (state) level until admin2 data is available and separately reviewed
- **Does not** accept client requests to add individual tracking

---

## 5. Uncertainty communication

Every forecast output **must** include:
1. A numeric probability (e.g., 0.73), not a vague label (e.g., "high risk")
2. A 90% credible interval or confidence interval
3. The data-quality flag for the relevant state-period
4. The model version and training-data cutoff date

---

## 6. Known limitations (mandatory disclosure)

Clients must be informed of:
- **Coverage bias**: Quiet records may reflect under-reporting, not absence of violence
- **Admin1 resolution**: State-level forecasts mask within-state spatial variation
- **Lag**: ACLED data lags real events by days to weeks; UCDP lags by ~12 months
- **Model uncertainty**: All models are wrong; the question is whether they are useful

---

## 7. Feedback and harm reporting

If a forecast is found to have contributed to harm, the operator must:
1. Immediately suspend delivery to the relevant client
2. Document the incident in `reports/incident_log.md`
3. Conduct a post-mortem and update the model card accordingly

---

*This policy was drafted using the Do No Digital Harm framework and the Alan Turing Institute's data ethics guidance.*
