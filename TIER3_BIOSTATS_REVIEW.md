# Tier 3 Script Review Request — Senior Biostatistics Perspective

> **For**: A reviewing agent with senior biostatistics expertise.
> **Script**: `core_pipeline_scripts/11_tier_3_analysis.py`
> **Supporting**: `core_pipeline_scripts/10_vaccine_severity_enrichment.py`
> **Context**: Read `CONTEXT.md`, `DATA_CONTEXT.md`, and `TIER3_ANALYSIS_PLAN.md` first.

---

## What This Script Does

`11_tier_3_analysis.py` takes the Tier 2 logistic regression model (`logit(IHD) = Age + Male + CCI`) and runs it within strata defined by SARS-CoV-2 variant era (Ancestral / Delta / Omicron) and vaccination status. It has five components:

- **A**: Era-stratified G1 vs G2 models (one model per era)
- **B**: Vaccination-stratified models within Delta and Omicron eras
- **C**: Era-stratified G1 vs G3 comparison (COVID exposure model among IHD patients)
- **D**: Severity as an exploratory covariate (flagged as potential mediator)
- **E**: Formal interaction tests (era × CCI, vaccination × CCI via LR tests)

## Completed Code Integrity Fixes

Before this review, the following bugs were identified and fixed:

1. **G3 CCI zeroing bug**: The CCI recomputation originally used `covid_date` as reference for ALL patients. Since G3 has no COVID date (NaT), their CCI flags were all 0, making the G1 vs G3 comparison invalid. Fixed by using `covid_date.combine_first(ihd_date)` as the index date, with a sanity check logging G3 mean CCI.

2. **Severity misclassification for G3**: `_assign_severity` in Step 10 treated NaN `required_O2` as truthy, labelling unmatched G3 patients as "Severe". Fixed with explicit `is True` check.

3. **Duplicate sklearn imports**: `from sklearn.metrics import roc_auc_score` was re-imported inside three loops. Moved to module-level import.

---

## Questions for Biostatistical Review

Please evaluate the script and plan from a senior biostatistics standpoint. Below are specific areas where feedback is requested, followed by a general invitation for any issues you identify.

### 1. Stratification Approach vs. Interaction Model

The plan uses both:
- **Stratified models** (separate regressions per era) — Component A/B
- **Pooled interaction model** (era dummies + era×CCI terms) — Component E

**Q**: Is reporting both approaches appropriate, or does one supersede the other? The stratified models allow ORs to vary freely across strata (different intercepts and slopes), while the interaction model tests whether the CCI slope specifically differs. Should the interaction test be the primary test, with stratified results as descriptive, or vice versa?

### 2. G1 vs G3 Design — G3 as Time-Invariant Reference

In Component C, G1 patients are era-specific (infection date determines era), but G3 patients (IHD without COVID) span the entire 2015–2023 study period and have no natural era assignment. The current approach pools ALL G3 as a common reference for each era-specific G1 subset.

**Q**: Is this valid, or does it introduce temporal confounding? G3 patients from 2015–2019 differ from those in 2022 (e.g., changes in IHD coding practices, population aging). Should G3 be time-matched to each era (e.g., restrict G3 to IHD events in the same calendar period as the era)?

### 3. Vaccination as Effect Modifier — DAG Justification

The DAG treats vaccination as an effect modifier (stratified, not adjusted). The rationale: vaccination influences both infection probability and post-infection immune response, but adjusting for it could introduce collider bias if vaccination also relates to health-seeking behaviour.

**Q**: Is the stratification-only approach defensible? Should we also run a model adjusting for vaccination as a covariate (for comparison/sensitivity), noting the potential collider bias? If vaccination is protective against IHD (not just against severe COVID), then collider bias from adjusting may be small and the adjusted model could provide a useful point estimate.

### 4. Severity as Potential Mediator (Component D)

The script includes severity (LOS, ICU, O2) as an ordinal covariate in an exploratory model. The causal caveat is documented: severity is on the causal path (COVID → severity → IHD), so conditioning on it may block the total effect.

**Q**: Is the ordinal encoding appropriate (Unknown=0, Mild=1, Moderate=2, Severe=3, Critical=4)? Should Unknown be excluded rather than coded as 0? Would a binary indicator (e.g., ICU admission yes/no) be preferable to an ordinal scale with potentially unequal intervals?

### 5. Multiple Comparisons

Tier 3 produces many models across strata: 3 eras × 2 groups (G1G2, G1G3) + 4 vaccination strata + interaction tests. No correction for multiple comparisons is applied.

**Q**: Should any multiplicity adjustment be applied (Bonferroni, Holm, FDR)? Or is this exploratory stratification where multiplicity correction would be overly conservative? If not formal correction, should we at least flag findings that would not survive adjustment?

### 6. Sample Size Adequacy

Expected event counts per stratum from Tier 0 crude analysis:
- Ancestral G1: ~180 events
- Delta G1: ~250 events
- Omicron G1: ~1,390 events

The script uses `MIN_EVENTS_FOR_MODEL = 30` as the minimum to attempt logistic regression with 3 predictors (~10 EPV).

**Q**: Is 30 events sufficient for stable OR estimation with 3 covariates? Some guidelines recommend 10-20 EPV; others argue ≥50 is needed for reliable CIs. Should Firth's penalized likelihood be used for Ancestral/Delta strata? The script has a `FIRTH_THRESHOLD = 80` defined but not yet implemented — should it be?

### 7. CCI Exclusions Consistency

The modified CCI excludes Myocardial Infarction, Congestive Heart Failure, and AIDS/HIV (same as Tier 2). MI and CHF are excluded because they overlap with the IHD outcome. AIDS is excluded due to data restrictions.

**Q**: In the G1 vs G3 model (Component C), where the outcome is `covid_exposed` (not IHD), should MI and CHF be re-included in CCI? The exclusion rationale (overlap with outcome) doesn't apply when the outcome is COVID exposure, not IHD. However, changing CCI composition between components would make ORs non-comparable. What is the preferred approach?

### 8. Model Diagnostics

The current script reports AUC and Pseudo R² per stratum but does not include Hosmer-Lemeshow tests, calibration plots, or residual analysis per stratum (unlike Tier 2 which had full diagnostics).

**Q**: Should full diagnostics be added for each stratified model? Or is AUC + the overall interaction test sufficient for a stratified analysis? In small strata (Ancestral), HL tests may be unstable.

### 9. Missing Vaccination Data Handling

G3 patients (IHD without COVID) do get NIR vaccination data, and their `ref_date` falls back to `ihd_date`. So `doses_before_ref` for G3 counts doses before their IHD event. This means G3 vaccination status reflects doses before IHD, not doses before a non-existent COVID infection.

**Q**: Is this semantically valid for Component C? When comparing G1 (doses before COVID) vs G3 (doses before IHD), the reference dates differ conceptually. Should vaccination be excluded from G1 vs G3 models entirely, or is this comparison still informative?

### 10. Race as a Covariate

Race data is available from COVIDFACILLOS and COVID Reinfections. The current Tier 3 models do not include race.

**Q**: Should race be considered as an additional covariate or stratification variable? Singapore's multi-ethnic composition (Chinese, Malay, Indian, Others) may have differential COVID-IHD risk. However, adding race would change the model from Tier 2's specification. Should it be a sensitivity analysis?

---

## General Review Request

Beyond the specific questions above, please review the overall analytical strategy for:

- **Internal validity**: Any threats to causal inference not addressed above
- **Statistical assumptions**: Logistic regression appropriateness given the rare event rates (~0.39% overall, varying by era)
- **Reporting standards**: Does the output structure align with what a journal reviewer would expect for a stratified analysis?
- **Alternative approaches**: Would propensity score methods, Poisson regression for rates, or marginal structural models be more appropriate for any component?
- **Any code-level statistical errors**: Incorrect formula specifications, inappropriate use of test statistics, etc.

Please provide specific revision suggestions with rationale, prioritised by impact on validity.
