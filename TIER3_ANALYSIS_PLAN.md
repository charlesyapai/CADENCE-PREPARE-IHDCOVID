# Tier 3 Analysis Plan: Era & Vaccination Stratification

## 1. What Tier 3 Is

Tier 3 takes the Tier 2 model (`logit(IHD) = Age + Gender + CCI`) and runs it **separately within strata** defined by variant era and vaccination status. The logic: if COVID→IHD risk varies across eras or by vaccination, then variant virulence and immune priming modify the association. This is stratification — not covariate adjustment — consistent with the DAG framework where vaccination is treated as an effect modifier, not a confounder.

## 2. What We Already Know (Tiers 0–2 Findings)

These results frame the interpretation of Tier 3.

**Tier 0 (Crude)**
- Overall SIR = 1.85 (95% CI 1.77–1.94). COVID patients develop IHD at nearly twice the expected rate.
- Era-specific crude SIRs: Ancestral 2.51, Delta 1.59, Omicron 2.08. The drop during Delta and partial rebound in Omicron is a signal worth decomposing.

**Tier 1 (Demographics)**
- Age (OR 1.07/year) and male sex (OR 1.96) explain most of the IHD prediction (AUC 0.867).
- These ORs are stable and expected — age and sex are fundamental IHD risk factors.

**Tier 2 (+ CCI)**
- CCI contributes significantly (LR test p = 3.10×10⁻⁷⁸, OR 1.262 per CCI point).
- Adding CCI attenuates Age OR by 14.8% and Gender OR by 9.5% — comorbidity burden partially explains the age and sex gradient.
- G1 vs G3 comparison: CCI OR = 0.704. Post-COVID IHD patients have **lower** baseline comorbidity than non-COVID IHD patients. This is the strongest evidence that COVID introduces a cardiovascular pathway independent of traditional risk — you don't need to be as sick to develop IHD after COVID.

**Key implication for Tier 3**: If the protective-appearing CCI OR in G1 vs G3 varies by era or vaccination status, it tells us whether newer variants or vaccination alter the degree to which COVID bypasses traditional risk factors.

## 3. New Data Available for Tier 3

Step 10 enriches the cohort with these variables from five new datasets:

| Variable | Source | Coverage |
|----------|--------|----------|
| `doses_before_ref` | NIRListtruncated (6.17M rows) | Near-complete for all patients |
| `vaccinated_before_covid` | Derived from NIR | Binary flag (≥1 dose before infection) |
| `fully_vaccinated_before_covid` | Derived from NIR | Binary flag (≥2 doses) |
| `vaccine_brand_primary` | NIRListtruncated | Brand of first dose |
| `LOS` (length of stay) | COVIDFACILLOS (2.35M rows) | COVID patients |
| `DaysInICU` | COVIDFACILLOS | COVID patients with ICU admission |
| `Deceased` | COVIDFACILLOS | In-hospital mortality flag |
| `required_O2` | Derived from COVIDFACILLOS O2 dates | Oxygen supplementation flag |
| `severity_category` | Derived (Critical/Severe/Moderate/Mild) | COVID patients |
| `race` | COVIDFACILLOS + COVID Reinfections | Demographics |
| `variant_era` | Derived from covid_date | Ancestral / Delta / Omicron |
| `is_reinfection` | COVID Reinfections + FacilityUtilizationRI | Binary flag |
| `serologyresult` | Serology_Tests_COVID (1.07M rows) | Serology test outcome |
| `serologyctvalue` | Serology_Tests_COVID | CT value (proxy for viral load) |

### Strata Definitions

**Variant Era** (based on COVID infection date):
- Ancestral: before 2021-05-01
- Delta: 2021-05-01 to 2021-12-31
- Omicron: 2022-01-01 onwards

**Vaccination Status** (at time of COVID infection):
- Unvaccinated: 0 doses before infection
- Partially vaccinated: 1 dose before infection
- Fully vaccinated: ≥2 doses before infection

## 4. Analytical Steps

### Step 10: Data Enrichment (script: `10_vaccine_severity_enrichment.py`)
- Ingest COVIDFACILLOS, NIRListtruncated, COVID Reinfections, FacilityUtilizationLOSSubsequentRI, Serology_Tests_COVID
- Merge onto CCI-enriched cohort (Step 8 output)
- Derive vaccination status at time of infection, severity category, era assignment, reinfection flag
- Output: `cohort_tier3_ready.csv` with detailed data profiling report

### Step 11: Tier 3 Analysis (script: `11_tier_3_analysis.py`)

#### 11A. Era-Stratified Logistic Regression (G1 vs G2)

For each variant era, run on G1+G2 patients infected in that era:

```
logit(IHD) = β₀ + β₁·Age + β₂·Male + β₃·CCI
```

Report per era:
- N (total, G1, G2), event rate
- ORs with 95% CI for Age, Male, CCI
- AUC, Pseudo R², Hosmer-Lemeshow
- Comparison of ORs across eras (forest plot)

**What to look for**:
- If CCI OR is highest in Ancestral and lowest in Omicron, it suggests that the ancestral variant affected sicker patients more disproportionately, while Omicron affected patients regardless of comorbidity.
- If Age OR drops across eras, the age gradient in COVID-IHD may have flattened as the pandemic progressed (younger people infected more during Omicron).

#### 11B. Vaccination-Stratified Models Within Eras

Within each era where vaccination existed (primarily Delta and Omicron):

```
logit(IHD) = β₀ + β₁·Age + β₂·Male + β₃·CCI
```

Separately for:
- Unvaccinated patients
- Vaccinated patients (≥2 doses)

**What to look for**:
- If the IHD event rate is lower in vaccinated strata, vaccination attenuates COVID→IHD risk.
- If the CCI OR shifts toward 1.0 in vaccinated patients, it suggests vaccination restores the "normal" comorbidity-IHD relationship (i.e., COVID no longer bypasses traditional risk).

#### 11C. Era-Stratified G1 vs G3 Comparison

Run the G1 vs G3 model per era:

```
logit(COVID_IHD) = β₀ + β₁·Age + β₂·Male + β₃·CCI
```

Where COVID_IHD=1 for G1 (post-COVID IHD) and COVID_IHD=0 for G3 (non-COVID IHD).

**What to look for**:
- The overall CCI OR is 0.704 (COVID-IHD patients are healthier). If this OR varies by era, it tells us whether the "healthy patient" signal is variant-specific.
- If Omicron G1 patients have CCI closer to G3, it would suggest Omicron-era IHD behaves more like traditional IHD.

#### 11D. Severity as Exploratory Covariate

Within G1+G2 COVID patients, optionally add severity:

```
logit(IHD) = β₀ + β₁·Age + β₂·Male + β₃·CCI + β₄·severity_category
```

This is exploratory (severity is a potential mediator on the causal path from COVID→IHD, so including it may introduce bias). It answers: conditional on surviving severe COVID, does severity independently predict IHD? Report it separately from the main models with appropriate caveats.

#### 11E. Interaction Tests

Formal statistical tests for effect modification:
- Likelihood ratio test: full model (with era×CCI interaction) vs reduced model
- Wald test on interaction terms in pooled model:

```
logit(IHD) = Age + Male + CCI + era + era×CCI
```

These confirm whether the OR differences across eras are statistically significant, not just descriptive.

## 5. Potential Findings and Interpretation Framework

| Finding | Interpretation |
|---------|---------------|
| CCI OR stable across eras | Comorbidity contribution to IHD risk is constant — COVID-IHD pathway is not variant-dependent |
| CCI OR higher in Ancestral, lower in Omicron | Ancestral variant disproportionately affected sicker patients; Omicron is more indiscriminate |
| Lower IHD rate in vaccinated strata | Vaccination provides direct cardioprotection or reduces systemic inflammation |
| CCI OR closer to 1.0 in vaccinated | Vaccination "normalises" the IHD risk profile — COVID no longer bypasses traditional risk |
| Severity predicts IHD (positive OR) | Severe acute illness drives cardiac damage (direct mechanism) |
| Severity does NOT predict IHD | IHD may arise via chronic immune-mediated pathways, not acute damage |
| G1 vs G3 CCI OR varies by era | The "healthy patient" paradox is variant-specific |
| Reinfection associated with higher IHD rate | Cumulative COVID exposure amplifies cardiovascular risk |

## 6. Sample Size Considerations

Expected approximate sample sizes per stratum (from crude era analysis):

| Era | G1 (IHD) | G2 (No IHD) | Total COVID |
|-----|----------|-------------|-------------|
| Ancestral | ~180 | ~20,000 | ~20,180 |
| Delta | ~250 | ~95,000 | ~95,250 |
| Omicron | ~1,390 | ~350,000 | ~351,390 |

G1 counts are small, especially Ancestral. This means:
- Ancestral-era models will have wide confidence intervals. A minimum of ~10 events per predictor (EPV) rule suggests 3 predictors need ~30 events; 180 is adequate.
- Vaccination sub-stratification within Ancestral is not feasible (vaccines weren't yet available).
- Delta vaccination stratification is feasible but marginal — check cell sizes before proceeding.
- Omicron has sufficient sample size for all planned stratifications.

If any stratum has <50 events, report descriptive statistics only (no logistic regression). Use exact or Firth's penalized logistic regression if events are borderline (50–100).

## 7. Output Specification

```
data/03_results/step_11_tier3/
├── era_models/
│   ├── tier3_ancestral_g1g2.txt    -- Model output
│   ├── tier3_delta_g1g2.txt
│   ├── tier3_omicron_g1g2.txt
│   └── era_comparison_forest.png   -- ORs across eras
│
├── vacc_models/
│   ├── tier3_delta_unvacc.txt
│   ├── tier3_delta_vacc.txt
│   ├── tier3_omicron_unvacc.txt
│   ├── tier3_omicron_vacc.txt
│   └── vacc_comparison_forest.png
│
├── g1_vs_g3/
│   ├── g1g3_ancestral.txt
│   ├── g1g3_delta.txt
│   ├── g1g3_omicron.txt
│   └── g1g3_era_comparison_forest.png
│
├── interaction_tests/
│   ├── era_interaction_lr_test.txt
│   └── pooled_interaction_model.txt
│
├── severity_exploratory/
│   ├── severity_model_full.txt
│   └── severity_by_era.txt
│
├── descriptive/
│   ├── table1_by_era.csv
│   ├── table1_by_vacc_status.csv
│   ├── vacc_coverage_by_era.csv
│   └── severity_by_era.csv
│
└── tier3_summary_report.txt        -- Executive summary of all results
```

## 8. Script Conventions

- Entry point: `run_step_11(config)` in `11_tier_3_analysis.py`
- Uses `DataCatalog` for any additional data loads
- Uses `setup_logger`, `ensure_dir`, `save_with_report` from `src/utils.py`
- Reads from: `data/02_processed/step_10_enriched/cohort_tier3_ready.csv`
- All plots saved as both PNG (for reports) and HTML (for interactive inspection)
- Models use `statsmodels.formula.api.logit` for consistency with Tier 1/2
- Forest plots use `matplotlib` with error bars for OR comparisons
