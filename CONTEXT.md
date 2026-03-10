# IHD-COVID Analysis — Project Context

> **Purpose**: Self-contained briefing for any agent working in `main_analysis_scripts_v2/`.
> Read this first, then `DATA_CONTEXT.md` for data specifics.

---

## 1. Study Overview

**Research question**: Does SARS-CoV-2 infection independently elevate the risk of Ischaemic Heart Disease (IHD / acute myocardial infarction)?

We analyse Singapore's national administrative health data (MediClaims + SingCLOUD + COVID case registry + Death Registry) across the pandemic period (2015-2023) using a tiered modelling strategy that incrementally adds covariates to isolate COVID-specific cardiovascular risk.

### Cohort Definitions

| Group | Label | Definition | N (confirmed) |
|-------|-------|-----------|---------------|
| G1 | Post-COVID IHD | COVID-positive → IHD within 365 days | 1,870 |
| G2 | COVID No-IHD | COVID-positive → no IHD within 365 days | 483,981 |
| G3 | Naive IHD | IHD diagnosis with NO COVID in 1 year prior | 70,838 |
| Unknown | Unclassified | Missing group assignment | 75 |

- **G1 vs G2** answers: "Among COVID patients, who develops IHD?"
- **G1 vs G3** answers: "Are COVID-related IHD patients different from non-COVID IHD patients?"

### Key Demographics
- G1: mean age 71.5 ± 14.5, 65.2% male
- G2: mean age 41.5 ± 22.2, 44.5% male
- G3: mean age 69.0 ± 13.3, 63.1% male

---

## 2. Tiered Modelling Strategy (DAG-Based)

We follow a **Directed Acyclic Graph (DAG)** approach: only pre-existing confounders are adjusted for, never mediators or colliders. Vaccination is treated as an effect modifier (stratification), not a covariate.

### Tier 0 — Crude Epidemiology (COMPLETE)
- **Standardised Incidence Ratio (SIR)**: 1.85 (95% CI: 1.77–1.94)
  - 1,891 observed IHD events vs 1,020.6 expected
- **ASIR**: 463.57 per 100,000 person-years
- Age-standardised against Singapore Census 2020 population
- Kaplan-Meier survival: G1 shows steeper early mortality than G3

### Tier 1 — Demographic Adjustment (COMPLETE)
- Model: `logit(IHD) = β₀ + β₁·Age + β₂·Male`
- Full population: AUC = 0.867, Age OR = 1.07, Male OR = 1.96, Pseudo R² = 0.1465
- Age ≥ 35 subset: AUC = 0.781, Age OR = 1.06, Male OR = 1.92, Pseudo R² = 0.083

### Tier 2 — Clinical Baseline (COMPLETE)
- Model: `logit(IHD) = β₀ + β₁·Age + β₂·Male + β₃·CCI`
- AUC = 0.881 (+0.014 from Tier 1), CCI OR = 1.262
- LR test vs Tier 1: p = 3.10×10⁻⁷⁸ (CCI significantly improves fit)
- Age OR attenuation: 14.8%, Gender OR attenuation: 9.5%
- Variance decomposition: Age 49.8%, CCI 9.1%, Gender 4.1%
- Individual CCI components: Prior MI OR = 12.34, Diabetes Complicated OR = 2.19, CHF OR = 1.94

**Critical G1 vs G3 finding**: CCI OR = 0.704 — COVID-IHD patients are *healthier* at baseline than non-COVID IHD patients (CCI 2.22 vs 3.53). This is the strongest evidence of a COVID-specific cardiovascular pathway: you don't need to be as sick to get IHD if you had COVID.

### Tier 3 — Era & Vaccination Stratification (NEXT)
- Run the Tier 2 model separately for each variant era:
  - **Ancestral**: infection before 2021-05-01 (SIR = 2.51)
  - **Delta**: 2021-05-01 to 2021-12-31 (SIR = 1.59)
  - **Omicron**: ≥ 2022-01-01 (SIR = 2.08)
- Additionally stratify by vaccination status within each era
- Compare ORs across strata to assess whether variant virulence or vaccination modifies the COVID→IHD association
- This is a **stratification**, not a covariate adjustment (consistent with DAG)

---

## 3. Execution Environment

**You cannot run scripts locally.** The execution environment is:

- **Platform**: Amazon SageMaker VDI (Virtual Desktop Infrastructure)
- **Data storage**: AWS S3 bucket (accessed via `s3fs`)
- **Workflow**: Scripts are written locally → copied to SageMaker → executed there
- **Data access**: Via `catalog.py` which reads `catalog.yaml` and loads datasets from S3
- **Python environment**: pandas, numpy, scipy, statsmodels, lifelines, plotly, matplotlib, s3fs

When writing scripts, assume they will run in a SageMaker notebook/terminal with S3 access. All file paths for data should use the DataCatalog or config-specified relative paths, not local absolute paths.

---

## 4. Workspace Structure

```
main_analysis_scripts_v2/
├── CONTEXT.md              ← You are here
├── DATA_CONTEXT.md         ← Data dictionary & catalog reference
├── TIER3_ANALYSIS_PLAN.md  ← Tier 3 analytical strategy & expected findings
├── TIER3_BIOSTATS_REVIEW.md ← Questions for senior biostatistics review
├── config.yaml             ← All study parameters, ICD codes, paths
├── pipeline.py             ← Step orchestrator (runs steps 1-11)
├── main.py                 ← Entry point (placeholder)
├── simplified_analysis_plan.md  ← Detailed tier-wise analysis plan
├── catalog.py              ← DataCatalog class (copied from project root)
│
├── core_pipeline_scripts/
│   ├── 1_extract_index_events.py    ← Find IHD events in MediClaims
│   ├── 2_generate_cohorts.py        ← Build G1/G2/G3 cohorts
│   ├── 3_enrich_features.py         ← Add demographics, comorbidities, meds
│   ├── 4_statistical_analysis.py    ← Tier 0: SIR, ASIR, KM curves
│   ├── 5_variant_era_analysis.py    ← Crude era-stratified rates & Cox
│   ├── 6_tier_1_analysis.py         ← Tier 1: Age + Gender logistic reg
│   ├── 7_cci_diagcode_discovery.py  ← Discover ICD-10 codes for CCI
│   ├── 7_cci_discovery_config.yaml  ← CCI component ICD pattern config
│   ├── 8_apply_cci_codes.py         ← Compute CCI scores from discovered codes
│   ├── 8_cci_curated_codes.yaml     ← Manually curated CCI ICD-10 codes
│   ├── 9_tier_2_analysis.py         ← Tier 2: Age + Gender + CCI
│   ├── 10_vaccine_severity_enrichment.py ← Ingest vacc/severity/race/serology
│   └── 11_tier_3_analysis.py        ← Tier 3: Era & vaccination stratification
│
├── src/
│   ├── utils.py             ← Logging, viz helpers, DataFrame reporting
│   └── __init__.py
│
└── reports/
    ├── tier0_paper_writeup.pdf      ← Full Tier 0 results write-up
    ├── tier1_and2_results.pdf       ← Full Tier 1 & 2 results (53 pages)
    └── ai-writeups/
        ├── COVID-data.md            ← Dataset catalog for PREPARE ecosystem
        ├── cohort_flowchart.md      ← Cohort construction logic
        ├── covid_variant_era_writeup.md  ← Singapore pandemic timeline
        ├── study_redesign_plan.md   ← Analysis redesign rationale
        └── tier2_writeup.md         ← Tier 2 results narrative
```

---

## 5. Pipeline Steps (In Order)

| Step | Script | Purpose | Key Outputs |
|------|--------|---------|-------------|
| 1 | `1_extract_index_events.py` | Scan MediClaims (2015-2023) for IHD diagnoses using ICD-10 codes & regex | `ihd_index_events.csv` |
| 2 | `2_generate_cohorts.py` | Link COVID registry + IHD events → assign patients to G1/G2/G3 | `cohort_master.csv` |
| 3 | `3_enrich_features.py` | Add demographics (age, sex), comorbidities, medications from SingCLOUD | `cohort_enriched.csv` |
| 4 | `4_statistical_analysis.py` | Tier 0: SIR, ASIR, KM curves, Table 1, forest plots | PDFs, CSVs in results dir |
| 5 | `5_variant_era_analysis.py` | Era-stratified crude rates, era-specific SIR, Cox PH models | Era-specific CSVs, plots |
| 6 | `6_tier_1_analysis.py` | Tier 1 logistic regression: Age + Gender | Model results, diagnostics |
| 7 | `7_cci_diagcode_discovery.py` | Discover which ICD-10 codes appear for each CCI component | Discovery YAML + reports |
| 8 | `8_apply_cci_codes.py` | Compute CCI scores per patient using curated codes | CCI-augmented cohort CSV |
| 9 | `9_tier_2_analysis.py` | Tier 2 logistic regression: Age + Gender + CCI | Model results, diagnostics |
| 10 | `10_vaccine_severity_enrichment.py` | Ingest vaccination, severity, race, serology, reinfection data | `cohort_tier3_ready.csv` |
| 11 | `11_tier_3_analysis.py` | Tier 3: Era & vaccination stratified models + interaction tests | Stratified model results |

---

## 6. Config Reference (config.yaml)

Key sections in `config.yaml`:
- **`paths`**: Output directories (output, raw_cache, processed, results)
- **`datasets`**: Aliases/patterns for MediClaims, COVID registry, death registry, SingCLOUD
- **`study_period`**: 2015-2023
- **`cohort_definitions`**: follow_up_days=365, washout_period_days=365
- **`definitions.ihd_icd10_codes`**: I21.x, I22.x series (with and without dots)
- **`definitions.comorbidities`**: 16+ CCI component regex patterns
- **`definitions.medications`**: Statin, Antiplatelet, Antihypertensive keyword lists
- **`population_denominator`**: Singapore Census 2020 age-sex distribution for standardisation

---

## 7. What Tier 3 Needs To Do

1. **Load the CCI-enriched cohort** (output from Step 8/9) with variant era labels (from Step 5)
2. **Subset by era** (Ancestral / Delta / Omicron) using COVID infection dates
3. **Within each era**: run `logit(IHD) = Age + Gender + CCI` (the Tier 2 model)
4. **Compare ORs** across eras — if OR for COVID→IHD varies, variant virulence matters
5. **Further stratify by vaccination status** within each era
6. **Report**: Forest plots comparing ORs across eras, interaction tests, sample sizes per stratum
7. **Sensitivity**: Consider whether era-specific confounding (e.g., testing rates, treatment protocols) biases results

The existing `5_variant_era_analysis.py` computes crude era-specific rates and Cox models but does NOT run the Tier 2 logistic regression by era. Tier 3 is implemented in Steps 10 and 11.

### Step 10 Run Status (Confirmed)

Step 10 was run on SageMaker. The cohort loaded successfully (556,764 patients) and variant era assignment worked:
- G1 by era: Ancestral = 58, Delta = 445, Omicron = 1,367
- G2 by era: Ancestral = 3,289, Delta = 40,862, Omicron = 439,830

**However, all 5 external dataset merges returned 0 matches** — vaccination, severity, race, serology, and reinfection columns are all empty/null. Root cause is likely a catalog alias mismatch or uin format mismatch on SageMaker. See `DATA_CONTEXT.md` Section 7 for details and debugging steps.

**Impact on Tier 3**: Components A (era-stratified G1 vs G2), C (era-stratified G1 vs G3), and E (interaction tests) can proceed using existing Age/Gender/CCI data. Components B (vaccination-stratified) and D (severity exploratory) are blocked until the merge is fixed.

---

## 8. Important Caveats for Agents

- **Never hardcode S3 paths** — always use DataCatalog or config.yaml
- **ICD-10 codes appear both with and without dots** in the data (e.g., `I21.0` and `I210`). All code must handle both formats.
- **MediClaims has 9 years** (2015-2023), loaded one year at a time to manage memory
- **SingCLOUD medications are split into 29 chunks** (`SingCLOUD_medication_items1` through `SingCLOUD_medication_items29`)
- **The G1 vs G3 comparison uses reversed outcome coding** (G1=1 means COVID-IHD, G3=0 means non-COVID-IHD) — the OR interpretation flips
- **`pipeline.py` uses dynamic imports** — new steps must follow the naming convention (`N_step_name.py` with `run_step_N(config)` entry point)
- **All outputs should use `save_with_report()`** from `src/utils.py` for traceability
