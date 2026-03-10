# Data Context — Datasets, Schemas & Catalog Reference

> **Purpose**: Describes every dataset used in the IHD-COVID analysis, how they are accessed, and what intermediate files the pipeline produces.

---

## 1. Data Access Architecture

```
┌──────────────────┐      catalog.yaml       ┌──────────────┐
│  Python Script   │ ──── DataCatalog() ───→  │   S3 Bucket  │
│  (SageMaker VDI) │         │                │  (moh-2030)  │
└──────────────────┘         │                └──────────────┘
                             ▼
                      pandas DataFrame
```

- **`catalog.py`** reads `catalog.yaml`, which maps dataset aliases to S3 paths
- Scripts call `cat.load("alias_name")` to get a DataFrame
- Supports: CSV, XLSX, Parquet, JSONL
- S3 access via `s3fs.S3FileSystem(anon=False)` (uses IAM role on SageMaker)

### Quick Usage
```python
from catalog import DataCatalog

cat = DataCatalog("catalog.yaml")
df = cat.load("mediclaims_diag_2020")

# Or inject multiple into namespace:
cat.inject_into("mediclaims_diag_2019", "mediclaims_diag_2020", tgt_globals=globals())
```

---

## 2. Source Datasets

### 2.1 MediClaims (Diagnosis Records)

| Property | Value |
|----------|-------|
| Alias pattern | `mediclaims_diag_{year}` (2015-2023) |
| Source | National health insurance claims |
| Granularity | One row per diagnosis event |
| Key columns | `PATIENT_ID`, `DIAGNOSIS_CODE` (ICD-10), `DIAGNOSIS_DESCRIPTION`, `CLAIM_DATE`, `ADMISSION_DATE`, `DISCHARGE_DATE` |
| Size | Millions of rows per year |
| Notes | ICD-10 codes appear both with dots (`I21.0`) and without (`I210`). Both formats must be handled. |

**Usage in pipeline**:
- Step 1: Scan all 9 years for IHD codes (I21.x, I22.x)
- Step 3: Scan for comorbidity ICD-10 codes (CCI components)
- Step 7: Discovery scan for all ICD-10 codes matching CCI patterns

### 2.2 COVID Case Registry

| Property | Value |
|----------|-------|
| Alias patterns | `ConfirmedCaseHeadersForAgenciesCNo0to100000`, etc. (7 chunks) |
| Source | National COVID-19 confirmed case registry |
| Granularity | One row per confirmed COVID case |
| Key columns | `PATIENT_ID` (or similar), `CASE_DATE` / `CONFIRMATION_DATE`, demographic fields |
| Coverage | All confirmed COVID-19 cases in Singapore |
| Notes | Split into chunks by case number ranges + two date-specific files (Dec 2022, Jan 2023) |

**Full list of COVID registry aliases** (from config.yaml):
1. `ConfirmedCaseHeadersForAgenciesCNo0to100000`
2. `ConfirmedCaseHeadersForAgenciesCNo100000to200000`
3. `ConfirmedCaseHeadersForAgenciesCNo200000to300000`
4. `ConfirmedCaseHeadersForAgenciesCNo300000to400000`
5. `ConfirmedCaseHeadersForAgenciesCNo400000to500000`
6. `ConfirmedCaseHeadersForAgencies1Dec2022`
7. `ConfirmedCaseHeadersForAgencies1Jan2023`

### 2.3 Death Registry

| Property | Value |
|----------|-------|
| Alias | `death_registry` |
| Source | National death registration |
| Granularity | One row per death event |
| Key columns | `PATIENT_ID`, `DATE_OF_DEATH`, `CAUSE_OF_DEATH` (ICD-10) |
| Usage | Mortality outcome in KM analysis, Table 1 statistics |

### 2.4 SingCLOUD — Demographics

| Property | Value |
|----------|-------|
| Gender alias | `SingCLOUD_gender` |
| DOB alias | `SingCLOUD_DOB` |
| Source | National patient master index |
| Key columns | `PATIENT_ID`, `GENDER` / `SEX`, `DATE_OF_BIRTH` |
| Usage | Step 3: Enrich cohort with age and sex |

### 2.5 SingCLOUD — Medications

| Property | Value |
|----------|-------|
| Alias pattern | `SingCLOUD_medication_items{n}` (n = 1 to 29) |
| Source | National medication dispensing records |
| Granularity | One row per dispensed medication |
| Key columns | `PATIENT_ID`, `MEDICATION_NAME` / `DRUG_NAME`, `DISPENSE_DATE` |
| Size | 29 chunks (large dataset) |
| Usage | Step 3: Flag medication classes (Statin, Antiplatelet, Antihypertensive) by keyword matching |

**Medication keywords** (from config.yaml):
- **Statin**: `statin`
- **Antiplatelet**: `aspirin`, `clopidogrel`, `ticagrelor`, `prasugrel`, `dipyridamole`, `cilostazol`
- **Antihypertensive**: `pril`, `sartan`, `olol`, `dipine`, `furosemide`, `hydrochlorothiazide`

### 2.6 COVIDFACILLOS — COVID Facility & Severity Data

| Property | Value |
|----------|-------|
| Alias | `COVIDFACILLOS` |
| Source | COVID-19 facility management system |
| Granularity | One row per COVID case (may have duplicates per patient if multiple admissions) |
| Rows | ~2,349,112 |
| Date range | 2020-01-22 to 2024-02-28 |
| Date format | `%d%b%Y` (e.g. `15Mar2021`) |
| Key columns | `uin`, `notificationdate`, `Casenumber`, `CaseClass`, `age`, `gender`, `race` |
| Severity columns | `LOS` (length of stay), `DaysInICU`, `Deceased`, `O2StartDate`, `O2EndDate` |
| Vaccination columns | `vacc_date1`–`vacc_date5`, `vaccbrand1`–`vaccbrand5`, `VaccinationFourthDoseDate` |
| Other columns | `passtype`, `flattype`, `Exposuretype`, `CPlus`, `AGPlus`, `EarliestEDVisit`, `EDVisits`, `HistoricalStatus`, `Status` |
| Usage | Step 10: Primary source for COVID severity, race, and backup vaccination data |

### 2.7 NIRListtruncated — National Immunisation Registry

| Property | Value |
|----------|-------|
| Alias | `NIRListtruncated` |
| Source | National Immunisation Registry (NIR) |
| Granularity | One row per person |
| Rows | ~6,174,098 |
| Key columns | `uin`, `vacc_date1`–`vacc_date6`, `vaccbrand1`–`vaccbrand6` |
| Notes | Most comprehensive vaccination source. Covers up to 6 doses (vs 5 in COVIDFACILLOS). Covers ALL residents, not just COVID cases. |
| Usage | Step 10: Primary vaccination data source for all patients (G1, G2, G3) |

### 2.8 COVID Reinfections

| Property | Value |
|----------|-------|
| Alias | `COVID Reinfections` |
| Source | COVID-19 reinfection surveillance |
| Granularity | One row per reinfection event |
| Rows | ~128,101 |
| Key columns | `uin`, `notificationdate`, `Casenumber`, `CaseClass`, `age`, `reinfection_type`, `HistoricalStatus` |
| Demographics | `cat_gender`, `cat_race`, `cat_passtype`, `cat_flattype` (note: prefixed with `cat_`) |
| Severity columns | `DaysAtPHI`, `DaysAtCTF`, `DaysAtCIF`, `DaysAtCIFM`, `DaysAtDRF`, `DaysAtHRP`, `DaysO2`, `DaysInICU`, `Deceased` |
| Vaccination columns | `vacc_date1`–`vacc_date5`, `vaccbrand1`–`vaccbrand5` |
| Usage | Step 10: Reinfection flag + supplementary race data |

### 2.9 FacilityUtilizationLOSSubsequentRI — Reinfection Facility Use

| Property | Value |
|----------|-------|
| Alias | `FacilityUtilizationLOSSubsequentRI` |
| Source | Facility utilisation for confirmed reinfected COVID patients |
| Granularity | One row per reinfection episode |
| Rows | ~2,319 |
| Key columns | `uin`, `notificationdate`, `Age`, `Gender`, `reinfection_type`, `reinfno`, `prev_reinf_notif` |
| Severity columns | `DaysAtPHI`, `DaysAtCTF`, `DaysAtCIF`, `DaysAtCIFM`, `DaysInICU`, `DaysO2`, `NRDate` |
| Vaccination columns | `vacc_date1`–`vacc_date5`, `vaccbrand1`–`vaccbrand5` |
| Usage | Step 10: Additional reinfection flag source |

### 2.10 Serology_Tests_COVID — Serology Results

| Property | Value |
|----------|-------|
| Alias | `Serology_Tests_COVID` |
| Source | COVID-19 serology testing programme |
| Granularity | One row per serology test |
| Rows | ~1,067,600 |
| Key columns | `uin`, `accesionno`, `serologyswabdate`, `serologyresultdate` |
| Result columns | `serologyswabstatus`, `serologyresult`, `serologyctvalue`, `serologyresultindicator`, `serologyvalue` |
| Lab columns | `serologylab`, `serologyswablocation`, `serologylabinterpretationnote` |
| Metadata | `createdat`, `createdby`, `updatedat`, `updatedby` |
| Usage | Step 10: Serology result and CT value (viral load proxy) linked to earliest test per patient |

---

## 3. Clinical Code Definitions

### 3.1 IHD (Index Event) — ICD-10 Codes

```
I21   - Acute myocardial infarction (parent)
I21.0 / I210 - Anterior wall STEMI
I21.1 / I211 - Inferior wall STEMI
I21.2 / I212 - Other sites STEMI
I21.3 / I213 - Unspecified site STEMI
I21.4 / I214 - NSTEMI
I21.9 / I219 - AMI unspecified
I22   - Subsequent MI (parent)
I220, I221, I228, I229 - Subsequent MI subtypes
```

Additionally, a regex is used for text-based discovery:
```
(?i)myocardial|infarction|stemi|nstemi|heart\s+attack
```

### 3.2 Charlson Comorbidity Index (CCI) — 16 Components

Each component is defined by an ICD-10 regex pattern in `config.yaml` under `definitions.comorbidities`:

| Component | ICD-10 Pattern (prefix) |
|-----------|------------------------|
| Myocardial Infarction | I21, I22, I25.2 |
| Congestive Heart Failure | I50, I11.0, I13.0, I13.2 |
| Peripheral Vascular Disease | I71, I73.9, I79.0, R02, Z95.8, Z99.2 |
| Cerebrovascular Disease | I60-I69, G45, G46 |
| Chronic Pulmonary Disease | J40-J47, J60-J67 |
| Diabetes Uncomplicated | E10-E14 (.0, .1, .9) |
| Diabetes Complicated | E10-E14 (.2-.8) |
| Dementia | F00-F03, F05.1, G30, G31.1 |
| Paraplegia/Hemiplegia | G04.1, G11.4, G80.1, G80.2, G81-G83 |
| Renal Disease | N18, N19, I12, I13, N03.2, N05.2, Z49, Z94.0, Z99.2 |
| Liver Disease (Mild) | B18, K70.x, K71.x, K73, K74, K76.x |
| Liver Disease (Severe) | K70.4, K71.1, K72.x, K76.5-7 |
| Peptic Ulcer Disease | K25-K28 |
| Rheumatic Disease | M05, M06, M31.5, M32-M34, M35.1/3, M36.0 |
| AIDS/HIV | B20-B24 |
| Malignancy (Any) | C00-C76, C81-C85, C88, C90-C97 |
| Metastatic Solid Tumor | C77-C79, C80 |
| Hypertension* | I10-I15 |
| Hyperlipidemia* | E78 |
| Obesity* | E66 |

*\* Not part of standard CCI but included for IHD risk profiling.*

### 3.3 Variant Era Definitions

| Era | Date Range | Defining Event |
|-----|-----------|----------------|
| Ancestral | Before 2021-05-01 | Pre-Delta Singapore |
| Delta | 2021-05-01 to 2021-12-31 | Delta wave in Singapore |
| Omicron | 2022-01-01 onwards | Omicron dominance |

Eras are assigned based on each patient's **COVID infection date**.

---

## 4. Pipeline-Generated Intermediate Files

These are produced by the pipeline steps and stored in the configured `processed_dir` and `results_dir`:

| File | Produced By | Description |
|------|-------------|-------------|
| `ihd_index_events.csv` | Step 1 | All IHD diagnosis events found in MediClaims |
| `cohort_master.csv` | Step 2 | Patient-level file with group assignment (G1/G2/G3), COVID date, IHD date |
| `cohort_enriched.csv` | Step 3 | Cohort + demographics + comorbidity flags + medication flags |
| `patient_comorbidities.csv` | Step 3 | Per-patient comorbidity binary flags |
| `patient_medications.csv` | Step 3 | Per-patient medication class flags |
| `patient_mortality.csv` | Step 3 | Per-patient mortality date (if deceased) |
| `tier0_results/` | Step 4 | SIR tables, KM curves, Table 1, forest plots |
| `era_analysis/` | Step 5 | Era-stratified rates, SIR by era, Cox model outputs |
| `tier1_results/` | Step 6 | Logistic regression outputs, ROC, diagnostics |
| `cci_discovery/` | Step 7 | ICD-10 code frequencies per CCI component |
| `cohort_with_cci.csv` | Step 8 | Cohort with computed CCI total score |
| `tier2_results/` | Step 9 | Tier 2 model outputs, attenuation analysis, G1vsG3 comparison |
| `cohort_tier3_ready.csv` | Step 10 | Fully enriched cohort with vaccination, severity, era, serology |
| `vaccination_summary.csv` | Step 10 | Vaccination coverage by group × era |
| `severity_summary.csv` | Step 10 | Severity breakdown by group × era |
| `enrichment_data_report.txt` | Step 10 | Detailed profiling of all merged data |

---

## 5. Population Denominator (for Standardisation)

Singapore Census 2020 resident population (≈4.04 million), broken down by age group and sex:

| Age Group | Male | Female |
|-----------|------|--------|
| 0-39 | 567,000 | 620,000 |
| 40-49 | 305,000 | 326,000 |
| 50-59 | 295,000 | 307,000 |
| 60-69 | 265,000 | 275,000 |
| 70-79 | 135,000 | 155,000 |
| 80+ | 55,000 | 80,000 |

Used for age-sex standardised incidence rates (ASIR) and expected case calculations (SIR).

---

## 6. Data Considerations

1. **Memory management**: MediClaims and SingCLOUD medication data are very large. Load one year/chunk at a time using the DataCatalog. The `inject_into()` method prints memory usage and headroom.

2. **Column naming**: Column names may vary slightly between datasets and years. Always inspect `df.columns` before assuming a column exists. Common variations: `PATIENT_ID` vs `patient_id` vs `PatientID`.

3. **Date formats**: Dates may appear as strings in various formats (`YYYY-MM-DD`, `DD/MM/YYYY`, etc.). Always parse explicitly with `pd.to_datetime(df['col'], format=...)` or use `dayfirst=True` if ambiguous.

4. **Duplicate patients**: A patient can appear in MediClaims multiple times (multiple diagnoses). Cohort assignment uses the *first* qualifying event. Always deduplicate to patient-level before analysis.

5. **Missing data**: SingCLOUD demographic linkage is not 100%. Some patients in MediClaims may not have gender/DOB records. Handle missing demographics gracefully.

6. **ICD-10 dot inconsistency**: Critical — codes appear as both `I21.0` and `I210`. All regex patterns and lookups must account for optional dots. The config.yaml lists both variants.

---

## 7. Step 10 Run Results (Confirmed)

Step 10 was executed against the actual cohort. The enrichment report confirmed the following.

### 7.1 Confirmed Cohort Sizes

| Group | N |
|-------|---|
| Group 1 (Post-COVID IHD) | 1,870 |
| Group 2 (COVID No-IHD) | 483,981 |
| Group 3 (Naive IHD) | 70,838 |
| Unknown | 75 |
| **Total** | **556,764** |

### 7.2 Confirmed Variant Era Distribution (COVID patients)

| Era | Group 1 | Group 2 |
|-----|---------|---------|
| Ancestral | 58 | 60,686 |
| Delta | 445 | 215,750 |
| Omicron | 1,367 | 207,545 |

These are the actual stratification counts for Tier 3. Note:
- Ancestral G1 has only 58 events (below the `MIN_EVENTS_FOR_MODEL=30` threshold, but borderline for 3-predictor models — may need Firth's penalized regression)
- Delta G1 has 445 events — adequate for stratified modelling
- Omicron G1 has 1,367 events — strong sample for all sub-stratifications

### 7.3 External Dataset Merge Failure (UNRESOLVED)

The five new datasets (COVIDFACILLOS, NIRListtruncated, COVID Reinfections, FacilityUtilizationLOSSubsequentRI, Serology_Tests_COVID) all failed to merge any data onto the cohort:

| Variable | Non-null count | Expected |
|----------|----------------|----------|
| `doses_before_ref` | 0 (0%) | ~485K+ for COVID patients |
| `vaccinated_before_covid` | 0 (0%) | ~485K+ |
| `fully_vaccinated_before_covid` | 0 (0%) | ~485K+ |
| `severity_category` | 556,764 (all "Unknown") | Mix of Mild/Moderate/Severe/Critical |
| `race` | 0 (0%) | ~485K+ for COVID patients |
| `serologyresult` | 0 (0%) | ~subset of COVID patients |
| `is_reinfection` | 0 reinfected | ~some subset |

**Root cause investigation needed.** Likely causes (check in order):

1. **Catalog alias mismatch**: The aliases in `config.yaml` (e.g., `"COVIDFACILLOS"`, `"COVID Reinfections"`) may not match the actual entries in the working `catalog.yaml` on SageMaker. Verify exact alias names with `cat.list_datasets()`.
2. **`uin` column name mismatch**: The cohort uses `uin` as the patient identifier. The new datasets also use `uin` (per the screenshots), but there may be casing differences (`UIN` vs `uin`) or the column may have a different name in practice. Check `df.columns` after loading each dataset.
3. **`uin` value format mismatch**: The cohort `uin` values (from COVID case registry + MediClaims) may be formatted differently from the `uin` values in the new datasets (e.g., leading zeros, string vs int types). Compare `df['uin'].dtype` and sample values across datasets.
4. **Catalog.yaml not updated**: The five new dataset entries may not have been added to the actual `catalog.yaml` file on SageMaker. The `config.yaml` references aliases, but those aliases must also exist in `catalog.yaml` with valid S3 paths.
5. **S3 path errors**: The datasets may be at different S3 paths than what's in the catalog.

**Debugging steps**:
```python
from catalog import DataCatalog
cat = DataCatalog("catalog.yaml")
print(cat.list_datasets())  # Check if aliases exist

# Try loading one dataset
df_test = cat.load("COVIDFACILLOS")
print(df_test.columns.tolist())  # Check column names
print(df_test['uin'].dtype, df_test['uin'].head())  # Check uin format

# Compare with cohort
df_cohort = pd.read_csv("data/02_processed/step_8_cci/cohort_enriched_cci.csv")
print(df_cohort['uin'].dtype, df_cohort['uin'].head())

# Check overlap
cohort_uins = set(df_cohort['uin'].unique())
new_uins = set(df_test['uin'].unique())
print(f"Overlap: {len(cohort_uins & new_uins):,} / {len(cohort_uins):,}")
```

### 7.4 Variables Successfully Derived (from existing data)

Despite the merge failures, these variables were derived from data already in the cohort:

| Variable | Source | Status |
|----------|--------|--------|
| `variant_era` | Derived from `covid_date` | Fully populated for all COVID patients |
| `severity_category` | Would use COVIDFACILLOS LOS/ICU/Deceased | All "Unknown" (no source data merged) |
| `is_reinfection` | Would use COVID Reinfections + FacilityUtilization | All 0 (no source data merged) |

### 7.5 Implications for Tier 3

Until the external dataset merge is resolved:
- **Component A (era-stratified G1 vs G2)**: Can proceed — uses only Age, Gender, CCI, and variant_era (all available)
- **Component B (vaccination-stratified)**: BLOCKED — requires vaccination data
- **Component C (era-stratified G1 vs G3)**: Can proceed — same covariates as A
- **Component D (severity exploratory)**: BLOCKED — requires severity data
- **Component E (interaction tests for era)**: Can proceed — era × CCI interaction uses existing data
