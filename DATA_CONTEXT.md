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
