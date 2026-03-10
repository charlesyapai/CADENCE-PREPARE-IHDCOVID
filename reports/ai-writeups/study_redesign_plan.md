# IHD-COVID Study Redesign & Implementation Guide

## 1. Study Objective & Design

**Objective:** To quantify the independent impact of COVID-19 infection on the 1-year incidence of _De Novo_ NSTEMI and calculate the Population Attributable Fraction (PAF).

**Design:** Three-Group Retrospective Cohort Analysis.
**Timeline:**

- **Index Date:** Date of first positive COVID-19 test (for Cases) or assigned matched date (for Controls).
- **Follow-up:** 365 days from Index Date.
- **Outcome:** Incident _De Novo_ NSTEMI (ICD-10 I21.4).

### 1.1 Contrast Groups (The "Three Groups")

1.  **Group 1 (Post-COVID IHD):** COVID(+) patients who developed NSTEMI within 1 year.
    - _Source:_ Intersection of [COVID Cohort] and [IHD Registry].
2.  **Group 2 (Background IHD):** COVID(-) matched controls who developed NSTEMI within 1 year.
    - _Source:_ Intersection of [Control Cohort] and [IHD Registry].
3.  **Group 3 (COVID Non-IHD):** COVID(+) patients who did _not_ develop NSTEMI within 1 year.
    - _Source:_ [COVID Cohort] members NOT present in [IHD Registry] (or event >1 year).

_(Implicit Group 4: COVID(-) Controls without NSTEMI - needed for denominators)_

## 2. Data Sources & Extraction Strategy

We will utilize the following data sources. **Crucially, we must ensure we have the full denominators (Groups 3 & 4), not just the outcome events.**

### 2.1 Data Sources

| Dataset Name       | Description                           | Source File/Pattern          | Role                                             |
| :----------------- | :------------------------------------ | :--------------------------- | :----------------------------------------------- |
| **COVID Cohort**   | All patients with confirmed COVID-19. | `covid_events_matched.csv`   | Defines the Exposed Population (Groups 1 & 3).   |
| **Control Cohort** | Matched COVID-naive individuals.      | `control_cohort_matched.csv` | Defines the Unexposed Population (Groups 2 & 4). |
| **IHD Registry**   | List of all NSTEMI events.            | `index_events_extracted.csv` | Defines the Outcome (Numerator).                 |
| **Comorbidities**  | Baseline diagnoses (DM, HTN, etc.).   | `patient_comorbidities.csv`  | Adjustment Covariates.                           |
| **Medications**    | Baseline meds (Statins, etc.).        | `patient_medications.csv`    | Adjustment Covariates.                           |

### 2.2 Re-Processing Requirements (Addressing the "Missing Cohort")

The user indicated potential missingness of "COVID but no IHD". We must verify that `covid_events_matched.csv` is indeed the **full list of infected patients**, not just infected patients who had heart attacks.

**Action Item:**
If `covid_events_matched.csv` is already the full list of matched cases from the master COVID registry, we are good. If it was pre-filtered to equal the IHD list, we must return to the raw `covid_notifications` source (if available) or the `generate_control_cohort.py` input.

_Assumption for this plan:_ `covid_events_matched.csv` is the _full_ matched cohort (N=Large). `index_events_extracted.csv` is the _full_ list of heart attacks (N=Small). Merging them LEFT (Cohort -> Outcomes) naturally creates Group 3.

## 3. Implementation Steps & Code Edits

### Step 1: Update Data Extraction (`data_extraction_scripts.py`)

**Goal:** Ensure we extract covariates for _everyone_ in the full cohort, not just those with events.

- **Edit in `data_extraction_scripts.py`**:
  - Currently, it loads `INDEX_EVENTS_FILE` and `COVID_MATCHED_FILE` to build `target_uins`.
  - **Verify:** Ensure `COVID_MATCHED_FILE` yields the _full_ denominator of COVID patients.
  - **Verify:** Ensure `control_cohort_matched.csv` UINs are ALSO added to `target_uins`.
  - **Change:**
    ```python
    # Add Control Cohort to target UINs
    CONTROL_FILE = "control_cohort_matched.csv"
    if os.path.exists(CONTROL_FILE):
         df_ctrl = pd.read_csv(CONTROL_FILE, usecols=['uin'])
         target_uins.update(df_ctrl['uin'].unique())
    ```
  - _Rationale:_ If we don't scan comorbidities/meds for the Controls and the Non-Event COVID patients, we cannot run the logistic regression (missing X matrix).

### Step 2: Update Final Analysis (`final_analysis.py`)

**Goal:** Implement the 3-Group logic and Logistic/PAF model.

#### A. Construct the Logic for 3 Groups

- **Location:** After "Step 3: Exclusion".
- **Logic:**

  ```python
  # Define Binary Outcome (1 Year)
  cohort['Outcome_1Y'] = (cohort['Duration'] <= 365) & (cohort['Status'] == 1)
  cohort['Outcome_1Y'] = cohort['Outcome_1Y'].astype(int)

  # Define Analysis Groups
  conditions = [
      (cohort['Is_Case'] == 1) & (cohort['Outcome_1Y'] == 1), # Group 1
      (cohort['Is_Case'] == 0) & (cohort['Outcome_1Y'] == 1), # Group 2
      (cohort['Is_Case'] == 1) & (cohort['Outcome_1Y'] == 0), # Group 3
      (cohort['Is_Case'] == 0) & (cohort['Outcome_1Y'] == 0)  # Group 4 (Implicit)
  ]
  choices = ['G1: COVID+ IHD', 'G2: Naive IHD', 'G3: COVID+ No IHD', 'G4: Naive No IHD']
  cohort['Analysis_Group'] = np.select(conditions, choices, default='Unclassified')
  ```

#### B. Descriptive Stats (Table 1)

- **Edit:** Replace the generic `groupby('Group')` with a specific comparison of the 3 groups.
- **Task:** Create a helper to compute p-values (Chi2/T-test) between G1 vs G2 and G1 vs G3.

#### C. Statistical Modeling (Multivariate Logistic)

- **Edit:** Replace Cox & KM sections with Logistic Regression.
- **Library:** `import statsmodels.api as sm` (or `statsmodels.formula.api as smf`).
- **Code Structure:**

  ```python
  import statsmodels.formula.api as smf

  # Model 1: The Impact Model (Dependent: Outcome_1Y)
  # Predictors: Is_Case + Age + Gender + Comorbidities
  formula = "Outcome_1Y ~ Is_Case + Age + Gender + Diabetes + HTN + CKD + ..."
  model = smf.logit(formula, data=cohort).fit()

  print(model.summary())
  # Extract ORs: np.exp(model.params)
  ```

#### D. PAF Calculation

- **Edit:** Add a new section "Step 7: Contribution Analysis".
- **Method:**
  - $PAF = P(D) \times \frac{RR - 1}{RR}$ (Simple) OR
  - Average Attributable Fraction from the Model:
    1. Predict $P(Y=1 | \text{Actual Data})$
    2. Predict $P(Y=1 | \text{Counterfactual: Everyone COVID-free})$
    3. Sum of differences / Sum of actual cases.
- **Code:**

  ```python
  # Method: Direct Estimate from Model
  # 1. Total Predicted Probability with Actual Exposure
  cohort['prob_actual'] = model.predict(cohort)

  # 2. Counterfactual: Set Is_Case = 0 for everyone
  counterfactual_data = cohort.copy()
  counterfactual_data['Is_Case'] = 0
  cohort['prob_counterfactual'] = model.predict(counterfactual_data)

  # 3. Sum of Excess Cases due to COVID
  total_cases_observed = cohort['prob_actual'].sum()
  total_cases_no_covid = cohort['prob_counterfactual'].sum()
  excess_cases = total_cases_observed - total_cases_no_covid

  PAF = excess_cases / total_cases_observed
  print(f"Population Attributable Fraction (PAF): {PAF:.2%}")
  ```

## 4. Execution Plan

1.  **Modify `data_extraction_scripts.py`**: Add Control UINs to the scan list.
2.  **Verify Data**: Run extraction (simulated/check logic) to ensure we have `patient_comorbidities.csv` for _all_ UINs.
3.  **Modify `final_analysis.py`**: Refactor the pipeline to Logistic/PAF.
4.  **Run Analysis**: Execute and review `study_redesign_results.txt`.
