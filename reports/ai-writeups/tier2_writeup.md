# Tier 2 Analysis Report: CCI as a Confounder of IHD Risk Post-COVID

## Executive Summary

Tier 2 extends the Tier 1 demographic model by adding the **Charlson Comorbidity Index (CCI)** — a weighted measure of pre-existing disease burden. The central question:

> _"Among COVID patients of the same age, sex, **and baseline sickness level**, are IHD cases still emerging at excess rates?"_

**Bottom line**: CCI is a **highly significant** independent predictor of post-COVID IHD (OR = 1.26, p = 8.24×10⁻⁹²). Adding it improves the model (AUC 0.867 → 0.881, LR test p = 3.10×10⁻⁷⁸), but **demographics remain the dominant driver**. The AUC gain of +0.014 is modest — pre-existing sickness explains _some_ but **not all** of the excess risk.

---

## 1. Study Population

|                                |             |
| :----------------------------- | ----------: |
| Total COVID cohort (G1 + G2)   | **467,352** |
| G1 — COVID → IHD (cases)       |       1,826 |
| G2 — COVID → No IHD (controls) |     465,526 |
| CCI components available       |     16 / 17 |
| Event rate                     |       0.39% |

> [!NOTE]
> 16 of 17 Charlson categories were successfully mapped from the curated diagcodes (Step 7 → Step 8 pipeline). The CCI score was pre-computed by Step 8 using exact ICD code matching against mediclaims 2015–2023.

---

## 2. CCI Profile: Cases vs Controls

The CCI distributions reveal a **dramatic difference** between G1 and G2:

| Metric         |    G1 (IHD)    |  G2 (No IHD)   |
| :------------- | :------------: | :------------: |
| Mean CCI       |    **2.22**    |      0.48      |
| Median CCI     |     **2**      |       0        |
| CCI = 0 (%)    |       —        | ~78.9% overall |
| **SMD**        |   **1.005**    | (large effect) |
| Mann-Whitney p | **< 1×10⁻³⁰⁰** |       —        |

> [!IMPORTANT]
> An SMD of **1.005** is considered a _very large_ effect size (>0.8 = large). This means G1 patients carry substantially more comorbidity burden _before_ their COVID infection — they were already sicker.

### Most Prevalent CCI Conditions (all patients)

| Condition                   |      N | Prevalence |
| :-------------------------- | -----: | ---------: |
| Diabetes (uncomplicated)    | 47,240 |     10.11% |
| Malignancy (any)            | 30,721 |      6.57% |
| Dementia                    | 22,028 |      4.71% |
| Renal Disease               | 17,288 |      3.70% |
| Cerebrovascular Disease     | 11,242 |      2.41% |
| Chronic Pulmonary Disease   | 11,149 |      2.39% |
| Peripheral Vascular Disease |  7,538 |      1.61% |

**Extra conditions (not in CCI score):** Hypertension (15.6%), Hyperlipidemia (14.0%), Obesity (0.19%).

**Key correlation**: CCI and age have a Pearson r = **0.4975** — substantial but well below the multicollinearity threshold (~0.8), confirming they carry independent information.

---

## 3. Tier 2 Regression Results

**Model:** `logit(P(IHD)) = β₀ + β₁(Age) + β₂(Male) + β₃(CCI)`

| Variable                  | Coefficient |        OR |   95% CI    | z-stat |           p-value |
| :------------------------ | ----------: | --------: | :---------: | -----: | ----------------: |
| Intercept                 |     −9.6273 |    0.0001 |      —      | −86.17 |            <0.001 |
| **Age** (per year)        |      0.0590 | **1.061** | 1.058–1.064 |  39.97 |     <0.001 \*\*\* |
| **Male Sex**              |      0.6250 | **1.868** | 1.692–2.064 |  12.32 | 6.79×10⁻³⁵ \*\*\* |
| **CCI Score** (per point) |      0.2325 | **1.262** | 1.234–1.290 |  20.32 | 8.24×10⁻⁹² \*\*\* |

### Interpretation of Odds Ratios

- **Age**: Each additional year of age increases the odds of post-COVID IHD by **6.1%**, down from 7.1% in Tier 1. The attenuation is **14.8%** — meaning ~15% of the apparent "age effect" in Tier 1 was actually a "comorbidity effect" (older patients have higher CCI).

- **Male Sex**: Males have **87%** higher odds of IHD than females after adjusting for age and CCI (OR 1.868). This is attenuated **9.5%** from the Tier 1 estimate (OR 1.960), suggesting a small portion of the male excess risk was due to higher comorbidity burden in men.

- **CCI Score**: Each additional CCI point increases IHD odds by **26.2%**, independent of age and sex. This is a large and highly significant effect (z = 20.32), confirming that pre-existing comorbidity burden is a meaningful predictor.

---

## 4. Model Comparison: Tier 1 vs Tier 2

| Metric             |     Tier 1 |          Tier 2 |             Change |
| :----------------- | ---------: | --------------: | -----------------: |
| **AUC**            |     0.8669 |      **0.8807** |        **+0.0138** |
| **McFadden R²**    |   0.146520 |        0.161193 |          +0.014673 |
| **AIC**            |  20,399.94 |   **20,051.31** |            −348.62 |
| **BIC**            |  20,433.10 |       20,095.53 |            −337.57 |
| **Log-Likelihood** | −10,196.97 |      −10,021.66 |            +175.31 |
| **LR Test**        |          — | χ²=350.62, df=1 | **p = 3.10×10⁻⁷⁸** |

### What the Numbers Mean

1. **AUC +0.014**: Discrimination improved from 86.7% to 88.1%. While _statistically_ overwhelming (p ≈ 10⁻⁷⁸), this is a **modest** practical gain — the ROC curves are very close visually.

2. **AIC drop of 349**: AIC decreases > 10 are conventionally strong evidence. This 349-point drop is decisive — CCI unambiguously improves fit.

3. **LR Test p = 3.10×10⁻⁷⁸**: Overwhelmingly rejects the null, but with N = 467,352 even tiny effects reach extreme significance. _Statistical_ significance ≠ _practical_ importance.

4. **R² +0.015**: CCI explains ~1.5 additional percentage points of variance — meaningful but not transformative.

### OR Attenuation: The Key Table

| Variable            | OR (Tier 1) | OR (Tier 2) | Attenuation |
| :------------------ | ----------: | ----------: | ----------: |
| **Age** (per year)  |      1.0713 |      1.0608 |  **+14.8%** |
| **Male Sex**        |      1.9595 |      1.8683 |   **+9.5%** |
| **CCI** (per point) |           — |      1.2617 |     _(new)_ |

> [!IMPORTANT]
> The Age OR attenuated by **14.8%** — meaning roughly 1 in 7 of the apparent "age effect" was actually a "sickness effect" (older patients simply had more comorbidities). The Gender OR attenuated by **9.5%**, indicating a small portion of male excess risk was similarly confounded. Both attenuations are **modest**, leaving the core demographic signal largely intact.

---

## 5. Predicted Probability Curves

The predicted probability plots show the model-estimated 1-year IHD risk across the age spectrum, stratified by sex and CCI level (0, 2, 5).

### Males

| Age | CCI=0 | CCI=2 | CCI=5 |
| :-- | ----: | ----: | ----: |
| 35  | ~0.1% | ~0.2% | ~0.3% |
| 60  | ~0.5% | ~0.8% | ~1.3% |
| 75  | ~1.5% | ~2.3% | ~3.8% |
| 95  | ~3.1% | ~4.8% | ~9.5% |

### Females

| Age | CCI=0 | CCI=2 | CCI=5 |
| :-- | ----: | ----: | ----: |
| 35  | ~0.1% | ~0.1% | ~0.2% |
| 60  | ~0.3% | ~0.5% | ~0.7% |
| 75  | ~0.8% | ~1.3% | ~2.1% |
| 95  | ~1.7% | ~2.8% | ~5.3% |

**Key observations:**

- The CCI gradient is clearly visible: at age 95, a male with CCI=5 has roughly **3× the risk** of a male with CCI=0 (9.5% vs 3.1%)
- The **sex gap persists** across all CCI levels — at every age and CCI combination, males have roughly double the female risk
- The curves are exponential, with risk accelerating sharply after age ~65
- Even at the highest risk (95-year-old male, CCI=5), the predicted probability is ~9.5% — reflecting the very low base rate (0.39%) in this cohort

---

## 6. Calibration Plot

The calibration plot assesses whether predicted probabilities match observed event rates. The plot shows:

- **One very large bubble near the origin** — this represents the vast majority of patients (>460,000) with very low predicted risk (~0.3%), which matches the observed rate closely
- **A few small trailing points** at higher predicted probabilities (up to ~3%), also tracking the diagonal
- The plot reflects the **extreme class imbalance** (0.39% event rate), which compresses the bulk of predictions into a narrow low-risk band

> [!NOTE]
> The calibration is **adequate** where data density is sufficient (near the origin), but should be interpreted cautiously at higher predicted probabilities where sample sizes are very small. This is a known limitation of rare-event logistic regression.

---

## 7. Variance Decomposition: Who Drives the Model?

The drop-one partial R² analysis quantifies each predictor's unique contribution to the full model.

| Predictor      | R² without it |   Partial R² | % of Full R² |
| :------------- | ------------: | -----------: | -----------: |
| **Age**        |      0.080886 | **0.080308** |    **49.8%** |
| Gender (Male)  |      0.154505 |     0.006688 |         4.1% |
| CCI Score      |      0.146520 |     0.014673 |         9.1% |
| **Full Model** |             — | **0.161193** |            — |

> [!IMPORTANT]
> **Age alone accounts for half the model's explanatory power** (49.8%). Removing age collapses the R² from 0.161 to 0.081 — a catastrophic loss. CCI contributes 9.1% and Gender just 4.1%. This confirms that chronological age is the overwhelmingly dominant predictor of IHD risk post-COVID.

**Implication**: Even though CCI is highly significant (p = 10⁻⁹²), it contributes roughly **one-fifth** the explanatory power of age. The clinical picture is clear — age drives the risk, CCI modulates it, and sex plays a smaller but significant role.

---

## 8. Individual CCI Component Model: Which Comorbidities Matter?

Instead of a single composite CCI score, this model uses all 16 CCI comorbidity categories as separate binary predictors, revealing which specific conditions independently predict post-COVID IHD.

**Model**: `logit(IHD) = Age + Gender + 16 CCI binary flags`
**AUC**: 0.8878 (vs 0.8807 composite) | **R²**: 0.172994 | **AIC**: 19,799.34

### Significant Predictors (p < 0.05)

| Comorbidity                     |        OR |   95% CI   |    p-value | Interpretation                 |
| :------------------------------ | --------: | :--------: | ---------: | :----------------------------- |
| **Myocardial Infarction**       | **12.34** | 7.20–21.15 | 5.84×10⁻²⁰ | Prior MI → 12× higher IHD odds |
| **Diabetes (Complicated)**      |  **2.19** | 1.84–2.60  | 3.62×10⁻¹⁹ | End-organ damage doubles risk  |
| **Congestive Heart Failure**    |  **1.94** | 1.63–2.30  | 6.23×10⁻¹⁴ | Pre-existing cardiac disease   |
| **Rheumatic Disease**           |  **1.84** | 1.24–2.74  |  2.52×10⁻³ | Inflammatory/autoimmune        |
| **Renal Disease**               |  **1.82** | 1.62–2.05  | 1.26×10⁻²² | Cardiorenal axis               |
| **Diabetes (Uncomplicated)**    |  **1.54** | 1.38–1.71  | 7.21×10⁻¹⁵ | Metabolic risk                 |
| **Peripheral Vascular Disease** |  **1.34** | 1.12–1.60  |  1.61×10⁻³ | Systemic atherosclerosis       |
| **Cerebrovascular Disease**     |  **1.28** | 1.11–1.49  |  1.01×10⁻³ | Shared vascular substrate      |

### Protective / Non-significant

| Comorbidity               |       OR |  95% CI   |      p-value |
| :------------------------ | -------: | :-------: | -----------: |
| **Malignancy (Any)**      | **0.85** | 0.75–0.97 | **0.016 \*** |
| Dementia                  |     0.97 | 0.86–1.10 |        0.619 |
| Chronic Pulmonary Disease |     1.12 | 0.94–1.34 |        0.194 |
| Peptic Ulcer Disease      |     0.76 | 0.52–1.11 |        0.150 |

> [!WARNING]
> **Malignancy is _protective_ against IHD** (OR = 0.85, p = 0.016). This likely reflects **competing risk** / survivorship bias — cancer patients who survive to their COVID date may be under close medical surveillance, or may die of cancer before manifesting IHD. This is a known epidemiological finding and does _not_ mean cancer prevents heart disease.

**Key insight**: The cardiovascular comorbidities (prior MI, CHF, PVD, cerebrovascular disease) and metabolic conditions (diabetes, renal disease) are the strongest independent predictors. Notably, prior **MI has a 12× effect** — overwhelmingly the single strongest risk factor, but present in only 0.02% of the cohort, so its population-level impact is small.

---

## 9. G3 Comparison: The Critical COVID-Specificity Test

This section introduces **Group 3** (IHD patients who never had COVID) as a reference population, enabling a direct comparison of comorbidity profiles between COVID-associated and non-COVID IHD.

### 9a. CCI Profile: All Three Groups

| Group                  |         N | Mean CCI | Median |  CCI=0 % |
| :--------------------- | --------: | -------: | -----: | -------: |
| G1 (COVID → IHD)       |     1,826 |     2.22 |      2 |    29.0% |
| G2 (COVID → No IHD)    |   465,526 |     0.48 |      0 |    79.1% |
| **G3 (IHD, no COVID)** | **6,439** | **3.53** |  **3** | **0.0%** |

| Comparison   |    CCI SMD |   Age SMD |
| :----------- | ---------: | --------: |
| G1 vs G2     |     +1.005 |    +1.597 |
| **G1 vs G3** | **−0.586** | **0.188** |

> [!CAUTION]
> **G1 patients are _less_ sick than G3 patients** (CCI SMD = −0.586). Non-COVID IHD patients have a mean CCI of 3.53 compared to G1's 2.22 — they carry substantially _more_ comorbidity burden. Meanwhile, G1 and G3 are **almost the same age** (SMD = 0.188). This is a striking asymmetry: COVID-IHD patients develop heart disease despite being _healthier_ at baseline.

### 9b. COVID Exposure Model

**Question**: Among all IHD patients (G1 + G3), can we predict which ones had COVID exposure?

**Model**: `logit(P(COVID-exposed)) = Age + Gender + CCI`
**Population**: 8,265 IHD patients (G1 = 1,826, G3 = 6,439)

| Variable       |        OR |   95% CI    |         p-value |
| :------------- | --------: | :---------: | --------------: |
| Age (per year) |     1.021 | 1.017–1.026 |      1.60×10⁻²³ |
| Male Sex       | **0.739** | 0.653–0.836 |       1.57×10⁻⁶ |
| **CCI Score**  | **0.704** | 0.682–0.726 | **2.72×10⁻¹⁰⁷** |

**AUC**: 0.6963

> [!IMPORTANT]
> **This is the single most important finding in the Tier 2 analysis.**
>
> The CCI OR is **0.704** — each additional CCI point _decreases_ the odds of the IHD patient having been COVID-exposed by **30%**. In other words, **COVID-IHD patients are systematically _healthier_ at baseline than non-COVID IHD patients.**
>
> If COVID merely accelerated IHD in already-sick patients, we would expect CCI OR ≥ 1. The fact that it is **significantly below 1** (p = 2.72×10⁻¹⁰⁷) suggests COVID may cause IHD in **otherwise healthier patients** — patients who would not have developed IHD without the viral insult.

Also notable: **Male sex is protective** (OR = 0.739) in this model, meaning that among IHD patients, males are _less_ likely to have had COVID. This likely reflects the higher baseline IHD rate in non-COVID males (G3 has more males at baseline).

---

## 10. Updated Clinical Interpretation

### The Complete Picture: Three Lines of Evidence

**Line 1 — Tier 2 Composite Model (Sections 3-6)**:
CCI is a significant confounder (OR 1.26), but explains only 9.1% of model variance. Age dominates at 49.8%. The d emographic signal survives adjustment.

**Line 2 — Individual CCI Components (Section 8)**:
Cardiovascular comorbidities (MI, CHF) and metabolic conditions (diabetes, renal disease) drive the CCI effect. The individual model (AUC 0.888) performs only marginally better than the composite (0.881), confirming the CCI score captures the relevant information.

**Line 3 — G3 Comparison (Section 9)** ⭐:
**COVID-IHD patients are _less_ sick than non-COVID IHD patients** (CCI 2.22 vs 3.53, OR 0.704). This inverts the expected direction and provides the strongest evidence yet for a COVID-specific cardiovascular pathway — the virus appears to cause IHD in patients who would otherwise not have developed it.

### Revised Presentation Talking Points

> _"Our tiered analysis reveals three key findings:_
>
> _First, pre-existing comorbidity burden (CCI) modestly improves IHD prediction (AUC +0.014), but age alone accounts for 50% of the model's explanatory power._
>
> _Second, among individual comorbidities, prior MI (OR 12.3), diabetes with complications (OR 2.2), and CHF (OR 1.9) are the strongest independent risk factors._
>
> **_Third and most critically — COVID-IHD patients (G1) are significantly LESS sick at baseline than non-COVID IHD patients (G3): CCI 2.22 vs 3.53, with each CCI point reducing COVID-exposure odds by 30% (OR 0.70, p < 10⁻¹⁰⁷). This suggests COVID triggers IHD in otherwise healthier individuals who would not have developed heart disease through traditional pathways._**
>
> _Together, these findings support a COVID-specific cardiovascular mechanism that operates independently of conventional risk factors."_

---

## 11. Next Steps

> [!TIP]
> **Proceed to Tier 3**: The question now shifts from _who_ gets IHD (demographics, comorbidities) to _when_ and _under what conditions_ — specifically, whether the COVID era (wild-type vs Delta vs Omicron) and vaccination status modify the risk. The G3 comparison finding (COVID-IHD patients being healthier) provides strong motivation: if the excess risk diminishes in post-vaccination eras, the causal argument becomes very compelling.

---

_Report compiled from Step 9 outputs (27 Feb 2026). All numbers verified against pipeline logs and output files._
