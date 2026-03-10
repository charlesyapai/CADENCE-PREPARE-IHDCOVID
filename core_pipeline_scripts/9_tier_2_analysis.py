"""
9_tier_2_analysis.py
====================
Tier 2 of the Stepwise DAG-Aligned Analysis:
    logit(P(IHD=1)) = b0 + b1(Age) + b2(Gender) + b3(CCI_Score)

Purpose:
    Determine whether the strong demographic signal from Tier 1 (AUC ~0.87)
    is further explained by pre-existing comorbidity burden, as measured
    by the Charlson Comorbidity Index (CCI).

    This is the CRITICAL test. Tier 1 showed age/gender strongly predict
    IHD within the COVID cohort. Tier 2 asks:

        "Among COVID patients of the SAME age, sex, AND baseline sickness
         level -- are IHD cases still emerging at excess rates?"

    If adding CCI barely changes the model (small AUC gain, non-significant
    LR test), then comorbidities don't explain the residual risk, and the
    COVID-specific pathway argument strengthens.

    If CCI substantially improves the model, then the excess risk is
    partly driven by sicker patients catching COVID and developing IHD
    at their baseline expected rate.

Population:
    COVID cohort only (Group 1 + Group 2), same as Tier 1.

CCI Computation:
    Uses the Comorb_*_Date columns from the Step 3 enriched cohort.
    Each comorbidity is binarized (present if date <= covid_date).
    CCI score = weighted sum using standard Charlson 1987 weights.

    MODIFIED CCI: The following categories are EXCLUDED from the index:
      - Myocardial_Infarction (weight 1): Overlaps with IHD outcome (ICD I21/I22)
      - Congestive_Heart_Failure (weight 1): Closely related to IHD outcome
      - AIDS_HIV (weight 6): Excluded due to sensitive patient data restrictions

Outputs (to data/03_results/step_9_tier2/):
    1.  tier2_model_summary.txt          -- Full logistic regression output
    2.  tier2_odds_ratios.csv            -- ORs with 95% CI
    3.  tier2_forest_plot.png            -- Forest plot of adjusted ORs
    4.  tier2_forest_comparison.png      -- Side-by-side Tier 1 vs Tier 2 ORs
    5.  tier2_cci_distribution.png       -- CCI score distribution G1 vs G2
    6.  tier2_cci_prevalence.png         -- Comorbidity prevalence by group
    7.  tier2_predicted_prob.png         -- Predicted probability by age, sex, CCI
    8.  tier2_roc_curve.png              -- ROC curve with AUC (+ Tier 1 overlay)
    9.  tier2_calibration_plot.png       -- Calibration plot
    10. tier2_attenuation_report.txt     -- Tier 1 vs Tier 2 comparison
    11. tier2_verbose_report.txt         -- Full narrative report
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import yaml
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger, ensure_dir

# ==============================================================================
# CCI WEIGHTS (Charlson et al., 1987)
# ==============================================================================
CCI_WEIGHTS = {
    # NOTE: Myocardial_Infarction, Congestive_Heart_Failure, and AIDS_HIV
    #       have been EXCLUDED from this modified CCI. See CCI_EXCLUSIONS below.
    'Peripheral_Vascular_Disease': 1,
    'Cerebrovascular_Disease': 1,
    'Dementia': 1,
    'Chronic_Pulmonary_Disease': 1,
    'Rheumatic_Disease': 1,
    'Peptic_Ulcer_Disease': 1,
    'Liver_Disease_Mild': 1,
    'Diabetes_Uncomplicated': 1,
    'Diabetes_Complicated': 2,
    'Paraplegia_Hemiplegia': 2,
    'Renal_Disease': 2,
    'Malignancy_Any': 2,
    'Liver_Disease_Severe': 3,
    'Metastatic_Solid_Tumor': 6,
}

# Categories EXCLUDED from this modified CCI:
#   - Myocardial_Infarction (weight 1): Overlaps with IHD outcome (ICD I21/I22)
#   - Congestive_Heart_Failure (weight 1): Closely related to IHD outcome
#   - AIDS_HIV (weight 6): Excluded due to sensitive patient data restrictions
CCI_EXCLUSIONS = ['Myocardial_Infarction', 'Congestive_Heart_Failure', 'AIDS_HIV']

# Additional conditions tracked but NOT in classic CCI
EXTRA_CONDITIONS = ['Hypertension', 'Hyperlipidemia', 'Obesity']


def compute_smd(g1_vals, g2_vals):
    """Standardised Mean Difference (Cohen's d)."""
    m1, m2 = g1_vals.mean(), g2_vals.mean()
    s1, s2 = g1_vals.std(), g2_vals.std()
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / pooled if pooled > 0 else 0.0


def make_forest_plot(odds_df, save_path, title="Forest Plot"):
    """Forest plot from DataFrame with Variable, OR, Lower_CI, Upper_CI."""
    fig, ax = plt.subplots(figsize=(9, max(3, len(odds_df) * 0.7 + 1.5)))
    y_pos = np.arange(len(odds_df))

    ax.errorbar(
        odds_df['OR'], y_pos,
        xerr=[odds_df['OR'] - odds_df['Lower_CI'],
              odds_df['Upper_CI'] - odds_df['OR']],
        fmt='o', color='#2c3e50', ecolor='#7f8c8d',
        elinewidth=2, capsize=4, markersize=8, zorder=3
    )
    ax.axvline(x=1.0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(odds_df['Variable'], fontsize=10)

    for i, row in odds_df.iterrows():
        ax.text(
            max(odds_df['Upper_CI']) * 1.08, y_pos[i],
            f"{row['OR']:.2f} ({row['Lower_CI']:.2f}-{row['Upper_CI']:.2f})",
            va='center', fontsize=8, color='#2c3e50'
        )

    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def run_step_9(config):
    # ------------------------------------------------------------------
    # 1. SETUP
    # ------------------------------------------------------------------
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_9_tier2")
    ensure_dir(results_dir)

    logger = setup_logger("tier_2", results_dir)
    logger.info("=" * 70)
    logger.info("TIER 2 ANALYSIS: Clinical Baseline Adjustment")
    logger.info("Model: logit(P(IHD)) = b0 + b1(Age) + b2(Male) + b3(CCI)")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 2. LOAD DATA (prefer Step 8 CCI-enriched, fallback to Step 3)
    # ------------------------------------------------------------------
    cci_file = os.path.join(processed_dir, "step_8_cci", "cohort_enriched_cci.csv")
    step3_file = os.path.join(processed_dir, "step_3_features", "cohort_enriched.csv")

    if os.path.exists(cci_file):
        input_file = cci_file
        logger.info("Using Step 8 CCI-enriched cohort")
    elif os.path.exists(step3_file):
        input_file = step3_file
        logger.warning("Step 8 output not found. Falling back to Step 3 cohort.")
        logger.warning("CCI columns may be missing. Run Step 8 first for best results.")
    else:
        logger.error("No enriched cohort found. Run Steps 3 and 8 first.")
        return None

    df_all = pd.read_csv(input_file)
    logger.info(f"Loaded cohort: {len(df_all):,} patients from {os.path.basename(input_file)}")

    # ------------------------------------------------------------------
    # 3. PREPARE COVID COHORT (G1 + G2)
    # ------------------------------------------------------------------
    covid = df_all[df_all['group'].isin(['Group 1', 'Group 2'])].copy()
    covid['outcome'] = (covid['group'] == 'Group 1').astype(int)

    # Age
    covid['age'] = pd.to_numeric(covid['age'], errors='coerce')

    # Gender
    gender_map = {'M': 1, 'Male': 1, 'MALE': 1, 'm': 1, '1': 1,
                  'F': 0, 'Female': 0, 'FEMALE': 0, 'f': 0, '0': 0}
    covid['gender_male'] = covid['gender'].astype(str).str.strip().map(gender_map)

    # Drop missing
    covid = covid.dropna(subset=['age', 'gender_male']).copy()
    covid['gender_male'] = covid['gender_male'].astype(int)

    n_g1 = covid['outcome'].sum()
    n_g2 = len(covid) - n_g1
    logger.info(f"COVID cohort: {len(covid):,} (G1={n_g1:,}, G2={n_g2:,})")

    # ------------------------------------------------------------------
    # 4. COMPUTE CCI SCORE
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("SECTION A: Computing Charlson Comorbidity Index")
    logger.info("-" * 50)

    covid_date = pd.to_datetime(covid['covid_date'], errors='coerce')

    # Try both column prefixes: Comorb_CCI_ (Step 8) and Comorb_ (Step 3)
    cci_flags = {}
    cci_available = []
    cci_missing = []

    for condition, weight in CCI_WEIGHTS.items():
        # Step 8 uses Comorb_CCI_ prefix, Step 3 uses Comorb_
        col_cci = f"Comorb_CCI_{condition}_Date"
        col_old = f"Comorb_{condition}_Date"
        col = col_cci if col_cci in covid.columns else (col_old if col_old in covid.columns else None)

        if col is not None:
            comorb_date = pd.to_datetime(covid[col], errors='coerce')
            flag = ((comorb_date.notnull()) & (comorb_date <= covid_date)).astype(int)
            covid[f"cci_{condition}"] = flag
            cci_flags[condition] = weight
            cci_available.append(condition)
            n_pos = flag.sum()
            logger.info(f"  {condition:35s} weight={weight}  N={n_pos:>6,} ({n_pos/len(covid)*100:.2f}%)")
        else:
            cci_missing.append(condition)
            logger.warning(f"  {condition}: NOT FOUND (tried {col_cci} and {col_old})")

    if cci_missing:
        logger.warning(f"  Missing {len(cci_missing)} CCI components: {cci_missing}")

    # Also binarize extra conditions (not in CCI score, but logged)
    for condition in EXTRA_CONDITIONS:
        col = f"Comorb_{condition}_Date"
        if col in covid.columns:
            comorb_date = pd.to_datetime(covid[col], errors='coerce')
            covid[f"extra_{condition}"] = ((comorb_date.notnull()) & (comorb_date <= covid_date)).astype(int)
            n_pos = covid[f"extra_{condition}"].sum()
            logger.info(f"  {condition:35s} (extra) N={n_pos:>6,} ({n_pos/len(covid)*100:.2f}%)")

    # Always recompute CCI score (Step 8 pre-computed scores include excluded
    # categories: MI, CHF, AIDS_HIV -- so we must recalculate from flags)
    logger.info("  Recomputing CCI score (excluding MI, CHF, AIDS_HIV)")
    covid['cci_score'] = 0
    for condition, weight in cci_flags.items():
        covid['cci_score'] += covid[f"cci_{condition}"] * weight

    logger.info(f"\n  CCI Score summary:")
    logger.info(f"    Mean:   {covid['cci_score'].mean():.2f}")
    logger.info(f"    Median: {covid['cci_score'].median():.0f}")
    logger.info(f"    Max:    {covid['cci_score'].max():.0f}")
    logger.info(f"    % with CCI=0: {(covid['cci_score'] == 0).mean() * 100:.1f}%")

    g1 = covid[covid['outcome'] == 1]
    g2 = covid[covid['outcome'] == 0]

    smd_cci = compute_smd(g1['cci_score'], g2['cci_score'])
    _, p_cci = stats.mannwhitneyu(g1['cci_score'], g2['cci_score'], alternative='two-sided')

    logger.info(f"\n  CCI by group:")
    logger.info(f"    G1 (IHD):    mean={g1['cci_score'].mean():.2f}, median={g1['cci_score'].median():.0f}")
    logger.info(f"    G2 (No-IHD): mean={g2['cci_score'].mean():.2f}, median={g2['cci_score'].median():.0f}")
    logger.info(f"    SMD: {smd_cci:.3f}, Mann-Whitney p={p_cci:.2e}")

    # ------------------------------------------------------------------
    # 5. CCI DISTRIBUTION & PREVALENCE PLOTS
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("SECTION B: CCI Visualizations")
    logger.info("-" * 50)

    # -- CCI Distribution --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_cci = int(min(covid['cci_score'].max(), 15))
    bins = np.arange(-0.5, max_cci + 1.5, 1)

    axes[0].hist(g2['cci_score'], bins=bins, alpha=0.5, color='#3498db',
                 label=f'G2 No-IHD (N={len(g2):,})', density=True)
    axes[0].hist(g1['cci_score'], bins=bins, alpha=0.7, color='#e74c3c',
                 label=f'G1 IHD (N={len(g1):,})', density=True)
    axes[0].set_xlabel('CCI Score', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('CCI Score Distribution: G1 vs G2', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    bp1 = axes[1].boxplot([g2['cci_score']], positions=[1], widths=0.5,
                          patch_artist=True, medianprops=dict(color='black', linewidth=2))
    bp1['boxes'][0].set(facecolor='#3498db', alpha=0.4)
    bp2 = axes[1].boxplot([g1['cci_score']], positions=[2], widths=0.5,
                          patch_artist=True, medianprops=dict(color='black', linewidth=2))
    bp2['boxes'][0].set(facecolor='#e74c3c', alpha=0.5)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['G2: No-IHD', 'G1: IHD'])
    axes[1].set_ylabel('CCI Score', fontsize=11)
    axes[1].set_title(f'CCI Comparison (SMD={smd_cci:.2f}, p={p_cci:.1e})',
                      fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier2_cci_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier2_cci_distribution.png")

    # -- Comorbidity Prevalence by Group --
    conditions_to_plot = [c for c in cci_available if f"cci_{c}" in covid.columns]
    if conditions_to_plot:
        g1_prev = [g1[f"cci_{c}"].mean() * 100 for c in conditions_to_plot]
        g2_prev = [g2[f"cci_{c}"].mean() * 100 for c in conditions_to_plot]
        labels = [c.replace('_', ' ') for c in conditions_to_plot]

        fig, ax = plt.subplots(figsize=(12, max(5, len(conditions_to_plot) * 0.4)))
        y = np.arange(len(labels))
        h = 0.35

        ax.barh(y - h/2, g1_prev, h, label=f'G1: IHD (N={len(g1):,})', color='#e74c3c', alpha=0.7)
        ax.barh(y + h/2, g2_prev, h, label=f'G2: No-IHD (N={len(g2):,})', color='#3498db', alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Prevalence (%)', fontsize=11)
        ax.set_title('Pre-COVID Comorbidity Prevalence: G1 vs G2', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', linestyle=':', alpha=0.4)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "tier2_cci_prevalence.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: tier2_cci_prevalence.png")

    # ------------------------------------------------------------------
    # 6. TIER 1 BASELINE (re-fit for comparison)
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("SECTION C: Tier 1 Baseline (Age + Gender only)")
    logger.info("-" * 50)

    model_t1 = smf.logit('outcome ~ age + gender_male', data=covid).fit(disp=0)
    aic_t1 = model_t1.aic
    bic_t1 = model_t1.bic
    ll_t1 = model_t1.llf

    from sklearn.metrics import roc_auc_score, roc_curve

    covid['pred_t1'] = model_t1.predict(covid)
    auc_t1 = roc_auc_score(covid['outcome'], covid['pred_t1'])
    r2_t1 = model_t1.prsquared

    logger.info(f"  Tier 1 AUC:  {auc_t1:.4f}")
    logger.info(f"  Tier 1 AIC:  {aic_t1:.2f}")
    logger.info(f"  Tier 1 R2:   {r2_t1:.6f}")

    # Extract Tier 1 ORs for comparison
    or_t1 = pd.DataFrame({
        'Variable': model_t1.params.index,
        'OR_T1': np.exp(model_t1.params.values),
        'Lower_T1': np.exp(model_t1.conf_int()[0].values),
        'Upper_T1': np.exp(model_t1.conf_int()[1].values),
    })

    # ------------------------------------------------------------------
    # 7. TIER 2 LOGISTIC REGRESSION
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SECTION D: TIER 2 -- Logistic Regression (Age + Gender + CCI)")
    logger.info("    logit(P(IHD)) = b0 + b1(Age) + b2(Male) + b3(CCI)")
    logger.info("=" * 70)

    # Diagnostic: check for issues before fitting
    logger.info(f"\n  Pre-fit diagnostics:")
    logger.info(f"    cci_score variance:    {covid['cci_score'].var():.6f}")
    logger.info(f"    cci_score unique vals: {covid['cci_score'].nunique()}")
    logger.info(f"    cci_score all zeros?   {(covid['cci_score'] == 0).all()}")
    logger.info(f"    corr(age, cci_score):  {covid['age'].corr(covid['cci_score']):.4f}")

    if covid['cci_score'].var() == 0:
        logger.error("  CCI score has ZERO variance (all same value). Cannot include in model.")
        logger.error("  This means no comorbidity dates were found before COVID dates.")
        logger.error("  Check that Comorb_*_Date columns are properly populated in the enriched cohort.")
        return None

    # Fit with robust optimizer (BFGS avoids Hessian inversion issues)
    try:
        model_t2 = smf.logit('outcome ~ age + gender_male + cci_score', data=covid).fit(
            method='bfgs', disp=0, maxiter=200
        )
    except Exception as e1:
        logger.warning(f"  BFGS fit failed: {e1}. Trying regularized fit...")
        try:
            model_t2 = smf.logit('outcome ~ age + gender_male + cci_score', data=covid).fit_regularized(
                method='l1', alpha=0.01, disp=0
            )
        except Exception as e2:
            logger.error(f"  Regularized fit also failed: {e2}")
            return None

    logger.info(f"\nConverged: {model_t2.mle_retvals.get('converged', 'N/A')}")

    # Full summary
    summary_text = model_t2.summary().as_text()
    with open(os.path.join(results_dir, "tier2_model_summary.txt"), "w") as f:
        f.write("TIER 2 LOGISTIC REGRESSION -- FULL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model: logit(P(IHD)) = b0 + b1(Age) + b2(Male) + b3(CCI)\n")
        f.write(f"Population: COVID cohort (G1 + G2), N = {len(covid):,}\n")
        f.write(f"Events (G1): {n_g1:,}  |  Non-Events (G2): {n_g2:,}\n")
        f.write(f"CCI components available: {len(cci_available)}/{len(CCI_WEIGHTS)}\n\n")
        f.write(summary_text)
    logger.info("  Saved: tier2_model_summary.txt")

    # Extract ORs
    params = model_t2.params
    conf = model_t2.conf_int()
    pvalues = model_t2.pvalues

    or_df = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'OR': np.exp(params.values),
        'Lower_CI': np.exp(conf[0].values),
        'Upper_CI': np.exp(conf[1].values),
        'p_value': pvalues.values,
        'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in pvalues.values]
    })

    or_df.to_csv(os.path.join(results_dir, "tier2_odds_ratios.csv"), index=False)
    logger.info("  Saved: tier2_odds_ratios.csv")

    for _, row in or_df.iterrows():
        logger.info(f"  {row['Variable']:20s}  OR={row['OR']:.4f}  "
                    f"(95% CI: {row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                    f"p={row['p_value']:.2e} {row['Significant']}")

    # Forest plot (Tier 2 only)
    plot_df = or_df[or_df['Variable'] != 'Intercept'].reset_index(drop=True)
    rename = {'age': 'Age (per year)', 'gender_male': 'Male Sex', 'cci_score': 'CCI Score (per point)'}
    plot_df['Variable'] = plot_df['Variable'].map(lambda x: rename.get(x, x))
    make_forest_plot(plot_df, os.path.join(results_dir, "tier2_forest_plot.png"),
                     title="Tier 2: Adjusted Odds Ratios\n(Age + Gender + CCI)")
    logger.info("  Saved: tier2_forest_plot.png")

    # ------------------------------------------------------------------
    # 8. MODEL DIAGNOSTICS
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("SECTION E: Model Diagnostics")
    logger.info("-" * 50)

    covid['pred_t2'] = model_t2.predict(covid)
    auc_t2 = roc_auc_score(covid['outcome'], covid['pred_t2'])
    r2_t2 = model_t2.prsquared
    aic_t2 = model_t2.aic
    bic_t2 = model_t2.bic

    logger.info(f"  AUC: {auc_t2:.4f}")
    logger.info(f"  R2:  {r2_t2:.6f}")
    logger.info(f"  AIC: {aic_t2:.2f}")
    logger.info(f"  BIC: {bic_t2:.2f}")

    # LR test: Tier 2 vs Tier 1
    lr_stat = -2 * (ll_t1 - model_t2.llf)
    lr_df = 1  # 1 extra parameter (cci_score)
    lr_pval = stats.chi2.sf(lr_stat, df=lr_df)
    logger.info(f"  LR Test (Tier 2 vs Tier 1): chi2={lr_stat:.2f}, df={lr_df}, p={lr_pval:.2e}")

    # ROC curve (both tiers overlaid)
    fpr_t1, tpr_t1, _ = roc_curve(covid['outcome'], covid['pred_t1'])
    fpr_t2, tpr_t2, _ = roc_curve(covid['outcome'], covid['pred_t2'])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_t1, tpr_t1, color='#7f8c8d', linewidth=1.5, linestyle='--',
            label=f'Tier 1: Age+Gender (AUC={auc_t1:.3f})')
    ax.plot(fpr_t2, tpr_t2, color='#2c3e50', linewidth=2,
            label=f'Tier 2: +CCI (AUC={auc_t2:.3f})')
    ax.plot([0, 1], [0, 1], color='#bdc3c7', linestyle=':', linewidth=1)
    ax.fill_between(fpr_t2, tpr_t2, alpha=0.08, color='#2c3e50')
    ax.set_xlabel('1 - Specificity (FPR)', fontsize=11)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=11)
    ax.set_title(f'ROC Curves: Tier 1 vs Tier 2', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier2_roc_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier2_roc_curve.png")

    # Predicted probability by age, sex, and CCI
    age_range = np.arange(35, 95, 1)
    cci_levels = [0, 2, 5]
    colors_m = ['#2980b9', '#1a5276', '#0b2e4a']
    colors_f = ['#e74c3c', '#a93226', '#641e16']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, cci in enumerate(cci_levels):
        pred_m = model_t2.predict(pd.DataFrame({'age': age_range, 'gender_male': 1, 'cci_score': cci}))
        pred_f = model_t2.predict(pd.DataFrame({'age': age_range, 'gender_male': 0, 'cci_score': cci}))
        axes[0].plot(age_range, pred_m * 100, color=colors_m[i], linewidth=2,
                     label=f'Male, CCI={cci}')
        axes[1].plot(age_range, pred_f * 100, color=colors_f[i], linewidth=2,
                     label=f'Female, CCI={cci}')

    for ax, title in zip(axes, ['Males', 'Females']):
        ax.set_xlabel('Age (years)', fontsize=11)
        ax.set_ylabel('Predicted 1-Year IHD Risk (%)', fontsize=11)
        ax.set_title(f'Tier 2: Predicted Risk -- {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(linestyle=':', alpha=0.4)
        ax.set_xlim(35, 95)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier2_predicted_prob.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier2_predicted_prob.png")

    # Calibration
    try:
        n_bins = 10
        bin_edges = np.linspace(0, covid['pred_t2'].max() * 1.01, n_bins + 1)
        covid['prob_bin'] = pd.cut(covid['pred_t2'], bins=bin_edges, include_lowest=True)
        cal = covid.groupby('prob_bin', observed=True).agg(
            mean_pred=('pred_t2', 'mean'), mean_obs=('outcome', 'mean'), n=('outcome', 'count')
        ).dropna()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, cal['mean_pred'].max()], [0, cal['mean_pred'].max()],
                color='#bdc3c7', linestyle='--', linewidth=1)
        ax.scatter(cal['mean_pred'], cal['mean_obs'], s=cal['n']/cal['n'].max()*300,
                   color='#2c3e50', alpha=0.7, edgecolor='white', zorder=3)
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Proportion', fontsize=11)
        ax.set_title('Tier 2: Calibration Plot', fontsize=13, fontweight='bold')
        ax.grid(linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "tier2_calibration_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: tier2_calibration_plot.png")
    except Exception as e:
        logger.warning(f"  Calibration plot failed: {e}")

    # ------------------------------------------------------------------
    # 8b. MODEL DIAGNOSTICS (Logistic Regression Goodness-of-Fit)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SECTION E2: Logistic Regression Model Diagnostics")
    logger.info("=" * 70)

    diag_lines = []
    diag_lines.append("TIER 2: LOGISTIC REGRESSION MODEL DIAGNOSTICS")
    diag_lines.append("=" * 60)
    diag_lines.append("")

    # --- 8b-i. Hosmer-Lemeshow Goodness-of-Fit Test ---
    logger.info("\n  --- Hosmer-Lemeshow Goodness-of-Fit Test ---")
    diag_lines.append("1. HOSMER-LEMESHOW GOODNESS-OF-FIT TEST")
    diag_lines.append("-" * 45)

    try:
        n_hl_groups = 10
        covid['pred_t2_grp'] = pd.qcut(covid['pred_t2'], q=n_hl_groups, duplicates='drop')
        hl_table = covid.groupby('pred_t2_grp', observed=True).agg(
            n=('outcome', 'count'),
            obs_events=('outcome', 'sum'),
            mean_pred=('pred_t2', 'mean'),
        ).reset_index()
        hl_table['exp_events'] = hl_table['mean_pred'] * hl_table['n']
        hl_table['obs_nonevents'] = hl_table['n'] - hl_table['obs_events']
        hl_table['exp_nonevents'] = hl_table['n'] - hl_table['exp_events']

        # HL statistic: sum of (O-E)^2/E for events + nonevents
        hl_stat = 0
        for _, row in hl_table.iterrows():
            if row['exp_events'] > 0:
                hl_stat += (row['obs_events'] - row['exp_events'])**2 / row['exp_events']
            if row['exp_nonevents'] > 0:
                hl_stat += (row['obs_nonevents'] - row['exp_nonevents'])**2 / row['exp_nonevents']

        hl_df = len(hl_table) - 2  # g - 2
        hl_pval = stats.chi2.sf(hl_stat, df=hl_df)

        logger.info(f"  Groups: {len(hl_table)}")
        logger.info(f"  HL statistic: {hl_stat:.2f}")
        logger.info(f"  df: {hl_df}")
        logger.info(f"  p-value: {hl_pval:.4e}")

        diag_lines.append(f"  Groups: {len(hl_table)}")
        diag_lines.append(f"  HL Statistic: {hl_stat:.2f}")
        diag_lines.append(f"  Degrees of freedom: {hl_df}")
        diag_lines.append(f"  p-value: {hl_pval:.4e}")
        diag_lines.append("")

        if hl_pval >= 0.05:
            hl_interp = "PASS: No evidence of poor fit (p >= 0.05). Model calibration is adequate."
        else:
            hl_interp = ("CAUTION: Significant HL test (p < 0.05) suggests some miscalibration.\n"
                         "  However, with N=467,352, even minor deviations reach significance.\n"
                         "  Assess practical calibration via the calibration plot above.")
        logger.info(f"  {hl_interp}")
        diag_lines.append(f"  Interpretation: {hl_interp}")
        diag_lines.append("")

        # HL table detail
        diag_lines.append("  Decile table:")
        diag_lines.append(f"  {'Decile':<8} {'N':>8} {'Obs':>8} {'Exp':>10} {'O/E':>8}")
        diag_lines.append(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
        for idx, row in hl_table.iterrows():
            oe = row['obs_events'] / row['exp_events'] if row['exp_events'] > 0 else 0
            diag_lines.append(f"  {idx+1:<8} {int(row['n']):>8} {int(row['obs_events']):>8} "
                              f"{row['exp_events']:>10.1f} {oe:>8.3f}")
        diag_lines.append("")

        covid.drop(columns=['pred_t2_grp'], inplace=True, errors='ignore')

    except Exception as e:
        logger.warning(f"  Hosmer-Lemeshow test failed: {e}")
        diag_lines.append(f"  Failed: {e}")
        diag_lines.append("")

    # --- 8b-ii. Deviance Residuals ---
    logger.info("\n  --- Deviance Residuals ---")
    diag_lines.append("2. DEVIANCE RESIDUALS")
    diag_lines.append("-" * 45)

    try:
        # Compute deviance residuals manually
        pred = covid['pred_t2'].values
        y = covid['outcome'].values
        pred_clipped = np.clip(pred, 1e-10, 1 - 1e-10)

        d_i = np.where(
            y == 1,
            np.sqrt(-2 * np.log(pred_clipped)),
            -np.sqrt(-2 * np.log(1 - pred_clipped))
        )
        covid['deviance_resid'] = d_i

        dr_mean = d_i.mean()
        dr_std = d_i.std()
        dr_skew = stats.skew(d_i)
        dr_kurtosis = stats.kurtosis(d_i)

        logger.info(f"  Mean:     {dr_mean:.4f} (should be near 0)")
        logger.info(f"  Std:      {dr_std:.4f}")
        logger.info(f"  Skewness: {dr_skew:.4f} (should be near 0)")
        logger.info(f"  Kurtosis: {dr_kurtosis:.4f} (excess, should be near 0)")

        diag_lines.append(f"  Mean:     {dr_mean:.4f} (ideal: ~0)")
        diag_lines.append(f"  Std:      {dr_std:.4f}")
        diag_lines.append(f"  Skewness: {dr_skew:.4f} (ideal: ~0)")
        diag_lines.append(f"  Kurtosis: {dr_kurtosis:.4f} (excess, ideal: ~0)")
        diag_lines.append("")

        if abs(dr_skew) > 2:
            diag_lines.append("  NOTE: High skewness is expected with rare events (0.39% rate).")
            diag_lines.append("  This does NOT indicate model failure for logistic regression.")
        diag_lines.append("")

        # Plot: Deviance residual distribution + Q-Q plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) Histogram of deviance residuals
        axes[0].hist(d_i, bins=100, color='#2c3e50', alpha=0.7, edgecolor='white', density=True)
        axes[0].axvline(0, color='#e74c3c', linestyle='--', linewidth=1.5)
        axes[0].set_xlabel('Deviance Residual', fontsize=11)
        axes[0].set_ylabel('Density', fontsize=11)
        axes[0].set_title('Deviance Residual Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', linestyle=':', alpha=0.4)

        # (b) Q-Q plot of deviance residuals
        # Sample if too large for plotting performance
        n_sample = min(len(d_i), 50000)
        d_sample = np.sort(np.random.choice(d_i, size=n_sample, replace=False))
        theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, n_sample))

        axes[1].scatter(theoretical, d_sample, s=1, alpha=0.3, color='#2c3e50')
        # Reference line
        q25, q75 = np.percentile(d_sample, [25, 75])
        t25, t75 = stats.norm.ppf([0.25, 0.75])
        slope = (q75 - q25) / (t75 - t25)
        intercept = q25 - slope * t25
        x_line = np.array([theoretical.min(), theoretical.max()])
        axes[1].plot(x_line, slope * x_line + intercept, color='#e74c3c',
                     linewidth=2, linestyle='--', label='Reference line')
        axes[1].set_xlabel('Theoretical Quantiles (Normal)', fontsize=11)
        axes[1].set_ylabel('Deviance Residual Quantiles', fontsize=11)
        axes[1].set_title('Q-Q Plot: Deviance Residuals', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(linestyle=':', alpha=0.4)

        # (c) Residuals vs predicted probability
        # Sample for scatter
        idx_sample = np.random.choice(len(pred), size=min(20000, len(pred)), replace=False)
        axes[2].scatter(pred[idx_sample], d_i[idx_sample], s=1, alpha=0.2, color='#2c3e50')
        axes[2].axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5)
        axes[2].axhline(2, color='#e67e22', linestyle=':', alpha=0.5)
        axes[2].axhline(-2, color='#e67e22', linestyle=':', alpha=0.5)
        axes[2].set_xlabel('Predicted Probability', fontsize=11)
        axes[2].set_ylabel('Deviance Residual', fontsize=11)
        axes[2].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[2].grid(linestyle=':', alpha=0.4)

        plt.suptitle('Tier 2 Model Diagnostics', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "tier2_diagnostics_residuals.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: tier2_diagnostics_residuals.png")

    except Exception as e:
        logger.warning(f"  Deviance residual diagnostics failed: {e}")
        diag_lines.append(f"  Failed: {e}")
        diag_lines.append("")

    # --- 8b-iii. Influence Diagnostics (Cook's Distance) ---
    logger.info("\n  --- Influence Diagnostics ---")
    diag_lines.append("3. INFLUENCE DIAGNOSTICS")
    diag_lines.append("-" * 45)

    try:
        influence = model_t2.get_influence()
        cooks_d = influence.cooks_distance[0]
        hat_vals = influence.hat_matrix_diag

        # Thresholds
        n_obs = len(covid)
        n_params = len(model_t2.params)
        cook_thresh = 4 / n_obs
        hat_thresh = 2 * n_params / n_obs

        n_influential_cook = (cooks_d > cook_thresh).sum()
        n_high_leverage = (hat_vals > hat_thresh).sum()

        logger.info(f"  Cook's D threshold (4/n): {cook_thresh:.6f}")
        logger.info(f"  Influential obs (Cook's D > threshold): {n_influential_cook:,} ({n_influential_cook/n_obs*100:.2f}%)")
        logger.info(f"  Hat threshold (2p/n): {hat_thresh:.6f}")
        logger.info(f"  High leverage obs: {n_high_leverage:,} ({n_high_leverage/n_obs*100:.2f}%)")
        logger.info(f"  Max Cook's D: {cooks_d.max():.6f}")

        diag_lines.append(f"  Cook's D threshold (4/n): {cook_thresh:.6f}")
        diag_lines.append(f"  Influential observations: {n_influential_cook:,} ({n_influential_cook/n_obs*100:.2f}%)")
        diag_lines.append(f"  Hat threshold (2p/n): {hat_thresh:.6f}")
        diag_lines.append(f"  High leverage observations: {n_high_leverage:,} ({n_high_leverage/n_obs*100:.2f}%)")
        diag_lines.append(f"  Max Cook's D: {cooks_d.max():.6f}")
        diag_lines.append("")

        if cooks_d.max() < 0.5:
            diag_lines.append("  PASS: No single observation has undue influence (max Cook's D < 0.5).")
        else:
            diag_lines.append("  WARNING: Some observations have high influence. Inspect them.")
        diag_lines.append("")

        # Influence plot: Cook's D vs hat values
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) Cook's distance plot (sorted, top 50)
        cook_sorted = np.sort(cooks_d)[::-1][:50]
        axes[0].bar(range(len(cook_sorted)), cook_sorted, color='#2c3e50', alpha=0.7)
        axes[0].axhline(cook_thresh, color='#e74c3c', linestyle='--', linewidth=1.5,
                        label=f'Threshold (4/n = {cook_thresh:.2e})')
        axes[0].set_xlabel('Observation Rank', fontsize=11)
        axes[0].set_ylabel("Cook's Distance", fontsize=11)
        axes[0].set_title("Top 50 Cook's Distances", fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(axis='y', linestyle=':', alpha=0.4)

        # (b) Leverage vs residuals squared (bubble = Cook's D)
        idx_sample = np.random.choice(n_obs, size=min(20000, n_obs), replace=False)
        axes[1].scatter(hat_vals[idx_sample], d_i[idx_sample]**2, s=2, alpha=0.2, color='#2c3e50')
        axes[1].axvline(hat_thresh, color='#e74c3c', linestyle='--', alpha=0.6)
        axes[1].axhline(4, color='#e67e22', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('Leverage (Hat Value)', fontsize=11)
        axes[1].set_ylabel('Residual^2', fontsize=11)
        axes[1].set_title('Leverage vs Residuals', fontsize=12, fontweight='bold')
        axes[1].grid(linestyle=':', alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "tier2_diagnostics_influence.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: tier2_diagnostics_influence.png")

    except Exception as e:
        logger.warning(f"  Influence diagnostics failed: {e}")
        diag_lines.append(f"  Failed: {e}")
        diag_lines.append("")

    # --- 8b-iv. Link Test (Specification Error) ---
    logger.info("\n  --- Link Test ---")
    diag_lines.append("4. LINK TEST (Specification Error)")
    diag_lines.append("-" * 45)

    try:
        # Pregibon link test: regress outcome on yhat and yhat^2
        # If yhat^2 is significant, model may be misspecified
        covid['yhat'] = model_t2.predict(covid)
        covid['yhat_sq'] = covid['yhat'] ** 2

        link_model = smf.logit('outcome ~ yhat + yhat_sq', data=covid).fit(disp=0, maxiter=100)

        yhat_p = link_model.pvalues.get('yhat', 1.0)
        yhat_sq_p = link_model.pvalues.get('yhat_sq', 1.0)

        logger.info(f"  yhat coefficient p-value:    {yhat_p:.4e} (should be significant)")
        logger.info(f"  yhat^2 coefficient p-value:  {yhat_sq_p:.4e} (should be NON-significant)")

        diag_lines.append(f"  yhat p-value:   {yhat_p:.4e} (should be significant)")
        diag_lines.append(f"  yhat^2 p-value: {yhat_sq_p:.4e} (should be NON-significant)")
        diag_lines.append("")

        if yhat_sq_p >= 0.05:
            link_interp = "PASS: No evidence of specification error (yhat^2 not significant)."
        else:
            link_interp = ("CAUTION: yhat^2 is significant, suggesting possible misspecification.\n"
                           "  Consider interactions (e.g., Age*CCI) or nonlinear terms.\n"
                           "  With very large N, minor nonlinearities can reach significance.\n"
                           "  Assess practical impact by comparing AUC with enriched models.")
        logger.info(f"  {link_interp}")
        diag_lines.append(f"  Interpretation: {link_interp}")
        diag_lines.append("")

        covid.drop(columns=['yhat', 'yhat_sq'], inplace=True, errors='ignore')

    except Exception as e:
        logger.warning(f"  Link test failed: {e}")
        diag_lines.append(f"  Failed: {e}")
        diag_lines.append("")

    # --- Save diagnostic report ---
    diag_lines.append("")
    diag_lines.append("OVERALL ASSESSMENT")
    diag_lines.append("=" * 45)
    diag_lines.append("For logistic regression, deviance residuals are NOT expected")
    diag_lines.append("to be normally distributed (unlike linear regression). The Q-Q")
    diag_lines.append("plot will show bimodality due to the binary outcome. This is")
    diag_lines.append("NORMAL behavior, not a model deficiency.")
    diag_lines.append("")
    diag_lines.append("Key diagnostics to focus on:")
    diag_lines.append("  1. Hosmer-Lemeshow: tests calibration across risk deciles")
    diag_lines.append("  2. Cook's D: detects single obs with disproportionate influence")
    diag_lines.append("  3. Link test: detects misspecification (missing terms)")
    diag_lines.append("  4. Calibration plot: visual check of predicted vs observed")

    diag_path = os.path.join(results_dir, "tier2_diagnostics_report.txt")
    with open(diag_path, 'w') as f:
        f.write('\n'.join(diag_lines))
    logger.info(f"  Saved: tier2_diagnostics_report.txt")
    # ------------------------------------------------------------------
    # 9. TIER 1 vs TIER 2: SIDE-BY-SIDE FOREST PLOT
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("SECTION F: Tier 1 vs Tier 2 Comparison")
    logger.info("-" * 50)

    # Merge ORs for shared variables
    shared_vars = ['age', 'gender_male']
    comp_rows = []
    for var in shared_vars:
        nice = rename.get(var, var)
        t1_or = np.exp(model_t1.params[var])
        t1_lo = np.exp(model_t1.conf_int().loc[var, 0])
        t1_hi = np.exp(model_t1.conf_int().loc[var, 1])
        t2_or = np.exp(model_t2.params[var])
        t2_lo = np.exp(model_t2.conf_int().loc[var, 0])
        t2_hi = np.exp(model_t2.conf_int().loc[var, 1])
        comp_rows.append({
            'Variable': nice, 'Tier': 'Tier 1', 'OR': t1_or,
            'Lower_CI': t1_lo, 'Upper_CI': t1_hi
        })
        comp_rows.append({
            'Variable': nice, 'Tier': 'Tier 2', 'OR': t2_or,
            'Lower_CI': t2_lo, 'Upper_CI': t2_hi
        })
    # Add CCI (Tier 2 only)
    cci_or = np.exp(model_t2.params['cci_score'])
    cci_lo = np.exp(model_t2.conf_int().loc['cci_score', 0])
    cci_hi = np.exp(model_t2.conf_int().loc['cci_score', 1])
    comp_rows.append({
        'Variable': 'CCI Score', 'Tier': 'Tier 2', 'OR': cci_or,
        'Lower_CI': cci_lo, 'Upper_CI': cci_hi
    })

    comp_df = pd.DataFrame(comp_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    tier_colors = {'Tier 1': '#7f8c8d', 'Tier 2': '#2c3e50'}
    variables = comp_df['Variable'].unique()
    y_base = np.arange(len(variables))

    for tier, offset in [('Tier 1', -0.15), ('Tier 2', 0.15)]:
        sub = comp_df[comp_df['Tier'] == tier]
        for _, row in sub.iterrows():
            y_idx = np.where(variables == row['Variable'])[0][0]
            ax.errorbar(row['OR'], y_idx + offset,
                       xerr=[[row['OR'] - row['Lower_CI']], [row['Upper_CI'] - row['OR']]],
                       fmt='o', color=tier_colors[tier], elinewidth=2, capsize=4,
                       markersize=8, label=tier if row['Variable'] == variables[0] else '')
            ax.text(max(comp_df['Upper_CI']) * 1.1, y_idx + offset,
                   f"{row['OR']:.3f}", va='center', fontsize=8, color=tier_colors[tier])

    ax.axvline(1.0, color='#e74c3c', linestyle='--', alpha=0.6)
    ax.set_yticks(y_base)
    ax.set_yticklabels(variables, fontsize=11)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
    ax.set_title('OR Comparison: Tier 1 (Age+Gender) vs Tier 2 (+CCI)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.invert_yaxis()
    handles = [plt.Line2D([0], [0], marker='o', color=c, linestyle='', markersize=8)
               for c in tier_colors.values()]
    ax.legend(handles, tier_colors.keys(), fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier2_forest_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier2_forest_comparison.png")

    # ------------------------------------------------------------------
    # 10. ATTENUATION REPORT
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SECTION G: Attenuation Analysis (Tier 1 -> Tier 2)")
    logger.info("=" * 70)

    age_or_t1 = np.exp(model_t1.params['age'])
    age_or_t2 = np.exp(model_t2.params['age'])
    gender_or_t1 = np.exp(model_t1.params['gender_male'])
    gender_or_t2 = np.exp(model_t2.params['gender_male'])

    # Attenuation: % change in OR-1 (on log-OR scale)
    age_atten = ((age_or_t1 - 1) - (age_or_t2 - 1)) / (age_or_t1 - 1) * 100 if age_or_t1 != 1 else 0
    gender_atten = ((gender_or_t1 - 1) - (gender_or_t2 - 1)) / (gender_or_t1 - 1) * 100 if gender_or_t1 != 1 else 0

    atten_path = os.path.join(results_dir, "tier2_attenuation_report.txt")
    with open(atten_path, "w") as f:
        f.write("TIER 2 ATTENUATION REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write("QUESTION: After accounting for pre-existing comorbidity burden\n")
        f.write("(CCI), does the demographic signal from Tier 1 attenuate?\n\n")

        f.write("--- Model Comparison ---\n\n")
        f.write(f"  {'Metric':<25} {'Tier 1':>12} {'Tier 2':>12} {'Change':>12}\n")
        f.write(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}\n")
        f.write(f"  {'AUC':<25} {auc_t1:>12.4f} {auc_t2:>12.4f} {auc_t2-auc_t1:>+12.4f}\n")
        f.write(f"  {'McFadden R2':<25} {r2_t1:>12.6f} {r2_t2:>12.6f} {r2_t2-r2_t1:>+12.6f}\n")
        f.write(f"  {'AIC':<25} {aic_t1:>12.2f} {aic_t2:>12.2f} {aic_t2-aic_t1:>+12.2f}\n")
        f.write(f"  {'BIC':<25} {bic_t1:>12.2f} {bic_t2:>12.2f} {bic_t2-bic_t1:>+12.2f}\n")
        f.write(f"  {'Log-Likelihood':<25} {ll_t1:>12.2f} {model_t2.llf:>12.2f} {model_t2.llf-ll_t1:>+12.2f}\n\n")

        f.write(f"  LR Test (Tier 2 vs Tier 1): chi2={lr_stat:.2f}, df={lr_df}, p={lr_pval:.2e}\n")
        sig_lr = "YES -- CCI significantly improves model" if lr_pval < 0.05 else "NO -- CCI does NOT significantly improve model"
        f.write(f"  Significant? {sig_lr}\n\n")

        f.write("--- OR Attenuation ---\n\n")
        f.write(f"  {'Variable':<20} {'OR (Tier 1)':>12} {'OR (Tier 2)':>12} {'Attenuation':>12}\n")
        f.write(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}\n")
        f.write(f"  {'Age (per year)':<20} {age_or_t1:>12.4f} {age_or_t2:>12.4f} {age_atten:>+11.1f}%\n")
        f.write(f"  {'Male Sex':<20} {gender_or_t1:>12.4f} {gender_or_t2:>12.4f} {gender_atten:>+11.1f}%\n")
        f.write(f"  {'CCI (per point)':<20} {'--':>12} {cci_or:>12.4f} {'(new)':>12}\n\n")

        f.write("--- CCI Effect ---\n\n")
        f.write(f"  CCI OR: {cci_or:.4f} (95% CI: {cci_lo:.4f}-{cci_hi:.4f})\n")
        f.write(f"  p-value: {model_t2.pvalues['cci_score']:.2e}\n\n")
        if cci_or > 1 and model_t2.pvalues['cci_score'] < 0.05:
            f.write(f"  Each additional CCI point increases the odds of IHD by\n")
            f.write(f"  {(cci_or - 1) * 100:.1f}%, independent of age and gender.\n\n")
        elif model_t2.pvalues['cci_score'] >= 0.05:
            f.write(f"  CCI is NOT a significant independent predictor of IHD\n")
            f.write(f"  after adjusting for age and gender.\n\n")

        f.write("--- INTERPRETATION ---\n\n")

        auc_gain = auc_t2 - auc_t1
        if auc_gain < 0.01 and lr_pval >= 0.05:
            f.write("  FINDING: Adding CCI provides NEGLIGIBLE improvement.\n\n")
            f.write("  Pre-existing comorbidity burden does NOT explain the\n")
            f.write("  excess IHD risk beyond what demographics already capture.\n")
            f.write("  This STRENGTHENS the argument for a COVID-specific\n")
            f.write("  cardiovascular pathway.\n\n")
            f.write("  -> PROCEED to Tier 3 (Era/Vaccination stratification)\n")
        elif auc_gain < 0.02:
            f.write("  FINDING: CCI adds MODEST additional information.\n\n")
            f.write("  Some of the excess risk is attributable to sicker\n")
            f.write("  patients, but the effect is small. Demographics remain\n")
            f.write("  the dominant predictor.\n\n")
            f.write("  -> PROCEED to Tier 3 to test temporal/vaccination effects.\n")
        else:
            f.write("  FINDING: CCI adds SUBSTANTIAL explanatory power.\n\n")
            f.write("  Pre-existing disease burden is an important confound.\n")
            f.write("  The age OR attenuated by {:.1f}%, suggesting that some\n".format(age_atten))
            f.write("  of the 'age effect' was really a 'sickness effect.'\n\n")
            f.write("  -> Tier 3 should still be performed, but the causal\n")
            f.write("     argument is weaker.\n")

        f.write(f"\n\n--- PRESENTATION TALKING POINT ---\n\n")
        f.write(f"  'Adding comorbidity burden (CCI) to the demographics model\n")
        f.write(f"   changed the AUC from {auc_t1:.3f} to {auc_t2:.3f} (delta={auc_gain:+.3f}).\n")
        f.write(f"   The Age OR moved from {age_or_t1:.4f} to {age_or_t2:.4f} ({age_atten:+.1f}%),\n")
        f.write(f"   and the Gender OR from {gender_or_t1:.4f} to {gender_or_t2:.4f} ({gender_atten:+.1f}%).\n")
        if cci_or > 1 and model_t2.pvalues['cci_score'] < 0.05:
            f.write(f"   Each CCI point independently increases IHD odds by {(cci_or-1)*100:.1f}%.\n")
        lr_label = "Significant" if lr_pval < 0.05 else "Not Significant"
        f.write(f"   The LR test was {lr_label} (p={lr_pval:.2e}).'\n")

    logger.info("  Saved: tier2_attenuation_report.txt")

    # ------------------------------------------------------------------
    # 11. VERBOSE REPORT
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("       TIER 2 ANALYSIS -- COMPREHENSIVE REPORT")
    logger.info("       Clinical Baseline: Age + Gender + CCI")
    logger.info("=" * 70)

    report_path = os.path.join(results_dir, "tier2_verbose_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("       TIER 2 ANALYSIS -- COMPREHENSIVE REPORT\n")
        f.write("       Clinical Baseline: Age + Gender + CCI\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. STUDY DESIGN\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Population:   COVID-19 confirmed patients (2020-2023)\n")
        f.write(f"  Outcome:      Incident IHD (I21/I22) within 365 days\n")
        f.write(f"  G1 (Cases):   {n_g1:,}\n")
        f.write(f"  G2 (Controls):{n_g2:,}\n")
        f.write(f"  Total:        {len(covid):,}\n")
        f.write(f"  CCI components used: {len(cci_available)}/{len(CCI_WEIGHTS)}\n\n")

        f.write("2. CCI PROFILE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Group':<20} {'Mean CCI':>10} {'Median':>8} {'CCI=0 %':>10}\n")
        f.write(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*10}\n")
        f.write(f"  {'G1 (IHD)':<20} {g1['cci_score'].mean():>10.2f} {g1['cci_score'].median():>8.0f} {(g1['cci_score']==0).mean()*100:>9.1f}%\n")
        f.write(f"  {'G2 (No-IHD)':<20} {g2['cci_score'].mean():>10.2f} {g2['cci_score'].median():>8.0f} {(g2['cci_score']==0).mean()*100:>9.1f}%\n")
        f.write(f"  SMD: {smd_cci:.3f}, p={p_cci:.2e}\n\n")

        f.write("3. TIER 2 REGRESSION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model: logit(P(IHD)) = b0 + b1(Age) + b2(Male) + b3(CCI)\n\n")
        f.write(f"  {'Variable':<20} {'OR':>8} {'95% CI':>20} {'p-value':>12}\n")
        f.write(f"  {'-'*20} {'-'*8} {'-'*20} {'-'*12}\n")
        for _, row in or_df.iterrows():
            f.write(f"  {row['Variable']:<20} {row['OR']:>8.4f} "
                    f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f}) {row['p_value']:>12.2e} {row['Significant']}\n")

        f.write(f"\n4. MODEL COMPARISON (Tier 1 vs Tier 2)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Metric':<20} {'Tier 1':>12} {'Tier 2':>12} {'Delta':>10}\n")
        f.write(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}\n")
        f.write(f"  {'AUC':<20} {auc_t1:>12.4f} {auc_t2:>12.4f} {auc_t2-auc_t1:>+10.4f}\n")
        f.write(f"  {'R2':<20} {r2_t1:>12.6f} {r2_t2:>12.6f} {r2_t2-r2_t1:>+10.6f}\n")
        f.write(f"  {'AIC':<20} {aic_t1:>12.2f} {aic_t2:>12.2f} {aic_t2-aic_t1:>+10.2f}\n")
        f.write(f"  {'Age OR':<20} {age_or_t1:>12.4f} {age_or_t2:>12.4f} {age_atten:>+9.1f}%\n")
        f.write(f"  {'Gender OR':<20} {gender_or_t1:>12.4f} {gender_or_t2:>12.4f} {gender_atten:>+9.1f}%\n")
        f.write(f"  LR test: chi2={lr_stat:.2f}, p={lr_pval:.2e}\n\n")

        f.write(f"5. CONCLUSION\n")
        f.write("-" * 40 + "\n")

        if auc_gain < 0.01:
            f.write("  CCI adds NEGLIGIBLE discriminative power.\n")
            f.write("  Pre-existing sickness does NOT explain the excess.\n")
            f.write("  -> Supports COVID-specific cardiovascular mechanism.\n")
        elif auc_gain < 0.02:
            f.write("  CCI adds MODEST discriminative power.\n")
            f.write("  Some confounding from baseline sickness, but small.\n")
        else:
            f.write("  CCI adds SUBSTANTIAL discriminative power.\n")
            f.write("  Baseline sickness is an important confounder.\n")
        f.write(f"  -> PROCEED to Tier 3 (era/vaccination stratification).\n")

        f.write(f"\n\n" + "=" * 70 + "\n")
        f.write("Generated Plots:\n")
        for p in ["tier2_cci_distribution.png", "tier2_cci_prevalence.png",
                   "tier2_forest_plot.png", "tier2_forest_comparison.png",
                   "tier2_predicted_prob.png", "tier2_roc_curve.png",
                   "tier2_calibration_plot.png"]:
            f.write(f"  - {p}\n")
        f.write("=" * 70 + "\n")

    logger.info(f"  Saved: tier2_verbose_report.txt")

    # ------------------------------------------------------------------
    # 12. VARIANCE DECOMPOSITION (Partial R-squared)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SECTION H: Variance Decomposition (Drop-One Partial R2)")
    logger.info("=" * 70)

    r2_full = model_t2.prsquared

    # Drop each predictor and refit
    model_no_age = smf.logit('outcome ~ gender_male + cci_score', data=covid).fit(method='bfgs', disp=0, maxiter=200)
    model_no_gender = smf.logit('outcome ~ age + cci_score', data=covid).fit(method='bfgs', disp=0, maxiter=200)
    model_no_cci = model_t1  # Already fitted (Age + Gender only)

    r2_no_age = model_no_age.prsquared
    r2_no_gender = model_no_gender.prsquared
    r2_no_cci = model_no_cci.prsquared

    partial_age = r2_full - r2_no_age
    partial_gender = r2_full - r2_no_gender
    partial_cci = r2_full - r2_no_cci

    total_partial = partial_age + partial_gender + partial_cci
    pct_age = partial_age / r2_full * 100 if r2_full > 0 else 0
    pct_gender = partial_gender / r2_full * 100 if r2_full > 0 else 0
    pct_cci = partial_cci / r2_full * 100 if r2_full > 0 else 0

    logger.info(f"  Full Model R2:        {r2_full:.6f}")
    logger.info(f"  R2 without Age:       {r2_no_age:.6f}  -> partial Age   = {partial_age:.6f} ({pct_age:.1f}%)")
    logger.info(f"  R2 without Gender:    {r2_no_gender:.6f}  -> partial Gender= {partial_gender:.6f} ({pct_gender:.1f}%)")
    logger.info(f"  R2 without CCI:       {r2_no_cci:.6f}  -> partial CCI   = {partial_cci:.6f} ({pct_cci:.1f}%)")

    # Save report
    decomp_path = os.path.join(results_dir, "tier2_variance_decomposition.txt")
    with open(decomp_path, "w") as f:
        f.write("TIER 2: VARIANCE DECOMPOSITION (Partial R2)\n")
        f.write("=" * 55 + "\n\n")
        f.write("Method: Drop-one McFadden pseudo-R2 comparison.\n")
        f.write("Each predictor's partial R2 = R2_full - R2_without_predictor.\n\n")
        f.write(f"  {'Predictor':<20} {'Partial R2':>12} {'% of Full R2':>14} {'% of Total':>12}\n")
        f.write(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*12}\n")
        prop_age = partial_age / total_partial * 100 if total_partial > 0 else 0
        prop_gender = partial_gender / total_partial * 100 if total_partial > 0 else 0
        prop_cci = partial_cci / total_partial * 100 if total_partial > 0 else 0
        f.write(f"  {'Age':<20} {partial_age:>12.6f} {pct_age:>13.1f}% {prop_age:>11.1f}%\n")
        f.write(f"  {'Gender (Male)':<20} {partial_gender:>12.6f} {pct_gender:>13.1f}% {prop_gender:>11.1f}%\n")
        f.write(f"  {'CCI Score':<20} {partial_cci:>12.6f} {pct_cci:>13.1f}% {prop_cci:>11.1f}%\n")
        f.write(f"\n  Full model R2:       {r2_full:.6f}\n")
        f.write(f"  Sum of partials:     {total_partial:.6f}\n\n")
        f.write("NOTE: Partial R2 values may not sum to full R2 due to shared\n")
        f.write("variance between correlated predictors (e.g. Age ~ CCI r=0.50).\n")
        f.write("'% of Full R2' = partial / R2_full. '% of Total' = partial / sum(partials).\n")
    logger.info(f"  Saved: tier2_variance_decomposition.txt")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ['Age', 'Gender (Male)', 'CCI Score'],
        [partial_age, partial_gender, partial_cci],
        color=['#2980b9', '#e74c3c', '#27ae60'],
        alpha=0.8, edgecolor='white', linewidth=1.5
    )
    for bar, pct in zip(bars, [pct_age, pct_gender, pct_cci]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Partial McFadden R-squared', fontsize=11)
    ax.set_title('Tier 2: Variance Decomposition\n(Contribution of each predictor)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier2_variance_decomposition.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier2_variance_decomposition.png")

    # ------------------------------------------------------------------
    # 13. INDIVIDUAL CCI COMPONENT MODEL
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SECTION I: Individual CCI Component Model")
    logger.info("  logit(IHD) = Age + Gender + each CCI comorbidity separately")
    logger.info("=" * 70)

    # Build formula with individual binary flags
    cci_flag_cols = [f"cci_{c}" for c in cci_available if f"cci_{c}" in covid.columns]
    if len(cci_flag_cols) >= 2:
        formula_indiv = 'outcome ~ age + gender_male + ' + ' + '.join(cci_flag_cols)
        logger.info(f"  Formula has {len(cci_flag_cols)} CCI components + Age + Gender")

        try:
            model_indiv = smf.logit(formula_indiv, data=covid).fit(method='bfgs', disp=0, maxiter=300)

            covid['pred_indiv'] = model_indiv.predict(covid)
            auc_indiv = roc_auc_score(covid['outcome'], covid['pred_indiv'])
            r2_indiv = model_indiv.prsquared

            logger.info(f"  Converged: {model_indiv.mle_retvals.get('converged', 'N/A')}")
            logger.info(f"  AUC: {auc_indiv:.4f} (vs Tier 2 composite: {auc_t2:.4f})")
            logger.info(f"  R2:  {r2_indiv:.6f} (vs Tier 2 composite: {r2_t2:.6f})")
            logger.info(f"  AIC: {model_indiv.aic:.2f}")

            # Extract ORs for all terms
            indiv_params = model_indiv.params
            indiv_conf = model_indiv.conf_int()
            indiv_pvals = model_indiv.pvalues

            indiv_or_df = pd.DataFrame({
                'Variable': indiv_params.index,
                'Coefficient': indiv_params.values,
                'OR': np.exp(indiv_params.values),
                'Lower_CI': np.exp(indiv_conf[0].values),
                'Upper_CI': np.exp(indiv_conf[1].values),
                'p_value': indiv_pvals.values,
                'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in indiv_pvals.values]
            })

            indiv_or_df.to_csv(os.path.join(results_dir, "tier2b_individual_cci_odds_ratios.csv"), index=False)
            logger.info("  Saved: tier2b_individual_cci_odds_ratios.csv")

            # Log each CCI component OR
            logger.info(f"\n  {'Variable':<35} {'OR':>8} {'95% CI':>20} {'p-value':>12}")
            logger.info(f"  {'-'*35} {'-'*8} {'-'*20} {'-'*12}")
            for _, row in indiv_or_df.iterrows():
                if row['Variable'] == 'Intercept':
                    continue
                logger.info(f"  {row['Variable']:<35} {row['OR']:>8.4f} "
                            f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f}) "
                            f"{row['p_value']:>12.2e} {row['Significant']}")

            # Save full summary
            with open(os.path.join(results_dir, "tier2b_individual_cci_summary.txt"), "w") as f:
                f.write("INDIVIDUAL CCI COMPONENT MODEL\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Model: logit(IHD) = Age + Gender + {len(cci_flag_cols)} CCI components\n")
                f.write(f"AUC: {auc_indiv:.4f} | R2: {r2_indiv:.6f} | AIC: {model_indiv.aic:.2f}\n\n")
                f.write(model_indiv.summary().as_text())
            logger.info("  Saved: tier2b_individual_cci_summary.txt")

            # Forest plot -- CCI components only (exclude Age, Gender, Intercept)
            cci_or_plot = indiv_or_df[indiv_or_df['Variable'].str.startswith('cci_')].copy()
            cci_or_plot = cci_or_plot.sort_values('OR', ascending=True).reset_index(drop=True)
            # Clean labels
            cci_or_plot['Label'] = cci_or_plot['Variable'].str.replace('cci_', '').str.replace('_', ' ')

            if len(cci_or_plot) > 0:
                fig, ax = plt.subplots(figsize=(10, max(4, len(cci_or_plot) * 0.45 + 1.5)))
                y_pos = np.arange(len(cci_or_plot))

                # Color by significance
                colors = ['#2c3e50' if p < 0.05 else '#bdc3c7' for p in cci_or_plot['p_value']]

                ax.errorbar(
                    cci_or_plot['OR'], y_pos,
                    xerr=[cci_or_plot['OR'] - cci_or_plot['Lower_CI'],
                          cci_or_plot['Upper_CI'] - cci_or_plot['OR']],
                    fmt='none', ecolor='#7f8c8d', elinewidth=1.5, capsize=3
                )
                ax.scatter(cci_or_plot['OR'], y_pos, c=colors, s=60, zorder=3, edgecolor='white')
                ax.axvline(x=1.0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(cci_or_plot['Label'], fontsize=9)

                for i, row in cci_or_plot.iterrows():
                    idx = cci_or_plot.index.get_loc(i)
                    sig = row['Significant']
                    ax.text(max(cci_or_plot['Upper_CI']) * 1.08, idx,
                            f"{row['OR']:.2f} ({row['Lower_CI']:.2f}-{row['Upper_CI']:.2f}) {sig}",
                            va='center', fontsize=7.5, color='#2c3e50')

                ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
                ax.set_title('Individual CCI Components: Adjusted ORs for IHD\n(Each adjusted for Age + Gender + all other components)',
                             fontsize=12, fontweight='bold')
                ax.set_xscale('log')
                ax.grid(axis='x', linestyle=':', alpha=0.4)
                ax.invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, "tier2b_individual_cci_forest.png"), dpi=150, bbox_inches='tight')
                plt.close()
                logger.info("  Saved: tier2b_individual_cci_forest.png")

        except Exception as e:
            logger.warning(f"  Individual CCI model failed: {e}")
            logger.warning("  This may be due to sparse categories or perfect separation.")
    else:
        logger.warning("  Fewer than 2 CCI flag columns available. Skipping individual model.")

    # ------------------------------------------------------------------
    # 14. G3 COMPARISON (IHD without COVID)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SECTION J: G3 Comparison (IHD without COVID)")
    logger.info("=" * 70)

    g3 = df_all[df_all['group'] == 'Group 3'].copy()
    n_g3 = len(g3)
    logger.info(f"  G3 (IHD, no COVID): {n_g3:,} patients")

    if n_g3 > 0:
        # Prepare G3 features
        g3['age'] = pd.to_numeric(g3['age'], errors='coerce')
        g3['gender_male'] = g3['gender'].astype(str).str.strip().map(gender_map)
        g3 = g3.dropna(subset=['age', 'gender_male']).copy()
        g3['gender_male'] = g3['gender_male'].astype(int)

        # Compute CCI for G3
        g3_index_date = pd.to_datetime(g3['discharge_date'], errors='coerce')
        g3['cci_score_computed'] = 0
        for condition, weight in CCI_WEIGHTS.items():
            col_cci = f"Comorb_CCI_{condition}_Date"
            col_old = f"Comorb_{condition}_Date"
            col = col_cci if col_cci in g3.columns else (col_old if col_old in g3.columns else None)
            if col is not None:
                cdt = pd.to_datetime(g3[col], errors='coerce')
                flag = ((cdt.notnull()) & (cdt <= g3_index_date)).astype(int)
                g3[f"cci_{condition}"] = flag
                g3['cci_score_computed'] += flag * weight

        # Always use recomputed CCI (excludes MI, CHF, AIDS_HIV)
        g3['cci_score'] = g3['cci_score_computed']

        # --- 14a. CCI Profile: G1 vs G2 vs G3 ---
        logger.info("\n  --- CCI Profile: G1 vs G2 vs G3 ---")

        groups_data = {
            'G1 (COVID->IHD)': g1,
            'G2 (COVID->No IHD)': g2,
            'G3 (No COVID->IHD)': g3,
        }
        logger.info(f"  {'Group':<25} {'N':>8} {'Mean CCI':>10} {'Median':>8} {'CCI=0 %':>10}")
        for gname, gdf in groups_data.items():
            logger.info(f"  {gname:<25} {len(gdf):>8,} {gdf['cci_score'].mean():>10.2f} "
                        f"{gdf['cci_score'].median():>8.0f} {(gdf['cci_score']==0).mean()*100:>9.1f}%")

        # SMDs: G1 vs G3
        smd_g1_g3 = compute_smd(g1['cci_score'], g3['cci_score'])
        _, p_g1_g3 = stats.mannwhitneyu(g1['cci_score'], g3['cci_score'], alternative='two-sided')
        logger.info(f"\n  G1 vs G3 CCI: SMD={smd_g1_g3:.3f}, p={p_g1_g3:.2e}")

        smd_g1_g2_age = compute_smd(g1['age'], g2['age'])
        smd_g1_g3_age = compute_smd(g1['age'], g3['age'])
        logger.info(f"  G1 vs G2 Age SMD: {smd_g1_g2_age:.3f}")
        logger.info(f"  G1 vs G3 Age SMD: {smd_g1_g3_age:.3f}")

        # --- 14b. 3-Group CCI Distribution Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        max_cci_all = int(min(max(g1['cci_score'].max(), g2['cci_score'].max(), g3['cci_score'].max()), 15))
        bins_all = np.arange(-0.5, max_cci_all + 1.5, 1)

        for gdf, label, color, alpha in [
            (g2, f'G2: No-IHD (N={len(g2):,})', '#3498db', 0.4),
            (g3, f'G3: IHD no-COVID (N={len(g3):,})', '#27ae60', 0.5),
            (g1, f'G1: COVID->IHD (N={len(g1):,})', '#e74c3c', 0.6),
        ]:
            axes[0].hist(gdf['cci_score'], bins=bins_all, alpha=alpha, color=color,
                         label=label, density=True)
        axes[0].set_xlabel('CCI Score', fontsize=11)
        axes[0].set_ylabel('Density', fontsize=11)
        axes[0].set_title('CCI Distribution: All 3 Groups', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=8)

        # Box plot
        bp_data = [g2['cci_score'], g3['cci_score'], g1['cci_score']]
        bp_labels = ['G2: No-IHD', 'G3: IHD\nno-COVID', 'G1: COVID\n->IHD']
        bp_colors = ['#3498db', '#27ae60', '#e74c3c']
        bp = axes[1].boxplot(bp_data, positions=[1, 2, 3], widths=0.5,
                             patch_artist=True, medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], bp_colors):
            patch.set(facecolor=color, alpha=0.5)
        axes[1].set_xticks([1, 2, 3])
        axes[1].set_xticklabels(bp_labels, fontsize=9)
        axes[1].set_ylabel('CCI Score', fontsize=11)
        axes[1].set_title('CCI by Group', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', linestyle=':', alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "tier2c_g3_cci_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: tier2c_g3_cci_comparison.png")

        # --- 14c. 3-Group Comorbidity Prevalence ---
        conditions_for_g3 = [c for c in cci_available if f"cci_{c}" in g3.columns]
        if conditions_for_g3:
            g1_prev3 = [g1[f"cci_{c}"].mean() * 100 for c in conditions_for_g3]
            g2_prev3 = [g2[f"cci_{c}"].mean() * 100 for c in conditions_for_g3]
            g3_prev3 = [g3[f"cci_{c}"].mean() * 100 if f"cci_{c}" in g3.columns else 0 for c in conditions_for_g3]
            labels3 = [c.replace('_', ' ') for c in conditions_for_g3]

            fig, ax = plt.subplots(figsize=(14, max(6, len(conditions_for_g3) * 0.5)))
            y = np.arange(len(labels3))
            h = 0.25

            ax.barh(y - h, g1_prev3, h, label=f'G1: COVID->IHD (N={len(g1):,})', color='#e74c3c', alpha=0.7)
            ax.barh(y, g3_prev3, h, label=f'G3: IHD no-COVID (N={len(g3):,})', color='#27ae60', alpha=0.7)
            ax.barh(y + h, g2_prev3, h, label=f'G2: No-IHD (N={len(g2):,})', color='#3498db', alpha=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(labels3, fontsize=9)
            ax.set_xlabel('Prevalence (%)', fontsize=11)
            ax.set_title('Pre-Index Comorbidity Prevalence: G1 vs G3 vs G2', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='x', linestyle=':', alpha=0.4)
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "tier2c_g3_prevalence.png"), dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("  Saved: tier2c_g3_prevalence.png")

        # --- 14d. COVID Exposure Model (G1 + G3: IHD patients only) ---
        logger.info("\n  --- COVID Exposure Model (G1 vs G3) ---")
        logger.info("  Among IHD patients: does COVID exposure differ by CCI?")

        ihd_pool = pd.concat([
            g1[['age', 'gender_male', 'cci_score']].assign(covid_exposed=1),
            g3[['age', 'gender_male', 'cci_score']].assign(covid_exposed=1 if False else 0),
        ], ignore_index=True)
        # Fix: Set covid_exposed properly
        ihd_pool.loc[:len(g1)-1, 'covid_exposed'] = 1
        ihd_pool.loc[len(g1):, 'covid_exposed'] = 0

        ihd_pool = ihd_pool.dropna()
        logger.info(f"  IHD pool: {len(ihd_pool):,} (G1={len(g1):,}, G3={len(g3):,})")

        try:
            model_covid = smf.logit('covid_exposed ~ age + gender_male + cci_score', data=ihd_pool).fit(
                method='bfgs', disp=0, maxiter=200
            )
            covid_or_df = pd.DataFrame({
                'Variable': model_covid.params.index,
                'OR': np.exp(model_covid.params.values),
                'Lower_CI': np.exp(model_covid.conf_int()[0].values),
                'Upper_CI': np.exp(model_covid.conf_int()[1].values),
                'p_value': model_covid.pvalues.values,
            })

            logger.info(f"\n  COVID Exposure Model (among IHD patients):")
            logger.info(f"  {'Variable':<20} {'OR':>8} {'95% CI':>20} {'p-value':>12}")
            for _, row in covid_or_df.iterrows():
                logger.info(f"  {row['Variable']:<20} {row['OR']:>8.4f} "
                            f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f}) {row['p_value']:>12.2e}")

            covid_auc = roc_auc_score(ihd_pool['covid_exposed'], model_covid.predict(ihd_pool))
            logger.info(f"  AUC: {covid_auc:.4f}")

            cci_p_covid = model_covid.pvalues.get('cci_score', 1.0)
            cci_or_covid = np.exp(model_covid.params.get('cci_score', 0))

            # Interpretation
            if cci_p_covid >= 0.05:
                interp = ("CCI is NOT a significant predictor of COVID exposure among IHD patients.\n"
                          "  This means COVID-IHD and non-COVID-IHD patients have SIMILAR\n"
                          "  baseline sickness -- strengthening the COVID-pathway argument.")
            elif cci_or_covid < 1:
                interp = ("CCI is significant but INVERSELY associated with COVID exposure.\n"
                          "  COVID-IHD patients are LESS sick at baseline than non-COVID-IHD,\n"
                          "  suggesting COVID may cause IHD in otherwise healthier patients.")
            else:
                interp = ("CCI is significant and POSITIVELY associated with COVID exposure.\n"
                          "  COVID-IHD patients are SICKER at baseline, suggesting some of\n"
                          "  the excess risk is driven by vulnerable patients catching COVID.")
            logger.info(f"\n  INTERPRETATION: {interp}")

        except Exception as e:
            logger.warning(f"  COVID exposure model failed: {e}")
            interp = "Model could not be fitted."
            covid_or_df = pd.DataFrame()
            covid_auc = None

        # --- Save G3 Comparison Report ---
        g3_report_path = os.path.join(results_dir, "tier2c_g3_comparison.txt")
        with open(g3_report_path, "w") as f:
            f.write("G3 COMPARISON REPORT\n")
            f.write("=" * 55 + "\n\n")
            f.write("G1 = COVID -> IHD (post-COVID cardiac events)\n")
            f.write("G2 = COVID -> No IHD (COVID without cardiac events)\n")
            f.write("G3 = IHD without COVID (cardiac events, no COVID)\n\n")

            f.write("--- CCI Profile ---\n\n")
            f.write(f"  {'Group':<25} {'N':>8} {'Mean CCI':>10} {'Median':>8} {'CCI=0%':>8}\n")
            f.write(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8}\n")
            for gname, gdf in groups_data.items():
                f.write(f"  {gname:<25} {len(gdf):>8,} {gdf['cci_score'].mean():>10.2f} "
                        f"{gdf['cci_score'].median():>8.0f} {(gdf['cci_score']==0).mean()*100:>7.1f}%\n")

            f.write(f"\n  G1 vs G3 CCI SMD: {smd_g1_g3:.3f}, p={p_g1_g3:.2e}\n")
            f.write(f"  G1 vs G2 Age SMD: {smd_g1_g2_age:.3f}\n")
            f.write(f"  G1 vs G3 Age SMD: {smd_g1_g3_age:.3f}\n\n")

            f.write("--- COVID Exposure Model (G1 + G3, IHD patients only) ---\n\n")
            f.write("  Model: logit(P(COVID)) = Age + Gender + CCI\n")
            f.write("  Question: Are COVID-IHD patients different from non-COVID-IHD?\n\n")
            if len(covid_or_df) > 0:
                for _, row in covid_or_df.iterrows():
                    f.write(f"  {row['Variable']:<20} OR={row['OR']:.4f} "
                            f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f}) p={row['p_value']:.2e}\n")
                if covid_auc is not None:
                    f.write(f"\n  AUC: {covid_auc:.4f}\n")
            f.write(f"\n  INTERPRETATION:\n  {interp}\n")

        logger.info(f"  Saved: tier2c_g3_comparison.txt")

    else:
        logger.warning("  No G3 patients found. Skipping G3 comparison.")

    # ------------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("TIER 2 ANALYSIS COMPLETE (with enhancements)")
    logger.info("=" * 70)

    return {
        'model_t1': model_t1, 'model_t2': model_t2,
        'auc_t1': auc_t1, 'auc_t2': auc_t2,
        'r2_t1': r2_t1, 'r2_t2': r2_t2,
        'aic_t1': aic_t1, 'aic_t2': aic_t2,
        'lr_stat': lr_stat, 'lr_pval': lr_pval,
        'cci_or': cci_or,
        'age_atten': age_atten, 'gender_atten': gender_atten,
        'partial_r2_age': partial_age, 'partial_r2_gender': partial_gender, 'partial_r2_cci': partial_cci,
    }


# ==============================================================================
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    results = run_step_9(conf)
    if results:
        print(f"\nTier 2 complete. AUC: {results['auc_t1']:.4f} -> {results['auc_t2']:.4f}")
        print(f"CCI OR: {results['cci_or']:.4f}")
        print(f"LR test p={results['lr_pval']:.2e}")
