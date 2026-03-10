"""
6_tier_1_analysis.py
====================
Tier 1 of the Stepwise DAG-Aligned Analysis:
    logit(P(IHD=1)) = β₀ + β₁(Age) + β₂(Gender_Male)

Purpose:
    Determine whether the observed ~85% excess IHD risk (SIR=1.85) in
    the COVID cohort is explained purely by demographic confounders
    (age and sex distribution differences between Group 1 and Group 2).

Population:
    - Group 1 (Post-COVID IHD): N ≈ 1,870
    - Group 2 (COVID No-IHD):   N ≈ 483,981

Outputs (to data/03_results/step_6_tier1/):
    1. tier1_model_summary.txt        — Full statsmodels logistic summary
    2. tier1_odds_ratios.csv          — Odds ratios with 95% CI
    3. tier1_attenuation_report.txt   — Comparison to crude SIR: attenuation %
    4. tier1_forest_plot.png          — Forest plot of adjusted ORs
    5. tier1_age_distribution.png     — Age distribution comparison G1 vs G2
    6. tier1_gender_distribution.png  — Gender bar chart comparison
    7. tier1_predicted_prob.png       — Predicted probability curve by age & sex
    8. tier1_roc_curve.png            — ROC curve with AUC
    9. tier1_calibration_plot.png     — Calibration (observed vs predicted)
    10. tier1_diagnostic_report.txt   — Model diagnostics summary
    11. tier1_verbose_report.txt      — Human-readable narrative report
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

# --- Local imports ------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger, report_df_info, save_with_report, ensure_dir


# ==============================================================================
# CONFIGURABLE AGE CUTOFF
# ==============================================================================
# Set AGE_CUTOFF to an integer to filter the cohort by age.
#   - AGE_DIRECTION = "older"   → keep patients >= AGE_CUTOFF  (e.g. 35 and older)
#   - AGE_DIRECTION = "younger" → keep patients <  AGE_CUTOFF  (e.g. younger than 35)
# Set AGE_CUTOFF = None to disable age filtering entirely.
AGE_CUTOFF = 35           # Current default: 35 years
AGE_DIRECTION = "older"   # "older" or "younger"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_smd(group1_vals, group2_vals):
    """Standardised Mean Difference (Cohen's d) between two groups."""
    m1, m2 = group1_vals.mean(), group2_vals.mean()
    s1, s2 = group1_vals.std(), group2_vals.std()
    pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
    if pooled_sd == 0:
        return 0.0
    return (m1 - m2) / pooled_sd


def make_forest_plot(odds_df, save_path, title="Tier 1: Forest Plot of Adjusted Odds Ratios"):
    """Generate a publication-quality forest plot from a dataframe of ORs.
    
    Expected columns: Variable, OR, Lower_CI, Upper_CI.
    """
    fig, ax = plt.subplots(figsize=(8, max(3, len(odds_df) * 0.8 + 1.5)))
    
    y_pos = np.arange(len(odds_df))
    
    # Plot points and CIs
    ax.errorbar(
        odds_df['OR'], y_pos,
        xerr=[odds_df['OR'] - odds_df['Lower_CI'],
              odds_df['Upper_CI'] - odds_df['OR']],
        fmt='o', color='#2c3e50', ecolor='#7f8c8d',
        elinewidth=2, capsize=4, markersize=8, zorder=3
    )
    
    # Reference line at OR=1
    ax.axvline(x=1.0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='OR = 1.0 (null)')
    
    # Labels — variable name on left, OR (CI) on right
    ax.set_yticks(y_pos)
    ax.set_yticklabels(odds_df['Variable'], fontsize=11)
    
    for i, row in odds_df.iterrows():
        ax.text(
            max(odds_df['Upper_CI']) * 1.08, y_pos[i],
            f"{row['OR']:.2f} ({row['Lower_CI']:.2f}–{row['Upper_CI']:.2f})",
            va='center', fontsize=9, color='#2c3e50'
        )
    
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def run_tier_1(config):
    # --------------------------------------------------------------------------
    # 1. SETUP
    # --------------------------------------------------------------------------
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_6_tier1")
    ensure_dir(results_dir)
    
    logger = setup_logger("tier_1", results_dir)
    logger.info("=" * 70)
    logger.info("TIER 1 ANALYSIS: Demographic Adjustment (Age + Gender)")
    logger.info("Model: logit(P(IHD=1)) = β₀ + β₁(Age) + β₂(Gender_Male)")
    logger.info("=" * 70)
    
    # --------------------------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------------------------
    input_file = os.path.join(processed_dir, "step_3_features", "cohort_enriched.csv")
    if not os.path.exists(input_file):
        logger.error(f"Missing input: {input_file}. Run Step 3 first.")
        return None
    
    df_all = pd.read_csv(input_file)
    logger.info(f"Loaded full cohort: {len(df_all):,} patients")
    logger.info(f"Groups: {df_all['group'].value_counts().to_dict()}")
    
    # --------------------------------------------------------------------------
    # 3. PREPARE COVID COHORT (G1 + G2)
    # --------------------------------------------------------------------------
    covid = df_all[df_all['group'].isin(['Group 1', 'Group 2'])].copy()
    logger.info(f"COVID cohort: {len(covid):,} patients (G1 + G2)")
    
    # Outcome: 1 = IHD (Group 1), 0 = No IHD (Group 2)
    covid['outcome'] = (covid['group'] == 'Group 1').astype(int)
    
    # Age — ensure numeric, drop missing
    covid['age'] = pd.to_numeric(covid['age'], errors='coerce')
    n_missing_age = covid['age'].isna().sum()
    if n_missing_age > 0:
        logger.warning(f"Dropping {n_missing_age} patients with missing age ({n_missing_age/len(covid)*100:.2f}%)")
    
    # Gender — standardise to binary, handle edge cases
    gender_map = {'M': 1, 'Male': 1, 'MALE': 1, 'm': 1, '1': 1,
                  'F': 0, 'Female': 0, 'FEMALE': 0, 'f': 0, '0': 0}
    covid['gender_str'] = covid['gender'].astype(str).str.strip()
    covid['gender_male'] = covid['gender_str'].map(gender_map)
    n_missing_gender = covid['gender_male'].isna().sum()
    if n_missing_gender > 0:
        logger.warning(f"Dropping {n_missing_gender} patients with unmappable gender ({n_missing_gender/len(covid)*100:.2f}%)")
    
    # Drop incomplete records
    covid = covid.dropna(subset=['age', 'gender_male']).copy()
    covid['gender_male'] = covid['gender_male'].astype(int)
    
    # -- Apply age cutoff filter --
    if AGE_CUTOFF is not None:
        n_before_age_filter = len(covid)
        if AGE_DIRECTION == "older":
            covid = covid[covid['age'] >= AGE_CUTOFF].copy()
            filter_desc = f"age >= {AGE_CUTOFF}"
        elif AGE_DIRECTION == "younger":
            covid = covid[covid['age'] < AGE_CUTOFF].copy()
            filter_desc = f"age < {AGE_CUTOFF}"
        else:
            raise ValueError(f"Invalid AGE_DIRECTION '{AGE_DIRECTION}'. Must be 'older' or 'younger'.")
        n_dropped_age_filter = n_before_age_filter - len(covid)
        logger.info(f"Age cutoff applied: keeping {filter_desc} → dropped {n_dropped_age_filter:,} patients ({n_dropped_age_filter/n_before_age_filter*100:.1f}%)")
    else:
        logger.info("No age cutoff applied (AGE_CUTOFF is None)")
    
    n_g1 = covid['outcome'].sum()
    n_g2 = len(covid) - n_g1
    logger.info(f"Analysis cohort after cleaning: {len(covid):,} (G1={n_g1:,}, G2={n_g2:,})")
    
    # --------------------------------------------------------------------------
    # 4. DESCRIPTIVE STATISTICS (Pre-Regression)
    # --------------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("SECTION A: Descriptive Statistics")
    logger.info("-" * 50)
    
    g1 = covid[covid['outcome'] == 1]
    g2 = covid[covid['outcome'] == 0]
    
    # -- Age Distribution --
    age_stats = {
        'Group 1 (COVID->IHD)': {
            'N': len(g1),
            'Mean ± SD': f"{g1['age'].mean():.1f} ± {g1['age'].std():.1f}",
            'Median (IQR)': f"{g1['age'].median():.0f} ({g1['age'].quantile(0.25):.0f}–{g1['age'].quantile(0.75):.0f})",
        },
        'Group 2 (COVID No-IHD)': {
            'N': len(g2),
            'Mean ± SD': f"{g2['age'].mean():.1f} ± {g2['age'].std():.1f}",
            'Median (IQR)': f"{g2['age'].median():.0f} ({g2['age'].quantile(0.25):.0f}–{g2['age'].quantile(0.75):.0f})",
        },
    }
    
    # Statistical test
    t_stat, p_age = stats.ttest_ind(g1['age'].dropna(), g2['age'].dropna(), equal_var=False)
    smd_age = compute_smd(g1['age'].dropna(), g2['age'].dropna())
    
    logger.info(f"  Age -- G1: {age_stats['Group 1 (COVID->IHD)']['Mean ± SD']}, "
                f"G2: {age_stats['Group 2 (COVID No-IHD)']['Mean ± SD']}")
    logger.info(f"  Welch t-test: t={t_stat:.2f}, p={p_age:.2e}")
    logger.info(f"  SMD (Cohen's d): {smd_age:.3f}")
    
    # -- Gender Distribution --
    g1_male_pct = g1['gender_male'].mean() * 100
    g2_male_pct = g2['gender_male'].mean() * 100
    _, p_gender = stats.chi2_contingency(pd.crosstab(covid['outcome'], covid['gender_male']))[:2]
    smd_gender = compute_smd(g1['gender_male'].astype(float), g2['gender_male'].astype(float))
    
    logger.info(f"  Gender (Male%) — G1: {g1_male_pct:.1f}%, G2: {g2_male_pct:.1f}%")
    logger.info(f"  Chi-squared p={p_gender:.2e}")
    logger.info(f"  SMD: {smd_gender:.3f}")
    
    # -- Plot: Age Distribution --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(g2['age'].dropna(), bins=30, alpha=0.5, color='#3498db', label=f'G2 (N={len(g2):,})', density=True)
    axes[0].hist(g1['age'].dropna(), bins=30, alpha=0.7, color='#e74c3c', label=f'G1 (N={len(g1):,})', density=True)
    axes[0].set_xlabel('Age (years)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Age Distribution: G1 vs G2', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].axvline(g1['age'].median(), color='#e74c3c', linestyle='--', alpha=0.8, label=f'G1 median={g1["age"].median():.0f}')
    axes[0].axvline(g2['age'].median(), color='#3498db', linestyle='--', alpha=0.8, label=f'G2 median={g2["age"].median():.0f}')
    axes[0].legend(fontsize=9)
    
    # Box plot — draw each group separately to allow different colors
    bp1 = axes[1].boxplot(
        [g2['age'].dropna()], positions=[1], widths=0.5,
        patch_artist=True, medianprops=dict(color='black', linewidth=2)
    )
    bp1['boxes'][0].set(facecolor='#3498db', alpha=0.4)
    
    bp2 = axes[1].boxplot(
        [g1['age'].dropna()], positions=[2], widths=0.5,
        patch_artist=True, medianprops=dict(color='black', linewidth=2)
    )
    bp2['boxes'][0].set(facecolor='#e74c3c', alpha=0.5)
    
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['G2: COVID\nNo-IHD', 'G1: COVID\n→ IHD'])
    axes[1].set_ylabel('Age (years)', fontsize=11)
    axes[1].set_title(f'Age Comparison (SMD={smd_age:.2f}, p={p_age:.1e})', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier1_age_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier1_age_distribution.png")
    
    # -- Plot: Gender Distribution --
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    
    g1_counts = [len(g1[g1['gender_male'] == 0]), len(g1[g1['gender_male'] == 1])]
    g2_counts = [len(g2[g2['gender_male'] == 0]), len(g2[g2['gender_male'] == 1])]
    
    g1_pcts = [c / len(g1) * 100 for c in g1_counts]
    g2_pcts = [c / len(g2) * 100 for c in g2_counts]
    
    bars1 = ax.bar(x - width/2, g1_pcts, width, label=f'G1: COVID->IHD (N={len(g1):,})', color='#e74c3c', alpha=0.7)
    bars2 = ax.bar(x + width/2, g2_pcts, width, label=f'G2: COVID No-IHD (N={len(g2):,})', color='#3498db', alpha=0.7)
    
    ax.set_xlabel('Gender', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title(f'Gender Distribution: G1 vs G2 (χ² p={p_gender:.1e}, SMD={smd_gender:.2f})', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Female', 'Male'])
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    # Annotate bars with counts
    for bar, count in zip(bars1, g1_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={count:,}', ha='center', va='bottom', fontsize=9)
    for bar, count in zip(bars2, g2_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier1_gender_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier1_gender_distribution.png")
    
    # --------------------------------------------------------------------------
    # 5. CRUDE (UNADJUSTED) ODDS RATIO
    # --------------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("SECTION B: Crude (Unadjusted) Incidence & Odds")
    logger.info("-" * 50)
    
    crude_or = (n_g1 / n_g2)  # Simplified: P(IHD|COVID) / P(noIHD|COVID)
    # More precisely: a/(a+b) / (1 - a/(a+b)) = a/b
    crude_risk = n_g1 / (n_g1 + n_g2)
    crude_odds = n_g1 / n_g2  # This is the odds, not OR from a model
    
    logger.info(f"  Crude 1-year IHD risk in COVID cohort: {crude_risk*100:.3f}% ({n_g1}/{n_g1+n_g2})")
    logger.info(f"  Crude odds: {crude_odds:.5f}")
    logger.info(f"  (For external reference: crude SIR from Step 4 ≈ 1.85)")
    
    # Run unadjusted model (intercept only + no predictors)
    # This gives us a formal baseline log-likelihood for the LR test
    try:
        model_null = smf.logit('outcome ~ 1', data=covid).fit(disp=0)
        ll_null = model_null.llf
        logger.info(f"  Null model log-likelihood: {ll_null:.2f}")
    except Exception as e:
        logger.warning(f"  Could not fit null model: {e}")
        ll_null = None
    
    # --------------------------------------------------------------------------
    # 6. TIER 1 LOGISTIC REGRESSION
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SECTION C: TIER 1 — Logistic Regression (Age + Gender)")
    logger.info("    logit(P(IHD=1)) = β₀ + β₁(Age) + β₂(Gender_Male)")
    logger.info("=" * 70)
    
    try:
        model_t1 = smf.logit('outcome ~ age + gender_male', data=covid).fit(disp=0)
    except Exception as e:
        logger.error(f"Tier 1 model fitting failed: {e}")
        return None
    
    logger.info(f"\nModel converged: {model_t1.mle_retvals['converged']}")
    logger.info(f"Iterations: {model_t1.mle_retvals.get('iterations', 'N/A')}")
    
    # -- Full Summary --
    summary_text = model_t1.summary().as_text()
    with open(os.path.join(results_dir, "tier1_model_summary.txt"), "w") as f:
        f.write("TIER 1 LOGISTIC REGRESSION — FULL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model: logit(P(IHD=1)) = β₀ + β₁(Age) + β₂(Gender_Male)\n")
        f.write(f"Population: COVID cohort (G1 + G2), N = {len(covid):,}\n")
        f.write(f"Events (G1): {n_g1:,}  |  Non-Events (G2): {n_g2:,}\n\n")
        f.write(summary_text)
    logger.info("  Saved: tier1_model_summary.txt")
    
    # -- Extract Odds Ratios --
    params = model_t1.params
    conf = model_t1.conf_int()
    pvalues = model_t1.pvalues
    
    or_df = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'OR': np.exp(params.values),
        'Lower_CI': np.exp(conf[0].values),
        'Upper_CI': np.exp(conf[1].values),
        'p_value': pvalues.values,
        'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in pvalues.values]
    })
    
    or_df.to_csv(os.path.join(results_dir, "tier1_odds_ratios.csv"), index=False)
    logger.info("  Saved: tier1_odds_ratios.csv")
    
    # Log key results
    for _, row in or_df.iterrows():
        logger.info(f"  {row['Variable']:20s}  OR={row['OR']:.4f}  "
                    f"(95% CI: {row['Lower_CI']:.4f}–{row['Upper_CI']:.4f})  "
                    f"p={row['p_value']:.2e} {row['Significant']}")
    
    # -- Forest Plot --
    plot_df = or_df[or_df['Variable'] != 'Intercept'].reset_index(drop=True)
    # Rename for readability
    rename_map = {'age': 'Age (per year)', 'gender_male': 'Male Sex'}
    plot_df['Variable'] = plot_df['Variable'].map(lambda x: rename_map.get(x, x))
    
    make_forest_plot(plot_df, os.path.join(results_dir, "tier1_forest_plot.png"),
                     title="Tier 1: Adjusted Odds Ratios\n(Age + Gender Only)")
    logger.info("  Saved: tier1_forest_plot.png")
    
    # --------------------------------------------------------------------------
    # 7. MODEL DIAGNOSTICS
    # --------------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("SECTION D: Model Diagnostics")
    logger.info("-" * 50)
    
    # a) Predicted probabilities
    covid['pred_prob'] = model_t1.predict(covid)
    
    # b) AUC / C-statistic
    from sklearn.metrics import roc_auc_score, roc_curve
    
    try:
        auc = roc_auc_score(covid['outcome'], covid['pred_prob'])
    except Exception:
        auc = float('nan')
    logger.info(f"  AUC (C-statistic): {auc:.4f}")
    
    # c) Pseudo R-squared
    pseudo_r2 = model_t1.prsquared
    logger.info(f"  McFadden Pseudo R²: {pseudo_r2:.6f}")
    
    # d) Likelihood Ratio Test vs Null
    if ll_null is not None:
        lr_stat = -2 * (ll_null - model_t1.llf)
        lr_pval = stats.chi2.sf(lr_stat, df=2)  # 2 predictors
        logger.info(f"  LR Test vs Null: χ²={lr_stat:.2f}, df=2, p={lr_pval:.2e}")
    else:
        lr_stat, lr_pval = None, None
    
    # e) AIC / BIC
    logger.info(f"  AIC: {model_t1.aic:.2f}")
    logger.info(f"  BIC: {model_t1.bic:.2f}")
    
    # -- ROC Curve --
    try:
        fpr, tpr, _ = roc_curve(covid['outcome'], covid['pred_prob'])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='#2c3e50', linewidth=2, label=f'Tier 1 Model (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#bdc3c7', linestyle='--', linewidth=1, label='Random (AUC = 0.5)')
        ax.fill_between(fpr, tpr, alpha=0.1, color='#2c3e50')
        ax.set_xlabel('1 − Specificity (FPR)', fontsize=11)
        ax.set_ylabel('Sensitivity (TPR)', fontsize=11)
        ax.set_title(f'Tier 1: ROC Curve (AUC = {auc:.3f})', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "tier1_roc_curve.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: tier1_roc_curve.png")
    except Exception as e:
        logger.warning(f"  ROC curve generation failed: {e}")
    
    # -- Predicted Probability by Age & Sex --
    age_range = np.arange(20, 95, 1)
    pred_male = model_t1.predict(pd.DataFrame({'age': age_range, 'gender_male': 1}))
    pred_female = model_t1.predict(pd.DataFrame({'age': age_range, 'gender_male': 0}))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(age_range, pred_male * 100, color='#2980b9', linewidth=2, label='Male')
    ax.plot(age_range, pred_female * 100, color='#e74c3c', linewidth=2, label='Female')
    ax.fill_between(age_range, pred_male * 100, pred_female * 100, alpha=0.1, color='#7f8c8d')
    ax.set_xlabel('Age (years)', fontsize=11)
    ax.set_ylabel('Predicted 1-Year IHD Risk (%)', fontsize=11)
    ax.set_title('Tier 1: Predicted IHD Probability by Age and Sex\n(Demographics-Only Model)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(linestyle=':', alpha=0.4)
    ax.set_xlim(20, 95)
    
    # Annotate key ages
    for key_age in [40, 60, 80]:
        male_pred = model_t1.predict(pd.DataFrame({'age': [key_age], 'gender_male': [1]}))[0] * 100
        female_pred = model_t1.predict(pd.DataFrame({'age': [key_age], 'gender_male': [0]}))[0] * 100
        ax.annotate(f'{male_pred:.2f}%', xy=(key_age, male_pred),
                    xytext=(key_age + 3, male_pred + 0.3),
                    fontsize=8, color='#2980b9', arrowprops=dict(arrowstyle='-', color='#2980b9', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier1_predicted_prob.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier1_predicted_prob.png")
    
    # -- Observed IHD Rate by 5-Year Age Buckets & Sex --
    logger.info("  Generating observed IHD rate by 5-year age buckets...")
    
    age_min = int(np.floor(covid['age'].min() / 5) * 5)
    age_max = int(np.ceil(covid['age'].max() / 5) * 5) + 5
    age_bins = np.arange(age_min, age_max, 5)
    bin_labels = [f"{b}-{b+4}" for b in age_bins[:-1]]
    
    covid['age_bin'] = pd.cut(covid['age'], bins=age_bins, right=False, labels=bin_labels)
    
    # Compute observed rates by age bin and sex
    rate_table = covid.groupby(['age_bin', 'gender_male'], observed=True).agg(
        n_ihd=('outcome', 'sum'),
        n_total=('outcome', 'count')
    ).reset_index()
    rate_table['rate_pct'] = rate_table['n_ihd'] / rate_table['n_total'] * 100
    
    male_rates = rate_table[rate_table['gender_male'] == 1].set_index('age_bin')
    female_rates = rate_table[rate_table['gender_male'] == 0].set_index('age_bin')
    
    # Log the table
    logger.info("  Observed IHD rate (%) by 5-year age bucket:")
    logger.info(f"    {'Age Bin':<10} {'Male %':>8} {'(n/N)':>14} {'Female %':>10} {'(n/N)':>14}")
    for lbl in bin_labels:
        m_row = male_rates.loc[lbl] if lbl in male_rates.index else None
        f_row = female_rates.loc[lbl] if lbl in female_rates.index else None
        m_str = f"{m_row['rate_pct']:>7.2f}%  ({int(m_row['n_ihd'])}/{int(m_row['n_total'])})" if m_row is not None else "     N/A"
        f_str = f"{f_row['rate_pct']:>7.2f}%  ({int(f_row['n_ihd'])}/{int(f_row['n_total'])})" if f_row is not None else "     N/A"
        logger.info(f"    {lbl:<10} {m_str}     {f_str}")
    
    # Plot: grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(bin_labels))
    width = 0.35
    
    male_vals = [male_rates.loc[lbl, 'rate_pct'] if lbl in male_rates.index else 0 for lbl in bin_labels]
    female_vals = [female_rates.loc[lbl, 'rate_pct'] if lbl in female_rates.index else 0 for lbl in bin_labels]
    male_n = [f"{int(male_rates.loc[lbl, 'n_ihd'])}/{int(male_rates.loc[lbl, 'n_total'])}" if lbl in male_rates.index else "" for lbl in bin_labels]
    female_n = [f"{int(female_rates.loc[lbl, 'n_ihd'])}/{int(female_rates.loc[lbl, 'n_total'])}" if lbl in female_rates.index else "" for lbl in bin_labels]
    
    bars_m = ax.bar(x - width/2, male_vals, width, label='Male', color='#2980b9', alpha=0.8)
    bars_f = ax.bar(x + width/2, female_vals, width, label='Female', color='#e74c3c', alpha=0.8)
    
    # Annotate bars with count info
    for bar, n_label in zip(bars_m, male_n):
        if n_label and bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{bar.get_height():.1f}%\n({n_label})',
                    ha='center', va='bottom', fontsize=7, color='#2c3e50')
    for bar, n_label in zip(bars_f, female_n):
        if n_label and bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{bar.get_height():.1f}%\n({n_label})',
                    ha='center', va='bottom', fontsize=7, color='#2c3e50')
    
    ax.set_xlabel('Age Group (years)', fontsize=11)
    ax.set_ylabel('Observed 1-Year IHD Rate (%)', fontsize=11)
    ax.set_title('Tier 1: Observed IHD Rate by 5-Year Age Bucket and Sex\n(Empirical Rates — Not Model-Smoothed)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tier1_observed_rate_by_age_sex.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: tier1_observed_rate_by_age_sex.png")
    
    # Clean up temporary column
    covid.drop(columns=['age_bin'], inplace=True)
    
    # -- Calibration Plot --
    try:
        n_bins = 10
        bin_edges = np.linspace(0, covid['pred_prob'].max() * 1.01, n_bins + 1)
        covid['prob_bin'] = pd.cut(covid['pred_prob'], bins=bin_edges, include_lowest=True)
        
        cal_df = covid.groupby('prob_bin', observed=True).agg(
            mean_pred=('pred_prob', 'mean'),
            mean_obs=('outcome', 'mean'),
            n=('outcome', 'count')
        ).dropna()
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, cal_df['mean_pred'].max()], [0, cal_df['mean_pred'].max()],
                color='#bdc3c7', linestyle='--', linewidth=1, label='Perfect calibration')
        ax.scatter(cal_df['mean_pred'], cal_df['mean_obs'], s=cal_df['n'] / cal_df['n'].max() * 300,
                   color='#2c3e50', alpha=0.7, zorder=3, edgecolor='white', linewidth=1)
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Proportion', fontsize=11)
        ax.set_title('Tier 1: Calibration Plot', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "tier1_calibration_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: tier1_calibration_plot.png")
    except Exception as e:
        logger.warning(f"  Calibration plot generation failed: {e}")
    
    # -- Save Diagnostic Report --
    diag_path = os.path.join(results_dir, "tier1_diagnostic_report.txt")
    with open(diag_path, "w") as f:
        f.write("TIER 1 MODEL DIAGNOSTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model:     logit(P(IHD=1)) = β₀ + β₁(Age) + β₂(Gender_Male)\n")
        f.write(f"N:         {len(covid):,}\n")
        f.write(f"Events:    {n_g1:,} (G1: COVID->IHD)\n")
        f.write(f"Controls:  {n_g2:,} (G2: COVID No-IHD)\n\n")
        f.write(f"AUC (C-statistic):        {auc:.4f}\n")
        f.write(f"McFadden Pseudo R²:       {pseudo_r2:.6f}\n")
        f.write(f"AIC:                      {model_t1.aic:.2f}\n")
        f.write(f"BIC:                      {model_t1.bic:.2f}\n")
        f.write(f"Log-Likelihood (model):   {model_t1.llf:.2f}\n")
        if ll_null is not None:
            f.write(f"Log-Likelihood (null):    {ll_null:.2f}\n")
            f.write(f"LR Test χ²:               {lr_stat:.2f} (df=2, p={lr_pval:.2e})\n")
        f.write(f"\nConverged:  {model_t1.mle_retvals['converged']}\n")
        f.write(f"\n--- Interpretation ---\n")
        f.write(f"AUC of {auc:.3f} indicates that age and gender alone have ")
        if auc < 0.6:
            f.write("POOR discriminative ability.\n")
            f.write("The demographics-only model barely separates G1 from G2.\n")
        elif auc < 0.7:
            f.write("FAIR discriminative ability.\n")
            f.write("Demographics explain some but not all of the risk separation.\n")
        elif auc < 0.8:
            f.write("ACCEPTABLE discriminative ability.\n")
            f.write("Demographics are meaningful predictors but other factors contribute.\n")
        else:
            f.write("GOOD discriminative ability.\n")
            f.write("Demographics alone strongly separate IHD from non-IHD patients.\n")
    logger.info("  Saved: tier1_diagnostic_report.txt")
    
    # --------------------------------------------------------------------------
    # 8. ATTENUATION ANALYSIS (Tier 0 → Tier 1)
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SECTION E: Attenuation Analysis (Crude → Tier 1)")
    logger.info("=" * 70)
    
    # The crude SIR from Step 4 was ~1.85.
    # We reference it here. For a formal within-study comparison we also
    # calculate the crude OR from the unadjusted logistic model.
    
    try:
        model_crude = smf.logit('outcome ~ 1', data=covid).fit(disp=0)
        crude_intercept_or = np.exp(model_crude.params['Intercept'])
        # This is just odds(IHD) = G1/G2, not an OR against an external population.
        # For SIR comparison we use the external SIR value.
    except Exception:
        crude_intercept_or = crude_odds
    
    # We cannot directly compare SIR (external reference) to logistic OR
    # (internal cohort comparison). Instead, we note:
    # - Crude within-cohort P(IHD|COVID) is the baseline
    # - The adjusted ORs for age and gender tell us how much of the
    #   age/sex imbalance between G1 and G2 contributes
    
    # The meaningful comparison: how much do age and gender EXPLAIN
    # the difference in IHD rates within the COVID cohort?
    # We quantify via the change in the intercept and the model fit.
    
    # For a more clinically meaningful attenuation, we compare:
    # - Age-standardised rate (ASIR) from Step 4 to
    # - Non-standardised rate (crude) from Step 4
    # These are already computed. Here we focus on what the regression tells us.
    
    # Practical attenuation metric: 
    # If we had included COVID_exposure as a predictor in a cross-cohort model,
    # the attenuation would be (OR_unadj - OR_adj) / (OR_unadj - 1).
    # Since this is within-COVID only (no external comparison), we report:
    # 1. The SMD pre-adjustment
    # 2. How much of the variation in IHD each demographic explains
    
    # Wald test contribution of each variable
    age_or = np.exp(model_t1.params['age'])
    gender_or = np.exp(model_t1.params['gender_male'])
    
    # Compute age-adjusted rate: predicted average probability
    p_adjusted_avg = covid['pred_prob'].mean()
    p_crude = n_g1 / len(covid)
    
    atten_report_path = os.path.join(results_dir, "tier1_attenuation_report.txt")
    with open(atten_report_path, "w") as f:
        f.write("TIER 1 ATTENUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("QUESTION: Does adjusting for Age and Gender explain away the\n")
        f.write("          excess IHD risk observed in the COVID cohort?\n\n")
        
        f.write("--- Reference Values ---\n")
        f.write(f"  Crude SIR (from Step 4, vs general population): ~1.85\n")
        f.write(f"  Crude IHD risk in COVID cohort (P(IHD|COVID)):  {p_crude*100:.4f}%\n")
        f.write(f"  Crude odds (G1/G2):                             {crude_odds:.6f}\n\n")
        
        f.write("--- Tier 1 Adjusted Results ---\n")
        f.write(f"  OR for Age (per year):    {age_or:.4f} (95% CI: {np.exp(model_t1.conf_int().loc['age', 0]):.4f}–{np.exp(model_t1.conf_int().loc['age', 1]):.4f})\n")
        f.write(f"  OR for Male Sex:          {gender_or:.4f} (95% CI: {np.exp(model_t1.conf_int().loc['gender_male', 0]):.4f}–{np.exp(model_t1.conf_int().loc['gender_male', 1]):.4f})\n\n")
        
        f.write("--- Pre-Adjustment Imbalance (SMD) ---\n")
        f.write(f"  Age SMD (G1 vs G2):       {smd_age:.3f}")
        f.write(f"  {'(Large imbalance)' if abs(smd_age) > 0.5 else '(Moderate)' if abs(smd_age) > 0.2 else '(Small)'}\n")
        f.write(f"  Gender SMD (G1 vs G2):    {smd_gender:.3f}")
        f.write(f"  {'(Large imbalance)' if abs(smd_gender) > 0.5 else '(Moderate)' if abs(smd_gender) > 0.2 else '(Small)'}\n\n")
        
        f.write("--- Model Fit ---\n")
        f.write(f"  AUC:     {auc:.4f}\n")
        f.write(f"  R²:      {pseudo_r2:.6f}\n")
        if lr_pval is not None:
            f.write(f"  LR test: p = {lr_pval:.2e} (vs null model)\n")
        f.write(f"\n")
        
        f.write("--- Interpretation ---\n\n")
        
        if auc < 0.65 and pseudo_r2 < 0.01:
            f.write("  FINDING: Age and Gender have MINIMAL explanatory power over\n")
            f.write("  IHD risk within the COVID cohort.\n\n")
            f.write("  The AUC is near 0.5 (random) and R² is near zero, meaning\n")
            f.write("  demographics alone do NOT explain the excess risk.\n\n")
            f.write("  CONCLUSION for Tier 1: The excess IHD risk is NOT simply\n")
            f.write("  because the COVID->IHD patients were older or more male.\n")
            f.write("  → PROCEED to Tier 2 (add clinical baseline: CCI + Medications).\n")
        elif auc < 0.75:
            f.write("  FINDING: Age and Gender provide MODERATE explanatory power.\n\n")
            f.write("  Some of the excess risk is attributable to demographic\n")
            f.write("  differences (older age, more males in G1), but substantial\n")
            f.write("  residual risk remains unexplained.\n\n")
            f.write("  CONCLUSION for Tier 1: Demographics partially confound the\n")
            f.write("  association but do NOT fully explain it.\n")
            f.write("  → PROCEED to Tier 2.\n")
        else:
            f.write("  FINDING: Age and Gender have STRONG explanatory power.\n\n")
            f.write("  Demographics alone can substantially separate G1 from G2.\n")
            f.write("  The excess risk may be largely driven by age/sex composition.\n\n")
            f.write("  CONCLUSION: Consider whether the crude SIR was inflated by\n")
            f.write("  demographic confounding. Tier 2 will determine if adding\n")
            f.write("  clinical factors further attenuates the signal.\n")
            f.write("  → PROCEED to Tier 2 for confirmation.\n")
        
        f.write(f"\n\n--- For the Presentation ---\n\n")
        f.write(f"Key talking point:\n")
        f.write(f"  'After adjusting for age and gender, every additional year of\n")
        f.write(f"   age increases the odds of IHD by {(age_or - 1) * 100:.1f}%.\n")
        f.write(f"   Males have {gender_or:.2f}× the odds of IHD compared to females.\n")
        f.write(f"   However, the model AUC of {auc:.3f} shows that demographics\n")
        f.write(f"   alone {'poorly discriminate' if auc < 0.65 else 'partially discriminate' if auc < 0.75 else 'well discriminate'} between COVID patients who develop IHD\n")
        f.write(f"   and those who do not.'\n")
    
    logger.info("  Saved: tier1_attenuation_report.txt")
    
    # --------------------------------------------------------------------------
    # 9. COMPREHENSIVE VERBOSE REPORT
    # --------------------------------------------------------------------------
    report_path = os.path.join(results_dir, "tier1_verbose_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("       TIER 1 ANALYSIS — COMPREHENSIVE REPORT\n")
        f.write("       Demographic Adjustment: Age + Gender\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. STUDY DESIGN\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Population:        COVID-19 confirmed patients (2020–2023)\n")
        f.write(f"  Exposure:          COVID-19 infection\n")
        f.write(f"  Outcome:           Incident IHD (I21/I22) within 365 days\n")
        f.write(f"  Group 1 (Cases):   COVID -> IHD within 1 year (N={n_g1:,})\n")
        f.write(f"  Group 2 (Controls):COVID → No IHD within 1 year (N={n_g2:,})\n")
        f.write(f"  Total:             {len(covid):,}\n\n")

        f.write("2. DEMOGRAPHIC PROFILE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Characteristic':<25} {'G1 (IHD)':>15} {'G2 (No IHD)':>15} {'SMD':>8} {'p-value':>12}\n")
        f.write(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*8} {'-'*12}\n")
        f.write(f"  {'Age (mean +/- SD)':<25} {age_stats['Group 1 (COVID->IHD)']['Mean ± SD']:>15} {age_stats['Group 2 (COVID No-IHD)']['Mean ± SD']:>15} {smd_age:>8.3f} {p_age:>12.2e}\n")
        f.write(f"  {'Male (%)':<25} {g1_male_pct:>14.1f}% {g2_male_pct:>14.1f}% {smd_gender:>8.3f} {p_gender:>12.2e}\n\n")
        
        f.write("3. LOGISTIC REGRESSION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model: logit(P(IHD=1)) = β₀ + β₁(Age) + β₂(Gender_Male)\n\n")
        f.write(f"  {'Variable':<20} {'β':>10} {'OR':>8} {'95% CI':>20} {'p-value':>12}\n")
        f.write(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*20} {'-'*12}\n")
        for _, row in or_df.iterrows():
            f.write(f"  {row['Variable']:<20} {row['Coefficient']:>10.4f} {row['OR']:>8.4f} "
                    f"({row['Lower_CI']:.4f}–{row['Upper_CI']:.4f}){'':<1} {row['p_value']:>12.2e} {row['Significant']}\n")
        
        f.write(f"\n4. MODEL DIAGNOSTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  AUC (C-statistic):      {auc:.4f}\n")
        f.write(f"  McFadden Pseudo R²:     {pseudo_r2:.6f}\n")
        f.write(f"  AIC:                    {model_t1.aic:.2f}\n")
        f.write(f"  BIC:                    {model_t1.bic:.2f}\n")
        if lr_pval is not None:
            f.write(f"  LR Test (vs null):      χ²={lr_stat:.2f}, p={lr_pval:.2e}\n")
        
        f.write(f"\n5. CONCLUSION\n")
        f.write("-" * 40 + "\n")
        if auc < 0.65:
            f.write("  Age and Gender alone have POOR discriminative ability.\n")
            f.write("  The excess IHD risk in COVID patients is NOT explained by\n")
            f.write("  demographic confounding alone.\n")
            f.write("  → TIER 2 is needed (add comorbidity burden + medications).\n")
        elif auc < 0.75:
            f.write("  Demographics provide MODERATE discrimination.\n")
            f.write("  Some confounding exists but residual risk persists.\n")
            f.write("  → TIER 2 is needed to test clinical baseline.\n")
        else:
            f.write("  Demographics provide STRONG discrimination.\n")
            f.write("  Much of the observed risk difference may be\n")
            f.write("  attributable to age/sex composition.\n")
            f.write("  → TIER 2 will confirm whether clinical factors\n")
            f.write("    further explain or the residual is truly independent.\n")
        
        f.write(f"\n\n" + "=" * 70 + "\n")
        f.write("Generated Plots:\n")
        f.write("  1. tier1_age_distribution.png\n")
        f.write("  2. tier1_gender_distribution.png\n")
        f.write("  3. tier1_forest_plot.png\n")
        f.write("  4. tier1_predicted_prob.png\n")
        f.write("  5. tier1_observed_rate_by_age_sex.png\n")
        f.write("  6. tier1_roc_curve.png\n")
        f.write("  7. tier1_calibration_plot.png\n")
        f.write("=" * 70 + "\n")
    
    logger.info(f"  Saved: tier1_verbose_report.txt")
    
    logger.info("\n" + "=" * 70)
    logger.info("TIER 1 ANALYSIS COMPLETE")
    logger.info("=" * 70)
    
    return {
        'model': model_t1,
        'or_df': or_df,
        'auc': auc,
        'pseudo_r2': pseudo_r2,
        'n_g1': n_g1,
        'n_g2': n_g2,
        'smd_age': smd_age,
        'smd_gender': smd_gender,
        'lr_pval': lr_pval,
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    
    results = run_tier_1(conf)
    
    if results:
        print(f"\n✓ Tier 1 complete. AUC={results['auc']:.4f}")
        print(f"  Results saved to: {conf['paths']['results_dir']}/step_6_tier1/")
