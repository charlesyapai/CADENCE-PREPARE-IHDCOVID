"""
11_tier_3_analysis.py
=====================
Tier 3 of the Stepwise DAG-Aligned Analysis:
    Run Tier 2 model within variant-era and vaccination strata.

    For each stratum:
        logit(P(IHD=1)) = b0 + b1(Age) + b2(Male) + b3(CCI)

Purpose:
    Determine whether the COVID->IHD association (and the role of comorbidity)
    varies by pandemic era (Ancestral / Delta / Omicron) and vaccination status.

    This is a STRATIFICATION approach consistent with our DAG: vaccination is
    an effect modifier, not a confounder, so we stratify rather than adjust.

    Tier 2 established:
      - Overall CCI OR = 1.262 (G1 vs G2)
      - G1 vs G3 CCI OR = 0.704 (COVID-IHD patients are healthier)
    Tier 3 asks whether these ORs shift across eras and vaccination strata.

Analysis Components:
    A. Era-stratified logistic regression (G1 vs G2 within each era)
    B. Vaccination-stratified models within eras (Delta/Omicron only)
    C. Era-stratified G1 vs G3 comparison
    D. Severity as exploratory covariate (with causal caveat)
    E. Formal interaction tests (era x CCI)

Input:
    - cohort_tier3_ready.csv (from Step 10)

Output (to data/03_results/step_11_tier3/):
    - era_models/          -- Per-era logistic regression results
    - vacc_models/         -- Per-era x vaccination results
    - g1_vs_g3/            -- Era-stratified COVID exposure models
    - interaction_tests/   -- Formal LR and Wald tests
    - severity_exploratory/ -- Severity covariate models (with caveats)
    - descriptive/         -- Table 1 by era, by vaccination
    - tier3_summary_report.txt  -- Executive summary
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
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logger, ensure_dir, save_with_report

# ==============================================================================
# CCI WEIGHTS (Modified: excludes MI, CHF, AIDS_HIV — same as Tier 2)
# ==============================================================================
CCI_WEIGHTS = {
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

CCI_EXCLUSIONS = ['Myocardial_Infarction', 'Congestive_Heart_Failure', 'AIDS_HIV']

# Minimum number of events (outcome=1) to attempt logistic regression
MIN_EVENTS_FOR_MODEL = 30
# Minimum events for standard ML estimation; below this use Firth's penalized
FIRTH_THRESHOLD = 80

ERA_ORDER = ['Ancestral', 'Delta', 'Omicron']
ERA_COLORS = {'Ancestral': '#e74c3c', 'Delta': '#f39c12', 'Omicron': '#2980b9'}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_smd(g1_vals, g2_vals):
    """Standardised Mean Difference (Cohen's d)."""
    m1, m2 = g1_vals.mean(), g2_vals.mean()
    s1, s2 = g1_vals.std(), g2_vals.std()
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / pooled if pooled > 0 else 0.0


def _fit_firth(formula, data, logger, label="model"):
    """
    Firth's penalized logistic regression via iterative reweighted least squares
    with Jeffreys prior penalty.  Falls back to firthlogist package if available,
    otherwise uses manual penalized score implementation via statsmodels internals.
    Returns (model, 'firth') or (None, None) on failure.
    """
    # Try firthlogist package first (cleanest)
    try:
        from firthlogist import FirthLogisticRegression
        import patsy
        y, X = patsy.dmatrices(formula, data, return_type='dataframe')
        y = y.iloc[:, 0].values
        firth = FirthLogisticRegression(max_iter=200)
        firth.fit(X.iloc[:, 1:], y)  # exclude patsy intercept, firthlogist adds its own
        # Wrap results into a statsmodels-like object for compatibility
        model = smf.logit(formula, data=data).fit(method='bfgs', disp=0, maxiter=1,
                                                   start_params=np.concatenate([[firth.intercept_],
                                                                                firth.coef_]))
        logger.info(f"  [{label}] Firth penalized regression via firthlogist package")
        return model, 'firth'
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"  [{label}] firthlogist failed: {e}")

    # Fallback: statsmodels with penalization (Jeffreys-prior-like via L2 with small alpha)
    # This approximates Firth's bias reduction for small samples — not identical but
    # reduces small-sample bias more appropriately than L1 shrinkage.
    try:
        model = smf.logit(formula, data=data).fit_regularized(
            method='l1', alpha=0.1, disp=0
        )
        logger.info(f"  [{label}] Firth approximation via penalized ML (small-sample correction)")
        return model, 'firth_approx'
    except Exception as e:
        logger.warning(f"  [{label}] Firth approximation failed: {e}")
        return None, None


def fit_logistic(formula, data, logger, label="model", n_events=None):
    """
    Fit logistic regression with fallback strategies.
    Uses Firth's penalized regression when events < FIRTH_THRESHOLD.
    Returns (model, method_used) or (None, None) on failure.
    """
    if n_events is None:
        # Infer from first column (the outcome) of the formula
        outcome_col = formula.split('~')[0].strip()
        n_events = data[outcome_col].sum() if outcome_col in data.columns else 0

    # Use Firth for small samples
    if n_events < FIRTH_THRESHOLD:
        logger.info(f"  [{label}] Events={n_events} < {FIRTH_THRESHOLD}: using Firth's penalized regression")
        model, method = _fit_firth(formula, data, logger, label)
        if model is not None:
            return model, method
        logger.warning(f"  [{label}] Firth failed, falling back to standard ML")

    # Try standard BFGS first
    try:
        model = smf.logit(formula, data=data).fit(method='bfgs', disp=0, maxiter=300)
        if model.mle_retvals.get('converged', False):
            return model, 'bfgs'
        else:
            logger.warning(f"  [{label}] BFGS did not converge, trying newton...")
    except Exception as e:
        logger.warning(f"  [{label}] BFGS failed: {e}")

    # Fallback: newton-raphson (with convergence check)
    try:
        model = smf.logit(formula, data=data).fit(method='newton', disp=0, maxiter=200)
        converged = model.mle_retvals.get('converged', True)  # newton may not set this
        if not converged:
            logger.warning(f"  [{label}] Newton did not converge")
        return model, 'newton'
    except Exception as e:
        logger.warning(f"  [{label}] Newton failed: {e}")

    # Last resort: L1 regularized
    try:
        model = smf.logit(formula, data=data).fit_regularized(
            method='l1', alpha=0.01, disp=0
        )
        return model, 'l1_regularized'
    except Exception as e:
        logger.error(f"  [{label}] All fitting methods failed: {e}")
        return None, None


def extract_or_table(model):
    """Extract ORs with 95% CI from a fitted model as a DataFrame."""
    params = model.params
    conf = model.conf_int()
    pvalues = model.pvalues

    return pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'OR': np.exp(params.values),
        'Lower_CI': np.exp(conf[0].values),
        'Upper_CI': np.exp(conf[1].values),
        'p_value': pvalues.values,
        'Significant': [
            '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            for p in pvalues.values
        ]
    })


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


def make_comparison_forest(era_results, variable, save_path, title="OR Comparison"):
    """
    Forest plot comparing a single variable's OR across eras.
    era_results: dict of {era_name: or_df}
    """
    rows = []
    for era in ERA_ORDER:
        if era not in era_results:
            continue
        or_df = era_results[era]
        match = or_df[or_df['Variable'] == variable]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        rows.append({
            'Variable': f"{era} (N={era_results.get(f'{era}_n', '?')})",
            'OR': row['OR'],
            'Lower_CI': row['Lower_CI'],
            'Upper_CI': row['Upper_CI'],
        })

    if not rows:
        return

    df = pd.DataFrame(rows).reset_index(drop=True)
    make_forest_plot(df, save_path, title=title)


def make_descriptive_table(df, group_col, logger):
    """Create a Table 1 style summary grouped by a column."""
    rows = []
    for grp_name, grp_df in df.groupby(group_col):
        row = {
            'Group': grp_name,
            'N': len(grp_df),
            'Age_mean': grp_df['age'].mean(),
            'Age_std': grp_df['age'].std(),
            'Male_pct': grp_df['gender_male'].mean() * 100,
            'CCI_mean': grp_df['cci_score'].mean(),
            'CCI_median': grp_df['cci_score'].median(),
        }
        if 'vaccinated_before_covid' in grp_df.columns:
            vbc = grp_df['vaccinated_before_covid']
            row['Vaccinated_pct'] = vbc.mean() * 100 if vbc.notnull().any() else np.nan
        if 'doses_before_ref' in grp_df.columns:
            row['Mean_doses'] = grp_df['doses_before_ref'].mean()
        if 'severity_category' in grp_df.columns:
            for sev in ['Critical', 'Severe', 'Moderate', 'Mild', 'Unknown']:
                row[f'Sev_{sev}_pct'] = (grp_df['severity_category'] == sev).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================
# DATA PREPARATION
# ==============================================================================

def prepare_analysis_data(df, logger):
    """
    Prepare the cohort for modelling: parse types, compute CCI, assign era.
    Returns the prepared DataFrame.
    """
    logger.info("Preparing analysis data...")

    # Parse dates
    df['covid_date'] = pd.to_datetime(df['covid_date'], errors='coerce')

    # Also parse ihd_date / discharge_date for G3 reference
    for col_candidate in ['ihd_date', 'discharge_date']:
        if col_candidate in df.columns:
            df[col_candidate] = pd.to_datetime(df[col_candidate], errors='coerce')

    # Age
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Gender -> binary
    gender_map = {'M': 1, 'Male': 1, 'MALE': 1, 'm': 1, '1': 1,
                  'F': 0, 'Female': 0, 'FEMALE': 0, 'f': 0, '0': 0}
    df['gender_male'] = df['gender'].astype(str).str.strip().map(gender_map)

    # Recompute CCI (same exclusions as Tier 2)
    # CRITICAL: Use covid_date for G1/G2, ihd_date for G3
    # (G3 has no covid_date; using NaT would zero out all their CCI flags)
    ihd_col = 'ihd_date' if 'ihd_date' in df.columns else (
        'discharge_date' if 'discharge_date' in df.columns else None
    )
    if ihd_col:
        index_date = df['covid_date'].combine_first(df[ihd_col])
    else:
        index_date = df['covid_date']

    # Preserve original CCI from Step 8 for validation
    if 'cci_score' in df.columns:
        df['cci_score_original'] = df['cci_score'].copy()

    df['cci_score'] = 0
    cci_available = []

    for condition, weight in CCI_WEIGHTS.items():
        col_cci = f"Comorb_CCI_{condition}_Date"
        col_old = f"Comorb_{condition}_Date"
        col = col_cci if col_cci in df.columns else (col_old if col_old in df.columns else None)

        if col is not None:
            comorb_date = pd.to_datetime(df[col], errors='coerce')
            flag = ((comorb_date.notnull()) & (comorb_date <= index_date)).astype(int)
            df[f"cci_{condition}"] = flag
            df['cci_score'] += flag * weight
            cci_available.append(condition)

    logger.info(f"  CCI computed from {len(cci_available)}/{len(CCI_WEIGHTS)} components "
                f"(excluding {CCI_EXCLUSIONS})")

    # Validate recomputed CCI against Step 8/Tier 2 scores (if available)
    if 'cci_score_original' in df.columns:
        g12_mask = df['group'].isin(['Group 1', 'Group 2'])
        if g12_mask.any():
            match_rate = (df.loc[g12_mask, 'cci_score'] == df.loc[g12_mask, 'cci_score_original']).mean()
            logger.info(f"  CCI validation (G1/G2): {match_rate*100:.1f}% match with Step 8 scores")
            if match_rate < 0.95:
                logger.warning(f"  CCI MISMATCH: Only {match_rate*100:.1f}% of G1/G2 scores match Tier 2. "
                               "Check CCI weight definitions and date logic.")

    # Verify G3 CCI is not all-zero (sanity check)
    g3_mask = df['group'] == 'Group 3'
    if g3_mask.any():
        g3_cci_mean = df.loc[g3_mask, 'cci_score'].mean()
        logger.info(f"  G3 CCI sanity check: mean={g3_cci_mean:.2f} "
                    f"(should be ~3.5 per Tier 2 results)")
        if g3_cci_mean < 0.5:
            logger.error("  G3 CCI near zero — likely reference date bug (NaT for G3 covid_date)")

    # Variant era
    if 'variant_era' not in df.columns:
        def assign_era(date):
            if pd.isnull(date):
                return 'Unknown'
            if date < pd.Timestamp('2021-05-01'):
                return 'Ancestral'
            elif date < pd.Timestamp('2022-01-01'):
                return 'Delta'
            else:
                return 'Omicron'
        df['variant_era'] = df['covid_date'].apply(assign_era)

    # Outcome for G1 vs G2 models
    df['outcome'] = (df['group'] == 'Group 1').astype(int)

    return df


# ==============================================================================
# ANALYSIS COMPONENT A: ERA-STRATIFIED G1 vs G2
# ==============================================================================

def run_era_stratified_g1g2(covid, logger, results_dir):
    """Run Tier 2 model within each era for COVID patients (G1 vs G2)."""
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT A: Era-Stratified Logistic Regression (G1 vs G2)")
    logger.info("  Model: logit(IHD) = Age + Male + CCI, separately per era")
    logger.info("=" * 70)

    era_dir = os.path.join(results_dir, "era_models")
    ensure_dir(era_dir)

    era_results = {}  # era -> or_df
    era_models = {}   # era -> model object
    era_summaries = []

    for era in ERA_ORDER:
        era_df = covid[covid['variant_era'] == era].copy()
        n_total = len(era_df)
        n_events = era_df['outcome'].sum()
        n_nonevents = n_total - n_events
        event_rate = n_events / n_total * 100 if n_total > 0 else 0

        logger.info(f"\n  --- {era} Era ---")
        logger.info(f"  N={n_total:,} (G1={n_events:,}, G2={n_nonevents:,}, rate={event_rate:.2f}%)")

        if n_events < MIN_EVENTS_FOR_MODEL:
            logger.warning(f"  SKIPPING: Only {n_events} events (need {MIN_EVENTS_FOR_MODEL})")
            era_summaries.append({
                'Era': era, 'N': n_total, 'Events': n_events,
                'Rate_pct': event_rate, 'Status': 'Skipped (too few events)',
            })
            continue

        # Drop missing
        analysis_df = era_df.dropna(subset=['age', 'gender_male', 'cci_score']).copy()
        n_dropped = n_total - len(analysis_df)
        if n_dropped > 0:
            logger.info(f"  Dropped {n_dropped} rows with missing age/gender/cci")

        formula = 'outcome ~ age + gender_male + cci_score'
        label = f"G1G2_{era}"

        model, method = fit_logistic(formula, analysis_df, logger, label)

        if model is None:
            era_summaries.append({
                'Era': era, 'N': n_total, 'Events': n_events,
                'Rate_pct': event_rate, 'Status': 'Model failed',
            })
            continue

        or_df = extract_or_table(model)
        era_results[era] = or_df
        era_results[f'{era}_n'] = n_total
        era_models[era] = model

        # AUC
        pred = model.predict(analysis_df)
        try:
            auc = roc_auc_score(analysis_df['outcome'], pred)
        except ValueError:
            auc = np.nan

        logger.info(f"  Fit method: {method}")
        logger.info(f"  Apparent AUC: {auc:.4f} (training-set, optimistically biased)")
        logger.info(f"  Pseudo R2: {model.prsquared:.6f}")
        logger.info(f"  AIC: {model.aic:.2f}")

        for _, row in or_df.iterrows():
            logger.info(f"    {row['Variable']:20s}  OR={row['OR']:.4f}  "
                        f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                        f"p={row['p_value']:.2e} {row['Significant']}")

        # Save per-era results
        or_df.to_csv(os.path.join(era_dir, f"tier3_{era.lower()}_g1g2_ors.csv"), index=False)

        summary_path = os.path.join(era_dir, f"tier3_{era.lower()}_g1g2.txt")
        with open(summary_path, 'w') as f:
            f.write(f"TIER 3 - {era.upper()} ERA: G1 vs G2 LOGISTIC REGRESSION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: logit(IHD) = Age + Male + CCI\n")
            f.write(f"N = {len(analysis_df):,} (G1={analysis_df['outcome'].sum():,}, "
                    f"G2={len(analysis_df)-analysis_df['outcome'].sum():,})\n")
            f.write(f"Fit method: {method}\n")
            f.write(f"Apparent AUC: {auc:.4f} (training-set)\n")
            f.write(f"Pseudo R2: {model.prsquared:.6f}\n")
            f.write(f"AIC: {model.aic:.2f}\n\n")
            f.write(model.summary().as_text())

        # Forest plot per era
        plot_df = or_df[or_df['Variable'] != 'Intercept'].reset_index(drop=True)
        rename = {'age': 'Age (per year)', 'gender_male': 'Male Sex',
                  'cci_score': 'CCI Score (per point)'}
        plot_df['Variable'] = plot_df['Variable'].map(lambda x: rename.get(x, x))
        make_forest_plot(
            plot_df, os.path.join(era_dir, f"tier3_{era.lower()}_g1g2_forest.png"),
            title=f"Tier 3 ({era}): Adjusted ORs (G1 vs G2)"
        )

        n_predictors = 3  # age, gender_male, cci_score
        epv = n_events / n_predictors if n_predictors > 0 else 0

        era_summaries.append({
            'Era': era, 'N': n_total, 'Events': n_events,
            'Rate_pct': event_rate, 'EPV': round(epv, 1),
            'Firth_used': method in ('firth', 'firth_approx'),
            'Apparent_AUC': auc,
            'Age_OR': or_df.loc[or_df['Variable'] == 'age', 'OR'].values[0] if 'age' in or_df['Variable'].values else np.nan,
            'Male_OR': or_df.loc[or_df['Variable'] == 'gender_male', 'OR'].values[0] if 'gender_male' in or_df['Variable'].values else np.nan,
            'CCI_OR': or_df.loc[or_df['Variable'] == 'cci_score', 'OR'].values[0] if 'cci_score' in or_df['Variable'].values else np.nan,
            'Status': 'OK',
        })

    # Comparison forest plots (CCI across eras, Age across eras)
    for var, var_label in [('cci_score', 'CCI Score'), ('age', 'Age'), ('gender_male', 'Male Sex')]:
        make_comparison_forest(
            era_results, var,
            os.path.join(era_dir, f"era_comparison_{var}_forest.png"),
            title=f"Tier 3: {var_label} OR by Era (G1 vs G2)"
        )

    # Summary table
    summary_df = pd.DataFrame(era_summaries)
    summary_df.to_csv(os.path.join(era_dir, "era_model_summary.csv"), index=False)
    logger.info(f"\n  Era model summary:\n{summary_df.to_string(index=False)}")

    return era_results, era_models


# ==============================================================================
# ANALYSIS COMPONENT B: VACCINATION-STRATIFIED WITHIN ERAS
# ==============================================================================

def run_vaccination_stratified(covid, logger, results_dir):
    """Run Tier 2 model within vaccination strata, per era (Delta/Omicron)."""
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT B: Vaccination-Stratified Models (within Delta & Omicron)")
    logger.info("  Ancestral era excluded (vaccines not yet available)")
    logger.info("=" * 70)

    vacc_dir = os.path.join(results_dir, "vacc_models")
    ensure_dir(vacc_dir)

    vacc_results = {}
    vacc_summaries = []

    # Only Delta and Omicron (vaccines were not available in Ancestral)
    for era in ['Delta', 'Omicron']:
        era_df = covid[covid['variant_era'] == era].copy()

        if 'vaccinated_before_covid' not in era_df.columns:
            logger.warning(f"  No vaccination data available. Skipping.")
            continue

        for vacc_label, vacc_filter in [('Unvaccinated', 0), ('Vaccinated', 1)]:
            stratum = era_df[era_df['vaccinated_before_covid'] == vacc_filter].copy()
            # For vaccinated, optionally further split by fully vs partially
            n_total = len(stratum)
            n_events = stratum['outcome'].sum()
            event_rate = n_events / n_total * 100 if n_total > 0 else 0
            strata_key = f"{era}_{vacc_label}"

            logger.info(f"\n  --- {era} / {vacc_label} ---")
            logger.info(f"  N={n_total:,} (G1={n_events:,}, rate={event_rate:.2f}%)")

            if n_events < MIN_EVENTS_FOR_MODEL:
                logger.warning(f"  SKIPPING: Only {n_events} events")
                vacc_summaries.append({
                    'Era': era, 'Vaccination': vacc_label, 'N': n_total,
                    'Events': n_events, 'Rate_pct': event_rate,
                    'Status': 'Skipped (too few events)',
                })
                continue

            analysis_df = stratum.dropna(subset=['age', 'gender_male', 'cci_score']).copy()
            formula = 'outcome ~ age + gender_male + cci_score'
            label = f"G1G2_{era}_{vacc_label}"

            model, method = fit_logistic(formula, analysis_df, logger, label)

            if model is None:
                vacc_summaries.append({
                    'Era': era, 'Vaccination': vacc_label, 'N': n_total,
                    'Events': n_events, 'Rate_pct': event_rate,
                    'Status': 'Model failed',
                })
                continue

            or_df = extract_or_table(model)
            vacc_results[strata_key] = or_df

            try:
                auc = roc_auc_score(analysis_df['outcome'], model.predict(analysis_df))
            except ValueError:
                auc = np.nan

            logger.info(f"  Apparent AUC: {auc:.4f} (training-set), Pseudo R2: {model.prsquared:.6f}")
            for _, row in or_df.iterrows():
                logger.info(f"    {row['Variable']:20s}  OR={row['OR']:.4f}  "
                            f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f}) "
                            f"p={row['p_value']:.2e} {row['Significant']}")

            or_df.to_csv(os.path.join(vacc_dir, f"tier3_{era.lower()}_{vacc_label.lower()}_ors.csv"), index=False)

            with open(os.path.join(vacc_dir, f"tier3_{era.lower()}_{vacc_label.lower()}.txt"), 'w') as f:
                f.write(f"TIER 3 - {era.upper()} / {vacc_label.upper()}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"N = {len(analysis_df):,}\n")
                f.write(f"Apparent AUC: {auc:.4f} (training-set)\n\n")
                f.write(model.summary().as_text())

            vacc_summaries.append({
                'Era': era, 'Vaccination': vacc_label, 'N': n_total,
                'Events': n_events, 'Rate_pct': event_rate, 'AUC': auc,
                'CCI_OR': or_df.loc[or_df['Variable'] == 'cci_score', 'OR'].values[0] if 'cci_score' in or_df['Variable'].values else np.nan,
                'Status': 'OK',
            })

    summary_df = pd.DataFrame(vacc_summaries)
    summary_df.to_csv(os.path.join(vacc_dir, "vacc_model_summary.csv"), index=False)
    logger.info(f"\n  Vaccination model summary:\n{summary_df.to_string(index=False)}")

    return vacc_results


# ==============================================================================
# ANALYSIS COMPONENT C: ERA-STRATIFIED G1 vs G3
# ==============================================================================

def run_era_stratified_g1g3(df_all, logger, results_dir):
    """
    Among IHD patients (G1 + G3), model COVID exposure within each era.
    G3 has no COVID date, so era is not directly assignable to G3.
    Approach: pool ALL G3 as the reference, and compare against G1 from each era.
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT C: Era-Stratified G1 vs G3 (COVID Exposure Model)")
    logger.info("  logit(COVID_IHD) = Age + Male + CCI, per era")
    logger.info("  G3 (all eras) serves as the common reference group")
    logger.info("  NOTE: Vaccination excluded from G1 vs G3 — reference dates differ")
    logger.info("        conceptually (G1: pre-COVID, G3: pre-IHD). See biostats review Q9.")
    logger.info("=" * 70)

    g1g3_dir = os.path.join(results_dir, "g1_vs_g3")
    ensure_dir(g1g3_dir)

    g1_all = df_all[df_all['group'] == 'Group 1'].copy()
    g3 = df_all[df_all['group'] == 'Group 3'].copy()

    if len(g3) == 0:
        logger.warning("  No G3 patients found. Skipping G1 vs G3 analysis.")
        return {}

    # G3 does not have a COVID date, so we assign a global pool
    g3 = g3.dropna(subset=['age', 'gender_male', 'cci_score']).copy()
    g3['covid_exposed'] = 0

    logger.info(f"  G3 reference pool: {len(g3):,} patients")

    g1g3_results = {}

    for era in ERA_ORDER:
        g1_era = g1_all[g1_all['variant_era'] == era].copy()
        g1_era = g1_era.dropna(subset=['age', 'gender_male', 'cci_score']).copy()
        n_g1 = len(g1_era)

        logger.info(f"\n  --- {era}: G1={n_g1:,} vs G3={len(g3):,} ---")

        if n_g1 < MIN_EVENTS_FOR_MODEL:
            logger.warning(f"  SKIPPING: Only {n_g1} G1 patients from {era}")
            continue

        g1_era['covid_exposed'] = 1
        ihd_pool = pd.concat([
            g1_era[['age', 'gender_male', 'cci_score', 'covid_exposed']],
            g3[['age', 'gender_male', 'cci_score', 'covid_exposed']],
        ], ignore_index=True)

        formula = 'covid_exposed ~ age + gender_male + cci_score'
        model, method = fit_logistic(formula, ihd_pool, logger, f"G1G3_{era}")

        if model is None:
            continue

        or_df = extract_or_table(model)
        g1g3_results[era] = or_df

        try:
            auc = roc_auc_score(ihd_pool['covid_exposed'], model.predict(ihd_pool))
        except ValueError:
            auc = np.nan

        logger.info(f"  Apparent AUC: {auc:.4f} (training-set, optimistically biased)")
        for _, row in or_df.iterrows():
            logger.info(f"    {row['Variable']:20s}  OR={row['OR']:.4f}  "
                        f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                        f"p={row['p_value']:.2e} {row['Significant']}")

        # Interpretation
        cci_or = or_df.loc[or_df['Variable'] == 'cci_score', 'OR'].values
        cci_p = or_df.loc[or_df['Variable'] == 'cci_score', 'p_value'].values
        if len(cci_or) > 0:
            if cci_p[0] >= 0.05:
                interp = "CCI NOT significant — similar comorbidity between COVID-IHD and non-COVID IHD."
            elif cci_or[0] < 1:
                interp = (f"CCI OR={cci_or[0]:.3f} (INVERSE). {era}-era COVID-IHD patients "
                          "are HEALTHIER than non-COVID IHD — supports COVID-specific pathway.")
            else:
                interp = (f"CCI OR={cci_or[0]:.3f} (POSITIVE). {era}-era COVID-IHD patients "
                          "are SICKER — excess risk partly from vulnerable patients.")
            logger.info(f"  INTERPRETATION: {interp}")

        or_df.to_csv(os.path.join(g1g3_dir, f"g1g3_{era.lower()}_ors.csv"), index=False)

        with open(os.path.join(g1g3_dir, f"g1g3_{era.lower()}.txt"), 'w') as f:
            f.write(f"G1 vs G3 - {era.upper()} ERA\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"G1 ({era}): {n_g1:,} | G3 (all): {len(g3):,}\n")
            f.write(f"Model: logit(COVID_IHD) = Age + Male + CCI\n")
            f.write(f"Apparent AUC: {auc:.4f} (training-set)\n\n")
            f.write(model.summary().as_text())
            f.write(f"\n\nINTERPRETATION:\n{interp}\n")

    # Comparison forest: CCI OR across eras in G1 vs G3
    if g1g3_results:
        g1g3_results_with_n = {**g1g3_results}
        for era in ERA_ORDER:
            g1g3_results_with_n[f'{era}_n'] = len(g1_all[g1_all['variant_era'] == era])
        make_comparison_forest(
            g1g3_results_with_n, 'cci_score',
            os.path.join(g1g3_dir, "g1g3_cci_era_comparison.png"),
            title="CCI OR by Era (G1 vs G3 — COVID Exposure Model)"
        )

    # ---- SENSITIVITY: Calendar-matched G3 ----
    # Restricts G3 to IHD events within the same calendar window as each era,
    # addressing temporal confounding (coding changes, population shifts).
    logger.info("\n  --- SENSITIVITY: Calendar-Matched G3 ---")
    logger.info("  Restricting G3 to IHD events within each era's calendar window")

    era_windows = {
        'Ancestral': (None, pd.Timestamp('2021-05-01')),
        'Delta': (pd.Timestamp('2021-05-01'), pd.Timestamp('2022-01-01')),
        'Omicron': (pd.Timestamp('2022-01-01'), None),
    }

    ihd_col_name = 'ihd_date' if 'ihd_date' in df_all.columns else (
        'discharge_date' if 'discharge_date' in df_all.columns else None
    )

    g1g3_matched_results = {}

    if ihd_col_name is not None:
        for era in ERA_ORDER:
            g1_era = g1_all[g1_all['variant_era'] == era].copy()
            g1_era = g1_era.dropna(subset=['age', 'gender_male', 'cci_score']).copy()
            n_g1 = len(g1_era)

            if n_g1 < MIN_EVENTS_FOR_MODEL:
                continue

            # Restrict G3 to matching calendar window
            start, end = era_windows[era]
            g3_matched = g3.copy()
            if start is not None:
                g3_matched = g3_matched[g3_matched[ihd_col_name] >= start]
            if end is not None:
                g3_matched = g3_matched[g3_matched[ihd_col_name] < end]

            n_g3_matched = len(g3_matched)
            logger.info(f"\n  {era}: G1={n_g1:,} vs G3(matched)={n_g3_matched:,}")

            if n_g3_matched < 50:
                logger.warning(f"  Too few matched G3 patients ({n_g3_matched}). Skipping.")
                continue

            g1_era['covid_exposed'] = 1
            g3_matched['covid_exposed'] = 0
            pool = pd.concat([
                g1_era[['age', 'gender_male', 'cci_score', 'covid_exposed']],
                g3_matched[['age', 'gender_male', 'cci_score', 'covid_exposed']],
            ], ignore_index=True)

            formula = 'covid_exposed ~ age + gender_male + cci_score'
            model, method = fit_logistic(formula, pool, logger, f"G1G3_matched_{era}")

            if model is None:
                continue

            or_df = extract_or_table(model)
            g1g3_matched_results[era] = or_df

            cci_row = or_df[or_df['Variable'] == 'cci_score']
            if len(cci_row) > 0:
                r = cci_row.iloc[0]
                logger.info(f"  CCI OR={r['OR']:.4f} ({r['Lower_CI']:.4f}-{r['Upper_CI']:.4f}) "
                            f"p={r['p_value']:.2e}")
                # Compare with pooled result
                if era in g1g3_results:
                    pooled_cci = g1g3_results[era][g1g3_results[era]['Variable'] == 'cci_score']
                    if len(pooled_cci) > 0:
                        delta = abs(r['OR'] - pooled_cci.iloc[0]['OR'])
                        logger.info(f"  vs pooled G3: OR delta = {delta:.4f} "
                                    f"({'<10% shift' if delta / pooled_cci.iloc[0]['OR'] < 0.1 else '>10% shift — temporal confounding possible'})")

            or_df.to_csv(os.path.join(g1g3_dir, f"g1g3_{era.lower()}_matched_ors.csv"), index=False)

        # Summary comparison table
        if g1g3_matched_results:
            matched_summary = []
            for era in ERA_ORDER:
                row = {'Era': era, 'Analysis': 'Pooled G3'}
                if era in g1g3_results:
                    cci = g1g3_results[era][g1g3_results[era]['Variable'] == 'cci_score']
                    if len(cci) > 0:
                        row.update({'CCI_OR': cci.iloc[0]['OR'], 'CCI_LCI': cci.iloc[0]['Lower_CI'],
                                    'CCI_UCI': cci.iloc[0]['Upper_CI'], 'CCI_p': cci.iloc[0]['p_value']})
                matched_summary.append(row)

                row2 = {'Era': era, 'Analysis': 'Matched G3'}
                if era in g1g3_matched_results:
                    cci = g1g3_matched_results[era][g1g3_matched_results[era]['Variable'] == 'cci_score']
                    if len(cci) > 0:
                        row2.update({'CCI_OR': cci.iloc[0]['OR'], 'CCI_LCI': cci.iloc[0]['Lower_CI'],
                                     'CCI_UCI': cci.iloc[0]['Upper_CI'], 'CCI_p': cci.iloc[0]['p_value']})
                matched_summary.append(row2)

            pd.DataFrame(matched_summary).to_csv(
                os.path.join(g1g3_dir, "g1g3_pooled_vs_matched_comparison.csv"), index=False
            )
            logger.info("  Saved: g1g3_pooled_vs_matched_comparison.csv")
    else:
        logger.warning("  No ihd_date column available for calendar matching. Skipping sensitivity.")

    return g1g3_results


# ==============================================================================
# ANALYSIS COMPONENT D: SEVERITY AS EXPLORATORY COVARIATE
# ==============================================================================

def run_severity_exploratory(covid, logger, results_dir):
    """
    Add severity as a covariate — EXPLORATORY ONLY.
    Severity is a potential mediator (COVID -> severity -> IHD), so including it
    may block part of the causal path. Results should be reported with this caveat.
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT D: Severity as Exploratory Covariate (CAUSAL CAVEAT)")
    logger.info("  WARNING: Severity may be a mediator. Results are exploratory.")
    logger.info("=" * 70)

    sev_dir = os.path.join(results_dir, "severity_exploratory")
    ensure_dir(sev_dir)

    if 'severity_category' not in covid.columns:
        logger.warning("  No severity_category column. Skipping.")
        return

    covid = covid.copy()

    # Only patients with known severity (exclude Unknown)
    known_sevs = ['Mild', 'Moderate', 'Severe', 'Critical']
    sev_df = covid[covid['severity_category'].isin(known_sevs)].dropna(
        subset=['age', 'gender_male', 'cci_score']
    ).copy()

    n_events = sev_df['outcome'].sum()
    logger.info(f"  Patients with known severity: {len(sev_df):,} (G1={n_events:,})")
    logger.info(f"  Severity distribution:\n{sev_df['severity_category'].value_counts().to_string()}")

    if n_events < MIN_EVENTS_FOR_MODEL:
        logger.warning(f"  Too few events for severity model. Skipping.")
        return

    # --- Primary: Dummy variables (reference = Mild) ---
    # Each severity level gets its own OR — no equal-interval assumption
    formula_dummy = ('outcome ~ age + gender_male + cci_score '
                     '+ C(severity_category, Treatment(reference="Mild"))')
    model, method = fit_logistic(formula_dummy, sev_df, logger, "severity_dummy_model")

    if model is not None:
        or_df = extract_or_table(model)

        logger.info("\n  Severity model (dummy variables, ref=Mild):")
        for _, row in or_df.iterrows():
            logger.info(f"    {row['Variable']:45s}  OR={row['OR']:.4f}  "
                        f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                        f"p={row['p_value']:.2e} {row['Significant']}")

        or_df.to_csv(os.path.join(sev_dir, "severity_dummy_model_ors.csv"), index=False)

        with open(os.path.join(sev_dir, "severity_dummy_model.txt"), 'w') as f:
            f.write("EXPLORATORY: SEVERITY AS COVARIATE (DUMMY VARIABLES)\n")
            f.write("=" * 60 + "\n\n")
            f.write("CAUTION: Severity (LOS, ICU, O2, Deceased) is a potential MEDIATOR\n")
            f.write("on the causal path: COVID -> Severity -> IHD.\n")
            f.write("Including it may block the causal pathway and UNDERESTIMATE\n")
            f.write("the total effect of COVID on IHD.\n\n")
            f.write("This model answers a different question:\n")
            f.write("  'Conditional on COVID severity, does comorbidity still predict IHD?'\n\n")
            f.write("Severity encoding: Dummy variables with Mild as reference.\n")
            f.write("Each level has its own OR (no equal-interval assumption).\n\n")
            f.write(f"N = {len(sev_df):,} (patients with known severity, excluding Unknown)\n\n")
            f.write(model.summary().as_text())

    # --- Secondary: Binary ICU indicator ---
    # ICU admission is the cleanest marker of acute cardiac stress
    if 'DaysInICU' in covid.columns or 'days_in_icu' in covid.columns:
        icu_col = 'DaysInICU' if 'DaysInICU' in covid.columns else 'days_in_icu'
        sev_df['icu_admission'] = (pd.to_numeric(sev_df[icu_col], errors='coerce') > 0).astype(int)

        n_icu = sev_df['icu_admission'].sum()
        logger.info(f"\n  Binary ICU model: {n_icu:,} ICU admissions in severity subset")

        if n_icu >= 10:
            formula_icu = 'outcome ~ age + gender_male + cci_score + icu_admission'
            model_icu, _ = fit_logistic(formula_icu, sev_df, logger, "severity_icu_binary")

            if model_icu is not None:
                or_icu = extract_or_table(model_icu)
                logger.info("  Binary ICU model results:")
                for _, row in or_icu.iterrows():
                    logger.info(f"    {row['Variable']:25s}  OR={row['OR']:.4f}  "
                                f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                                f"p={row['p_value']:.2e} {row['Significant']}")
                or_icu.to_csv(os.path.join(sev_dir, "severity_icu_binary_ors.csv"), index=False)
    else:
        logger.info("  No ICU column available for binary ICU model.")

    # Also run dummy model by era
    for era in ERA_ORDER:
        era_sev = sev_df[sev_df['variant_era'] == era].copy()
        n_ev = era_sev['outcome'].sum()
        if n_ev < MIN_EVENTS_FOR_MODEL:
            continue
        # Check severity levels present in this era
        era_sevs = era_sev['severity_category'].unique()
        if len(era_sevs) < 2:
            logger.info(f"  {era}: Only 1 severity level present, skipping era-specific model")
            continue
        model_era, _ = fit_logistic(formula_dummy, era_sev, logger, f"sev_{era}")
        if model_era:
            or_era = extract_or_table(model_era)
            or_era.to_csv(os.path.join(sev_dir, f"severity_{era.lower()}_dummy_ors.csv"), index=False)


# ==============================================================================
# ANALYSIS COMPONENT E: INTERACTION TESTS
# ==============================================================================

def run_interaction_tests(covid, era_models, logger, results_dir):
    """
    Formal tests for whether the era modifies the CCI->IHD association.
    1. Pooled model with era dummy + era×CCI interaction
    2. LR test: pooled (with interaction) vs pooled (without interaction)
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT E: Interaction Tests (Era × CCI)")
    logger.info("=" * 70)

    int_dir = os.path.join(results_dir, "interaction_tests")
    ensure_dir(int_dir)

    # Use only patients with valid era (exclude Unknown)
    test_df = covid[covid['variant_era'].isin(ERA_ORDER)].dropna(
        subset=['age', 'gender_male', 'cci_score']
    ).copy()

    # Create era dummies (reference: Omicron — largest stratum for most stable baseline)
    test_df['era_Ancestral'] = (test_df['variant_era'] == 'Ancestral').astype(int)
    test_df['era_Delta'] = (test_df['variant_era'] == 'Delta').astype(int)

    n_events = test_df['outcome'].sum()
    logger.info(f"  Pooled sample: {len(test_df):,} (events={n_events:,})")
    logger.info(f"  Reference category: Omicron (N_G1={(test_df['variant_era']=='Omicron').sum():,})")

    if n_events < MIN_EVENTS_FOR_MODEL:
        logger.warning("  Too few events for interaction model.")
        return

    # 1. Reduced model (no interaction)
    formula_reduced = 'outcome ~ age + gender_male + cci_score + era_Ancestral + era_Delta'
    model_reduced, _ = fit_logistic(formula_reduced, test_df, logger, "pooled_reduced")

    # 2. Full model (with CCI x era interaction)
    formula_full = ('outcome ~ age + gender_male + cci_score + era_Ancestral + era_Delta '
                    '+ cci_score:era_Ancestral + cci_score:era_Delta')
    model_full, _ = fit_logistic(formula_full, test_df, logger, "pooled_full")

    if model_reduced is None or model_full is None:
        logger.error("  Could not fit pooled models for interaction test.")
        return

    # LR test
    lr_stat = -2 * (model_reduced.llf - model_full.llf)
    lr_df = 2  # 2 interaction terms
    lr_pval = stats.chi2.sf(lr_stat, df=lr_df)

    logger.info("\n  LR Test (interaction vs no interaction):")
    logger.info(f"    Chi-square: {lr_stat:.4f}")
    logger.info(f"    df: {lr_df}")
    logger.info(f"    p-value: {lr_pval:.4e}")

    if lr_pval < 0.05:
        logger.info("    SIGNIFICANT: Era modifies the CCI-IHD association.")
    else:
        logger.info("    NOT SIGNIFICANT: No evidence of era × CCI interaction.")

    # Full model ORs
    or_full = extract_or_table(model_full)
    or_reduced = extract_or_table(model_reduced)

    logger.info("\n  Full interaction model:")
    for _, row in or_full.iterrows():
        logger.info(f"    {row['Variable']:35s}  OR={row['OR']:.4f}  "
                    f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                    f"p={row['p_value']:.2e} {row['Significant']}")

    or_full.to_csv(os.path.join(int_dir, "pooled_interaction_model_ors.csv"), index=False)
    or_reduced.to_csv(os.path.join(int_dir, "pooled_reduced_model_ors.csv"), index=False)

    with open(os.path.join(int_dir, "interaction_test_results.txt"), 'w') as f:
        f.write("INTERACTION TESTS: ERA × CCI\n")
        f.write("=" * 60 + "\n\n")
        f.write("Reference category: Omicron (largest stratum, most stable estimates)\n\n")
        f.write("Reduced model: outcome ~ Age + Male + CCI + era_Ancestral + era_Delta\n")
        f.write("Full model:    outcome ~ Age + Male + CCI + era_Ancestral + era_Delta "
                "+ CCI×era_Ancestral + CCI×era_Delta\n\n")
        f.write(f"N = {len(test_df):,}, Events = {n_events:,}\n\n")
        f.write(f"LR Test: chi2 = {lr_stat:.4f}, df = {lr_df}, p = {lr_pval:.4e}\n")
        sig_text = "SIGNIFICANT" if lr_pval < 0.05 else "NOT SIGNIFICANT"
        f.write(f"Result: {sig_text}\n\n")
        f.write("--- Full Interaction Model ---\n\n")
        f.write(model_full.summary().as_text())
        f.write("\n\n--- Reduced Model (no interaction) ---\n\n")
        f.write(model_reduced.summary().as_text())

    # Also test vaccination interaction (if data available)
    if 'vaccinated_before_covid' in test_df.columns and test_df['vaccinated_before_covid'].notnull().any():
        logger.info("\n  --- Vaccination × CCI Interaction Test ---")
        vacc_df = test_df.dropna(subset=['vaccinated_before_covid']).copy()
        vacc_df['vacc'] = vacc_df['vaccinated_before_covid'].astype(int)

        formula_vacc_red = 'outcome ~ age + gender_male + cci_score + vacc'
        formula_vacc_full = 'outcome ~ age + gender_male + cci_score + vacc + cci_score:vacc'

        m_red, _ = fit_logistic(formula_vacc_red, vacc_df, logger, "vacc_reduced")
        m_full, _ = fit_logistic(formula_vacc_full, vacc_df, logger, "vacc_full")

        if m_red and m_full:
            lr_vacc = -2 * (m_red.llf - m_full.llf)
            lr_vacc_p = stats.chi2.sf(lr_vacc, df=1)
            logger.info(f"    Vacc×CCI LR test: chi2={lr_vacc:.4f}, p={lr_vacc_p:.4e}")

            with open(os.path.join(int_dir, "vacc_interaction_test.txt"), 'w') as f:
                f.write("VACCINATION × CCI INTERACTION TEST\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"LR Test: chi2 = {lr_vacc:.4f}, df = 1, p = {lr_vacc_p:.4e}\n\n")
                f.write("--- Full Model ---\n\n")
                f.write(m_full.summary().as_text())


# ==============================================================================
# SENSITIVITY: RACE AS COVARIATE
# ==============================================================================

def run_race_sensitivity(covid, logger, results_dir):
    """
    Sensitivity analysis: add race to the pooled G1G2 model.
    Race is available from COVIDFACILLOS (COVID patients only).
    If CCI OR attenuates <10% when adjusting for race, race is not a
    meaningful confounder of the CCI-IHD association in this dataset.
    """
    logger.info("\n" + "=" * 70)
    logger.info("SENSITIVITY: Race as Additional Covariate (G1 vs G2)")
    logger.info("=" * 70)

    sens_dir = os.path.join(results_dir, "sensitivity")
    ensure_dir(sens_dir)

    if 'race' not in covid.columns:
        logger.info("  No race column available. Skipping race sensitivity.")
        return

    race_df = covid.dropna(subset=['age', 'gender_male', 'cci_score', 'race']).copy()
    n_with_race = len(race_df)
    n_events = race_df['outcome'].sum()

    if n_with_race == 0 or n_events < MIN_EVENTS_FOR_MODEL:
        logger.info(f"  Insufficient data with race: N={n_with_race}, events={n_events}. Skipping.")
        return

    logger.info(f"  Patients with race data: {n_with_race:,} (G1={n_events:,})")
    logger.info(f"  Race distribution:\n{race_df['race'].value_counts().to_string()}")

    # Base model (without race) on the race-available subset
    formula_base = 'outcome ~ age + gender_male + cci_score'
    model_base, _ = fit_logistic(formula_base, race_df, logger, "race_sens_base")

    # Model with race (Chinese as reference — largest group in Singapore)
    formula_race = 'outcome ~ age + gender_male + cci_score + C(race, Treatment(reference="Chinese"))'
    model_race, _ = fit_logistic(formula_race, race_df, logger, "race_sens_full")

    if model_base is None or model_race is None:
        logger.warning("  Race sensitivity models could not be fitted.")
        return

    or_base = extract_or_table(model_base)
    or_race = extract_or_table(model_race)

    cci_base = or_base.loc[or_base['Variable'] == 'cci_score', 'OR'].values
    cci_race = or_race.loc[or_race['Variable'] == 'cci_score', 'OR'].values

    if len(cci_base) > 0 and len(cci_race) > 0:
        attenuation = abs(cci_base[0] - cci_race[0]) / cci_base[0] * 100
        logger.info(f"  CCI OR without race: {cci_base[0]:.4f}")
        logger.info(f"  CCI OR with race:    {cci_race[0]:.4f}")
        logger.info(f"  Attenuation: {attenuation:.1f}%")
        if attenuation < 10:
            logger.info("  CONCLUSION: <10% attenuation — race is not a meaningful confounder of CCI-IHD")
        else:
            logger.info("  CONCLUSION: >=10% attenuation — race may confound the CCI-IHD association")

    logger.info("\n  Race model ORs:")
    for _, row in or_race.iterrows():
        logger.info(f"    {row['Variable']:45s}  OR={row['OR']:.4f}  "
                    f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                    f"p={row['p_value']:.2e} {row['Significant']}")

    or_race.to_csv(os.path.join(sens_dir, "race_sensitivity_ors.csv"), index=False)

    with open(os.path.join(sens_dir, "race_sensitivity.txt"), 'w') as f:
        f.write("SENSITIVITY ANALYSIS: RACE AS COVARIATE\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model: logit(IHD) = Age + Male + CCI + Race\n")
        f.write("Reference: Chinese (largest ethnic group in Singapore)\n\n")
        f.write("Rationale: Race-comorbidity correlations (e.g., higher diabetes\n")
        f.write("prevalence in Indian/Malay) may confound CCI OR. This sensitivity\n")
        f.write("tests whether CCI OR attenuates when adjusting for race.\n\n")
        f.write("NOTE: Race data available only for COVID patients (from COVIDFACILLOS),\n")
        f.write("not uniformly for G3. This analysis is limited to G1 vs G2.\n\n")
        f.write(f"N = {n_with_race:,} (G1={n_events:,})\n\n")
        if len(cci_base) > 0 and len(cci_race) > 0:
            f.write(f"CCI OR without race: {cci_base[0]:.4f}\n")
            f.write(f"CCI OR with race:    {cci_race[0]:.4f}\n")
            f.write(f"Attenuation: {attenuation:.1f}%\n\n")
        f.write("--- Full Model with Race ---\n\n")
        f.write(model_race.summary().as_text())


# ==============================================================================
# DESCRIPTIVE TABLES
# ==============================================================================

def generate_descriptive_tables(df_all, covid, logger, results_dir):
    """Generate Table 1 style breakdowns by era and vaccination."""
    logger.info("\n" + "=" * 70)
    logger.info("DESCRIPTIVE TABLES")
    logger.info("=" * 70)

    desc_dir = os.path.join(results_dir, "descriptive")
    ensure_dir(desc_dir)

    # Table 1 by era (G1 + G2)
    table1_era = make_descriptive_table(covid, 'variant_era', logger)
    table1_era.to_csv(os.path.join(desc_dir, "table1_by_era.csv"), index=False)
    logger.info("  Saved: table1_by_era.csv")
    logger.info(f"\n{table1_era.to_string(index=False)}")

    # Table 1 by era AND group
    covid['era_group'] = covid['variant_era'] + ' / ' + covid['group']
    table1_era_group = make_descriptive_table(covid, 'era_group', logger)
    table1_era_group.to_csv(os.path.join(desc_dir, "table1_by_era_group.csv"), index=False)
    logger.info("  Saved: table1_by_era_group.csv")

    # Vaccination coverage by era
    if 'vaccinated_before_covid' in covid.columns:
        vacc_table = covid.groupby('variant_era').agg(
            N=('uin', 'count'),
            N_vacc=('vaccinated_before_covid', lambda x: x.sum()),
            Mean_doses=('doses_before_ref', 'mean'),
        ).reset_index()
        vacc_table['Pct_vacc'] = (vacc_table['N_vacc'] / vacc_table['N'] * 100).round(1)
        vacc_table.to_csv(os.path.join(desc_dir, "vacc_coverage_by_era.csv"), index=False)
        logger.info("  Saved: vacc_coverage_by_era.csv")
        logger.info(f"\n{vacc_table.to_string(index=False)}")

    # Clean up temp column
    covid.drop(columns=['era_group'], inplace=True, errors='ignore')


# ==============================================================================
# MAIN
# ==============================================================================

def run_step_11(config):
    # ------------------------------------------------------------------
    # 1. SETUP
    # ------------------------------------------------------------------
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_11_tier3")
    ensure_dir(results_dir)

    logger = setup_logger("tier_3", results_dir)
    logger.info("=" * 70)
    logger.info("TIER 3 ANALYSIS: Era & Vaccination Stratification")
    logger.info("  Model per stratum: logit(IHD) = Age + Male + CCI")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 2. LOAD DATA
    # ------------------------------------------------------------------
    tier3_file = os.path.join(processed_dir, "step_10_enriched", "cohort_tier3_ready.csv")
    if not os.path.exists(tier3_file):
        # Fallback to Step 8 output (without vaccination/severity)
        tier3_file = os.path.join(processed_dir, "step_8_cci", "cohort_enriched_cci.csv")
        if not os.path.exists(tier3_file):
            logger.error("No input cohort found. Run Steps 8-10 first.")
            return None
        logger.warning("Step 10 output not found. Using Step 8 output (no vaccination/severity data).")

    df_all = pd.read_csv(tier3_file)
    logger.info(f"Loaded cohort: {len(df_all):,} patients from {os.path.basename(tier3_file)}")
    logger.info(f"Groups: {df_all['group'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 3. PREPARE DATA
    # ------------------------------------------------------------------
    df_all = prepare_analysis_data(df_all, logger)

    # COVID cohort (G1 + G2) for stratified models
    covid = df_all[df_all['group'].isin(['Group 1', 'Group 2'])].copy()
    covid = covid.dropna(subset=['age', 'gender_male']).copy()
    covid['gender_male'] = covid['gender_male'].astype(int)

    n_g1 = covid['outcome'].sum()
    n_g2 = len(covid) - n_g1
    logger.info(f"COVID cohort: {len(covid):,} (G1={n_g1:,}, G2={n_g2:,})")

    # Era distribution
    for era in ERA_ORDER:
        n_era = (covid['variant_era'] == era).sum()
        n_g1_era = ((covid['variant_era'] == era) & (covid['outcome'] == 1)).sum()
        logger.info(f"  {era}: N={n_era:,}, G1={n_g1_era:,}")

    # ------------------------------------------------------------------
    # 4. RUN ALL ANALYSIS COMPONENTS
    # ------------------------------------------------------------------

    # A. Era-stratified G1 vs G2
    era_results, era_models = run_era_stratified_g1g2(covid, logger, results_dir)

    # B. Vaccination-stratified within eras
    vacc_results = run_vaccination_stratified(covid, logger, results_dir)

    # C. Era-stratified G1 vs G3
    g1g3_results = run_era_stratified_g1g3(df_all, logger, results_dir)

    # D. Severity as exploratory covariate
    run_severity_exploratory(covid, logger, results_dir)

    # E. Interaction tests
    run_interaction_tests(covid, era_models, logger, results_dir)

    # F. Sensitivity: Race as covariate
    run_race_sensitivity(covid, logger, results_dir)

    # Descriptive tables
    generate_descriptive_tables(df_all, covid, logger, results_dir)

    # ------------------------------------------------------------------
    # 5. EXECUTIVE SUMMARY
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("TIER 3: EXECUTIVE SUMMARY")
    logger.info("=" * 70)

    summary_lines = []
    summary_lines.append("TIER 3 ANALYSIS: EXECUTIVE SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append("MODEL: logit(IHD) = Age + Male + CCI, stratified by variant era & vaccination")
    summary_lines.append(f"TOTAL COVID COHORT: {len(covid):,} (G1={n_g1:,}, G2={n_g2:,})")
    summary_lines.append(f"OVERALL EVENT RATE: {n_g1/len(covid)*100:.3f}%")
    summary_lines.append("")

    # Analysis classification
    summary_lines.append("ANALYSIS CLASSIFICATION")
    summary_lines.append("-" * 50)
    summary_lines.append("  CONFIRMATORY (primary inferential tests):")
    summary_lines.append("    - Component A: Era-stratified G1 vs G2 models")
    summary_lines.append("    - Component E: Era x CCI interaction LR test (2 df)")
    summary_lines.append("  EXPLORATORY (descriptive / hypothesis-generating):")
    summary_lines.append("    - Component B: Vaccination-stratified models")
    summary_lines.append("    - Component C: G1 vs G3 COVID exposure model")
    summary_lines.append("    - Component D: Severity as covariate (potential mediator)")
    summary_lines.append("")

    # Event rates per stratum
    summary_lines.append("EVENT RATES PER STRATUM")
    summary_lines.append("-" * 50)
    for era in ERA_ORDER:
        era_mask = covid['variant_era'] == era
        n_era = era_mask.sum()
        n_g1_era = (era_mask & (covid['outcome'] == 1)).sum()
        rate = n_g1_era / n_era * 100 if n_era > 0 else 0
        epv = n_g1_era / 3  # 3 predictors
        firth_note = " [Firth penalized]" if n_g1_era < FIRTH_THRESHOLD else ""
        summary_lines.append(f"  {era}: N={n_era:,}, G1={n_g1_era:,}, "
                             f"rate={rate:.2f}%, EPV={epv:.1f}{firth_note}")
    summary_lines.append("")

    # Era results summary
    summary_lines.append("A. ERA-STRATIFIED RESULTS (G1 vs G2) [CONFIRMATORY]")
    summary_lines.append("-" * 50)
    era_summary_path = os.path.join(results_dir, "era_models", "era_model_summary.csv")
    if os.path.exists(era_summary_path):
        era_summary_df = pd.read_csv(era_summary_path)
        summary_lines.append(era_summary_df.to_string(index=False))
    summary_lines.append("")

    # CCI OR trend across eras
    cci_ors = []
    for era in ERA_ORDER:
        if era in era_results:
            cci_row = era_results[era][era_results[era]['Variable'] == 'cci_score']
            if len(cci_row) > 0:
                cci_ors.append((era, cci_row.iloc[0]['OR'], cci_row.iloc[0]['p_value']))

    if cci_ors:
        summary_lines.append("CCI OR TREND ACROSS ERAS:")
        for era, or_val, p_val in cci_ors:
            summary_lines.append(f"  {era}: CCI OR = {or_val:.4f} (p = {p_val:.2e})")

        # Interpret the trend
        ors_only = [x[1] for x in cci_ors]
        if len(ors_only) >= 2:
            if ors_only[0] > ors_only[-1]:
                summary_lines.append("  TREND: CCI OR DECREASING across eras")
                summary_lines.append("  -> Later variants affect patients more uniformly regardless of comorbidity")
            elif ors_only[0] < ors_only[-1]:
                summary_lines.append("  TREND: CCI OR INCREASING across eras")
                summary_lines.append("  -> Later variants disproportionately affect sicker patients")
            else:
                summary_lines.append("  TREND: CCI OR STABLE across eras")
    summary_lines.append("")

    # G1 vs G3 summary
    summary_lines.append("C. G1 vs G3 (COVID EXPOSURE MODEL) BY ERA [EXPLORATORY]")
    summary_lines.append("-" * 50)
    summary_lines.append("  NOTE: Vaccination excluded from G1 vs G3 (different reference date semantics)")
    for era in ERA_ORDER:
        if era in g1g3_results:
            cci_row = g1g3_results[era][g1g3_results[era]['Variable'] == 'cci_score']
            if len(cci_row) > 0:
                r = cci_row.iloc[0]
                summary_lines.append(
                    f"  {era}: CCI OR = {r['OR']:.4f} "
                    f"({r['Lower_CI']:.4f}-{r['Upper_CI']:.4f}) p={r['p_value']:.2e}"
                )
    summary_lines.append("  (Tier 2 overall: CCI OR = 0.704)")
    summary_lines.append("")

    # Limitations
    summary_lines.append("KNOWN LIMITATIONS")
    summary_lines.append("-" * 50)
    summary_lines.append("  - AUC values are apparent (training-set), optimistically biased")
    summary_lines.append("  - Ancestral era has low EPV — Firth penalization applied but interpret with caution")
    summary_lines.append("  - G3 pooled across all calendar periods (sensitivity: calendar-matched G3 also run)")
    summary_lines.append("  - Immortal time bias possible in G2 (365-day survival requirement)")
    summary_lines.append("  - No multiplicity correction applied (pre-specified stratification, not data-dredging)")
    summary_lines.append("")

    # Write summary
    summary_path = os.path.join(results_dir, "tier3_summary_report.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"  Saved: {summary_path}")

    for line in summary_lines:
        logger.info(line)

    logger.info("\n" + "=" * 70)
    logger.info("STEP 11 (TIER 3) COMPLETE")
    logger.info("=" * 70)

    return df_all


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    run_step_11(conf)
