"""
11_tier_3_analysis.py
=====================
Tier 3 of the Stepwise DAG-Aligned Analysis:
    Run Tier 2 model within variant-era strata, with vaccination and severity
    as covariates (not stratification variables).

    Base model per era:
        logit(P(IHD=1)) = b0 + b1(Age) + b2(Male) + b3(CCI)

    Extended models:
        + b4(Vaccinated)                           -- vaccination effect
        + b5(ICU_admitted) + b6(LOS_survivors)      -- severity effect
        + b7(Vaccinated × ICU)                      -- interaction

Purpose:
    Determine whether the COVID->IHD association varies by pandemic era,
    and whether vaccination status and COVID severity independently predict
    post-COVID IHD risk.

    Tier 2 established:
      - Overall CCI OR = 1.262 (G1 vs G2)
      - G1 vs G3 CCI OR = 0.704 (COVID-IHD patients are healthier)
    Tier 3 asks whether these ORs shift across eras and whether vaccination
    and severity modify IHD risk.

Analysis Components:
    A. Era-stratified logistic regression (G1 vs G2 within each era)
    B. Vaccination as binary covariate (0/1) added to Tier 2 model
    C. Era-stratified G1 vs G3 comparison
    D. Severity as covariates (ICU admission binary, LOS excluding deaths)
    E. Vaccination × Severity interaction term
    F. Formal interaction tests (era × CCI)

Input:
    - cohort_tier3_ready.csv (from Step 10)

Output (to data/03_results/step_11_tier3/):
    - era_models/          -- Per-era logistic regression results
    - vaccination_models/  -- Models with vaccination as covariate
    - g1_vs_g3/            -- Era-stratified COVID exposure models
    - severity_models/     -- Severity covariate models
    - interaction_tests/   -- Formal LR and Wald tests
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

RACE_REFERENCE = 'Chinese'
RACE_FORMULA_TERM = f'C(race, Treatment(reference="{RACE_REFERENCE}"))'

BASE_PREDICTORS = ['age', 'gender_male', 'cci_score']
BASE_FORMULA = 'outcome ~ age + gender_male + cci_score'


def build_formula(outcome_col, df, logger, label=""):
    """
    Build the model formula, adding race if available with sufficient coverage.
    Returns (formula_string, n_predictors, has_race).
    """
    base = f'{outcome_col} ~ age + gender_male + cci_score'
    n_pred = 3

    if 'race' in df.columns:
        race_coverage = df['race'].notnull().mean()
        n_race_levels = df['race'].dropna().nunique()
        if race_coverage >= 0.5 and n_race_levels >= 2:
            formula = f'{base} + {RACE_FORMULA_TERM}'
            n_pred += n_race_levels - 1  # dummy variables (minus reference)
            if label:
                logger.info(f"  [{label}] Including race ({n_race_levels} levels, "
                            f"ref={RACE_REFERENCE}, coverage={race_coverage*100:.0f}%)")
            return formula, n_pred, True

    if label:
        logger.info(f"  [{label}] Race not available or insufficient coverage — using base model")
    return base, n_pred, False


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

    # Normalize race values (defensive: handles UPPERCASE from COVIDFACILLOS)
    if 'race' in df.columns:
        race_map = {
            'CHINESE': 'Chinese', 'INDIANS': 'Indian', 'MALAYS': 'Malay',
            'OTHERS': 'Others', 'EURASIANS': 'Eurasian',
        }
        df['race'] = df['race'].map(race_map).fillna(df['race'])
        logger.info(f"  Race distribution: {df['race'].value_counts().to_dict()}")

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
    logger.info("  Model: logit(IHD) = Age + Male + CCI + Race, separately per era")
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

        # Drop missing (base covariates)
        drop_subset = ['age', 'gender_male', 'cci_score']
        analysis_df = era_df.dropna(subset=drop_subset).copy()
        n_dropped = n_total - len(analysis_df)
        if n_dropped > 0:
            logger.info(f"  Dropped {n_dropped} rows with missing age/gender/cci")

        # Also drop rows with missing race if race is available
        if 'race' in analysis_df.columns:
            n_before_race = len(analysis_df)
            analysis_df = analysis_df.dropna(subset=['race']).copy()
            n_race_dropped = n_before_race - len(analysis_df)
            if n_race_dropped > 0:
                logger.info(f"  Dropped {n_race_dropped} rows with missing race")

        label = f"G1G2_{era}"
        formula, n_predictors_used, has_race = build_formula('outcome', analysis_df, logger, label)

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
            model_desc = "logit(IHD) = Age + Male + CCI + Race" if has_race else "logit(IHD) = Age + Male + CCI"
            f.write(f"TIER 3 - {era.upper()} ERA: G1 vs G2 LOGISTIC REGRESSION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_desc}\n")
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
        # Clean up race variable names for display
        def _clean_var(x):
            if x in rename:
                return rename[x]
            if 'race' in x.lower() and 'T.' in x:
                # e.g. "C(race, Treatment(reference="Chinese"))[T.Indian]" -> "Race: Indian"
                race_val = x.split('[T.')[-1].rstrip(']')
                return f"Race: {race_val}"
            return x
        plot_df['Variable'] = plot_df['Variable'].map(_clean_var)
        make_forest_plot(
            plot_df, os.path.join(era_dir, f"tier3_{era.lower()}_g1g2_forest.png"),
            title=f"Tier 3 ({era}): Adjusted ORs (G1 vs G2)"
        )

        epv = n_events / n_predictors_used if n_predictors_used > 0 else 0

        era_summaries.append({
            'Era': era, 'N': n_total, 'Events': n_events,
            'Rate_pct': event_rate, 'EPV': round(epv, 1),
            'N_predictors': n_predictors_used,
            'Race_included': has_race,
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
# ANALYSIS COMPONENT B: VACCINATION AS BINARY COVARIATE
# ==============================================================================

def run_vaccination_covariate(covid, logger, results_dir):
    """
    Add vaccination status as a binary covariate (0/1) to the Tier 2 model.
    Model: logit(IHD) = Age + Male + CCI + Vaccinated

    This directly tests whether vaccination independently affects IHD risk,
    rather than splitting into separate models which loses the direct comparison.

    Vaccination defined as: >=1 dose before COVID infection date.
    Prefers vaccinated_6mo_before_covid if available, falls back to vaccinated_before_covid.
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT B: Vaccination as Binary Covariate")
    logger.info("  Model: logit(IHD) = Age + Male + CCI + Vaccinated")
    logger.info("=" * 70)

    vacc_dir = os.path.join(results_dir, "vaccination_models")
    ensure_dir(vacc_dir)

    # Determine which vaccination column to use
    if 'vaccinated_6mo_before_covid' in covid.columns and covid['vaccinated_6mo_before_covid'].notnull().any():
        vacc_col = 'vaccinated_6mo_before_covid'
        logger.info(f"  Using 6-month vaccination window ({vacc_col})")
    elif 'vaccinated_before_covid' in covid.columns and covid['vaccinated_before_covid'].notnull().any():
        vacc_col = 'vaccinated_before_covid'
        logger.info(f"  Using any-time-before vaccination ({vacc_col})")
    else:
        logger.warning("  No vaccination data available. Skipping Component B.")
        return {}

    vacc_results = {}
    vacc_summaries = []

    # Prepare data
    required_cols = ['age', 'gender_male', 'cci_score', vacc_col]
    vacc_df = covid.dropna(subset=required_cols).copy()
    vacc_df['vaccinated'] = vacc_df[vacc_col].astype(int)

    n_total = len(vacc_df)
    n_events = vacc_df['outcome'].sum()
    n_vacc = vacc_df['vaccinated'].sum()
    logger.info(f"  Pooled: N={n_total:,}, G1={n_events:,}, "
                f"Vaccinated={n_vacc:,} ({n_vacc/n_total*100:.1f}%)")

    # --- 1. Pooled model: base vs base+vaccination ---
    formula_base = 'outcome ~ age + gender_male + cci_score'
    formula_vacc = 'outcome ~ age + gender_male + cci_score + vaccinated'

    model_base, _ = fit_logistic(formula_base, vacc_df, logger, "pooled_base")
    model_vacc, method = fit_logistic(formula_vacc, vacc_df, logger, "pooled_vacc")

    if model_vacc is not None:
        or_df = extract_or_table(model_vacc)
        vacc_results['pooled'] = or_df

        try:
            auc = roc_auc_score(vacc_df['outcome'], model_vacc.predict(vacc_df))
        except ValueError:
            auc = np.nan

        # LR test: does vaccination improve fit?
        lr_stat = lr_p = np.nan
        if model_base is not None:
            lr_stat = -2 * (model_base.llf - model_vacc.llf)
            lr_p = stats.chi2.sf(lr_stat, df=1)
            logger.info(f"  LR test (vaccination contribution): chi2={lr_stat:.4f}, p={lr_p:.4e}")

        logger.info(f"  Pooled vaccination model: Apparent AUC={auc:.4f}")
        for _, row in or_df.iterrows():
            logger.info(f"    {row['Variable']:25s}  OR={row['OR']:.4f}  "
                        f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                        f"p={row['p_value']:.2e} {row['Significant']}")

        or_df.to_csv(os.path.join(vacc_dir, "pooled_vacc_model_ors.csv"), index=False)
        with open(os.path.join(vacc_dir, "pooled_vacc_model.txt"), 'w') as f:
            f.write("VACCINATION AS COVARIATE — POOLED MODEL\n")
            f.write("=" * 60 + "\n\n")
            f.write("Model: logit(IHD) = Age + Male + CCI + Vaccinated\n")
            f.write(f"Vaccination column: {vacc_col}\n")
            f.write(f"N = {n_total:,}, Events = {n_events:,}\n")
            f.write(f"Vaccinated: {n_vacc:,} ({n_vacc/n_total*100:.1f}%)\n")
            f.write(f"Apparent AUC: {auc:.4f}\n")
            if not np.isnan(lr_stat):
                f.write(f"LR test vs base: chi2={lr_stat:.4f}, p={lr_p:.4e}\n")
            f.write("\n")
            f.write(model_vacc.summary().as_text())

        vacc_summaries.append({
            'Era': 'Pooled', 'N': n_total, 'Events': n_events,
            'N_vacc': n_vacc, 'AUC': auc,
            'Vacc_OR': or_df.loc[or_df['Variable'] == 'vaccinated', 'OR'].values[0] if 'vaccinated' in or_df['Variable'].values else np.nan,
            'CCI_OR': or_df.loc[or_df['Variable'] == 'cci_score', 'OR'].values[0] if 'cci_score' in or_df['Variable'].values else np.nan,
            'Status': 'OK',
        })

    # --- 2. Per-era models with vaccination covariate (Delta/Omicron only) ---
    for era in ['Delta', 'Omicron']:
        era_df = vacc_df[vacc_df['variant_era'] == era].copy()
        n_era = len(era_df)
        n_ev = era_df['outcome'].sum()
        n_v = era_df['vaccinated'].sum()

        logger.info(f"\n  --- {era}: N={n_era:,}, G1={n_ev:,}, Vaccinated={n_v:,} ---")

        if n_ev < MIN_EVENTS_FOR_MODEL:
            logger.warning(f"  SKIPPING: Only {n_ev} events")
            continue

        model_era, method = fit_logistic(formula_vacc, era_df, logger, f"vacc_{era}")
        if model_era is None:
            continue

        or_era = extract_or_table(model_era)
        vacc_results[era] = or_era

        try:
            auc_era = roc_auc_score(era_df['outcome'], model_era.predict(era_df))
        except ValueError:
            auc_era = np.nan

        logger.info(f"  {era} Apparent AUC: {auc_era:.4f}")
        for _, row in or_era.iterrows():
            logger.info(f"    {row['Variable']:25s}  OR={row['OR']:.4f}  "
                        f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                        f"p={row['p_value']:.2e} {row['Significant']}")

        or_era.to_csv(os.path.join(vacc_dir, f"vacc_{era.lower()}_ors.csv"), index=False)
        with open(os.path.join(vacc_dir, f"vacc_{era.lower()}.txt"), 'w') as f:
            f.write(f"VACCINATION AS COVARIATE — {era.upper()} ERA\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"N = {n_era:,}, Events = {n_ev:,}, Vaccinated = {n_v:,}\n")
            f.write(f"Apparent AUC: {auc_era:.4f}\n\n")
            f.write(model_era.summary().as_text())

        vacc_summaries.append({
            'Era': era, 'N': n_era, 'Events': n_ev,
            'N_vacc': n_v, 'AUC': auc_era,
            'Vacc_OR': or_era.loc[or_era['Variable'] == 'vaccinated', 'OR'].values[0] if 'vaccinated' in or_era['Variable'].values else np.nan,
            'CCI_OR': or_era.loc[or_era['Variable'] == 'cci_score', 'OR'].values[0] if 'cci_score' in or_era['Variable'].values else np.nan,
            'Status': 'OK',
        })

    summary_df = pd.DataFrame(vacc_summaries)
    summary_df.to_csv(os.path.join(vacc_dir, "vacc_covariate_summary.csv"), index=False)
    logger.info(f"\n  Vaccination covariate summary:\n{summary_df.to_string(index=False)}")

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
# ANALYSIS COMPONENT D: SEVERITY AS COVARIATES (LOS + ICU)
# ==============================================================================

def run_severity_models(covid, logger, results_dir):
    """
    Model severity using LOS and ICU admission as proper covariates.

    Key design decisions:
    - LOS comparisons EXCLUDE in-hospital deaths (their LOS is truncated by death)
    - ICU admission is a binary covariate (cleanest acute severity marker)
    - LOS is used as continuous covariate for survivors only

    Models:
    1. Base + ICU (binary): does ICU admission predict IHD?
    2. Base + LOS (survivors only): does length of stay predict IHD?
    3. Base + ICU + LOS_survivors: combined severity model

    CAUTION: Severity is a potential mediator (COVID -> severity -> IHD).
    Results should be interpreted with this caveat.
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT D: Severity as Covariates (ICU + LOS)")
    logger.info("  LOS excludes in-hospital deaths (truncated by death)")
    logger.info("  CAUTION: Severity may be a mediator on COVID -> IHD path")
    logger.info("=" * 70)

    sev_dir = os.path.join(results_dir, "severity_models")
    ensure_dir(sev_dir)

    covid = covid.copy()

    # --- Derive ICU admission binary ---
    icu_col = None
    for candidate in ['DaysInICU', 'days_in_icu']:
        if candidate in covid.columns:
            icu_col = candidate
            break

    has_icu = False
    if icu_col:
        covid['icu_admitted'] = (pd.to_numeric(covid[icu_col], errors='coerce').fillna(0) > 0).astype(int)
        has_icu = True
        n_icu = covid['icu_admitted'].sum()
        logger.info(f"  ICU admissions: {n_icu:,} ({n_icu/len(covid)*100:.2f}%)")
    else:
        logger.warning("  No ICU column found.")

    # --- Derive LOS for survivors (exclude in-hospital deaths) ---
    has_los = False
    if 'LOS' in covid.columns:
        covid['LOS_num'] = pd.to_numeric(covid['LOS'], errors='coerce')

        # Identify deceased patients
        deceased_col = None
        for candidate in ['Deceased', 'deceased']:
            if candidate in covid.columns:
                deceased_col = candidate
                break

        if deceased_col:
            # Deceased can be 1, 'Y', 'YES', 'True', etc.
            deceased_vals = covid[deceased_col].astype(str).str.strip().str.upper()
            covid['is_deceased'] = deceased_vals.isin(['1', 'Y', 'YES', 'TRUE', '1.0'])
            n_deceased = covid['is_deceased'].sum()
            logger.info(f"  In-hospital deaths: {n_deceased:,}")
        else:
            covid['is_deceased'] = False
            logger.info("  No Deceased column — using all LOS values")

        # LOS for survivors only (exclude deaths whose LOS is truncated)
        covid['los_survivors'] = covid['LOS_num'].copy()
        covid.loc[covid['is_deceased'], 'los_survivors'] = np.nan

        survivors_with_los = covid['los_survivors'].notnull() & (covid['los_survivors'] >= 0)
        n_los_valid = survivors_with_los.sum()
        has_los = n_los_valid > 0

        if has_los:
            los_valid = covid.loc[survivors_with_los, 'los_survivors']
            logger.info(f"  LOS (survivors only): N={n_los_valid:,}, "
                        f"mean={los_valid.mean():.1f}, median={los_valid.median():.0f}, "
                        f"IQR={los_valid.quantile(0.25):.0f}-{los_valid.quantile(0.75):.0f}")
    else:
        logger.warning("  No LOS column found.")

    # Check we have enough data for modelling
    base_cols = ['age', 'gender_male', 'cci_score']
    base_df = covid.dropna(subset=base_cols).copy()

    if not has_icu and not has_los:
        logger.warning("  No severity data available. Skipping Component D.")
        return

    # === Model 1: Base + ICU (binary) ===
    if has_icu:
        logger.info("\n  --- Model D1: Base + ICU Admission ---")
        icu_df = base_df.dropna(subset=['icu_admitted']).copy()
        n_events = icu_df['outcome'].sum()
        n_icu_m = icu_df['icu_admitted'].sum()
        logger.info(f"  N={len(icu_df):,}, G1={n_events:,}, ICU={n_icu_m:,}")

        if n_events >= MIN_EVENTS_FOR_MODEL and n_icu_m >= 5:
            formula = 'outcome ~ age + gender_male + cci_score + icu_admitted'
            model, _ = fit_logistic(formula, icu_df, logger, "D1_icu")

            if model is not None:
                or_df = extract_or_table(model)
                _log_and_save_model(or_df, model, icu_df, sev_dir, "icu_model",
                                    "BASE + ICU ADMISSION", logger)

    # === Model 2: Base + LOS (survivors only) ===
    if has_los:
        logger.info("\n  --- Model D2: Base + LOS (survivors, excluding deaths) ---")
        los_df = base_df[base_df['los_survivors'].notnull() & (base_df['los_survivors'] >= 0)].copy()
        n_events = los_df['outcome'].sum()
        logger.info(f"  N={len(los_df):,}, G1={n_events:,}")

        if n_events >= MIN_EVENTS_FOR_MODEL:
            formula = 'outcome ~ age + gender_male + cci_score + los_survivors'
            model, _ = fit_logistic(formula, los_df, logger, "D2_los")

            if model is not None:
                or_df = extract_or_table(model)
                _log_and_save_model(or_df, model, los_df, sev_dir, "los_survivors_model",
                                    "BASE + LOS (SURVIVORS ONLY)", logger)

    # === Model 3: Combined ICU + LOS ===
    if has_icu and has_los:
        logger.info("\n  --- Model D3: Base + ICU + LOS (survivors) ---")
        combined_df = base_df.dropna(subset=['icu_admitted']).copy()
        combined_df = combined_df[combined_df['los_survivors'].notnull() & (combined_df['los_survivors'] >= 0)].copy()
        n_events = combined_df['outcome'].sum()
        logger.info(f"  N={len(combined_df):,}, G1={n_events:,}")

        if n_events >= MIN_EVENTS_FOR_MODEL:
            formula = 'outcome ~ age + gender_male + cci_score + icu_admitted + los_survivors'
            model, _ = fit_logistic(formula, combined_df, logger, "D3_combined")

            if model is not None:
                or_df = extract_or_table(model)
                _log_and_save_model(or_df, model, combined_df, sev_dir, "icu_los_combined_model",
                                    "BASE + ICU + LOS (SURVIVORS)", logger)

    # === LOS descriptive table: by group × era, excluding deaths ===
    if has_los:
        logger.info("\n  --- LOS Descriptive Table (survivors only, excluding deaths) ---")
        los_desc = []
        for grp in ['Group 1', 'Group 2']:
            for era in ERA_ORDER:
                sub = covid[(covid['group'] == grp) & (covid['variant_era'] == era)
                            & covid['los_survivors'].notnull() & (covid['los_survivors'] >= 0)]
                if len(sub) > 0:
                    los_desc.append({
                        'Group': grp, 'Era': era, 'N_survivors': len(sub),
                        'N_deaths_excluded': ((covid['group'] == grp) & (covid['variant_era'] == era)
                                              & covid['is_deceased']).sum(),
                        'LOS_mean': sub['los_survivors'].mean(),
                        'LOS_median': sub['los_survivors'].median(),
                        'LOS_q25': sub['los_survivors'].quantile(0.25),
                        'LOS_q75': sub['los_survivors'].quantile(0.75),
                        'LOS_max': sub['los_survivors'].max(),
                    })
        if los_desc:
            los_desc_df = pd.DataFrame(los_desc)
            los_desc_df.to_csv(os.path.join(sev_dir, "los_survivors_by_group_era.csv"), index=False)
            logger.info(f"\n  LOS (survivors) by Group x Era:\n{los_desc_df.to_string(index=False)}")

    # === Severity distribution descriptive table ===
    if 'severity_category' in covid.columns:
        logger.info("\n  --- Severity Distribution by Group x Era ---")
        sev_desc = covid.groupby(['group', 'variant_era', 'severity_category']).size().reset_index(name='count')
        sev_pivot = sev_desc.pivot_table(
            index=['group', 'variant_era'], columns='severity_category',
            values='count', fill_value=0
        ).reset_index()
        sev_pivot.to_csv(os.path.join(sev_dir, "severity_distribution_by_group_era.csv"), index=False)
        logger.info(f"\n{sev_pivot.to_string(index=False)}")

        # Event rate by severity
        sev_event_rates = []
        for sev_cat in ['Mild', 'Moderate', 'Severe', 'Critical', 'Unknown']:
            sub = covid[covid['severity_category'] == sev_cat]
            if len(sub) > 0:
                n_ev = sub['outcome'].sum()
                sev_event_rates.append({
                    'Severity': sev_cat, 'N': len(sub),
                    'G1_events': n_ev,
                    'Event_rate_pct': n_ev / len(sub) * 100,
                })
        if sev_event_rates:
            sev_rate_df = pd.DataFrame(sev_event_rates)
            sev_rate_df.to_csv(os.path.join(sev_dir, "event_rate_by_severity.csv"), index=False)
            logger.info(f"\n  Event rate by severity:\n{sev_rate_df.to_string(index=False)}")


def _log_and_save_model(or_df, model, data, out_dir, name, title, logger):
    """Helper to log OR table and save model outputs."""
    try:
        auc = roc_auc_score(data['outcome'], model.predict(data))
    except ValueError:
        auc = np.nan

    logger.info(f"  {title}: Apparent AUC={auc:.4f}")
    for _, row in or_df.iterrows():
        logger.info(f"    {row['Variable']:30s}  OR={row['OR']:.4f}  "
                    f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                    f"p={row['p_value']:.2e} {row['Significant']}")

    or_df.to_csv(os.path.join(out_dir, f"{name}_ors.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}.txt"), 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n\n")
        f.write("CAUTION: Severity indicators are potential MEDIATORS on the\n")
        f.write("causal path: COVID -> Severity -> IHD. Including them may\n")
        f.write("underestimate the total effect of COVID on IHD.\n\n")
        f.write(f"N = {len(data):,}, Events = {data['outcome'].sum():,}\n")
        f.write(f"Apparent AUC: {auc:.4f}\n\n")
        f.write(model.summary().as_text())


# ==============================================================================
# ANALYSIS COMPONENT E: VACCINATION × SEVERITY INTERACTION
# ==============================================================================

def run_vacc_severity_interaction(covid, logger, results_dir):
    """
    Test whether vaccination status modifies the effect of COVID severity on IHD.
    Model: logit(IHD) = Age + Male + CCI + Vaccinated + ICU + Vaccinated × ICU

    If the interaction is significant, it means the effect of severe COVID on
    IHD risk differs between vaccinated and unvaccinated patients.
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT E: Vaccination × Severity Interaction")
    logger.info("  Model: logit(IHD) = Age + Male + CCI + Vacc + ICU + Vacc×ICU")
    logger.info("=" * 70)

    int_dir = os.path.join(results_dir, "vacc_severity_interaction")
    ensure_dir(int_dir)

    # Need both vaccination and severity data
    vacc_col = None
    for candidate in ['vaccinated_6mo_before_covid', 'vaccinated_before_covid']:
        if candidate in covid.columns and covid[candidate].notnull().any():
            vacc_col = candidate
            break

    if vacc_col is None:
        logger.warning("  No vaccination data. Skipping.")
        return

    # Derive ICU if not already done
    if 'icu_admitted' not in covid.columns:
        for icu_candidate in ['DaysInICU', 'days_in_icu']:
            if icu_candidate in covid.columns:
                covid = covid.copy()
                covid['icu_admitted'] = (pd.to_numeric(covid[icu_candidate], errors='coerce').fillna(0) > 0).astype(int)
                break

    if 'icu_admitted' not in covid.columns:
        logger.warning("  No ICU data. Skipping.")
        return

    # Prepare data
    required = ['age', 'gender_male', 'cci_score', vacc_col, 'icu_admitted']
    int_df = covid.dropna(subset=required).copy()
    int_df['vaccinated'] = int_df[vacc_col].astype(int)

    n_total = len(int_df)
    n_events = int_df['outcome'].sum()
    n_vacc = int_df['vaccinated'].sum()
    n_icu = int_df['icu_admitted'].sum()
    n_vacc_icu = ((int_df['vaccinated'] == 1) & (int_df['icu_admitted'] == 1)).sum()

    logger.info(f"  N={n_total:,}, G1={n_events:,}")
    logger.info(f"  Vaccinated={n_vacc:,}, ICU={n_icu:,}, Vacc+ICU={n_vacc_icu:,}")

    if n_events < MIN_EVENTS_FOR_MODEL:
        logger.warning("  Too few events. Skipping.")
        return

    # Reduced model (no interaction)
    formula_red = 'outcome ~ age + gender_male + cci_score + vaccinated + icu_admitted'
    model_red, _ = fit_logistic(formula_red, int_df, logger, "vacc_sev_reduced")

    # Full model (with interaction)
    formula_full = ('outcome ~ age + gender_male + cci_score + vaccinated + icu_admitted '
                    '+ vaccinated:icu_admitted')
    model_full, _ = fit_logistic(formula_full, int_df, logger, "vacc_sev_full")

    if model_red is None or model_full is None:
        logger.error("  Model fitting failed.")
        return

    # LR test for interaction
    lr_stat = -2 * (model_red.llf - model_full.llf)
    lr_p = stats.chi2.sf(lr_stat, df=1)

    logger.info(f"\n  LR test (Vacc×ICU interaction): chi2={lr_stat:.4f}, p={lr_p:.4e}")
    if lr_p < 0.05:
        logger.info("    SIGNIFICANT: Vaccination modifies the ICU->IHD relationship")
    else:
        logger.info("    NOT SIGNIFICANT: No evidence of Vacc×ICU interaction")

    # Log full model
    or_full = extract_or_table(model_full)
    or_red = extract_or_table(model_red)

    logger.info("\n  Full interaction model:")
    for _, row in or_full.iterrows():
        logger.info(f"    {row['Variable']:35s}  OR={row['OR']:.4f}  "
                    f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                    f"p={row['p_value']:.2e} {row['Significant']}")

    or_full.to_csv(os.path.join(int_dir, "vacc_severity_interaction_ors.csv"), index=False)
    or_red.to_csv(os.path.join(int_dir, "vacc_severity_reduced_ors.csv"), index=False)

    with open(os.path.join(int_dir, "vacc_severity_interaction.txt"), 'w') as f:
        f.write("VACCINATION × SEVERITY (ICU) INTERACTION MODEL\n")
        f.write("=" * 60 + "\n\n")
        f.write("Question: Does vaccination modify the effect of severe COVID\n")
        f.write("(ICU admission) on subsequent IHD risk?\n\n")
        f.write(f"Vaccination column: {vacc_col}\n")
        f.write(f"N = {n_total:,}, Events = {n_events:,}\n")
        f.write(f"Vaccinated: {n_vacc:,}, ICU: {n_icu:,}, Both: {n_vacc_icu:,}\n\n")
        f.write(f"LR test (interaction): chi2={lr_stat:.4f}, df=1, p={lr_p:.4e}\n")
        sig_text = "SIGNIFICANT" if lr_p < 0.05 else "NOT SIGNIFICANT"
        f.write(f"Result: {sig_text}\n\n")
        f.write("--- Full Interaction Model ---\n\n")
        f.write(model_full.summary().as_text())
        f.write("\n\n--- Reduced Model (no interaction) ---\n\n")
        f.write(model_red.summary().as_text())

    # Also test with LOS (survivors) if available
    if 'los_survivors' in covid.columns:
        los_int_df = int_df[int_df['los_survivors'].notnull() & (int_df['los_survivors'] >= 0)].copy()
        n_ev_los = los_int_df['outcome'].sum()

        if n_ev_los >= MIN_EVENTS_FOR_MODEL:
            logger.info("\n  --- Vaccination × LOS (survivors) interaction ---")
            formula_los_full = ('outcome ~ age + gender_male + cci_score + vaccinated '
                                '+ los_survivors + vaccinated:los_survivors')
            model_los, _ = fit_logistic(formula_los_full, los_int_df, logger, "vacc_los_interaction")

            if model_los is not None:
                or_los = extract_or_table(model_los)
                logger.info("  Vacc × LOS interaction model:")
                for _, row in or_los.iterrows():
                    logger.info(f"    {row['Variable']:35s}  OR={row['OR']:.4f}  "
                                f"({row['Lower_CI']:.4f}-{row['Upper_CI']:.4f})  "
                                f"p={row['p_value']:.2e} {row['Significant']}")
                or_los.to_csv(os.path.join(int_dir, "vacc_los_interaction_ors.csv"), index=False)


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

    # Vaccination coverage by era (both any-time and 6-month window)
    if 'vaccinated_before_covid' in covid.columns:
        agg_dict = {
            'N': ('uin', 'count'),
            'N_vacc_anytime': ('vaccinated_before_covid', lambda x: x.sum()),
            'Mean_doses': ('doses_before_ref', 'mean'),
        }
        if 'vaccinated_6mo_before_covid' in covid.columns:
            agg_dict['N_vacc_6mo'] = ('vaccinated_6mo_before_covid', lambda x: x.sum())
        vacc_table = covid.groupby('variant_era').agg(**agg_dict).reset_index()
        vacc_table['Pct_vacc_anytime'] = (vacc_table['N_vacc_anytime'] / vacc_table['N'] * 100).round(1)
        if 'N_vacc_6mo' in vacc_table.columns:
            vacc_table['Pct_vacc_6mo'] = (vacc_table['N_vacc_6mo'] / vacc_table['N'] * 100).round(1)
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
    logger.info("  Model per stratum: logit(IHD) = Age + Male + CCI + Race (when available)")
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

    # B. Vaccination as binary covariate
    vacc_results = run_vaccination_covariate(covid, logger, results_dir)

    # C. Era-stratified G1 vs G3
    g1g3_results = run_era_stratified_g1g3(df_all, logger, results_dir)

    # D. Severity as covariates (ICU + LOS excluding deaths)
    run_severity_models(covid, logger, results_dir)

    # E. Vaccination × Severity interaction
    run_vacc_severity_interaction(covid, logger, results_dir)

    # F. Era interaction tests (era × CCI)
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
    summary_lines.append("MODEL: logit(IHD) = Age + Male + CCI [+ Vaccinated] [+ ICU/LOS], stratified by variant era")
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
