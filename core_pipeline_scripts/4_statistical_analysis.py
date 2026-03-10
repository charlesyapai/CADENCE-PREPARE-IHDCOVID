
import os
import sys
import pandas as pd
import numpy as np
import yaml
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt

# Add local directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logger, report_df_info, plot_bar_chart, save_with_report, ensure_dir

def run_step_4(config):
    # 1. Setup
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_4_analysis")
    ensure_dir(results_dir)
    
    logger = setup_logger("step_4", results_dir)
    logger.info("Starting Step 4: Statistical Analysis")
    
    # 2. Load Enriched Cohort
    input_file = os.path.join(processed_dir, "step_3_features", "cohort_enriched.csv")
    if not os.path.exists(input_file):
        logger.error(f"Missing input: {input_file}. Run Step 3 first.")
        return
        
    df = pd.read_csv(input_file)
    logger.info(f"Loaded Cohort: {len(df):,} patients.")
    
    # Pre-processing for Analysis
    # ---------------------------
    # Define Outcome: IHD (Group 1 or Group 3 are the IHD cases)
    # But strictly, we are comparing:
    # A. Risk of IHD in COVID (Group 1 vs Group 2) -> "Does COVID increase IHD risk?"
    #    Calculate Incidence Rate in COVID patients (G1+G2) vs Incidence in Non-COVID?
    #    Actually, user design is:
    #    G1: COVID -> IHD
    #    G2: COVID -> No IHD
    #    G3: No COVID -> IHD (Naive IHD)
    
    # Wait, where is the "No COVID -> No IHD" group (General Population Control)? 
    # The user script generated "control_cohort_matched.csv" which seemed to be G2 (Naive IHD) in the old script?
    # Re-reading Plan: 
    # "The idea is that we will have 3 groups: ... G3: patients with IHD and no diagnosis of COVID".
    # This design lacks a purely healthy control (No COVID, No IHD).
    # Thus we can only compare:
    # 1. Profile of Post-COVID IHD (G1) vs Naive IHD (G3) -> "Is the phenotype different?"
    # 2. Incidence? We can calculate incidence within the COVID cohort (G1 / (G1+G2)).
    #    But we cannot compare it to "Non-COVID Incidence" unless we have the denominator for G3 (The healthy non-covid population).
    
    # User Request: "The goal is to use incidence rates to determine the causality of covid first to find a risk of IHD if you have COVID... then afterwards we want to find the age-standardized risk"
    # To calculate Risk of IHD given COVID, we need P(IHD|COVID).
    # We have G1 (IHD|COVID) and G2 (No IHD|COVID). So we can calc P(IHD|COVID).
    # To say if it's "Increased", we need P(IHD|No COVID).
    # We have G3 (IHD|No COVID). But we DO NOT have the denominator for No COVID (The healthy population).
    # Unless... the user implies we use National Statistics or the "Universe" as denominator?
    # OR, maybe the "matching" implied we brought in controls?
    # Original script `generate_control_cohort.py` seemed to pull random controls.
    # But current pipeline Step 2 only pulls G1/G2/G3 based on intersections.
    
    # CRITICAL: If G3 is just "IHD patients without COVID", we lack the denominator to calculate "Rate of IHD in Non-COVID".
    # However, I must proceed with what I have. 
    # Perhaps we treat the "G3" count as the numerator, and we need to Estimate the denominator?
    # OR, maybe I missed a step where we ingest a "General Population" sample.
    # Re-reading prompt: "The current script does case-control matching but because we have pivoted away from Cox, we no longer really need to do this."
    # "Taking the incidence rates through matching of the data from the different sources"
    
    # Assumption for V2:
    # We calculate the incidence in the COVID cohort.
    # We standardized it.
    # We compare it to... what? 
    # Maybe we are just profiling? 
    # "find a risk of IHD if you have COVID... then... multilogistic regression model."
    # Logistic Regression usually needs Cases (IHD) and Controls (No IHD).
    # In COVID pop (G1 vs G2), we can model IHD ~ Age + Gender + Comorbs.
    # In G1 vs G3 (COVID IHD vs Naive IHD), we can model "Is COVID associated with this IHD?" (Case-Case study?)
    
    # Let's implement the calculations we CAN do:
    # 1. Incidence in COVID cohort (G1 / Total COVID).
    # 2. Compare G1 vs G3 (Demographics, Comorbs).
    # 3. Logistic Regression within COVID cohort (Predictors of IHD).
    
    # A. Incidence Rate Calculation (Per 100,000 person-years)
    # --------------------------------------------------------
    # Denominator: Person-Time of COVID patients (G1 + G2).
    # G1 Time: Time to IHD (or 365 days if censored? No, they had event).
    # G2 Time: 365 days (or until death? We have death registry).
    
    logger.info("Calculating Incidence Rates in COVID Cohort...")
    
    # Filter COVID Cohort (G1 + G2)
    covid_cohort = df[df['group'].isin(['Group 1', 'Group 2'])].copy()
    
    # Calculate Follow-up Days
    # Start: covid_date
    # End: Min(IHD_Date, Death_Date, COVID+365)
    
    # Fill Dates
    covid_cohort['start_date'] = pd.to_datetime(covid_cohort['covid_date'])
    covid_cohort['end_date'] = covid_cohort['start_date'] + pd.Timedelta(days=365)
    
    # If IHD (G1), end is event date
    # Valid IHD date for G1 is 'discharge_date' (index event)
    # Check if we have discharge_date column merged correctly. 
    # In Step 2 we merged 'discharge_date'. Step 3 preserved it.
    covid_cohort['discharge_date'] = pd.to_datetime(covid_cohort['discharge_date'], errors='coerce')
    
    # Adjust end date for events
    mask_event = covid_cohort['group'] == 'Group 1'
    # Take the earlier of 365 days or Event Date
    # Note: Event date should be within 365 days by definition of G1.
    covid_cohort.loc[mask_event, 'end_date'] = covid_cohort.loc[mask_event, 'discharge_date']
    
    # Convert to Person-Years
    covid_cohort['person_days'] = (covid_cohort['end_date'] - covid_cohort['start_date']).dt.days
    # Clamp to 0-365 just in case
    covid_cohort['person_days'] = covid_cohort['person_days'].clip(lower=0, upper=365)
    
    total_pys = covid_cohort['person_days'].sum() / 365.25
    n_events = mask_event.sum()
    
    rate = (n_events / total_pys) * 100000
    
    logger.info(f"COVID Cohort Incidence Analysis:")
    logger.info(f"  Events: {n_events}")
    logger.info(f"  Person-Years: {total_pys:,.2f}")
    logger.info(f"  Rate: {rate:,.2f} per 100,000 PY")
    
    # Save Rate
    with open(os.path.join(results_dir, "incidence_rate_covid.txt"), "w") as f:
        f.write(f"Incidence Rate in COVID Cohort:\n")
        f.write(f"Events: {n_events}\n")
        f.write(f"Person-Years: {total_pys:,.2f}\n")
        f.write(f"Rate: {rate:,.2f} per 100,000 PY\n")

    # B. Age-Standardization & Risk Analysis (ASIR & SIR)
    # ---------------------------------------------------
    # User Request: comparison is needed.
    # 1. ASIR (Direct Standardization): Using Cohort or Fixed Standard Weights.
    # 2. SIR (Standardized Incidence Ratio): Comparison to General Population.
    
    logger.info("Running Standardization & Risk Analysis (SIR/ASIR)...")
    
    # 1. Prepare Age Groups
    # align with config bins if possible, but hardcoded for now to match Census config
    # Config bins: 0, 40, 50, 60, 70, 80
    bins = [0, 40, 50, 60, 70, 80, 100]
    labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80+']
    # Ensure age is numeric
    covid_cohort['age'] = pd.to_numeric(covid_cohort['age'], errors='coerce')
    covid_cohort['age_group'] = pd.cut(covid_cohort['age'], bins=bins, labels=labels, right=False)
    
    # 2. Get Observed Events in COVID Cohort (Group 1)
    # We want breakdown by Age Group
    obs_counts = covid_cohort[covid_cohort['group'] == 'Group 1'].groupby('age_group')['uin'].count()
    
    # 3. Get Person-Years in COVID Cohort
    pys_counts = covid_cohort.groupby('age_group')['person_days'].sum() / 365.25
    
    # 4. Reference Population (Non-COVID) Logic
    # -----------------------------------------
    # We use Census for Total Population.
    # Non-COVID Pop = Census - COVID Pop.
    # Non-COVID Events = Group 3 Events.
    # Non-COVID Rate = G3 / (Census - COVID).
    
    # Load Census
    census_cfg = config.get('population_denominator', {})
    age_dist = census_cfg.get('age_sex_distribution', {})
    
    # Map Census keys (0, 40...) to our labels (<40, 40-49...)
    # keys are integers.
    map_age_key = {
        0: '<40', 40: '40-49', 50: '50-59', 60: '60-69', 70: '70-79', 80: '80+'
    }
    
    # Parse Census into DataFrame
    census_rows = []
    for age_key, counts in age_dist.items():
        total_pop = counts.get('M', 0) + counts.get('F', 0)
        label = map_age_key.get(age_key, 'Unknown')
        census_rows.append({'age_group': label, 'census_pop': total_pop})
        
    df_census = pd.DataFrame(census_rows).set_index('age_group')
    
    # Get COVID Population Count by Age (Total Unique Patients in Cohort)
    covid_pop_counts = covid_cohort.groupby('age_group')['uin'].count()
    
    # Get Group 3 Events by Age (Naive IHD)
    # Note: df has ALL groups. We need to cut df again.
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    g3_events = df[df['group'] == 'Group 3'].groupby('age_group')['uin'].count()

    # --- Derivation Log Writer ---
    stats_log = []
    stats_log.append("RISK ANALYSIS DERIVATION LOG")
    stats_log.append("============================")
    stats_log.append(f"Total COVID Cohort PYs: {total_pys:,.2f}")
    stats_log.append(f"Total Observed Events (G1): {n_events}")
    stats_log.append("\n[STEP 1] Reference Rate Calculation (Non-COVID)")
    stats_log.append("Logic: Rate_Ref = G3_Events / (Census_Pop - COVID_Pop)")
    stats_log.append(f"{'Age_Group':<10} {'Census':<12} {'COVID_Pop':<10} {'Non-COVID':<10} {'G3_Events':<10} {'Observed_G1':<12} {'Ref_Rate_Per_100k':<20}")
    stats_log.append("-" * 92)

    expected_total = 0
    df_results = []
    
    for label in labels:
        census_n = df_census.loc[label, 'census_pop'] if label in df_census.index else 0
        covid_n = covid_pop_counts.get(label, 0)
        non_covid_n = max(0, census_n - covid_n)
        
        g3_n = g3_events.get(label, 0)
        
        # Observed
        observed = obs_counts.get(label, 0)
        
        # Rate per Person-Year. 
        # Assumption: Census is "Mid-Year Population". So Person-Years ~ Population * 1 Year.
        # This is a standard approximation.
        
        ref_rate = 0
        if non_covid_n > 0:
            ref_rate = g3_n / non_covid_n # Events per Person-Year (approx)
            
        ref_rate_100k = ref_rate * 100000
        
        # Calculate Expected for COVID Cohort
        # Expected = COVID_PYs * Ref_Rate
        covid_pys = pys_counts.get(label, 0)
        expected = covid_pys * ref_rate
        expected_total += expected
        
        stats_log.append(f"{label:<10} {census_n:<12} {covid_n:<10} {non_covid_n:<10} {g3_n:<10} {observed:<12} {ref_rate_100k:<20.2f}")
        
        df_results.append({
            'Age Group': label,
            'Observed (G1)': observed,
            'COVID PYs': covid_pys,
            'Expected (G1)': expected,
            'Ref Rate (Non-COVID)': ref_rate,
            'Non-COVID Pop': non_covid_n,
            'G3 Events': g3_n
        })

    stats_log.append("-" * 80)
    
    # 5. SIR Calculation
    # ------------------
    sir = n_events / expected_total if expected_total > 0 else 0
    
    # 95% CI for SIR
    # Byar's approximation or Exact Poisson
    # Lower = Obs * (1 - 1/(9obs) - 1.96/3sqrt(obs))^3
    # Upper = (Obs+1) * (1 - 1/(9(obs+1)) + 1.96/3sqrt(obs+1))^3
    # Simple approx: SIR +/- 1.96 * SE(SIR), where SE(SIR) = SIR / sqrt(Obs)
    
    lower_ci = 0
    upper_ci = 0
    if n_events > 0:
        factor = 1.96 * (sir / np.sqrt(n_events))
        lower_ci = max(0, sir - factor)
        upper_ci = sir + factor
    
    stats_log.append("\n[STEP 2] Standardized Incidence Ratio (SIR)")
    stats_log.append(f"Total Observed (G1): {n_events}")
    stats_log.append(f"Total Expected:      {expected_total:.2f}")
    stats_log.append(f"SIR Calculation:     {n_events} / {expected_total:.2f}")
    stats_log.append(f"SIR Result:          {sir:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})")
    
    if sir > 1:
        stats_log.append(f"interpretation: COVID-19 group has {sir:.2f}x higher risk of IHD than general population.")
    else:
        stats_log.append(f"interpretation: COVID-19 group has similar or lower risk ({sir:.2f}x).")

    # 6. ASIR (Direct Method)
    # -----------------------
    # ASIR = Sum(AgeSpecRate_i * StandardWeight_i)
    # Weights = Census Proportion
    
    total_census = df_census['census_pop'].sum()
    df_census['weight'] = df_census['census_pop'] / total_census
    
    asir_val = 0
    var_asir = 0 # Variance for CI
    
    stats_log.append("\n[STEP 3] Age-Standardized Incidence Rate (ASIR)")
    stats_log.append("Method: Direct Standardization using Census 2020 weights")
    stats_log.append(f"{'Age_Group':<10} {'COVID_Rate(100k)':<18} {'Non-COVID_Rate(100k)':<20} {'Weight':<10}")
    stats_log.append("-" * 70)
    
    asir_ref_val = 0
    var_asir_ref = 0
    
    for item in df_results:
        label = item['Age Group']
        pys = item['COVID PYs']
        obs = item['Observed (G1)']
        
        # Non-COVID Data
        ref_rate = item['Ref Rate (Non-COVID)']
        g3_n = item['G3 Events']
        non_covid_n = item['Non-COVID Pop']
        
        rate_raw = (obs / pys) if pys > 0 else 0
        weight = df_census.loc[label, 'weight'] if label in df_census.index else 0
        
        # Weighted Rates
        weighted_rate = rate_raw * weight
        asir_val += weighted_rate
        
        weighted_ref_rate = ref_rate * weight
        asir_ref_val += weighted_ref_rate
        
        # Variance (COVID) - Poisson approx: Var(Rate) = Events / PY^2
        if pys > 0:
            var_asir += (weight**2 * obs) / (pys**2)
            
        # Variance (Non-COVID) - Poisson approx: Var(Rate) = Events / Pop^2
        # Note: Non-COVID denominator is Population count, treating as Person-Years (1 year follow-up approx)
        if non_covid_n > 0:
            var_asir_ref += (weight**2 * g3_n) / (non_covid_n**2)
            
        stats_log.append(f"{label:<10} {rate_raw*100000:<18.2f} {ref_rate*100000:<20.2f} {weight:<10.4f}")
        
    # Final ASIR COVID
    asir_100k = asir_val * 100000
    se_asir = np.sqrt(var_asir) * 100000
    asir_lci = max(0, asir_100k - 1.96 * se_asir)
    asir_uci = asir_100k + 1.96 * se_asir
    
    # Final ASIR Non-COVID
    asir_ref_100k = asir_ref_val * 100000
    se_asir_ref = np.sqrt(var_asir_ref) * 100000
    asir_ref_lci = max(0, asir_ref_100k - 1.96 * se_asir_ref)
    asir_ref_uci = asir_ref_100k + 1.96 * se_asir_ref
    
    stats_log.append("-" * 70)
    stats_log.append(f"COVID Cohort ASIR:    {asir_100k:.2f} per 100,000 PY (95% CI: {asir_lci:.2f} - {asir_uci:.2f})")
    stats_log.append(f"Non-COVID Pop ASIR:   {asir_ref_100k:.2f} per 100,000 PY (95% CI: {asir_ref_lci:.2f} - {asir_ref_uci:.2f})")
    
    # Calculate Rate Ratio (ASIR COVID / ASIR Non-COVID)
    if asir_ref_100k > 0:
        rr = asir_100k / asir_ref_100k
        stats_log.append(f"Standardized Rate Ratio: {rr:.2f}")
    
    # Save Logs
    log_path = os.path.join(results_dir, "sir_derivation_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(stats_log))
    logger.info(f"Saved Derivation Log: {log_path}")
    
    # Save Results CSV
    res_df = pd.DataFrame(df_results)
    save_with_report(res_df, os.path.join(results_dir, "risk_analysis_details.csv"), "Breakdown of Observed vs Expected", logger)
    
    # Save Metrics
    metrics = pd.DataFrame({
        'Metric': ['Crude Rate (100k)', 'ASIR COVID (100k)', 'ASIR COVID LCI', 'ASIR COVID UCI',
                   'ASIR Non-COVID (100k)', 'ASIR Non-COVID LCI', 'ASIR Non-COVID UCI',
                   'SIR', 'SIR_LCI', 'SIR_UCI'],
        'Value': [rate, asir_100k, asir_lci, asir_uci,
                  asir_ref_100k, asir_ref_lci, asir_ref_uci,
                  sir, lower_ci, upper_ci]
    })
    save_with_report(metrics, os.path.join(results_dir, "final_risk_metrics.csv"), "Comparison Metrics", logger)

    rate_plot_path = os.path.join(results_dir, "age_specific_incidence_plot.png")
    
    # Actually PLOT the age-specific rates (Fix for blank plot)
    # Actually PLOT the age-specific comparison
    plt.figure(figsize=(12, 6))
    
    age_grps = [r['Age Group'] for r in df_results]
    
    covid_rates = []
    non_covid_rates = []
    
    for r in df_results:
        pys = r['COVID PYs']
        obs = r['Observed (G1)']
        rate_c = (obs / pys) * 100000 if pys > 0 else 0
        covid_rates.append(rate_c)
        
        # Non-COVID
        rate_nc = r['Ref Rate (Non-COVID)'] * 100000
        non_covid_rates.append(rate_nc)
        
    x = np.arange(len(age_grps))
    width = 0.35
    
    plt.bar(x - width/2, covid_rates, width, label='COVID Cohort', color='teal', alpha=0.8)
    plt.bar(x + width/2, non_covid_rates, width, label='Non-COVID Population', color='gray', alpha=0.6)
    
    plt.xlabel('Age Group')
    plt.ylabel('Incidence Rate per 100,000 PY')
    plt.title('Age-Specific IHD Incidence Rate: COVID vs Non-COVID')
    plt.xticks(x, age_grps)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(rate_plot_path)
    plt.close()
    
    # Visual 2: Observed vs Expected (Risk Analysis)
    obs_vals = [r['Observed (G1)'] for r in df_results]
    exp_vals = [r['Expected (G1)'] for r in df_results]
    
    plt.figure(figsize=(10, 6))
    x_idx = np.arange(len(age_grps))
    width = 0.35
    
    plt.bar(x_idx - width/2, obs_vals, width, label='Observed (COVID)', color='salmon')
    plt.bar(x_idx + width/2, exp_vals, width, label='Expected (Based on Non-COVID)', color='gray')
    
    plt.xlabel('Age Group')
    plt.ylabel('Number of Cases')
    plt.title('Observed vs Expected IHD Cases in COVID Cohort')
    plt.xticks(x_idx, age_grps)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "observed_vs_expected_plot.png"))
    plt.close()

    # C. Logistic Regression (Risk Factors) - SPLIT
    # ---------------------------------------------
    logger.info("Generating Table 1...")
    # (Table 1 generation code remains - omitted for brevity if unchanged, but included below for context)
    
    def get_summary(df_sub, label):
        summ = {'Group': label, 'N': len(df_sub)}
        summ['Age_Mean'] = df_sub['age'].mean()
        summ['Age_SD'] = df_sub['age'].std()
        summ['Male_%'] = (df_sub['gender'].isin(['M', 'Male', '1']).sum() / len(df_sub)) * 100 if len(df_sub) > 0 else 0
        comorb_cols = [c for c in df_sub.columns if c.startswith('Comorb_') and c.endswith('_Date')]
        for c in comorb_cols:
            clean_name = c.replace('Comorb_', '').replace('_Date', '')
            count = df_sub[c].notnull().sum()
            summ[f"{clean_name}_%"] = (count / len(df_sub)) * 100 if len(df_sub) > 0 else 0
        return summ
        
    table1_rows = []
    for g in sorted(df['group'].unique()):
        sub = df[df['group'] == g]
        table1_rows.append(get_summary(sub, g))
    df_table1 = pd.DataFrame(table1_rows)
    
    # Save CSV
    t1_save = df_table1.set_index('Group').T.reset_index()
    save_with_report(t1_save, os.path.join(results_dir, "table_1_baseline.csv"), 
                     "Table 1: Baseline Characteristics", logger)
                     
    # Save Table as Image
    plt.figure(figsize=(12, len(t1_save) * 0.5 + 2))
    ax = plt.gca()
    # Hide axes
    ax.axis('off') 
    
    # Prepare text table
    col_labels = t1_save.columns
    cell_text = []
    for row in t1_save.itertuples(index=False):
        # Format numeric to 1 decimal place
        formatted_row = []
        for val in row:
            if isinstance(val, (float, int)) and not isinstance(val, str):
                formatted_row.append(f"{val:.1f}")
            else:
                formatted_row.append(str(val))
        cell_text.append(formatted_row)
        
    table_plot = plt.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1, 1.5)
    
    plt.title("Table 1: Baseline Characteristics Baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "table_1_baseline.png"))
    plt.close()

    logger.info("Running Logistic Regression (Split Models)...")
    
    # Common Prep
    covid_cohort['outcome'] = (covid_cohort['group'] == 'Group 1').astype(int)
    # Ensure numeric
    covid_cohort['age'] = pd.to_numeric(covid_cohort['age'], errors='coerce')
    covid_cohort['gender_num'] = covid_cohort['gender'].map({'M': 1, 'F': 0})
    
    # Helper to Run Model
    def run_logit_model(model_name, feature_prefix, filename_suffix):
        logger.info(f"  Running Model: {model_name}...")
        
        # 1. Select Features
        cols = [c for c in covid_cohort.columns if c.startswith(feature_prefix) and c.endswith('_Date')]
        
        # 2. Binarize based on Timing (Timeline: History BEFORE COVID)
        # Because we are predicting IHD *after* COVID, based on Status *at* COVID.
        reg_df = covid_cohort[['uin', 'outcome', 'age', 'gender_num', 'covid_date']].copy()
        
        feature_names = []
        for col in cols:
            clean_name = col.replace(feature_prefix, '').replace('_Date', '')
            # Logic: Present AND Date <= COVID Date
            date_series = pd.to_datetime(covid_cohort[col], errors='coerce')
            covid_date = pd.to_datetime(covid_cohort['covid_date'], errors='coerce')
            
            # 0/1 Flag
            reg_df[clean_name] = ((date_series.notnull()) & (date_series <= covid_date)).astype(int)
            
            # Only add if at least some variance (otherwise regression fails)
            if reg_df[clean_name].sum() > 0 and reg_df[clean_name].sum() < len(reg_df):
                feature_names.append(clean_name)
            else:
                logger.warning(f"    Skipping {clean_name}: No variance (All 0 or All 1)")
        
        # 3. Formula
        predictors = ['age', 'gender_num'] + feature_names
        formula = f"outcome ~ {' + '.join(predictors)}"
        
        try:
            model = smf.logit(formula=formula, data=reg_df).fit(disp=0)
            
            # Save Summary
            with open(os.path.join(results_dir, f"logit_{filename_suffix}_summary.txt"), "w") as f:
                f.write(model.summary().as_text())
                
            # Extract ORs
            params = model.params
            conf = model.conf_int()
            conf['OR'] = params
            conf.columns = ['Lower CI', 'Upper CI', 'OR']
            odds_ratios = np.exp(conf)
            
            # Save CSV
            save_with_report(odds_ratios.reset_index(), os.path.join(results_dir, f"odds_ratios_{filename_suffix}.csv"), 
                             f"Odds Ratios ({model_name})", logger)
            
            # Forest Plot
            # -----------
            # 1. Drop Intercept
            plot_df = odds_ratios.drop('Intercept', errors='ignore')
            
            # 2. Drop Invalid Rows (Inf, NaN, or CI spanning Inf)
            #    Usually happens with Perfect Separation -> Huge SE -> Huge CI.
            valid_mask = (np.isfinite(plot_df['OR'])) & \
                         (np.isfinite(plot_df['Lower CI'])) & \
                         (np.isfinite(plot_df['Upper CI'])) & \
                         (plot_df['Upper CI'] < 1000) # Arbitrary cutoff for stability
            
            dropped = plot_df[~valid_mask]
            if not dropped.empty:
                logger.warning(f"    Dropped {len(dropped)} features from plot due to instability (Inf/NaN/Huge CI): {list(dropped.index)}")
                
            plot_df = plot_df[valid_mask]
            
            if not plot_df.empty:
                plt.figure(figsize=(10, max(4, len(plot_df)*0.5 + 2)))
                features = plot_df.index
                ors = plot_df['OR']
                errs = [ors - plot_df['Lower CI'], plot_df['Upper CI'] - ors]
                
                plt.errorbar(ors, range(len(features)), xerr=errs, fmt='o', color='black', capsize=5)
                plt.yticks(range(len(features)), features)
                plt.axvline(x=1, color='red', linestyle='--')
                plt.xlabel("Odds Ratio (95% CI)")
                plt.title(f"Predictors of Post-COVID IHD: {model_name}")
                plt.grid(axis='x', linestyle='--', alpha=0.5)
                
                # Check scale
                if plot_df['Upper CI'].max() > 10:
                     plt.xscale('log') # Use log scale if range is wide
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"forest_plot_{filename_suffix}.png"))
                plt.close()
            else:
                logger.warning(f"    No valid features to plot for {model_name} (All dropped or empty).")
                
        except Exception as e:
            logger.error(f"    Model {model_name} Failed: {e}")

    # Run 1: Comorbidities
    run_logit_model("Comorbidities", "Comorb_", "comorbs")
    
    # Run 2: Medications
    run_logit_model("Medications", "Med_", "meds")

    # D. Correlation Matrix (Split or Combined?) 
    # Let's do Combined for overview, or separate? 
    # Just keep combined for now as general EDA.
    logger.info("Generating Correlation Matrix...")
    feature_cols = [c for c in df.columns if c.startswith('Comorb_') or c.startswith('Med_')]
    bin_features = df[feature_cols].notnull().astype(int)
    bin_features.columns = [c.replace('Comorb_', '').replace('Med_', '').replace('_Date', '') for c in bin_features.columns]
    
    corr_mat = bin_features.corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_mat, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr_mat)), corr_mat.columns, rotation=90)
    plt.yticks(range(len(corr_mat)), corr_mat.columns)
    plt.title("Risk Factor Correlation Matrix (Combined)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "correlation_matrix.png"))
    plt.close()

    # D2. Comorbidity Comparison (Butterfly Plot)
    # -------------------------------------------
    logger.info("Generating Comorbidity Diverging Bar Chart (Butterfly Plot)...")
    
    # 1. Prepare Data
    # Groups: Group 1 (COVID -> IHD), Group 2 (COVID -> No IHD)
    groups = ['Group 1', 'Group 2']
    comorb_cols = [c for c in covid_cohort.columns if c.startswith('Comorb_') and c.endswith('_Date')]
    
    comparative_data = []
    
    # Calculate Prevalence
    g1_pop = covid_cohort[covid_cohort['group'] == 'Group 1']
    g2_pop = covid_cohort[covid_cohort['group'] == 'Group 2']
    
    n_g1 = len(g1_pop)
    n_g2 = len(g2_pop)
    
    if n_g1 > 0 and n_g2 > 0:
        for col in comorb_cols:
            clean_name = col.replace('Comorb_', '').replace('_Date', '').replace('_', ' ')
            
            # Count History (Date <= COVID Date) for each group
            # Note: We can reuse the `reg_df` logic or recalc. Let's recalc to be safe.
            
            # Group 1 Count
            dates_g1 = pd.to_datetime(g1_pop[col], errors='coerce')
            cov_dates_g1 = pd.to_datetime(g1_pop['covid_date'], errors='coerce')
            count_g1 = ((dates_g1.notnull()) & (dates_g1 <= cov_dates_g1)).sum()
            prev_g1 = (count_g1 / n_g1) * 100
            
            # Group 2 Count
            dates_g2 = pd.to_datetime(g2_pop[col], errors='coerce')
            cov_dates_g2 = pd.to_datetime(g2_pop['covid_date'], errors='coerce')
            count_g2 = ((dates_g2.notnull()) & (dates_g2 <= cov_dates_g2)).sum()
            prev_g2 = (count_g2 / n_g2) * 100
            
            comparative_data.append({
                'Condition': clean_name,
                'G1_Prev': prev_g1,
                'G2_Prev': prev_g2,
                'Diff': prev_g1 - prev_g2 # To sort
            })
            
        plot_df = pd.DataFrame(comparative_data)
        plot_df.sort_values('G1_Prev', inplace=True) # Sort by G1 prevalence
        
        # 2. Plotting
        # -----------
        plt.figure(figsize=(12, max(6, len(plot_df)*0.5)))
        ax = plt.gca()
        
        y_pos = np.arange(len(plot_df))
        
        # Group 2 (Left side, make negative)
        g2_bars = ax.barh(y_pos, -plot_df['G2_Prev'], align='center', color='skyblue', label=f'Group 2 (No IHD) N={n_g2:,}')
        # Group 1 (Right side)
        g1_bars = ax.barh(y_pos, plot_df['G1_Prev'], align='center', color='salmon', label=f'Group 1 (Post-COVID IHD) N={n_g1:,}')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['Condition'])
        ax.axvline(0, color='black', linewidth=0.8)
        
        # Fix X-axis labels to be positive relative to 0
        xticks = ax.get_xticks()
        # Ensure we don't set ticklabels without setting ticks if possible, but safe here relative to old code
        ax.set_xticklabels([f"{abs(x):.0f}%" for x in xticks])
        
        plt.xlabel("Prevalence (%)")
        plt.title("Comorbidity Profile Comparison: Group 1 vs Group 2")
        plt.legend(loc='lower right')
        
        # Annotate
        for bars, col_name in zip([g1_bars, g2_bars], ['G1_Prev', 'G2_Prev']):
            for bar, val in zip(bars, plot_df[col_name]):
                width = bar.get_width()
                # Determine position
                if width < 0: # Left side
                    x_loc = width - 1 
                    text_val = f"{val:.1f}%"
                    align = 'right'
                else: # Right side
                    x_loc = width + 1
                    text_val = f"{val:.1f}%"
                    align = 'left'
                    
                if val > 0.5: # Only label if visible
                    ax.text(x_loc, bar.get_y() + bar.get_height()/2, text_val, 
                            va='center', ha=align, fontsize=9, color='black')
        
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plot_path = os.path.join(results_dir, "comorbidity_butterfly_plot.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved Butterfly Plot: {plot_path}")
    else:
        logger.warning("Skipping Butterfly Plot: Insufficient data in Group 1 or Group 2.")

    # E. Mortality Analysis (KM Curves & Risk Table)
    # ----------------------------------------------
    logger.info("Running Mortality Analysis (Statistics & KM Curves)...")
    
    # 1. Define Correct Index Date for Fair Comparison
    # ------------------------------------------------
    # Group 1 (Post-COVID IHD) & Group 3 (Naive IHD): Index = IHD Diagnosis Date.
    # Group 2 (COVID Non-IHD): Index = COVID Date.
    
    df['discharge_date'] = pd.to_datetime(df['discharge_date'], errors='coerce')
    df['covid_date'] = pd.to_datetime(df['covid_date'], errors='coerce')
    df['death_date'] = pd.to_datetime(df['death_date'], errors='coerce')
    
    def get_analysis_index(row):
        if row['group'] in ['Group 1', 'Group 3']:
            return row['discharge_date']
        return row['covid_date']
        
    df['analysis_index_date'] = df.apply(get_analysis_index, axis=1)
    
    # Study End for Censoring (Adjust as needed, default 2023-12-31)
    study_end = pd.Timestamp(f"{config['study_period']['end_year']}-12-31")
    
    # Prepare Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Setup for "Number at Risk" Table
    # Times: 0, 1, 2, 3, 4 Years
    risk_times_yr = [0, 1, 2, 3, 4] 
    risk_times_days = [t * 365 for t in risk_times_yr]
    risk_table_data = []
    
    colors = {'Group 1': 'red', 'Group 3': 'blue', 'Group 2': 'green'}
    groups_to_plot = ['Group 1', 'Group 2', 'Group 3'] # Focus on IHD comparison as per context
    # Add Group 2 if desired, but scales might differ.
    
    for group in groups_to_plot:
        sub = df[df['group'] == group].copy()
        if sub.empty: continue
        
        # Calculate Time to Event
        # End Date = Min(Death, StudyEnd)
        # Note: If Death > StudyEnd, we censor at StudyEnd.
        # If Death < Index, data error or pre-existing? 
        sub['end_date_ac'] = sub['death_date'].fillna(study_end)
        
        # Duration in Days
        sub['duration'] = (sub['end_date_ac'] - sub['analysis_index_date']).dt.days
        
        # Event: 1 if Dead AND Death <= StudyEnd
        sub['event'] = (sub['death_date'].notnull()) & (sub['death_date'] <= study_end)
        sub['event'] = sub['event'].astype(int)
        
        # Filter Logic for Plotting (User Request: Fix "starts at 0.9")
        # If a patient dies on Day 0 (Index Date), survival drops immediately.
        # To show a curve starting at 1.0, we condition on "Surviving the Index Event Procedure".
        # We exclude duration <= 0 strictly for the plot, OR we handle t=0 explicitly.
        # Decision: Filter duration > 0 for KM Plot lines to look "clean".
        # But for 'At Risk' counts, we count them at T=0.
        
        # Valid Analysis Set (Non-negative time)
        sub = sub[sub['duration'] >= 0].copy()
        
        # --- Number at Risk Calculation ---
        at_risk_counts = []
        for t_day in risk_times_days:
            # At risk if Duration >= t_day
            n = len(sub[sub['duration'] >= t_day])
            at_risk_counts.append(n)
        risk_table_data.append(at_risk_counts + [group]) # Append Group Name for row Label later
        
        # --- Manual Kaplan-Meier with 95% CI (Greenwood) ---
        sub.sort_values('duration', inplace=True)
        
        times = np.sort(sub['duration'].unique())
        # Filter max 4 years for plot (4 * 365 = 1460 days)
        max_days = 4 * 365
        times = times[times <= max_days]
        
        surv_probs = []
        ci_lower = []
        ci_upper = []
        
        n_at_risk = len(sub)
        current_km = 1.0 # S(t)
        current_var = 0.0 # Sum(d_i / (n_i * (n_i - d_i)))
        
        # Initialize at t=0
        surv_probs.append(1.0)
        ci_lower.append(1.0)
        ci_upper.append(1.0)
        plot_times = [0]
        
        for t in times:
            if t == 0: continue # Already handled initialization
            
            # Events and Censored at exact time t
            n_events = len(sub[(sub['duration'] == t) & (sub['event'] == 1)])
            n_censored = len(sub[(sub['duration'] == t) & (sub['event'] == 0)])
            n_dropped = n_events + n_censored
            
            if n_at_risk > 0:
                # Update Survival
                # S(t) = S(t-1) * (1 - d_i / n_i)
                p_i = 1 - (n_events / n_at_risk)
                current_km *= p_i
                
                # Update Variance (Greenwood)
                # Var(S(t)) = S(t)^2 * Sum(...)
                if n_events > 0 and (n_at_risk - n_events) > 0:
                    term = n_events / (n_at_risk * (n_at_risk - n_events))
                    current_var += term
                
                # SE
                se_s = current_km * np.sqrt(current_var)
                
                # 95% CI (1.96 * SE)
                low = max(0, current_km - 1.96 * se_s)
                high = min(1.0, current_km + 1.96 * se_s)
                
                surv_probs.append(current_km)
                ci_lower.append(low)
                ci_upper.append(high)
                plot_times.append(t)
                
            n_at_risk -= n_dropped
            if n_at_risk <= 0: break
            
        # Plot
        # Convert Days to Years
        plot_times_yr = np.array(plot_times) / 365.0
        
        # Step Plot for Survival
        ax.step(plot_times_yr, surv_probs, where='post', label=group, color=colors.get(group, 'black'))
        
        # Shade CI
        ax.fill_between(plot_times_yr, ci_lower, ci_upper, alpha=0.2, step='post', color=colors.get(group, 'black'))

    # Formatting Plot
    ax.set_title("Kaplan-Meier Survival Curves (Post-Index Event)")
    ax.set_xlabel("Years from Index Date")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0.5, 1.05)
    ax.set_xlim(0, 4)
    ax.grid(linestyle='--', alpha=0.7)
    ax.legend(loc="lower left")
    
    # Add Risk Table
    # Rows: Groups, Cols: Years 0,1,2,3,4
    # risk_table_data is [[N0, N1, N2, N3, N4, Name], ...]
    if risk_table_data:
        table_rows = [x[:-1] for x in risk_table_data]
        row_labels = [x[-1] for x in risk_table_data]
        col_labels = [str(x) for x in risk_times_yr]
        
        # Adjust table position
        plt.subplots_adjust(bottom=0.25)
        the_table = plt.table(cellText=table_rows,
                              rowLabels=row_labels,
                              colLabels=col_labels,
                              loc='bottom',
                              bbox=[0.0, -0.35, 1.0, 0.25],
                              cellLoc='center')
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
    
    plt.savefig(os.path.join(results_dir, "km_survival_curves_fixed.png"), bbox_inches='tight')
    plt.close()
    
    # 2. 1-3-5 Year Windows (Existing Logic, slightly updated)
    # --------------------------------------------------------
    windows = config['definitions'].get('mortality_windows', [1, 23])
    mortality_results = []
    
    for group in ['Group 1', 'Group 3']: 
        sub = df[df['group'] == group].copy()
        if sub.empty: continue
        
        sub['end_date_ac'] = sub['death_date'].fillna(study_end)
        sub['duration'] = (sub['end_date_ac'] - sub['analysis_index_date']).dt.days
        n_total = len(sub)
        
        for yr in windows:
            days = yr * 365
            # Dead within X days
            n_dead = ((sub['death_date'].notnull()) & (sub['duration'] <= days)).sum()
            rate = (n_dead / n_total) * 100
            
            mortality_results.append({
                'Group': group,
                'Window': f"{yr}-Year",
                'Rate_Percent': rate,
                'Deaths': n_dead,
                'Total': n_total
            })
            
    mort_df = pd.DataFrame(mortality_results)
    if not mort_df.empty:
        save_with_report(mort_df, os.path.join(results_dir, "mortality_rates_fixed.csv"), "Mortality Rates by Group", logger)

    logger.info("Step 4 Analysis Complete.")
    return df

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    run_step_4(conf)
