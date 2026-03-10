
import os
import sys
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

# Add local directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logger, save_with_report, ensure_dir

def run_step_5(config):
    # 1. Setup
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_5_variant_analysis")
    ensure_dir(results_dir)
    
    logger = setup_logger("step_5", results_dir)
    logger.info("Starting Step 5: Variant Era Analysis")
    
    # 2. Load Enriched Cohort
    input_file = os.path.join(processed_dir, "step_3_features", "cohort_enriched.csv")
    if not os.path.exists(input_file):
        logger.error(f"Missing input: {input_file}. Run Step 3 first.")
        return
        
    df = pd.read_csv(input_file)
    logger.info(f"Loaded Cohort: {len(df):,} patients.")
    
    # 3. Define Variant Eras
    # ----------------------
    # Ancestral: < 2021-05-01
    # Delta: 2021-05-01 to 2021-12-31
    # Omicron: >= 2022-01-01
    
    # Focus on COVID Patients (Group 1 & Group 2)
    covid_df = df[df['group'].isin(['Group 1', 'Group 2'])].copy()
    covid_df['covid_date'] = pd.to_datetime(covid_df['covid_date'])
    
    def assign_era(date):
        if pd.isnull(date): return 'Unknown'
        if date < pd.Timestamp('2021-05-01'):
            return 'Ancestral'
        elif date < pd.Timestamp('2022-01-01'):
            return 'Delta'
        else:
            return 'Omicron'
            
    covid_df['variant_era'] = covid_df['covid_date'].apply(assign_era)
    
    # Log Distribution
    era_counts = covid_df['variant_era'].value_counts()
    logger.info("Variant Era Distribution (COVID Patients):")
    for era, count in era_counts.items():
        logger.info(f"  {era}: {count:,}")
        
    # Ordered Eras
    eras_order = ['Ancestral', 'Delta', 'Omicron']
    covid_df['variant_era'] = pd.Categorical(covid_df['variant_era'], categories=eras_order, ordered=True)
    
    # 3.1 Descriptive Visualizations (New)
    # ------------------------------------
    logger.info("Generating Descriptive Visualizations...")
    
    # A. COVID Case Timeline
    plt.figure(figsize=(10, 6))
    # Resample to monthly counts
    timeline = covid_df.set_index('covid_date').resample('M').size()
    # Or better: Stacked histogram by Era
    # We can just plot histogram of dates
    
    # Partition data by Era
    data_ancestral = covid_df[covid_df['variant_era'] == 'Ancestral']['covid_date']
    data_delta = covid_df[covid_df['variant_era'] == 'Delta']['covid_date']
    data_omicron = covid_df[covid_df['variant_era'] == 'Omicron']['covid_date']
    
    plt.hist([data_ancestral, data_delta, data_omicron], bins=30, stacked=True, 
             color=['#1f77b4', '#ff7f0e', '#2ca02c'], label=eras_order)
    
    plt.title("Distribution of COVID Cases by Variant Era")
    plt.xlabel("Date of Infection")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "covid_cases_timeline.png"))
    plt.close()
    
    # B. Demographics by Era
    # Age Distribution (Boxplot)
    plt.figure(figsize=(8, 6))
    data_age = [covid_df[covid_df['variant_era'] == e]['age'].dropna() for e in eras_order]
    
    plt.boxplot(data_age, labels=eras_order, patch_artist=True, 
                boxprops=dict(facecolor='lightblue'))
    plt.title("Age Distribution by Variant Era")
    plt.ylabel("Age (Years)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(results_dir, "demographics_age_boxplot.png"))
    plt.close()
    
    # C. Comorbidity Heatmap
    # Calculate prevalence of key comorbs per Era
    comorbs_plot = ['Diabetes_Uncomplicated', 'Hypertension', 'Hyperlipidemia', 'Renal_Disease', 'Congestive_Heart_Failure', 'Myocardial_Infarction']
    
    heatmap_data = []
    for era in eras_order:
        sub = covid_df[covid_df['variant_era'] == era]
        row = {'Era': era}
        for c in comorbs_plot:
            # Find col
            matches = [x for x in covid_df.columns if c in x and x.endswith('_Date')]
            if matches:
                col = matches[0]
                # Check history
                d_s = pd.to_datetime(sub[col], errors='coerce')
                # Count if date <= covid_date (using pre-calc loop logic would be cleaner, but re-doing for safety)
                # But wait, we didn't pre-calc all of them in the main df yet, only in the loop later.
                # Let's do it quick here.
                c_dates = pd.to_datetime(sub['covid_date'], errors='coerce')
                is_hist = ((d_s.notnull()) & (d_s <= c_dates))
                prev = (is_hist.sum() / len(sub)) * 100 if len(sub) > 0 else 0
                row[c] = prev
            else:
                row[c] = 0
        heatmap_data.append(row)
        
    hm_df = pd.DataFrame(heatmap_data).set_index('Era')
    
    # Plot Heatmap
    plt.figure(figsize=(10, 5))
    plt.imshow(hm_df.values, cmap='YlOrRd', aspect='auto')
    
    # Annotate
    for i in range(len(eras_order)):
        for j in range(len(comorbs_plot)):
            val = hm_df.iloc[i, j]
            color = 'white' if val > 50 else 'black'
            plt.text(j, i, f"{val:.1f}%", ha='center', va='center', color=color)
            
    plt.xticks(range(len(comorbs_plot)), labels=[x.replace('_',' ') for x in comorbs_plot], rotation=45, ha='right')
    plt.yticks(range(len(eras_order)), labels=eras_order)
    plt.title("Comorbidity Prevalence by Era (%)")
    plt.colorbar(label='Prevalence %')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comorbidity_heatmap.png"))
    plt.close()
    # ---------------------------------------------------
    # Objective: Compare rate of IHD (Group 1) across Eras.
    # Metrics: Incidence Rate (per 100k PY), Adjusted HR.
    # Fairness: Censor at 1 Year for EVERYONE (since Omicron has shorter follow-up, and G1 def is 1 year).
    
    logger.info("Running Incidence Analysis...")
    
    # Setup Time-to-Event
    # Start: COVID Date
    # End: Min(IHD Date, Death Date, COVID+365)
    # Event: IHD (Group 1 status within 365 days)
    
    covid_df['start_date'] = covid_df['covid_date']
    covid_df['end_date_1y'] = covid_df['start_date'] + pd.Timedelta(days=365)
    
    # For G1, event date is Discharge Date
    covid_df['discharge_date'] = pd.to_datetime(covid_df['discharge_date'], errors='coerce')
    
    # Define Event and Duration
    # Default end: MIN(Death, End_1y, Study_End)
    # Note: If patient died within 1 year without IHD -> Censored.
    # If patient had IHD -> Event.
    
    covid_df['death_date'] = pd.to_datetime(covid_df['death_date'], errors='coerce')
    
    # Determine efficient end date for person-years
    # Init with 365 days out
    covid_df['stop_date'] = covid_df['end_date_1y']
    
    # If died earlier, censor at death
    mask_dead = (covid_df['death_date'].notnull()) & (covid_df['death_date'] < covid_df['stop_date'])
    covid_df.loc[mask_dead, 'stop_date'] = covid_df.loc[mask_dead, 'death_date']
    
    # If IHD Event (Group 1), stop at event
    # Only if IHD date is valid and within window (which fits G1 def)
    mask_ihd = (covid_df['group'] == 'Group 1') & (covid_df['discharge_date'].notnull())
    # Ensure IHD is before the current stop date (death/365)
    # If IHD happens, that's the event.
    covid_df.loc[mask_ihd, 'stop_date'] = covid_df.loc[mask_ihd, 'discharge_date']
    
    # Calculate Days
    covid_df['person_days'] = (covid_df['stop_date'] - covid_df['start_date']).dt.days
    # Clip to 0 (data errors) and 365
    covid_df['person_days'] = covid_df['person_days'].clip(lower=0, upper=365)
    
    # Event Flag
    covid_df['ihd_event'] = 0
    covid_df.loc[mask_ihd, 'ihd_event'] = 1
    
    # A. Crude Rates Plot
    # -------------------
    inc_results = []
    
    for era in eras_order:
        sub = covid_df[covid_df['variant_era'] == era]
        events = sub['ihd_event'].sum()
        pys = sub['person_days'].sum() / 365.25
        rate = (events / pys) * 100000 if pys > 0 else 0
        
        # CI for Rate (poisson exact approx)
        se = rate / np.sqrt(events) if events > 0 else 0
        
        inc_results.append({
            'Era': era,
            'Events': events,
            'PYs': pys,
            'Rate_100k': rate,
            'SE': se
        })
        
    inc_df = pd.DataFrame(inc_results)
    save_with_report(inc_df, os.path.join(results_dir, "incidence_rates_by_era.csv"), "Incidence Rates by Variant Era", logger)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(inc_df['Era'], inc_df['Rate_100k'], yerr=1.96*inc_df['SE'], capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    plt.title("IHD Incidence Rate by COVID Variant Era (1-Year Window)")
    plt.ylabel("Incidence per 100,000 Person-Years")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(results_dir, "incidence_rate_by_era_plot.png"))
    plt.close()
    
    # A2. ASIR and SIR by Era (Standardized)
    # -------------------------------------
    logger.info("Running Standardization (ASIR & SIR) by Era...")
    
    # 1. Load Census (Same as Step 4)
    census_cfg = config.get('population_denominator', {})
    age_dist = census_cfg.get('age_sex_distribution', {})
    map_age_key = {0: '<40', 40: '40-49', 50: '50-59', 60: '60-69', 70: '70-79', 80: '80+'}
    bins = [0, 40, 50, 60, 70, 80, 100]
    labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80+']
    
    # Parse Census
    census_rows = []
    for age_key, counts in age_dist.items():
        total_pop = counts.get('M', 0) + counts.get('F', 0)
        label = map_age_key.get(age_key, 'Unknown')
        census_rows.append({'age_group': label, 'census_pop': total_pop})
        
    if not census_rows:
        logger.warning("No Census Data found in Config (population_denominator). Skipping ASIR/SIR.")
        df_census = pd.DataFrame()
    else:
        df_census = pd.DataFrame(census_rows).set_index('age_group')
    
    if df_census.empty:
         # Skip ASIR/SIR section but don't crash
         inc_results_std = [] # Empty for safety
    else:
        # Weights for ASIR
        total_census = df_census['census_pop'].sum()
        df_census['weight'] = df_census['census_pop'] / total_census
        
        # 2. Reference Rates (Non-COVID) from Group 3
        # Need to re-calculate Ref Rate per Age Group from Whole Cohort (Step 3 file has all groups)
        # Since we loaded filtering only G1/G2 in line 41, we need to go back to full df for G3
        
        g3_df = df[df['group'] == 'Group 3'].copy()
        g3_df['age'] = pd.to_numeric(g3_df['age'], errors='coerce')
        g3_df['age_group'] = pd.cut(g3_df['age'], bins=bins, labels=labels, right=False)
        g3_events = g3_df.groupby('age_group')['uin'].count()
        
        # Covid Total Pop by Age (for denominator subtraction)
        # Note: "Covid Pop" changes over time? No, Step 4 logic subtracts TOTAL unique Covid patients from Census.
        # We will stick to that approximations.
        covid_all_unique = df[df['group'].isin(['Group 1', 'Group 2'])].copy()
        covid_all_unique['age'] = pd.to_numeric(covid_all_unique['age'], errors='coerce')
        covid_all_unique['age_group'] = pd.cut(covid_all_unique['age'], bins=bins, labels=labels, right=False)
        covid_pop_counts = covid_all_unique.groupby('age_group')['uin'].count()
        
        ref_rates = {} # Rate per person-year
        for label in labels:
            census_n = df_census.loc[label, 'census_pop'] if label in df_census.index else 0
            covid_n = covid_pop_counts.get(label, 0)
            denom_n = max(0, census_n - covid_n)
            exp_n = g3_events.get(label, 0)
            rate = (exp_n / denom_n) if denom_n > 0 else 0
            ref_rates[label] = rate
            
        # 3. Calculate ASIR & SIR per Era
        # We need Age Group for our covid_df (Incidence Cohort)
        covid_df['age_group'] = pd.cut(covid_df['age'], bins=bins, labels=labels, right=False)
        
        std_results = []
        
        for era in eras_order:
            sub = covid_df[covid_df['variant_era'] == era].copy()
            
            # SIR Prep
            obs_total = sub['ihd_event'].sum()
            pys_by_age = sub.groupby('age_group')['person_days'].sum() / 365.25
            exp_total = 0
            
            # ASIR Prep
            asir_weighted_sum = 0
            asir_var_sum = 0
            
            for label in labels:
                pys = pys_by_age.get(label, 0)
                
                # SIR Expected
                r_ref = ref_rates.get(label, 0)
                exp_total += (pys * r_ref)
                
                # ASIR (Direct)
                # Obs rate in Era-Age
                obs_in_cell = sub[(sub['age_group'] == label)]['ihd_event'].sum()
                rate_cell = (obs_in_cell / pys) if pys > 0 else 0
                
                w = df_census.loc[label, 'weight'] if label in df_census.index else 0
                asir_weighted_sum += (rate_cell * w)
                if pys > 0:
                    asir_var_sum += (w**2 * obs_in_cell) / (pys**2)
                    
            # Final SIR
            sir = obs_total / exp_total if exp_total > 0 else 0
            # SIR CI
            sir_l, sir_u = 0, 0
            if obs_total > 0:
                 factor = 1.96 * (sir / np.sqrt(obs_total))
                 sir_l = max(0, sir - factor)
                 sir_u = sir + factor
                 
            # Final ASIR
            asir_100k = asir_weighted_sum * 100000
            asir_se = np.sqrt(asir_var_sum) * 100000
            asir_l = max(0, asir_100k - 1.96 * asir_se)
            asir_u = asir_100k + 1.96 * asir_se
            
            std_results.append({
                'Era': era,
                'Observed': obs_total,
                'Expected': exp_total,
                'SIR': sir,
                'SIR_LCI': sir_l,
                'SIR_UCI': sir_u,
                'ASIR_100k': asir_100k,
                'ASIR_LCI': asir_l,
                'ASIR_UCI': asir_u
            })
            
        std_df = pd.DataFrame(std_results)
        save_with_report(std_df, os.path.join(results_dir, "standardized_rates_by_era.csv"), "ASIR and SIR by Era", logger)
        
        # Comparison Statistics (Rate Ratios & Significance)
        # Compare Delta vs Ancestral, Omicron vs Ancestral
        stats_rows = []
        
        # Get Baseline (Ancestral)
        base = std_df[std_df['Era'] == 'Ancestral'].iloc[0] if not std_df[std_df['Era'] == 'Ancestral'].empty else None
        
        if base is not None:
            for i, row in std_df.iterrows():
                if row['Era'] == 'Ancestral': continue
                
                # RR for ASIR
                rr_asir = row['ASIR_100k'] / base['ASIR_100k'] if base['ASIR_100k'] > 0 else np.nan
                
                # Significance (Z-test for Rates)
                # Z = (R1 - R2) / sqrt(SE1^2 + SE2^2)
                # Est SE from CI: SE = (UCI - LCI) / (2*1.96)
                se1 = (base['ASIR_UCI'] - base['ASIR_LCI']) / 3.92
                se2 = (row['ASIR_UCI'] - row['ASIR_LCI']) / 3.92
                
                z_score = (row['ASIR_100k'] - base['ASIR_100k']) / np.sqrt(se1**2 + se2**2) if (se1+se2) > 0 else 0
                
                # Use scipy.stats.norm for CDF
                from scipy.stats import norm
                p_val = 2 * (1 - norm.cdf(abs(z_score))) # Two-tailed
                
                stats_rows.append({
                    'Comparison': f"{row['Era']} vs Ancestral",
                    'ASIR_Rate_Ratio': rr_asir,
                    'Z_Score': z_score,
                    'P_Value': p_val,
                    'Significant': p_val < 0.05
                })
        
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            save_with_report(stats_df, os.path.join(results_dir, "era_comparison_stats.csv"), "Significance of Rate Differences", logger)

        # Plot ASIR
        plt.figure(figsize=(8, 6))
        plt.bar(std_df['Era'], std_df['ASIR_100k'], 
                yerr=[std_df['ASIR_100k']-std_df['ASIR_LCI'], std_df['ASIR_UCI']-std_df['ASIR_100k']], 
                capsize=5, color='purple', alpha=0.7)
        plt.title("Age-Standardized Incidence Rate (ASIR) by Era")
        plt.ylabel("ASIR per 100,000 Person-Years")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "asir_by_era_plot.png"))
        plt.close()
        
        # Plot SIR
        plt.figure(figsize=(8, 6))
        plt.errorbar(std_df['Era'], std_df['SIR'], 
                     yerr=[std_df['SIR']-std_df['SIR_LCI'], std_df['SIR_UCI']-std_df['SIR_LCI']],
                     fmt='o', color='darkred', capsize=5)
        plt.axhline(1, color='gray', linestyle='--')
        plt.title("Standardized Incidence Ratio (SIR) by Era (Ref: Non-COVID)")
        plt.ylabel("SIR (Observed / Expected)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "sir_by_era_plot.png"))
        plt.close()
    
    # B. Adjusted Cox/Poisson Model
    # -----------------------------
    # Use Cox (PHReg) or Poisson?
    # Given we have time-to-event, Cox is better.
    # Model: Hazard(IHD) ~ Era + Age + Gender + Comorbidities
    
    logger.info("Running Adjusted Cox Model for Incidence...")
    
    # Prepare Vars
    covid_df['age'] = pd.to_numeric(covid_df['age'], errors='coerce')
    covid_df['gender_male'] = (covid_df['gender'].isin(['M', 'Male', '1'])).astype(int)
    
    # Select Comorbidities (Top 5-6 relevant ones)
    comorbs = ['Diabetes_Uncomplicated', 'Hypertension', 'Hyperlipidemia', 'Renal_Disease', 'Congestive_Heart_Failure']
    model_comorbs = []
    for c in comorbs:
        col_name = f"Comorb_{c}_Date"
        # Check if column exists
        matches = [x for x in covid_df.columns if c in x and x.endswith('_Date')]
        if matches:
            actual_col = matches[0] # Use first match
            clean_name = c
            # Define binary (History before COVID)
            d_series = pd.to_datetime(covid_df[actual_col], errors='coerce')
            covid_df[clean_name] = ((d_series.notnull()) & (d_series <= covid_df['start_date'])).astype(int)
            if covid_df[clean_name].sum() > 10: # Min count
                model_comorbs.append(clean_name)
    
    # Formula
    # Treat era as categorical, Ref=Ancestral (automatic if ordered categorical?)
    # Statsmodels handles categorical with C()
    
    predictors = [
        "C(variant_era, Treatment(reference='Ancestral'))",
        "age",
        "gender_male"
    ] + model_comorbs
    
    formula = f"person_days ~ {' + '.join(predictors)}"
    
    # Note: statsmodels PHReg syntax is specific.
    # We need status and time.
    # smf.phreg(formula, data, status=...)
    
    try:
        # Filter for valid data
        reg_df = covid_df.dropna(subset=['person_days', 'ihd_event', 'age', 'variant_era']).copy()
        
        # PHReg
        mod = smf.phreg(
            formula,
            reg_df,
            status=reg_df['ihd_event']
        )
        res = mod.fit()
        
        # Save Summary
        with open(os.path.join(results_dir, "incidence_cox_summary.txt"), "w") as f:
            f.write(res.summary().as_text())
            
        # Extract HRs for Plot
        params = res.params
        conf = res.conf_int()
        conf['HR'] = np.exp(params)
        conf.columns = ['Lower', 'Upper', 'HR']
        
        # Filter for Era coefficients
        era_rows = conf[conf.index.str.contains("variant_era")]
        
        if not era_rows.empty:
            save_with_report(era_rows.reset_index(), os.path.join(results_dir, "era_hazard_ratios.csv"), "Adjusted Hazard Ratios (Ref: Ancestral)", logger)
            
            # Forest Plot
            plt.figure(figsize=(8, 4))
            # Clean names
            names = [x.split('[')[1].split(']')[0].replace("T.", "") for x in era_rows.index]
            
            plt.errorbar(era_rows['HR'], range(len(names)), 
                         xerr=[era_rows['HR'] - era_rows['Lower'], era_rows['Upper'] - era_rows['HR']], 
                         fmt='o', capsize=5, color='darkred')
            
            plt.yticks(range(len(names)), names)
            plt.axvline(1, color='gray', linestyle='--')
            plt.xlabel("Adjusted Hazard Ratio (95% CI)")
            plt.title("Relative Risk of IHD vs Ancestral Era")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "incidence_hr_forest_plot.png"))
            plt.close()
            
    except Exception as e:
        logger.error(f"Failed to run Cox Model: {e}")
        # import traceback
        # logger.error(traceback.format_exc())

    # 5. Outcomes Analysis (Mortality in Post-COVID IHD)
    # --------------------------------------------------
    # Population: Group 1 Only
    # Event: Death (All Cause)
    # Stratify by Era
    
    logger.info("Running Outcomes Analysis (Mortality)...")
    
    g1_df = covid_df[covid_df['group'] == 'Group 1'].copy()
    
    if g1_df.empty:
        logger.warning("No Group 1 patients found. Skipping Outcomes Analysis.")
        return

    # Time logic changes:
    # Start: IHD Date (discharge_date)
    # End: Death or Study End
    # Note: Era is defined by COVID date, which is correct (Era of Infection).
    
    # Recalculate duration from IHD event
    study_end = pd.Timestamp('2023-12-31')
    
    g1_df['outcome_start'] = g1_df['discharge_date']
    g1_df['outcome_end'] = study_end
    
    # Check Dead
    mask_dead_g1 = (g1_df['death_date'].notnull()) & (g1_df['death_date'] <= study_end)
    
    # Set Stop Date
    g1_df.loc[mask_dead_g1, 'outcome_end'] = g1_df.loc[mask_dead_g1, 'death_date']
    
    # Check for consistency (Death before IHD? Should not happen in G1 def, but safety check)
    g1_df = g1_df[g1_df['outcome_end'] >= g1_df['outcome_start']]
    
    g1_df['surv_days'] = (g1_df['outcome_end'] - g1_df['outcome_start']).dt.days
    g1_df['dead_event'] = mask_dead_g1.astype(int)
    
    # 1-Year Mortality Rate by Era
    # (Dead within 365 days of IHD Analysis)
    
    # Adjust for Censoring:
    # If censored < 365 days and NOT dead, we can't definitively say they survived 1 year?
    # But usually we just do KM estimate at t=365.
    
    # Let's do KM Curves per Era
    
    plt.figure(figsize=(10, 6))
    colors_map = {'Ancestral': 'blue', 'Delta': 'orange', 'Omicron': 'green'}
    
    for era in eras_order:
        sub = g1_df[g1_df['variant_era'] == era].copy()
        if sub.empty: continue
        
        sub.sort_values('surv_days', inplace=True)
        times = np.sort(sub['surv_days'].unique())
        
        # Simple manual KM (reuse logic or use lib if available, stick to manual for consistency with Step 4)
        surv_probs = [1.0]
        plot_times = [0]
        at_risk = len(sub)
        curr_s = 1.0
        
        for t in times:
            if t > 365 * 2: break # Limit plot to 2 years for clarity (Omicron is short)
            
            n_events = len(sub[(sub['surv_days'] == t) & (sub['dead_event'] == 1)])
            n_censored = len(sub[(sub['surv_days'] == t) & (sub['dead_event'] == 0)])
            n_dropped = n_events + n_censored
            
            if at_risk > 0:
                p_i = 1 - (n_events / at_risk)
                curr_s *= p_i
                surv_probs.append(curr_s)
                plot_times.append(t)
            
            at_risk -= n_dropped
            
        plt.step(plot_times, surv_probs, where='post', label=f"{era} (N={len(sub)})", color=colors_map.get(era, 'gray'))
        
    plt.title("Kaplan-Meier Survival after Post-COVID IHD (by Era)")
    plt.xlabel("Days since IHD Diagnosis")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(results_dir, "mortality_km_by_era.png"))
    plt.close()

    # 5.B. Mortality Cox Model (Adjusted HR for Death)
    # ------------------------------------------------
    # Model: Death ~ Era + Age + Gender + Comorbidities
    # Population: Group 1 IHD Survivors (Time from Discharge to Death)
    
    logger.info("Running Mortality Cox Model (Adjusted)...")
    
    try:
        # Prepare Reg DF
        g1_reg = g1_df.copy()
        
        # Check size
        if len(g1_reg) < 10 or g1_reg['dead_event'].sum() < 2:
            logger.warning(f"Skipping Mortality Cox Model: Insufficient data (N={len(g1_reg)}, Events={g1_reg['dead_event'].sum()})")
            raise ValueError("Insufficient Data")
        
        # Ensure vars are ready (Age, Gender already in)
        # Add comorbs (History <= Covid date is fine, or History <= Discharge Date?
        # Usually baseline at Index (Discharge).
        # Reuse comorbs list but re-check timeline relative to DISCHARGE for G1
        
        model_comorbs_g1 = []
        for c in comorbs:
            col_name = f"Comorb_{c}_Date"
            matches = [x for x in g1_reg.columns if c in x and x.endswith('_Date')]
            if matches:
                 col = matches[0]
                 clean_name = c + "_G1" # Rename to avoid conflict if merging
                 d_s = pd.to_datetime(g1_reg[col], errors='coerce')
                 # Condition: History BEFORE Discharge (Index)
                 # Note: Discharge date is outcome_start
                 g1_reg[clean_name] = ((d_s.notnull()) & (d_s <= g1_reg['outcome_start'])).astype(int)
                 if g1_reg[clean_name].sum() > 5: # Lower threshold for smaller G1
                     model_comorbs_g1.append(clean_name)

        # Formula
        # Ref: Ancestral
        predictors_g1 = [
            "C(variant_era, Treatment(reference='Ancestral'))",
            "age",
            "gender_male"
        ] + model_comorbs_g1
        
        formula_g1 = f"surv_days ~ {' + '.join(predictors_g1)}"
        
        mod_g1 = smf.phreg(
            formula_g1,
            g1_reg,
            status=g1_reg['dead_event']
        )
        res_g1 = mod_g1.fit()
        
        # Save Summary
        with open(os.path.join(results_dir, "mortality_cox_summary.txt"), "w") as f:
            f.write(res_g1.summary().as_text())
            
        # Forest Plot (Mortality)
        conf_g1 = res_g1.conf_int()
        conf_g1['HR'] = np.exp(res_g1.params)
        conf_g1.columns = ['Lower', 'Upper', 'HR']
        
        era_rows_g1 = conf_g1[conf_g1.index.str.contains("variant_era")]
        
        if not era_rows_g1.empty:
            save_with_report(era_rows_g1.reset_index(), os.path.join(results_dir, "mortality_era_hazard_ratios.csv"), "Mortality Hazard Ratios (Ref: Ancestral)", logger)
            
            plt.figure(figsize=(8, 4))
            names = [x.split('[')[1].split(']')[0].replace("T.", "") for x in era_rows_g1.index]
            
            plt.errorbar(era_rows_g1['HR'], range(len(names)), 
                         xerr=[era_rows_g1['HR'] - era_rows_g1['Lower'], era_rows_g1['Upper'] - era_rows_g1['HR']], 
                         fmt='s', capsize=5, color='darkblue') # Different color/marker
            
            plt.yticks(range(len(names)), names)
            plt.axvline(1, color='gray', linestyle='--')
            plt.xlabel("Adjusted Mortality Hazard Ratio (95% CI)")
            plt.title("Relative Risk of Death Post-IHD vs Ancestral Era")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "mortality_hr_forest_plot.png"))
            plt.close()
            
    except Exception as e:
        logger.error(f"Failed to run Mortality Cox Model: {e}")

    logger.info("Completed Step 5.")

if __name__ == "__main__":
    # Allow running standalone
    import sys
    
    # Load Config
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        
    print(f"Loading Config from: {config_path}")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        run_step_5(config)
    else:
        print("Config not found.")
