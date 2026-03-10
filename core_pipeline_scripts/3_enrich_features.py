
import os
import sys
import pandas as pd
import yaml
import gc

# Add local directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logger, report_df_info, plot_bar_chart, save_with_report, ensure_dir, find_project_root

# Add ROOT to path for catalog
ROOT_DIR = find_project_root(__file__)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

try:
    from src.catalog import DataCatalog
except ImportError:
    try:
        from catalog import DataCatalog
    except ImportError:
         print(f"[CRITICAL] Could not import DataCatalog. Root: {ROOT_DIR}")
         sys.exit(1)

def run_step_3(config):
    # 1. Setup
    # --------
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_3_features")
    ensure_dir(results_dir)
    
    logger = setup_logger("step_3", results_dir)
    logger.info("Starting Step 3: Feature Enrichment (Comorbidities & Meds)")
    
    # 2. Load Cohort (Target Population)
    # ----------------------------------
    cohort_file = os.path.join(processed_dir, "step_2_cohorts", "cohort_definitions.csv")
    if not os.path.exists(cohort_file):
        logger.error(f"Missing cohort file: {cohort_file}. Run Step 2 first.")
        return None
        
    df_cohort = pd.read_csv(cohort_file)
    logger.info(f"Target Cohort: {len(df_cohort):,} patients.")
    
    # Convert dates for logic
    df_cohort['index_date'] = pd.to_datetime(df_cohort['covid_date']).combine_first(pd.to_datetime(df_cohort['discharge_date']))
    target_uins = set(df_cohort['uin'].unique())
    
    # Initialize components
    catalog_path = os.path.join(os.getcwd(), config['paths']['catalog_config'])
    cat = DataCatalog(catalog_path)
    
    # 3. Comorbidity Extraction
    # -------------------------
    logger.info("Scanning for Comorbidities...")
    comorb_defs = config['definitions']['comorbidities']
    extracted_comorbs = []
    
    start_year = config['study_period']['start_year']
    end_year = config['study_period']['end_year']
    
    for year in range(start_year, end_year + 1):
        alias = config['datasets']['mediclaims_pattern'].format(year)
        logger.info(f"  Scanning {alias}...")
        
        try:
            df = cat.load(alias, usecols=['uin', 'diagcode', 'discharge_date'])
             # Pre-filter for performance
            df = df[df['uin'].isin(target_uins)]
            
            if not df.empty:
                for name, regex in comorb_defs.items():
                    matches = df[df['diagcode'].str.contains(regex, regex=True, na=False)]
                    if not matches.empty:
                        sub = matches[['uin', 'discharge_date']].copy()
                        sub['condition'] = name
                        extracted_comorbs.append(sub)
            
            del df
            gc.collect()
            
        except Exception as e:
            logger.warning(f"  Failed {alias}: {e}")

    # Process Comorbidities
    if extracted_comorbs:
        logger.info("Aggregating Comorbidities...")
        all_comorbs = pd.concat(extracted_comorbs, ignore_index=True)
        all_comorbs['discharge_date'] = pd.to_datetime(all_comorbs['discharge_date'], errors='coerce')
        all_comorbs.sort_values('discharge_date', inplace=True)
        
        # Keep Earliest per Patient per Condition
        earliest = all_comorbs.groupby(['uin', 'condition'])['discharge_date'].first().reset_index()
        
        # Pivot
        comorb_matrix = earliest.pivot(index='uin', columns='condition', values='discharge_date')
        comorb_matrix.columns = [f"Comorb_{c}_Date" for c in comorb_matrix.columns]
        comorb_matrix.reset_index(inplace=True)
        
        logger.info(f"Comorbidity Matrix: {comorb_matrix.shape}")
    else:
        comorb_matrix = pd.DataFrame({'uin': list(target_uins)})
    
    # 4. Medication Extraction
    # ------------------------
    logger.info("Scanning for Medications...")
    med_defs = config['definitions']['medications']
    extracted_meds = []
    
    med_template = config['datasets']['singcloud']['medications'] # e.g. "SingCLOUD_medication_items{}"
    
    # Assume 1-29 chunks
    for i in range(1, 30):
        # Skip excessive logging
        if i % 10 == 1: logger.info(f"  Scanning Pharmacy Chunk {i}...")
        
        alias = med_template.format(i)
        try:
            # Need columns: ITEM_NAME_ORI_TXT, PATIENT_ID_EXTN_X, MEDITEM_DATE_Z
            # Check catalog for exact standard names or aliases
            # We'll assume standard catalog setup or handle errors
            df = cat.load(alias)
            
            # Map columns
            # Assumption: catalog load returns standard names or we rename
            col_map = {
                'PATIENT_ID_EXTN_X': 'uin',
                'MEDITEM_DATE_Z': 'med_date',
                'ITEM_NAME_ORI_TXT': 'drug_name'
            }
            # Rename intersection
            df.rename(columns=col_map, inplace=True)
            
            # Filter
            df = df[df['uin'].isin(target_uins)]
            
            if not df.empty:
                df['drug_name'] = df['drug_name'].astype(str).str.lower()
                
                for cls_name, keywords in med_defs.items():
                    # Pattern: match any keyword
                    pattern = '|'.join(keywords)
                    matches = df[df['drug_name'].str.contains(pattern, regex=True, na=False)]
                    
                    if not matches.empty:
                        sub = matches[['uin', 'med_date']].copy()
                        sub['med_class'] = cls_name
                        extracted_meds.append(sub)
                        
            del df
            gc.collect()
        except:
            pass # Chunk might not exist

    # Process Meds
    if extracted_meds:
        logger.info("Aggregating Medications...")
        all_meds = pd.concat(extracted_meds, ignore_index=True)
        all_meds['med_date'] = pd.to_datetime(all_meds['med_date'], errors='coerce')
        all_meds.sort_values('med_date', inplace=True)
        
        earliest_med = all_meds.groupby(['uin', 'med_class'])['med_date'].first().reset_index()
        
        med_matrix = earliest_med.pivot(index='uin', columns='med_class', values='med_date')
        med_matrix.columns = [f"Med_{c}_Date" for c in med_matrix.columns]
        med_matrix.reset_index(inplace=True)
        
        logger.info(f"Medication Matrix: {med_matrix.shape}")
    else:
        med_matrix = pd.DataFrame({'uin': list(target_uins)})

    # 5. Mortality Extraction
    # -----------------------
    logger.info("Scanning Death Registry...")
    mortality_alias = config['datasets'].get('death_registry', 'death_registry')
    try:
        # Load Death Registry
        # Expecting: uin (or nric_masked), death_date, causeofdeath
        df_death = cat.load(mortality_alias)
        
        # Standardize
        cols_map = {'nric_masked': 'uin', 'death_date': 'death_date', 'causeofdeath': 'death_cause'}
        df_death.rename(columns=cols_map, inplace=True)
        
        # Filter
        df_death = df_death[df_death['uin'].isin(target_uins)][['uin', 'death_date', 'death_cause']]
        df_death['death_date'] = pd.to_datetime(df_death['death_date'], errors='coerce')
        
        # Deduplicate Death Registry: Keep Earliest
        # Preventing row explosion if multiple death records exist
        n_raw_death = len(df_death)
        df_death.sort_values('death_date', inplace=True)
        df_death = df_death.drop_duplicates(subset=['uin'], keep='first')
        n_unique_death = len(df_death)
        
        logger.info(f"  Found {n_raw_death:,} death records. Unique UINs: {n_unique_death:,} (Dropped {n_raw_death - n_unique_death} duplicates)")
    
    except Exception as e:
        logger.warning(f"  Failed to load Death Registry: {e}")
        df_death = pd.DataFrame(columns=['uin', 'death_date', 'death_cause'])

    # 6. Merge Everything
    # -------------------
    logger.info("Merging features into Cohort...")
    final_df = df_cohort.merge(comorb_matrix, on='uin', how='left')
    final_df = final_df.merge(med_matrix, on='uin', how='left')
    final_df = final_df.merge(df_death, on='uin', how='left')
    
    # 7. Output
    # ---------
    step_out_dir = os.path.join(processed_dir, "step_3_features")
    ensure_dir(step_out_dir)
    
    outfile = os.path.join(step_out_dir, "cohort_enriched.csv")
    save_with_report(final_df, outfile, "Cohort with Comorbidities and Medications", logger)
    
    # 7. Viz
    # ------
    # Prevalence Plot (Binary)
    # Convert dates to binary (Prevalent = Date < Index)
    # Note: Index date depends on group? Or just "Has history found"?
    # For reporting, usually we want Baseline Prevalence (History before Index)
    
    # TBD: Binary logic will often happen in analysis step, but let's do a simple count here
    # Just count non-nulls
    
    counts = {}
    feature_cols = [c for c in final_df.columns if 'Comorb_' in c or 'Med_' in c]
    for c in feature_cols:
        counts[c] = final_df[c].notnull().sum()
        
    plot_bar_chart(counts, "Feature Availability (Count of non-null dates)", "Feature", "Count", 
                   os.path.join(results_dir, "feature_counts.png"))

    # Extra Plot: Prevalence by Group
    # -------------------------------
    try:
        # We need groups from final_df
        # Calculate % of each comorb/med per group
        groups = sorted(final_df['group'].unique())
        features_to_plot = [c for c in feature_cols if 'Comorb_' in c] # Focus on conditions
        
        prevalence_data = {g: [] for g in groups}
        labels = [c.replace('Comorb_', '').replace('_Date', '') for c in features_to_plot]
        
        for g in groups:
            sub = final_df[final_df['group'] == g]
            n = len(sub)
            if n == 0:
                prevalence_data[g] = [0] * len(features_to_plot)
            else:
                vals = []
                for f in features_to_plot:
                    # Non specific "is present" check (Date Exists)
                    vals.append((sub[f].notnull().sum() / n) * 100)
                prevalence_data[g] = vals
                
        # Grouped Bar Plot
        x = np.arange(len(labels))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        
        for i, g in enumerate(groups):
            offset = (i - 1) * width # Centers 3 groups (-1, 0, 1) approx
            plt.bar(x + offset, prevalence_data[g], width, label=g)
            
        plt.ylabel("Prevalence (%)")
        plt.title("Comorbidity Prevalence by Group")
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        prev_path = os.path.join(results_dir, "comorbidity_prevalence_comparison.png")
        plt.savefig(prev_path)
        plt.close()
        logger.info(f"Saved Prevalence Plot: {prev_path}")
        
    except Exception as e:
        logger.warning(f"Failed Prevalence Plot: {e}")

    return final_df

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    run_step_3(conf)
