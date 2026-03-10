
import os
import sys
import pandas as pd
import yaml
import gc

# Add local directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logger, report_df_info, plot_bar_chart, ensure_dir, find_project_root

# Add ROOT to path for catalog import
ROOT_DIR = find_project_root(__file__)
sys.path.append(ROOT_DIR)
# Also add ROOT/src to allow direct import of modules inside it (bypassing local src conflict)
sys.path.append(os.path.join(ROOT_DIR, "src"))

try:
    from src.catalog import DataCatalog
except ImportError:
    try:
        from catalog import DataCatalog
    except ImportError:
        # Debugging
        print(f"[CRITICAL] Could not import DataCatalog. Root: {ROOT_DIR}")
        print(f"Sys Path: {sys.path}")
        sys.exit(1)

def run_step_1(config):
    # 1. Setup
    # --------
    out_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_1_extraction")
    ensure_dir(out_dir)
    ensure_dir(results_dir)
    
    logger = setup_logger("step_1", results_dir)
    logger.info("Starting Step 1: Index Event Extraction")
    
    # Initialize Catalog
    catalog_path = os.path.join(os.getcwd(), config['paths']['catalog_config'])
    logger.info(f"Loading Catalog from: {catalog_path}")
    cat = DataCatalog(catalog_path)
    
    # 2. Parameters
    # -------------
    start_year = config['study_period']['start_year']
    end_year = config['study_period']['end_year']
    ihd_codes = config['definitions']['ihd_icd10_codes']
    
    logger.info(f"Years: {start_year}-{end_year}")
    logger.info(f"Target Codes: {ihd_codes}")
    
    # 3. Extraction Loop
    # ------------------
    extracted_chunks = []
    stats_per_year = {}
    
    for year in range(start_year, end_year + 1):
        alias = config['datasets']['mediclaims_pattern'].format(year)
        logger.info(f"Scanning {alias}...")
        
        try:
            # We ONLY need diagcode and uin and dates at this stage
            df = cat.load(alias, usecols=['uin', 'diagcode', 'discharge_date'])
            
            # Filter
            mask = df['diagcode'].isin(ihd_codes)
            subset = df[mask].copy()
            
            count = len(subset)
            stats_per_year[str(year)] = count
            logger.info(f"  -> Found {count:,} events.")
            
            if not subset.empty:
                extracted_chunks.append(subset)
            
            del df
            del mask
            del subset
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to process {alias}: {e}")
            
    # 4. Aggregation & Seasonality Check
    # ----------------------------------
    if not extracted_chunks:
        logger.error("No events extracted!")
        return None
        
    logger.info("Concatenating chunks...")
    full_df = pd.concat(extracted_chunks, ignore_index=True)
    full_df['discharge_date'] = pd.to_datetime(full_df['discharge_date'], errors='coerce')
    
    # Seasonality Plot (Monthly Counts)
    # ---------------------------------
    full_df['month_year'] = full_df['discharge_date'].dt.to_period('M').astype(str)
    mnth_counts = full_df['month_year'].value_counts().sort_index().to_dict()
    
    plot_path_month = os.path.join(results_dir, "ihd_events_monthly_trend.png")
    # Custom plot function since bar chart might be too crowded?
    # Let's use bar chart key-value, utils handles it.
    # If too many bars, maybe we should filter or use line?
    # plot_bar_chart is generic. Let's use it.
    plot_bar_chart(mnth_counts, "Monthly IHD Events Trend", "Month", "Count", plot_path_month)
    logger.info(f"Saved monthly trend plot: {plot_path_month}")

    # Top ICD Codes Plot
    # ------------------
    top_codes = full_df['diagcode'].value_counts().head(10).to_dict()
    plot_path_codes = os.path.join(results_dir, "top_icd_codes.png")
    plot_bar_chart(top_codes, "Top 10 ICD-10 Codes Extracted", "ICD Code", "Count", plot_path_codes)
    logger.info(f"Saved Top ICD Codes plot: {plot_path_codes}")

    # Deduplicate: Keep EARLIEST event per patient
    n_raw_rows = len(full_df)
    full_df.sort_values('discharge_date', inplace=True)
    unique_patients = full_df.groupby('uin').first().reset_index()
    n_unique_pats = len(unique_patients)
    
    logger.info("--- DEDUPLICATION & ISOLATION READOUT ---")
    logger.info(f"Total IHD Events Extracted: {n_raw_rows:,}")
    logger.info(f"Unique Patients (Earliest Event): {n_unique_pats:,}")
    logger.info(f"Dropped {n_raw_rows - n_unique_pats:,} duplicate events.")
    
    # HARD STOP IF DUPLICATES EXIST
    if not unique_patients['uin'].is_unique:
        logger.error(f"[CRITICAL] Isolation Failed! {unique_patients['uin'].duplicated().sum()} duplicates found.")
        raise ValueError("Index Isolation Failed: Output contains duplicate UINs.")
    
    report_df_info(unique_patients, "Unique IHD Patients", logger)
    
    # 5. Output
    # ---------
    # Use granular folder
    step_out_dir = os.path.join(out_dir, "step_1_extraction")
    ensure_dir(step_out_dir)
    
    outfile = os.path.join(step_out_dir, "index_events.csv")
    
    # Save with sidecar report
    from src.utils import save_with_report
    save_with_report(unique_patients, outfile, "Extracted Index Events (IHD)", logger)
    
    # 6. Visualization
    # ----------------
    plot_path = os.path.join(results_dir, "ihd_events_per_year.png")
    plot_bar_chart(stats_per_year, "Extracted IHD Events by Year", "Year", "Count", plot_path)
    logger.info(f"Saved plot: {plot_path}")
    
    return unique_patients

if __name__ == "__main__":
    # Test execution
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    run_step_1(conf)
