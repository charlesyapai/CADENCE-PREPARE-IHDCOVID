
import os
import sys
import pandas as pd
import yaml
import gc
import numpy as np
import matplotlib.pyplot as plt

# Add local directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logger, report_df_info, plot_sankey, plot_venn2, ensure_dir, find_project_root

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
try:
    from src.catalog import DataCatalog
except ImportError:
    try:
        from catalog import DataCatalog
    except ImportError:
         print(f"[CRITICAL] Could not import DataCatalog. Root: {ROOT_DIR}")
         sys.exit(1)

def run_step_2(config):
    # 1. Setup
    # --------
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_2_cohorts")
    ensure_dir(results_dir)
    
    logger = setup_logger("step_2", results_dir)
    logger.info("Starting Step 2: Cohort Generation")
    
    # 2. Load Inputs
    # --------------
    # A. Index Events (From Step 1 Folder)
    events_file = os.path.join(processed_dir, "step_1_extraction", "index_events.csv")
    if not os.path.exists(events_file):
        logger.error(f"Missing input: {events_file}. Run Step 1 first.")
        return None
    
    df_events = pd.read_csv(events_file)
    df_events['discharge_date'] = pd.to_datetime(df_events['discharge_date'])
    logger.info(f"Loaded {len(df_events):,} IHD events.")
    
    # B. COVID Registry
    # We need to load ALL COVID files defined in config
    catalog_path = os.path.join(os.getcwd(), config['paths']['catalog_config'])
    cat = DataCatalog(catalog_path)
    
    covid_patterns = config['datasets']['covid_patterns']
    covid_dfs = []
    
    logger.info("Loading COVID datasets...")
    for alias in covid_patterns:
        try:
            logger.info(f"  Loading {alias}...")
            # Minimal columns: UIN, NotificationDate, Age, Gender
            # Note: Column names might vary (case sensitivity), standardizing below
            df = cat.load(alias)
            
            # Standardize Columns
            cols_map = {
                'notificationdate': 'covid_date',
                'age': 'age', 
                'gender': 'gender',
                'sex': 'gender',
                'cat_gender': 'gender'
            }
            df.rename(columns=lambda x: x.lower(), inplace=True) # First lower all
            df.rename(columns=cols_map, inplace=True)
            
            # Keep only relevant
            keep_cols = ['uin', 'covid_date', 'age', 'gender']
            # Intersection of what we have vs what we want
            available = [c for c in keep_cols if c in df.columns]
            df = df[available]
            
            covid_dfs.append(df)
            del df
            gc.collect()
        except Exception as e:
            logger.warning(f"  Failed to load {alias}: {e}")
            
    if not covid_dfs:
        logger.error("No COVID data loaded!")
        return None
        
    df_covid = pd.concat(covid_dfs, ignore_index=True)
    
    # Deduplicate COVID: Keep EARLIEST notification
    n_raw_covid = len(df_covid)
    df_covid['covid_date'] = pd.to_datetime(df_covid['covid_date'], errors='coerce')
    df_covid.sort_values('covid_date', inplace=True)
    df_covid = df_covid.groupby('uin').first().reset_index()
    n_unique_covid = len(df_covid)
    
    logger.info("--- COVID REGISTRY DEDUPLICATION READOUT ---")
    logger.info(f"Raw COVID Rows: {n_raw_covid:,}")
    logger.info(f"Unique COVID Patients (Earliest): {n_unique_covid:,}")
    logger.info(f"Dropped {n_raw_covid - n_unique_covid:,} duplicates.")
    
    if not df_covid['uin'].is_unique:
        raise ValueError("COVID Deduplication Failed")
    
    # 3. Matching & Logic
    # -------------------
    logger.info("Merging datasets...")
    
    # Merge on UIN
    # Master UIN list = Union(COVID, IHD)
    
    merged = pd.merge(df_covid, df_events, on='uin', how='outer', suffixes=('_covid', '_ihd'))
    
    # HARD STOP IF MERGE CREATED DUPLICATES
    if not merged['uin'].is_unique:
        logger.error(f"Merge created duplicates! Unique UINs: {merged['uin'].nunique()}, Rows: {len(merged)}")
        raise ValueError("Merge Integrity Failed: UINs are not unique.")
        
    
    # Date logic parameters
    follow_up_days = config['cohort_definitions']['follow_up_days']
    washout = config['cohort_definitions']['washout_period_days']
    
    # Classification Logic
    def classify(row):
        has_covid = not pd.isnull(row['covid_date'])
        has_ihd = not pd.isnull(row['discharge_date'])
        
        if has_covid and has_ihd:
            # Check timing
            # COVID First?
            days_diff = (row['discharge_date'] - row['covid_date']).days
            
            if 0 <= days_diff <= follow_up_days:
                return 'Group 1' # Post-COVID IHD
            elif days_diff < 0:
                # IHD happened BEFORE COVID
                # User says: "Group 3: Patients with IHD and no diagnosis of COVID within 1 year prior"
                # If IHD is way before, they are just IHD.
                # If IHD is shortly before COVID, they fail the "No COVID within 1yr prior" rule?
                # Actually, standard design:
                # If IHD is index, we check history. If COVID is index, we check future.
                
                # Let's strictly follow User's updated definition in Plan:
                # Group 3: IHD + No COVID within 1 yr prior.
                # If COVID was 2 days after IHD, then COVID was "within 1 year prior" is FALSE (it was after).
                # So they qualify for Group 3? 
                # Wait, "No COVID within 1 year PRIOR to IHD".
                
                # If COVID date > IHD date, then COVID is in the FUTURE relative to IHD.
                # So relative to IHD date, they had NO COVID in the PAST.
                # So they are Group 3.
                return 'Group 3' 
                
            else:
                # days_diff > 365
                # IHD happened > 1 year after COVID.
                # They had COVID. They didn't have IHD within 1 year.
                # So they form part of Group 2 (COVID Non-IHD) for that first year?
                # AND they are also an IHD case later?
                # This is "Immortal time bias" territory.
                # Standard approach: They enter Group 2 at COVID date. They are censored or become a case later.
                # For this specific static group definitions:
                # User: "Group 2 = COVID + No IHD within 1 year". 
                # This patient fits that description.
                return 'Group 2'

        elif has_covid and not has_ihd:
            return 'Group 2' # COVID, No IHD ever (in dataset)
            
        elif not has_covid and has_ihd:
            return 'Group 3' # IHD, Never had COVID
            
        return 'Unknown'

    merged['group'] = merged.apply(classify, axis=1)
    
    # Fill Demographics for Group 3 (Naive IHD) who came from IHD registry (no age/gender there usually)
    # We need to fetch from SingCLOUD
    missing_demo_mask = (merged['group'] == 'Group 3') & (merged['age'].isna())
    missing_uins = merged.loc[missing_demo_mask, 'uin'].unique()
    
    if len(missing_uins) > 0:
        logger.info(f"Fetching demographics for {len(missing_uins):,} Group 3 patients...")
        try:
            # Gender
            df_gen = cat.load(config['datasets']['singcloud']['gender'])
            # DOB
            df_dob = cat.load(config['datasets']['singcloud']['dob'])
            
            # Prepare lookup
            gen_map = df_gen.set_index('PAT_ID_X')['PAT_GENDER'].to_dict()
            dob_map = df_dob.set_index('PAT_ID_X')['PAT_DOB_X'].to_dict() # Assumes Year
            
            # Apply
            def fill_gender(r):
                if pd.isna(r['gender']) and r['uin'] in gen_map:
                    return gen_map[r['uin']]
                return r['gender']
                
            def fill_age(r):
                if pd.isna(r['age']) and r['uin'] in dob_map:
                    # Age = EventYear - BirthYear
                    if pd.notnull(r['discharge_date']):
                        return r['discharge_date'].year - int(dob_map[r['uin']])
                return r['age']

            merged['gender'] = merged.apply(fill_gender, axis=1)
            merged['age'] = merged.apply(fill_age, axis=1)
            
        except Exception as e:
            logger.warning(f"Demographic fetch failed: {e}")
    
    # 4. Sankey & Venn
    # ----------------
    # Nodes: Total -> COVID / No COVID -> IHD / No IHD -> Groups
    
    # Refined Flow:
    # L1: Total Population
    # L2: COVID+ vs No COVID
    # L3: Overlap Logic
    #     COVID+ -> [COVID Only (No IHD)] | [COVID + IHD Overlap]
    #     No COVID -> [IHD Only (Naive IHD Source)] | [Healthy/Others (Implicitly removed)]
    # L4: Group Assignment
    #     [COVID+IHD Overlap] -> [G1 (<1y)] | [G2 (>1y)] | [G3 (<0, IHD before)]
    #     [COVID Only] -> [G2]
    #     [IHD Only] -> [G3]
    
    # Counts
    total_pop = len(merged)
    
    # Global Group Counts for Attrition Report
    n_g1 = (merged['group'] == 'Group 1').sum()
    n_g2 = (merged['group'] == 'Group 2').sum()
    n_g3 = (merged['group'] == 'Group 3').sum()
    
    # 1. Level 1 Split
    covid_pos_mask = merged['covid_date'].notnull()
    ihd_pos_mask = merged['discharge_date'].notnull()
    
    n_covid_pos = covid_pos_mask.sum()
    n_no_covid = (~covid_pos_mask).sum()
    
    # 2. Level 2 Split (From COVID+)
    # Overlap = COVID+ AND IHD+
    mask_overlap = covid_pos_mask & ihd_pos_mask
    n_overlap = mask_overlap.sum()
    n_covid_only = n_covid_pos - n_overlap
    
    # From No COVID
    # No COVID + IHD = Naive IHD Source
    mask_naive_ihd = (~covid_pos_mask) & ihd_pos_mask
    n_naive_ihd_source = mask_naive_ihd.sum()
    # Note: There are also No-COVID, No-IHD patients? 
    # The merge was outer. If we loaded specific IHD events and specific COVID list.
    # If a patient is in neither, they aren't in `merged`.
    # So `n_no_covid` must be `n_naive_ihd_source` unless we loaded a base population?
    # `merged` = outer join of COVID and IHD. So everyone has at least one.
    # So n_no_covid should exactly equal n_naive_ihd_source.
    
    # 3. Level 3 Split (From Overlap)
    # We need to see where they went.
    # Split Overlap by TIME
    overlap_df = merged[mask_overlap].copy()
    overlap_df['diff'] = (overlap_df['discharge_date'] - overlap_df['covid_date']).dt.days
    
    # G1: 0 <= diff <= 365
    n_ov_g1 = ((overlap_df['diff'] >= 0) & (overlap_df['diff'] <= follow_up_days)).sum()
    # G2 (Late): diff > 365
    n_ov_g2_late = (overlap_df['diff'] > follow_up_days).sum()
    # G3 (Prior): diff < 0
    n_ov_g3_prior = (overlap_df['diff'] < 0).sum()
    
    # Sanity check
    if n_overlap != (n_ov_g1 + n_ov_g2_late + n_ov_g3_prior):
        logger.warning(f"Sankey Math Mismatch: Overlap {n_overlap} != splits {n_ov_g1+n_ov_g2_late+n_ov_g3_prior}")
        
    # Nodes List
    # 0: Total Patients
    # 1: COVID Positive
    # 2: No COVID Recorded
    # 3: COVID + IHD Overlap
    # 4: COVID Only (No IHD)
    # 5: IHD Only (No COVID)
    # 6: Group 1 (Post-COVID IHD)
    # 7: Group 2 (COVID Non-IHD)
    # 8: Group 3 (Naive IHD)

    def lbl(name, n):
        # Use HTML break for multi-line labels to reduce width
        return f"{name}<br>(N={n:,})"

    labels = [
        lbl("Total Cohort", total_pop),      # 0
        lbl("COVID Positive", n_covid_pos),  # 1
        lbl("No COVID", n_no_covid),         # 2
        lbl("COVID + IHD Overlap", n_overlap), # 3
        lbl("COVID Only", n_covid_only),     # 4
        lbl("IHD Only", n_naive_ihd_source), # 5
        lbl("Group 1 (Post-COVID IHD)", n_ov_g1), # 6
        lbl("Group 2 (COVID Non-IHD)", n_covid_only + n_ov_g2_late), # 7
        lbl("Group 3 (Naive IHD)", n_naive_ihd_source + n_ov_g3_prior) # 8
    ]
    
    # --- Mermaid Flowchart Generation ---
    # ------------------------------------
    # Visualize the logic as requested
    mermaid_code = f"""graph TD
    subgraph Raw Inputs
        A["Raw COVID Notifications: {n_raw_covid:,}"]
        B["Raw IHD Events: {len(df_events):,}"]
    end

    subgraph Deduplication
        A -->|Limit to Earliest| C["Unique COVID Patients: {n_unique_covid:,}"]
        B -->|Limit to Earliest| D["Unique IHD Patients: {len(df_events):,}"]
    end
    
    subgraph Merging
        C --> E{{"Merge on UIN"}}
        D --> E
        E --> F["Total Union Cohort: {total_pop:,}"]
    end
    
    subgraph Attrition_and_Logic
        F --> G{{"COVID Status?"}}
        G -->|"COVID+ ({n_covid_pos:,})"| H{{"IHD Status?"}}
        G -->|"No COVID ({n_no_covid:,})"| I{{"IHD Status?"}}
        
        I -->|IHD Only| J["Potential Group 3"]
        J --> K["Group 3: Naive IHD<br/>(N={n_naive_ihd_source:,})"]
        
        H -->|No IHD ever| L["Group 2: COVID Only<br/>(N={n_covid_only:,})"]
        H -->|COVID + IHD Overlap| M{{"Time Logic<br/>(IHD Date - COVID Date)"}}
        
        M -->|Diff 0-365 days| N["Group 1: Post-COVID IHD<br/>(N={n_ov_g1:,})"]
        M -->|Diff > 365 days| O["Group 2: Late IHD<br/>(N={n_ov_g2_late:,})"]
        M -->|Diff < 0 days| P["Group 3: Prior IHD<br/>(N={n_ov_g3_prior:,})"]
    end
    
    subgraph Final_Groups
        N --> G1_Final["Group 1 Total: {n_g1:,}"]
        L --> G2_Final["Group 2 Total: {n_g2:,}"]
        O --> G2_Final
        K --> G3_Final["Group 3 Total: {n_g3:,}"]
        P --> G3_Final
    end
    
    style G1_Final fill:#ffb366,stroke:#333
    style G2_Final fill:#66b3ff,stroke:#333
    style G3_Final fill:#00cc99,stroke:#333
    """
    
    mermaid_path = os.path.join(results_dir, "cohort_flowchart.mmd")
    with open(mermaid_path, "w") as f:
        f.write(mermaid_code)
    logger.info(f"Saved Mermaid Flowchart to {mermaid_path}")
    
    # Colors (Pastel/Vibrant scheme)
    # 0:Gray, 1:Red, 2:Blue, 3:Purple, 4:Pink, 5:LightBlue, 6:Orange, 7:Green, 8:Teal
    colors = [
        "lightgray", # Total
        "#ff9999",   # COVID+ (Reddish)
        "#99ccff",   # No COVID (Blueish)
        "#c2c2f0",   # Overlap (Purple)
        "#ffb3e6",   # COVID Only (Pink)
        "#99ffcc",   # IHD Only (Mint)
        "#ffb366",   # G1 (Orange)
        "#66b3ff",   # G2 (Blue)
        "#00cc99"    # G3 (Teal)
    ]
    
    source = []
    target = []
    value = []
    
    # L1 -> L2
    # Total -> COVID+
    source.append(0); target.append(1); value.append(n_covid_pos)
    # Total -> No COVID
    source.append(0); target.append(2); value.append(n_no_covid)
    
    # L2 -> L3
    # COVID+ -> Overlap
    source.append(1); target.append(3); value.append(n_overlap)
    # COVID+ -> COVID Only
    source.append(1); target.append(4); value.append(n_covid_only)
    
    # No COVID -> IHD Only
    source.append(2); target.append(5); value.append(n_naive_ihd_source)
    
    # L3 -> L4 (Group Assignment)
    # Overlap -> G1
    source.append(3); target.append(6); value.append(n_ov_g1)
    # Overlap -> G2 (Late IHD)
    source.append(3); target.append(7); value.append(n_ov_g2_late)
    # Overlap -> G3 (Prior IHD)
    source.append(3); target.append(8); value.append(n_ov_g3_prior)
    
    # COVID Only -> G2
    source.append(4); target.append(7); value.append(n_covid_only)
    
    # IHD Only -> G3
    source.append(5); target.append(8); value.append(n_naive_ihd_source)
    
    # Generate Link Colors (Transparency logic)
    def hex_to_rgba(hex_code, alpha=0.4):
        hex_code = hex_code.lstrip('#')
        if len(hex_code) == 6:
            r = int(hex_code[0:2], 16)
            g = int(hex_code[2:4], 16)
            b = int(hex_code[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return hex_code # Fallback (e.g. named colors)
        
    # Special handling for named color 'lightgray' -> just simple gray rgba
    color_map = {c: c for c in colors}
    color_map['lightgray'] = 'rgba(200, 200, 200, 0.4)'
    
    link_colors = []
    for s in source:
        src_color = colors[s]
        if src_color.startswith('#'):
            link_colors.append(hex_to_rgba(src_color))
        else:
            link_colors.append(color_map.get(src_color, src_color))

    plot_sankey(labels, source, target, value, "Cohort Generation Flow", os.path.join(results_dir, "cohort_sankey.png"), 
                node_colors=colors, link_colors=link_colors)
    
    # Venn
    u_covid = set(df_covid['uin'])
    u_ihd = set(df_events['uin'])
    plot_venn2(u_covid, u_ihd, ('COVID Registry', 'IHD Registry'), "Patient Overlap", os.path.join(results_dir, "overlap_venn.png"))
    
    # Attrition Report
    # ----------------
    attrition_file = os.path.join(results_dir, "cohort_attrition.txt")
    with open(attrition_file, "w") as f:
        f.write("COHORT ATTRITION & LOGIC REPORT\n")
        f.write("===============================\n")
        f.write(f"Total COVID Patients:       {len(df_covid):,}\n")
        f.write(f"Total IHD Patients (Events):{len(df_events):,}\n")
        f.write(f"Intersection (Overlapping): {len(merged[merged['covid_date'].notnull() & merged['discharge_date'].notnull()]):,}\n\n")
        f.write("Group Assignment Logic:\n")
        f.write(f"  Group 1 (COVID -> IHD <= {follow_up_days}d): {n_g1:,}\n")
        f.write(f"  Group 2 (COVID -> No IHD in window): {n_g2:,}\n")
        f.write(f"  Group 3 (IHD -> No COVID in {washout}d prior): {n_g3:,} (Naive IHD)\n")
        f.write("\nEdges Cases / Exclusions:\n")
        f.write(f"  IHD happened BEFORE COVID (Relegated to G3?): {n_ov_g3_prior:,}\n")
        f.write(f"  IHD happened > {follow_up_days}d after COVID (Relegated to G2?): {n_ov_g2_late:,}\n")
        # Let's count explicitly
        mask_late = (merged['group'] == 'Group 2') & (merged['discharge_date'].notnull())
        f.write(f"  Patients with IHD > {follow_up_days}d after COVID (Counted as G2 initially): {mask_late.sum():,}\n")
        
    logger.info(f"Saved Attrition Report: {attrition_file}")
    
    # Extra Plots
    # -----------
    # 1. Age Distribution by Group
    try:
        plt.figure(figsize=(10, 6))
        # Drop missing ages
        plot_df = merged.dropna(subset=['age', 'group'])
        
        # Use simple boxplot
        groups = sorted(plot_df['group'].unique())
        data_to_plot = [plot_df[plot_df['group'] == g]['age'] for g in groups]
        
        plt.boxplot(data_to_plot, labels=groups)
        plt.title("Age Distribution by Cohort Group")
        plt.ylabel("Age (Years)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        box_path = os.path.join(results_dir, "age_distribution_boxplot.png")
        plt.savefig(box_path)
        plt.close()
        logger.info(f"Saved Age Boxplot: {box_path}")
        
    except Exception as e:
        logger.warning(f"Failed Age Plot: {e}")
        
    
    # 2. Time Lag Distribution (Group 1 Only / Overlap Analysis)
    try:
        # Focus on patients with BOTH IHD and COVID (regardless of Group assignment initially)
        overlap_mask = merged['covid_date'].notnull() & merged['discharge_date'].notnull()
        overlap_df = merged[overlap_mask].copy()
        
        if not overlap_df.empty:
            # Calculate Lag (IHD Date - COVID Date)
            overlap_df['lag_days'] = (overlap_df['discharge_date'] - overlap_df['covid_date']).dt.days
            
            # Filter for POSITIVE lag (Post-COVID IHD) as per request context
            # "how many... were apart by less than 1 year (after)"
            # Assuming interest is in the sequence COVID -> IHD
            pos_lag = overlap_df[overlap_df['lag_days'] > 0].copy()
            
            if not pos_lag.empty:
                # 1. Buckets
                pos_lag['lag_years'] = pos_lag['lag_days'] / 365.0
                
                lt_1 = (pos_lag['lag_years'] < 1).sum()
                lt_2 = ((pos_lag['lag_years'] >= 1) & (pos_lag['lag_years'] < 2)).sum()
                lt_3 = ((pos_lag['lag_years'] >= 2) & (pos_lag['lag_years'] < 3)).sum()
                gte_3 = (pos_lag['lag_years'] >= 3).sum()
                
                # Stats
                mean_lag = pos_lag['lag_days'].mean()
                median_lag = pos_lag['lag_days'].median()
                
                # 2. Report
                logger.info("--- Overlap Lag Analysis (COVID -> IHD) ---")
                logger.info(f"Total Post-COVID IHD pairs: {len(pos_lag)}")
                logger.info(f"  < 1 Year:  {lt_1} ({lt_1/len(pos_lag)*100:.1f}%)")
                logger.info(f"  1-2 Years: {lt_2} ({lt_2/len(pos_lag)*100:.1f}%)")
                logger.info(f"  2-3 Years: {lt_3} ({lt_3/len(pos_lag)*100:.1f}%)")
                logger.info(f"  >= 3 Years:{gte_3} ({gte_3/len(pos_lag)*100:.1f}%)")
                logger.info(f"  Mean Lag: {mean_lag:.1f} days")
                logger.info(f"  Median:   {median_lag:.1f} days")
                
                # 3. Stacked Bar / Histogram Visualization
                # User asked for a "Venn", but meaningful visual is bar.
                # We will make a visually distinct bar chart "Breakdown of Overlap Duration"
                
                plt.figure(figsize=(8, 6))
                categories = ['< 1 Year', '1-2 Years', '2-3 Years', '>= 3 Years']
                counts = [lt_1, lt_2, lt_3, gte_3]
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'] # Soft colors
                
                bars = plt.bar(categories, counts, color=colors, edgecolor='black')
                plt.title("Duration between COVID-19 and Subsequent IHD Event")
                plt.xlabel("Time Interval")
                plt.ylabel("Number of Patients")
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                
                # Add counts on top
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{int(height)}',
                             ha='center', va='bottom')
                
                lag_plot_path = os.path.join(results_dir, "overlap_duration_breakdown.png")
                plt.savefig(lag_plot_path)
                plt.close()
                logger.info(f"Saved Overlap Breakdown Plot: {lag_plot_path}")
                
                # Also save the raw stats to text
                with open(os.path.join(results_dir, "overlap_stats.txt"), "w") as f:
                    f.write("OVERLAP DURATION STATISTICS\n")
                    f.write("===========================\n")
                    f.write(f"Total Patients (COVID -> IHD): {len(pos_lag)}\n")
                    f.write(f"Mean Lag:   {mean_lag:.1f} days\n")
                    f.write(f"Median Lag: {median_lag:.1f} days\n\n")
                    f.write("Breakdown:\n")
                    f.write(f"  < 1 Year:   {lt_1}\n")
                    f.write(f"  1-2 Years:  {lt_2}\n")
                    f.write(f"  2-3 Years:  {lt_3}\n")
                    f.write(f"  >= 3 Years: {gte_3}\n")
                    
            else:
                logger.info("No Post-COVID IHD patients found for overlap analysis.")

    except Exception as e:
        logger.warning(f"Failed Overlap Analysis: {e}")
    
    # 5. Save Output
    # --------------
    # Granular output folder
    step_out_dir = os.path.join(processed_dir, "step_2_cohorts")
    ensure_dir(step_out_dir)
    outfile = os.path.join(step_out_dir, "cohort_definitions.csv")
    
    from src.utils import save_with_report
    save_with_report(merged, outfile, "Final Cohort Definitions (G1, G2, G3)", logger)
    
    logger.info(f"Group Counts:\n{merged['group'].value_counts()}")
    logger.info(f"Saved: {outfile}")
    
    return merged

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    run_step_2(conf)
