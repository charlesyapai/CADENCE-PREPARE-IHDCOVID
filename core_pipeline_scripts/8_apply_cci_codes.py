"""
8_apply_cci_codes.py
====================
Step 8: CCI Comorbidity Re-Enrichment

Reads curated CCI diagcodes from Step 7 output (TSV files), scans
mediclaims year-by-year, creates Comorb_CCI_<Category>_Date columns
for each CCI category, computes a CCI score, and merges everything
onto the existing enriched cohort.

Input:
    - cohort_enriched.csv (from Step 3)
    - Step 7 curated TSV files (from step_7_cci_discovery/)

Output (to data/02_processed/step_8_cci/):
    - cohort_enriched_cci.csv   -- Enriched cohort + CCI columns + cci_score
    - cci_enrichment_report.txt -- Summary report

The curated TSV files should be manually edited BEFORE running this step:
    1. Run Step 7 -> outputs _candidates.tsv per CCI category
    2. Open each TSV -> delete rows that are NOT relevant
    3. Run Step 8 -> scans mediclaims using remaining diagcodes
"""

import os
import sys
import gc
import csv
import yaml
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logger, ensure_dir

try:
    from catalog import DataCatalog
except ImportError:
    from src.catalog import DataCatalog

# ==============================================================================
# CCI WEIGHTS (Charlson et al., 1987)
# ==============================================================================
CCI_WEIGHTS = {
    'Myocardial_Infarction': 1,
    'Congestive_Heart_Failure': 1,
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
    'AIDS_HIV': 6,
}


def _read_tsv_codes(filepath, logger):
    """Read a curated TSV file and return a set of diagcodes."""
    codes = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quotechar='"')
            header = next(reader, None)
            if header is None:
                return codes

            # Find diagcode column
            code_idx = None
            for i, h in enumerate(header):
                clean = h.strip().strip('"').lower()
                if 'code' in clean:
                    code_idx = i
                    break

            if code_idx is None:
                logger.warning(f"  No diagcode column in {filepath}")
                return codes

            for row in reader:
                if len(row) > code_idx:
                    code = row[code_idx].strip().strip('"')
                    if code:
                        codes.add(code)

    except Exception as e:
        logger.warning(f"  Failed to read {filepath}: {e}")

    return codes


def run_step_8(config):
    # ------------------------------------------------------------------
    # 1. SETUP
    # ------------------------------------------------------------------
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_8_cci_codes")
    output_dir = os.path.join(processed_dir, "step_8_cci")
    ensure_dir(results_dir)
    ensure_dir(output_dir)

    logger = setup_logger("step_8", results_dir)
    logger.info("=" * 70)
    logger.info("STEP 8: CCI Comorbidity Re-Enrichment")
    logger.info("  Scan mediclaims with curated codes -> enrich cohort")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 2. LOAD EXISTING ENRICHED COHORT
    # ------------------------------------------------------------------
    cohort_file = os.path.join(processed_dir, "step_3_features", "cohort_enriched.csv")
    if not os.path.exists(cohort_file):
        logger.error(f"Missing: {cohort_file}. Run Steps 1-3 first.")
        return None

    df_cohort = pd.read_csv(cohort_file)
    logger.info(f"Loaded enriched cohort: {len(df_cohort):,} patients")
    target_uins = set(df_cohort['uin'].unique())

    # ------------------------------------------------------------------
    # 3. LOAD CURATED CCI CODES FROM STEP 7 TSVs
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("Loading curated CCI codes from Step 7 output")
    logger.info("-" * 50)

    discovery_dir = os.path.join(config['paths']['results_dir'], "step_7_cci_discovery")
    if not os.path.isdir(discovery_dir):
        logger.error(f"Step 7 output not found: {discovery_dir}")
        logger.error("Run Step 7 first, curate the TSV files, then re-run Step 8.")
        return None

    # Build lookup: category -> set of diagcodes
    cci_code_lookup = {}
    total_codes = 0

    for cat_name in CCI_WEIGHTS.keys():
        tsv_path = os.path.join(discovery_dir, f"{cat_name}_candidates.tsv")
        csv_path = os.path.join(discovery_dir, f"{cat_name}_candidates.csv")

        if os.path.exists(tsv_path):
            fpath = tsv_path
        elif os.path.exists(csv_path):
            fpath = csv_path
        else:
            logger.warning(f"  {cat_name}: No curated file found, skipping.")
            continue

        codes = _read_tsv_codes(fpath, logger)
        if codes:
            cci_code_lookup[cat_name] = codes
            total_codes += len(codes)
            logger.info(f"  {cat_name:35s} -> {len(codes):>5} diagcodes")
        else:
            logger.warning(f"  {cat_name:35s} -> EMPTY (0 codes)")

    if not cci_code_lookup:
        logger.error("No CCI codes loaded from any category. Nothing to scan.")
        return None

    logger.info(f"\n  Total: {len(cci_code_lookup)} categories, {total_codes} diagcodes")

    # Build a flat reverse map: diagcode -> category (for fast lookup)
    code_to_category = {}
    for cat, codes in cci_code_lookup.items():
        for c in codes:
            # If a code maps to multiple categories, first match wins
            if c not in code_to_category:
                code_to_category[c] = cat

    logger.info(f"  Reverse map: {len(code_to_category)} unique codes")

    # ------------------------------------------------------------------
    # 4. SCAN MEDICLAIMS YEAR-BY-YEAR
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Scanning Mediclaims for CCI comorbidities")
    logger.info("=" * 70)

    catalog_path = os.path.join(os.getcwd(), config['paths']['catalog_config'])
    cat = DataCatalog(catalog_path)

    start_year = config['study_period']['start_year']
    end_year = config['study_period']['end_year']

    all_code_set = set(code_to_category.keys())
    extracted = []  # List of (uin, condition, discharge_date) tuples

    for year in range(start_year, end_year + 1):
        alias = config['datasets']['mediclaims_pattern'].format(year)
        logger.info(f"  Scanning {alias}...")

        try:
            df = cat.load(alias, usecols=['uin', 'diagcode', 'discharge_date'])

            # Pre-filter: only target patients
            df = df[df['uin'].isin(target_uins)]

            if df.empty:
                logger.info(f"    -> 0 rows for target patients")
                del df
                gc.collect()
                continue

            # Strip diagcode whitespace
            df['diagcode'] = df['diagcode'].astype(str).str.strip()

            # Match exact diagcodes via isin (much faster than regex)
            matches = df[df['diagcode'].isin(all_code_set)].copy()

            if not matches.empty:
                # Map each matched code to its CCI category
                matches['condition'] = matches['diagcode'].map(code_to_category)
                sub = matches[['uin', 'condition', 'discharge_date']].copy()
                extracted.append(sub)
                n_match = len(sub)
                n_cats = sub['condition'].nunique()
                logger.info(f"    -> {n_match:,} matches across {n_cats} categories")
            else:
                logger.info(f"    -> 0 CCI matches")

            del df, matches
            gc.collect()

        except Exception as e:
            logger.warning(f"    Failed {alias}: {e}")

    # ------------------------------------------------------------------
    # 5. BUILD CCI COLUMNS
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("Building CCI comorbidity matrix")
    logger.info("-" * 50)

    if extracted:
        all_comorbs = pd.concat(extracted, ignore_index=True)
        all_comorbs['discharge_date'] = pd.to_datetime(
            all_comorbs['discharge_date'], errors='coerce'
        )
        all_comorbs.sort_values('discharge_date', inplace=True)

        # Keep earliest date per patient per condition
        earliest = all_comorbs.groupby(
            ['uin', 'condition']
        )['discharge_date'].first().reset_index()

        # Pivot: one column per CCI category
        cci_matrix = earliest.pivot(
            index='uin', columns='condition', values='discharge_date'
        )
        cci_matrix.columns = [f"Comorb_CCI_{c}_Date" for c in cci_matrix.columns]
        cci_matrix.reset_index(inplace=True)

        logger.info(f"  CCI Matrix shape: {cci_matrix.shape}")
        logger.info(f"  Patients with any CCI: {len(cci_matrix):,}")

        # Log per-category counts
        for col in sorted(cci_matrix.columns):
            if col == 'uin':
                continue
            n_pos = cci_matrix[col].notnull().sum()
            cat_name = col.replace('Comorb_CCI_', '').replace('_Date', '')
            logger.info(f"    {cat_name:35s} N={n_pos:>6,}")

    else:
        logger.warning("  No CCI matches found in any mediclaims year!")
        cci_matrix = pd.DataFrame({'uin': list(target_uins)})

    # ------------------------------------------------------------------
    # 6. COMPUTE CCI SCORE
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("Computing CCI score")
    logger.info("-" * 50)

    # Merge CCI matrix onto cohort
    df_enriched = df_cohort.merge(cci_matrix, on='uin', how='left')

    # Binarize each CCI condition (has date = 1, else 0) and compute score
    # For the score, we check if comorbidity date <= index_date (covid or discharge)
    index_date = pd.to_datetime(
        df_enriched['covid_date'], errors='coerce'
    ).combine_first(
        pd.to_datetime(df_enriched['discharge_date'], errors='coerce')
    )

    df_enriched['cci_score'] = 0

    for cat_name, weight in CCI_WEIGHTS.items():
        col = f"Comorb_CCI_{cat_name}_Date"
        if col in df_enriched.columns:
            comorb_dt = pd.to_datetime(df_enriched[col], errors='coerce')
            # Condition is "pre-existing" if it occurred before or at the index date
            flag = ((comorb_dt.notnull()) & (comorb_dt <= index_date)).astype(int)
            df_enriched['cci_score'] += flag * weight

    logger.info(f"  CCI Score distribution:")
    logger.info(f"    Mean:   {df_enriched['cci_score'].mean():.2f}")
    logger.info(f"    Median: {df_enriched['cci_score'].median():.0f}")
    logger.info(f"    Max:    {df_enriched['cci_score'].max():.0f}")
    logger.info(f"    CCI=0:  {(df_enriched['cci_score'] == 0).mean() * 100:.1f}%")

    # By group
    for grp in ['Group 1', 'Group 2', 'Group 3']:
        sub = df_enriched[df_enriched['group'] == grp]
        if len(sub) > 0:
            logger.info(f"    {grp}: mean={sub['cci_score'].mean():.2f}, "
                        f"median={sub['cci_score'].median():.0f}, N={len(sub):,}")

    # ------------------------------------------------------------------
    # 7. SAVE
    # ------------------------------------------------------------------
    logger.info("\n" + "-" * 50)
    logger.info("Saving enriched cohort")
    logger.info("-" * 50)

    output_path = os.path.join(output_dir, "cohort_enriched_cci.csv")
    df_enriched.to_csv(output_path, index=False)
    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Shape: {df_enriched.shape}")

    # Also save the code lookup as YAML for reference
    codelist_path = os.path.join(results_dir, "cci_master_codelist.yaml")
    codelist_dict = {cat: sorted(list(codes)) for cat, codes in cci_code_lookup.items()}
    with open(codelist_path, 'w') as f:
        yaml.dump({'cci_diagcodes': codelist_dict}, f,
                  default_flow_style=False, sort_keys=False)
    logger.info(f"  Saved codelist: {codelist_path}")

    # ------------------------------------------------------------------
    # 8. SUMMARY REPORT
    # ------------------------------------------------------------------
    report_lines = []
    report_lines.append("CCI ENRICHMENT REPORT")
    report_lines.append("=" * 55)
    report_lines.append(f"Cohort size:         {len(df_enriched):,}")
    report_lines.append(f"CCI categories used: {len(cci_code_lookup)}")
    report_lines.append(f"Total diagcodes:     {total_codes}")
    report_lines.append(f"Mediclaims scanned:  {start_year}-{end_year}")
    report_lines.append("")
    report_lines.append(f"CCI Score: mean={df_enriched['cci_score'].mean():.2f}, "
                        f"median={df_enriched['cci_score'].median():.0f}")
    report_lines.append(f"CCI=0: {(df_enriched['cci_score'] == 0).mean() * 100:.1f}%")
    report_lines.append("")
    report_lines.append("Per-category counts (patients with condition):")
    for cat_name in CCI_WEIGHTS.keys():
        col = f"Comorb_CCI_{cat_name}_Date"
        if col in df_enriched.columns:
            n = df_enriched[col].notnull().sum()
            report_lines.append(f"  {cat_name:35s} N={n:>6,} ({n/len(df_enriched)*100:.2f}%)")
    report_lines.append("")
    report_lines.append(f"Output: {output_path}")
    report_lines.append(f"Codelist: {codelist_path}")
    report_lines.append("")
    report_lines.append("NEXT STEP: Run Step 9 (Tier 2 analysis) which reads")
    report_lines.append("cohort_enriched_cci.csv and uses cci_score + CCI columns.")

    report_path = os.path.join(results_dir, "cci_enrichment_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"  Saved report: {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("STEP 8 COMPLETE")
    logger.info("=" * 70)

    return df_enriched


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    run_step_8(conf)
