"""
7_cci_diagcode_discovery.py
===========================
Step 7: CCI Diagcode Discovery

Scans all MediClaims datasets (2015-2023) and searches the `diagdesc`
(human-readable diagnosis description) column using regex patterns
defined in `7_cci_discovery_config.yaml`.

For each CCI category, outputs a file of unique (diagdesc, diagcode)
pairs that matched. The user reviews these files, deletes irrelevant
rows, and pastes the curated results into `8_cci_curated_codes.yaml`
for Step 8 to consume.

Outputs (to data/03_results/step_7_cci_discovery/):
    - One TSV per CCI category:  <Category>_candidates.tsv
    - discovery_summary.txt:     Counts per category
"""

import os
import sys
import re
import pandas as pd
import yaml
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logger, ensure_dir, find_project_root

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


def run_step_7(config):
    # ──────────────────────────────────────────────────────────────────
    # 1. SETUP
    # ──────────────────────────────────────────────────────────────────
    results_dir = os.path.join(config['paths']['results_dir'], "step_7_cci_discovery")
    ensure_dir(results_dir)

    logger = setup_logger("step_7", results_dir)
    logger.info("=" * 60)
    logger.info("STEP 7: CCI Diagcode Discovery")
    logger.info("=" * 60)

    # Load CCI discovery config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cci_config_path = os.path.join(script_dir, "7_cci_discovery_config.yaml")

    if not os.path.exists(cci_config_path):
        logger.error(f"Missing CCI config: {cci_config_path}")
        return None

    with open(cci_config_path, 'r') as f:
        cci_config = yaml.safe_load(f)

    diagdesc_col = cci_config.get('diagdesc_column', 'diagdesc')
    diagcode_col = cci_config.get('diagcode_column', 'diagcode')
    output_fmt = cci_config.get('output_format', 'tsv')
    cci_categories = cci_config.get('cci_categories', {})

    logger.info(f"diagdesc column:  {diagdesc_col}")
    logger.info(f"diagcode column:  {diagcode_col}")
    logger.info(f"CCI categories:   {len(cci_categories)}")
    logger.info(f"Output format:    {output_fmt}")

    # Initialize catalog
    catalog_path = os.path.join(os.getcwd(), config['paths']['catalog_config'])
    cat = DataCatalog(catalog_path)

    start_year = config['study_period']['start_year']
    end_year = config['study_period']['end_year']

    # ──────────────────────────────────────────────────────────────────
    # 2. SCAN ALL MEDICLAIMS FOR UNIQUE DIAGDESC+DIAGCODE PAIRS
    # ──────────────────────────────────────────────────────────────────
    # We collect ALL unique (diagdesc, diagcode) pairs first, then match.
    # This avoids running regex per-year per-category (N*M passes).

    logger.info("\nPhase 1: Collecting unique (diagdesc, diagcode) pairs...")

    all_pairs = set()  # (diagdesc, diagcode) tuples

    for year in range(start_year, end_year + 1):
        alias = config['datasets']['mediclaims_pattern'].format(year)
        logger.info(f"  Scanning {alias}...")

        try:
            # Load only the two columns we need
            df = cat.load(alias, usecols=[diagcode_col, diagdesc_col])

            # Drop rows where diagdesc is missing
            df = df.dropna(subset=[diagdesc_col])

            # Normalize
            df[diagdesc_col] = df[diagdesc_col].astype(str).str.strip()
            df[diagcode_col] = df[diagcode_col].astype(str).str.strip()

            # Collect unique pairs
            pairs = set(zip(df[diagdesc_col], df[diagcode_col]))
            new_count = len(pairs - all_pairs)
            all_pairs.update(pairs)

            logger.info(f"    Rows: {len(df):,} | Unique pairs in this year: {len(pairs):,} | New: {new_count:,}")

            del df
            gc.collect()

        except Exception as e:
            logger.warning(f"    Failed to load {alias}: {e}")

    logger.info(f"\nTotal unique (diagdesc, diagcode) pairs across all years: {len(all_pairs):,}")

    # Convert to DataFrame for regex matching
    df_pairs = pd.DataFrame(list(all_pairs), columns=['diagdesc', 'diagcode'])
    df_pairs.sort_values(['diagdesc', 'diagcode'], inplace=True)

    # ──────────────────────────────────────────────────────────────────
    # 3. MATCH EACH CCI CATEGORY
    # ──────────────────────────────────────────────────────────────────
    logger.info("\nPhase 2: Matching CCI categories against diagdesc...")

    summary_lines = []
    summary_lines.append("CCI DIAGCODE DISCOVERY — SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Total unique (diagdesc, diagcode) pairs scanned: {len(df_pairs):,}")
    summary_lines.append(f"Years scanned: {start_year}-{end_year}")
    summary_lines.append("")

    sep = "\t" if output_fmt == "tsv" else ","

    for cat_name, cat_def in cci_categories.items():
        regex_pattern = cat_def.get('diagdesc_regex', '')
        if not regex_pattern:
            logger.warning(f"  {cat_name}: No regex defined, skipping.")
            continue

        # Apply regex (case-insensitive)
        try:
            mask = df_pairs['diagdesc'].str.contains(
                regex_pattern, flags=re.IGNORECASE, regex=True, na=False
            )
        except re.error as e:
            logger.error(f"  {cat_name}: Invalid regex '{regex_pattern}': {e}")
            continue

        matched = df_pairs[mask].copy()
        n_matched = len(matched)
        n_unique_codes = matched['diagcode'].nunique()

        logger.info(f"  {cat_name}: {n_matched} unique (desc, code) pairs matched "
                     f"({n_unique_codes} unique diagcodes)")

        # Save candidate file
        ext = "tsv" if output_fmt == "tsv" else "csv"
        outfile = os.path.join(results_dir, f"{cat_name}_candidates.{ext}")

        # Write with quoting to handle spaces/special chars in diagdesc
        with open(outfile, 'w', encoding='utf-8') as fout:
            # Header
            fout.write(f'"diagdesc"{sep}"diagcode"{sep}"keep"\n')
            for _, row in matched.iterrows():
                desc = row['diagdesc'].replace('"', '""')  # Escape internal quotes
                code = row['diagcode'].replace('"', '""')
                fout.write(f'"{desc}"{sep}"{code}"{sep}\n')

        summary_lines.append(f"  {cat_name}:")
        summary_lines.append(f"    Matched pairs:    {n_matched}")
        summary_lines.append(f"    Unique diagcodes: {n_unique_codes}")
        summary_lines.append(f"    Output file:      {cat_name}_candidates.{ext}")
        summary_lines.append("")

    # ──────────────────────────────────────────────────────────────────
    # 4. SAVE SUMMARY
    # ──────────────────────────────────────────────────────────────────
    summary_lines.append("=" * 50)
    summary_lines.append("NEXT STEPS:")
    summary_lines.append("  1. Open each _candidates.tsv file")
    summary_lines.append("  2. Review the diagdesc column — delete rows that are NOT")
    summary_lines.append("     clinically relevant to the CCI category")
    summary_lines.append("  3. Paste the curated diagdesc + diagcode pairs into")
    summary_lines.append("     8_cci_curated_codes.yaml (see template)")
    summary_lines.append("  4. Run Step 8 to generate the definitive code lists")

    summary_path = os.path.join(results_dir, "discovery_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    logger.info(f"\nSaved summary: {summary_path}")
    logger.info("=" * 60)
    logger.info("STEP 7 COMPLETE — Review the candidate files before Step 8.")
    logger.info("=" * 60)

    return df_pairs


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    run_step_7(conf)
