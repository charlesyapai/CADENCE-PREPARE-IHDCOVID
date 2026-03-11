"""
debug_step10_merges.py
======================
Quick diagnostic script to inspect the 5 external datasets that Step 10
uses. Focuses on the COLUMNS Step 10/11 actually need: race, vaccination,
severity (LOS, ICU, Deceased, O2), and serology — showing unique value
counts so we know how to handle them.

Usage:
    python debug_step10_merges.py
"""

import os
import sys
import yaml
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from catalog import DataCatalog
except ImportError:
    from src.catalog import DataCatalog


# Columns of interest per dataset (what Step 10/11 actually use)
COLUMNS_OF_INTEREST = {
    "COVIDFACILLOS": [
        'race', 'LOS', 'DaysInICU', 'Deceased', 'O2StartDate', 'O2EndDate',
        'notificationdate',
    ],
    "NIRListtruncated": [],  # vacc_date* and vaccbrand* handled by prefix search
    "COVID Reinfections": [
        'cat_race', 'cat_gender', 'DaysInICU', 'Deceased', 'reinfection_type',
    ],
    "FacilityUtilizationLOSSubsequentRI": [],  # small dataset, show everything
    "Serology_Tests_COVID": [
        'serologyresult', 'serologyctvalue', 'serologyvalue',
        'serologyswabdate', 'serologyresultindicator',
    ],
}


def show_value_counts(df, col, max_values=20):
    """Print value counts for a column."""
    vc = df[col].value_counts(dropna=False).head(max_values)
    n_null = df[col].isna().sum()
    print(f"\n    '{col}' (dtype: {df[col].dtype})")
    print(f"      non-null: {df[col].notnull().sum():,} / {len(df):,}")
    print(f"      null: {n_null:,}")
    print(f"      unique: {df[col].nunique():,}")
    print(f"      value counts (top {min(max_values, len(vc))}):")
    for val, cnt in vc.items():
        pct = cnt / len(df) * 100
        print(f"        {repr(val)}: {cnt:,} ({pct:.1f}%)")


def inspect_dataset(cat, alias, label):
    """Load a dataset and print value counts for columns Step 10/11 use."""
    print(f"\n{'='*70}")
    print(f"  DATASET: {label}")
    print(f"  Catalog alias: '{alias}'")
    print(f"{'='*70}")

    try:
        df = cat.load(alias)
    except Exception as e:
        print(f"  *** FAILED TO LOAD: {e}")
        return

    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  All columns: {list(df.columns)}")

    # --- Show target columns of interest ---
    target_cols = COLUMNS_OF_INTEREST.get(alias, [])

    # For NIR: find vacc_date* and vaccbrand* columns dynamically
    vacc_date_cols = sorted([c for c in df.columns if 'vacc_date' in c.lower() or 'vaccdate' in c.lower()])
    vacc_brand_cols = sorted([c for c in df.columns if 'vaccbrand' in c.lower() or 'vacc_brand' in c.lower()])
    if vacc_date_cols or vacc_brand_cols:
        print(f"\n  Vaccination date columns found: {vacc_date_cols}")
        print(f"  Vaccination brand columns found: {vacc_brand_cols}")
        # Show coverage for each dose
        for col in vacc_date_cols:
            n_has = df[col].notnull().sum()
            print(f"    {col}: {n_has:,} non-null ({n_has/len(df)*100:.1f}%)")
        # Show brand distribution for first brand col
        if vacc_brand_cols:
            target_cols = target_cols + vacc_brand_cols[:2]  # show first 2 brand cols

    # For small datasets, show all columns
    if len(df) < 5000 and not target_cols:
        target_cols = list(df.columns)

    # Check which target columns actually exist
    found = [c for c in target_cols if c in df.columns]
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        print(f"\n  *** EXPECTED COLUMNS NOT FOUND: {missing}")

    # Also find columns that look like race/severity even if named differently
    race_like = [c for c in df.columns if 'race' in c.lower() or 'ethnic' in c.lower()]
    severity_like = [c for c in df.columns if any(k in c.lower() for k in
                     ['los', 'icu', 'deceas', 'death', 'dead', 'o2', 'oxygen', 'sever'])]
    extras = [c for c in (race_like + severity_like) if c not in found]
    if extras:
        print(f"\n  Additional race/severity-like columns: {extras}")
        found = found + extras

    # Deduplicate
    found = list(dict.fromkeys(found))

    print(f"\n  --- Value counts for columns of interest ---")
    for col in found:
        show_value_counts(df, col)

    # Show first 3 rows for context
    print(f"\n  First 3 rows (all cols):")
    print(df.head(3).to_string(index=False))

    del df


def main():
    # --- Load config ---
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Init catalog ---
    catalog_path = os.path.join(os.getcwd(), config['paths']['catalog_config'])
    print(f"Catalog path: {catalog_path}")
    cat = DataCatalog(catalog_path)
    print(f"Catalog: {cat}")

    try:
        print(f"All datasets: {cat.list_datasets()}")
    except Exception:
        print("(could not list datasets)")

    # --- Hardcoded aliases (avoids KeyError from config nesting) ---
    datasets_to_check = [
        ("COVIDFACILLOS",                       "COVIDFACILLOS (severity/vacc/race)"),
        ("NIRListtruncated",                     "NIRListtruncated (vaccination)"),
        ("COVID Reinfections",                   "COVID Reinfections"),
        ("FacilityUtilizationLOSSubsequentRI",   "FacilityUtilizationLOSSubsequentRI"),
        ("Serology_Tests_COVID",                 "Serology_Tests_COVID"),
    ]

    for alias, label in datasets_to_check:
        inspect_dataset(cat, alias, label)

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
