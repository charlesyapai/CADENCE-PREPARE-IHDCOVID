"""
10_vaccine_severity_enrichment.py
=================================
Step 10: Vaccination, Severity & Serology Enrichment

Ingests five new datasets and merges vaccine status, COVID severity indicators,
race, reinfection flags, and serology results onto the CCI-enriched cohort.

New data sources (from catalog):
    1. COVIDFACILLOS              (2.35M rows) - Primary COVID facility/LOS dataset
    2. NIRListtruncated           (6.17M rows) - National Immunisation Registry
    3. COVID Reinfections         (128K rows)  - Reinfection records with severity
    4. FacilityUtilizationLOSSubsequentRI (2,319 rows) - Reinfection facility use
    5. Serology_Tests_COVID       (1.07M rows) - Serology test results

Derived variables:
    - Vaccination: doses_before_covid, vaccinated_before_covid (bool), vaccine_brand_primary
    - Severity:    LOS, DaysInICU, Deceased, required_O2 (bool), severity_category
    - Demographics: race
    - Reinfection:  is_reinfection (bool), reinfection_type
    - Serology:     serology_result, serology_ct_value (earliest test)
    - Era:          variant_era (Ancestral / Delta / Omicron)

Input:
    - cohort_enriched_cci.csv (from Step 8)

Output (to data/02_processed/step_10_enriched/):
    - cohort_tier3_ready.csv        -- Fully enriched cohort for Tier 3
    - enrichment_data_report.txt    -- Detailed data profiling report
    - vaccination_summary.csv       -- Vaccination coverage by era/group
    - severity_summary.csv          -- Severity breakdown by era/group
"""

import os
import sys
import gc
import yaml
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logger, save_with_report, ensure_dir, report_df_info

try:
    from catalog import DataCatalog
except ImportError:
    from src.catalog import DataCatalog


# ==============================================================================
# HELPERS
# ==============================================================================

def _count_doses_before_date(vacc_dates, ref_date):
    """
    Given a Series of vaccination date columns and a reference date,
    count how many doses were administered strictly before ref_date.

    Parameters
    ----------
    vacc_dates : pd.DataFrame
        Columns like vacc_date1 ... vacc_date6, one row per patient.
    ref_date : pd.Series
        Reference date per patient (e.g. covid_date or ihd_date).

    Returns
    -------
    pd.Series of int
    """
    count = pd.Series(0, index=vacc_dates.index, dtype=int)
    for col in vacc_dates.columns:
        dt = pd.to_datetime(vacc_dates[col], errors='coerce')
        count += ((dt.notnull()) & (dt < ref_date)).astype(int)
    return count


def _get_primary_brand(row, vacc_brand_cols):
    """Return first non-null vaccine brand (i.e. brand of dose 1)."""
    for col in vacc_brand_cols:
        val = row.get(col)
        if pd.notnull(val) and str(val).strip():
            return str(val).strip()
    return np.nan


def _assign_era(date):
    """Map a COVID infection date to a variant era."""
    if pd.isnull(date):
        return 'Unknown'
    if date < pd.Timestamp('2021-05-01'):
        return 'Ancestral'
    elif date < pd.Timestamp('2022-01-01'):
        return 'Delta'
    else:
        return 'Omicron'


def _assign_severity(row):
    """
    Categorise COVID severity from facility indicators.
    Categories: Critical / Severe / Moderate / Mild / Unknown
    """
    icu = row.get('DaysInICU', 0)
    deceased = row.get('Deceased', 0)
    los = row.get('LOS', 0)
    o2 = row.get('required_O2', False)

    # Coerce to numeric
    try:
        icu = float(icu) if pd.notnull(icu) else 0
    except (ValueError, TypeError):
        icu = 0
    try:
        los = float(los) if pd.notnull(los) else 0
    except (ValueError, TypeError):
        los = 0

    if deceased == 1 or (isinstance(deceased, str) and deceased.strip().upper() in ('Y', 'YES', '1')):
        return 'Critical'
    if icu > 0:
        return 'Critical'
    if o2:
        return 'Severe'
    if los > 7:
        return 'Moderate'
    if los > 0:
        return 'Mild'
    return 'Unknown'


# ==============================================================================
# DATA PROFILING
# ==============================================================================

def _profile_dataset(df, name, logger):
    """Log a detailed profile of a loaded dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET PROFILE: {name}")
    logger.info(f"{'='*60}")
    logger.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    logger.info(f"  Columns: {list(df.columns)}")

    # Unique patients
    uin_col = None
    for candidate in ['uin', 'UIN', 'patient_id', 'PATIENT_ID']:
        if candidate in df.columns:
            uin_col = candidate
            break
    if uin_col:
        logger.info(f"  Unique patients ({uin_col}): {df[uin_col].nunique():,}")

    # Missingness
    logger.info(f"  Missing values:")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            logger.info(f"    {col}: {n_miss:,} ({n_miss/len(df)*100:.1f}%)")

    # Date columns - show range
    for col in df.columns:
        if 'date' in col.lower():
            dt = pd.to_datetime(df[col], errors='coerce')
            valid = dt.dropna()
            if len(valid) > 0:
                logger.info(f"  Date range [{col}]: {valid.min()} to {valid.max()}")

    # Categorical columns - show value counts (top 10)
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() < 30:
            vc = df[col].value_counts().head(10)
            logger.info(f"  Value counts [{col}]:")
            for val, cnt in vc.items():
                logger.info(f"    {val}: {cnt:,}")

    return


# ==============================================================================
# MAIN
# ==============================================================================

def run_step_10(config):
    # ------------------------------------------------------------------
    # 1. SETUP
    # ------------------------------------------------------------------
    processed_dir = config['paths']['processed_dir']
    results_dir = os.path.join(config['paths']['results_dir'], "step_10_enrichment")
    output_dir = os.path.join(processed_dir, "step_10_enriched")
    ensure_dir(results_dir)
    ensure_dir(output_dir)

    logger = setup_logger("step_10", results_dir)
    logger.info("=" * 70)
    logger.info("STEP 10: Vaccination, Severity & Serology Enrichment")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 2. LOAD CCI-ENRICHED COHORT (Step 8 output)
    # ------------------------------------------------------------------
    cohort_file = os.path.join(processed_dir, "step_8_cci", "cohort_enriched_cci.csv")
    if not os.path.exists(cohort_file):
        logger.error(f"Missing: {cohort_file}. Run Steps 1-8 first.")
        return None

    df = pd.read_csv(cohort_file)
    logger.info(f"Loaded CCI cohort: {len(df):,} patients, {len(df.columns)} columns")
    logger.info(f"  Groups: {df['group'].value_counts().to_dict()}")

    # Parse key dates
    df['covid_date'] = pd.to_datetime(df['covid_date'], errors='coerce')
    df['ihd_date'] = pd.to_datetime(
        df.get('ihd_date', df.get('discharge_date', pd.NaT)),
        errors='coerce'
    )
    # Reference date: covid_date for G1/G2, ihd_date for G3
    df['ref_date'] = df['covid_date'].combine_first(df['ihd_date'])

    target_uins = set(df['uin'].unique())
    logger.info(f"  Target patients: {len(target_uins):,}")

    # ------------------------------------------------------------------
    # 3. INITIALIZE CATALOG
    # ------------------------------------------------------------------
    catalog_path = os.path.join(os.getcwd(), config['paths']['catalog_config'])
    cat = DataCatalog(catalog_path)
    logger.info(f"DataCatalog: {cat}")

    # ==================================================================
    # 4. LOAD & PROFILE: COVIDFACILLOS (severity + vacc + race)
    # ==================================================================
    logger.info("\n" + "#" * 70)
    logger.info("LOADING: COVIDFACILLOS (COVID Facility LOS)")
    logger.info("#" * 70)

    try:
        df_facil = cat.load(config['datasets']['covid_facility_los'])
        _profile_dataset(df_facil, "COVIDFACILLOS", logger)

        # Filter to cohort patients
        df_facil = df_facil[df_facil['uin'].isin(target_uins)].copy()
        logger.info(f"  After filtering to cohort: {len(df_facil):,} rows, "
                     f"{df_facil['uin'].nunique():,} unique patients")

        # Handle duplicates: keep the record with the earliest notification date
        df_facil['notificationdate'] = pd.to_datetime(
            df_facil['notificationdate'], errors='coerce'
        )
        df_facil.sort_values('notificationdate', inplace=True)
        df_facil_dedup = df_facil.drop_duplicates(subset='uin', keep='first')
        logger.info(f"  After dedup (first notification): {len(df_facil_dedup):,} patients")

        # Extract severity columns
        severity_cols = ['uin', 'LOS', 'DaysInICU', 'Deceased',
                         'O2StartDate', 'O2EndDate', 'race']
        # Only keep columns that exist
        severity_cols = [c for c in severity_cols if c in df_facil_dedup.columns]
        df_severity = df_facil_dedup[severity_cols].copy()

        # Derive O2 requirement flag
        if 'O2StartDate' in df_severity.columns:
            df_severity['required_O2'] = df_severity['O2StartDate'].notnull()
        else:
            df_severity['required_O2'] = False

        # Extract vaccination columns from COVIDFACILLOS (backup, NIR is primary)
        facil_vacc_date_cols = [c for c in df_facil_dedup.columns if c.startswith('vacc_date')]
        facil_vacc_brand_cols = [c for c in df_facil_dedup.columns if c.startswith('vaccbrand')]
        df_facil_vacc = df_facil_dedup[['uin'] + facil_vacc_date_cols + facil_vacc_brand_cols].copy()

        del df_facil, df_facil_dedup
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to load COVIDFACILLOS: {e}")
        df_severity = pd.DataFrame(columns=['uin'])
        df_facil_vacc = pd.DataFrame(columns=['uin'])

    # ==================================================================
    # 5. LOAD & PROFILE: NIRListtruncated (primary vaccination source)
    # ==================================================================
    logger.info("\n" + "#" * 70)
    logger.info("LOADING: NIRListtruncated (National Immunisation Registry)")
    logger.info("#" * 70)

    try:
        df_nir = cat.load(config['datasets']['nir_vaccination'])
        _profile_dataset(df_nir, "NIRListtruncated", logger)

        # Filter to cohort
        df_nir = df_nir[df_nir['uin'].isin(target_uins)].copy()
        logger.info(f"  After filtering to cohort: {len(df_nir):,} rows, "
                     f"{df_nir['uin'].nunique():,} unique patients")

        # Dedup (should be 1 row per patient, but just in case)
        df_nir.drop_duplicates(subset='uin', keep='first', inplace=True)

        # This dataset has vacc_date1-6, vaccbrand1-6 (one more dose than COVIDFACILLOS)
        nir_vacc_date_cols = sorted([c for c in df_nir.columns if c.startswith('vacc_date')])
        nir_vacc_brand_cols = sorted([c for c in df_nir.columns if c.startswith('vaccbrand')])

        logger.info(f"  Vaccination date columns: {nir_vacc_date_cols}")
        logger.info(f"  Vaccination brand columns: {nir_vacc_brand_cols}")

        # Log vaccination coverage
        for col in nir_vacc_date_cols:
            n_has = df_nir[col].notnull().sum()
            logger.info(f"    {col}: {n_has:,} patients ({n_has/len(df_nir)*100:.1f}%)")

    except Exception as e:
        logger.error(f"Failed to load NIRListtruncated: {e}")
        df_nir = pd.DataFrame(columns=['uin'])
        nir_vacc_date_cols = []
        nir_vacc_brand_cols = []

    # ==================================================================
    # 6. LOAD & PROFILE: COVID Reinfections
    # ==================================================================
    logger.info("\n" + "#" * 70)
    logger.info("LOADING: COVID Reinfections")
    logger.info("#" * 70)

    try:
        df_reinf = cat.load(config['datasets']['covid_reinfections'])
        _profile_dataset(df_reinf, "COVID Reinfections", logger)

        df_reinf = df_reinf[df_reinf['uin'].isin(target_uins)].copy()
        logger.info(f"  After filtering to cohort: {len(df_reinf):,} rows, "
                     f"{df_reinf['uin'].nunique():,} unique patients")

        # Create reinfection flag: any patient appearing here has been reinfected
        reinf_uins = set(df_reinf['uin'].unique())

        # Extract useful severity data from reinfections too
        reinf_severity_cols = ['uin', 'DaysInICU', 'Deceased', 'reinfection_type']
        reinf_severity_cols = [c for c in reinf_severity_cols if c in df_reinf.columns]

        # For race: COVID Reinfections uses cat_gender, cat_race prefixes
        if 'cat_race' in df_reinf.columns:
            df_reinf_race = df_reinf[['uin', 'cat_race']].drop_duplicates(
                subset='uin', keep='first'
            ).rename(columns={'cat_race': 'race_reinf'})
        else:
            df_reinf_race = pd.DataFrame(columns=['uin', 'race_reinf'])

        del df_reinf
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to load COVID Reinfections: {e}")
        reinf_uins = set()
        df_reinf_race = pd.DataFrame(columns=['uin', 'race_reinf'])

    # ==================================================================
    # 7. LOAD & PROFILE: FacilityUtilizationLOSSubsequentRI
    # ==================================================================
    logger.info("\n" + "#" * 70)
    logger.info("LOADING: FacilityUtilizationLOSSubsequentRI")
    logger.info("#" * 70)

    try:
        df_facil_ri = cat.load(config['datasets']['facility_utilization_ri'])
        _profile_dataset(df_facil_ri, "FacilityUtilizationLOSSubsequentRI", logger)

        df_facil_ri = df_facil_ri[df_facil_ri['uin'].isin(target_uins)].copy()
        logger.info(f"  After filtering to cohort: {len(df_facil_ri):,} rows")

        # Add these UINs to reinfection set
        if len(df_facil_ri) > 0:
            reinf_uins.update(df_facil_ri['uin'].unique())

        del df_facil_ri
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to load FacilityUtilizationLOSSubsequentRI: {e}")

    # ==================================================================
    # 8. LOAD & PROFILE: Serology_Tests_COVID
    # ==================================================================
    logger.info("\n" + "#" * 70)
    logger.info("LOADING: Serology_Tests_COVID")
    logger.info("#" * 70)

    try:
        df_sero = cat.load(config['datasets']['serology_tests'])
        _profile_dataset(df_sero, "Serology_Tests_COVID", logger)

        df_sero = df_sero[df_sero['uin'].isin(target_uins)].copy()
        logger.info(f"  After filtering to cohort: {len(df_sero):,} rows, "
                     f"{df_sero['uin'].nunique():,} unique patients")

        # Keep the earliest serology test per patient
        if 'serologyswabdate' in df_sero.columns:
            df_sero['serologyswabdate'] = pd.to_datetime(
                df_sero['serologyswabdate'], errors='coerce'
            )
            df_sero.sort_values('serologyswabdate', inplace=True)

        sero_keep_cols = ['uin', 'serologyresult', 'serologyctvalue',
                          'serologyvalue', 'serologyswabdate',
                          'serologyresultindicator']
        sero_keep_cols = [c for c in sero_keep_cols if c in df_sero.columns]
        df_sero_dedup = df_sero[sero_keep_cols].drop_duplicates(
            subset='uin', keep='first'
        )
        logger.info(f"  After dedup (earliest test): {len(df_sero_dedup):,} patients")

        # Log serology result distribution
        if 'serologyresult' in df_sero_dedup.columns:
            logger.info(f"  Serology result distribution:")
            for val, cnt in df_sero_dedup['serologyresult'].value_counts().head(10).items():
                logger.info(f"    {val}: {cnt:,}")

        del df_sero
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to load Serology_Tests_COVID: {e}")
        df_sero_dedup = pd.DataFrame(columns=['uin'])

    # ==================================================================
    # 9. MERGE EVERYTHING ONTO COHORT
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MERGING ALL ENRICHMENT DATA ONTO COHORT")
    logger.info("=" * 70)

    n_before = len(df)

    # 9a. Merge severity (from COVIDFACILLOS)
    if len(df_severity) > 0 and 'uin' in df_severity.columns:
        drop_cols = [c for c in df_severity.columns if c in df.columns and c != 'uin']
        if drop_cols:
            logger.info(f"  Dropping existing columns before severity merge: {drop_cols}")
        df = df.drop(columns=[c for c in drop_cols if c != 'uin'], errors='ignore')
        df = df.merge(df_severity, on='uin', how='left')
        logger.info(f"  After severity merge: {len(df):,} rows")

    # 9b. Merge NIR vaccination (primary source, more doses tracked)
    if len(df_nir) > 0 and 'uin' in df_nir.columns:
        # Rename NIR columns to avoid collision with COVIDFACILLOS vacc cols
        nir_cols_to_merge = ['uin'] + nir_vacc_date_cols + nir_vacc_brand_cols
        nir_cols_to_merge = [c for c in nir_cols_to_merge if c in df_nir.columns]
        df_nir_merge = df_nir[nir_cols_to_merge].copy()

        # Use NIR prefix to distinguish
        rename_map = {}
        for c in nir_vacc_date_cols + nir_vacc_brand_cols:
            rename_map[c] = f"nir_{c}"
        df_nir_merge.rename(columns=rename_map, inplace=True)

        df = df.merge(df_nir_merge, on='uin', how='left')
        logger.info(f"  After NIR vaccination merge: {len(df):,} rows")
        del df_nir_merge

    # 9c. Merge race from reinfection dataset (supplement if missing)
    if len(df_reinf_race) > 0:
        df = df.merge(df_reinf_race, on='uin', how='left')
        # Fill missing race from reinfection data
        if 'race' in df.columns and 'race_reinf' in df.columns:
            df['race'] = df['race'].fillna(df['race_reinf'])
            df.drop(columns='race_reinf', inplace=True)

    # 9d. Merge serology
    if len(df_sero_dedup) > 0 and 'uin' in df_sero_dedup.columns:
        df = df.merge(df_sero_dedup, on='uin', how='left')
        logger.info(f"  After serology merge: {len(df):,} rows")

    # 9e. Reinfection flag
    df['is_reinfection'] = df['uin'].isin(reinf_uins).astype(int)
    logger.info(f"  Reinfection patients in cohort: {df['is_reinfection'].sum():,}")

    assert len(df) == n_before, \
        f"Row count changed during merges: {n_before} -> {len(df)}"

    # ==================================================================
    # 10. DERIVE VACCINATION VARIABLES
    # ==================================================================
    logger.info("\n" + "-" * 50)
    logger.info("Deriving vaccination variables")
    logger.info("-" * 50)

    # Use NIR vaccination data as primary (has up to 6 doses)
    nir_date_cols_in_df = sorted([c for c in df.columns if c.startswith('nir_vacc_date')])
    nir_brand_cols_in_df = sorted([c for c in df.columns if c.startswith('nir_vaccbrand')])

    if nir_date_cols_in_df:
        # Count doses before reference date
        vacc_date_df = df[nir_date_cols_in_df].copy()
        for col in nir_date_cols_in_df:
            vacc_date_df[col] = pd.to_datetime(vacc_date_df[col], errors='coerce')

        df['doses_before_ref'] = _count_doses_before_date(vacc_date_df, df['ref_date'])
        df['vaccinated_before_covid'] = (df['doses_before_ref'] >= 1).astype(int)
        df['fully_vaccinated_before_covid'] = (df['doses_before_ref'] >= 2).astype(int)

        # Primary vaccine brand
        if nir_brand_cols_in_df:
            df['vaccine_brand_primary'] = df.apply(
                lambda row: _get_primary_brand(row, nir_brand_cols_in_df), axis=1
            )

        logger.info(f"  Dose distribution before ref_date:")
        for n_doses in sorted(df['doses_before_ref'].unique()):
            cnt = (df['doses_before_ref'] == n_doses).sum()
            logger.info(f"    {n_doses} doses: {cnt:,} ({cnt/len(df)*100:.1f}%)")

        logger.info(f"  Vaccinated (≥1 dose): {df['vaccinated_before_covid'].sum():,} "
                     f"({df['vaccinated_before_covid'].mean()*100:.1f}%)")
        logger.info(f"  Fully vaccinated (≥2 doses): {df['fully_vaccinated_before_covid'].sum():,}")

        if 'vaccine_brand_primary' in df.columns:
            logger.info(f"  Vaccine brand distribution:")
            for brand, cnt in df['vaccine_brand_primary'].value_counts().head(10).items():
                logger.info(f"    {brand}: {cnt:,}")

    else:
        logger.warning("  No NIR vaccination columns found; skipping dose derivation")
        df['doses_before_ref'] = np.nan
        df['vaccinated_before_covid'] = np.nan
        df['fully_vaccinated_before_covid'] = np.nan

    # ==================================================================
    # 11. DERIVE SEVERITY CATEGORY
    # ==================================================================
    logger.info("\n" + "-" * 50)
    logger.info("Deriving severity category")
    logger.info("-" * 50)

    df['severity_category'] = df.apply(_assign_severity, axis=1)

    logger.info(f"  Severity distribution:")
    for cat_name, cnt in df['severity_category'].value_counts().items():
        logger.info(f"    {cat_name}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # By group
    for grp in ['Group 1', 'Group 2', 'Group 3']:
        sub = df[df['group'] == grp]
        if len(sub) > 0:
            logger.info(f"  {grp} severity:")
            for cat_name, cnt in sub['severity_category'].value_counts().items():
                logger.info(f"    {cat_name}: {cnt:,} ({cnt/len(sub)*100:.1f}%)")

    # ==================================================================
    # 12. ASSIGN VARIANT ERA
    # ==================================================================
    logger.info("\n" + "-" * 50)
    logger.info("Assigning variant eras")
    logger.info("-" * 50)

    df['variant_era'] = df['covid_date'].apply(_assign_era)

    logger.info(f"  Era distribution:")
    for era, cnt in df['variant_era'].value_counts().items():
        logger.info(f"    {era}: {cnt:,}")

    # By group
    for grp in ['Group 1', 'Group 2']:
        sub = df[df['group'] == grp]
        if len(sub) > 0:
            logger.info(f"  {grp} by era:")
            for era, cnt in sub['variant_era'].value_counts().items():
                logger.info(f"    {era}: {cnt:,}")

    # ==================================================================
    # 13. GENERATE SUMMARY TABLES
    # ==================================================================
    logger.info("\n" + "-" * 50)
    logger.info("Generating summary tables")
    logger.info("-" * 50)

    # 13a. Vaccination summary: by group x era
    covid_mask = df['group'].isin(['Group 1', 'Group 2'])
    if 'vaccinated_before_covid' in df.columns and df['vaccinated_before_covid'].notnull().any():
        vacc_summary = df[covid_mask].groupby(['group', 'variant_era']).agg(
            n_patients=('uin', 'count'),
            n_vaccinated=('vaccinated_before_covid', 'sum'),
            n_fully_vaccinated=('fully_vaccinated_before_covid', 'sum'),
            mean_doses=('doses_before_ref', 'mean'),
        ).reset_index()
        vacc_summary['pct_vaccinated'] = (
            vacc_summary['n_vaccinated'] / vacc_summary['n_patients'] * 100
        ).round(1)
        vacc_summary['pct_fully_vaccinated'] = (
            vacc_summary['n_fully_vaccinated'] / vacc_summary['n_patients'] * 100
        ).round(1)

        vacc_path = os.path.join(output_dir, "vaccination_summary.csv")
        vacc_summary.to_csv(vacc_path, index=False)
        logger.info(f"  Saved: {vacc_path}")
        logger.info(f"\n  Vaccination Summary:\n{vacc_summary.to_string(index=False)}")

    # 13b. Severity summary: by group x era
    if df['severity_category'].notnull().any():
        sev_summary = df[covid_mask].groupby(
            ['group', 'variant_era', 'severity_category']
        ).size().reset_index(name='count')

        sev_path = os.path.join(output_dir, "severity_summary.csv")
        sev_summary.to_csv(sev_path, index=False)
        logger.info(f"  Saved: {sev_path}")

    # ==================================================================
    # 14. SAVE FINAL ENRICHED COHORT
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SAVING FINAL ENRICHED COHORT")
    logger.info("=" * 70)

    # Drop intermediate NIR raw columns to keep output clean
    # (we derived doses_before_ref, vaccinated_before_covid, vaccine_brand_primary)
    drop_raw_vacc = [c for c in df.columns
                     if c.startswith('nir_vacc_date') or c.startswith('nir_vaccbrand')]
    # Also drop COVIDFACILLOS raw vacc columns if present
    drop_raw_vacc += [c for c in df.columns
                      if c.startswith('vacc_date') or c.startswith('vaccbrand')]
    if drop_raw_vacc:
        logger.info(f"  Dropping {len(drop_raw_vacc)} raw vaccination columns")
        df.drop(columns=drop_raw_vacc, inplace=True, errors='ignore')

    # Also drop O2 date columns (we have the derived flag)
    df.drop(columns=['O2StartDate', 'O2EndDate'], inplace=True, errors='ignore')

    output_path = os.path.join(output_dir, "cohort_tier3_ready.csv")
    save_with_report(df, output_path, "Tier 3 Ready Cohort (Step 10)", logger)
    logger.info(f"  Final shape: {df.shape}")

    # ==================================================================
    # 15. DETAILED DATA REPORT
    # ==================================================================
    logger.info("\n" + "-" * 50)
    logger.info("Writing enrichment data report")
    logger.info("-" * 50)

    report = []
    report.append("STEP 10: ENRICHMENT DATA REPORT")
    report.append("=" * 70)
    report.append(f"Cohort size: {len(df):,} patients")
    report.append(f"Columns: {len(df.columns)}")
    report.append("")

    report.append("GROUP BREAKDOWN:")
    for grp, cnt in df['group'].value_counts().items():
        report.append(f"  {grp}: {cnt:,}")

    report.append("")
    report.append("NEW VARIABLES ADDED:")
    new_vars = ['LOS', 'DaysInICU', 'Deceased', 'required_O2', 'race',
                'is_reinfection', 'doses_before_ref', 'vaccinated_before_covid',
                'fully_vaccinated_before_covid', 'vaccine_brand_primary',
                'severity_category', 'variant_era',
                'serologyresult', 'serologyctvalue', 'serologyvalue']
    for var in new_vars:
        if var in df.columns:
            n_valid = df[var].notnull().sum()
            report.append(f"  {var}: {n_valid:,} non-null ({n_valid/len(df)*100:.1f}%)")

    report.append("")
    report.append("VARIANT ERA (COVID patients only):")
    for era in ['Ancestral', 'Delta', 'Omicron', 'Unknown']:
        for grp in ['Group 1', 'Group 2']:
            n = len(df[(df['group'] == grp) & (df['variant_era'] == era)])
            if n > 0:
                report.append(f"  {grp} / {era}: {n:,}")

    report.append("")
    report.append("VACCINATION (COVID patients with ref_date):")
    covid_df = df[df['group'].isin(['Group 1', 'Group 2'])]
    if 'doses_before_ref' in covid_df.columns:
        for d in range(7):
            n = (covid_df['doses_before_ref'] == d).sum()
            if n > 0:
                report.append(f"  {d} doses before COVID: {n:,}")

    report.append("")
    report.append("SEVERITY CATEGORY:")
    for cat_name, cnt in df['severity_category'].value_counts().items():
        report.append(f"  {cat_name}: {cnt:,}")

    report.append("")
    report.append("RACE DISTRIBUTION:")
    if 'race' in df.columns:
        for race, cnt in df['race'].value_counts().head(10).items():
            report.append(f"  {race}: {cnt:,}")

    report.append("")
    report.append("SEROLOGY:")
    if 'serologyresult' in df.columns:
        for val, cnt in df['serologyresult'].value_counts().head(10).items():
            report.append(f"  {val}: {cnt:,}")

    report.append("")
    report.append("REINFECTION:")
    report.append(f"  Reinfected: {df['is_reinfection'].sum():,}")
    report.append(f"  Not reinfected: {(df['is_reinfection'] == 0).sum():,}")

    report.append("")
    report.append(f"OUTPUT: {output_path}")
    report.append("")
    report.append("COLUMN LIST:")
    for col in sorted(df.columns):
        report.append(f"  - {col} ({df[col].dtype})")

    report_path = os.path.join(results_dir, "enrichment_data_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"  Saved report: {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("STEP 10 COMPLETE")
    logger.info("  Output: cohort_tier3_ready.csv")
    logger.info("  Next:   Step 11 (Tier 3 era-stratified analysis)")
    logger.info("=" * 70)

    return df


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    run_step_10(conf)
