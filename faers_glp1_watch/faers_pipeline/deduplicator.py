"""
deduplicator.py
---------------
Implement FDA-recommended deduplication of FAERS DEMO records,
log every removal decision, and produce a quarterly audit trail.

DECISION LOG — Deduplication strategy
---------------------------------------

Background
  FAERS contains duplicate case reports because:
  1. Manufacturers submit the same report as different case versions
     (CASEVERSION field tracks this).
  2. The same case may be re-submitted with corrections.
  3. Regulatory authorities in other countries submit reports already
     present from the manufacturer (cross-regulatory duplicates).
  4. Report amendments — FDA receives 'follow-up' reports that update
     an existing case.

  Failing to deduplicate causes signal inflation: a single real event
  appears as 2–5 records, artificially boosting ROR/PRR counts.

FDA-recommended rule (Reference: Potter et al. 2025, Clin Pharmacol Ther;
  FDA FAERS Best Practices document, 2024)
  ─────────────────────────────────────────────────────────────────────
  Step 1: For each CASEID, keep the record with the most recent FDA_DT
          (the date FDA received/processed the report).
  Step 2: If two records share the same CASEID AND FDA_DT, keep the one
          with the highest PRIMARYID (larger = more recently assigned).

  This ensures:
  - The most current version of each case is retained.
  - In case of ties, the most recently registered report wins.
  - We do NOT use CASEVERSION for deduplication because it is often
    missing, inconsistent across reporting sources, and not the field
    FDA's own surveillance team uses for this purpose.

What we deduplicate
  Only DEMO (demographics) is deduplicated here — it is the case-level
  file. DRUG, REAC, OUTC, THER, RPSR, INDI are joined to DEMO via
  PRIMARYID after deduplication, automatically inheriting the filtering.
  This is the correct approach: dedup at the case level, then pull all
  related records for surviving cases.

What we log per quarter
  - Total raw records
  - Records removed in Step 1 (older FDA_DT)
  - Records removed in Step 2 (lower PRIMARYID on tied FDA_DT)
  - Final record count
  - Deduplication rate (% removed)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_fda_dt(series: pd.Series) -> pd.Series:
    """
    Parse the FDA_DT field to a sortable integer (YYYYMMDD).

    DECISION: FDA_DT is stored as YYYYMMDD in some quarters and
    YYYYMMDDHHMMSS in others. We normalize to the date portion only
    (first 8 chars) and convert to integer for fast comparison.
    Missing values → 0 (treated as oldest, so they lose any tie).
    """
    parsed = (
        series
        .astype(str)
        .str.strip()
        .str[:8]           # Take date portion only
        .str.replace(r"\D", "", regex=True)  # Remove any non-digit chars
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return parsed


def deduplicate_demo(
    df: pd.DataFrame,
    quarter_label: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply FDA-recommended 2-step deduplication to a DEMO DataFrame.

    Args:
        df:            Raw DEMO DataFrame for one quarter (or combined).
        quarter_label: String label for logging (e.g. '2024Q3').

    Returns:
        Tuple of:
          - Deduplicated DataFrame
          - Audit dict with row counts at each step
    """
    n_raw = len(df)

    # Verify required columns exist
    required = {"primaryid", "caseid", "fda_dt"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DEMO file for {quarter_label} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # ── Normalize key columns ─────────────────────────────────────────────────
    df = df.copy()
    df["_fda_dt_int"]   = _parse_fda_dt(df["fda_dt"])
    df["_primaryid_int"] = pd.to_numeric(df["primaryid"], errors="coerce").fillna(0).astype(int)
    df["_caseid_str"]   = df["caseid"].astype(str).str.strip()

    # ── Step 1: Per CASEID, keep latest FDA_DT ────────────────────────────────
    # Find the max FDA_DT for each case
    max_fda_dt = (
        df.groupby("_caseid_str")["_fda_dt_int"]
        .transform("max")
    )
    mask_step1  = df["_fda_dt_int"] == max_fda_dt
    df_step1    = df[mask_step1].copy()
    n_after_s1  = len(df_step1)
    removed_s1  = n_raw - n_after_s1

    logger.debug(
        f"  {quarter_label} Step 1: {n_raw:,} → {n_after_s1:,} "
        f"(removed {removed_s1:,} older FDA_DT records)"
    )

    # ── Step 2: Among ties on FDA_DT, keep highest PRIMARYID ─────────────────
    max_primaryid = (
        df_step1.groupby("_caseid_str")["_primaryid_int"]
        .transform("max")
    )
    mask_step2   = df_step1["_primaryid_int"] == max_primaryid
    df_deduped   = df_step1[mask_step2].copy()

    # Handle edge case: if duplicates remain after step 2 (identical PRIMARYID
    # on same CASEID+FDA_DT — extremely rare but possible with data errors),
    # keep first occurrence only.
    df_deduped = df_deduped.drop_duplicates(subset=["_caseid_str"], keep="first")

    n_final    = len(df_deduped)
    removed_s2 = n_after_s1 - n_final

    logger.debug(
        f"  {quarter_label} Step 2: {n_after_s1:,} → {n_final:,} "
        f"(removed {removed_s2:,} lower PRIMARYID ties)"
    )

    # ── Drop helper columns ───────────────────────────────────────────────────
    df_deduped = df_deduped.drop(
        columns=["_fda_dt_int", "_primaryid_int", "_caseid_str"],
        errors="ignore",
    )

    # ── Audit record ─────────────────────────────────────────────────────────
    dedup_rate = (n_raw - n_final) / n_raw if n_raw > 0 else 0.0
    audit = {
        "quarter"      : quarter_label,
        "n_raw"        : n_raw,
        "n_after_step1": n_after_s1,
        "removed_step1": removed_s1,
        "removed_step2": removed_s2,
        "n_final"      : n_final,
        "dedup_rate"   : round(dedup_rate, 4),
    }

    logger.info(
        f"  DEDUP {quarter_label}: "
        f"{n_raw:,} raw → {n_final:,} unique cases "
        f"({dedup_rate:.1%} removed)"
    )

    return df_deduped, audit


def filter_related_by_primaryid(
    df: pd.DataFrame,
    surviving_primaryids: set,
    file_type: str,
    quarter_label: str,
) -> pd.DataFrame:
    """
    Filter a non-DEMO file type (DRUG, REAC, etc.) to only rows whose
    PRIMARYID appears in the deduplicated DEMO set.

    DECISION: We filter by PRIMARYID (not CASEID) because PRIMARYID is
    the unique report-level key that maps 1:1 to a DEMO record. CASEID
    can map to multiple PRIMARYIDs across case versions — the dedup
    already selected exactly one PRIMARYID per CASEID, so filtering
    related files by that exact PRIMARYID set is the correct join.
    """
    if "primaryid" not in df.columns:
        logger.warning(
            f"  {file_type} {quarter_label}: no 'primaryid' column — "
            f"returning unfiltered"
        )
        return df

    n_before = len(df)
    df_primaryid = pd.to_numeric(df["primaryid"], errors="coerce")
    mask         = df_primaryid.isin(surviving_primaryids)
    df_filtered  = df[mask].copy()
    n_after      = len(df_filtered)

    logger.debug(
        f"  FILTER {file_type} {quarter_label}: "
        f"{n_before:,} → {n_after:,} rows "
        f"({n_before - n_after:,} orphaned records removed)"
    )
    return df_filtered
