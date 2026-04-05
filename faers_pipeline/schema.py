"""
schema.py
---------
Column schema normalization for the AERS → FAERS era transition.

DECISION LOG — Why this file exists
--------------------------------------
FDA ran two separate adverse event reporting systems:

  AERS  (Adverse Event Reporting System)
        Data era: 2004 Q1 → 2012 Q2
        Key columns: isr, case, gndr_cod, foll_seq

  FAERS (FDA Adverse Event Reporting System)
        Data era: 2012 Q3 → 2025 Q4
        Key columns: primaryid, caseid, sex, caseversion

  AEMS  (Adverse Event Monitoring System)
        Data era: 2026 Q1 → present
        Key columns: same as FAERS (no schema change, just rebrand)

The column names changed when FDA rebuilt the system. The data is
semantically equivalent — 'isr' and 'primaryid' are the same concept
(a unique identifier for one submitted report). But the name change
means any code that assumes FAERS column names crashes on AERS data.

Our scope starts at 2005 Q2, so we have 28 AERS quarters
(2005 Q2 → 2012 Q2) and 50 FAERS quarters (2012 Q3 → 2024 Q3,
excluding the missing 2012 Q3).

Strategy: rename AERS columns to their FAERS equivalents immediately
after parsing, before any other processing. All downstream code
(deduplicator, normalizer, writer) then works with FAERS names only.

Also handle: missing 2012 Q3
  FDA never published the 2012 Q3 quarterly file. This quarter falls
  exactly at the AERS→FAERS transition and is simply absent from the
  FIS download server (returns 404). The pipeline must skip it
  gracefully. We mark it in KNOWN_MISSING_QUARTERS so the download
  step doesn't retry it and the parse step doesn't warn about it.

Sources
-------
- FDA AERS ASCII Data Dictionary (legacy, on FDA website)
- FDA FAERS ASCII Data Dictionary (current, on FDA website)
- Potter et al. 2025: FAERS Essentials (documents the transition)
- PLoS One 2025: FAERS study noting isr→primaryid mapping
"""

from .quarters import Quarter

# ── Known missing quarters ────────────────────────────────────────────────────

KNOWN_MISSING_QUARTERS: frozenset[Quarter] = frozenset({
    # 2012 Q3: FDA never published this quarter.
    # The AERS→FAERS system transition occurred Sept 10, 2012.
    # The Q3 file (covering July–Sept 2012) was never released separately.
    # Reports from this period appear in later cumulative files.
    Quarter(2012, 3),
})

# ── Era detection ─────────────────────────────────────────────────────────────

def is_aers_era(quarter: Quarter) -> bool:
    """
    Return True if this quarter's data uses the legacy AERS schema.
    AERS era: 2004 Q1 → 2012 Q2 (inclusive).
    """
    return (quarter.year, quarter.q) <= (2012, 2)


# ── Column rename maps ────────────────────────────────────────────────────────
# Maps: old_aers_name → new_faers_name
# Only columns that actually changed name are listed.
# Columns that exist in AERS but not FAERS are kept as-is (they won't
# interfere with downstream processing since we only look for specific cols).
# Columns new in FAERS but absent in AERS will be NaN after rename.

DEMO_COLUMN_RENAMES: dict[str, str] = {
    "isr"     : "primaryid",    # Individual Safety Report number
    "case"    : "caseid",       # Case identifier
    "gndr_cod": "sex",          # Gender code → sex
    "foll_seq": "caseversion",  # Follow-up sequence → caseversion
}

DRUG_COLUMN_RENAMES: dict[str, str] = {
    "isr": "primaryid",
    # Note: AERS DRUG has no 'caseid' or 'prod_ai' columns.
    # After rename, caseid and prod_ai will be absent (NaN-filled downstream).
}

REAC_COLUMN_RENAMES: dict[str, str] = {
    "isr": "primaryid",
    # 'pt' (MedDRA preferred term) is the same in both eras.
}

OUTC_COLUMN_RENAMES: dict[str, str] = {
    "isr": "primaryid",
    # 'outc_cod' is the same in both eras.
}

THER_COLUMN_RENAMES: dict[str, str] = {
    "isr": "primaryid",
    # drug_seq, start_dt, end_dt, dur, dur_cod same in both eras.
}

RPSR_COLUMN_RENAMES: dict[str, str] = {
    "isr": "primaryid",
    # rpsr_cod same in both eras.
}

INDI_COLUMN_RENAMES: dict[str, str] = {
    "isr": "primaryid",
    # indi_pt same in both eras.
}

# Dispatch table: file_type → rename map
COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "DEMO": DEMO_COLUMN_RENAMES,
    "DRUG": DRUG_COLUMN_RENAMES,
    "REAC": REAC_COLUMN_RENAMES,
    "OUTC": OUTC_COLUMN_RENAMES,
    "THER": THER_COLUMN_RENAMES,
    "RPSR": RPSR_COLUMN_RENAMES,
    "INDI": INDI_COLUMN_RENAMES,
}

# ── Ensure required columns exist ─────────────────────────────────────────────
# After rename, some FAERS columns may still be absent from AERS data.
# We add them as NaN columns so downstream code doesn't crash on missing cols.

DEMO_ENSURE_COLUMNS: dict[str, object] = {
    "primaryid"  : None,
    "caseid"     : None,
    "fda_dt"     : None,
    "caseversion": None,
    "age"        : None,
    "sex"        : None,
    # FAERS-only columns — absent in AERS, filled with None
    "age_grp"         : None,
    "init_fda_dt"     : None,
    "reporter_country": None,
    "occr_country"    : None,
}

DRUG_ENSURE_COLUMNS: dict[str, object] = {
    "primaryid": None,
    "caseid"   : None,
    "drug_seq" : None,
    "role_cod" : None,
    "drugname" : None,
    # prod_ai is FAERS-only — absent in AERS, normalizer handles None gracefully
    "prod_ai"  : None,
}

ENSURE_COLUMNS: dict[str, dict[str, object]] = {
    "DEMO": DEMO_ENSURE_COLUMNS,
    "DRUG": DRUG_ENSURE_COLUMNS,
    # Other file types only need primaryid guaranteed
    "REAC": {"primaryid": None, "caseid": None, "pt": None},
    "OUTC": {"primaryid": None, "caseid": None, "outc_cod": None},
    "THER": {"primaryid": None, "caseid": None, "drug_seq": None},
    "RPSR": {"primaryid": None, "caseid": None, "rpsr_cod": None},
    "INDI": {"primaryid": None, "caseid": None, "drug_seq": None, "indi_pt": None},
}
