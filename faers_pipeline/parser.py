"""
parser.py
---------
Extract and parse all 7 FAERS file types from quarterly ZIP files.

DECISION LOG — Parsing choices
--------------------------------

1. Encoding
   FAERS ASCII files are ISO-8859-1 (Latin-1), NOT UTF-8.
   Using UTF-8 causes silent data corruption on drug names, reporter
   addresses, and narrative fields that contain accented characters.
   Reference: FDA FAERS ASCII data dictionary, p.2.

2. Delimiter
   Files use the '$' character as delimiter (not pipe '|' as commonly
   assumed in older documentation). Post-2012 FAERS files consistently
   use '$'. Pre-2012 AERS files used '\\t' (tab). We handle both.

3. Header row
   Every file has a header row. Column names are lowercase in the
   raw files but we standardize to lowercase throughout.

4. Dtype strategy
   We read everything as strings initially, then cast specific columns.
   Reason: many ID fields (caseid, primaryid) are numeric but have
   leading zeros in some quarters. String-first prevents silent
   truncation of edge cases.

5. What we keep
   We read ALL columns from each file type. Filtering to GLP-1 drugs
   happens in the deduplication / normalization stage, not here.
   Reason: the parser is a generic FAERS reader; drug scoping is
   a separate concern with its own decision log.

6. File naming quirks
   - Pre-2014: files may be named DEMO12Q1.TXT (uppercase extension)
   - Some quarters have DEMO12Q1.txt and DEMO12Q1.TXT both present (take first)
   - Some quarters have files in the root of the ZIP (not under ASCII/)
   We handle all three cases.
"""

import logging
import zipfile
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

from .quarters import Quarter, FILE_TYPES
from .schema import COLUMN_RENAMES, ENSURE_COLUMNS, is_aers_era

logger = logging.getLogger(__name__)


# ── Column dtypes for controlled casting post-read ──────────────────────────
# These are cast AFTER initial string read.
# Everything else stays as object (string).

NUMERIC_COLS = {
    "DEMO": ["primaryid", "caseid", "caseversion", "age", "wt"],
    "DRUG": ["primaryid", "caseid", "drug_seq"],
    "REAC": ["primaryid", "caseid"],
    "OUTC": ["primaryid", "caseid"],
    "THER": ["primaryid", "caseid", "drug_seq"],
    "RPSR": ["primaryid", "caseid"],
    "INDI": ["primaryid", "caseid", "drug_seq"],
}


def _find_file_in_zip(
    zf: zipfile.ZipFile,
    file_type: str,
    suffix: str,
) -> Optional[str]:
    """
    Locate the right .txt file inside a ZIP for a given file type.
    Handles case variations and both root and ASCII/ subdirectory.

    Args:
        zf:        Open ZipFile object.
        file_type: e.g. 'DEMO', 'DRUG', etc.
        suffix:    Quarter suffix e.g. '24Q3'.

    Returns:
        The zip member path string, or None if not found.
    """
    candidates = [
        f"ASCII/{file_type}{suffix}.txt",
        f"ascii/{file_type}{suffix}.txt",
        f"ASCII/{file_type}{suffix}.TXT",
        f"{file_type}{suffix}.txt",
        f"{file_type}{suffix}.TXT",
    ]

    names_lower = {n.lower(): n for n in zf.namelist()}

    for candidate in candidates:
        if candidate in zf.namelist():
            return candidate
        # Case-insensitive fallback
        if candidate.lower() in names_lower:
            return names_lower[candidate.lower()]

    # Last resort: find any file starting with the type name
    for name in zf.namelist():
        basename = name.split("/")[-1].upper()
        if basename.startswith(file_type) and basename.endswith(".TXT"):
            return name

    return None


def _detect_delimiter(header_line: str) -> str:
    """
    Auto-detect delimiter from the header row.
    FAERS uses '$'; legacy AERS used tab.
    """
    if "$" in header_line:
        return "$"
    if "\t" in header_line:
        return "\t"
    if "|" in header_line:
        return "|"
    return "$"  # default


def parse_file_type(
    zip_path: Path,
    quarter: Quarter,
    file_type: str,
) -> Optional[pd.DataFrame]:
    """
    Extract one file type from a quarterly ZIP and return as DataFrame.

    Returns None if the file is not found in the ZIP (some early quarters
    are missing certain file types — log and continue).
    """
    suffix = quarter.txt_suffix()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            member = _find_file_in_zip(zf, file_type, suffix)

            if member is None:
                logger.warning(
                    f"  {file_type} not found in {zip_path.name} "
                    f"(suffix={suffix}) — skipping"
                )
                return None

            # Read raw bytes, decode as Latin-1
            raw_bytes = zf.read(member)
            raw_text  = raw_bytes.decode("iso-8859-1", errors="replace")

            # Detect delimiter from header
            first_line = raw_text.split("\n")[0]
            delimiter  = _detect_delimiter(first_line)

            # Parse into DataFrame
            df = pd.read_csv(
                StringIO(raw_text),
                sep=delimiter,
                dtype=str,          # All string initially
                low_memory=False,
                on_bad_lines="warn",
            )

            # Normalize column names: strip whitespace, lowercase
            df.columns = [c.strip().lower() for c in df.columns]

            # ── AERS → FAERS schema normalization ────────────────────────────
            # Pre-2012 AERS files use different column names than FAERS.
            # Rename them immediately so all downstream code is era-agnostic.
            # DECISION: rename at parse time, not at dedup time, so every
            # module downstream can assume FAERS column names unconditionally.
            if is_aers_era(quarter):
                renames = COLUMN_RENAMES.get(file_type, {})
                # Only rename columns that actually exist in the DataFrame
                actual_renames = {k: v for k, v in renames.items() if k in df.columns}
                if actual_renames:
                    df = df.rename(columns=actual_renames)
                    logger.debug(
                        f"  SCHEMA {file_type} {quarter}: renamed "
                        + ", ".join(f"{k}→{v}" for k, v in actual_renames.items())
                    )

            # Ensure required downstream columns exist (add as None if absent)
            # This handles FAERS-only columns missing from AERS files
            # (e.g. prod_ai, age_grp, reporter_country).
            for col, default in ENSURE_COLUMNS.get(file_type, {}).items():
                if col not in df.columns:
                    df[col] = default

            # Strip whitespace from all string values
            str_cols = df.select_dtypes(include=["object", "string"]).columns
            df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

            # Cast numeric columns where safe (coerce errors → NaN)
            for col in NUMERIC_COLS.get(file_type, []):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Add quarter metadata columns
            df["_quarter"]   = str(quarter)
            df["_file_type"] = file_type

            logger.info(
                f"  Parsed {file_type:<4} {quarter}: "
                f"{len(df):>7,} rows, {len(df.columns)} cols"
            )
            return df

    except zipfile.BadZipFile:
        logger.error(f"  BAD ZIP: {zip_path.name} — skipping {file_type}")
        return None
    except Exception as e:
        logger.error(f"  ERROR parsing {file_type} from {zip_path.name}: {e}")
        return None


def parse_quarter(
    zip_path: Path,
    quarter: Quarter,
    file_types: list[str] = None,
) -> dict[str, Optional[pd.DataFrame]]:
    """
    Parse all file types from one quarterly ZIP.

    Returns:
        Dict mapping file_type → DataFrame (or None if file missing).
    """
    file_types = file_types or FILE_TYPES
    return {
        ft: parse_file_type(zip_path, quarter, ft)
        for ft in file_types
    }
