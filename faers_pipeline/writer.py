"""
writer.py
---------
Save processed DataFrames as versioned Parquet files and persist
the deduplication audit log as both JSON and CSV.

DECISION LOG — Output format choices
--------------------------------------

Parquet vs. CSV
  We output Parquet, not CSV, because:
  1. Parquet is ~3-10x smaller than CSV for FAERS data (column-oriented
     compression handles the many repeated string values well).
  2. Parquet preserves dtypes — no silent re-casting of IDs on reload.
  3. Parquet is the native format for BigQuery import (Phase 6 add-on).
  4. pandas, DuckDB, Spark, and BigQuery all read Parquet natively.
  Downside: not human-readable without tooling. We mitigate this by
  always providing a CSV export of the audit log and the data dictionary.

Versioning strategy
  Output filenames embed the pipeline run date:
    DEMO_deduplicated_v20240930.parquet
  "v" + the last quarter's end date (not today's date) so the version
  is tied to data coverage, not to when the pipeline was run.
  This allows multiple pipeline runs to coexist if we add new quarters.

Partitioning
  For Phase 1 we write a SINGLE file per file type spanning all quarters.
  Rationale: at ~17M DEMO records total for our scope, a single Parquet
  file is well within pandas/DuckDB comfort zone.
  Phase 6 (BigQuery) will use date-partitioned tables — partitioning
  is a BigQuery-side concern, not a local Parquet concern.

Compression
  snappy (default in pyarrow) — fast read/write, good compression.
  We do NOT use gzip because it prevents parallel reads in BigQuery.
"""

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .quarters import Quarter

logger = logging.getLogger(__name__)


def _version_tag(end_quarter: Quarter) -> str:
    """
    Version string tied to data coverage end date.
    e.g. Quarter(2024, 3) → 'v20240930'
    """
    # Map quarter to its last calendar day
    end_month = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
    return f"v{end_quarter.year}{end_month[end_quarter.q]}"


def save_parquet(
    df: pd.DataFrame,
    file_type: str,
    processed_dir: Path,
    end_quarter: Quarter,
    compression: str = "snappy",
) -> Path:
    """
    Save a DataFrame as a versioned Parquet file.

    Args:
        df:            DataFrame to save.
        file_type:     e.g. 'DEMO', 'DRUG', 'REAC'.
        processed_dir: Output directory.
        end_quarter:   Last quarter included (used for version tag).
        compression:   Parquet compression codec.

    Returns:
        Path to the written file.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    version   = _version_tag(end_quarter)
    filename  = f"{file_type}_deduplicated_{version}.parquet"
    out_path  = processed_dir / filename

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        out_path,
        compression=compression,
    )

    size_mb = out_path.stat().st_size / (1024 ** 2)
    n_rows  = len(df)
    logger.info(
        f"  SAVED {filename}: {n_rows:,} rows, {size_mb:.1f} MB"
    )
    return out_path


def save_audit_log(
    audit_records: list[dict],
    logs_dir: Path,
    end_quarter: Quarter,
) -> tuple[Path, Path]:
    """
    Save the deduplication audit log as both JSON and CSV.

    JSON: machine-readable, used by the EDA charting script.
    CSV:  human-readable, for quick Excel/Sheets review.

    Returns:
        Tuple of (json_path, csv_path).
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    version  = _version_tag(end_quarter)
    json_path = logs_dir / f"dedup_audit_{version}.json"
    csv_path  = logs_dir / f"dedup_audit_{version}.csv"

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(
            {
                "pipeline_version" : "0.1.0",
                "generated_at"     : datetime.now(timezone.utc).isoformat(),
                "scope_end_quarter": str(end_quarter),
                "total_quarters"   : len(audit_records),
                "records"          : audit_records,
            },
            f,
            indent=2,
        )

    # Write CSV
    df_audit = pd.DataFrame(audit_records)
    df_audit.to_csv(csv_path, index=False)

    logger.info(f"  AUDIT LOG → {json_path.name} + {csv_path.name}")
    return json_path, csv_path


def save_data_dictionary(
    dataframes: dict[str, pd.DataFrame],
    docs_dir: Path,
) -> Path:
    """
    Generate a data dictionary CSV documenting every column in the
    processed output: file type, column name, dtype, non-null count,
    sample values.

    This is a portfolio artifact — it shows you document your work.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for file_type, df in dataframes.items():
        if df is None or df.empty:
            continue
        for col in df.columns:
            if col.startswith("_"):
                continue  # skip internal columns
            non_null  = df[col].notna().sum()
            null_rate = 1 - (non_null / len(df)) if len(df) > 0 else 0
            samples   = (
                df[col]
                .dropna()
                .astype(str)
                .unique()[:5]
                .tolist()
            )
            rows.append({
                "file_type"   : file_type,
                "column"      : col,
                "dtype"       : str(df[col].dtype),
                "non_null_n"  : int(non_null),
                "null_rate"   : round(float(null_rate), 4),
                "sample_values": " | ".join(samples),
            })

    df_dict = pd.DataFrame(rows)
    out_path = docs_dir / "data_dictionary.csv"
    df_dict.to_csv(out_path, index=False)
    logger.info(f"  DATA DICTIONARY → {out_path.name} ({len(rows)} columns documented)")
    return out_path
