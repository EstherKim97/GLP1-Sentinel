"""
pipeline.py
-----------
Phase 1 orchestrator: Download → Parse → Deduplicate → Write Parquet.

This is the single entry point for running the full Phase 1 pipeline.
It coordinates all modules and produces:
  1. Deduplicated Parquet files per file type (data/processed/)
  2. Deduplication audit log (data/logs/)
  3. Data dictionary (docs/)

Usage (from project root):
  python -m faers_pipeline.pipeline --help
  python -m faers_pipeline.pipeline --dry-run
  python -m faers_pipeline.pipeline
  python -m faers_pipeline.pipeline --quarters 2023Q1 2023Q2 2024Q3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .quarters import Quarter, ALL_QUARTERS, SCOPE_START, SCOPE_END, FILE_TYPES
from .downloader import download_all
from .parser import parse_quarter
from .deduplicator import deduplicate_demo, filter_related_by_primaryid
from .writer import save_parquet, save_audit_log, save_data_dictionary

# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(logs_dir: Path, verbose: bool = False) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / "pipeline.log", mode="a"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ── Quarter parsing helper ────────────────────────────────────────────────────

def _parse_quarter_arg(s: str) -> Quarter:
    """Parse '2024Q3' or '2024q3' → Quarter(2024, 3)."""
    s = s.upper().replace(" ", "")
    if "Q" not in s:
        raise argparse.ArgumentTypeError(f"Invalid quarter format: {s!r}. Use e.g. 2024Q3")
    year_str, q_str = s.split("Q")
    return Quarter(int(year_str), int(q_str))


# ── Phase 1 core logic ────────────────────────────────────────────────────────

def run_phase1(
    project_root: Path,
    quarters: list[Quarter] = None,
    dry_run: bool = False,
    force_download: bool = False,
    skip_download: bool = False,
    verbose: bool = False,
) -> None:
    """
    Execute Phase 1: Download → Parse → Deduplicate → Write.

    Args:
        project_root:    Root directory of the project.
        quarters:        Quarters to process (default: all in scope).
        dry_run:         Show what would be downloaded without doing it.
        force_download:  Re-download even if ZIP already exists.
        skip_download:   Skip download step (use existing ZIPs).
        verbose:         Enable DEBUG logging.
    """
    raw_dir       = project_root / "data" / "raw"
    interim_dir   = project_root / "data" / "interim"
    processed_dir = project_root / "data" / "processed"
    logs_dir      = project_root / "data" / "logs"
    docs_dir      = project_root / "docs"

    _setup_logging(logs_dir, verbose=verbose)
    logger = logging.getLogger(__name__)

    quarters = quarters or ALL_QUARTERS
    end_q    = quarters[-1]

    logger.info("=" * 60)
    logger.info("FAERS-GLP1-Watch  |  Phase 1: Ingestion & Deduplication")
    logger.info("=" * 60)
    logger.info(f"Scope   : {quarters[0]} → {end_q}  ({len(quarters)} quarters)")
    logger.info(f"Dry run : {dry_run}")
    logger.info(f"Root    : {project_root}")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if not skip_download:
        logger.info("\n── Step 1: Download ZIPs ──────────────────────────────────")
        download_results = download_all(
            raw_dir  = raw_dir,
            quarters = quarters,
            dry_run  = dry_run,
            force    = force_download,
        )
        if dry_run:
            logger.info("Dry run complete. Exiting.")
            return

        failed = [r for r in download_results if r["status"] == "failed"]
        if failed:
            logger.warning(
                f"  {len(failed)} quarter(s) failed to download: "
                + ", ".join(r["quarter"] for r in failed)
            )
    else:
        logger.info("\n── Step 1: SKIPPED (--skip-download) ─────────────────────")

    # ── Step 2 & 3: Parse + Deduplicate + Collect ────────────────────────────
    logger.info("\n── Step 2–3: Parse, Deduplicate, Collect ──────────────────")

    # Accumulators — one list per file type
    accumulated: dict[str, list[pd.DataFrame]] = {ft: [] for ft in FILE_TYPES}
    audit_records: list[dict] = []

    for quarter in quarters:
        zip_path = raw_dir / quarter.zip_filename()

        if not zip_path.exists():
            logger.warning(f"  MISSING ZIP: {zip_path.name} — skipping quarter")
            continue

        logger.info(f"\n  Processing {quarter.label()} ...")

        # Parse all 7 file types
        parsed = parse_quarter(zip_path, quarter)

        # Must have DEMO to proceed
        demo = parsed.get("DEMO")
        if demo is None or demo.empty:
            logger.warning(f"  DEMO missing for {quarter} — skipping entire quarter")
            continue

        # Deduplicate DEMO
        demo_deduped, audit = deduplicate_demo(demo, str(quarter))
        audit_records.append(audit)
        accumulated["DEMO"].append(demo_deduped)

        # Get surviving PRIMARYID set for filtering related files
        surviving_ids = set(
            pd.to_numeric(demo_deduped["primaryid"], errors="coerce").dropna().astype(int)
        )

        # Filter and accumulate related file types
        for ft in FILE_TYPES:
            if ft == "DEMO":
                continue
            df = parsed.get(ft)
            if df is not None and not df.empty:
                df_filtered = filter_related_by_primaryid(df, surviving_ids, ft, str(quarter))
                accumulated[ft].append(df_filtered)

    if not any(accumulated.values()):
        logger.error("No data parsed. Check that ZIP files are present in data/raw/")
        return

    # ── Step 4: Concatenate all quarters per file type ────────────────────────
    logger.info("\n── Step 4: Concatenate & Write Parquet ────────────────────")

    all_dataframes: dict[str, pd.DataFrame] = {}

    for ft in FILE_TYPES:
        frames = [f for f in accumulated[ft] if f is not None and not f.empty]
        if not frames:
            logger.warning(f"  {ft}: no data collected across all quarters")
            continue

        combined = pd.concat(frames, ignore_index=True)
        all_dataframes[ft] = combined
        logger.info(f"  {ft}: {len(combined):,} total rows across {len(frames)} quarters")

        # Save as Parquet
        save_parquet(combined, ft, processed_dir, end_q)

    # ── Step 5: Save audit log ────────────────────────────────────────────────
    logger.info("\n── Step 5: Save Audit Log ─────────────────────────────────")
    save_audit_log(audit_records, logs_dir, end_q)

    # ── Step 6: Save data dictionary ─────────────────────────────────────────
    logger.info("\n── Step 6: Save Data Dictionary ───────────────────────────")
    save_data_dictionary(all_dataframes, docs_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 complete")
    logger.info(f"  Quarters processed : {len(audit_records)}")
    if audit_records:
        total_raw    = sum(a["n_raw"] for a in audit_records)
        total_final  = sum(a["n_final"] for a in audit_records)
        total_removed = total_raw - total_final
        logger.info(f"  Total raw cases    : {total_raw:,}")
        logger.info(f"  Total after dedup  : {total_final:,}")
        logger.info(f"  Total removed      : {total_removed:,} ({total_removed/total_raw:.1%})")
    logger.info(f"  Parquet files      : {processed_dir}")
    logger.info(f"  Audit log          : {logs_dir}")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "FAERS-GLP1-Watch — Phase 1: Download, parse, deduplicate, "
            "and write versioned Parquet files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what would be downloaded (no actual download)
  python -m faers_pipeline.pipeline --dry-run

  # Run full pipeline (downloads if ZIPs missing, uses cache otherwise)
  python -m faers_pipeline.pipeline

  # Run only specific quarters (useful for testing)
  python -m faers_pipeline.pipeline --quarters 2024Q1 2024Q2 2024Q3

  # Re-download all ZIPs even if they exist
  python -m faers_pipeline.pipeline --force-download

  # Parse/process only (skip download, ZIPs already in data/raw/)
  python -m faers_pipeline.pipeline --skip-download

  # Verbose debug logging
  python -m faers_pipeline.pipeline --verbose
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print download URLs without fetching files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download ZIPs even if they already exist.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step; parse ZIPs already in data/raw/.",
    )
    parser.add_argument(
        "--quarters",
        nargs="+",
        type=_parse_quarter_arg,
        metavar="YYYYQN",
        help="Specific quarters to process (e.g. 2024Q1 2024Q2). Default: full scope.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    run_phase1(
        project_root   = args.root.resolve(),
        quarters       = args.quarters,
        dry_run        = args.dry_run,
        force_download = args.force_download,
        skip_download  = args.skip_download,
        verbose        = args.verbose,
    )


if __name__ == "__main__":
    main()
