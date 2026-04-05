"""
pipeline.py
-----------
Full pipeline orchestrator: Phases 1, 2, and 3.

Phase 1: Download → Parse → Schema-normalize → Deduplicate → Parquet
Phase 2: Drug name normalization → GLP-1 scoping → PS filter
Phase 3: MedDRA SOC join → Missingness audit → Scoped output Parquet

Usage (from project root):
  python -m faers_pipeline.pipeline --help
  python -m faers_pipeline.pipeline --dry-run
  python -m faers_pipeline.pipeline
  python -m faers_pipeline.pipeline --quarters 2023Q1 2023Q2 2024Q3
  python -m faers_pipeline.pipeline --mdhier /path/to/mdhier.asc
  python -m faers_pipeline.pipeline --skip-download
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
from .normalizer import normalize_drug_file, filter_to_glp1_ps, build_normalization_audit
from .meddra import join_meddra, soc_summary
from .signal_detection import run_signal_detection
from .eda_report import build_report
from .writer import save_parquet, save_audit_log, save_data_dictionary, _version_tag

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
    project_root   : Path,
    quarters       : list[Quarter] = None,
    dry_run        : bool = False,
    force_download : bool = False,
    skip_download  : bool = False,
    verbose        : bool = False,
    mdhier_path    : Path | None = None,
    no_report      : bool = False,
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

    # ── Step 6: Phase 2 — Drug name normalization & GLP-1 scoping ────────────
    logger.info("\n── Step 6: Phase 2 — Drug Normalization & GLP-1 Scope ────")

    norm_audit_records = []

    if "DRUG" in all_dataframes:
        drug_all = all_dataframes["DRUG"]

        # Annotate every DRUG row with GLP-1 identification columns
        logger.info(f"  Normalizing {len(drug_all):,} DRUG rows across all quarters...")
        drug_annotated = normalize_drug_file(drug_all, "ALL_QUARTERS")

        # Save full annotated DRUG (all drugs, all annotations) as interim
        save_parquet(drug_annotated, "DRUG_annotated", processed_dir, end_q)

        # Build normalization audit
        norm_audit = build_normalization_audit(drug_annotated)
        norm_audit_records.append(norm_audit)
        logger.info(
            f"  GLP-1 identified: {norm_audit['glp1_identified']:,} rows "
            f"({norm_audit['glp1_rate']:.1%} of all DRUG rows)"
        )
        logger.info(
            f"  Primary suspect GLP-1: {norm_audit['primary_suspect_glp1']:,} rows"
        )
        if norm_audit.get("compounded", 0):
            logger.info(f"  Compounded products: {norm_audit['compounded']:,} rows")

        # Filter to GLP-1 primary suspect — the analysis-ready dataset
        drug_glp1_ps = filter_to_glp1_ps(drug_annotated)
        save_parquet(drug_glp1_ps, "DRUG_glp1_ps", processed_dir, end_q)
        logger.info(f"  Saved DRUG_glp1_ps: {len(drug_glp1_ps):,} rows")

        # Scope DEMO to only cases with at least one GLP-1 PS drug
        # DECISION: a "GLP-1 case" is defined as any DEMO record whose
        # PRIMARYID appears in the GLP-1 PS DRUG set. This is the correct
        # denominator for patient-level analyses.
        glp1_ps_primaryids = set(
            pd.to_numeric(drug_glp1_ps["primaryid"], errors="coerce")
            .dropna().astype(int)
        )
        if "DEMO" in all_dataframes:
            demo_all = all_dataframes["DEMO"]
            demo_glp1 = demo_all[
                pd.to_numeric(demo_all["primaryid"], errors="coerce")
                .isin(glp1_ps_primaryids)
            ].copy()
            save_parquet(demo_glp1, "DEMO_glp1", processed_dir, end_q)
            logger.info(f"  Saved DEMO_glp1: {len(demo_glp1):,} unique GLP-1 cases")
            all_dataframes["DEMO_glp1"] = demo_glp1

        all_dataframes["DRUG_annotated"] = drug_annotated
        all_dataframes["DRUG_glp1_ps"]   = drug_glp1_ps

    # Save normalization audit
    if norm_audit_records:
        norm_audit_path = logs_dir / f"normalization_audit_{_version_tag(end_q)}.json"
        with open(norm_audit_path, "w") as f:
            json.dump(norm_audit_records, f, indent=2)
        logger.info(f"  Normalization audit → {norm_audit_path.name}")

    # ── Step 7: Phase 3 — MedDRA SOC join on scoped REAC ─────────────────────
    logger.info("\n── Step 7: Phase 3 — MedDRA SOC Join ─────────────────────")

    if "REAC" in all_dataframes and "DRUG_glp1_ps" in all_dataframes:
        reac_all = all_dataframes["REAC"]

        # Scope REAC to GLP-1 cases only (join on primaryid)
        reac_glp1 = reac_all[
            pd.to_numeric(reac_all["primaryid"], errors="coerce")
            .isin(glp1_ps_primaryids)
        ].copy()
        logger.info(f"  Scoped REAC to GLP-1 cases: {len(reac_glp1):,} reaction rows")

        # Apply MedDRA SOC hierarchy
        reac_with_soc, meddra_audit = join_meddra(reac_glp1, mdhier_path)
        save_parquet(reac_with_soc, "REAC_glp1_soc", processed_dir, end_q)
        logger.info(
            f"  MedDRA mapping rate: {meddra_audit['mapping_rate']:.1%} "
            f"({meddra_audit['mapped_via_mdhier']:,} via mdhier, "
            f"{meddra_audit['mapped_via_bundled']:,} via bundled)"
        )

        # Save MedDRA audit
        meddra_audit_path = logs_dir / f"meddra_audit_{_version_tag(end_q)}.json"
        with open(meddra_audit_path, "w") as f:
            json.dump(meddra_audit, f, indent=2)
        logger.info(f"  MedDRA audit → {meddra_audit_path.name}")

        # SOC summary table
        soc_df = soc_summary(reac_with_soc)
        if not soc_df.empty:
            soc_path = processed_dir / f"SOC_summary_{_version_tag(end_q)}.csv"
            soc_df.to_csv(soc_path, index=False)
            logger.info(f"  SOC summary → {soc_path.name}")

        all_dataframes["REAC_glp1_soc"] = reac_with_soc

    # ── Step 8: Save data dictionary ─────────────────────────────────────────
    logger.info("\n── Step 8: Save Data Dictionary ───────────────────────────")
    save_data_dictionary(all_dataframes, docs_dir)

    # ── Step 9: Phase 4 — Signal detection ───────────────────────────────────
    logger.info("\n── Step 9: Phase 4 — Signal Detection ─────────────────────")

    if "DRUG_glp1_ps" in all_dataframes and "REAC_glp1_soc" in all_dataframes:
        signal_results = run_signal_detection(
            drug_glp1_ps  = all_dataframes["DRUG_glp1_ps"],
            reac_with_soc = all_dataframes["REAC_glp1_soc"],
            demo_all      = all_dataframes.get("DEMO", pd.DataFrame()),
            ther_df       = all_dataframes.get("THER"),
            demo_glp1     = all_dataframes.get("DEMO_glp1"),
            full_reac     = all_dataframes.get("REAC"),
        )

        # Save signal tables
        for key in ["signals_pt", "signals_soc", "tto", "tto_summary"]:
            if key in signal_results and not signal_results[key].empty:
                save_parquet(signal_results[key], key, processed_dir, end_q)

        # Save signal audit
        signal_audit_path = logs_dir / f"signal_audit_{_version_tag(end_q)}.json"
        with open(signal_audit_path, "w") as f:
            audit_data = signal_results.get("audit", {})
            # Convert any non-serialisable values
            json.dump(
                {k: int(v) if hasattr(v, 'item') else v
                 for k, v in audit_data.items()},
                f, indent=2,
            )
        logger.info(f"  Signal audit → {signal_audit_path.name}")

        if "signals_pt" in signal_results:
            spt = signal_results["signals_pt"]
            n_sig = int(spt["is_signal"].sum())
            logger.info(
                f"  PT-level signals: {n_sig:,} / {len(spt):,} pairs "
                f"({n_sig/max(len(spt),1):.1%})"
            )
        all_dataframes.update({
            k: v for k, v in signal_results.items()
            if isinstance(v, pd.DataFrame) and not v.empty
        })

    # ── Step 10: Phase 5 — EDA report ────────────────────────────────────────
    if not no_report:
        logger.info("\n── Step 10: Phase 5 — EDA Report ──────────────────────")
        try:
            report_path = docs_dir / f"eda_report_{_version_tag(end_q)}.html"
            build_report(
                processed_dir = processed_dir,
                logs_dir      = logs_dir,
                out_path      = report_path,
                scope_start   = str(quarters[0]).replace("Q", " Q"),
                scope_end     = str(end_q).replace("Q", " Q"),
            )
        except Exception as e:
            logger.warning(f"  EDA report failed (non-fatal): {e}")
    else:
        logger.info("\n── Step 10: Phase 5 — SKIPPED (--no-report) ───────────")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete — Phases 1 → 5")
    logger.info(f"  Quarters processed    : {len(audit_records)}")
    if audit_records:
        total_raw   = sum(a["n_raw"]   for a in audit_records)
        total_final = sum(a["n_final"] for a in audit_records)
        logger.info(f"  Total raw DEMO cases  : {total_raw:,}")
        logger.info(f"  After dedup           : {total_final:,}")
    if "DEMO_glp1" in all_dataframes:
        logger.info(f"  GLP-1 cases (Ph 2)    : {len(all_dataframes['DEMO_glp1']):,}")
    if "REAC_glp1_soc" in all_dataframes:
        logger.info(f"  GLP-1 reactions (Ph 3): {len(all_dataframes['REAC_glp1_soc']):,}")
    if "signals_pt" in all_dataframes:
        spt = all_dataframes["signals_pt"]
        logger.info(
            f"  PT signals (Ph 4)     : {int(spt['is_signal'].sum()):,} "
            f"/ {len(spt):,} pairs"
        )
    logger.info(f"  Outputs               : {processed_dir}")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "FAERS-GLP1-Watch — full pharmacovigilance pipeline. "
            "Phases 1–5: Download, parse, deduplicate, normalize, "
            "MedDRA join, signal detection, EDA report."
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

  # Use MedDRA subscription file for full PT→SOC coverage
  python -m faers_pipeline.pipeline --mdhier /path/to/mdhier.asc

  # Skip download (ZIPs already in data/raw/) and skip EDA report
  python -m faers_pipeline.pipeline --skip-download --no-report

  # Re-download all ZIPs even if they exist
  python -m faers_pipeline.pipeline --force-download

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
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip EDA report generation (Phase 5). Useful for headless CI runs.",
    )
    parser.add_argument(
        "--mdhier",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to MedDRA mdhier.asc file (from MedDRA subscription). "
            "If omitted, the bundled GLP-1 PT→SOC mapping is used for Phase 3."
        ),
    )

    args = parser.parse_args()

    run_phase1(
        project_root   = args.root.resolve(),
        quarters       = args.quarters,
        dry_run        = args.dry_run,
        force_download = args.force_download,
        skip_download  = args.skip_download,
        verbose        = args.verbose,
        mdhier_path    = args.mdhier,
        no_report      = args.no_report,
    )


if __name__ == "__main__":
    main()
