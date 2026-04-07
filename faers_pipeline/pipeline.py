"""
pipeline.py — FAERS-GLP1-Watch full pipeline, Phases 1–5.
Streams everything. Never loads a full file type into RAM.

Flags:
  --skip-download   skip Step 1 (ZIPs already in data/raw/)
  --skip-parse      skip Steps 2-3 (interim/ already populated)
  --skip-merge      skip Step 4 (processed/ Parquet already written)
  --no-report       skip Step 10 EDA report
  --dry-run         show download URLs only
  --verbose         debug logging
  --quarters        e.g. 2024Q1 2024Q2
  --mdhier          path to MedDRA mdhier.asc
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .quarters import Quarter, ALL_QUARTERS, FILE_TYPES
from .downloader import download_all
from .parser import parse_quarter
from .deduplicator import deduplicate_demo, filter_related_by_primaryid
from .normalizer import normalize_drug_file, filter_to_glp1_ps, build_normalization_audit
from .meddra import join_meddra, soc_summary
from .signal_detection import run_signal_detection
from .eda_report import build_report
from .writer import save_audit_log, save_data_dictionary, _version_tag

CHUNK_ROWS = 2_000_000


def _setup_logging(logs_dir: Path, verbose: bool = False) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "pipeline.log", mode="a"),
        ],
    )


def _parse_quarter_arg(s: str) -> Quarter:
    s = s.upper().replace(" ", "")
    if "Q" not in s:
        raise argparse.ArgumentTypeError(f"Invalid quarter: {s!r}. Use e.g. 2024Q3")
    year_str, q_str = s.split("Q")
    return Quarter(int(year_str), int(q_str))


def _unified_schema(part_paths: list) -> pa.Schema:
    """Read schemas only (no data) and promote null columns to large_string."""
    unified: dict = {}
    for p in part_paths:
        for field in pq.read_schema(p):
            existing = unified.get(field.name)
            if existing is None:
                unified[field.name] = field.type
            elif existing == pa.null() and field.type != pa.null():
                unified[field.name] = field.type
    return pa.schema([
        pa.field(name, pa.large_utf8() if typ == pa.null() else typ)
        for name, typ in unified.items()
    ])


def _cast_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    cols = []
    for name in schema.names:
        target = schema.field(name).type
        if name in table.schema.names:
            col = table.column(name)
            if col.type != target:
                col = col.cast(target, safe=False)
        else:
            col = pa.array([None] * len(table), type=target)
        cols.append(col)
    return pa.table(dict(zip(schema.names, cols)), schema=schema)


def _merge_quarters(part_paths: list, out_path: Path, label: str) -> int:
    """Stream-merge per-quarter Parquet files. One quarter in RAM at a time."""
    schema = _unified_schema(part_paths)
    writer = None
    total  = 0
    for i, p in enumerate(part_paths):
        table  = pq.read_table(p)
        table  = _cast_to_schema(table, schema)
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema, compression="snappy")
        writer.write_table(table)
        total += table.num_rows
        del table
        p.unlink(missing_ok=True)          # free interim disk immediately
        if (i + 1) % 10 == 0:
            logging.getLogger(__name__).info(
                f"    {label}: {i+1}/{len(part_paths)} quarters merged...")
        gc.collect()
    if writer:
        writer.close()
    return total


def run_phase1(
    project_root  : Path,
    quarters      : list = None,
    dry_run       : bool = False,
    force_download: bool = False,
    skip_download : bool = False,
    skip_parse    : bool = False,
    skip_merge    : bool = False,
    skip_to_signals: bool = False,
    verbose       : bool = False,
    mdhier_path   : Path = None,
    no_report     : bool = False,
) -> None:

    raw_dir       = project_root / "data" / "raw"
    interim_dir   = project_root / "data" / "interim"
    processed_dir = project_root / "data" / "processed"
    logs_dir      = project_root / "data" / "logs"
    docs_dir      = project_root / "docs"

    for d in [processed_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _setup_logging(logs_dir, verbose)
    log = logging.getLogger(__name__)

    quarters = quarters or ALL_QUARTERS
    end_q    = quarters[-1]
    vtag     = _version_tag(end_q)

    log.info("=" * 60)
    log.info("FAERS-GLP1-Watch  |  Phases 1–5")
    log.info(f"Scope : {quarters[0]} → {end_q}  ({len(quarters)} quarters)")
    log.info(f"Root  : {project_root}")
    log.info("=" * 60)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if not skip_download:
        log.info("\n── Step 1: Download ───────────────────────────────────────")
        results = download_all(raw_dir=raw_dir, quarters=quarters,
                               dry_run=dry_run, force=force_download)
        if dry_run:
            return
        failed = [r for r in results if r["status"] == "failed"]
        if failed:
            log.warning("  Failed: " + ", ".join(r["quarter"] for r in failed))
    else:
        log.info("\n── Step 1: SKIPPED (--skip-download) ─────────────────────")

    # ── Steps 2–3: Parse + Deduplicate → per-quarter interim Parquet ─────────
    audit_records: list = []

    if not skip_parse:
        log.info("\n── Step 2–3: Parse & Deduplicate ──────────────────────────")
        interim_dir.mkdir(parents=True, exist_ok=True)

        for quarter in quarters:
            zip_path = raw_dir / quarter.zip_filename()
            if not zip_path.exists():
                log.warning(f"  MISSING: {zip_path.name}")
                continue

            log.info(f"\n  Processing {quarter.label()} ...")
            parsed = parse_quarter(zip_path, quarter)

            demo = parsed.get("DEMO")
            if demo is None or demo.empty:
                log.warning(f"  DEMO missing — skip")
                continue

            demo_deduped, audit = deduplicate_demo(demo, str(quarter))
            audit_records.append(audit)

            surviving_ids = set(
                pd.to_numeric(demo_deduped["primaryid"], errors="coerce")
                .dropna().astype(int)
            )

            qdir = interim_dir / str(quarter)
            qdir.mkdir(exist_ok=True)
            demo_deduped.to_parquet(qdir / "DEMO.parquet", index=False)

            for ft in FILE_TYPES:
                if ft == "DEMO":
                    continue
                df = parsed.get(ft)
                if df is not None and not df.empty:
                    filtered = filter_related_by_primaryid(
                        df, surviving_ids, ft, str(quarter))
                    filtered.to_parquet(qdir / f"{ft}.parquet", index=False)
                    del filtered

            del parsed, demo, demo_deduped
            gc.collect()

        if audit_records:
            save_audit_log(audit_records, logs_dir, end_q)
    else:
        log.info("\n── Step 2–3: SKIPPED (--skip-parse) ──────────────────────")

    # Discover available quarters in interim/
    quarters_available = sorted([
        d.name for d in interim_dir.iterdir()
        if d.is_dir() and (d / "DEMO.parquet").exists()
    ]) if interim_dir.exists() else []

    if not quarters_available and not skip_merge:
        log.error("No interim files found. Run without --skip-parse first.")
        return

    log.info(f"\n  {len(quarters_available)} quarters in interim/")

    # ── Step 4: Merge per-quarter → one Parquet per file type ────────────────
    if not skip_merge:
        log.info("\n── Step 4: Merge ──────────────────────────────────────────")

        for ft in FILE_TYPES:
            out_path = processed_dir / f"{ft}_deduplicated_{vtag}.parquet"
            if out_path.exists():
                mb = out_path.stat().st_size / 1024 ** 2
                log.info(f"  {ft}: already exists ({mb:.0f} MB) — skip")
                continue

            part_paths = [
                interim_dir / q / f"{ft}.parquet"
                for q in quarters_available
                if (interim_dir / q / f"{ft}.parquet").exists()
            ]
            if not part_paths:
                log.warning(f"  {ft}: no quarter files found")
                continue

            total   = _merge_quarters(part_paths, out_path, ft)
            size_mb = out_path.stat().st_size / 1024 ** 2
            log.info(f"  {ft}: {total:,} rows → {out_path.name} ({size_mb:.0f} MB)")
            gc.collect()
    else:
        log.info("\n── Step 4: SKIPPED (--skip-merge) ────────────────────────")

    # ── Step 5: Audit log ─────────────────────────────────────────────────────
    log.info("\n── Step 5: Audit Log ──────────────────────────────────────")
    if not audit_records:
        existing = sorted(logs_dir.glob("dedup_audit_*.csv"))
        log.info(f"  Using existing: {existing[-1].name}" if existing
                 else "  No audit records available")

    # ── Step 6: Phase 2 — Drug normalization
    if skip_to_signals:
        log.info("\n── Steps 6–8: SKIPPED (--skip-to-signals) ─────────────")
        # Load the GLP-1 case IDs from the existing DRUG_glp1_ps file
        if drug_ps_path.exists():
            glp1_ps_primaryids = set(
                pd.to_numeric(
                    pd.read_parquet(drug_ps_path, columns=["primaryid"])["primaryid"],
                    errors="coerce"
                ).dropna().astype(int)
            )
            log.info(f"  Loaded {len(glp1_ps_primaryids):,} GLP-1 case IDs from existing file")
    else:
        log.info("\n── Step 6: Phase 2 — Drug Normalization ───────────────────")

        drug_raw_path  = processed_dir / f"DRUG_deduplicated_{vtag}.parquet"
        drug_ann_path  = processed_dir / f"DRUG_annotated_{vtag}.parquet"
        drug_ps_path   = processed_dir / f"DRUG_glp1_ps_{vtag}.parquet"
        glp1_ps_primaryids: set = set()

        if drug_raw_path.exists():
            drug_file  = pq.ParquetFile(drug_raw_path)
            n_drug     = drug_file.metadata.num_rows
            log.info(f"  {n_drug:,} DRUG rows → chunked normalization ({CHUNK_ROWS:,}/chunk)")

            ann_writer = None
            ps_writer  = None
            agg        = {k: 0 for k in ["total_drug_rows", "glp1_identified",
                                        "primary_suspect_glp1", "compounded",
                                        "combo_products", "withdrawn_drug"]}

            for batch in drug_file.iter_batches(batch_size=CHUNK_ROWS):
                chunk     = batch.to_pandas()
                chunk_ann = normalize_drug_file(chunk, "CHUNK")
                chunk_ps  = filter_to_glp1_ps(chunk_ann)

                t_ann = pa.Table.from_pandas(chunk_ann, preserve_index=False)
                if ann_writer is None:
                    ann_writer = pq.ParquetWriter(drug_ann_path, t_ann.schema,
                                                compression="snappy")
                ann_writer.write_table(t_ann)

                if len(chunk_ps) > 0:
                    t_ps = pa.Table.from_pandas(chunk_ps, preserve_index=False)
                    if ps_writer is None:
                        ps_writer = pq.ParquetWriter(drug_ps_path, t_ps.schema,
                                                    compression="snappy")
                    ps_writer.write_table(t_ps)
                    glp1_ps_primaryids.update(
                        pd.to_numeric(chunk_ps["primaryid"], errors="coerce")
                        .dropna().astype(int)
                    )

                for k in agg:
                    agg[k] += build_normalization_audit(chunk_ann).get(k, 0)

                del chunk, chunk_ann, chunk_ps
                gc.collect()

            if ann_writer: ann_writer.close()
            if ps_writer:  ps_writer.close()

            rate = agg["glp1_identified"] / max(agg["total_drug_rows"], 1)
            log.info(f"  GLP-1 identified : {agg['glp1_identified']:,} ({rate:.1%})")
            log.info(f"  Primary suspect  : {agg['primary_suspect_glp1']:,}")
            log.info(f"  Unique case IDs  : {len(glp1_ps_primaryids):,}")

            with open(logs_dir / f"normalization_audit_{vtag}.json", "w") as f:
                json.dump([agg], f, indent=2)
        else:
            log.warning(f"  DRUG file not found — skipping normalization")

        # Scope DEMO to GLP-1 cases (stream, never full DEMO in RAM)
        demo_raw_path  = processed_dir / f"DEMO_deduplicated_{vtag}.parquet"
        demo_glp1_path = processed_dir / f"DEMO_glp1_{vtag}.parquet"

        if demo_raw_path.exists() and glp1_ps_primaryids:
            log.info("  Scoping DEMO to GLP-1 cases (streaming)...")
            demo_file   = pq.ParquetFile(demo_raw_path)
            demo_writer = None
            n_glp1      = 0

            for batch in demo_file.iter_batches(batch_size=CHUNK_ROWS):
                chunk = batch.to_pandas()
                filt  = chunk[
                    pd.to_numeric(chunk["primaryid"], errors="coerce")
                    .isin(glp1_ps_primaryids)
                ]
                if len(filt) > 0:
                    t = pa.Table.from_pandas(filt, preserve_index=False)
                    if demo_writer is None:
                        demo_writer = pq.ParquetWriter(demo_glp1_path, t.schema,
                                                    compression="snappy")
                    demo_writer.write_table(t)
                    n_glp1 += len(filt)
                del chunk, filt
                gc.collect()

            if demo_writer: demo_writer.close()
            log.info(f"  DEMO_glp1: {n_glp1:,} GLP-1 cases")

        # ── Step 7: Phase 3 — MedDRA SOC join ────────────────────────────────────
        log.info("\n── Step 7: Phase 3 — MedDRA SOC Join ─────────────────────")

        reac_raw_path  = processed_dir / f"REAC_deduplicated_{vtag}.parquet"
        reac_glp1_path = processed_dir / f"REAC_glp1_soc_{vtag}.parquet"
        reac_glp1_df   = None

        if reac_raw_path.exists() and glp1_ps_primaryids:
            log.info("  Scoping REAC to GLP-1 cases (streaming)...")
            reac_file  = pq.ParquetFile(reac_raw_path)
            glp1_chunks = []

            for batch in reac_file.iter_batches(batch_size=CHUNK_ROWS):
                chunk = batch.to_pandas()
                filt  = chunk[
                    pd.to_numeric(chunk["primaryid"], errors="coerce")
                    .isin(glp1_ps_primaryids)
                ].copy()
                if len(filt) > 0:
                    glp1_chunks.append(filt)
                del chunk
                gc.collect()

            if glp1_chunks:
                reac_glp1_raw = pd.concat(glp1_chunks, ignore_index=True)
                del glp1_chunks
                gc.collect()

                log.info(f"  GLP-1 REAC rows: {len(reac_glp1_raw):,}")
                reac_glp1_df, meddra_audit = join_meddra(reac_glp1_raw, mdhier_path)
                reac_glp1_df.to_parquet(reac_glp1_path, index=False)
                log.info(f"  MedDRA mapping: {meddra_audit['mapping_rate']:.1%}")

                soc_df = soc_summary(reac_glp1_df)
                if not soc_df.empty:
                    soc_path = processed_dir / f"SOC_summary_{vtag}.csv"
                    soc_df.to_csv(soc_path, index=False)
                    log.info(f"  SOC summary → {soc_path.name}")

                with open(logs_dir / f"meddra_audit_{vtag}.json", "w") as f:
                    json.dump(meddra_audit, f, indent=2)
                del reac_glp1_raw
                gc.collect()

        elif reac_glp1_path.exists():
            log.info("  Loading existing REAC_glp1_soc...")
            reac_glp1_df = pd.read_parquet(reac_glp1_path)

        # ── Step 8: Data dictionary ───────────────────────────────────────────────
        log.info("\n── Step 8: Data Dictionary ────────────────────────────────")
        try:
            small = {}
            for name, path in [("DEMO_glp1", demo_glp1_path),
                                ("REAC_glp1_soc", reac_glp1_path)]:
                if path.exists():
                    small[name] = pd.read_parquet(path)
            if small:
                save_data_dictionary(small, docs_dir)
                log.info(f"  Data dictionary → {docs_dir / 'data_dictionary.csv'}")
        except Exception as e:
            log.warning(f"  Data dictionary failed (non-fatal): {e}")

    # ── Step 9: Phase 4 — Signal detection ───────────────────────────────────
    log.info("\n── Step 9: Phase 4 — Signal Detection ─────────────────────")

    if drug_ps_path.exists() and reac_glp1_df is not None:
        drug_glp1_ps = pd.read_parquet(drug_ps_path)
        log.info(f"  DRUG_glp1_ps: {len(drug_glp1_ps):,} rows")

        # N denominator: only primaryid column from full DEMO
        demo_ids = pd.read_parquet(demo_raw_path, columns=["primaryid"]) \
                   if demo_raw_path.exists() else pd.DataFrame()

        # c-cell marginal: stream full REAC, keep only primaryid+pt
        # Never load full REAC into RAM — read in chunks and filter
        if reac_raw_path.exists():
            log.info("  Building REAC marginal from full file (streaming)...")
            reac_file = pq.ParquetFile(reac_raw_path)
            reac_chunks = []
            for batch in reac_file.iter_batches(
                    batch_size=CHUNK_ROWS,
                    columns=["primaryid", "pt"]):
                reac_chunks.append(batch.to_pandas())
                gc.collect()
            full_reac_small = pd.concat(reac_chunks, ignore_index=True)
            del reac_chunks
            gc.collect()
            log.info(f"  REAC marginal: {len(full_reac_small):,} rows")
        else:
            full_reac_small = None

        signal_results = run_signal_detection(
            drug_glp1_ps  = drug_glp1_ps,
            reac_with_soc = reac_glp1_df,
            demo_all      = demo_ids,
            full_reac     = full_reac_small,
        )
        del full_reac_small, demo_ids
        gc.collect()

        for key in ["signals_pt", "signals_soc", "tto", "tto_summary"]:
            df = signal_results.get(key)
            if df is not None and not df.empty:
                df.to_parquet(processed_dir / f"{key}_{vtag}.parquet", index=False)

        with open(logs_dir / f"signal_audit_{vtag}.json", "w") as f:
            json.dump(
                {k: int(v) if hasattr(v, "item") else v
                 for k, v in signal_results.get("audit", {}).items()},
                f, indent=2,
            )

        if "signals_pt" in signal_results:
            spt   = signal_results["signals_pt"]
            n_sig = int(spt["is_signal"].sum())
            log.info(f"  PT signals: {n_sig:,} / {len(spt):,} pairs")
    else:
        log.warning("  Skipping — missing DRUG_glp1_ps or REAC_glp1_soc")

    # ── Step 10: Phase 5 — EDA report ────────────────────────────────────────
    if not no_report:
        log.info("\n── Step 10: Phase 5 — EDA Report ──────────────────────")
        try:
            report_path = docs_dir / f"eda_report_{vtag}.html"
            build_report(
                processed_dir = processed_dir,
                logs_dir      = logs_dir,
                out_path      = report_path,
                scope_start   = str(quarters[0]).replace("Q", " Q"),
                scope_end     = str(end_q).replace("Q", " Q"),
            )
        except Exception as e:
            log.warning(f"  EDA report failed (non-fatal): {e}")
    else:
        log.info("\n── Step 10: SKIPPED (--no-report) ────────────────────")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Pipeline complete")
    if audit_records:
        log.info(f"  Quarters   : {len(audit_records)}")
        log.info(f"  Raw cases  : {sum(a['n_raw'] for a in audit_records):,}")
        log.info(f"  After dedup: {sum(a['n_final'] for a in audit_records):,}")
    log.info(f"  Outputs    : {processed_dir}")
    log.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="FAERS-GLP1-Watch pipeline (Phases 1–5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m faers_pipeline.pipeline --dry-run
  python -m faers_pipeline.pipeline
  python -m faers_pipeline.pipeline --skip-download
  python -m faers_pipeline.pipeline --skip-download --skip-parse
  python -m faers_pipeline.pipeline --skip-download --skip-parse --skip-merge
""")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-download",  action="store_true")
    p.add_argument("--skip-parse",     action="store_true")
    p.add_argument("--skip-merge",     action="store_true")
    p.add_argument("--no-report",      action="store_true")
    p.add_argument("--verbose",        action="store_true")
    p.add_argument("--quarters", nargs="+", type=_parse_quarter_arg, metavar="YYYYQN")
    p.add_argument("--mdhier", type=Path, default=None)
    p.add_argument("--root",   type=Path, default=Path("."))
    p.add_argument("--skip-signals",   action="store_true",
                   help="Skip straight to Step 9 signal detection")
    args = p.parse_args()

    run_phase1(
        project_root   = args.root.resolve(),
        quarters       = args.quarters,
        dry_run        = args.dry_run,
        force_download = args.force_download,
        skip_download  = args.skip_download,
        skip_parse     = args.skip_parse,
        skip_merge     = args.skip_merge,
        verbose        = args.verbose,
        mdhier_path    = args.mdhier,
        no_report      = args.no_report,
    )


if __name__ == "__main__":
    main()
