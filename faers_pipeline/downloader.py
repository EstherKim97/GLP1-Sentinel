"""
downloader.py
-------------
Download FAERS quarterly ZIP files from FDA FIS with:
  - Resume support (skip already-downloaded files)
  - Retry with exponential backoff
  - Progress bar per file
  - Download manifest (JSON log of every file fetched)
  - Dry-run mode (print what would be downloaded without fetching)

DECISION LOG — Download strategy
----------------------------------
We download the full ASCII quarterly files (not the XML format) because:
  1. ASCII is pipe-delimited, significantly smaller, and directly parseable.
  2. XML FAERS files are larger and require more complex parsing with no
     additional signal-relevant fields for our use case.
  3. ASCII files are the format used in all major pharmacovigilance studies
     we cite (Potter et al. 2025, Scientific Reports GLP-1 2025, etc.).

We do NOT use the openFDA API because:
  - It enforces rate limits (240 requests/minute) that make bulk download slow.
  - It returns JSON which is memory-inefficient vs. flat files for 17M+ records.
  - The raw quarterly files are the authoritative source; openFDA is a derived
    product with its own preprocessing decisions we want to control ourselves.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from .quarters import Quarter, ALL_QUARTERS, FILE_TYPES
from .schema import KNOWN_MISSING_QUARTERS

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

CHUNK_SIZE        = 1024 * 1024  # 1 MB chunks for streaming download
MAX_RETRIES       = 5
BACKOFF_BASE_SECS = 2.0          # Wait: 2, 4, 8, 16, 32 seconds
REQUEST_TIMEOUT   = 60           # Seconds before connection timeout
USER_AGENT        = (
    "FAERS-GLP1-Watch/0.1.0 "
    "(pharmacovigilance research; "
    "contact: see project README)"
)


# ── Download single ZIP ───────────────────────────────────────────────────────

def download_quarter(
    quarter: Quarter,
    raw_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Download one quarterly ZIP file.

    Returns a result dict with fields:
      quarter, filename, url, status, bytes_downloaded, path, error
    Status values: 'skipped' | 'downloaded' | 'dry_run' | 'failed'
    """
    filename = quarter.zip_filename()
    dest     = raw_dir / filename
    url      = quarter.download_url()

    result = {
        "quarter"         : str(quarter),
        "filename"        : filename,
        "url"             : url,
        "status"          : None,
        "bytes_downloaded": 0,
        "path"            : str(dest),
        "error"           : None,
    }

    # ── Dry run ──────────────────────────────────────────────────────────────
    if dry_run:
        if quarter in KNOWN_MISSING_QUARTERS:
            print(f"  [DRY RUN] SKIP {url} (known missing quarter)")
        else:
            print(f"  [DRY RUN] Would download: {url}")
        result["status"] = "dry_run"
        return result

    # ── Known missing quarters — skip without attempting download ─────────────
    # 2012 Q3 was never published by FDA. Attempting download returns 404
    # and wastes a retry cycle. We mark it as 'known_missing' and move on.
    if quarter in KNOWN_MISSING_QUARTERS:
        logger.info(f"  SKIP  {filename} (known missing quarter — FDA never published)")
        result["status"] = "known_missing"
        result["error"]  = "Quarter never published by FDA (AERS→FAERS transition gap)"
        return result

    # ── Skip if already exists ───────────────────────────────────────────────
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 ** 2)
        logger.info(f"  SKIP  {filename} (already exists, {size_mb:.1f} MB)")
        result["status"]          = "skipped"
        result["bytes_downloaded"] = dest.stat().st_size
        return result

    # ── Download with retry ───────────────────────────────────────────────────
    headers = {"User-Agent": USER_AGENT}
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"  GET   {url}  (attempt {attempt}/{MAX_RETRIES})")
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            total_bytes = int(response.headers.get("content-length", 0))
            downloaded  = 0

            with open(dest, "wb") as f, tqdm(
                desc        = f"{str(quarter):<8}",
                total       = total_bytes,
                unit        = "B",
                unit_scale  = True,
                unit_divisor= 1024,
                leave       = True,
            ) as bar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        bar.update(len(chunk))

            result["status"]          = "downloaded"
            result["bytes_downloaded"] = downloaded
            logger.info(f"  OK    {filename} ({downloaded / 1024**2:.1f} MB)")
            return result

        except requests.exceptions.HTTPError as e:
            # 404 = quarter doesn't exist yet; don't retry
            if e.response is not None and e.response.status_code == 404:
                logger.warning(f"  404   {url} — quarter may not exist")
                result["status"] = "failed"
                result["error"]  = f"HTTP 404 — file not found"
                return result
            last_error = str(e)

        except Exception as e:
            last_error = str(e)

        # Exponential backoff before retry
        wait = BACKOFF_BASE_SECS ** attempt
        logger.warning(f"  RETRY {filename} in {wait:.0f}s — {last_error}")
        time.sleep(wait)

    # All retries exhausted
    # Clean up partial file if it exists
    if dest.exists():
        dest.unlink()
    result["status"] = "failed"
    result["error"]  = last_error
    logger.error(f"  FAIL  {filename} — {last_error}")
    return result


# ── Download all quarters ─────────────────────────────────────────────────────

def download_all(
    raw_dir: Path,
    quarters: list[Quarter] = None,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    """
    Download all quarterly ZIPs for the project scope.

    Args:
        raw_dir:   Directory to store ZIP files (created if missing).
        quarters:  Specific quarters to download; defaults to ALL_QUARTERS.
        dry_run:   If True, print URLs but don't download.
        force:     Re-download even if file exists.

    Returns:
        List of result dicts, one per quarter.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    quarters = quarters or ALL_QUARTERS

    print(f"\n{'='*60}")
    print(f"FAERS-GLP1-Watch — Phase 1: Download")
    print(f"{'='*60}")
    print(f"  Scope       : {quarters[0]} → {quarters[-1]}")
    print(f"  Quarters    : {len(quarters)}")
    print(f"  Destination : {raw_dir}")
    print(f"  Dry run     : {dry_run}")
    print(f"  Force       : {force}")
    print(f"{'='*60}\n")

    results = []
    for i, quarter in enumerate(quarters, 1):
        print(f"[{i:>3}/{len(quarters)}] {quarter.label()}")
        result = download_quarter(quarter, raw_dir, dry_run=dry_run, force=force)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ["downloaded", "skipped", "failed", "dry_run", "known_missing"]}
    total_mb = sum(r["bytes_downloaded"] for r in results) / (1024 ** 2)

    print(f"\n{'='*60}")
    print(f"Download summary")
    print(f"  Downloaded    : {counts['downloaded']}")
    print(f"  Skipped       : {counts['skipped']} (already present)")
    print(f"  Known missing : {counts['known_missing']} (FDA never published)")
    print(f"  Failed        : {counts['failed']}")
    print(f"  Total size    : {total_mb:,.0f} MB on disk")
    print(f"{'='*60}\n")

    # ── Save manifest ─────────────────────────────────────────────────────────
    if not dry_run:
        manifest_path = raw_dir / "download_manifest.json"
        _save_manifest(manifest_path, results)
        print(f"  Manifest saved → {manifest_path}\n")

    return results


def _save_manifest(path: Path, results: list[dict]) -> None:
    """
    Persist download manifest as JSON.
    Merges with any existing manifest so re-runs append/update entries.
    """
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = {r["quarter"]: r for r in json.load(f)}

    # Update with new results
    for r in results:
        existing[r["quarter"]] = r

    with open(path, "w") as f:
        json.dump(list(existing.values()), f, indent=2)
