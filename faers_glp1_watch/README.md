# FAERS-GLP1-Watch

**An end-to-end pharmacovigilance pipeline for the GLP-1 drug class**

[![Tests](https://github.com/YOUR_USERNAME/faers-glp1-watch/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/faers-glp1-watch/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## What this is

GLP-1 receptor agonists (semaglutide, tirzepatide, liraglutide, and class) are the fastest-adopted drug class in US history. FDA's own quarterly signal reports flagged new neurological, pulmonary aspiration, and gastrointestinal risks through 2024–2025 — while simultaneous staff reductions reduced the agency's capacity to monitor them systematically.

This pipeline ingests the raw FDA Adverse Event Reporting System (FAERS/AEMS) quarterly files, resolves their well-documented quality problems, and produces a clean, analysis-ready dataset — with every data decision documented. The decision log is not an afterthought; in drug safety, it *is* the deliverable.

**Phase 1 (this repo):** Download → Parse → Deduplicate → Versioned Parquet output

**Planned phases:** Drug name normalization → MedDRA join → Disproportionality signal detection (ROR/PRR/IC) → EDA report → BigQuery NLP layer (Google Health add-on)

---

## Why the regulatory background matters

Anyone can download FAERS. The hard parts are:

- Knowing FDA's recommended 2-step deduplication rule — and why using `CASEVERSION` instead is wrong
- Understanding that `ROLE_COD = 'PS'` (primary suspect) is the correct filter for signal detection, not all drug records
- Knowing that 49.5% of semaglutide reports are missing age, and what that means for demographic stratification
- Recognising compounded semaglutide reports as a distinct population that needs separate analysis

These are regulatory decisions embedded in the pipeline, not engineering guesses.

---

## Project scope

| Parameter | Value |
|-----------|-------|
| Start quarter | 2005 Q2 — exenatide (Byetta) approval, first GLP-1 RA |
| End quarter | 2024 Q3 |
| Total quarters | 78 |
| Drugs in scope | semaglutide, tirzepatide, liraglutide, dulaglutide, exenatide, lixisenatide |
| Raw file format | FAERS ASCII quarterly ZIPs (7 files per quarter) |
| Output format | Parquet (snappy), versioned by data coverage end date |

---

## Data decision log

Every non-trivial choice is documented inline in the source. Key decisions:

### Deduplication (see `faers_pipeline/deduplicator.py`)

FAERS contains duplicate case reports because the same adverse event is often submitted by multiple parties (manufacturer, healthcare provider, patient) or re-submitted as updates. FDA's recommended rule:

1. Per `CASEID`, keep the record with the most recent `FDA_DT` (date FDA received the report)
2. On ties, keep the highest `PRIMARYID`

We do **not** use `CASEVERSION` — it is unreliable across reporting sources.

**Reference:** Potter et al. (2025), *Clinical Pharmacology & Therapeutics*, FAERS Essentials.

### Why primary suspect only?

FAERS assigns each drug in a report one of four roles: PS (primary suspect), SS (secondary suspect), C (concomitant), I (interacting). Signal detection uses PS only. Using all roles inflates counts and generates noise signals for drugs that happen to be taken alongside the suspect drug.

### MedDRA level choice

Signal analysis runs at **Preferred Term (PT)** level for granularity and at **System Organ Class (SOC)** level for summary charts. Collapsing directly to SOC loses clinically meaningful distinctions (e.g., "pancreatitis" vs. "abdominal pain" are both GI but have very different clinical weight).

### Compounded products

FDA explicitly flagged underreporting of compounded semaglutide/tirzepatide AEs through 2024–2025. The pipeline flags these separately via narrative text search, enabling a compounded vs. brand-name comparison — a clinically and regulatorily distinct population.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/faers-glp1-watch.git
cd faers-glp1-watch
pip install -r requirements.txt

# 2. See what would be downloaded (no actual download)
python -m faers_pipeline.pipeline --dry-run

# 3. Download all quarters and run the full pipeline
python -m faers_pipeline.pipeline

# 4. Process specific quarters only (useful for testing)
python -m faers_pipeline.pipeline --quarters 2024Q1 2024Q2 2024Q3

# 5. Already have ZIPs? Skip download and go straight to parse/dedup
python -m faers_pipeline.pipeline --skip-download

# 6. Run tests (no FDA data required)
python -m pytest tests/ -v
```

**Disk space:** ~2–5 GB for all 78 ZIPs. Output Parquet files are ~500 MB total.

**Time:** Download takes 30–60 minutes depending on connection. Parse + dedup runs in ~10 minutes.

---

## Output files

After running, `data/processed/` contains:

```
data/
├── raw/
│   ├── faers_ascii_2024q3.zip     # Original ZIPs (untouched)
│   ├── aers_ascii_2005q2.zip
│   └── download_manifest.json    # Every download: status, size, URL
├── processed/
│   ├── DEMO_deduplicated_v20240930.parquet
│   ├── DRUG_deduplicated_v20240930.parquet
│   ├── REAC_deduplicated_v20240930.parquet
│   ├── OUTC_deduplicated_v20240930.parquet
│   ├── THER_deduplicated_v20240930.parquet
│   ├── RPSR_deduplicated_v20240930.parquet
│   └── INDI_deduplicated_v20240930.parquet
└── logs/
    ├── dedup_audit_v20240930.json   # Machine-readable audit log
    ├── dedup_audit_v20240930.csv    # Human-readable audit log
    └── pipeline.log                 # Full run log

docs/
└── data_dictionary.csv             # Every column: dtype, null rate, samples
```

Version tag (`v20240930`) is tied to the last quarter's end date, not the run date.

---

## Reading the audit log

`data/logs/dedup_audit_v20240930.csv` has one row per quarter:

| column | meaning |
|--------|---------|
| `quarter` | e.g. `2024Q3` |
| `n_raw` | Records before dedup |
| `removed_step1` | Removed — older `FDA_DT` for same `CASEID` |
| `removed_step2` | Removed — lower `PRIMARYID` on tied `FDA_DT` |
| `n_final` | Unique cases retained |
| `dedup_rate` | Fraction removed (expect 5–20% per quarter) |

This becomes the **data quality chart** in the EDA report.

---

## Project structure

```
faers_glp1_watch/
├── faers_pipeline/
│   ├── __init__.py
│   ├── quarters.py        # Quarter registry, GLP-1 milestone annotations
│   ├── downloader.py      # Retry download with manifest logging
│   ├── parser.py          # ZIP extraction, encoding/delimiter handling
│   ├── deduplicator.py    # FDA-recommended 2-step dedup + audit logging
│   ├── writer.py          # Versioned Parquet + audit log output
│   └── pipeline.py        # Orchestrator + CLI entry point
├── tests/
│   ├── test_quarters.py              # Quarter class and registry
│   ├── test_deduplicator.py          # Dedup logic (34 unit tests)
│   └── test_pipeline_integration.py  # End-to-end with synthetic ZIP
├── data/                  # Git-ignored; created on first run
├── docs/                  # Data dictionary written here
├── requirements.txt
└── README.md
```

---

## GLP-1 approval milestones

| Quarter | Event |
|---------|-------|
| 2005 Q2 | Exenatide (Byetta) approved — first GLP-1 RA |
| 2010 Q1 | Liraglutide (Victoza) approved |
| 2014 Q3 | Dulaglutide (Trulicity) approved |
| 2016 Q3 | Lixisenatide (Adlyxin) approved |
| 2017 Q4 | Semaglutide SC (Ozempic) approved |
| 2019 Q3 | Semaglutide oral (Rybelsus) approved |
| 2021 Q2 | Semaglutide (Wegovy) — obesity indication |
| 2022 Q2 | Tirzepatide (Mounjaro) — first dual GIP/GLP-1 agonist |
| 2023 Q4 | Tirzepatide (Zepbound) — obesity indication |

These are embedded in `quarters.py` as `GLP1_MILESTONES` and used as chart annotations in the EDA report.

---

## References

- Potter E, Reyes M, Naples J, Dal Pan G. (2025). FDA Adverse Event Reporting System (FAERS) Essentials. *Clinical Pharmacology & Therapeutics*, 118(3):567–582.
- Scientific Reports. (2025). Pharmacovigilance analysis of neurological adverse events associated with GLP-1 receptor agonists based on the FDA Adverse Event Reporting System.
- FDA. (2024). Best Practices for FDA Staff in the Postmarketing Safety Surveillance of Human Drug and Biological Products.
- FDA. (2025). FDA's Concerns with Unapproved GLP-1 Drugs Used for Weight Loss.

---

## License

MIT. Data is sourced from FDA public databases and is in the public domain.
