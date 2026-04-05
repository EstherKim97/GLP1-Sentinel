# FAERS-GLP1-Watch

**An end-to-end pharmacovigilance pipeline for the GLP-1 drug class**

[![Tests](https://github.com/YOUR_USERNAME/faers-glp1-watch/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/faers-glp1-watch/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## What this is

GLP-1 receptor agonists — semaglutide, tirzepatide, liraglutide, and class — are the fastest-adopted drug class in US history. In 2024–2025, FDA's own quarterly FAERS signal reports flagged new neurological, pulmonary aspiration, and gastrointestinal serious risk signals for this class. Simultaneously, staff reductions reduced the agency's capacity to monitor them systematically.

This pipeline ingests the raw FDA Adverse Event Reporting System (FAERS/AEMS) quarterly files, resolves their well-documented quality problems, applies GLP-1 drug name normalization, joins the MedDRA reaction hierarchy, detects disproportionality signals, and produces a self-contained HTML EDA report — with every data decision documented inline.

**The decision log is not an afterthought. In drug safety, it is the deliverable.**

---

## Pipeline overview

```
Phase 1  Download → Parse → Schema-normalize → Deduplicate → Parquet
Phase 2  Drug name normalization → GLP-1 scoping → PS filter
Phase 3  MedDRA SOC join → Missingness audit
Phase 4  Signal detection: ROR / PRR / IC (≥2/3 methods)
Phase 5  Self-contained HTML EDA report (Plotly interactive charts)
Phase 6  BigQuery NLP layer — Google Health add-on (planned)
```

Run the entire pipeline with one command:

```bash
python -m faers_pipeline.pipeline
```

---

## Why the regulatory background matters

Anyone can download FAERS. The hard parts are:

- Knowing FDA's recommended 2-step deduplication rule — and why using `CASEVERSION` is wrong
- Knowing that AERS (pre-2012) uses `isr` / `case` / `gndr_cod` — different column names than FAERS — and that 2012 Q3 was never published
- Understanding that `role_cod = 'PS'` (primary suspect) is the correct filter for signal detection, not all drug records
- Recognising that compounded semaglutide/tirzepatide is a clinically distinct population that must be flagged separately
- Knowing which MedDRA level is appropriate for which analysis (PT vs. SOC)

These are regulatory decisions embedded in the code, not engineering guesses. Each one has a citation.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/faers-glp1-watch.git
cd faers-glp1-watch
pip install -r requirements.txt

# 2. Verify tests pass (no FDA data required)
python -m pytest tests/ -v

# 3. See what would be downloaded — sanity check before committing disk space
python -m faers_pipeline.pipeline --dry-run

# 4. Run full pipeline (~3-5 GB download, ~30-60 min depending on connection)
python -m faers_pipeline.pipeline

# 5. After pipeline completes, open the EDA report
open docs/eda_report_v20240930.html   # macOS
# or: xdg-open docs/eda_report_v20240930.html
```

### Useful flags

```bash
# Skip download if ZIPs already in data/raw/ (re-run analysis only)
python -m faers_pipeline.pipeline --skip-download

# Run specific quarters only (useful for development)
python -m faers_pipeline.pipeline --quarters 2023Q1 2023Q2 2024Q3

# Use MedDRA subscription file for full PT→SOC coverage
python -m faers_pipeline.pipeline --mdhier /path/to/mdhier.asc

# Skip EDA report (faster, useful for CI)
python -m faers_pipeline.pipeline --skip-download --no-report

# Generate EDA report standalone (after pipeline has run)
python -m faers_pipeline.eda_report

# Generate data quality audit chart standalone
python -m faers_pipeline.audit_chart
```

---

## Output files

After running, `data/processed/` contains:

```
data/
├── raw/
│   ├── faers_ascii_2024q3.zip          Original ZIPs (untouched)
│   ├── aers_ascii_2005q2.zip
│   └── download_manifest.json          Every download: status, size, URL
├── processed/
│   ├── DEMO_deduplicated_v20240930.parquet      All cases, all drugs
│   ├── DRUG_deduplicated_v20240930.parquet      All drugs, all roles
│   ├── DRUG_annotated_v20240930.parquet         All drugs + GLP-1 flags
│   ├── DRUG_glp1_ps_v20240930.parquet           GLP-1 primary suspect only
│   ├── DEMO_glp1_v20240930.parquet              Cases with ≥1 GLP-1 PS drug
│   ├── REAC_glp1_soc_v20240930.parquet          GLP-1 reactions + SOC codes
│   ├── SOC_summary_v20240930.csv                Cases per SOC (chart-ready)
│   ├── signals_pt_v20240930.parquet             PT-level signals (ROR/PRR/IC)
│   ├── signals_soc_v20240930.parquet            SOC-level signals
│   ├── tto_v20240930.parquet                    Time-to-onset records
│   └── tto_summary_v20240930.parquet            Per-drug TTO statistics
└── logs/
    ├── dedup_audit_v20240930.csv           Phase 1: dedup counts per quarter
    ├── normalization_audit_v20240930.json  Phase 2: match rates by tier
    ├── meddra_audit_v20240930.json         Phase 3: SOC mapping rate + unmapped
    └── signal_audit_v20240930.json         Phase 4: signal counts, thresholds

docs/
├── data_dictionary.csv             Every column: dtype, null rate, samples
└── eda_report_v20240930.html       Self-contained interactive report
```

Version tags (`v20240930`) are tied to the last quarter's end date, not the run date.

---

## Data decision log

### Phase 1 — Deduplication

FDA-recommended 2-step rule (Potter et al. 2025, *Clin Pharmacol Ther*):
1. Per `CASEID`, keep the record with the most recent `FDA_DT`
2. On ties, keep the highest `PRIMARYID`

We do **not** use `CASEVERSION` — it is unreliable across reporting sources.

**AERS era (pre-2012):** Legacy AERS files (2004 Q1 → 2012 Q2) use different column names: `isr` instead of `primaryid`, `case` instead of `caseid`, `gndr_cod` instead of `sex`. The pipeline renames these at parse time so all downstream code is era-agnostic. See `schema.py`.

**2012 Q3 — missing quarter:** FDA never published this quarter. It falls at the AERS→FAERS transition (Sept 10, 2012). The pipeline skips it with status `known_missing`.

### Phase 2 — Drug normalization

Three-tier lookup for each DRUG row:
- **Tier 1:** `prod_ai` (structured active ingredient field) — most reliable
- **Tier 2:** `drugname` exact match after cleaning (strip dose, trailing qualifiers)
- **Tier 3:** prefix matching for name variants not in the reference

We use **primary suspect (`role_cod = PS`) only** for signal analysis. Concomitant drugs are retained in the full annotated file for sensitivity analysis.

Compounded products (semaglutide acetate, compounded tirzepatide, etc.) are flagged with `is_compounded = True` and included by default. FDA documented underreporting of these products and distinct safety profiles (2024–2025 warnings).

### Phase 3 — MedDRA hierarchy

PT→SOC join using three sources in order:
1. `mdhier.asc` from your MedDRA subscription (pass with `--mdhier`)
2. Bundled GLP-1-relevant PT→SOC mapping (~110 PTs from published studies)
3. Unmapped PTs → `meddra_src = 'unmapped'`, logged to audit

**Primary SOC only.** A PT can belong to multiple SOCs; we use the primary classification — consistent with all published FAERS studies and FDA's own signal reports.

### Phase 4 — Signal detection

Three complementary algorithms; a signal requires **≥2 of 3 methods**:

| Method | Threshold | Reference |
|--------|-----------|-----------|
| ROR 95% CI lower bound | > 1.0 | Rothman et al., *Pharmacoepidemiol* |
| PRR with chi-square | PRR ≥ 2.0, χ² ≥ 4.0, a ≥ 3 | Evans et al. 2001 |
| IC025 (BCPNN lower bound) | > 0.0 | Norén et al. 2006, *Drug Saf* |

Minimum case count: **a ≥ 3** (below this, estimates are unreliable regardless of value).

**Denominator:** N = total unique cases in the full deduplicated DEMO file — not just GLP-1 cases. Disproportionality requires the full FAERS reporting universe.

**No Bonferroni correction by default.** With ~3,500 drug-PT pairs, Bonferroni is extremely conservative. Published GLP-1 FAERS studies use uncorrected thresholds. The `n_signals` column allows downstream Bonferroni application.

---

## Project structure

```
faers_glp1_watch/
├── faers_pipeline/
│   ├── quarters.py         Quarter registry, GLP-1 milestone annotations
│   ├── schema.py           AERS→FAERS column renames, known missing quarters
│   ├── downloader.py       Retry download with manifest + known-missing skip
│   ├── parser.py           ZIP extraction, encoding/delimiter handling
│   ├── deduplicator.py     FDA-recommended 2-step dedup + audit logging
│   ├── drug_reference.py   GLP-1 brand/generic/compounded lookup table
│   ├── normalizer.py       3-tier drug name normalization + PS filter
│   ├── meddra.py           MedDRA SOC hierarchy loader and PT join
│   ├── signal_detection.py ROR / PRR / IC signal detection + TTO analysis
│   ├── eda_report.py       Self-contained HTML EDA report (Plotly)
│   ├── audit_chart.py      Standalone dedup quality chart (matplotlib)
│   ├── writer.py           Versioned Parquet + audit log output
│   └── pipeline.py         Orchestrator + CLI (Phases 1–5)
├── tests/
│   ├── conftest.py                     Shared fixtures
│   ├── test_quarters.py                Quarter class and registry (10 tests)
│   ├── test_schema.py                  AERS/FAERS schema normalization (25 tests)
│   ├── test_deduplicator.py            Dedup logic (16 tests)
│   ├── test_pipeline_integration.py    End-to-end with synthetic ZIP (11 tests)
│   ├── test_drug_reference.py          Drug lookup map (26 tests)
│   ├── test_normalizer.py              3-tier normalization (48 tests)
│   ├── test_meddra.py                  MedDRA SOC join (27 tests)
│   ├── test_signal_detection.py        ROR/PRR/IC math + pipeline (42 tests)
│   └── test_eda_report.py              EDA report generation (25 tests)
├── data/                   Git-ignored; created on first run
├── docs/                   EDA report and data dictionary written here
├── requirements.txt
└── README.md
```

**241 tests, all passing.** Tests run without any FDA data — a synthetic ZIP is built in memory for integration tests.

---

## GLP-1 drug scope

| Drug | Mechanism | Approval | Key brands |
|------|-----------|----------|-----------|
| Exenatide | GLP-1 RA | 2005 Q2 | Byetta, Bydureon |
| Liraglutide | GLP-1 RA | 2010 Q1 | Victoza, Saxenda |
| Dulaglutide | GLP-1 RA | 2014 Q3 | Trulicity |
| Lixisenatide | GLP-1 RA | 2016 Q3 | Adlyxin |
| Semaglutide | GLP-1 RA | 2017 Q4 | Ozempic, Wegovy, Rybelsus |
| Albiglutide | GLP-1 RA | 2014 Q2 | Tanzeum (withdrawn 2017) |
| Tirzepatide | Dual GIP/GLP-1 | 2022 Q2 | Mounjaro, Zepbound |

Scope: **2005 Q2 → 2024 Q3** (78 quarters, 77 downloadable — 2012 Q3 never published)

---

## References

- Potter E, Reyes M, Naples J, Dal Pan G. (2025). FDA Adverse Event Reporting System (FAERS) Essentials. *Clin Pharmacol Ther* 118(3):567–582.
- Scientific Reports (2025). Pharmacovigilance analysis of neurological adverse events associated with GLP-1 receptor agonists based on the FDA FAERS database.
- Evans SJW et al. (2001). Use of proportional reporting ratios for signal generation. *Pharmacoepidemiol Drug Saf* 10(6):483–486.
- Norén GN et al. (2006). A statistical methodology for drug-drug interaction surveillance. *Stat Med* 25(9):1621–1632.
- FDA. (2025). FDA's Concerns with Unapproved GLP-1 Drugs Used for Weight Loss.
- FDA. (2024). Best Practices for FDA Staff in the Postmarketing Safety Surveillance of Human Drug and Biological Products.

---

## License

MIT. Data sourced from FDA public databases, in the public domain.
