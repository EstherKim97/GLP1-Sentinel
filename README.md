# FAERS-GLP1-Watch
### End-to-end pharmacovigilance pipeline for the GLP-1 drug class

[![Tests](https://img.shields.io/badge/tests-241%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Data](https://img.shields.io/badge/FAERS-2005Q2--2024Q3-orange)](https://fis.fda.gov/)

---

## Why this exists

In 2024, GLP-1 receptor agonists became the fastest-adopted prescription 
drug class in US history. Ozempic, Wegovy, and Mounjaro prescriptions grew 
300% year-over-year. At the same time, FDA staff reductions cut the 
pharmacovigilance workforce responsible for monitoring adverse event signals 
from this exact class.

FDA's own quarterly signal reports were already flagging new serious risks —
pulmonary aspiration under anesthesia, neurological adverse events, 
gastrointestinal complications requiring hospitalization. Compounded 
semaglutide (unapproved salt forms sold online) was generating a separate 
wave of reports the standard pipeline wasn't designed to distinguish.

I built this pipeline to do what that reduced workforce no longer could at 
scale: ingest 20 years of raw FDA adverse event data, resolve its known 
quality problems, and surface the signals that matter.

---

## What it found

Running across 20,904,555 total FAERS cases from 2005 Q2 through 2024 Q3,
the pipeline identified 288,173 GLP-1 primary-suspect cases and detected
**2,665 statistically significant drug-reaction signals** out of 10,131 
tested pairs.

The top findings are clinically meaningful:

| Drug | Reaction | Cases | ROR | CI | Methods |
|------|----------|-------|-----|----|---------|
| Liraglutide | Thyroid C-cell hyperplasia | 4 | 811 | 182–3,624 | 3/3 |
| Exenatide | Early satiety | 698 | 868 | 742–1,016 | 3/3 |
| Semaglutide | Idiopathic pancreatitis | 4 | 247 | 76–802 | 3/3 |
| Semaglutide | Allodynia | 168 | 149 | 125–176 | 3/3 |
| Tirzepatide | Injection site coldness | 136 | 124 | 102–151 | 3/3 |
| Dulaglutide | Diabetic amyotrophy | 6 | 114 | 45–291 | 3/3 |

Thyroid C-cell hyperplasia in liraglutide matches the FDA black box warning.
Pancreatitis in semaglutide matches the 2024 FDA signal report. Allodynia 
(nerve pain) in semaglutide is consistent with the 2025 Scientific Reports 
neurological AE study. These aren't accidents — they're what a correctly 
built pipeline should find.

The SOC heatmap shows gastrointestinal disorders are the dominant signal 
class (79,690 unique cases), followed by investigations (45,827) and 
metabolism and nutrition disorders (39,226). Semaglutide and liraglutide 
show the broadest SOC signal coverage, consistent with their longer market 
history and higher reporting volume.

---

## The data problem nobody talks about

### 1. The AERS era uses completely different column names

Pre-2012 FDA AERS files call the case identifier `isr` instead of 
`primaryid` and `case` instead of `caseid`. Every data row also ends with 
a trailing `$` delimiter. This causes pandas to silently treat the first 
data column as the DataFrame index, shifting every value one position to 
the right. `caseid` gets the `i_f_cod` value (`'I'`), which fails numeric 
casting to `NaN`. `fda_dt` gets `rept_cod` (`'DIR'`/`'EXP'`).

The result: **100% of AERS records were being dropped as duplicates with 
zero error messages.** The deduplication audit showed `80,524 raw → 0 
unique cases` for every quarter from 2005 through 2012.

Finding this required reading raw bytes from the ZIP files:
Header: ISRCASECASE
CASEI_F_COD......
...CONFID   (22 fields)
Row 1:  454824158536335853633
5853633I$$...NN
NN$    (23 fields — trailing $)

The fix was one parameter: `index_col=False` in `pd.read_csv()`.

### 2. 2012 Q3 was never published

This quarter falls at the AERS→FAERS transition (September 10, 2012) and 
was never released by FDA. Without explicit handling, the pipeline retried 
it five times on every run and logged a failure. It's now in 
`KNOWN_MISSING_QUARTERS` and skipped silently.

### 3. Deduplication is not obvious

FAERS contains follow-up reports — the same adverse event submitted 
multiple times as the case evolves. FDA's recommended two-step rule 
(Potter et al. 2025, *Clin Pharmacol Ther*): keep the latest `FDA_DT` 
per `CASEID`, break ties on highest `PRIMARYID`. Using `CASEVERSION` 
instead — which looks like the right field — produces wrong results 
because it's unreliable across reporting sources.

Final dedup rates: 12–17% removal for AERS era quarters (cumulative 
files with genuine follow-up reports), 0% for FAERS era quarters (each 
quarter contains only new reports).

### 4. The signal denominator matters enormously

The `c` cell in the 2×2 contingency table — cases *without* the drug 
that still had the reaction — must come from the full 67-million-row REAC 
database, not just GLP-1 cases. Using GLP-1-only REAC sets the background 
rate to near-zero and makes every reaction look like a signal.

The fix: pre-aggregate the full REAC file to a `{PT: unique_case_count}` 
dictionary (38,457 unique preferred terms) before signal detection, 
reducing the memory footprint from 67M rows to a small dict.

### 5. Memory on 7.8 GB RAM

The full dataset is 2.9 GB of ZIPs producing 5 GB of Parquet. The naive 
approach — accumulate all 77 quarters in RAM then write — terminates 
around 2010 Q4. The solution was a full streaming rewrite:

- Parse → write per-quarter Parquet immediately → delete from RAM
- Merge using `ParquetWriter` one quarter at a time
- Normalize DRUG in 2M-row chunks via `iter_batches()`
- Stream REAC and keep only GLP-1-scoped rows in memory
- Pre-aggregate REAC marginal to avoid loading 67M rows for signal math

Peak RAM usage: ~2 GB at any point during the full pipeline run.

---

## Pipeline overview
Phase 1  Download (77 quarters, 2.9 GB) → Parse → Deduplicate → Parquet
Phase 2  Drug normalization → GLP-1 scoping → Primary suspect filter
Phase 3  MedDRA SOC join → Missingness audit
Phase 4  ROR / PRR / IC signal detection (≥2/3 methods, a ≥ 3)
Phase 5  Self-contained interactive HTML report (Plotly)

data/processed/
DEMO_deduplicated_v20240930.parquet     20,907,551 rows  680 MB
DRUG_deduplicated_v20240930.parquet     82,161,150 rows  1,216 MB
DRUG_glp1_ps_v20240930.parquet            288,175 rows
DEMO_glp1_v20240930.parquet               288,173 rows
REAC_glp1_soc_v20240930.parquet           788,355 rows
SOC_summary_v20240930.csv                  18 SOCs
signals_pt_v20240930.parquet            2,665 signals / 10,131 pairs
signals_soc_v20240930.parquet           SOC-level signals
docs/
eda_report.html                         Interactive report (Plotly)

---

## Signal detection methodology

Three algorithms, signal requires ≥2/3 to agree:

| Method | Threshold | Reference |
|--------|-----------|-----------|
| ROR 95% CI lower bound | > 1.0 | Rothman et al. |
| PRR + chi-square | PRR ≥ 2.0, χ² ≥ 4.0, a ≥ 3 | Evans et al. 2001 |
| IC025 (BCPNN) | > 0.0 | Norén et al. 2006 |

Denominator: N = 20,904,555 unique cases in full FAERS database.
No Bonferroni correction applied (standard in published GLP-1 FAERS 
studies). The `n_signals` column allows downstream correction.

---

## GLP-1 drug scope

| Drug | Approval | Cases found |
|------|----------|-------------|
| Exenatide | 2005 Q2 | 83,997 |
| Dulaglutide | 2014 Q3 | 68,605 |
| Tirzepatide | 2022 Q2 | 54,127 |
| Semaglutide | 2017 Q4 | 37,537 |
| Liraglutide | 2010 Q1 | 34,314 |
| Albiglutide | 2014 Q2 | 9,446 |
| Lixisenatide | 2016 Q3 | 108 |

---

## Missingness

From the GLP-1 DEMO records: age_grp 83.7% missing, weight 79.7% missing,
event_dt 50.4% missing, age 43.9% missing. Consistent with published 
FAERS literature. Does not affect signal detection which operates on 
reaction counts not demographics.

---

## Tests

241 tests across all modules. Run without any FDA data:
```bash
python -m pytest tests/ -v
```

---

## References

- Potter E et al. (2025). FDA FAERS Essentials. *Clin Pharmacol Ther*
- Evans SJW et al. (2001). PRR for signal generation.  *Pharmacoepidemiol Drug Saf*
- Norén GN et al. (2006). Drug-drug interaction surveillance. *Stat Med*
- Scientific Reports (2025). Neurological AEs associated with GLP-1 RAs
- FDA (2024). Best Practices for Postmarketing Safety Surveillance