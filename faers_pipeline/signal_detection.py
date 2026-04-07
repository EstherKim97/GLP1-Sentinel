"""
signal_detection.py
-------------------
Phase 4: Disproportionality signal detection for GLP-1 adverse events.

Implements three complementary signal detection algorithms on the
2×2 contingency table built from FAERS data:
  - ROR  (Reporting Odds Ratio)      — primary metric
  - PRR  (Proportional Reporting Ratio) — secondary confirmation
  - IC   (Information Component, BCPNN) — Bayesian measure

A signal is flagged when ≥2 of 3 methods exceed their threshold.
This multi-algorithm approach mirrors FDA's own surveillance practice.

DECISION LOG — Algorithm choices
-----------------------------------

Why three algorithms instead of one?
  No single metric is optimal in all scenarios. ROR is powerful but
  inflated by large N. PRR is intuitive but sensitive to rare events.
  IC is robust for rare events but harder to interpret. Requiring
  consensus (≥2/3) reduces both false positives and false negatives.
  This is the approach used in:
    - Scientific Reports 2025 (GLP-1 neurological AEs)
    - PLoS One 2025 (Definity pharmacovigilance)
    - FDA FAERS Best Practices 2024

Signal thresholds (all from published pharmacovigilance literature):
  ROR:  lower bound of 95% CI > 1.0  (Rothman et al., Pharmacoepidemiol)
  PRR:  PRR ≥ 2.0  AND  chi-square ≥ 4.0  AND  a ≥ 3
        (Evans et al., 2001, Pharmacoepidemiol Drug Saf)
  IC:   IC025 > 0  (Norén et al., 2006, Drug Saf)
        (IC025 = lower bound of 95% credibility interval)

Minimum case count (a ≥ 3):
  Applied to ALL metrics. Below 3 cases, statistical estimates are
  unreliable regardless of the computed value. This is the standard
  minimum in published FAERS analyses and is stated explicitly in
  the Evans et al. PRR threshold paper.

Denominator (N):
  N = total unique cases in the full deduplicated DEMO file.
  NOT just GLP-1 cases. This is the critical design decision —
  the denominator must be the full FAERS reporting universe for
  disproportionality to be meaningful. Using only GLP-1 cases as
  the denominator would make every reaction appear as a signal.

Analysis levels:
  PT level  (Preferred Term): ~500 distinct PTs in GLP-1 reports.
            Granular — "Pancreatitis" rather than "Hepatobiliary disorders".
            Used for clinical interpretation.
  SOC level (System Organ Class): 27 SOCs.
            Summary — "Gastrointestinal disorders".
            Used for EDA charts and portfolio presentation.

Time-to-onset:
  Weibull distribution fit per (drug, SOC) pair using THER start dates.
  Reports median onset days and interquartile range.
  Based on methodology in Scientific Reports 2025 GLP-1 neurological study.

DECISION: We do NOT apply Bonferroni correction by default.
  Reason: With ~3,500 drug-PT pairs tested, Bonferroni is extremely
  conservative and would suppress many real signals. Published GLP-1
  FAERS studies (Scientific Reports 2025) use uncorrected thresholds
  and note Bonferroni as a sensitivity analysis. We flag the uncorrected
  signals and note in the output that Bonferroni would require
  ROR CI lower bound > (threshold × n_tests). Applying it is a
  downstream analysis decision, not a pipeline decision.

Output columns
  drug             active ingredient name
  pt / soc_name    reaction term
  a                cases with drug AND reaction
  b                cases with drug, NOT reaction
  c                cases without drug, WITH reaction
  d                cases without drug, NOT reaction
  N                total cases in database
  ror              Reporting Odds Ratio
  ror_lb / ror_ub  95% CI bounds
  prr              Proportional Reporting Ratio
  prr_chi2         Chi-square statistic for PRR
  ic               Information Component
  ic025            Lower 95% credibility interval
  signal_ror       bool — ROR threshold met
  signal_prr       bool — PRR threshold met
  signal_ic        bool — IC threshold met
  n_signals        int  — how many methods flagged this pair (0-3)
  is_signal        bool — True if n_signals >= 2
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Signal thresholds ─────────────────────────────────────────────────────────

MIN_CASES        = 3       # Minimum a for any calculation
ROR_CI_THRESHOLD = 1.0     # ROR lower 95% CI must exceed this
PRR_THRESHOLD    = 2.0     # PRR must be >= this
PRR_CHI2_THRESH  = 4.0     # Chi-square must be >= this
IC025_THRESHOLD  = 0.0     # IC025 must exceed this
MIN_METHODS      = 2       # Signals must meet >= this many criteria


# ── Cell-level calculations ───────────────────────────────────────────────────

def _ror(a: int, b: int, c: int, d: int
         ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute Reporting Odds Ratio and 95% CI.

    ROR = (a * d) / (b * c)
    SE(ln ROR) = sqrt(1/a + 1/b + 1/c + 1/d)
    CI = exp(ln(ROR) ± 1.96 * SE)

    Returns (ror, lb, ub) or (None, None, None) if not computable.
    """
    if b == 0 or c == 0 or a == 0 or d == 0:
        return None, None, None
    ror = (a * d) / (b * c)
    se  = math.sqrt(1/a + 1/b + 1/c + 1/d)
    lb  = math.exp(math.log(ror) - 1.96 * se)
    ub  = math.exp(math.log(ror) + 1.96 * se)
    return round(ror, 4), round(lb, 4), round(ub, 4)


def _prr(a: int, b: int, c: int, d: int
         ) -> tuple[Optional[float], Optional[float]]:
    """
    Compute Proportional Reporting Ratio and chi-square statistic.

    PRR = (a / (a + b)) / (c / (c + d))
    Chi-square using expected cell count E = (a+b) * c/(c+d)

    Returns (prr, chi2) or (None, None).
    """
    n_drug   = a + b
    n_no_drug= c + d
    if n_drug == 0 or n_no_drug == 0:
        return None, None
    prop_d   = a / n_drug
    prop_nod = c / n_no_drug
    if prop_nod == 0:
        return None, None
    prr      = prop_d / prop_nod
    expected = n_drug * prop_nod
    chi2     = (a - expected) ** 2 / expected if expected > 0 else 0.0
    return round(prr, 4), round(chi2, 4)


def _ic(a: int, b: int, c: int, d: int
        ) -> tuple[Optional[float], Optional[float]]:
    """
    Compute Information Component (IC) and IC025 lower credibility bound.

    IC = log2(O / E)  where:
      O = observed co-occurrences (a)
      E = expected under independence = N * P(D) * P(E)
        = N * (a+b)/N * (a+c)/N
        = (a+b)(a+c) / N

    IC025 = IC - 1.96 * sqrt(Var(IC))

    Variance approximation (Norén et al. 2006):
      Var(IC) ≈ sum(1 / ((x_i + 0.5) * ln(2)^2)) for each cell

    Returns (ic, ic025) or (None, None).
    """
    N = a + b + c + d
    if N == 0 or a == 0:
        return None, None
    expected = ((a + b) * (a + c)) / N
    if expected <= 0:
        return None, None
    ic   = math.log2(a / expected)
    var  = sum(
        1 / ((x + 0.5) * math.log(2) ** 2)
        for x in [a, b, c, d]
    )
    ic025 = ic - 1.96 * math.sqrt(var)
    return round(ic, 4), round(ic025, 4)


# ── Contingency table builder ─────────────────────────────────────────────────

def build_contingency_tables(
    drug_glp1_ps    : pd.DataFrame,
    reac_with_soc   : pd.DataFrame,
    n_total_cases   : int,
    level           : str = "pt",
    full_reac       : pd.DataFrame | None = None,
    reac_pt_marginal: dict | None = None,
) -> pd.DataFrame:
    reaction_col = "pt" if level == "pt" else "soc_name"
    drug_col     = "glp1_active_ingredient"

    if reaction_col not in reac_with_soc.columns:
        logger.error(f"  Column '{reaction_col}' not in REAC DataFrame")
        return pd.DataFrame()

    def _norm_pid(df, col="primaryid"):
        d = df.copy()
        d["_pid"] = pd.to_numeric(d[col], errors="coerce")
        return d

    drug_clean = _norm_pid(drug_glp1_ps)
    reac_clean = _norm_pid(reac_with_soc)

    # Cell a
    merged = drug_clean[["_pid", drug_col]].merge(
        reac_clean[["_pid", reaction_col]], on="_pid", how="inner")
    cell_a = (
        merged.groupby([drug_col, reaction_col])["_pid"].nunique()
        .reset_index()
        .rename(columns={"_pid": "a", drug_col: "drug", reaction_col: "reaction_term"})
    )

    # Drug marginal (a + b)
    drug_totals = (
        drug_clean.groupby(drug_col)["_pid"].nunique()
        .rename("drug_total").reset_index()
        .rename(columns={drug_col: "drug"})
    )

    # Reaction marginal (a + c)
    if reac_pt_marginal is not None:
        # Pre-aggregated dict — most memory efficient, ~500 rows
        if level == "soc":
            pt_to_soc = (
                reac_with_soc[["pt", "soc_name"]].dropna()
                .drop_duplicates("pt").set_index("pt")["soc_name"].to_dict()
            )
            soc_counts: dict = {}
            for pt, n in reac_pt_marginal.items():
                soc = pt_to_soc.get(pt)
                if soc:
                    soc_counts[soc] = soc_counts.get(soc, 0) + n
            reaction_totals = pd.DataFrame(
                list(soc_counts.items()), columns=["reaction_term", "reaction_total"])
        else:
            reaction_totals = pd.DataFrame(
                list(reac_pt_marginal.items()),
                columns=["reaction_term", "reaction_total"])
    elif full_reac is not None and not full_reac.empty:
        reac_for_marginal = _norm_pid(full_reac)
        if level == "soc" and "soc_name" not in full_reac.columns:
            pt_to_soc = (
                reac_with_soc[["pt", "soc_name"]].dropna()
                .drop_duplicates("pt").set_index("pt")["soc_name"].to_dict()
            )
            reac_for_marginal["soc_name"] = reac_for_marginal["pt"].map(pt_to_soc)
        marginal_col = reaction_col if reaction_col in reac_for_marginal.columns else "pt"
        reaction_totals = (
            reac_for_marginal.groupby(marginal_col)["_pid"].nunique().reset_index()
            .rename(columns={"_pid": "reaction_total", marginal_col: "reaction_term"})
        )
    else:
        logger.warning("  No full_reac — using GLP-1-scoped REAC for marginals (c undercount)")
        reaction_totals = (
            reac_clean.groupby(reaction_col)["_pid"].nunique().reset_index()
            .rename(columns={"_pid": "reaction_total", reaction_col: "reaction_term"})
        )

    # Assemble
    ct = (
        cell_a
        .merge(drug_totals,     on="drug",         how="left")
        .merge(reaction_totals, on="reaction_term", how="left")
    )
    ct["reaction_total"] = ct["reaction_total"].fillna(ct["a"])
    ct["N"] = n_total_cases

# ── Apply signal metrics to contingency table ─────────────────────────────────

def compute_signals(
    ct: pd.DataFrame,
    min_cases: int = MIN_CASES,
) -> pd.DataFrame:
    """
    Compute ROR, PRR, IC for each row of a contingency table DataFrame.

    Rows with a < min_cases are retained but marked as insufficient data.

    Args:
        ct:        DataFrame with columns a, b, c, d, N.
        min_cases: Minimum case count (a) for signal calculation.

    Returns:
        DataFrame with signal metric columns added.
    """
    n_rows    = len(ct)
    n_eligible = (ct["a"] >= min_cases).sum()
    logger.info(
        f"  Signal computation: {n_rows:,} pairs, "
        f"{n_eligible:,} meet min_cases={min_cases}"
    )

    # Pre-allocate result columns
    ror_vals   = np.full(n_rows, np.nan)
    ror_lb     = np.full(n_rows, np.nan)
    ror_ub     = np.full(n_rows, np.nan)
    prr_vals   = np.full(n_rows, np.nan)
    chi2_vals  = np.full(n_rows, np.nan)
    ic_vals    = np.full(n_rows, np.nan)
    ic025_vals = np.full(n_rows, np.nan)

    for i, row in enumerate(ct.itertuples(index=False)):
        if row.a < min_cases:
            continue
        a, b, c, d = int(row.a), int(row.b), int(row.c), int(row.d)

        ror_v, lb, ub   = _ror(a, b, c, d)
        prr_v, chi2_v   = _prr(a, b, c, d)
        ic_v, ic025_v   = _ic(a, b, c, d)

        if ror_v   is not None: ror_vals[i]   = ror_v
        if lb      is not None: ror_lb[i]     = lb
        if ub      is not None: ror_ub[i]     = ub
        if prr_v   is not None: prr_vals[i]   = prr_v
        if chi2_v  is not None: chi2_vals[i]  = chi2_v
        if ic_v    is not None: ic_vals[i]    = ic_v
        if ic025_v is not None: ic025_vals[i] = ic025_v

    ct = ct.copy()
    ct["ror"]      = ror_vals
    ct["ror_lb"]   = ror_lb
    ct["ror_ub"]   = ror_ub
    ct["prr"]      = prr_vals
    ct["prr_chi2"] = chi2_vals
    ct["ic"]       = ic_vals
    ct["ic025"]    = ic025_vals

    # ── Signal flags ──────────────────────────────────────────────────────────
    has_data     = ct["a"] >= min_cases

    ct["signal_ror"] = (
        has_data
        & ct["ror_lb"].notna()
        & (ct["ror_lb"] > ROR_CI_THRESHOLD)
    )
    ct["signal_prr"] = (
        has_data
        & ct["prr"].notna()
        & (ct["prr"] >= PRR_THRESHOLD)
        & (ct["prr_chi2"] >= PRR_CHI2_THRESH)
        & (ct["a"] >= MIN_CASES)
    )
    ct["signal_ic"] = (
        has_data
        & ct["ic025"].notna()
        & (ct["ic025"] > IC025_THRESHOLD)
    )

    ct["n_signals"] = (
        ct["signal_ror"].astype(int)
        + ct["signal_prr"].astype(int)
        + ct["signal_ic"].astype(int)
    )
    ct["is_signal"] = ct["n_signals"] >= MIN_METHODS

    n_signals = ct["is_signal"].sum()
    logger.info(
        f"  Signals detected: {n_signals:,} pairs flagged by ≥{MIN_METHODS} methods"
    )
    return ct


# ── Time-to-onset analysis ────────────────────────────────────────────────────

def time_to_onset(
    drug_glp1_ps: pd.DataFrame,
    ther_df: pd.DataFrame,
    demo_glp1: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute time-to-onset (TTO) in days for each GLP-1 drug.

    TTO = event_dt (from DEMO) - start_dt (from THER)
    Joined on primaryid + drug_seq for matching drug-specific therapy dates.

    Returns DataFrame with one row per (drug, primaryid):
      drug, primaryid, tto_days

    DECISION: use event_dt from DEMO as the adverse event date.
    THER has start_dt for when the drug was started. The difference
    gives onset from initiation, which is the clinically meaningful
    measure. Many reports lack exact dates — we include only rows
    where both dates are valid and TTO is non-negative (≥0 days).

    DECISION: We do NOT fit the Weibull model inside this function.
    That's a separate analysis function so it can be run independently
    and the TTO data can be inspected before modeling.
    """
    if ther_df is None or ther_df.empty:
        logger.warning("  TTO: THER file empty — skipping time-to-onset")
        return pd.DataFrame()

    drug_col = "glp1_active_ingredient"

    def _parse_date(series: pd.Series) -> pd.Series:
        """Parse YYYYMMDD integer dates to datetime, coerce invalids to NaT."""
        return pd.to_datetime(
            series.astype(str).str[:8].str.strip(),
            format="%Y%m%d",
            errors="coerce",
        )

    # Normalize primaryid in all three DataFrames
    drug_ps   = drug_glp1_ps[["primaryid", "drug_seq", drug_col]].copy()
    ther_work = ther_df[["primaryid", "drug_seq", "start_dt"]].copy()
    demo_work = demo_glp1[["primaryid", "event_dt"]].copy()

    for df in [drug_ps, ther_work, demo_work]:
        df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")

    ther_work["drug_seq"] = pd.to_numeric(ther_work["drug_seq"], errors="coerce")
    drug_ps["drug_seq"]   = pd.to_numeric(drug_ps["drug_seq"],   errors="coerce")

    # Join drug → therapy start dates (on primaryid + drug_seq)
    merged = drug_ps.merge(ther_work, on=["primaryid", "drug_seq"], how="inner")

    # Join adverse event date from DEMO
    merged = merged.merge(demo_work, on="primaryid", how="inner")

    # Parse dates
    merged["_start_dt"]  = _parse_date(merged["start_dt"])
    merged["_event_dt"]  = _parse_date(merged["event_dt"])

    # Compute TTO
    merged["tto_days"] = (merged["_event_dt"] - merged["_start_dt"]).dt.days

    # Keep only valid, non-negative TTO
    valid = (
        merged["tto_days"].notna()
        & (merged["tto_days"] >= 0)
        & (merged["tto_days"] <= 3650)   # cap at 10 years (data quality)
    )
    result = merged[valid][[
        drug_col, "primaryid", "tto_days"
    ]].rename(columns={drug_col: "drug"})

    logger.info(
        f"  TTO: {len(result):,} valid onset records across "
        f"{result['drug'].nunique()} drugs"
    )
    return result


def tto_summary(tto_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-drug TTO summary statistics.

    Returns DataFrame with one row per drug:
      drug, n, median_days, q25_days, q75_days, mean_days, pct_within_30d
    """
    if tto_df.empty:
        return pd.DataFrame()

    def _summarise(grp):
        days = grp["tto_days"]
        return pd.Series({
            "n"              : len(days),
            "median_days"    : round(days.median(), 1),
            "q25_days"       : round(days.quantile(0.25), 1),
            "q75_days"       : round(days.quantile(0.75), 1),
            "mean_days"      : round(days.mean(), 1),
            "pct_within_30d" : round((days <= 30).mean() * 100, 1),
        })

    return (
        tto_df.groupby("drug")
        .apply(_summarise)
        .reset_index()
        .sort_values("median_days")
    )


# ── Full Phase 4 runner ───────────────────────────────────────────────────────

def run_signal_detection(
    drug_glp1_ps     : pd.DataFrame,
    reac_with_soc    : pd.DataFrame,
    demo_all         : pd.DataFrame,
    ther_df          : pd.DataFrame | None = None,
    demo_glp1        : pd.DataFrame | None = None,
    full_reac        : pd.DataFrame | None = None,
    reac_pt_marginal : dict | None = None,
    min_cases        : int = MIN_CASES,
) -> dict[str, pd.DataFrame]:
    """
    Run full Phase 4 signal detection pipeline.

    Args:
        drug_glp1_ps:  GLP-1 PS DRUG rows (Phase 2 output).
        reac_with_soc: GLP-1 REAC rows with SOC (Phase 3 output).
        demo_all:      Full deduplicated DEMO (Phase 1 output).
                       Used for total N denominator.
        ther_df:       THER file (optional, for TTO analysis).
        demo_glp1:     GLP-1-scoped DEMO (optional, for TTO analysis).
        full_reac:     Full REAC file for all drugs (Phase 1 output).
                       Required for correct c-cell marginal calculation.
        min_cases:     Minimum cases (a) for signal calculation.

    Returns:
        Dict with keys:
          'signals_pt'  — signals at PT level
          'signals_soc' — signals at SOC level
          'tto'         — time-to-onset records
          'tto_summary' — per-drug TTO statistics
          'audit'       — signal audit record
    """
    # Total unique cases in full database — the denominator N
    n_total = int(
        pd.to_numeric(demo_all["primaryid"], errors="coerce")
        .dropna()
        .nunique()
    )
    logger.info(f"  Total unique cases in database (N): {n_total:,}")

    results = {}

    # ── PT-level signals ──────────────────────────────────────────────────────
    logger.info("\n  Computing PT-level contingency tables...")
    ct_pt = build_contingency_tables(
        drug_glp1_ps, reac_with_soc, n_total, level="pt",
        full_reac=full_reac, reac_pt_marginal=reac_pt_marginal
    )
    if not ct_pt.empty:
        signals_pt = compute_signals(ct_pt, min_cases)
        results["signals_pt"] = signals_pt

    # ── SOC-level signals ─────────────────────────────────────────────────────
    logger.info("\n  Computing SOC-level contingency tables...")
    ct_soc = build_contingency_tables(
        drug_glp1_ps, reac_with_soc, n_total, level="soc",
        full_reac=full_reac, reac_pt_marginal=reac_pt_marginal
    )
    if not ct_soc.empty:
        signals_soc = compute_signals(ct_soc, min_cases)
        results["signals_soc"] = signals_soc

    # ── Time-to-onset ─────────────────────────────────────────────────────────
    if ther_df is not None and demo_glp1 is not None:
        logger.info("\n  Computing time-to-onset...")
        tto_df = time_to_onset(drug_glp1_ps, ther_df, demo_glp1)
        results["tto"]         = tto_df
        results["tto_summary"] = tto_summary(tto_df)

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit = {
        "n_total_cases"      : n_total,
        "min_cases_threshold": min_cases,
        "ror_threshold"      : f"lb > {ROR_CI_THRESHOLD}",
        "prr_threshold"      : f"PRR >= {PRR_THRESHOLD}, chi2 >= {PRR_CHI2_THRESH}, a >= {MIN_CASES}",
        "ic_threshold"       : f"IC025 > {IC025_THRESHOLD}",
        "min_methods"        : MIN_METHODS,
    }
    if "signals_pt" in results:
        spt = results["signals_pt"]
        audit["pt_pairs_tested"]    = len(spt)
        audit["pt_pairs_eligible"]  = int((spt["a"] >= min_cases).sum())
        audit["pt_signals_detected"]= int(spt["is_signal"].sum())
    if "signals_soc" in results:
        ss = results["signals_soc"]
        audit["soc_pairs_tested"]    = len(ss)
        audit["soc_signals_detected"]= int(ss["is_signal"].sum())
    results["audit"] = audit

    return results
