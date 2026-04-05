"""
test_signal_detection.py
------------------------
Tests for Phase 4 disproportionality signal detection.

Tests are organised by concern:
  - Individual metric math (_ror, _prr, _ic)
  - Contingency table construction
  - Signal flag logic (thresholds)
  - Time-to-onset calculation
  - Full runner integration

The math tests use hand-computed expected values so any regression
in the formulas is immediately visible.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from faers_pipeline.signal_detection import (
    _ror, _prr, _ic,
    build_contingency_tables,
    compute_signals,
    time_to_onset,
    tto_summary,
    run_signal_detection,
    MIN_CASES, ROR_CI_THRESHOLD, PRR_THRESHOLD, PRR_CHI2_THRESH, IC025_THRESHOLD,
)


# ── Math: _ror ────────────────────────────────────────────────────────────────

class TestRor:

    def test_basic_calculation(self):
        # ROR = (a*d)/(b*c) = (10*1000)/(90*100) = 10000/9000 ≈ 1.1111
        ror, lb, ub = _ror(10, 90, 100, 1000)
        assert abs(ror - 1.1111) < 0.001

    def test_strong_signal(self):
        # High a, low b: clear signal
        ror, lb, ub = _ror(150, 45000, 2000, 8000000)
        assert ror > 10
        assert lb > 1.0    # Signal: lower bound exceeds 1

    def test_null_signal(self):
        # Proportionate reporting: no signal
        ror, lb, ub = _ror(10, 1000, 100, 10000)
        assert abs(ror - 1.0) < 0.01
        # CI should straddle 1
        assert lb < 1.0
        assert ub > 1.0

    def test_zero_b_returns_none(self):
        # b = 0: drug only ever appears with this reaction (division by zero)
        ror, lb, ub = _ror(10, 0, 100, 1000)
        assert ror is None

    def test_zero_c_returns_none(self):
        ror, lb, ub = _ror(10, 90, 0, 1000)
        assert ror is None

    def test_zero_a_returns_none(self):
        ror, lb, ub = _ror(0, 90, 100, 1000)
        assert ror is None

    def test_ci_lb_less_than_ub(self):
        ror, lb, ub = _ror(50, 5000, 200, 500000)
        assert lb < ror < ub

    def test_symmetry(self):
        # Swapping drug/no-drug should give inverse ROR
        ror1, _, _ = _ror(10, 90, 100, 1000)
        ror2, _, _ = _ror(100, 1000, 10, 90)
        assert abs(ror1 * ror2 - 1.0) < 0.01


# ── Math: _prr ────────────────────────────────────────────────────────────────

class TestPrr:

    def test_basic_calculation(self):
        # PRR = (a/(a+b)) / (c/(c+d))
        # = (10/100) / (100/1100) = 0.10 / 0.0909 ≈ 1.1
        prr, chi2 = _prr(10, 90, 100, 1000)
        assert abs(prr - 1.1) < 0.01

    def test_strong_signal(self):
        prr, chi2 = _prr(150, 45000, 2000, 8000000)
        assert prr > 10
        assert chi2 > 4

    def test_null_signal_prr_near_one(self):
        prr, chi2 = _prr(10, 1000, 100, 10000)
        assert abs(prr - 1.0) < 0.01

    def test_zero_c_plus_d_returns_none(self):
        prr, chi2 = _prr(10, 90, 0, 0)
        assert prr is None

    def test_chi2_positive(self):
        _, chi2 = _prr(50, 500, 100, 5000)
        assert chi2 >= 0


# ── Math: _ic ─────────────────────────────────────────────────────────────────

class TestIc:

    def test_ic_positive_for_signal(self):
        ic, ic025 = _ic(150, 45000, 2000, 8000000)
        assert ic > 0
        assert ic025 > 0    # Signal

    def test_ic_near_zero_for_null(self):
        # Proportionate → IC ≈ 0
        ic, ic025 = _ic(10, 1000, 100, 10000)
        assert abs(ic) < 0.5

    def test_ic025_less_than_ic(self):
        ic, ic025 = _ic(50, 5000, 200, 500000)
        assert ic025 < ic

    def test_zero_a_returns_none(self):
        ic, ic025 = _ic(0, 90, 100, 1000)
        assert ic is None

    def test_zero_N_returns_none(self):
        ic, ic025 = _ic(0, 0, 0, 0)
        assert ic is None


# ── build_contingency_tables ──────────────────────────────────────────────────

def make_drug_ps(rows: list[dict]) -> pd.DataFrame:
    defaults = {"drug_seq": "1", "role_cod": "PS", "is_glp1": True, "is_primary_suspect": True}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def make_reac_soc(rows: list[dict]) -> pd.DataFrame:
    defaults = {"caseid": "1001", "drug_rec_act": "", "meddra_src": "bundled"}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def make_demo(n: int) -> pd.DataFrame:
    """Make a DEMO DataFrame with n unique primaryids."""
    return pd.DataFrame({"primaryid": list(range(1, n + 1))})


class TestBuildContingencyTables:

    def test_cell_a_correct(self):
        """a = unique cases with both drug and reaction."""
        drug = make_drug_ps([
            {"primaryid": "1", "glp1_active_ingredient": "semaglutide"},
            {"primaryid": "2", "glp1_active_ingredient": "semaglutide"},
        ])
        reac = make_reac_soc([
            {"primaryid": "1", "pt": "Nausea", "soc_name": "Gastrointestinal disorders"},
            {"primaryid": "2", "pt": "Nausea", "soc_name": "Gastrointestinal disorders"},
            {"primaryid": "3", "pt": "Nausea", "soc_name": "Gastrointestinal disorders"},  # no drug
        ])
        demo = make_demo(1000)
        ct   = build_contingency_tables(drug, reac, n_total_cases=1000, level="pt")
        row  = ct[(ct["drug"] == "semaglutide") & (ct["reaction_term"] == "Nausea")]
        assert len(row) == 1
        assert row.iloc[0]["a"] == 2   # cases 1 and 2

    def test_cell_c_correct(self):
        """c = cases with reaction but WITHOUT this drug."""
        drug = make_drug_ps([
            {"primaryid": "1", "glp1_active_ingredient": "semaglutide"},
        ])
        reac = make_reac_soc([
            {"primaryid": "1", "pt": "Nausea", "soc_name": "GI"},
            {"primaryid": "5", "pt": "Nausea", "soc_name": "GI"},  # no drug match
            {"primaryid": "6", "pt": "Nausea", "soc_name": "GI"},
        ])
        demo = make_demo(100)
        ct   = build_contingency_tables(drug, reac, n_total_cases=100, level="pt")
        row  = ct[(ct["drug"] == "semaglutide") & (ct["reaction_term"] == "Nausea")]
        # a=1, reaction_total=3, so c = 3 - 1 = 2
        assert row.iloc[0]["c"] == 2

    def test_n_equals_parameter(self):
        drug = make_drug_ps([{"primaryid": "1", "glp1_active_ingredient": "semaglutide"}])
        reac = make_reac_soc([{"primaryid": "1", "pt": "Nausea", "soc_name": "GI"}])
        ct   = build_contingency_tables(drug, reac, n_total_cases=5000, level="pt")
        assert (ct["N"] == 5000).all()

    def test_soc_level_works(self):
        drug = make_drug_ps([{"primaryid": "1", "glp1_active_ingredient": "semaglutide"}])
        reac = make_reac_soc([{"primaryid": "1", "pt": "Nausea",
                                "soc_name": "Gastrointestinal disorders"}])
        ct   = build_contingency_tables(drug, reac, n_total_cases=100, level="soc")
        assert "reaction_term" in ct.columns
        assert "Gastrointestinal disorders" in ct["reaction_term"].values

    def test_multiple_drugs(self):
        drug = make_drug_ps([
            {"primaryid": "1", "glp1_active_ingredient": "semaglutide"},
            {"primaryid": "2", "glp1_active_ingredient": "tirzepatide"},
        ])
        reac = make_reac_soc([
            {"primaryid": "1", "pt": "Nausea", "soc_name": "GI"},
            {"primaryid": "2", "pt": "Nausea", "soc_name": "GI"},
        ])
        ct = build_contingency_tables(drug, reac, n_total_cases=1000, level="pt")
        drugs_in_ct = set(ct["drug"].unique())
        assert "semaglutide" in drugs_in_ct
        assert "tirzepatide" in drugs_in_ct

    def test_cells_non_negative(self):
        """b, c, d must never be negative."""
        drug = make_drug_ps([
            {"primaryid": str(i), "glp1_active_ingredient": "semaglutide"}
            for i in range(1, 20)
        ])
        reac = make_reac_soc([
            {"primaryid": str(i), "pt": "Nausea", "soc_name": "GI"}
            for i in range(1, 25)
        ])
        ct = build_contingency_tables(drug, reac, n_total_cases=500, level="pt")
        for col in ["a", "b", "c", "d"]:
            assert (ct[col] >= 0).all(), f"Column {col} has negative values"


# ── compute_signals ───────────────────────────────────────────────────────────

class TestComputeSignals:

    def _strong_signal_ct(self):
        return pd.DataFrame([{
            "drug": "semaglutide", "reaction_term": "Pancreatitis",
            "a": 150, "b": 45000, "c": 2000, "d": 8000000, "N": 8047150,
        }])

    def _null_signal_ct(self):
        return pd.DataFrame([{
            "drug": "semaglutide", "reaction_term": "Headache",
            "a": 10, "b": 1000, "c": 100, "d": 10000, "N": 11110,
        }])

    def _low_count_ct(self):
        return pd.DataFrame([{
            "drug": "semaglutide", "reaction_term": "RarePT",
            "a": 2, "b": 100, "c": 50, "d": 5000, "N": 5152,
        }])

    def test_strong_signal_all_three_methods(self):
        result = compute_signals(self._strong_signal_ct())
        assert result.iloc[0]["signal_ror"] == True
        assert result.iloc[0]["signal_prr"] == True
        assert result.iloc[0]["signal_ic"]  == True
        assert result.iloc[0]["is_signal"]  == True
        assert result.iloc[0]["n_signals"]  == 3

    def test_null_signal_no_methods(self):
        result = compute_signals(self._null_signal_ct())
        row = result.iloc[0]
        # Near-null PRR and ROR should not trigger
        assert row["n_signals"] <= 1

    def test_below_min_cases_no_signal(self):
        result = compute_signals(self._low_count_ct(), min_cases=3)
        # a=2, below min_cases=3 → no signal regardless of ratios
        assert result.iloc[0]["is_signal"] == False
        assert pd.isna(result.iloc[0]["ror"])

    def test_required_columns_present(self):
        result = compute_signals(self._strong_signal_ct())
        for col in ["ror", "ror_lb", "ror_ub", "prr", "prr_chi2",
                    "ic", "ic025", "signal_ror", "signal_prr",
                    "signal_ic", "n_signals", "is_signal"]:
            assert col in result.columns

    def test_n_signals_sum_matches_flags(self):
        result = compute_signals(self._strong_signal_ct())
        row = result.iloc[0]
        manual = int(row["signal_ror"]) + int(row["signal_prr"]) + int(row["signal_ic"])
        assert row["n_signals"] == manual

    def test_multiple_rows_independent(self):
        ct = pd.concat([self._strong_signal_ct(), self._null_signal_ct()], ignore_index=True)
        result = compute_signals(ct)
        assert result.iloc[0]["is_signal"] == True
        assert result.iloc[1]["n_signals"] <= 1


# ── time_to_onset ─────────────────────────────────────────────────────────────

class TestTimeToOnset:

    def _make_inputs(self):
        drug_ps = pd.DataFrame({
            "primaryid"             : ["1", "2", "3"],
            "drug_seq"              : ["1", "1", "1"],
            "glp1_active_ingredient": ["semaglutide"] * 3,
        })
        ther = pd.DataFrame({
            "primaryid": ["1",        "2",        "3"],
            "drug_seq" : ["1",        "1",        "1"],
            "start_dt" : ["20230101", "20230601", "20230101"],
        })
        demo_glp1 = pd.DataFrame({
            "primaryid": ["1",        "2",        "3"],
            "event_dt" : ["20230201", "20230701", "20221201"],  # case 3: before start
        })
        return drug_ps, ther, demo_glp1

    def test_returns_dataframe(self):
        drug_ps, ther, demo_glp1 = self._make_inputs()
        result = time_to_onset(drug_ps, ther, demo_glp1)
        assert isinstance(result, pd.DataFrame)

    def test_tto_days_correct(self):
        drug_ps, ther, demo_glp1 = self._make_inputs()
        result = time_to_onset(drug_ps, ther, demo_glp1)
        # Case 1: 20230101 → 20230201 = 31 days
        case1 = result[result["primaryid"] == 1.0]
        assert len(case1) == 1
        assert case1.iloc[0]["tto_days"] == 31

    def test_negative_tto_excluded(self):
        """Case 3 has event before drug start → must be excluded."""
        drug_ps, ther, demo_glp1 = self._make_inputs()
        result = time_to_onset(drug_ps, ther, demo_glp1)
        assert 3.0 not in result["primaryid"].values

    def test_empty_ther_returns_empty(self):
        drug_ps, _, demo_glp1 = self._make_inputs()
        result = time_to_onset(drug_ps, pd.DataFrame(), demo_glp1)
        assert result.empty

    def test_drug_column_present(self):
        drug_ps, ther, demo_glp1 = self._make_inputs()
        result = time_to_onset(drug_ps, ther, demo_glp1)
        assert "drug" in result.columns


class TestTtoSummary:

    def test_returns_dataframe(self):
        tto = pd.DataFrame({
            "drug"    : ["semaglutide"] * 10,
            "primaryid": range(10),
            "tto_days": [5, 10, 15, 20, 25, 30, 40, 50, 60, 100],
        })
        result = tto_summary(tto)
        assert isinstance(result, pd.DataFrame)
        assert "median_days" in result.columns

    def test_pct_within_30d_correct(self):
        tto = pd.DataFrame({
            "drug"     : ["semaglutide"] * 4,
            "primaryid": range(4),
            "tto_days" : [10, 20, 40, 60],
        })
        result = tto_summary(tto)
        # 2 of 4 within 30 days = 50%
        assert result.iloc[0]["pct_within_30d"] == 50.0

    def test_empty_input_returns_empty(self):
        result = tto_summary(pd.DataFrame())
        assert result.empty


# ── run_signal_detection integration ─────────────────────────────────────────

class TestRunSignalDetection:

    def _make_full_inputs(self):
        # 150 semaglutide + 50 tirzepatide cases
        drug_ps = pd.DataFrame({
            "primaryid"             : [str(i) for i in range(1, 201)],
            "drug_seq"              : ["1"] * 200,
            "glp1_active_ingredient": ["semaglutide"] * 150 + ["tirzepatide"] * 50,
            "role_cod"              : ["PS"] * 200,
        })
        # GLP-1 REAC:
        # - Cases 1-120: semaglutide + nausea (the signal)
        # - Cases 121-150: semaglutide + fatigue only (no nausea) → b = 30
        # - Cases 151-200: tirzepatide + dizziness
        glp1_nausea = pd.DataFrame({
            "primaryid" : [str(i) for i in range(1, 121)],       # 120 sema with nausea
            "pt"        : ["Nausea"] * 120,
            "soc_name"  : ["Gastrointestinal disorders"] * 120,
            "meddra_src": ["bundled"] * 120,
        })
        glp1_fatigue = pd.DataFrame({
            "primaryid" : [str(i) for i in range(121, 151)],     # 30 sema WITHOUT nausea
            "pt"        : ["Fatigue"] * 30,
            "soc_name"  : ["General disorders and administration site conditions"] * 30,
            "meddra_src": ["bundled"] * 30,
        })
        tirz_dizziness = pd.DataFrame({
            "primaryid" : [str(i) for i in range(151, 201)],     # 50 tirz cases
            "pt"        : ["Dizziness"] * 50,
            "soc_name"  : ["Nervous system disorders"] * 50,
            "meddra_src": ["bundled"] * 50,
        })
        reac_glp1 = pd.concat(
            [glp1_nausea, glp1_fatigue, tirz_dizziness], ignore_index=True
        )
        # Full REAC: add 2000 background (non-GLP-1) nausea cases → c = 2000
        background_nausea = pd.DataFrame({
            "primaryid": [str(i) for i in range(201, 2201)],
            "pt"       : ["Nausea"] * 2000,
            "soc_name" : ["Gastrointestinal disorders"] * 2000,
        })
        full_reac = pd.concat([reac_glp1, background_nausea], ignore_index=True)

        demo_all = pd.DataFrame({"primaryid": [str(i) for i in range(1, 10001)]})
        return drug_ps, reac_glp1, demo_all, full_reac

    def test_returns_expected_keys(self):
        drug_ps, reac, demo_all, full_reac = self._make_full_inputs()
        results = run_signal_detection(drug_ps, reac, demo_all, full_reac=full_reac)
        assert "signals_pt"  in results
        assert "signals_soc" in results
        assert "audit"       in results

    def test_audit_n_total_correct(self):
        drug_ps, reac, demo_all, full_reac = self._make_full_inputs()
        results = run_signal_detection(drug_ps, reac, demo_all, full_reac=full_reac)
        assert results["audit"]["n_total_cases"] == 10000

    def test_semaglutide_nausea_signal_detected(self):
        """With 150 semaglutide cases all having Nausea vs 2150 total nausea → strong signal."""
        drug_ps, reac, demo_all, full_reac = self._make_full_inputs()
        results = run_signal_detection(drug_ps, reac, demo_all, full_reac=full_reac)
        spt     = results["signals_pt"]
        sem_nausea = spt[
            (spt["drug"] == "semaglutide") & (spt["reaction_term"] == "Nausea")
        ]
        assert len(sem_nausea) == 1
        assert sem_nausea.iloc[0]["is_signal"] == True

    def test_signal_output_has_required_columns(self):
        drug_ps, reac, demo_all, full_reac = self._make_full_inputs()
        results = run_signal_detection(drug_ps, reac, demo_all, full_reac=full_reac)
        for col in ["drug", "reaction_term", "a", "b", "c", "d", "N",
                    "ror", "ror_lb", "prr", "ic", "is_signal"]:
            assert col in results["signals_pt"].columns
