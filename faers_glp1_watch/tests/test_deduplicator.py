"""
test_deduplicator.py
--------------------
Unit tests for FDA-recommended deduplication logic.

These tests run with zero external dependencies — no FDA data needed.
They verify the exact dedup logic against hand-crafted scenarios that
mirror known FAERS data quality patterns.

Run:
  cd faers_glp1_watch
  python -m pytest tests/ -v
"""

import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from faers_pipeline.deduplicator import deduplicate_demo, filter_related_by_primaryid


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_demo(**kwargs) -> pd.DataFrame:
    """
    Build a minimal DEMO DataFrame from keyword args.
    Required kwargs: primaryid, caseid, fda_dt.
    Extra columns (caseversion, age, sex) are auto-sized to match.
    """
    n = len(kwargs.get("primaryid", ["1001", "1002"]))
    defaults = {
        "primaryid"   : (kwargs.pop("primaryid",   None) or ["1001", "1002"])[:n],
        "caseid"      : (kwargs.pop("caseid",       None) or ["A001", "A002"])[:n],
        "fda_dt"      : (kwargs.pop("fda_dt",       None) or ["20240101"] * n)[:n],
        "caseversion" : ["1"] * n,
        "age"         : ["45"] * n,
        "sex"         : ["M"] * n,
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ── Step 1: Keep latest FDA_DT per CASEID ────────────────────────────────────

class TestStep1LatestFdaDt:

    def test_keeps_record_with_latest_fda_dt(self):
        """Classic case: two versions of the same case, different FDA_DT."""
        df = make_demo(
            primaryid = ["1001", "1002"],
            caseid    = ["A001", "A001"],   # SAME caseid
            fda_dt    = ["20230601", "20240101"],  # 1001 is older
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 1
        assert result.iloc[0]["primaryid"] == "1002"  # newer FDA_DT wins

    def test_keeps_all_when_no_duplicates(self):
        """Distinct CASEIDs — nothing should be removed."""
        df = make_demo(
            primaryid = ["1001", "1002"],
            caseid    = ["A001", "A002"],   # Different caseids
            fda_dt    = ["20240101", "20240101"],
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 2
        assert audit["removed_step1"] == 0
        assert audit["removed_step2"] == 0

    def test_three_versions_keeps_latest(self):
        """Three versions of same case — only latest survives."""
        df = make_demo(
            primaryid = ["1001", "1002", "1003"],
            caseid    = ["A001", "A001", "A001"],
            fda_dt    = ["20220101", "20230601", "20240901"],
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 1
        assert result.iloc[0]["primaryid"] == "1003"
        assert audit["removed_step1"] == 2

    def test_missing_fda_dt_treated_as_oldest(self):
        """Records with missing/null FDA_DT should lose to any dated record."""
        df = make_demo(
            primaryid = ["1001", "1002"],
            caseid    = ["A001", "A001"],
            fda_dt    = [None, "20240101"],
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 1
        assert result.iloc[0]["primaryid"] == "1002"

    def test_mixed_cases_independent(self):
        """Multiple distinct CASEIDs — dedup is per-case, not global."""
        df = make_demo(
            primaryid = ["1001", "1002", "2001", "2002"],
            caseid    = ["A001", "A001", "B001", "B001"],
            fda_dt    = ["20230101", "20240101", "20230601", "20230601"],
            # A001: 1002 wins (newer); B001: tie → step 2
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 2
        primaries = set(result["primaryid"].tolist())
        assert "1002" in primaries  # A001 winner


# ── Step 2: Highest PRIMARYID on tied FDA_DT ─────────────────────────────────

class TestStep2HighestPrimaryId:

    def test_tie_broken_by_higher_primaryid(self):
        """Two records with same CASEID and same FDA_DT — higher PRIMARYID wins."""
        df = make_demo(
            primaryid = ["1001", "9999"],
            caseid    = ["A001", "A001"],
            fda_dt    = ["20240101", "20240101"],  # Same date
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 1
        assert result.iloc[0]["primaryid"] == "9999"
        assert audit["removed_step2"] == 1

    def test_tie_three_records_highest_wins(self):
        df = make_demo(
            primaryid = ["1001", "5000", "9999"],
            caseid    = ["A001", "A001", "A001"],
            fda_dt    = ["20240101", "20240101", "20240101"],
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 1
        assert result.iloc[0]["primaryid"] == "9999"

    def test_no_tie_step2_removes_nothing(self):
        """When Step 1 resolves all dups, Step 2 should remove nothing."""
        df = make_demo(
            primaryid = ["1001", "1002"],
            caseid    = ["A001", "A001"],
            fda_dt    = ["20230101", "20240101"],  # Different → step 1 resolves
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert audit["removed_step2"] == 0


# ── Audit record correctness ──────────────────────────────────────────────────

class TestAuditRecord:

    def test_audit_fields_present(self):
        df = make_demo()
        _, audit = deduplicate_demo(df, "2024Q3")
        required_keys = {
            "quarter", "n_raw", "n_after_step1", "removed_step1",
            "removed_step2", "n_final", "dedup_rate"
        }
        assert required_keys <= set(audit.keys())

    def test_audit_math_consistent(self):
        """n_raw = removed_step1 + removed_step2 + n_final"""
        df = make_demo(
            primaryid = ["1001", "1002", "1003", "2001"],
            caseid    = ["A001", "A001", "A001", "B001"],
            fda_dt    = ["20220101", "20240101", "20240101", "20240101"],
            # A001: 1001 removed step1; 1002/1003 tie → 1003 wins (step2); B001 keeps
        )
        _, audit = deduplicate_demo(df, "TEST")
        assert audit["n_raw"] == 4
        assert audit["removed_step1"] + audit["removed_step2"] + audit["n_final"] == 4

    def test_dedup_rate_zero_when_no_dups(self):
        df = make_demo()  # 2 distinct cases
        _, audit = deduplicate_demo(df, "TEST")
        assert audit["dedup_rate"] == 0.0

    def test_dedup_rate_one_when_all_same_case(self):
        df = make_demo(
            primaryid = ["1001", "1002"],
            caseid    = ["A001", "A001"],
            fda_dt    = ["20230101", "20240101"],
        )
        _, audit = deduplicate_demo(df, "TEST")
        assert audit["n_final"] == 1
        assert audit["dedup_rate"] == 0.5  # 1 of 2 removed


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_row(self):
        df = make_demo(
            primaryid=["1001"], caseid=["A001"], fda_dt=["20240101"]
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 1
        assert audit["dedup_rate"] == 0.0

    def test_fda_dt_with_timestamp(self):
        """Some quarters store FDA_DT as YYYYMMDDHHMMSS — we take first 8 chars."""
        df = make_demo(
            primaryid = ["1001", "1002"],
            caseid    = ["A001", "A001"],
            fda_dt    = ["20240101120000", "20240101090000"],  # Same date, different time
        )
        # Both have same date portion → Step 2 breaks tie
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 1
        assert result.iloc[0]["primaryid"] == "1002"  # higher primaryid

    def test_whitespace_in_fields(self):
        """Whitespace around IDs should not cause duplication misses."""
        df = make_demo(
            primaryid = [" 1001 ", "1001"],
            caseid    = ["A001 ", " A001"],
            fda_dt    = ["20230101", "20240101"],
        )
        result, audit = deduplicate_demo(df, "TEST")
        assert len(result) == 1

    def test_missing_columns_raises_valueerror(self):
        df = pd.DataFrame({"primaryid": ["1"], "caseid": ["A"]})  # missing fda_dt
        with pytest.raises(ValueError, match="missing required columns"):
            deduplicate_demo(df, "TEST")


# ── filter_related_by_primaryid ───────────────────────────────────────────────

class TestFilterRelated:

    def test_filters_orphaned_records(self):
        """Records whose PRIMARYID is not in the surviving set should be removed."""
        df = pd.DataFrame({
            "primaryid": ["1001", "1002", "9999"],
            "pt"       : ["Nausea", "Vomiting", "Dizziness"],
        })
        surviving = {1001, 1002}
        result = filter_related_by_primaryid(df, surviving, "REAC", "TEST")
        assert len(result) == 2
        assert set(result["primaryid"].tolist()) == {"1001", "1002"}

    def test_no_orphans_unchanged(self):
        df = pd.DataFrame({
            "primaryid": ["1001", "1002"],
            "pt"       : ["Nausea", "Vomiting"],
        })
        surviving = {1001, 1002}
        result = filter_related_by_primaryid(df, surviving, "REAC", "TEST")
        assert len(result) == 2

    def test_all_orphaned_returns_empty(self):
        df = pd.DataFrame({
            "primaryid": ["9998", "9999"],
            "pt"       : ["Nausea", "Vomiting"],
        })
        surviving = {1001, 1002}
        result = filter_related_by_primaryid(df, surviving, "REAC", "TEST")
        assert len(result) == 0

    def test_multiple_reac_rows_per_primaryid(self):
        """One case can have multiple reactions — all should be kept."""
        df = pd.DataFrame({
            "primaryid": ["1001", "1001", "1001", "9999"],
            "pt"       : ["Nausea", "Vomiting", "Diarrhoea", "Headache"],
        })
        surviving = {1001}
        result = filter_related_by_primaryid(df, surviving, "REAC", "TEST")
        assert len(result) == 3
