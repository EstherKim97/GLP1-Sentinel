"""
test_schema.py
--------------
Tests for AERS → FAERS schema normalization.

Verifies:
  - is_aers_era() correctly identifies pre-2012Q3 quarters
  - KNOWN_MISSING_QUARTERS contains 2012Q3
  - Column rename maps have the right keys and values
  - parse_file_type applies renames correctly on AERS-era synthetic data
  - parse_file_type leaves FAERS-era data unchanged
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from faers_pipeline.quarters import Quarter
from faers_pipeline.schema import (
    is_aers_era,
    KNOWN_MISSING_QUARTERS,
    COLUMN_RENAMES,
    ENSURE_COLUMNS,
)
from faers_pipeline.parser import parse_file_type


# ── is_aers_era ───────────────────────────────────────────────────────────────

class TestIsAersEra:

    def test_2005q2_is_aers(self):
        assert is_aers_era(Quarter(2005, 2)) is True

    def test_2012q2_is_aers(self):
        # Last AERS quarter
        assert is_aers_era(Quarter(2012, 2)) is True

    def test_2012q3_is_not_aers(self):
        # FAERS launched 2012 Q3 (even though this quarter is missing)
        assert is_aers_era(Quarter(2012, 3)) is False

    def test_2012q4_is_not_aers(self):
        assert is_aers_era(Quarter(2012, 4)) is False

    def test_2024q3_is_not_aers(self):
        assert is_aers_era(Quarter(2024, 3)) is False


# ── KNOWN_MISSING_QUARTERS ────────────────────────────────────────────────────

class TestKnownMissingQuarters:

    def test_2012q3_is_known_missing(self):
        assert Quarter(2012, 3) in KNOWN_MISSING_QUARTERS

    def test_2012q2_not_missing(self):
        assert Quarter(2012, 2) not in KNOWN_MISSING_QUARTERS

    def test_2012q4_not_missing(self):
        assert Quarter(2012, 4) not in KNOWN_MISSING_QUARTERS


# ── COLUMN_RENAMES ────────────────────────────────────────────────────────────

class TestColumnRenames:

    def test_demo_renames_isr_to_primaryid(self):
        assert COLUMN_RENAMES["DEMO"]["isr"] == "primaryid"

    def test_demo_renames_case_to_caseid(self):
        assert COLUMN_RENAMES["DEMO"]["case"] == "caseid"

    def test_demo_renames_gndr_cod_to_sex(self):
        assert COLUMN_RENAMES["DEMO"]["gndr_cod"] == "sex"

    def test_demo_renames_foll_seq_to_caseversion(self):
        assert COLUMN_RENAMES["DEMO"]["foll_seq"] == "caseversion"

    def test_drug_renames_isr_to_primaryid(self):
        assert COLUMN_RENAMES["DRUG"]["isr"] == "primaryid"

    def test_all_file_types_have_rename_map(self):
        for ft in ["DEMO", "DRUG", "REAC", "OUTC", "THER", "RPSR", "INDI"]:
            assert ft in COLUMN_RENAMES


# ── Parser integration: AERS era synthetic data ───────────────────────────────

# Minimal AERS-format DEMO file (uses old column names)
AERS_DEMO = """\
isr\tcase\ti_f_cod\tfoll_seq\timage\tevent_dt\tmfr_dt\tfda_dt\trept_cod\tmfr_num\tmfr_sndr\tage\tage_cod\tgndr_cod\te_sub\twt\twt_cod\trept_dt\toccp_cod\tdeath_dt\tto_mfr\tconfid
10010001\t1001001\tI\t1\tN\t20060601\t20060501\t20060601\tEXP\t\tMANUF01\t55\tYR\tF\tN\t70\tKG\t20060615\tPH\t\tN\tY
10010002\t1001001\tI\t2\tN\t20060601\t20060501\t20060701\tEXP\t\tMANUF01\t55\tYR\tF\tN\t70\tKG\t20060715\tPH\t\tN\tY
10020001\t1002001\tI\t1\tN\t20060901\t20060901\t20061001\tMFR\t\tMANUF02\t42\tYR\tM\tN\t75\tKG\t20061015\tPH\t\tN\tN
"""

AERS_DRUG = """\
isr\tdrug_seq\trole_cod\tdrugname\tval_vbm\troute\tdose_vbm\tcum_dose_chr\tcum_dose_unit\tdechal\trechal\tlot_num\texp_dt\tnda_num\tdose_amt\tdose_unit\tdose_form\tdose_freq
10010001\t1\tPS\tBYETTA\t1\tSC\t5MCG\t\t\tU\tU\t\t\t21773\t5\tMCG\tSOLUTION\tTWICE DAILY
10010002\t1\tPS\tBYETTA\t1\tSC\t5MCG\t\t\tU\tU\t\t\t21773\t5\tMCG\tSOLUTION\tTWICE DAILY
10020001\t1\tPS\tBYETTA\t1\tSC\t10MCG\t\t\tU\tU\t\t\t21773\t10\tMCG\tSOLUTION\tTWICE DAILY
"""

AERS_REAC = """\
isr\tpt
10010001\tNausea
10010002\tNausea
10020001\tPancreatitis
"""


def _make_aers_zip(quarter: Quarter) -> bytes:
    """Build a minimal AERS-format ZIP for testing."""
    suffix = quarter.txt_suffix()
    buf    = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"ASCII/DEMO{suffix}.TXT", AERS_DEMO.encode("iso-8859-1"))
        zf.writestr(f"ASCII/DRUG{suffix}.TXT", AERS_DRUG.encode("iso-8859-1"))
        zf.writestr(f"ASCII/REAC{suffix}.TXT", AERS_REAC.encode("iso-8859-1"))
    return buf.getvalue()


@pytest.fixture
def aers_quarter():
    return Quarter(2006, 3)   # Firmly in AERS era


@pytest.fixture
def aers_zip_path(tmp_path, aers_quarter):
    raw_dir  = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    zip_path = raw_dir / aers_quarter.zip_filename()
    zip_path.write_bytes(_make_aers_zip(aers_quarter))
    return zip_path


class TestAersEraParser:

    def test_aers_demo_primaryid_column_exists(self, aers_zip_path, aers_quarter):
        """After parsing, 'isr' must be renamed to 'primaryid'."""
        df = parse_file_type(aers_zip_path, aers_quarter, "DEMO")
        assert df is not None
        assert "primaryid" in df.columns
        assert "isr" not in df.columns

    def test_aers_demo_caseid_column_exists(self, aers_zip_path, aers_quarter):
        """After parsing, 'case' must be renamed to 'caseid'."""
        df = parse_file_type(aers_zip_path, aers_quarter, "DEMO")
        assert "caseid" in df.columns
        assert "case" not in df.columns

    def test_aers_demo_sex_column_exists(self, aers_zip_path, aers_quarter):
        """After parsing, 'gndr_cod' must be renamed to 'sex'."""
        df = parse_file_type(aers_zip_path, aers_quarter, "DEMO")
        assert "sex" in df.columns
        assert "gndr_cod" not in df.columns

    def test_aers_demo_caseversion_column_exists(self, aers_zip_path, aers_quarter):
        """After parsing, 'foll_seq' must be renamed to 'caseversion'."""
        df = parse_file_type(aers_zip_path, aers_quarter, "DEMO")
        assert "caseversion" in df.columns
        assert "foll_seq" not in df.columns

    def test_aers_demo_values_preserved(self, aers_zip_path, aers_quarter):
        """Rename must not change values, only column names."""
        df = parse_file_type(aers_zip_path, aers_quarter, "DEMO")
        ids = set(df["primaryid"].astype(str).tolist())
        assert "10010001" in ids
        assert "10020001" in ids

    def test_aers_demo_fda_dt_preserved(self, aers_zip_path, aers_quarter):
        df = parse_file_type(aers_zip_path, aers_quarter, "DEMO")
        assert "fda_dt" in df.columns   # fda_dt name same in both eras

    def test_aers_drug_primaryid_renamed(self, aers_zip_path, aers_quarter):
        df = parse_file_type(aers_zip_path, aers_quarter, "DRUG")
        assert "primaryid" in df.columns
        assert "isr" not in df.columns

    def test_aers_drug_prod_ai_added_as_none(self, aers_zip_path, aers_quarter):
        """prod_ai doesn't exist in AERS — must be added as None column."""
        df = parse_file_type(aers_zip_path, aers_quarter, "DRUG")
        assert "prod_ai" in df.columns
        assert df["prod_ai"].isna().all()

    def test_aers_reac_primaryid_renamed(self, aers_zip_path, aers_quarter):
        df = parse_file_type(aers_zip_path, aers_quarter, "REAC")
        assert "primaryid" in df.columns
        assert "isr" not in df.columns

    def test_aers_quarter_survives_deduplication(self, aers_zip_path, aers_quarter):
        """After rename, deduplication must work on AERS data without error."""
        from faers_pipeline.deduplicator import deduplicate_demo
        df = parse_file_type(aers_zip_path, aers_quarter, "DEMO")
        result, audit = deduplicate_demo(df, str(aers_quarter))
        # 3 raw rows, 2 unique cases (1001001 has 2 versions)
        assert audit["n_raw"]   == 3
        assert audit["n_final"] == 2
        assert len(result) == 2


class TestFaersEraUnchanged:
    """FAERS-era data must not have columns renamed (they already have right names)."""

    def test_faers_era_primaryid_unchanged(self, synthetic_zip_path, synthetic_quarter):
        """synthetic_quarter is 2024Q3 — FAERS era. 'primaryid' must already exist."""
        from faers_pipeline.parser import parse_file_type
        df = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        assert "primaryid" in df.columns
        # These AERS names must not appear
        assert "isr"      not in df.columns
        assert "case"     not in df.columns
        assert "gndr_cod" not in df.columns
