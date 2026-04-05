"""
test_pipeline_integration.py
-----------------------------
End-to-end integration test using a synthetic FAERS ZIP file.

Fixtures are defined in conftest.py — this file contains only assertions.
Requires zero internet access and zero real FDA data.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from faers_pipeline.parser import parse_file_type, parse_quarter
from faers_pipeline.deduplicator import deduplicate_demo, filter_related_by_primaryid
from faers_pipeline.writer import save_parquet, save_audit_log


# ── Parser ────────────────────────────────────────────────────────────────────

class TestParser:

    def test_parse_demo_returns_dataframe(self, synthetic_zip_path, synthetic_quarter):
        df = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        assert df is not None
        assert len(df) == 6

    def test_parse_demo_columns_lowercased(self, synthetic_zip_path, synthetic_quarter):
        df = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        assert "primaryid" in df.columns
        assert "caseid"    in df.columns
        assert "fda_dt"    in df.columns

    def test_parse_adds_quarter_metadata(self, synthetic_zip_path, synthetic_quarter):
        df = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        assert "_quarter" in df.columns
        assert df["_quarter"].iloc[0] == "2024Q3"

    def test_parse_missing_file_returns_none(self, synthetic_zip_path, synthetic_quarter):
        result = parse_file_type(synthetic_zip_path, synthetic_quarter, "THER")
        assert result is None

    def test_parse_drug_returns_correct_rows(self, synthetic_zip_path, synthetic_quarter):
        df = parse_file_type(synthetic_zip_path, synthetic_quarter, "DRUG")
        assert df is not None
        assert len(df) == 7  # 6 GLP-1 rows + 1 aspirin orphan

    def test_parse_quarter_returns_all_available(self, synthetic_zip_path, synthetic_quarter):
        result = parse_quarter(synthetic_zip_path, synthetic_quarter)
        assert result["DEMO"] is not None
        assert result["DRUG"] is not None
        assert result["REAC"] is not None
        assert result["OUTC"] is not None
        assert result["THER"] is None
        assert result["RPSR"] is None
        assert result["INDI"] is None


# ── Deduplication on parsed data ──────────────────────────────────────────────

class TestDedupOnParsedData:

    def test_dedup_removes_older_caseid(self, synthetic_zip_path, synthetic_quarter):
        demo = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        result, _ = deduplicate_demo(demo, "2024Q3")
        surviving = set(result["primaryid"].astype(str).tolist())
        assert "10010002" in surviving
        assert "10010001" not in surviving

    def test_dedup_step2_handles_exact_tie(self, synthetic_zip_path, synthetic_quarter):
        demo = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        result, _ = deduplicate_demo(demo, "2024Q3")
        surviving = set(result["primaryid"].astype(str).tolist())
        assert "10040002" in surviving
        assert "10040001" not in surviving

    def test_dedup_final_case_count(self, synthetic_zip_path, synthetic_quarter):
        demo = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        result, _ = deduplicate_demo(demo, "2024Q3")
        assert len(result) == 4

    def test_related_files_filtered_to_survivors(self, synthetic_zip_path, synthetic_quarter):
        demo = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        drug = parse_file_type(synthetic_zip_path, synthetic_quarter, "DRUG")
        demo_deduped, _ = deduplicate_demo(demo, "2024Q3")
        surviving = set(
            pd.to_numeric(demo_deduped["primaryid"], errors="coerce")
            .dropna().astype(int)
        )
        drug_filtered = filter_related_by_primaryid(drug, surviving, "DRUG", "2024Q3")
        ids = drug_filtered["primaryid"].astype(str).tolist()
        assert "99999001" not in ids
        assert "10010001" not in ids
        assert "10010002" in ids

    def test_audit_math(self, synthetic_zip_path, synthetic_quarter):
        demo = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        _, audit = deduplicate_demo(demo, "2024Q3")
        assert audit["n_raw"]   == 6
        assert audit["n_final"] == 4
        assert audit["n_raw"] == audit["n_final"] + audit["removed_step1"] + audit["removed_step2"]


# ── Writer ────────────────────────────────────────────────────────────────────

class TestWriter:

    def test_save_parquet_creates_versioned_file(
        self, synthetic_zip_path, synthetic_quarter, processed_dir
    ):
        demo = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        demo_deduped, _ = deduplicate_demo(demo, "2024Q3")
        out_path = save_parquet(demo_deduped, "DEMO", processed_dir, synthetic_quarter)
        assert out_path.exists()
        assert out_path.suffix == ".parquet"
        assert "v20240930" in out_path.name

    def test_parquet_roundtrip_preserves_shape(
        self, synthetic_zip_path, synthetic_quarter, processed_dir
    ):
        demo = parse_file_type(synthetic_zip_path, synthetic_quarter, "DEMO")
        demo_deduped, _ = deduplicate_demo(demo, "2024Q3")
        out_path = save_parquet(demo_deduped, "DEMO", processed_dir, synthetic_quarter)
        reloaded = pd.read_parquet(out_path)
        assert len(reloaded)         == len(demo_deduped)
        assert set(reloaded.columns) == set(demo_deduped.columns)

    def test_save_audit_log_creates_both_formats(self, synthetic_quarter, logs_dir):
        audit_records = [{
            "quarter": "2024Q3", "n_raw": 6, "n_after_step1": 5,
            "removed_step1": 1, "removed_step2": 1, "n_final": 4,
            "dedup_rate": 0.333,
        }]
        json_path, csv_path = save_audit_log(audit_records, logs_dir, synthetic_quarter)
        assert json_path.exists()
        assert csv_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["total_quarters"]        == 1
        assert data["records"][0]["quarter"] == "2024Q3"
        df_audit = pd.read_csv(csv_path)
        assert len(df_audit)      == 1
        assert "dedup_rate" in df_audit.columns
