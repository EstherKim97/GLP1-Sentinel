"""
test_meddra.py
--------------
Tests for Phase 3 MedDRA SOC hierarchy join.

Tests cover:
  - Bundled PT→SOC mapping hits correctly
  - Unmapped PTs get 'unmapped' status
  - Audit counts are correct
  - SOC summary aggregates by unique case
  - mdhier.asc loader works on synthetic file
  - join_meddra columns are all present
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from faers_pipeline.meddra import (
    join_meddra,
    soc_summary,
    MEDDRA_SOCS,
    SOC_NAME_TO_CODE,
    load_mdhier,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_reac(rows: list[dict]) -> pd.DataFrame:
    defaults = {"primaryid": "10010001", "caseid": "1001001", "drug_rec_act": ""}
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ── MEDDRA_SOCS structure ─────────────────────────────────────────────────────

class TestSocRegistry:

    def test_27_socs_present(self):
        assert len(MEDDRA_SOCS) == 27

    def test_gi_soc_present(self):
        assert "10017947" in MEDDRA_SOCS
        assert "Gastrointestinal" in MEDDRA_SOCS["10017947"]

    def test_nervous_system_soc_present(self):
        assert "10029999" in MEDDRA_SOCS

    def test_reverse_lookup_works(self):
        assert SOC_NAME_TO_CODE.get("Gastrointestinal disorders") == "10017947"


# ── join_meddra with bundled mapping ─────────────────────────────────────────

class TestJoinMeddraBundled:

    def test_adds_required_columns(self):
        df   = make_reac([{"pt": "Nausea"}])
        result, _ = join_meddra(df)
        for col in ["soc_code", "soc_name", "hlt_name", "hlgt_name", "meddra_src"]:
            assert col in result.columns

    def test_nausea_maps_to_gi(self):
        df     = make_reac([{"pt": "Nausea"}])
        result, _ = join_meddra(df)
        assert result.iloc[0]["soc_name"] == "Gastrointestinal disorders"
        assert result.iloc[0]["meddra_src"] == "bundled"

    def test_dizziness_maps_to_nervous_system(self):
        df     = make_reac([{"pt": "Dizziness"}])
        result, _ = join_meddra(df)
        assert result.iloc[0]["soc_code"] == "10029999"

    def test_case_insensitive_matching(self):
        # FAERS stores PT names in various cases
        df     = make_reac([{"pt": "NAUSEA"}, {"pt": "nausea"}, {"pt": "Nausea"}])
        result, _ = join_meddra(df)
        assert (result["soc_name"] == "Gastrointestinal disorders").all()

    def test_unknown_pt_is_unmapped(self):
        df     = make_reac([{"pt": "Completely Unknown Reaction XYZ123"}])
        result, _ = join_meddra(df)
        assert result.iloc[0]["meddra_src"] == "unmapped"
        assert result.iloc[0]["soc_name"] is None

    def test_mixed_mapped_unmapped(self):
        df = make_reac([
            {"pt": "Nausea"},
            {"pt": "Pancreatitis"},
            {"pt": "Unknown PT that does not exist"},
        ])
        result, audit = join_meddra(df)
        assert audit["mapped_via_bundled"] == 2
        assert audit["unmapped"] == 1
        assert result[result["meddra_src"] == "bundled"].shape[0] == 2

    def test_original_rows_preserved(self):
        df     = make_reac([{"pt": "Nausea"}, {"pt": "Vomiting"}, {"pt": "Dizziness"}])
        result, _ = join_meddra(df)
        assert len(result) == 3

    def test_pt_column_unchanged(self):
        df     = make_reac([{"pt": "Nausea"}])
        result, _ = join_meddra(df)
        assert result.iloc[0]["pt"] == "Nausea"

    def test_working_column_removed(self):
        df     = make_reac([{"pt": "Nausea"}])
        result, _ = join_meddra(df)
        assert "_pt_lower" not in result.columns

    def test_glp1_specific_pts_covered(self):
        """Key PTs from published GLP-1 FAERS studies must be in bundled map."""
        key_pts = [
            "Nausea", "Vomiting", "Diarrhoea", "Pancreatitis",
            "Dizziness", "Hypoglycaemia", "Pulmonary aspiration",
            "Acute kidney injury", "Diabetic retinopathy",
        ]
        for pt in key_pts:
            df     = make_reac([{"pt": pt}])
            result, _ = join_meddra(df)
            assert result.iloc[0]["meddra_src"] == "bundled", \
                f"Expected '{pt}' to be in bundled map but got 'unmapped'"

    def test_diarrhea_us_spelling_covered(self):
        """Both UK (diarrhoea) and US (diarrhea) spellings must map."""
        for spelling in ["Diarrhoea", "Diarrhea"]:
            df     = make_reac([{"pt": spelling}])
            result, _ = join_meddra(df)
            assert result.iloc[0]["meddra_src"] == "bundled", \
                f"Spelling '{spelling}' not covered"


# ── Audit record correctness ──────────────────────────────────────────────────

class TestMeddraAudit:

    def test_audit_fields_present(self):
        df = make_reac([{"pt": "Nausea"}])
        _, audit = join_meddra(df)
        assert "total_reac_rows"    in audit
        assert "mapped_via_bundled" in audit
        assert "unmapped"           in audit
        assert "mapping_rate"       in audit
        assert "top_unmapped_pts"   in audit

    def test_audit_math(self):
        df = make_reac([
            {"pt": "Nausea"},
            {"pt": "Unknown ABC"},
            {"pt": "Vomiting"},
        ])
        _, audit = join_meddra(df)
        assert audit["total_reac_rows"] == 3
        assert audit["mapped_via_bundled"] + audit["unmapped"] == 3

    def test_mapping_rate_all_known(self):
        df = make_reac([{"pt": "Nausea"}, {"pt": "Vomiting"}, {"pt": "Dizziness"}])
        _, audit = join_meddra(df)
        assert audit["mapping_rate"] == 1.0

    def test_mapping_rate_none_known(self):
        df = make_reac([{"pt": "Unknown A"}, {"pt": "Unknown B"}])
        _, audit = join_meddra(df)
        assert audit["mapping_rate"] == 0.0

    def test_top_unmapped_pts_captured(self):
        df = make_reac([
            {"pt": "Unknown Rare PT"},
            {"pt": "Unknown Rare PT"},
            {"pt": "Another Unknown"},
        ])
        _, audit = join_meddra(df)
        assert "Unknown Rare PT" in audit["top_unmapped_pts"]
        assert audit["top_unmapped_pts"]["Unknown Rare PT"] == 2


# ── soc_summary ───────────────────────────────────────────────────────────────

class TestSocSummary:

    def _annotated_reac(self):
        df = make_reac([
            {"primaryid": "1001", "pt": "Nausea"},
            {"primaryid": "1001", "pt": "Vomiting"},    # same case, different PT
            {"primaryid": "1002", "pt": "Nausea"},      # different case, same PT
            {"primaryid": "1003", "pt": "Dizziness"},
        ])
        result, _ = join_meddra(df)
        return result

    def test_returns_dataframe(self):
        result = soc_summary(self._annotated_reac())
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        result = soc_summary(self._annotated_reac())
        assert "soc_name"      in result.columns
        assert "unique_cases"  in result.columns

    def test_counts_unique_cases_not_rows(self):
        """Cases 1001 and 1002 both have Nausea → GI should count 2, not 3."""
        result = soc_summary(self._annotated_reac())
        gi_row = result[result["soc_name"] == "Gastrointestinal disorders"]
        assert len(gi_row) == 1
        # primaryids 1001 and 1002 both have GI reactions
        assert gi_row.iloc[0]["unique_cases"] == 2

    def test_sorted_descending(self):
        result = soc_summary(self._annotated_reac())
        counts = result["unique_cases"].tolist()
        assert counts == sorted(counts, reverse=True)


# ── load_mdhier with synthetic file ──────────────────────────────────────────

class TestLoadMdhier:

    def _synthetic_mdhier(self, tmp_path) -> Path:
        """Build a minimal mdhier.asc-format file."""
        content = (
            "10009166$10001316$Agranulocytosis$10001316$Blood cell disorders$"
            "10001316$Blood and lymphatic system disorders$Y\n"
            "10028813$10029999$Nausea NEC$10028813$Nausea and vomiting symptoms$"
            "10017947$Gastrointestinal disorders$Y\n"
            "10013573$10029999$Dizziness NEC$10013573$Dizziness and giddiness$"
            "10029999$Nervous system disorders$Y\n"
        )
        path = tmp_path / "mdhier.asc"
        path.write_text(content, encoding="iso-8859-1")
        return path

    def test_load_returns_dataframe(self, tmp_path):
        path   = self._synthetic_mdhier(tmp_path)
        result = load_mdhier(path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_primary_soc_column_present(self, tmp_path):
        path   = self._synthetic_mdhier(tmp_path)
        result = load_mdhier(path)
        assert "soc_name" in result.columns

    def test_nonexistent_path_returns_empty(self, tmp_path):
        result = load_mdhier(tmp_path / "does_not_exist.asc")
        assert result.empty
