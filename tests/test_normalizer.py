"""
test_normalizer.py
------------------
Tests for the three-tier GLP-1 drug name normalization logic.

Tests are organized to mirror the three-tier strategy:
  - Tier 1: prod_ai exact match
  - Tier 2: drugname exact match after cleaning
  - Tier 3: prefix/substring match
  - Filter logic: PS only, combo exclusion, compounded handling
  - Edge cases: empty strings, None, garbage input
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from faers_pipeline.normalizer import (
    normalize_drugname,
    lookup_drug,
    normalize_drug_file,
    filter_to_glp1_ps,
    build_normalization_audit,
)


# ── normalize_drugname ────────────────────────────────────────────────────────

class TestNormalizeDrugname:

    def test_uppercase(self):
        assert normalize_drugname("ozempic") == "OZEMPIC"

    def test_strips_whitespace(self):
        assert normalize_drugname("  OZEMPIC  ") == "OZEMPIC"

    def test_strips_dose_mg(self):
        assert normalize_drugname("OZEMPIC 0.5MG") == "OZEMPIC"

    def test_strips_dose_with_space(self):
        assert normalize_drugname("MOUNJARO 5 MG") == "MOUNJARO"

    def test_strips_dose_per_ml(self):
        assert normalize_drugname("OZEMPIC 2MG/1.5ML") == "OZEMPIC"

    def test_strips_trailing_pen(self):
        assert normalize_drugname("OZEMPIC PEN") == "OZEMPIC"

    def test_strips_trailing_injection(self):
        assert normalize_drugname("SEMAGLUTIDE INJECTION") == "SEMAGLUTIDE"

    def test_strips_trailing_subcutaneous(self):
        assert normalize_drugname("LIRAGLUTIDE SUBCUTANEOUS") == "LIRAGLUTIDE"

    def test_strips_punctuation(self):
        assert normalize_drugname("OZEMPIC.") == "OZEMPIC"

    def test_handles_empty_string(self):
        assert normalize_drugname("") == ""

    def test_handles_none(self):
        assert normalize_drugname(None) == ""

    def test_preserves_hyphen_in_name(self):
        # Hyphenated drug names should not be split
        result = normalize_drugname("BYDUREON BCISE")
        assert "BYDUREON" in result

    def test_complex_name(self):
        result = normalize_drugname("OZEMPIC 1MG/0.75ML PREFILLED PEN")
        assert result == "OZEMPIC"


# ── lookup_drug — Tier 1 (prod_ai) ───────────────────────────────────────────

class TestTier1ProdAi:

    def test_prod_ai_semaglutide_hits(self):
        match, tier, source = lookup_drug("SOME DRUG NAME", "SEMAGLUTIDE")
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert tier == 1
        assert source == "prod_ai"

    def test_prod_ai_tirzepatide_hits(self):
        match, tier, source = lookup_drug("ANYTHING", "TIRZEPATIDE")
        assert match.active_ingredient == "tirzepatide"
        assert tier == 1

    def test_prod_ai_lowercase_hits(self):
        # prod_ai is uppercased internally — lowercase input still works
        match, tier, source = lookup_drug("OZEMPIC", "semaglutide")
        assert match is not None
        assert tier == 1

    def test_prod_ai_takes_priority_over_drugname(self):
        # Even if drugname doesn't match, prod_ai should succeed
        match, tier, source = lookup_drug("GARBAGE DRUG NAME 999", "LIRAGLUTIDE")
        assert match is not None
        assert match.active_ingredient == "liraglutide"
        assert tier == 1

    def test_non_glp1_prod_ai_falls_through(self):
        # METFORMIN in prod_ai should not match — falls to tier 2/3
        match, tier, source = lookup_drug("OZEMPIC", "METFORMIN")
        # drugname OZEMPIC should be caught by tier 2
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert tier == 2


# ── lookup_drug — Tier 2 (drugname exact) ────────────────────────────────────

class TestTier2DrugNameExact:

    def test_exact_brand_name_hits(self):
        match, tier, source = lookup_drug("OZEMPIC", "")
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert tier == 2
        assert source == "drugname"

    def test_exact_brand_lowercase_hits(self):
        match, tier, source = lookup_drug("ozempic", "")
        assert match is not None
        assert tier == 2

    def test_brand_with_dose_stripped_hits(self):
        # After normalize_drugname("WEGOVY 2.4MG") → "WEGOVY" → exact match
        match, tier, source = lookup_drug("WEGOVY 2.4MG", "")
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert tier == 2

    def test_generic_name_hits(self):
        match, tier, source = lookup_drug("SEMAGLUTIDE", "")
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert tier == 2

    def test_misspelling_in_reference_hits(self):
        # "SEMAGLUTDIE" is documented in FAERS and in our reference
        match, tier, source = lookup_drug("SEMAGLUTDIE", "")
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert tier == 2

    def test_compounded_exact_hits(self):
        match, tier, source = lookup_drug("COMPOUNDED SEMAGLUTIDE", "")
        assert match is not None
        assert match.is_compounded
        assert tier == 2

    def test_non_glp1_returns_none(self):
        match, tier, source = lookup_drug("ASPIRIN", "ASPIRIN")
        assert match is None
        assert tier == 0


# ── lookup_drug — Tier 3 (prefix match) ──────────────────────────────────────

class TestTier3Prefix:

    def test_brand_with_long_suffix_hits(self):
        # "OZEMPIC PEN DEVICE WITH NEEDLE" — not in our exact map
        # but "OZEMPIC" is a prefix → tier 3 hits
        match, tier, source = lookup_drug("OZEMPIC PEN DEVICE WITH NEEDLE", "")
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert tier == 3

    def test_mounjaro_with_color_suffix(self):
        # Reporters sometimes add pen color: "MOUNJARO GREEN PEN"
        match, tier, source = lookup_drug("MOUNJARO GREEN PEN", "")
        assert match is not None
        assert match.active_ingredient == "tirzepatide"
        assert tier == 3

    def test_short_string_does_not_match(self):
        # "GLP" is too short (<6 chars min)
        match, tier, source = lookup_drug("GLP", "")
        assert match is None

    def test_ambiguous_prefix_takes_longest(self):
        # "BYDUREON BCISE" vs "BYDUREON" — both in map, BYDUREON BCISE is longer
        match, tier, _ = lookup_drug("BYDUREON BCISE AUTOINJECTOR", "")
        assert match is not None
        assert match.active_ingredient == "exenatide"


# ── normalize_drug_file ───────────────────────────────────────────────────────

def make_drug_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DRUG DataFrame for testing."""
    defaults = {
        "primaryid": "10010001",
        "caseid"   : "1001001",
        "drug_seq" : "1",
        "role_cod" : "PS",
        "drugname" : "",
        "prod_ai"  : "",
        "route"    : "SC",
        "_quarter" : "2024Q3",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestNormalizeDrugFile:

    def test_adds_all_annotation_columns(self):
        df = make_drug_df([{"drugname": "OZEMPIC", "prod_ai": "SEMAGLUTIDE"}])
        result = normalize_drug_file(df, "TEST")
        expected_cols = {
            "glp1_active_ingredient", "glp1_match_tier", "glp1_match_source",
            "is_glp1", "is_primary_suspect", "is_compounded",
            "is_combo_product", "is_withdrawn", "drugname_normalized",
        }
        assert expected_cols <= set(result.columns)

    def test_glp1_drug_flagged(self):
        df = make_drug_df([{"drugname": "OZEMPIC", "prod_ai": "SEMAGLUTIDE"}])
        result = normalize_drug_file(df)
        assert result.iloc[0]["is_glp1"] == True
        assert result.iloc[0]["glp1_active_ingredient"] == "semaglutide"

    def test_non_glp1_not_flagged(self):
        df = make_drug_df([{"drugname": "ASPIRIN", "prod_ai": "ASPIRIN"}])
        result = normalize_drug_file(df)
        assert result.iloc[0]["is_glp1"] == False
        assert result.iloc[0]["glp1_active_ingredient"] is None

    def test_ps_role_flagged(self):
        df = make_drug_df([{"role_cod": "PS", "drugname": "OZEMPIC"}])
        result = normalize_drug_file(df)
        assert result.iloc[0]["is_primary_suspect"] == True

    def test_concomitant_role_not_flagged_as_ps(self):
        df = make_drug_df([{"role_cod": "C", "drugname": "OZEMPIC"}])
        result = normalize_drug_file(df)
        assert result.iloc[0]["is_glp1"] == True        # still GLP-1
        assert result.iloc[0]["is_primary_suspect"] == False  # but not PS

    def test_compounded_flagged(self):
        df = make_drug_df([{"drugname": "COMPOUNDED SEMAGLUTIDE", "prod_ai": ""}])
        result = normalize_drug_file(df)
        assert result.iloc[0]["is_compounded"] == True

    def test_combo_product_flagged(self):
        df = make_drug_df([{"drugname": "XULTOPHY", "prod_ai": "LIRAGLUTIDE"}])
        result = normalize_drug_file(df)
        assert result.iloc[0]["is_combo_product"] == True

    def test_original_rows_preserved(self):
        """normalize_drug_file must not drop any rows."""
        df = make_drug_df([
            {"drugname": "OZEMPIC",  "prod_ai": "SEMAGLUTIDE"},
            {"drugname": "ASPIRIN",  "prod_ai": "ASPIRIN"},
            {"drugname": "MOUNJARO", "prod_ai": "TIRZEPATIDE"},
        ])
        result = normalize_drug_file(df)
        assert len(result) == 3   # all rows preserved


# ── filter_to_glp1_ps ────────────────────────────────────────────────────────

class TestFilterToGlp1Ps:

    def _annotated_df(self):
        df = make_drug_df([
            {"drugname": "OZEMPIC",               "prod_ai": "SEMAGLUTIDE", "role_cod": "PS"},
            {"drugname": "OZEMPIC",               "prod_ai": "SEMAGLUTIDE", "role_cod": "C"},
            {"drugname": "ASPIRIN",               "prod_ai": "ASPIRIN",     "role_cod": "PS"},
            {"drugname": "XULTOPHY",              "prod_ai": "LIRAGLUTIDE", "role_cod": "PS"},
            {"drugname": "COMPOUNDED SEMAGLUTIDE","prod_ai": "",            "role_cod": "PS"},
            {"drugname": "TANZEUM",               "prod_ai": "ALBIGLUTIDE", "role_cod": "PS"},
        ])
        return normalize_drug_file(df)

    def test_keeps_glp1_ps_only_by_default(self):
        result = filter_to_glp1_ps(self._annotated_df())
        # Should keep: OZEMPIC PS, COMPOUNDED SEMAGLUTIDE PS, TANZEUM PS
        # Should exclude: XULTOPHY (combo), OZEMPIC C (not PS), ASPIRIN (not GLP1)
        assert all(result["is_glp1"])
        assert all(result["is_primary_suspect"])

    def test_excludes_concomitant_glp1(self):
        result = filter_to_glp1_ps(self._annotated_df())
        # OZEMPIC with role_cod=C should be excluded
        roles = result["role_cod"].unique().tolist()
        assert "C" not in roles

    def test_excludes_non_glp1(self):
        result = filter_to_glp1_ps(self._annotated_df())
        assert "ASPIRIN" not in result["drugname"].values

    def test_excludes_combo_by_default(self):
        result = filter_to_glp1_ps(self._annotated_df())
        assert not result["is_combo_product"].any()

    def test_includes_combo_when_requested(self):
        result = filter_to_glp1_ps(self._annotated_df(), include_combo=True)
        assert result["is_combo_product"].any()

    def test_includes_compounded_by_default(self):
        result = filter_to_glp1_ps(self._annotated_df())
        assert result["is_compounded"].any()

    def test_excludes_compounded_when_requested(self):
        result = filter_to_glp1_ps(self._annotated_df(), include_compounded=False)
        assert not result["is_compounded"].any()

    def test_includes_withdrawn_by_default(self):
        result = filter_to_glp1_ps(self._annotated_df())
        # TANZEUM (albiglutide, withdrawn) should be included
        assert result["is_withdrawn"].any()

    def test_excludes_withdrawn_when_requested(self):
        result = filter_to_glp1_ps(self._annotated_df(), include_withdrawn=False)
        assert not result["is_withdrawn"].any()


# ── build_normalization_audit ─────────────────────────────────────────────────

class TestNormalizationAudit:

    def test_audit_fields_present(self):
        df = make_drug_df([
            {"drugname": "OZEMPIC", "prod_ai": "SEMAGLUTIDE", "role_cod": "PS"},
            {"drugname": "ASPIRIN", "prod_ai": "ASPIRIN",     "role_cod": "PS"},
        ])
        annotated = normalize_drug_file(df)
        audit = build_normalization_audit(annotated)
        assert "total_drug_rows"      in audit
        assert "glp1_identified"      in audit
        assert "primary_suspect_glp1" in audit

    def test_audit_counts_correct(self):
        df = make_drug_df([
            {"drugname": "OZEMPIC", "prod_ai": "SEMAGLUTIDE", "role_cod": "PS"},
            {"drugname": "OZEMPIC", "prod_ai": "SEMAGLUTIDE", "role_cod": "C"},
            {"drugname": "ASPIRIN", "prod_ai": "ASPIRIN",     "role_cod": "PS"},
        ])
        annotated = normalize_drug_file(df)
        audit = build_normalization_audit(annotated)
        assert audit["total_drug_rows"]      == 3
        assert audit["glp1_identified"]      == 2
        assert audit["primary_suspect_glp1"] == 1
