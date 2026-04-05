"""
test_drug_reference.py
----------------------
Tests for the GLP-1 drug reference lookup map.

Verifies:
  - All expected active ingredients are registered
  - The normalization map is built without collisions on important terms
  - DrugMatch flags are set correctly per drug type
  - Withdrawn drug is flagged correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from faers_pipeline.drug_reference import (
    GLP1_REFERENCE,
    NORMALIZATION_MAP,
    GLP1_ACTIVE_INGREDIENTS,
    GLP1_ACTIVE_INGREDIENTS_CURRENT,
    DrugMatch,
)


class TestReferenceStructure:

    def test_all_seven_drugs_present(self):
        expected = {
            "semaglutide", "tirzepatide", "liraglutide",
            "dulaglutide", "exenatide", "lixisenatide", "albiglutide",
        }
        assert expected == set(GLP1_REFERENCE.keys())

    def test_active_ingredients_frozenset(self):
        assert "semaglutide" in GLP1_ACTIVE_INGREDIENTS
        assert "tirzepatide" in GLP1_ACTIVE_INGREDIENTS
        assert "aspirin"     not in GLP1_ACTIVE_INGREDIENTS

    def test_withdrawn_excluded_from_current(self):
        assert "albiglutide" not in GLP1_ACTIVE_INGREDIENTS_CURRENT
        assert "semaglutide" in GLP1_ACTIVE_INGREDIENTS_CURRENT

    def test_normalization_map_is_nonempty(self):
        assert len(NORMALIZATION_MAP) > 50   # we have 66+ terms


class TestBrandNameLookup:

    def test_ozempic_maps_to_semaglutide(self):
        match = NORMALIZATION_MAP.get("OZEMPIC")
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert not match.is_compounded
        assert not match.is_combo

    def test_wegovy_maps_to_semaglutide(self):
        match = NORMALIZATION_MAP.get("WEGOVY")
        assert match.active_ingredient == "semaglutide"

    def test_rybelsus_maps_to_semaglutide(self):
        match = NORMALIZATION_MAP.get("RYBELSUS")
        assert match.active_ingredient == "semaglutide"

    def test_mounjaro_maps_to_tirzepatide(self):
        match = NORMALIZATION_MAP.get("MOUNJARO")
        assert match.active_ingredient == "tirzepatide"

    def test_zepbound_maps_to_tirzepatide(self):
        match = NORMALIZATION_MAP.get("ZEPBOUND")
        assert match.active_ingredient == "tirzepatide"

    def test_victoza_maps_to_liraglutide(self):
        match = NORMALIZATION_MAP.get("VICTOZA")
        assert match.active_ingredient == "liraglutide"

    def test_saxenda_maps_to_liraglutide(self):
        match = NORMALIZATION_MAP.get("SAXENDA")
        assert match.active_ingredient == "liraglutide"

    def test_trulicity_maps_to_dulaglutide(self):
        match = NORMALIZATION_MAP.get("TRULICITY")
        assert match.active_ingredient == "dulaglutide"

    def test_byetta_maps_to_exenatide(self):
        match = NORMALIZATION_MAP.get("BYETTA")
        assert match.active_ingredient == "exenatide"

    def test_bydureon_maps_to_exenatide(self):
        match = NORMALIZATION_MAP.get("BYDUREON")
        assert match.active_ingredient == "exenatide"

    def test_adlyxin_maps_to_lixisenatide(self):
        match = NORMALIZATION_MAP.get("ADLYXIN")
        assert match.active_ingredient == "lixisenatide"

    def test_tanzeum_maps_to_albiglutide_withdrawn(self):
        match = NORMALIZATION_MAP.get("TANZEUM")
        assert match.active_ingredient == "albiglutide"
        assert match.is_withdrawn


class TestCompoundedFlags:

    def test_semaglutide_acetate_is_compounded(self):
        match = NORMALIZATION_MAP.get("SEMAGLUTIDE ACETATE")
        assert match is not None
        assert match.active_ingredient == "semaglutide"
        assert match.is_compounded

    def test_semaglutide_sodium_is_compounded(self):
        match = NORMALIZATION_MAP.get("SEMAGLUTIDE SODIUM")
        assert match.is_compounded

    def test_compounded_tirzepatide_is_compounded(self):
        match = NORMALIZATION_MAP.get("COMPOUNDED TIRZEPATIDE")
        assert match is not None
        assert match.active_ingredient == "tirzepatide"
        assert match.is_compounded

    def test_mounjaro_brand_not_compounded(self):
        match = NORMALIZATION_MAP.get("MOUNJARO")
        assert not match.is_compounded


class TestComboProductFlags:

    def test_xultophy_is_combo(self):
        match = NORMALIZATION_MAP.get("XULTOPHY")
        assert match is not None
        assert match.active_ingredient == "liraglutide"
        assert match.is_combo

    def test_soliqua_is_combo(self):
        match = NORMALIZATION_MAP.get("SOLIQUA")
        assert match is not None
        assert match.active_ingredient == "lixisenatide"
        assert match.is_combo

    def test_ozempic_not_combo(self):
        match = NORMALIZATION_MAP.get("OZEMPIC")
        assert not match.is_combo


class TestNonGlp1:

    def test_aspirin_not_in_map(self):
        assert NORMALIZATION_MAP.get("ASPIRIN") is None

    def test_metformin_not_in_map(self):
        assert NORMALIZATION_MAP.get("METFORMIN") is None

    def test_insulin_not_in_map(self):
        assert NORMALIZATION_MAP.get("INSULIN") is None
