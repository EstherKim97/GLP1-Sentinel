"""
drug_reference.py
-----------------
Canonical GLP-1 drug reference: every known brand name, generic variant,
compounded form, and misspelling seen in FAERS maps to a standardized
active ingredient name.

DECISION LOG — Why a static reference file instead of a database or API
------------------------------------------------------------------------
We use a hand-curated Python dictionary rather than pulling from
RxNorm, DrugBank, or openFDA because:

1.  FAERS drug names are *dirtier* than any of those sources expect.
    RxNorm normalizes "OZEMPIC 0.5 MG/0.19 ML SUBCUTANEOUS SOLUTION" —
    FAERS gives you "ozempic pen maybe", "OZEMPICK", "SEMAGLUDITE",
    "COMPOUNDED GLP1". No clean API handles those without a custom layer.

2.  The GLP-1 class is small and stable (7 active ingredients, ~12 brands).
    A curated dictionary is the most transparent, auditable, and
    reproducible approach. Every term in it can be traced to a source.

3.  We can encode domain knowledge that APIs don't have:
    - Salt forms of semaglutide (acetate, sodium) are compounded variants
      — flagged separately, not just mapped to semaglutide.
    - Combination products (Xultophy, Soliqua) contain a GLP-1 but
      also contain insulin — flagged as combo so signal analysis
      can exclude or stratify them appropriately.
    - Withdrawn drugs (albiglutide / Tanzeum) are included because
      their historical reports are still in FAERS and matter for
      longitudinal trend analysis.

Structure
---------
GLP1_REFERENCE: dict mapping active_ingredient → metadata dict
  approved_date:  FDA approval date (YYYY-MM format)
  withdrawn_date: FDA withdrawal date or None
  indications:    list of approved indications
  mechanism:      GLP-1 RA or dual GIP/GLP-1 RA
  brands:         list of known brand names
  generics:       list of known generic name variants (inc. misspellings)
  combo_products: list of combination product names (GLP-1 + something else)
  compounded:     list of compounded form name fragments

NORMALIZATION_MAP: flat dict mapping every known term → active_ingredient
  Built programmatically from GLP1_REFERENCE.
  Used by the normalizer for O(1) lookup.

Sources
-------
- FDA approval letters (drugs@fda.gov)
- Potter et al. 2025: FAERS Essentials, Clin Pharmacol Ther
- Scientific Reports 2025: Neurological AEs of GLP-1 RAs (FAERS 2005–2024)
- Frontiers Pharmacology 2024: Semaglutide route comparison
- FDA warnings on compounded GLP-1s (2024–2025)
- Published FAERS studies listing exact drugname strings found in database
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Active ingredient metadata ────────────────────────────────────────────────

GLP1_REFERENCE: dict[str, dict] = {

    "semaglutide": {
        "approved_date" : "2017-12",
        "withdrawn_date": None,
        "mechanism"     : "GLP-1 RA",
        "indications"   : ["type_2_diabetes", "obesity", "cardiovascular_risk_reduction"],
        "brands": [
            # Subcutaneous injection — diabetes
            "OZEMPIC",
            # Subcutaneous injection — obesity
            "WEGOVY",
            # Oral tablet — diabetes
            "RYBELSUS",
        ],
        "generics": [
            # Exact generic with route/form qualifiers seen in FAERS
            "SEMAGLUTIDE",
            "SEMAGLUTIDE SUBCUTANEOUS",
            "SEMAGLUTIDE INJECTION",
            "SEMAGLUTIDE INJ",
            "SEMAGLUTIDE SC",
            "SEMAGLUTIDE SC INJECTION",
            "SEMAGLUTIDE TABLET",
            "SEMAGLUTIDE ORAL",
            "SEMAGLUTIDE TABLETS",
            # Common misspellings documented in FAERS studies
            "SEMAGLUIDE",
            "SEMAGLUTDIE",
            "SEMALGUTIDE",
            "SEMAGULITDE",
            "SEMAGLUTID",
            "SEMAGLUTIDE.",
        ],
        "combo_products": [
            # Semaglutide does not currently have an approved combo product
            # (Cagrisema is in trials but not approved as of 2024)
        ],
        "compounded": [
            # These are flagged as compounded=True — clinically distinct
            # FDA issued specific warnings about these in 2024-2025
            "COMPOUNDED SEMAGLUTIDE",
            "SEMAGLUTIDE COMPOUND",
            "SEMAGLUTIDE COMPOUNDED",
            "COMPOUNDED OZEMPIC",
            # Salt forms — NOT approved active ingredients
            # FDA warned these are not bioequivalent to semaglutide free base
            "SEMAGLUTIDE ACETATE",
            "SEMAGLUTIDE SODIUM",
            "SEMAGLUTIDE SALT",
        ],
    },

    "tirzepatide": {
        "approved_date" : "2022-05",
        "withdrawn_date": None,
        "mechanism"     : "dual GIP/GLP-1 RA",  # NOT a pure GLP-1 RA
        "indications"   : ["type_2_diabetes", "obesity"],
        "brands": [
            # Subcutaneous injection — diabetes
            "MOUNJARO",
            # Subcutaneous injection — obesity
            "ZEPBOUND",
        ],
        "generics": [
            "TIRZEPATIDE",
            "TIRZEPATIDE SUBCUTANEOUS",
            "TIRZEPATIDE INJECTION",
            "TIRZEPATIDE INJ",
            "TIRZEPATIDE SC",
            # Misspellings
            "TIRZEPATDIE",
            "TIREZEPATIDE",
            "TIRZEPETIDE",
            "TIRZEPATIDE.",
        ],
        "combo_products": [],
        "compounded": [
            "COMPOUNDED TIRZEPATIDE",
            "TIRZEPATIDE COMPOUND",
            "TIRZEPATIDE COMPOUNDED",
            "COMPOUNDED MOUNJARO",
            # Salt forms — same regulatory concern as semaglutide
            "TIRZEPATIDE ACETATE",
            "TIRZEPATIDE SODIUM",
        ],
    },

    "liraglutide": {
        "approved_date" : "2010-01",
        "withdrawn_date": None,
        "mechanism"     : "GLP-1 RA",
        "indications"   : ["type_2_diabetes", "obesity", "cardiovascular_risk_reduction"],
        "brands": [
            # Subcutaneous injection — diabetes
            "VICTOZA",
            # Subcutaneous injection — obesity (higher dose)
            "SAXENDA",
        ],
        "generics": [
            "LIRAGLUTIDE",
            "LIRAGLUTIDE SUBCUTANEOUS",
            "LIRAGLUTIDE INJECTION",
            "LIRAGLUTIDE INJ",
            "LIRAGLUTIDE SC",
            # Misspellings
            "LIRAGLUTDIE",
            "LIRAGLUTID",
            "LIRAGLUIDE",
        ],
        "combo_products": [
            # Liraglutide 3.6mg + insulin degludec 100u/mL
            "XULTOPHY",
            "XULTOPHY 100/3.6",
        ],
        "compounded": [],
    },

    "dulaglutide": {
        "approved_date" : "2014-09",
        "withdrawn_date": None,
        "mechanism"     : "GLP-1 RA",
        "indications"   : ["type_2_diabetes", "cardiovascular_risk_reduction"],
        "brands": [
            "TRULICITY",
        ],
        "generics": [
            "DULAGLUTIDE",
            "DULAGLUTIDE SUBCUTANEOUS",
            "DULAGLUTIDE INJECTION",
            "DULAGLUTIDE INJ",
            "DULAGLUTIDE SC",
            # Misspellings
            "DULAGLUTDIE",
            "DULAGLUIDE",
        ],
        "combo_products": [],
        "compounded": [],
    },

    "exenatide": {
        "approved_date" : "2005-04",
        "withdrawn_date": None,
        "mechanism"     : "GLP-1 RA",
        "indications"   : ["type_2_diabetes"],
        "brands": [
            # Twice-daily injection
            "BYETTA",
            # Weekly extended-release injection
            "BYDUREON",
            "BYDUREON BCISE",
        ],
        "generics": [
            "EXENATIDE",
            "EXENATIDE SUBCUTANEOUS",
            "EXENATIDE INJECTION",
            "EXENATIDE INJ",
            "EXENATIDE SC",
            # Extended-release qualifiers seen in FAERS
            "EXENATIDE EXTENDED RELEASE",
            "EXENATIDE ER",
            "EXENATIDE XR",
            "EXENATIDE MICROSPHERES",
            # Misspellings
            "EXENATDIE",
            "EXENITIDE",
            "EXENATID",
        ],
        "combo_products": [],
        "compounded": [],
    },

    "lixisenatide": {
        "approved_date" : "2016-07",
        "withdrawn_date": None,
        "mechanism"     : "GLP-1 RA",
        "indications"   : ["type_2_diabetes"],
        "brands": [
            "ADLYXIN",
            # EU brand name — appears in FAERS from European reporters
            "LYXUMIA",
        ],
        "generics": [
            "LIXISENATIDE",
            "LIXISENATIDE SUBCUTANEOUS",
            "LIXISENATIDE INJECTION",
            "LIXISENATIDE INJ",
            "LIXISENATIDE SC",
            # Misspellings
            "LIXISENTATIDE",
            "LIXISENATDIE",
        ],
        "combo_products": [
            # Lixisenatide 33mcg + insulin glargine 100u/mL
            "SOLIQUA",
            "SOLIQUA 100/33",
            # EU brand name of the same combo
            "SULIQUA",
        ],
        "compounded": [],
    },

    "albiglutide": {
        "approved_date" : "2014-04",
        "withdrawn_date": "2017-07",  # Withdrawn — commercial reasons, not safety
        "mechanism"     : "GLP-1 RA",
        "indications"   : ["type_2_diabetes"],
        "brands": [
            "TANZEUM",
            # EU brand name
            "EPERZAN",
        ],
        "generics": [
            "ALBIGLUTIDE",
            "ALBIGLUTIDE SUBCUTANEOUS",
            "ALBIGLUTIDE INJECTION",
            "ALBIGLUTIDE INJ",
            # Misspellings
            "ALBIGLUIDE",
            "ALBIGLUTDIE",
        ],
        "combo_products": [],
        "compounded": [],
    },
}


# ── Build the flat normalization map ──────────────────────────────────────────
# Maps every known term (uppercased) → (active_ingredient, is_compounded, is_combo)

@dataclass(frozen=True)
class DrugMatch:
    active_ingredient: str
    is_compounded    : bool = False
    is_combo         : bool = False
    is_withdrawn     : bool = False


def _build_normalization_map() -> dict[str, DrugMatch]:
    """
    Build the flat lookup map from GLP1_REFERENCE.

    Key: uppercased term (all lookups normalize to uppercase first)
    Value: DrugMatch dataclass

    DECISION: uppercase normalization at build time means the lookup
    function only needs one .upper() call — O(1), no regex at lookup time.
    """
    norm_map: dict[str, DrugMatch] = {}

    for active_ingredient, meta in GLP1_REFERENCE.items():
        is_withdrawn = meta.get("withdrawn_date") is not None

        # Brand names → not compounded, not combo
        for term in meta.get("brands", []):
            norm_map[term.upper()] = DrugMatch(
                active_ingredient=active_ingredient,
                is_withdrawn=is_withdrawn,
            )

        # Generic name variants → not compounded, not combo
        for term in meta.get("generics", []):
            norm_map[term.upper()] = DrugMatch(
                active_ingredient=active_ingredient,
                is_withdrawn=is_withdrawn,
            )

        # Combo products → is_combo=True
        for term in meta.get("combo_products", []):
            norm_map[term.upper()] = DrugMatch(
                active_ingredient=active_ingredient,
                is_combo=True,
                is_withdrawn=is_withdrawn,
            )

        # Compounded forms → is_compounded=True
        for term in meta.get("compounded", []):
            norm_map[term.upper()] = DrugMatch(
                active_ingredient=active_ingredient,
                is_compounded=True,
                is_withdrawn=is_withdrawn,
            )

    return norm_map


# Build once at module load — O(1) lookups everywhere else
NORMALIZATION_MAP: dict[str, DrugMatch] = _build_normalization_map()

# Convenience set: all active ingredients in scope
GLP1_ACTIVE_INGREDIENTS: frozenset[str] = frozenset(GLP1_REFERENCE.keys())

# Active ingredients still on the market (not withdrawn)
GLP1_ACTIVE_INGREDIENTS_CURRENT: frozenset[str] = frozenset(
    k for k, v in GLP1_REFERENCE.items()
    if v.get("withdrawn_date") is None
)
