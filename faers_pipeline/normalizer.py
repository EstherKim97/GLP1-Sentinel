"""
normalizer.py
-------------
Normalize FAERS DRUG file records to canonical GLP-1 active ingredients
and filter to primary-suspect GLP-1 reports only.

DECISION LOG — Normalization strategy
--------------------------------------

The problem
  FAERS drug names are free text entered by reporters worldwide.
  The same drug appears as:
    "OZEMPIC", "ozempic", "Ozempic 0.5mg", "OZEMPIC PEN", "OZEMPICK"
    "SEMAGLUTIDE", "SEMAGLUTDIE", "semaglutide sc injection"
    "COMPOUNDED SEMAGLUTIDE", "SEMAGLUTIDE SODIUM"
  No single lookup covers all of these. We use a three-tier strategy.

Three-tier lookup strategy
  Tier 1 — prod_ai (active ingredient field, structured)
    FDA asks reporters to provide the active ingredient separately from
    the brand name. When populated, this field is more reliable than
    drugname. We try prod_ai first.
    Example: drugname="OZEMPIC 0.5MG/DOSE", prod_ai="SEMAGLUTIDE" → Tier 1 hits.

  Tier 2 — exact drugname lookup
    After normalizing drugname (uppercase, strip dose/strength/punctuation),
    look it up in NORMALIZATION_MAP directly.
    Example: drugname="WEGOVY", prod_ai="" → Tier 2 hits.

  Tier 3 — prefix/substring matching
    If exact lookup fails, check whether any known GLP-1 term appears
    as a prefix or substring of the normalized drugname.
    Example: drugname="OZEMPIC PEN 1MG" → strip dose → "OZEMPIC PEN"
             → no exact match → prefix scan → "OZEMPIC" found → hit.
    Example: drugname="SEMAGLUTIDE COMPOUNDED PHARMACY" → prefix →
             "SEMAGLUTIDE" found → hit, flagged as compounded.

  No match → record marked as non-GLP1, excluded from analysis.

  DECISION: We do NOT use fuzzy string matching (Levenshtein distance,
  difflib). Reason: fuzzy matching has an unacceptable false positive
  rate on short drug names. "BYETTA" has edit distance 2 from "BYETAA"
  (a typo) but also distance 2 from "ZYETTA" (a real different drug).
  In pharmacovigilance, a false positive signal is as harmful as a
  missed signal. Prefix matching is more conservative and auditable.

role_cod filter
  FAERS assigns each drug in a report one of four roles:
    PS  Primary Suspect    — reporter believes this drug caused the AE
    SS  Secondary Suspect  — possibly contributed
    C   Concomitant        — patient was taking it but not suspected
    I   Interacting        — suspected drug interaction

  DECISION: We filter to PS only for signal analysis.
  Rationale:
    - PS is what the reporter explicitly flagged as the cause.
    - Including SS, C, I inflates signal counts — a patient on Ozempic
      who also takes metformin generates a DRUG row for metformin too.
      If we included that, metformin would appear to have all the same
      signals as Ozempic, which is meaningless.
    - All published GLP-1 FAERS studies (Scientific Reports 2025,
      Frontiers Pharmacology 2024, PLoS One 2025) use PS only.
    - FDA's own surveillance team uses PS as the primary filter.

  We PRESERVE the SS/C/I rows in the output with a column flag so
  downstream analysis can include them as a sensitivity check.

Compounded product flag
  FDA issued explicit warnings in 2024-2025 that compounded semaglutide
  and tirzepatide AEs are underreported AND may reflect different
  safety profiles (dosing errors, non-bioequivalent salt forms).
  We flag these with is_compounded=True rather than excluding them,
  so they can be analyzed separately.

Output columns added
  glp1_active_ingredient  str|None   canonical name or None (not GLP-1)
  glp1_match_tier         int|None   1, 2, or 3 — which tier matched
  glp1_match_source       str|None   'prod_ai' or 'drugname'
  is_glp1                 bool       True if any tier matched
  is_primary_suspect      bool       True if role_cod == 'PS'
  is_compounded           bool       True if compounded form detected
  is_combo_product        bool       True if combo product (e.g. Xultophy)
  is_withdrawn            bool       True if drug withdrawn from market
  drugname_normalized     str        cleaned drugname used for lookup
"""

import logging
import re
from typing import Optional

import pandas as pd

from .drug_reference import NORMALIZATION_MAP, DrugMatch

logger = logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────

# Dose/strength patterns to strip before lookup
# e.g. "OZEMPIC 0.5MG/DOSE" → "OZEMPIC"
#      "SEMAGLUTIDE 1 MG"    → "SEMAGLUTIDE"
#      "MOUNJARO 2.5MG/0.5ML" → "MOUNJARO"
_DOSE_PATTERN = re.compile(
    r"""
    \s*             # optional leading space
    [\d,.]+         # number (may have comma or decimal)
    \s*             # optional space
    (MG|MCG|UG|ML|MG/ML|MG/DOSE|MG/0\.\d+ML|UNIT|IU|%)  # unit
    [^\s]*          # trailing characters (like /ML, /DOSE, /WEEK)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Trailing qualifiers that don't affect identity
_TRAILING_QUALIFIERS = re.compile(
    r"\s+(PREFILLED|PEN|SYRINGE|AUTOINJECTOR|KIT|SOLUTION|INJECTION|INJECTABLE|"
    r"SUBCUTANEOUS|ORAL|TABLET|TABLETS|POWDER|VIAL|DEVICE|"
    r"EXTENDED.RELEASE|XR|ER|ONCE.WEEKLY|WEEKLY|DAILY|TWICE.DAILY|"
    r"MONTHLY|DOSE|FORM)(\s+(PREFILLED|PEN|SYRINGE|DEVICE|KIT))?\s*$",
    re.IGNORECASE,
)

# Punctuation to strip (except hyphens which can be part of drug names)
_PUNCTUATION = re.compile(r"[^\w\s\-]")


def normalize_drugname(raw: str) -> str:
    """
    Clean a raw drugname string for lookup.

    Steps (applied in order):
      1. Uppercase
      2. Strip leading/trailing whitespace
      3. Remove dose/strength patterns (e.g. "0.5MG")
      4. Remove trailing route/form qualifiers ("PEN", "TABLET")
      5. Remove punctuation (periods, slashes, parenthetical content)
      6. Collapse multiple spaces
      7. Strip again

    DECISION: We strip in this specific order because dose patterns
    often come before qualifiers, and qualifier stripping uses word
    boundary anchors that would fail if dose patterns are still present.
    """
    if not raw or not isinstance(raw, str):
        return ""

    s = raw.upper().strip()
    s = _DOSE_PATTERN.sub("", s)
    s = _TRAILING_QUALIFIERS.sub("", s)
    s = _PUNCTUATION.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ── Three-tier lookup ─────────────────────────────────────────────────────────

def _lookup_exact(term: str) -> Optional[DrugMatch]:
    """Tier 1/2 exact lookup in NORMALIZATION_MAP."""
    return NORMALIZATION_MAP.get(term.upper())


def _lookup_prefix(normalized_name: str) -> Optional[DrugMatch]:
    """
    Tier 3: scan NORMALIZATION_MAP keys to find one that is a prefix
    of normalized_name, or that normalized_name is a prefix of.

    DECISION: We check BOTH directions:
      - key is prefix of name: "OZEMPIC" matches "OZEMPIC PEN 1MG"
      - name is prefix of key: "SEMAGLUTIDE SC" matches "SEMAGLUTIDE SC INJECTION"
    We take the LONGEST matching key to avoid short false matches
    (e.g. "BY" matching "BYETTA" vs "BYDUREON").

    DECISION: Minimum prefix length is 6 characters.
    Reason: Short drug name fragments like "GLUC" or "INSU" could
    match too broadly. GLP-1 drug names are all ≥6 chars.
    """
    if len(normalized_name) < 6:
        return None

    best_match: Optional[DrugMatch] = None
    best_len   : int                = 0

    for key, match in NORMALIZATION_MAP.items():
        if len(key) < 6:
            continue
        # key is a prefix of the name, or name is a prefix of key
        if normalized_name.startswith(key) or key.startswith(normalized_name):
            if len(key) > best_len:
                best_match = match
                best_len   = len(key)

    return best_match


def lookup_drug(
    drugname: str,
    prod_ai : str,
) -> tuple[Optional[DrugMatch], int, str]:
    """
    Three-tier GLP-1 drug lookup.

    Args:
        drugname: Raw drugname field from FAERS DRUG file.
        prod_ai:  Raw prod_ai (active ingredient) field from FAERS DRUG file.

    Returns:
        Tuple of (DrugMatch|None, tier:int, source:str)
          - DrugMatch: the matched drug info, or None if no GLP-1 match
          - tier: 1, 2, or 3 (which tier matched), or 0 if no match
          - source: 'prod_ai', 'drugname', or '' if no match

    DECISION: combo-product check on drugname always runs.
    Reason: prod_ai carries the active ingredient (e.g. "LIRAGLUTIDE"),
    not the product form. If drugname="XULTOPHY" and prod_ai="LIRAGLUTIDE",
    Tier 1 correctly identifies liraglutide but misses the combo flag.
    We therefore check drugname for combo products independently and
    upgrade the match if found.
    """
    normalized = normalize_drugname(drugname)

    # ── Combo product check on drugname (runs before tier logic) ─────────────
    # Combo products like Xultophy must be flagged even when prod_ai matches
    # a plain active ingredient.
    drugname_match = _lookup_exact(normalized) or _lookup_prefix(normalized)
    if drugname_match and drugname_match.is_combo:
        return drugname_match, 2, "drugname"

    # ── Tier 1: prod_ai exact lookup ─────────────────────────────────────────
    if prod_ai:
        prod_ai_clean = prod_ai.upper().strip()
        match = _lookup_exact(prod_ai_clean)
        if match:
            return match, 1, "prod_ai"

    # ── Tier 2: drugname exact lookup ────────────────────────────────────────
    if normalized:
        match = _lookup_exact(normalized)
        if match:
            return match, 2, "drugname"

    # ── Tier 3: prefix/substring match on drugname ───────────────────────────
    if normalized:
        match = _lookup_prefix(normalized)
        if match:
            return match, 3, "drugname"

    return None, 0, ""


# ── Row-level annotation ──────────────────────────────────────────────────────

def annotate_drug_row(row: pd.Series) -> pd.Series:
    """
    Annotate a single DRUG file row with GLP-1 identification columns.
    Used via df.apply() — see normalize_drug_file() for bulk version.
    """
    drugname = str(row.get("drugname", "") or "")
    prod_ai  = str(row.get("prod_ai",  "") or "")
    role_cod = str(row.get("role_cod", "") or "").upper().strip()

    match, tier, source = lookup_drug(drugname, prod_ai)

    return pd.Series({
        "glp1_active_ingredient": match.active_ingredient if match else None,
        "glp1_match_tier"       : tier if match else None,
        "glp1_match_source"     : source if match else None,
        "is_glp1"               : match is not None,
        "is_primary_suspect"    : role_cod == "PS",
        "is_compounded"         : match.is_compounded if match else False,
        "is_combo_product"      : match.is_combo      if match else False,
        "is_withdrawn"          : match.is_withdrawn   if match else False,
        "drugname_normalized"   : normalize_drugname(drugname),
    })


# ── Bulk DataFrame normalization ──────────────────────────────────────────────

def normalize_drug_file(
    drug_df: pd.DataFrame,
    quarter_label: str = "",
) -> pd.DataFrame:
    """
    Normalize an entire DRUG DataFrame.

    Adds GLP-1 annotation columns to every row, then returns
    the full DataFrame with annotations. Does NOT filter — that
    is a separate step so callers control what they keep.

    Args:
        drug_df:       DRUG DataFrame from parser (all drugs, all roles).
        quarter_label: For logging only.

    Returns:
        drug_df with 9 new columns added.

    DECISION: We annotate ALL rows, not just PS rows.
    Reason: The filter step (filter_to_glp1_ps) is separate and
    explicit. Keeping annotation and filtering as distinct operations
    means you can run sensitivity analyses (e.g., include SS drugs)
    by simply changing the filter call, not the normalization call.
    """
    n = len(drug_df)
    logger.info(f"  NORMALIZE {quarter_label}: {n:,} drug rows")

    # ── Vectorized lookup — much faster than row-by-row apply ────────────────
    # We build the annotation columns using vectorized pandas operations
    # rather than df.apply(annotate_drug_row) which is slow for large files.

    # Clean prod_ai and drugname
    prod_ai_upper  = drug_df["prod_ai"].fillna("").astype(str).str.upper().str.strip()
    drugname_clean = drug_df["drugname"].fillna("").astype(str).apply(normalize_drugname)
    role_cod_upper = drug_df["role_cod"].fillna("").astype(str).str.upper().str.strip()

    # Build result lists — iterate once through rows
    # (Pure vectorization isn't possible for the 3-tier lookup logic,
    #  but we minimize per-row overhead by pre-computing the cleaned strings.)
    results = []
    for prod_ai, drugname_norm in zip(prod_ai_upper, drugname_clean):
        match, tier, source = lookup_drug(drugname_norm, prod_ai)
        results.append((
            match.active_ingredient if match else None,
            tier if match else None,
            source if match else None,
            match is not None,
            match.is_compounded if match else False,
            match.is_combo      if match else False,
            match.is_withdrawn  if match else False,
        ))

    (
        active_ingredients,
        tiers,
        sources,
        is_glp1_flags,
        is_compounded_flags,
        is_combo_flags,
        is_withdrawn_flags,
    ) = zip(*results) if results else ([],[],[],[],[],[],[])

    drug_df = drug_df.copy()
    drug_df["glp1_active_ingredient"] = list(active_ingredients)
    drug_df["glp1_match_tier"]        = list(tiers)
    drug_df["glp1_match_source"]      = list(sources)
    drug_df["is_glp1"]                = list(is_glp1_flags)
    drug_df["is_primary_suspect"]     = role_cod_upper == "PS"
    drug_df["is_compounded"]          = list(is_compounded_flags)
    drug_df["is_combo_product"]       = list(is_combo_flags)
    drug_df["is_withdrawn"]           = list(is_withdrawn_flags)
    drug_df["drugname_normalized"]    = list(drugname_clean)

    n_glp1 = drug_df["is_glp1"].sum()
    n_ps   = (drug_df["is_glp1"] & drug_df["is_primary_suspect"]).sum()
    logger.info(
        f"  NORMALIZE {quarter_label}: "
        f"{n_glp1:,} GLP-1 rows found ({n_glp1/n:.1%}), "
        f"{n_ps:,} as primary suspect"
    )

    return drug_df


def filter_to_glp1_ps(
    drug_df: pd.DataFrame,
    include_compounded: bool = True,
    include_combo     : bool = False,
    include_withdrawn : bool = True,
) -> pd.DataFrame:
    """
    Filter annotated DRUG DataFrame to GLP-1 primary-suspect rows only.

    Args:
        drug_df:           Annotated DRUG DataFrame (output of normalize_drug_file).
        include_compounded: Include compounded product reports. Default True —
                            we want them for the compounded vs. brand comparison.
        include_combo:      Include combination products (e.g. Xultophy).
                            Default False — combo products have confounded signals
                            because the other component (insulin) may be the cause.
        include_withdrawn:  Include withdrawn drugs (albiglutide). Default True —
                            historical reports matter for longitudinal analysis.

    Returns:
        Filtered DataFrame.

    DECISION: Default is to include compounded but exclude combos.
    Rationale:
      - Compounded: same active ingredient, different manufacturer/source.
        Signal analysis should know about them. We have is_compounded flag
        to stratify or exclude later.
      - Combo products: different mechanism question. If someone on Xultophy
        (liraglutide + insulin) has hypoglycemia, is it the GLP-1 or the
        insulin? We can't answer that, so we exclude by default.
    """
    mask = (
        drug_df["is_glp1"]
        & drug_df["is_primary_suspect"]
    )

    if not include_compounded:
        mask = mask & ~drug_df["is_compounded"]

    if not include_combo:
        mask = mask & ~drug_df["is_combo_product"]

    if not include_withdrawn:
        mask = mask & ~drug_df["is_withdrawn"]

    return drug_df[mask].copy()


def build_normalization_audit(drug_df: pd.DataFrame) -> dict:
    """
    Produce a normalization audit summary for the decision log.

    Returns dict with counts at each tier, miss rate, compounded count, etc.
    This gets written to the audit log alongside the dedup audit.
    """
    total = len(drug_df)
    glp1  = drug_df["is_glp1"]

    tier_counts = {}
    if "glp1_match_tier" in drug_df.columns:
        vc = drug_df.loc[glp1, "glp1_match_tier"].value_counts()
        tier_counts = {f"tier_{k}": int(v) for k, v in vc.items()}

    return {
        "total_drug_rows"     : total,
        "glp1_identified"     : int(glp1.sum()),
        "glp1_rate"           : round(float(glp1.mean()), 4),
        "primary_suspect_glp1": int((glp1 & drug_df["is_primary_suspect"]).sum()),
        "compounded"          : int(drug_df.get("is_compounded", pd.Series([False]*total)).sum()),
        "combo_products"      : int(drug_df.get("is_combo_product", pd.Series([False]*total)).sum()),
        "withdrawn_drug"      : int(drug_df.get("is_withdrawn", pd.Series([False]*total)).sum()),
        **tier_counts,
    }
