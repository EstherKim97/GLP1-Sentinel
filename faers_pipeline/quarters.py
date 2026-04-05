"""
quarters.py
-----------
Canonical registry of all FAERS quarterly releases to ingest,
starting from 2005 Q2 (exenatide / Byetta approval, the first GLP-1 RA).

DECISION LOG — Start date choice
---------------------------------
We start at 2005 Q2, not 2004 Q1 (earliest FAERS data), because:
  - Exenatide (Byetta) was FDA-approved April 28, 2005 → first full quarter is Q2 2005.
  - Pre-approval quarters contain zero GLP-1 reports by definition.
  - Including them adds download/compute cost with no signal value.
  - Reference: FDA approval letter NDA 21-773, April 2005.

GLP-1 Drug Class Timeline (for context in EDA annotations)
------------------------------------------------------------
  2005 Q2  : Exenatide SC (Byetta) — first GLP-1 RA approved
  2010 Q1  : Liraglutide SC (Victoza) — diabetes
  2014 Q3  : Dulaglutide SC (Trulicity)
  2016 Q3  : Lixisenatide SC (Adlyxin/Soliqua)
  2017 Q4  : Semaglutide SC (Ozempic) — diabetes
  2019 Q3  : Semaglutide oral (Rybelsus)
  2021 Q2  : Semaglutide SC (Wegovy) — obesity indication
  2022 Q2  : Tirzepatide SC (Mounjaro) — first GIP/GLP-1 dual agonist
  2023 Q4  : Tirzepatide SC (Zepbound) — obesity indication

FAERS File Naming Convention
-----------------------------
  Pre-2012:   aers_ascii_YYYYQ[1-4].zip  (legacy AERS system)
  2012 Q3+:   faers_ascii_YYYYQ[1-4].zip (FAERS system, launched Sep 2012)
  2026 Q1+:   aems_ascii_YYYYQ[1-4].zip  (AEMS rebrand, launched Mar 2026)

  Inside each ZIP → ASCII/ folder → 7 pipe-delimited .txt files per quarter.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Quarter:
    year: int
    q: int  # 1-4

    def __str__(self) -> str:
        return f"{self.year}Q{self.q}"

    def label(self) -> str:
        """Human-readable label: '2024 Q3'"""
        return f"{self.year} Q{self.q}"

    def zip_filename(self) -> str:
        """
        Return the correct ZIP filename for this quarter.

        DECISION: Three naming eras:
          - 2004 Q1 – 2012 Q2: aers_ascii_YYYYQ[N].zip  (legacy AERS)
          - 2012 Q3 – 2025 Q4: faers_ascii_YYYYQ[N].zip  (FAERS)
          - 2026 Q1+          : aems_ascii_YYYYQ[N].zip   (AEMS, new as of Mar 2026)
        """
        if (self.year, self.q) < (2012, 3):
            prefix = "aers"
        elif (self.year, self.q) >= (2026, 1):
            prefix = "aems"
        else:
            prefix = "faers"
        return f"{prefix}_ascii_{self.year}q{self.q}.zip"

    def download_url(self) -> str:
        """
        Canonical FDA FIS download URL.
        Base: https://fis.fda.gov/content/Exports/
        """
        return f"https://fis.fda.gov/content/Exports/{self.zip_filename()}"

    def txt_suffix(self) -> str:
        """
        Suffix used in internal .txt filenames within the ZIP.
        E.g. 2024 Q3 → '24Q3'
        Pre-2010 quarters use 2-digit year only.
        """
        return f"{str(self.year)[2:]}Q{self.q}"

    def next(self) -> "Quarter":
        if self.q == 4:
            return Quarter(self.year + 1, 1)
        return Quarter(self.year, self.q + 1)

    def __lt__(self, other: "Quarter") -> bool:
        return (self.year, self.q) < (other.year, other.q)

    def __le__(self, other: "Quarter") -> bool:
        return (self.year, self.q) <= (other.year, other.q)


# ── Milestone annotations for EDA charting ──────────────────────────────────

GLP1_MILESTONES = {
    Quarter(2005, 2): "Exenatide (Byetta) approved",
    Quarter(2010, 1): "Liraglutide (Victoza) approved",
    Quarter(2014, 3): "Dulaglutide (Trulicity) approved",
    Quarter(2016, 3): "Lixisenatide (Adlyxin) approved",
    Quarter(2017, 4): "Semaglutide SC (Ozempic) approved",
    Quarter(2019, 3): "Semaglutide oral (Rybelsus) approved",
    Quarter(2021, 2): "Semaglutide (Wegovy) — obesity approved",
    Quarter(2022, 2): "Tirzepatide (Mounjaro) approved",
    Quarter(2023, 4): "Tirzepatide (Zepbound) — obesity approved",
}


def quarters_in_range(start: Quarter, end: Quarter) -> list[Quarter]:
    """
    Return ordered list of quarters from start (inclusive) to end (inclusive).
    """
    result = []
    current = start
    while current <= end:
        result.append(current)
        current = current.next()
    return result


# ── Project scope constants ──────────────────────────────────────────────────

SCOPE_START = Quarter(2005, 2)   # Exenatide approval — first GLP-1 RA
SCOPE_END   = Quarter(2024, 3)   # As specified in project proposal

ALL_QUARTERS = quarters_in_range(SCOPE_START, SCOPE_END)

# Internal txt file names within each quarter's ZIP
# Each file is: <TYPE><YYQ>.txt  e.g. DEMO24Q3.txt
FILE_TYPES = ["DEMO", "DRUG", "REAC", "OUTC", "THER", "RPSR", "INDI"]
