"""
conftest.py
-----------
Shared pytest fixtures available to all test files without import.
"""

import io
import zipfile
from pathlib import Path

import pytest

from faers_pipeline.quarters import Quarter


# ── Synthetic FAERS file content ──────────────────────────────────────────────
# These match the real FAERS ASCII format: $ delimiter, ISO-8859-1 encoding,
# header row, and the exact column names used in production data.

DEMO_ROWS = """\
primaryid$caseid$caseversion$i_f_cod$event_dt$mfr_dt$init_fda_dt$fda_dt$rept_cod$age$age_cod$age_grp$sex$wt$wt_cod$occr_country$reporter_country
10010001$1001001$1$I$20230601$20230501$20230601$20230601$EXP$55$YR$E$F$70$KG$US$US
10010002$1001001$2$I$20230601$20230501$20230701$20230701$EXP$55$YR$E$F$70$KG$US$US
10020001$1002001$1$I$20231001$20231001$20231015$20231015$MFR$42$YR$D$M$75$KG$US$US
10030001$1003001$1$I$20240101$20231201$20240110$20240110$CSR$68$YR$E$F$65$KG$DE$DE
10040001$1004001$1$I$20240201$20240101$20240215$20240215$EXP$35$YR$C$M$80$KG$US$US
10040002$1004001$1$I$20240201$20240101$20240215$20240215$EXP$35$YR$C$M$80$KG$US$US
"""

DRUG_ROWS = """\
primaryid$caseid$drug_seq$role_cod$drugname$prod_ai$val_vbm$route$dose_vbm$nda_num
10010001$1001001$1$PS$OZEMPIC$SEMAGLUTIDE$1$SC$0.5 MG WEEKLY$209637
10010002$1001001$1$PS$OZEMPIC$SEMAGLUTIDE$1$SC$0.5 MG WEEKLY$209637
10020001$1002001$1$PS$WEGOVY$SEMAGLUTIDE$1$SC$2.4 MG WEEKLY$215256
10030001$1003001$1$PS$MOUNJARO$TIRZEPATIDE$1$SC$5 MG WEEKLY$215866
10040001$1004001$1$PS$VICTOZA$LIRAGLUTIDE$1$SC$1.2 MG DAILY$22341
10040002$1004001$1$PS$VICTOZA$LIRAGLUTIDE$1$SC$1.2 MG DAILY$22341
99999001$9999001$1$PS$ASPIRIN$ASPIRIN$1$ORAL$100 MG DAILY$19537
"""

REAC_ROWS = """\
primaryid$caseid$pt$drug_rec_act
10010001$1001001$Nausea$
10010002$1001001$Nausea$
10010001$1001001$Vomiting$
10020001$1002001$Pancreatitis$
10020001$1002001$Abdominal pain$
10030001$1003001$Dizziness$
10040001$1004001$Constipation$
10040002$1004001$Constipation$
99999001$9999001$Headache$
"""

OUTC_ROWS = """\
primaryid$caseid$outc_cod
10010002$1001001$HO
10020001$1002001$OT
10030001$1003001$OT
10040002$1004001$OT
"""


@pytest.fixture(scope="session")
def synthetic_quarter() -> Quarter:
    """Standard test quarter: 2024 Q3."""
    return Quarter(2024, 3)


@pytest.fixture
def synthetic_zip_path(tmp_path, synthetic_quarter) -> Path:
    """
    Write a minimal valid FAERS-format ZIP to a temp directory.

    Contains DEMO, DRUG, REAC, OUTC.
    THER, RPSR, INDI are intentionally omitted to test missing-file handling.
    """
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    suffix   = synthetic_quarter.txt_suffix()
    zip_path = raw_dir / synthetic_quarter.zip_filename()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"ASCII/DEMO{suffix}.txt", DEMO_ROWS.encode("iso-8859-1"))
        zf.writestr(f"ASCII/DRUG{suffix}.txt", DRUG_ROWS.encode("iso-8859-1"))
        zf.writestr(f"ASCII/REAC{suffix}.txt", REAC_ROWS.encode("iso-8859-1"))
        zf.writestr(f"ASCII/OUTC{suffix}.txt", OUTC_ROWS.encode("iso-8859-1"))
    zip_path.write_bytes(buf.getvalue())

    return zip_path


@pytest.fixture
def processed_dir(tmp_path) -> Path:
    """Temp directory for processed Parquet output."""
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def logs_dir(tmp_path) -> Path:
    """Temp directory for log output."""
    d = tmp_path / "data" / "logs"
    d.mkdir(parents=True)
    return d
