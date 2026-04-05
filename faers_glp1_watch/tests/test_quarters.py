"""
test_quarters.py
----------------
Unit tests for the Quarter class and registry logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from faers_pipeline.quarters import Quarter, quarters_in_range, SCOPE_START, SCOPE_END


class TestQuarter:

    def test_str(self):
        assert str(Quarter(2024, 3)) == "2024Q3"

    def test_label(self):
        assert Quarter(2024, 3).label() == "2024 Q3"

    def test_txt_suffix(self):
        assert Quarter(2024, 3).txt_suffix() == "24Q3"
        assert Quarter(2005, 2).txt_suffix() == "05Q2"

    def test_zip_filename_legacy_aers(self):
        # Pre-2012Q3 → aers_ascii_
        assert Quarter(2005, 2).zip_filename() == "aers_ascii_2005q2.zip"
        assert Quarter(2012, 2).zip_filename() == "aers_ascii_2012q2.zip"

    def test_zip_filename_faers(self):
        # 2012Q3 onward → faers_ascii_
        assert Quarter(2012, 3).zip_filename() == "faers_ascii_2012q3.zip"
        assert Quarter(2024, 3).zip_filename() == "faers_ascii_2024q3.zip"

    def test_zip_filename_aems(self):
        # 2026Q1 onward → aems_ascii_
        assert Quarter(2026, 1).zip_filename() == "aems_ascii_2026q1.zip"

    def test_download_url_contains_filename(self):
        q = Quarter(2024, 3)
        assert q.zip_filename() in q.download_url()
        assert "fis.fda.gov" in q.download_url()

    def test_next_within_year(self):
        assert Quarter(2024, 1).next() == Quarter(2024, 2)
        assert Quarter(2024, 3).next() == Quarter(2024, 4)

    def test_next_year_rollover(self):
        assert Quarter(2024, 4).next() == Quarter(2025, 1)

    def test_ordering(self):
        assert Quarter(2023, 4) < Quarter(2024, 1)
        assert Quarter(2024, 1) <= Quarter(2024, 1)
        assert not Quarter(2024, 2) < Quarter(2024, 1)


class TestQuartersInRange:

    def test_single_quarter(self):
        result = quarters_in_range(Quarter(2024, 3), Quarter(2024, 3))
        assert result == [Quarter(2024, 3)]

    def test_within_year(self):
        result = quarters_in_range(Quarter(2024, 1), Quarter(2024, 4))
        assert len(result) == 4
        assert result[0] == Quarter(2024, 1)
        assert result[-1] == Quarter(2024, 4)

    def test_across_years(self):
        result = quarters_in_range(Quarter(2023, 3), Quarter(2024, 2))
        expected = [
            Quarter(2023, 3), Quarter(2023, 4),
            Quarter(2024, 1), Quarter(2024, 2),
        ]
        assert result == expected

    def test_scope_count(self):
        # 2005Q2 → 2024Q3 = 78 quarters
        result = quarters_in_range(SCOPE_START, SCOPE_END)
        assert len(result) == 78
