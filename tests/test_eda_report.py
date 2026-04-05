"""
test_eda_report.py
------------------
Tests for Phase 5 EDA HTML report generation.

Strategy: test that the report builds without errors on minimal
synthetic data and produces an HTML file containing the expected
section headers. We don't pixel-check charts — we verify structure
and that chart functions return strings without crashing.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from faers_pipeline.eda_report import (
    build_report,
    _stats_html,
    _signal_table_html,
    _chart_dedup,
    _chart_case_volume,
    _chart_top_signals,
    _chart_signal_heatmap,
    _chart_tto,
    _chart_missingness,
    _chart_compounded,
    _fmt,
)


# ── _fmt ─────────────────────────────────────────────────────────────────────

class TestFmt:
    def test_formats_large_number(self):
        assert _fmt(1234567) == "1,234,567"

    def test_formats_zero(self):
        assert _fmt(0) == "0"

    def test_handles_none(self):
        assert _fmt(None) == "N/A"


# ── _stats_html ───────────────────────────────────────────────────────────────

class TestStatsHtml:

    def test_returns_string(self):
        result = _stats_html(
            pd.DataFrame({"primaryid": range(1000)}),
            pd.DataFrame({"primaryid": range(100)}),
            pd.DataFrame({"primaryid": range(50)}),
            pd.DataFrame({"is_signal": [True, False, True]}),
        )
        assert isinstance(result, str)
        assert "1,000" in result   # total cases
        assert "2" in result       # 2 signals

    def test_handles_empty_dataframes(self):
        result = _stats_html(
            pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(),
        )
        assert isinstance(result, str)


# ── _signal_table_html ────────────────────────────────────────────────────────

class TestSignalTableHtml:

    def _make_signals(self, n_signal=5, n_noise=10):
        signals = pd.DataFrame({
            "drug"         : ["semaglutide"] * (n_signal + n_noise),
            "reaction_term": [f"Reaction_{i}" for i in range(n_signal + n_noise)],
            "a"            : [10] * (n_signal + n_noise),
            "ror"          : [5.0] * n_signal + [0.8] * n_noise,
            "ror_lb"       : [2.0] * n_signal + [0.4] * n_noise,
            "ror_ub"       : [10.0] * n_signal + [1.6] * n_noise,
            "prr"          : [4.0] * n_signal + [0.9] * n_noise,
            "ic025"        : [1.5] * n_signal + [-0.5] * n_noise,
            "n_signals"    : [3] * n_signal + [0] * n_noise,
            "is_signal"    : [True] * n_signal + [False] * n_noise,
        })
        return signals

    def test_returns_html_string(self):
        result = _signal_table_html(self._make_signals())
        assert isinstance(result, str)
        assert "<table>" in result

    def test_only_shows_signals(self):
        result = _signal_table_html(self._make_signals(n_signal=3, n_noise=10))
        # Should show 3 signal rows in table, not 13
        assert result.count("Reaction_") <= 4  # up to max_rows signals

    def test_empty_returns_message(self):
        result = _signal_table_html(pd.DataFrame())
        assert "No signal data" in result or "<p>" in result


# ── Individual chart functions ────────────────────────────────────────────────

class TestChartFunctions:
    """Each chart function must return a string without crashing.
    We test with minimal valid data — not pixel-perfect output."""

    def test_chart_dedup_empty_path(self, tmp_path):
        result = _chart_dedup(tmp_path / "nonexistent.csv")
        assert isinstance(result, str)

    def test_chart_dedup_valid(self, tmp_path):
        csv = tmp_path / "dedup_audit.csv"
        pd.DataFrame({
            "quarter": ["2024Q1", "2024Q2"],
            "n_raw": [1000, 1200],
            "n_final": [900, 1080],
            "removed_step1": [80, 100],
            "removed_step2": [20, 20],
        }).to_csv(csv, index=False)
        result = _chart_dedup(csv)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_chart_case_volume_empty(self):
        result = _chart_case_volume(pd.DataFrame())
        assert isinstance(result, str)

    def test_chart_case_volume_valid(self):
        df = pd.DataFrame({
            "primaryid"             : ["1","2","3"],
            "_quarter"              : ["2024Q1","2024Q1","2024Q2"],
            "glp1_active_ingredient": ["semaglutide","semaglutide","tirzepatide"],
            "is_compounded"         : [False, False, False],
        })
        result = _chart_case_volume(df)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_chart_top_signals_no_signals(self):
        df = pd.DataFrame({
            "drug": ["semaglutide"], "reaction_term": ["Nausea"],
            "a": [5], "ror": [1.2], "ror_lb": [0.9], "ror_ub": [1.6],
            "prr": [1.1], "ic025": [-0.2], "n_signals": [0], "is_signal": [False],
        })
        result = _chart_top_signals(df)
        assert isinstance(result, str)

    def test_chart_top_signals_with_signals(self):
        df = pd.DataFrame({
            "drug": ["semaglutide"] * 5,
            "reaction_term": [f"PT_{i}" for i in range(5)],
            "a": [10] * 5,
            "ror": [5.0, 4.0, 3.5, 3.0, 2.5],
            "ror_lb": [2.0] * 5, "ror_ub": [10.0] * 5,
            "prr": [4.0] * 5, "ic025": [1.5] * 5,
            "n_signals": [3] * 5, "is_signal": [True] * 5,
        })
        result = _chart_top_signals(df)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_chart_signal_heatmap_empty(self):
        result = _chart_signal_heatmap(pd.DataFrame())
        assert isinstance(result, str)

    def test_chart_tto_empty(self):
        result = _chart_tto(pd.DataFrame())
        assert isinstance(result, str)

    def test_chart_tto_valid(self):
        df = pd.DataFrame({
            "drug"          : ["semaglutide", "tirzepatide"],
            "n"             : [100, 50],
            "median_days"   : [27.0, 19.0],
            "q25_days"      : [7.0, 3.0],
            "q75_days"      : [77.0, 69.0],
            "mean_days"     : [35.0, 25.0],
            "pct_within_30d": [46.0, 55.0],
        })
        result = _chart_tto(df)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_chart_missingness_empty(self):
        result = _chart_missingness(pd.DataFrame())
        assert isinstance(result, str)

    def test_chart_missingness_valid(self):
        df = pd.DataFrame({
            "primaryid": range(100),
            "age"      : [None] * 50 + list(range(50)),
            "sex"      : ["M"] * 97 + [None] * 3,
            "event_dt" : ["20240101"] * 100,
        })
        result = _chart_missingness(df)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_chart_compounded_empty(self):
        result = _chart_compounded(pd.DataFrame())
        assert isinstance(result, str)

    def test_chart_compounded_valid(self):
        df = pd.DataFrame({
            "primaryid"             : ["1","2","3","4"],
            "_quarter"              : ["2024Q1","2024Q1","2024Q2","2024Q2"],
            "glp1_active_ingredient": ["semaglutide"]*4,
            "is_compounded"         : [False, False, True, False],
        })
        result = _chart_compounded(df)
        assert isinstance(result, str)
        assert len(result) > 100


# ── build_report ──────────────────────────────────────────────────────────────

class TestBuildReport:

    def _make_processed_dir(self, tmp_path):
        """Write minimal Parquet files so build_report has something to load."""
        processed = tmp_path / "processed"
        processed.mkdir()
        logs = tmp_path / "logs"
        logs.mkdir()
        docs = tmp_path / "docs"
        docs.mkdir()

        # Minimal DEMO
        pd.DataFrame({"primaryid": range(100)}).to_parquet(
            processed / "DEMO_deduplicated_v20240930.parquet"
        )
        # Minimal DEMO_glp1
        pd.DataFrame({
            "primaryid": range(10),
            "age": [45]*5 + [None]*5,
            "sex": ["M"]*7 + [None]*3,
            "event_dt": ["20240101"]*10,
        }).to_parquet(processed / "DEMO_glp1_v20240930.parquet")
        # Minimal DRUG_glp1_ps
        pd.DataFrame({
            "primaryid"             : range(10),
            "_quarter"              : ["2024Q3"]*10,
            "glp1_active_ingredient": ["semaglutide"]*10,
            "is_compounded"         : [False]*9 + [True],
        }).to_parquet(processed / "DRUG_glp1_ps_v20240930.parquet")
        # Minimal signals_pt
        pd.DataFrame({
            "drug": ["semaglutide"]*3,
            "reaction_term": ["Nausea","Pancreatitis","Dizziness"],
            "a": [50, 10, 20],
            "b": [100, 200, 130],
            "c": [2000, 500, 800],
            "d": [7850, 9290, 9050],
            "N": [10000]*3,
            "ror": [2.5, 5.0, 1.1],
            "ror_lb": [1.5, 2.5, 0.6],
            "ror_ub": [4.0, 10.0, 2.0],
            "prr": [2.2, 4.8, 1.0],
            "prr_chi2": [15.0, 45.0, 0.2],
            "ic": [1.2, 2.5, 0.1],
            "ic025": [0.5, 1.8, -0.8],
            "signal_ror": [True, True, False],
            "signal_prr": [True, True, False],
            "signal_ic":  [True, True, False],
            "n_signals":  [3, 3, 0],
            "is_signal":  [True, True, False],
        }).to_parquet(processed / "signals_pt_v20240930.parquet")
        # Minimal signals_soc
        pd.DataFrame({
            "drug": ["semaglutide"]*2,
            "reaction_term": ["Gastrointestinal disorders","Nervous system disorders"],
            "a": [80, 30], "b": [70, 120], "c": [3000, 1000], "d": [6850, 8850],
            "N": [10000]*2, "ror": [2.8, 1.5], "ror_lb": [1.8, 1.0],
            "ror_ub": [4.5, 2.2], "prr": [2.5, 1.4], "prr_chi2": [20.0, 5.0],
            "ic": [1.4, 0.5], "ic025": [0.8, -0.1],
            "signal_ror": [True, True], "signal_prr": [True, False],
            "signal_ic": [True, False], "n_signals": [3, 1], "is_signal": [True, False],
        }).to_parquet(processed / "signals_soc_v20240930.parquet")
        # Dedup audit CSV
        pd.DataFrame({
            "quarter": ["2024Q1","2024Q2","2024Q3"],
            "n_raw": [1000, 1200, 1100],
            "n_final": [900, 1080, 990],
            "removed_step1": [80, 100, 90],
            "removed_step2": [20, 20, 20],
        }).to_csv(logs / "dedup_audit_v20240930.csv", index=False)

        return processed, logs, docs

    def test_report_file_created(self, tmp_path):
        processed, logs, docs = self._make_processed_dir(tmp_path)
        out = docs / "eda_report.html"
        build_report(processed, logs, out)
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial content

    def test_report_contains_key_sections(self, tmp_path):
        processed, logs, docs = self._make_processed_dir(tmp_path)
        out = docs / "eda_report.html"
        build_report(processed, logs, out)
        html = out.read_text(encoding="utf-8")
        for section in [
            "Data quality",
            "GLP-1 case volume",
            "System Organ Class",
            "Top PT-level signals",
            "Signal heatmap",
            "Time-to-onset",
            "Missingness",
            "Compounded vs brand",
        ]:
            assert section in html, f"Section '{section}' not found in report"

    def test_report_contains_plotly_script(self, tmp_path):
        processed, logs, docs = self._make_processed_dir(tmp_path)
        out = docs / "eda_report.html"
        build_report(processed, logs, out)
        html = out.read_text(encoding="utf-8")
        assert "plotly" in html.lower()

    def test_report_handles_missing_files_gracefully(self, tmp_path):
        """Report should build even with empty processed dir."""
        processed = tmp_path / "processed"
        processed.mkdir()
        logs = tmp_path / "logs"
        logs.mkdir()
        docs = tmp_path / "docs"
        docs.mkdir()
        out = docs / "eda_report.html"
        # Should not crash even with no data files
        build_report(processed, logs, out)
        assert out.exists()
