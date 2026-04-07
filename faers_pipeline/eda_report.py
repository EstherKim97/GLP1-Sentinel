"""
eda_report.py
-------------
Phase 5: Generate a self-contained HTML EDA report.

Produces a single HTML file with embedded charts covering:
  1. Data quality — dedup rates by quarter
  2. GLP-1 case volume over time by drug
  3. SOC distribution — cases per system organ class
  4. Top signals — ROR-ranked PT signals per drug
  5. Signal heatmap — drugs × SOC flagged pairs
  6. Time-to-onset — onset distribution per drug
  7. Missingness — null rate heatmap for DEMO columns
  8. Compounded vs brand — case counts over time

DECISION LOG — Report format choice
--------------------------------------
We generate a single self-contained HTML file, not a Jupyter notebook,
because:
  1. No Jupyter required to view it — opens in any browser.
  2. Self-contained: plotly embeds as CDN script + JSON data.
     The HTML file is portable and shareable.
  3. GitHub renders HTML files via htmlpreview.github.io — this
     becomes a direct portfolio artifact with a shareable link.
  4. Reproducible: the script is called from pipeline.py or standalone,
     always generates from the same processed Parquet files.

DECISION — Plotly over matplotlib for the report.
  Plotly charts are interactive (hover, zoom, toggle traces).
  For a portfolio artifact, interactivity is more impressive than
  static PNGs. The CDN script is one line; no server needed.
  (matplotlib is still used for the CI/audit_chart PNG in Phase 1.)

Usage:
  # After pipeline.py runs:
  python -m faers_pipeline.eda_report

  # Point at a specific processed dir:
  python -m faers_pipeline.eda_report --processed data/processed --out docs/eda_report.html
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


# ── HTML scaffolding ──────────────────────────────────────────────────────────

_HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FAERS-GLP1-Watch — EDA Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem;
    background: #f9f9f8; color: #2c2c2a; line-height: 1.65;
  }}
  h1 {{ font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem; color: #1a3f6b; }}
  h2 {{ font-size: 1.3rem; font-weight: 500; margin: 2.5rem 0 0.75rem; color: #185fa5; border-bottom: 1px solid #d3d1c7; padding-bottom: 0.4rem; }}
  h3 {{ font-size: 1.1rem; font-weight: 500; margin: 1.5rem 0 0.5rem; color: #333; }}
  .meta {{ font-size: 0.875rem; color: #888; margin-bottom: 2rem; }}
  .chart {{ background: #fff; border: 1px solid #e0dfd8; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 1rem 0; }}
  .stat {{ background: #fff; border: 1px solid #e0dfd8; border-radius: 8px; padding: 1rem; }}
  .stat .val {{ font-size: 1.75rem; font-weight: 600; color: #185fa5; }}
  .stat .lbl {{ font-size: 0.8rem; color: #888; margin-top: 2px; }}
  .decision {{ background: #e6f1fb; border-left: 3px solid #185fa5; padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0; font-size: 0.9rem; }}
  .warning {{ background: #fef3e2; border-left: 3px solid #ef9f27; padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0; font-size: 0.9rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; margin: 1rem 0; }}
  th {{ background: #1a3f6b; color: #fff; padding: 8px 12px; text-align: left; font-weight: 500; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #e0dfd8; }}
  tr:nth-child(even) td {{ background: #f5f5f3; }}
</style>
</head>
<body>
<h1>FAERS-GLP1-Watch</h1>
<h1 style="font-size:1.1rem;font-weight:400;color:#555;margin-top:-0.5rem;">EDA Report — GLP-1 Drug Class Pharmacovigilance</h1>
<p class="meta">Generated {generated_at} &nbsp;·&nbsp; Data scope: {scope_start} → {scope_end} &nbsp;·&nbsp; Pipeline v0.1.0</p>
"""

_HTML_FOOTER = """
<hr style="margin:3rem 0;border:none;border-top:1px solid #d3d1c7;">
<p style="font-size:0.8rem;color:#888;">
  Source: FDA FAERS/AEMS quarterly ASCII files &nbsp;·&nbsp;
  Dedup: Potter et al. 2025 (Clin Pharmacol Ther) FDA-recommended 2-step rule &nbsp;·&nbsp;
  Signals: ROR/PRR/IC multi-algorithm (≥2/3 methods) &nbsp;·&nbsp;
  MedDRA v26 SOC hierarchy
</p>
</body>
</html>"""


def _plotly_div(fig, div_id: str) -> str:
    """Render a plotly figure to an HTML div string."""
    try:
        import plotly.io as pio
        return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                           div_id=div_id, config={"responsive": True})
    except Exception as e:
        logger.warning(f"  Chart '{div_id}' failed to render: {e}")
        return f'<div class="warning">Chart unavailable: {e}</div>'


def _fmt(n) -> str:
    """Format large numbers with comma separators."""
    if n is None:
        return "N/A"
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


# ── Individual chart builders ─────────────────────────────────────────────────

def _chart_dedup(audit_csv: Path) -> str:
    """Section 1: Dedup audit chart."""
    try:
        import plotly.graph_objects as go
        df = pd.read_csv(audit_csv).sort_values("quarter")
        fig = go.Figure()
        fig.add_bar(x=df["quarter"], y=df["n_final"],
                    name="Unique cases (retained)", marker_color="#378ADD")
        fig.add_bar(x=df["quarter"], y=df["removed_step1"],
                    name="Removed — Step 1 (older FDA_DT)", marker_color="#B5D4F4")
        fig.add_bar(x=df["quarter"], y=df["removed_step2"],
                    name="Removed — Step 2 (lower PRIMARYID)", marker_color="#EF9F27")
        fig.update_layout(
            barmode="stack", height=380, margin=dict(l=60, r=20, t=40, b=100),
            title="DEMO records per quarter — dedup breakdown",
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            legend=dict(orientation="h", y=1.08),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig.update_yaxes(title="Records", tickformat=",")
        return '<div class="chart">' + _plotly_div(fig, "dedup_chart") + "</div>"
    except Exception as e:
        return f'<div class="warning">Dedup chart unavailable: {e}</div>'


def _chart_case_volume(drug_glp1_ps: pd.DataFrame) -> str:
    """Section 2: GLP-1 case volume by quarter and drug."""
    try:
        import plotly.graph_objects as go
        if drug_glp1_ps.empty or "_quarter" not in drug_glp1_ps.columns:
            return '<div class="warning">GLP-1 case volume chart: no data</div>'

        vol = (
            drug_glp1_ps.groupby(["_quarter", "glp1_active_ingredient"])["primaryid"]
            .nunique()
            .reset_index()
            .rename(columns={"primaryid": "cases"})
        )
        # Sort chronologically — convert YYYYQN to sortable string YYYY-QN
        vol["_q_sort"] = vol["_quarter"].str.replace("Q", "-Q")
        vol = vol.sort_values("_q_sort")

        fig = go.Figure()
        COLORS = ["#185fa5","#1D9E75","#EF9F27","#D85A30","#7F77DD","#D4537E","#639922"]
        for i, drug in enumerate(sorted(vol["glp1_active_ingredient"].unique())):
            sub = vol[vol["glp1_active_ingredient"] == drug]
            fig.add_scatter(x=sub["_quarter"], y=sub["cases"],
                            name=drug.title(), mode="lines+markers",
                            line=dict(color=COLORS[i % len(COLORS)], width=2))
        fig.update_layout(
            title="GLP-1 primary-suspect cases per quarter by drug",
            height=380, margin=dict(l=60, r=20, t=40, b=100),
            xaxis=dict(tickangle=45, tickfont=dict(size=9),
                       categoryorder="array",
                       categoryarray=sorted(vol["_quarter"].unique(),
                                           key=lambda x: x.replace("Q","-Q"))),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig.update_yaxes(title="Unique cases", tickformat=",")
        return '<div class="chart">' + _plotly_div(fig, "vol_chart") + "</div>"
    except Exception as e:
        return f'<div class="warning">Case volume chart unavailable: {e}</div>'


def _chart_soc_dist(soc_csv: Path) -> str:
    """Section 3: SOC distribution — unique cases per SOC."""
    try:
        import plotly.graph_objects as go
        df = pd.read_csv(soc_csv).sort_values("unique_cases")
        df = df[df["soc_name"].notna() & (df["soc_name"] != "Unmapped")]
        fig = go.Figure(go.Bar(
            x=df["unique_cases"], y=df["soc_name"],
            orientation="h", marker_color="#185fa5",
            text=df["unique_cases"].apply(lambda n: f"{n:,}"),
            textposition="outside",
        ))
        fig.update_layout(
            title="Unique GLP-1 cases per System Organ Class (primary SOC)",
            height=max(400, len(df) * 22 + 80),
            margin=dict(l=320, r=80, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig.update_xaxes(title="Unique cases", tickformat=",")
        return '<div class="chart">' + _plotly_div(fig, "soc_chart") + "</div>"
    except Exception as e:
        return f'<div class="warning">SOC distribution chart unavailable: {e}</div>'


def _chart_top_signals(signals_pt: pd.DataFrame) -> str:
    """Section 4: Top 20 PT-level signals ranked by ROR."""
    try:
        import plotly.graph_objects as go
        if signals_pt.empty:
            return '<div class="warning">No PT signals available</div>'

        top = (
            signals_pt[signals_pt["is_signal"]]
            .nlargest(20, "ror")
            .sort_values("ror")
        )
        if top.empty:
            return '<div class="warning">No signals detected at current thresholds</div>'

        label = top["drug"].str.title() + " — " + top["reaction_term"]
        fig = go.Figure()
        fig.add_bar(
            x=top["ror"], y=label, orientation="h",
            marker_color="#D85A30",
            error_x=dict(
                type="data",
                symmetric=False,
                array=(top["ror_ub"] - top["ror"]).clip(lower=0).tolist(),
                arrayminus=(top["ror"] - top["ror_lb"]).clip(lower=0).tolist(),
                color="#999", thickness=1.5, width=4,
            ),
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="#888", line_width=1)
        fig.update_layout(
            title="Top 20 PT-level signals by ROR (≥2/3 methods, a ≥ 3)",
            height=max(400, len(top) * 24 + 80),
            margin=dict(l=340, r=80, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig.update_xaxes(title="Reporting Odds Ratio (ROR)", type="log")
        return '<div class="chart">' + _plotly_div(fig, "signal_chart") + "</div>"
    except Exception as e:
        return f'<div class="warning">Signal chart unavailable: {e}</div>'


def _chart_signal_heatmap(signals_soc: pd.DataFrame) -> str:
    """Section 5: Signal heatmap — drugs × SOC."""
    try:
        import plotly.graph_objects as go
        import numpy as np
        if signals_soc.empty:
            return '<div class="warning">No SOC signals for heatmap</div>'

        pivot = signals_soc.pivot_table(
            index="reaction_term", columns="drug",
            values="n_signals", aggfunc="max", fill_value=0,
        )
        pivot.columns = [c.title() for c in pivot.columns]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0,"#f0f0ee"],[0.33,"#B5D4F4"],[0.67,"#378ADD"],[1,"#1a3f6b"]],
            zmin=0, zmax=3,
            colorbar=dict(title="Methods<br>signalling", tickvals=[0,1,2,3]),
            hovertemplate="Drug: %{x}<br>SOC: %{y}<br>Methods: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title="Signal heatmap: methods flagging each drug × SOC pair (0–3)",
            height=max(400, len(pivot) * 28 + 100),
            margin=dict(l=320, r=80, t=40, b=80),
            paper_bgcolor="white",
            xaxis=dict(tickangle=30),
        )
        return '<div class="chart">' + _plotly_div(fig, "heatmap_chart") + "</div>"
    except Exception as e:
        return f'<div class="warning">Heatmap unavailable: {e}</div>'


def _chart_tto(tto_summary: pd.DataFrame) -> str:
    """Section 6: Time-to-onset box plot per drug."""
    try:
        import plotly.graph_objects as go
        if tto_summary.empty or "median_days" not in tto_summary.columns:
            return '<div class="warning">Time-to-onset data not available</div>'

        df = tto_summary.sort_values("median_days")
        fig = go.Figure()
        COLORS = ["#185fa5","#1D9E75","#EF9F27","#D85A30","#7F77DD","#D4537E","#639922"]
        for i, row in df.iterrows():
            color = COLORS[i % len(COLORS)]
            fig.add_scatter(
                x=[row["drug"].title()],
                y=[row["median_days"]],
                error_y=dict(
                    type="data", symmetric=False,
                    array=[row["q75_days"] - row["median_days"]],
                    arrayminus=[row["median_days"] - row["q25_days"]],
                    color=color, thickness=3, width=10,
                ),
                mode="markers",
                marker=dict(size=12, color=color),
                name=row["drug"].title(),
                hovertemplate=(
                    f"<b>{row['drug'].title()}</b><br>"
                    f"Median: {row['median_days']} days<br>"
                    f"IQR: {row['q25_days']}–{row['q75_days']} days<br>"
                    f"n={_fmt(row['n'])}<br>"
                    f"Within 30d: {row['pct_within_30d']}%"
                    "<extra></extra>"
                ),
            )
        fig.add_hline(y=30, line_dash="dot", line_color="#aaa", line_width=1,
                      annotation_text="30 days", annotation_position="right")
        fig.update_layout(
            title="Time-to-onset: median days from drug start to adverse event (IQR bars)",
            height=380, showlegend=False,
            margin=dict(l=60, r=80, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig.update_yaxes(title="Days to onset")
        return '<div class="chart">' + _plotly_div(fig, "tto_chart") + "</div>"
    except Exception as e:
        return f'<div class="warning">Time-to-onset chart unavailable: {e}</div>'


def _chart_missingness(demo_glp1: pd.DataFrame) -> str:
    """Section 7: Missingness heatmap for DEMO columns."""
    try:
        import plotly.graph_objects as go
        key_cols = [
            "age", "sex", "wt", "occr_country", "reporter_country",
            "age_grp", "occp_cod", "rept_cod", "caseversion",
            "event_dt", "fda_dt", "init_fda_dt",
        ]
        present = [c for c in key_cols if c in demo_glp1.columns]
        if not present:
            return '<div class="warning">No DEMO data for missingness chart</div>'

        null_rates = {
            c: round(demo_glp1[c].isna().mean() * 100, 1)
            for c in present
        }
        df = pd.DataFrame.from_dict(null_rates, orient="index", columns=["null_pct"])
        df = df.sort_values("null_pct", ascending=False)

        fig = go.Figure(go.Bar(
            x=df.index.tolist(),
            y=df["null_pct"].tolist(),
            marker_color=[
                "#E24B4A" if v >= 40 else "#EF9F27" if v >= 20 else "#378ADD"
                for v in df["null_pct"]
            ],
            text=[f"{v}%" for v in df["null_pct"]],
            textposition="outside",
        ))
        fig.add_hline(y=20, line_dash="dot", line_color="#EF9F27", line_width=1,
                      annotation_text="20% threshold", annotation_position="right")
        fig.update_layout(
            title="Missingness in GLP-1 DEMO records (red ≥ 40%, amber ≥ 20%)",
            height=350, margin=dict(l=60, r=80, t=40, b=80),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig.update_yaxes(title="Missing (%)", range=[0, min(110, df["null_pct"].max() + 15)])
        fig.update_xaxes(tickangle=30)
        return '<div class="chart">' + _plotly_div(fig, "missing_chart") + "</div>"
    except Exception as e:
        return f'<div class="warning">Missingness chart unavailable: {e}</div>'


def _chart_compounded(drug_glp1_ps: pd.DataFrame) -> str:
    """Section 8: Compounded vs brand-name cases over time."""
    try:
        import plotly.graph_objects as go
        if drug_glp1_ps.empty or "is_compounded" not in drug_glp1_ps.columns:
            return '<div class="warning">Compounded analysis: no data</div>'

        def _cases_by_quarter(mask):
            return (
                drug_glp1_ps[mask]
                .groupby("_quarter")["primaryid"]
                .nunique()
                .reset_index()
                .rename(columns={"primaryid": "cases"})
            )

        brand   = _cases_by_quarter(~drug_glp1_ps["is_compounded"])
        comp    = _cases_by_quarter(drug_glp1_ps["is_compounded"])

        fig = go.Figure()
        fig.add_scatter(x=brand["_quarter"], y=brand["cases"],
                        name="Brand / approved", mode="lines+markers",
                        line=dict(color="#185fa5", width=2))
        if not comp.empty:
            fig.add_scatter(x=comp["_quarter"], y=comp["cases"],
                            name="Compounded (unapproved)", mode="lines+markers",
                            line=dict(color="#D85A30", width=2, dash="dot"))
        fig.update_layout(
            title="GLP-1 primary-suspect cases: brand vs compounded over time",
            height=360, margin=dict(l=60, r=20, t=40, b=100),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig.update_yaxes(title="Unique cases", tickformat=",")
        return '<div class="chart">' + _plotly_div(fig, "comp_chart") + "</div>"
    except Exception as e:
        return f'<div class="warning">Compounded chart unavailable: {e}</div>'


# ── Signal table ──────────────────────────────────────────────────────────────

def _signal_table_html(signals_pt: pd.DataFrame, max_rows: int = 30) -> str:
    """Render top signals as an HTML table."""
    if signals_pt.empty:
        return "<p>No signal data available.</p>"
    top = (
        signals_pt[signals_pt["is_signal"]]
        .nlargest(max_rows, "ror")
        [["drug","reaction_term","a","ror","ror_lb","ror_ub",
          "prr","ic025","n_signals"]]
        .copy()
    )
    if top.empty:
        return "<p>No signals detected at current thresholds (≥2/3 methods, a ≥ 3).</p>"
    top.columns = ["Drug","Reaction (PT)","Cases (a)","ROR",
                   "ROR LB","ROR UB","PRR","IC025","Methods"]
    top["Drug"] = top["Drug"].str.title()
    rows = ""
    for _, r in top.iterrows():
        rows += "<tr>" + "".join(
            f"<td>{v:.2f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
            for v in r
        ) + "</tr>"
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{c}</th>" for c in top.columns)
        + "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


# ── Stats bar ─────────────────────────────────────────────────────────────────

def _stats_html(
    demo_all: pd.DataFrame,
    demo_glp1: pd.DataFrame,
    drug_glp1_ps: pd.DataFrame,
    signals_pt: pd.DataFrame,
) -> str:
    """Render top-line summary stat cards."""
    n_all   = len(demo_all) if demo_all is not None else 0
    n_glp1  = len(demo_glp1) if demo_glp1 is not None else 0
    n_rxns  = len(drug_glp1_ps) if drug_glp1_ps is not None else 0
    n_sig   = int(signals_pt["is_signal"].sum()) if (signals_pt is not None and not signals_pt.empty) else 0

    stats = [
        (_fmt(n_all),  "Total FAERS cases (all drugs)"),
        (_fmt(n_glp1), "GLP-1 primary-suspect cases"),
        (_fmt(n_rxns), "GLP-1 DRUG-level records (PS)"),
        (_fmt(n_sig),  "PT-level signals detected (≥2/3 methods)"),
    ]
    cards = "".join(
        f'<div class="stat"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
        for v, l in stats
    )
    return f'<div class="stat-grid">{cards}</div>'


# ── Main report builder ───────────────────────────────────────────────────────

def build_report(
    processed_dir : Path,
    logs_dir      : Path,
    out_path      : Path,
    scope_start   : str = "2005 Q2",
    scope_end     : str = "2024 Q3",
) -> None:
    """
    Build the full EDA HTML report from processed Parquet files.

    Loads each file independently — missing files produce a warning
    section, not a crash. The report is always generated even if
    only partial data is available.
    """
    try:
        import plotly
    except ImportError:
        print("ERROR: plotly required. Install: pip install plotly")
        return

    def _load(pattern: str, columns: list = None) -> pd.DataFrame:
        """Load the most recent Parquet matching pattern."""
        files = sorted(processed_dir.glob(pattern))
        if not files:
            logger.warning(f"  No file matching {pattern}")
            return pd.DataFrame()
        return pd.read_parquet(files[-1], columns=columns)

    def _load_csv(pattern: str) -> pd.DataFrame:
        files = sorted((logs_dir).glob(pattern))
        if not files:
            # Also check processed_dir
            files = sorted(processed_dir.glob(pattern))
        if not files:
            return pd.DataFrame()
        return pd.read_csv(files[-1])

    logger.info("  Loading processed files...")
    demo_all     = _load("DEMO_deduplicated*.parquet",
                         columns=["primaryid", "_quarter"])
    demo_glp1    = _load("DEMO_glp1*.parquet")
    drug_glp1_ps = _load("DRUG_glp1_ps*.parquet")
    signals_pt   = _load("signals_pt*.parquet")
    signals_soc  = _load("signals_soc*.parquet")
    tto_sum      = _load("tto_summary*.parquet")
    soc_csv      = processed_dir / sorted(processed_dir.glob("SOC_summary*.csv"),
                                          key=lambda p: p.name)[-1].name \
                   if list(processed_dir.glob("SOC_summary*.csv")) else None

    audit_csv = sorted(logs_dir.glob("dedup_audit*.csv"))
    audit_csv = audit_csv[-1] if audit_csv else None

    # ── Assemble HTML sections ────────────────────────────────────────────────
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = _HTML_HEADER.format(
        generated_at=generated_at,
        scope_start=scope_start,
        scope_end=scope_end,
    )

    # Summary stats
    html += "<h2>Summary</h2>"
    html += _stats_html(demo_all, demo_glp1, drug_glp1_ps, signals_pt)

    # Section 1 — Data quality
    html += "<h2>1 · Data quality — deduplication audit</h2>"
    html += """<div class="decision">
    <strong>Decision:</strong> FDA-recommended 2-step deduplication.
    Step 1: keep latest <code>FDA_DT</code> per <code>CASEID</code>.
    Step 2: break ties on highest <code>PRIMARYID</code>.
    Reference: Potter et al. 2025 (Clin Pharmacol Ther).
    </div>"""
    if audit_csv:
        html += _chart_dedup(audit_csv)
    else:
        html += '<div class="warning">Dedup audit CSV not found. Run the pipeline first.</div>'

    # Section 2 — GLP-1 case volume
    html += "<h2>2 · GLP-1 case volume over time</h2>"
    html += """<div class="decision">
    <strong>Decision:</strong> Scope = primary-suspect (<code>role_cod = PS</code>) GLP-1 reports only.
    Concomitant drugs are excluded from signal analysis but retained in full DRUG file.
    </div>"""
    html += _chart_case_volume(drug_glp1_ps)

    # Section 3 — SOC distribution
    html += "<h2>3 · System Organ Class distribution</h2>"
    if soc_csv and soc_csv.exists():
        html += _chart_soc_dist(soc_csv)
    else:
        html += '<div class="warning">SOC summary CSV not found.</div>'

    # Section 4 — Top signals
    html += "<h2>4 · Top PT-level signals (ROR ranked)</h2>"
    html += """<div class="decision">
    <strong>Signal criteria:</strong>
    A pair (drug, PT) is flagged when ≥2 of 3 methods agree:
    ROR lower 95% CI &gt; 1.0 &nbsp;|&nbsp;
    PRR ≥ 2.0 and χ² ≥ 4.0 &nbsp;|&nbsp;
    IC025 &gt; 0.
    Minimum cases a ≥ 3.
    </div>"""
    html += _chart_top_signals(signals_pt)
    html += "<h3>Full signal table (top 30 by ROR)</h3>"
    html += _signal_table_html(signals_pt, max_rows=30)

    # Section 5 — Heatmap
    html += "<h2>5 · Signal heatmap — drug × System Organ Class</h2>"
    html += _chart_signal_heatmap(signals_soc)

    # Section 6 — Time-to-onset
    html += "<h2>6 · Time-to-onset analysis</h2>"
    html += """<div class="decision">
    <strong>Method:</strong> TTO = event date (DEMO) − drug start date (THER).
    Includes only valid, non-negative TTO ≤ 3,650 days.
    Based on methodology in Scientific Reports 2025 GLP-1 neurological AE study.
    </div>"""
    html += _chart_tto(tto_sum)

    # Section 7 — Missingness
    html += "<h2>7 · Missingness in GLP-1 DEMO records</h2>"
    html += """<div class="decision">
    <strong>Known issue:</strong> Age is missing in ~49.5% of semaglutide reports
    (documented in published FAERS studies). Sex is missing in ~3–4%.
    These missingness rates are consistent with the broader FAERS literature
    and do not bias signal detection (which operates on reaction counts, not demographics).
    </div>"""
    html += _chart_missingness(demo_glp1)

    # Section 8 — Compounded vs brand
    html += "<h2>8 · Compounded vs brand-name cases</h2>"
    html += """<div class="decision">
    <strong>Decision:</strong> Compounded semaglutide/tirzepatide flagged separately.
    FDA issued explicit warnings 2024–2025 that these products are clinically distinct
    (unapproved salt forms, dosing errors, underreporting). Included in analysis by
    default; compare via <code>is_compounded</code> filter.
    </div>"""
    html += _chart_compounded(drug_glp1_ps)

    html += _HTML_FOOTER

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    logger.info(f"  EDA report saved → {out_path}  ({size_kb:,} KB)")
    print(f"\nEDA report → {out_path}  ({size_kb:,} KB)")
    print(f"  Open in browser: file://{out_path.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate FAERS-GLP1-Watch EDA HTML report."
    )
    parser.add_argument("--processed", type=Path, default=Path("data/processed"),
                        help="Processed data directory (default: data/processed)")
    parser.add_argument("--logs", type=Path, default=Path("data/logs"),
                        help="Logs directory (default: data/logs)")
    parser.add_argument("--out", type=Path, default=Path("docs/eda_report.html"),
                        help="Output HTML path (default: docs/eda_report.html)")
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Project root (default: current directory)")
    args = parser.parse_args()

    root = args.root.resolve()
    build_report(
        processed_dir = root / args.processed,
        logs_dir      = root / args.logs,
        out_path      = root / args.out,
    )


if __name__ == "__main__":
    main()
