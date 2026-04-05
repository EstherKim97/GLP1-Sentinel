"""
audit_chart.py
--------------
Generate the Phase 1 data quality chart from the deduplication audit log.

This is the first portfolio-visible output: a chart showing how many
duplicate records were removed per quarter across the full 2005–2024
scope, broken down by removal reason (Step 1: older FDA_DT vs.
Step 2: lower PRIMARYID tie-break).

The chart tells a story:
  - Early quarters (pre-2012 AERS era) tend to have lower dedup rates
    because the legacy system had fewer follow-up report mechanisms.
  - Post-2017 semaglutide / post-2022 tirzepatide quarters show
    elevated total record counts — visible as volume spikes.
  - Quarters with unusually high dedup rates (>20%) may indicate
    data quality events worth noting in the EDA report.

Usage:
    python -m faers_pipeline.audit_chart
    python -m faers_pipeline.audit_chart --audit data/logs/dedup_audit_v20240930.csv
    python -m faers_pipeline.audit_chart --out docs/dedup_quality_chart.png
"""

import argparse
import sys
from pathlib import Path


def find_latest_audit(logs_dir: Path) -> Path | None:
    """Find the most recently created audit CSV in the logs directory."""
    candidates = sorted(logs_dir.glob("dedup_audit_*.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def generate_chart(audit_csv: Path, out_path: Path, show: bool = False) -> None:
    """
    Read audit CSV and write the dedup quality chart.

    Requires: matplotlib, pandas (both in requirements.txt for Phase 5+).
    Install now if needed: pip install matplotlib
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend; safe in CI
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import pandas as pd
    except ImportError:
        print(
            "ERROR: matplotlib is required for this script.\n"
            "Install: pip install matplotlib pandas"
        )
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(audit_csv)
    df = df.sort_values("quarter").reset_index(drop=True)

    # Parse quarter to a sortable label
    quarters = df["quarter"].tolist()
    x        = range(len(quarters))

    removed_s1  = df["removed_step1"].tolist()
    removed_s2  = df["removed_step2"].tolist()
    n_final     = df["n_final"].tolist()
    dedup_rates = (df["dedup_rate"] * 100).tolist()

    # ── GLP-1 milestone annotations ───────────────────────────────────────────
    MILESTONES = {
        "2005Q2": "Exenatide\n(Byetta)",
        "2010Q1": "Liraglutide\n(Victoza)",
        "2017Q4": "Semaglutide SC\n(Ozempic)",
        "2019Q3": "Semaglutide oral\n(Rybelsus)",
        "2021Q2": "Wegovy\n(obesity)",
        "2022Q2": "Tirzepatide\n(Mounjaro)",
    }
    milestone_positions = {
        i: MILESTONES[q] for i, q in enumerate(quarters) if q in MILESTONES
    }

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize     = (18, 9),
        height_ratios = [2.5, 1],
        sharex      = True,
    )
    fig.patch.set_facecolor("#FAFAFA")

    BLUE_DARK   = "#1A3F6B"
    BLUE_MID    = "#378ADD"
    BLUE_LIGHT  = "#B5D4F4"
    AMBER       = "#EF9F27"
    GREEN       = "#3B6D11"
    GRID_COLOR  = "#DDDDDD"

    # ── Ax1: Stacked bar — record counts ─────────────────────────────────────
    ax1.set_facecolor("#FAFAFA")
    bar_width = 0.75

    b_final = ax1.bar(x, n_final,     width=bar_width, color=BLUE_MID,   label="Unique cases (retained)", zorder=2)
    b_s1    = ax1.bar(x, removed_s1,  width=bar_width, bottom=n_final,   color=BLUE_LIGHT, label="Removed — Step 1: older FDA_DT",     zorder=2)
    b_s2    = ax1.bar(x, removed_s2,  width=bar_width,
                      bottom=[a + b for a, b in zip(n_final, removed_s1)],
                      color=AMBER, label="Removed — Step 2: lower PRIMARYID tie", zorder=2)

    ax1.set_ylabel("DEMO records per quarter", fontsize=11, color="#444444")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(left=False, labelsize=9)

    # Milestone vertical lines
    for pos, label in milestone_positions.items():
        ax1.axvline(pos, color=GREEN, linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
        ax1.text(
            pos + 0.15, ax1.get_ylim()[1] * 0.97,
            label, fontsize=7, color=GREEN,
            va="top", ha="left", linespacing=1.3,
        )

    ax1.legend(
        handles=[b_final, b_s1, b_s2],
        loc="upper left", fontsize=9,
        framealpha=0.8, edgecolor=GRID_COLOR,
    )
    ax1.set_title(
        "FAERS-GLP1-Watch — Phase 1 Data Quality: Deduplication Audit\n"
        "2005 Q2 → 2024 Q3  |  FDA-recommended 2-step deduplication",
        fontsize=13, fontweight="bold", color=BLUE_DARK, pad=14,
    )

    # ── Ax2: Line — dedup rate % ──────────────────────────────────────────────
    ax2.set_facecolor("#FAFAFA")
    ax2.plot(x, dedup_rates, color=AMBER, linewidth=1.8, zorder=3)
    ax2.fill_between(x, dedup_rates, alpha=0.15, color=AMBER, zorder=2)

    # Flag quarters with dedup rate > 20% (anomalies worth investigating)
    HIGH_THRESHOLD = 20.0
    for i, rate in enumerate(dedup_rates):
        if rate > HIGH_THRESHOLD:
            ax2.scatter(i, rate, color="red", s=30, zorder=4)

    ax2.axhline(HIGH_THRESHOLD, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
    ax2.text(
        len(x) - 1, HIGH_THRESHOLD + 0.3,
        f">{HIGH_THRESHOLD:.0f}% — review",
        ha="right", va="bottom", fontsize=8, color="red", alpha=0.7,
    )

    ax2.set_ylabel("Dedup rate (%)", fontsize=10, color="#444444")
    ax2.set_ylim(0, max(dedup_rates) * 1.25 if dedup_rates else 30)
    ax2.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(left=False, labelsize=9)

    # ── X-axis ticks ─────────────────────────────────────────────────────────
    # Show every 4th quarter label (annual) to avoid crowding
    tick_step = 4
    ax2.set_xticks([i for i in x if i % tick_step == 0])
    ax2.set_xticklabels(
        [quarters[i] for i in x if i % tick_step == 0],
        rotation=45, ha="right", fontsize=8,
    )
    ax2.set_xlim(-0.5, len(x) - 0.5)

    # ── Footer annotation ─────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Source: FDA FAERS/AEMS quarterly ASCII files  |  "
        "Dedup rule: keep latest FDA_DT per CASEID; tie-break on highest PRIMARYID  |  "
        "Reference: Potter et al. 2025, Clin Pharmacol Ther",
        ha="center", fontsize=7.5, color="#888888",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    print(f"Chart saved → {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 deduplication quality chart from audit log."
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Path to dedup_audit_*.csv (default: auto-detect latest in data/logs/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/dedup_quality_chart.png"),
        help="Output path for chart PNG (default: docs/dedup_quality_chart.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open chart in interactive window after saving.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root (default: current directory)",
    )
    args = parser.parse_args()

    # Resolve audit CSV
    if args.audit:
        audit_csv = args.audit
    else:
        logs_dir  = args.root / "data" / "logs"
        audit_csv = find_latest_audit(logs_dir)
        if audit_csv is None:
            print(
                f"ERROR: No dedup_audit_*.csv found in {logs_dir}\n"
                "Run the pipeline first: python -m faers_pipeline.pipeline"
            )
            sys.exit(1)

    print(f"Reading audit log: {audit_csv}")
    generate_chart(audit_csv, args.out.resolve(), show=args.show)


if __name__ == "__main__":
    main()
