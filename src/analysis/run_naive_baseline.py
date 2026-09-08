"""Run the naive global-threshold detector on the six Phase 1 candidate series.

For each series this prints:
  * how many points the detector flags,
  * where the flagged points sit relative to the normal daily cycle and to the
    labelled anomaly windows,
  * precision, recall and false-positive count under the time-tolerant scoring
    in `src/evaluation/score.py`,
and writes an overlay plot (series, flagged points, labelled windows, threshold
band) to `reports/figures/`.

Nothing here is aggregated across series. The point of the exercise is to see
the naive baseline fail on the strongly seasonal series and hold up on the
weakly seasonal one, and an average would hide exactly that.

Usage:
    python -m src.analysis.run_naive_baseline [--k 3.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.nab_loader import (  # noqa: E402
    CANDIDATE_SERIES,
    load_series,
    load_windows,
    sampling_step,
)
from src.detectors import naive_threshold  # noqa: E402
from src.evaluation.score import EDGE_TOLERANCE_STEPS, score_series  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "reports" / "figures"
SCORES_PATH = REPO_ROOT / "data" / "processed" / "naive_baseline_scores.csv"

PEAK_HOUR_QUANTILE = 0.75  # hours whose median value is in the top 25% are "peak hours"


def peak_hours(df: pd.DataFrame) -> list[int]:
    """Hours of day whose median value is in the top quartile across the 24 hours.

    This is a rough description of when the normal daily cycle is high, used only
    to characterise where the naive detector's false positives land.
    """
    by_hour = df.assign(hour=df["timestamp"].dt.hour).groupby("hour")["value"].median()
    cutoff = by_hour.quantile(PEAK_HOUR_QUANTILE)
    return sorted(by_hour[by_hour >= cutoff].index.tolist())


def contiguous_ranges(hours: list[int]) -> str:
    """Render a sorted hour list like [6,7,8,17,18] as '06-08, 17-18'."""
    if not hours:
        return "(none)"
    out, start, prev = [], hours[0], hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
            continue
        out.append(f"{start:02d}-{prev:02d}" if start != prev else f"{start:02d}")
        start = prev = h
    out.append(f"{start:02d}-{prev:02d}" if start != prev else f"{start:02d}")
    return ", ".join(out)


def overlay_plot(key, df, fit, flags, windows, score, path):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    t = df["timestamp"].values
    v = df["value"].values

    ax.plot(t, v, lw=0.5, color="#3b76af", zorder=1)
    for start, end in windows:
        ax.axvspan(start, end, color="red", alpha=0.18, zorder=0)

    fp = score.fp_mask
    tp = score.tp_mask
    ax.scatter(t[fp], v[fp], s=14, color="#d62728", label=f"false positive ({fp.sum()})", zorder=3)
    ax.scatter(t[tp], v[tp], s=14, color="#2ca02c", label=f"true positive ({tp.sum()})", zorder=3)

    ax.axhline(fit.upper, color="#888", lw=0.9, ls="--", zorder=2)
    ax.axhline(fit.lower, color="#888", lw=0.9, ls="--", zorder=2)
    ax.text(t[0], fit.upper, f"  mean + {fit.k:g}sigma", va="bottom", ha="left", fontsize=7, color="#555")

    prec = f"{score.precision:.3f}" if np.isfinite(score.precision) else "n/a"
    rec = f"{score.recall:.2f}" if np.isfinite(score.recall) else "n/a"
    ax.set_title(
        f"{key}   naive threshold k={fit.k:g}   "
        f"flagged {score.n_flagged}/{score.n_points}   "
        f"precision {prec}   recall {rec}   "
        f"windows hit {score.windows_hit}/{score.n_windows}   "
        f"(red band = labelled window)",
        fontsize=9,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=naive_threshold.DEFAULT_K,
                        help="threshold in standard deviations (default 3.0)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    grid_specs = []
    for key in CANDIDATE_SERIES:
        df = load_series(key)
        windows = load_windows(key)
        step = sampling_step(df)
        edge_tol = EDGE_TOLERANCE_STEPS * step

        fit = naive_threshold.fit(df["value"].values, k=args.k)
        flags = naive_threshold.detect(df["value"].values, k=args.k)
        score = score_series(df["timestamp"].values, flags, windows, edge_tol, series=key)

        ph = peak_hours(df)
        flagged_hours = df.loc[flags, "timestamp"].dt.hour
        fp_hours = df.loc[score.fp_mask, "timestamp"].dt.hour
        frac_all_in_peak = df["timestamp"].dt.hour.isin(ph).mean()
        frac_fp_in_peak = fp_hours.isin(ph).mean() if len(fp_hours) else float("nan")
        top_flagged_hours = flagged_hours.value_counts().head(3)

        fig_path = FIG_DIR / ("naive_baseline__" + key.replace("/", "__").replace(".csv", "") + ".png")
        overlay_plot(key, df, fit, flags, windows, score, fig_path)
        grid_specs.append((key, df, fit, flags, windows, score))

        row = score.as_row()
        row.update({
            "sampling_step": str(step),
            "edge_tolerance": str(edge_tol),
            "k": args.k,
            "mean": fit.mean,
            "std": fit.std,
            "upper": fit.upper,
            "lower": fit.lower,
            "peak_hours": contiguous_ranges(ph),
            "frac_all_obs_in_peak_hours": round(float(frac_all_in_peak), 3),
            "frac_false_pos_in_peak_hours": round(float(frac_fp_in_peak), 3)
            if np.isfinite(frac_fp_in_peak) else None,
            "fig": fig_path.name,
        })
        rows.append(row)

        print("=" * 100)
        print(f"{key}")
        print(f"  points={score.n_points}  sampling_step={step}  edge_tolerance=±{edge_tol}")
        print(f"  global mean={fit.mean:.4g}  std={fit.std:.4g}  "
              f"flag band = value < {fit.lower:.4g} or value > {fit.upper:.4g}")
        print(f"  FLAGGED: {score.n_flagged} points "
              f"({100 * score.n_flagged / score.n_points:.2f}% of the series)")
        print(f"  labelled windows: {score.n_windows}   "
              f"windows containing >=1 flag: {score.windows_hit}")
        print(f"  true positives (flags inside a window +/- tol): {score.true_positives}")
        print(f"  false positives (flags outside every window):   {score.false_positives}")
        print(f"  normal daily peak hours (top-quartile median): {contiguous_ranges(ph)}")
        print(f"  share of ALL observations in those peak hours:  {frac_all_in_peak:.3f}")
        if np.isfinite(frac_fp_in_peak):
            print(f"  share of FALSE POSITIVES in those peak hours:   {frac_fp_in_peak:.3f}"
                  f"   ({'concentrated in the normal peak' if frac_fp_in_peak > frac_all_in_peak + 0.1 else 'not concentrated in the normal peak'})")
        print(f"  top hours-of-day among flagged points: "
              f"{', '.join(f'{h:02d}h x{n}' for h, n in top_flagged_hours.items())}")
        prec = f"{score.precision:.3f}" if np.isfinite(score.precision) else "n/a (nothing flagged)"
        rec = f"{score.recall:.3f}" if np.isfinite(score.recall) else "n/a (no windows)"
        print(f"  PRECISION={prec}   RECALL={rec}   FALSE_POSITIVES={score.false_positives}")

    scores = pd.DataFrame(rows)
    scores.to_csv(SCORES_PATH, index=False)

    print("\n" + "=" * 100)
    print("PER-SERIES SUMMARY (no aggregation)")
    cols = ["series", "n_points", "n_flagged", "n_windows", "windows_hit",
            "true_positives", "false_positives", "precision", "recall",
            "frac_all_obs_in_peak_hours", "frac_false_pos_in_peak_hours"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda x: f"{x:.3f}"):
        print(scores[cols].to_string(index=False))
    print(f"\nwrote {SCORES_PATH.relative_to(REPO_ROOT)}")

    # overview grid
    n = len(grid_specs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n))
    for ax, (key, df, fit, flags, windows, score) in zip(np.atleast_1d(axes), grid_specs):
        t = df["timestamp"].values
        v = df["value"].values
        ax.plot(t, v, lw=0.4, color="#3b76af")
        for start, end in windows:
            ax.axvspan(start, end, color="red", alpha=0.18)
        ax.scatter(t[score.fp_mask], v[score.fp_mask], s=8, color="#d62728")
        ax.scatter(t[score.tp_mask], v[score.tp_mask], s=8, color="#2ca02c")
        ax.axhline(fit.upper, color="#888", lw=0.8, ls="--")
        ax.axhline(fit.lower, color="#888", lw=0.8, ls="--")
        prec = f"{score.precision:.3f}" if np.isfinite(score.precision) else "n/a"
        rec = f"{score.recall:.2f}" if np.isfinite(score.recall) else "n/a"
        ax.set_title(f"{key}   flagged {score.n_flagged}   FP {score.false_positives}   "
                     f"precision {prec}   recall {rec}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(f"Naive global-threshold baseline, k={args.k:g}  "
                 f"(red band = labelled anomaly window, green = true positive, red dot = false positive)",
                 fontsize=10)
    fig.tight_layout()
    grid_path = FIG_DIR / "naive_baseline__overview_grid.png"
    fig.savefig(grid_path, dpi=95)
    plt.close(fig)
    print(f"wrote {grid_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
