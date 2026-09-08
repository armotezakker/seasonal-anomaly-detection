"""Head-to-head: naive global-threshold baseline vs STL residual-threshold detector.

Reads the per-series scores written by `run_naive_baseline.py` and
`run_stl_baseline.py` and prints a side-by-side table. It also re-runs the naive
detector on the exact same regular grid the STL detector uses, so the question
"did STL do worse than the baseline" can be answered on identical inputs rather
than across the raw-vs-resampled difference.

Finally it writes stacked before/after overlay plots (naive on top, STL below,
shared x-axis) for every series, and singles out occupancy_6005 and nyc_taxi.

Usage:
    python -m src.analysis.compare_naive_vs_stl [--k 3.0]
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

from src.data.nab_loader import CANDIDATE_SERIES, load_series, load_windows, resample_to_grid  # noqa: E402
from src.detectors import naive_threshold, stl_threshold  # noqa: E402
from src.evaluation.score import EDGE_TOLERANCE_STEPS, score_series  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "reports" / "figures"
NAIVE_RAW_PATH = REPO_ROOT / "data" / "processed" / "naive_baseline_scores.csv"
STL_PATH = REPO_ROOT / "data" / "processed" / "stl_baseline_scores.csv"
COMPARE_PATH = REPO_ROOT / "data" / "processed" / "naive_vs_stl_comparison.csv"

SPOTLIGHT = ["realTraffic/occupancy_6005.csv", "realKnownCause/nyc_taxi.csv"]


def run_both_on_grid(key, k):
    """Return (series, filled, windows, edge_tol, naive_grid_score, stl_score)."""
    df = load_series(key)
    windows = load_windows(key)
    series, filled, step = resample_to_grid(df, method="linear")
    period = int(round(pd.Timedelta("1D") / step))
    edge_tol = EDGE_TOLERANCE_STEPS * step
    observed = ~filled.to_numpy()

    naive_fit = naive_threshold.fit(series.to_numpy()[observed], k=k)
    naive_flags = naive_threshold.detect(series.to_numpy(), k=k) & observed
    naive_score = score_series(series.index.values, naive_flags, windows, edge_tol, series=key)

    stl_res = stl_threshold.decompose_and_detect(series, period=period, filled=filled, k=k)
    stl_score = score_series(series.index.values, stl_res.observed_flags, windows, edge_tol, series=key)

    return series, filled, windows, edge_tol, naive_fit, naive_flags, naive_score, stl_res, stl_score


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"


def before_after_plot(key, series, filled, windows, naive_fit, naive_flags, naive_score,
                      stl_res, stl_score, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    t = series.index.values
    v = series.to_numpy()

    ax = axes[0]
    ax.plot(t, v, lw=0.5, color="#3b76af")
    for s, e in windows:
        ax.axvspan(s, e, color="red", alpha=0.16)
    ax.axhline(naive_fit.upper, color="#888", lw=0.9, ls="--")
    ax.axhline(naive_fit.lower, color="#888", lw=0.9, ls="--")
    ax.scatter(t[naive_score.fp_mask], v[naive_score.fp_mask], s=12, color="#d62728")
    ax.scatter(t[naive_score.tp_mask], v[naive_score.tp_mask], s=12, color="#2ca02c")
    ax.set_title(f"NAIVE (global mean +/- {naive_fit.k:g} sd, on the same grid)   "
                 f"flagged {naive_score.n_flagged}   FP {naive_score.false_positives}   "
                 f"precision {fmt(naive_score.precision)}   recall {fmt(naive_score.recall, 2)}   "
                 f"windows {naive_score.windows_hit}/{naive_score.n_windows}", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.fill_between(t, stl_res.fitted + stl_res.lower_resid, stl_res.fitted + stl_res.upper_resid,
                    color="#bbbbbb", alpha=0.45, lw=0)
    ax.plot(t, v, lw=0.5, color="#3b76af")
    for s, e in windows:
        ax.axvspan(s, e, color="red", alpha=0.16)
    ax.scatter(t[stl_score.fp_mask], v[stl_score.fp_mask], s=12, color="#d62728")
    ax.scatter(t[stl_score.tp_mask], v[stl_score.tp_mask], s=12, color="#2ca02c")
    ax.set_title(f"STL (trend + seasonal +/- {stl_res.k:g} * resid sd, period {stl_res.period})   "
                 f"flagged {stl_score.n_flagged}   FP {stl_score.false_positives}   "
                 f"precision {fmt(stl_score.precision)}   recall {fmt(stl_score.recall, 2)}   "
                 f"windows {stl_score.windows_hit}/{stl_score.n_windows}", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(labelsize=7)

    fig.suptitle(f"{key}   before / after   (red band = labelled window, green = TP, red dot = FP)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=stl_threshold.DEFAULT_K)
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    naive_raw = pd.read_csv(NAIVE_RAW_PATH).set_index("series")
    stl_tbl = pd.read_csv(STL_PATH).set_index("series")

    rows = []
    for key in CANDIDATE_SERIES:
        (series, filled, windows, edge_tol, naive_fit, naive_flags, naive_grid,
         stl_res, stl_score) = run_both_on_grid(key, args.k)

        nr = naive_raw.loc[key]
        rows.append({
            "series": key,
            "naive_raw_P": nr["precision"], "naive_raw_R": nr["recall"], "naive_raw_FP": int(nr["false_positives"]),
            "naive_grid_P": naive_grid.precision, "naive_grid_R": naive_grid.recall,
            "naive_grid_FP": naive_grid.false_positives,
            "stl_P": stl_score.precision, "stl_R": stl_score.recall, "stl_FP": stl_score.false_positives,
            "dP_vs_raw": stl_score.precision - nr["precision"],
            "dR_vs_raw": stl_score.recall - nr["recall"],
            "dFP_vs_raw": stl_score.false_positives - int(nr["false_positives"]),
            "dP_vs_grid": stl_score.precision - naive_grid.precision,
            "dFP_vs_grid": stl_score.false_positives - naive_grid.false_positives,
        })

        path = FIG_DIR / ("compare__" + key.replace("/", "__").replace(".csv", "") + ".png")
        before_after_plot(key, series, filled, windows, naive_fit, naive_flags, naive_grid,
                          stl_res, stl_score, path)

    comp = pd.DataFrame(rows)
    comp.to_csv(COMPARE_PATH, index=False)

    pd.set_option("display.width", 240, "display.max_columns", 30)
    fmt3 = lambda x: f"{x:.3f}"

    print("=" * 120)
    print("HEADLINE: naive baseline (Phase 2, raw irregular series)  vs  STL detector (Phase 3, regular grid)")
    head = comp[["series", "naive_raw_P", "naive_raw_R", "naive_raw_FP", "stl_P", "stl_R", "stl_FP",
                 "dP_vs_raw", "dR_vs_raw", "dFP_vs_raw"]]
    with pd.option_context("display.float_format", fmt3):
        print(head.to_string(index=False))

    print("\n" + "=" * 120)
    print("MATCHED INPUTS: naive vs STL, both on the same regular grid (so 'did STL do worse' is apples to apples)")
    matched = comp[["series", "naive_grid_P", "naive_grid_R", "naive_grid_FP",
                    "stl_P", "stl_R", "stl_FP", "dP_vs_grid", "dFP_vs_grid"]]
    with pd.option_context("display.float_format", fmt3):
        print(matched.to_string(index=False))

    print("\nnote: naive_grid differs from naive_raw only where resampling changed the point set")
    print("      (the two realTraffic series, ~47% filled). For the other four the grid is >=98% observed.")
    print(f"\nwrote {COMPARE_PATH.relative_to(REPO_ROOT)}")
    print("wrote compare__*.png for all six series; spotlight: "
          + ", ".join(Path('compare__' + k.replace('/', '__').replace('.csv', '') + '.png').name for k in SPOTLIGHT))


if __name__ == "__main__":
    main()
