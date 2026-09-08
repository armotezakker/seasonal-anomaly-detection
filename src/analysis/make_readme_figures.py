"""Generate the two figures embedded in README.md.

  reports/figures/nyc_taxi_before_after.png   naive vs final detector on nyc_taxi
  reports/figures/six_series_comparison.png   precision, naive vs final, all six series

Both are committed (see .gitignore) so they render on GitHub. Run after
scripts/fetch_data.sh:

    python -m src.analysis.make_readme_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.nab_loader import CANDIDATE_SERIES, load_series, load_windows, resample_to_grid  # noqa: E402
from src.detectors import naive_threshold, seasonal_anomaly  # noqa: E402
from src.evaluation.score import EDGE_TOLERANCE_STEPS, score_series  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "reports" / "figures"
FINAL_SCORES = REPO_ROOT / "data" / "processed" / "final_detector_scores.csv"

SERIES_LINE = "#3b76af"
TP_COLOR = "#2ca02c"
FP_COLOR = "#d62728"
WINDOW_FILL = "#d62728"
NAIVE_BAR = "#9aa7b1"
FINAL_BAR = "#2a9d8f"

# README narrative order: the two improved series first, then the four where the
# naive baseline is still the best result.
BAR_ORDER = [
    "realKnownCause/nyc_taxi.csv",
    "realAdExchange/exchange-2_cpm_results.csv",
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realTraffic/occupancy_t4013.csv",
    "realTraffic/occupancy_6005.csv",
    "realTweets/Twitter_volume_AMZN.csv",
]
SHORT_NAME = {
    "realKnownCause/nyc_taxi.csv": "nyc_taxi",
    "realAdExchange/exchange-2_cpm_results.csv": "exchange-2_cpm",
    "realKnownCause/ambient_temperature_system_failure.csv": "ambient_temp",
    "realTraffic/occupancy_t4013.csv": "occupancy_t4013",
    "realTraffic/occupancy_6005.csv": "occupancy_6005",
    "realTweets/Twitter_volume_AMZN.csv": "Twitter_AMZN",
}


def _clean_axes(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def run_both(key):
    df = load_series(key)
    windows = load_windows(key)
    series, filled, step = resample_to_grid(df, method="linear")
    edge_tol = EDGE_TOLERANCE_STEPS * step
    ts = series.index.values

    naive_flags = naive_threshold.detect(series.to_numpy(), k=3.0) & ~filled.to_numpy()
    naive_score = score_series(ts, naive_flags, windows, edge_tol, series=key)

    fin = seasonal_anomaly.detect(series, step, filled=filled, k=3.0, weekly_cycle_min=10.0)
    final_score = score_series(ts, fin.observed_flags, windows, edge_tol, series=key)

    return {
        "series": series, "windows": windows,
        "naive_score": naive_score, "final_score": final_score,
        "decomposition": fin.decomposition_used,
    }


def _panel(ax, series, windows, score, title, marker_size):
    t = series.index.values
    v = series.to_numpy()
    ax.plot(t, v, lw=0.5, color=SERIES_LINE, zorder=2)
    for start, end in windows:
        ax.axvspan(start, end, color=WINDOW_FILL, alpha=0.13, zorder=0)
    ax.scatter(t[score.fp_mask], v[score.fp_mask], s=marker_size, color=FP_COLOR,
               label=f"false positive ({score.false_positives})", zorder=4, edgecolor="none")
    ax.scatter(t[score.tp_mask], v[score.tp_mask], s=marker_size, color=TP_COLOR,
               label=f"true positive ({score.true_positives})", zorder=5, edgecolor="none")
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    ax.set_ylabel("passengers per 30 min", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    _clean_axes(ax)


def figure_one(res):
    series, windows = res["series"], res["windows"]
    naive, final = res["naive_score"], res["final_score"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    fig.patch.set_facecolor("white")

    _panel(axes[0], series, windows, naive,
           f"Naive baseline: {naive.windows_hit} of {naive.n_windows} anomalies caught", marker_size=55)
    _panel(axes[1], series, windows, final,
           f"Final detector: {final.windows_hit} of {final.n_windows} anomalies caught", marker_size=16)

    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].set_xlabel("")

    fig.suptitle("Adding the weekly cycle fixes the taxi data's blind spot",
                 fontsize=13, fontweight="bold", x=0.5, y=0.99)
    fig.text(0.5, 0.028,
             "Shaded bands are the labelled anomaly windows. The naive 3-sigma band is so wide the "
             "daily peaks never cross it, so it fires once in 215 days.",
             ha="center", fontsize=8, color="#555")
    fig.text(0.5, 0.006,
             "Several final-detector false positives align with US holidays (Jul 4, Labor Day) not in "
             "NAB's label set, plausibly genuine ridership shifts the benchmark didn't annotate, not confirmed.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    out = FIG_DIR / "nyc_taxi_before_after.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out, naive, final


def figure_two(scores: pd.DataFrame):
    """Bars from data/processed/final_detector_scores.csv, so the figure matches
    the naive (Phase 2) vs final comparison used everywhere else in the project."""
    s = scores.set_index("series")
    labels = [SHORT_NAME[k] for k in BAR_ORDER]
    naive_p = np.array([s.loc[k, "naive_P"] for k in BAR_ORDER])
    final_p = np.array([s.loc[k, "final_P"] for k in BAR_ORDER])
    n_windows = np.array([int(str(s.loc[k, "final_windows"]).split("/")[1]) for k in BAR_ORDER])
    naive_hit = np.rint(np.array([s.loc[k, "naive_R"] for k in BAR_ORDER]) * n_windows).astype(int)
    final_hit = np.array([int(str(s.loc[k, "final_windows"]).split("/")[0]) for k in BAR_ORDER])

    x = np.arange(len(BAR_ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.patch.set_facecolor("white")

    # faint background bands: first two improved, last four naive-still-best
    ax.axvspan(-0.5, 1.5, color="#2a9d8f", alpha=0.06, zorder=0)
    ax.axvspan(1.5, 5.5, color="#9aa7b1", alpha=0.08, zorder=0)
    ax.text(0.5, 1.02, "improved", ha="center", va="bottom", fontsize=9,
            color="#1f6f66", transform=ax.get_xaxis_transform())
    ax.text(3.5, 1.02, "naive baseline still best", ha="center", va="bottom", fontsize=9,
            color="#5b6670", transform=ax.get_xaxis_transform())

    b1 = ax.bar(x - w / 2, naive_p, w, label="naive baseline", color=NAIVE_BAR, zorder=3)
    b2 = ax.bar(x + w / 2, final_p, w, label="final detector", color=FINAL_BAR, zorder=3)
    ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)

    # recall-change callouts for the two series whose recall improved
    for xi in x:
        if naive_hit[xi] != final_hit[xi]:
            ax.annotate(f"recall {naive_hit[xi]}/{n_windows[xi]} to {final_hit[xi]}/{n_windows[xi]}",
                        xy=(xi, max(naive_p[xi], final_p[xi]) + 0.07),
                        ha="center", fontsize=8.5, fontweight="bold", color="#1f6f66")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("precision", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.5, 5.5)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.set_title("One series clearly fixed, one improved, four where the simple detector still wins",
                 fontsize=13, fontweight="bold", loc="left", pad=26)
    _clean_axes(ax)

    fig.tight_layout()
    out = FIG_DIR / "six_series_comparison.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out, naive_p, final_p


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    nyc = run_both("realKnownCause/nyc_taxi.csv")
    out1, naive1, final1 = figure_one(nyc)
    print(f"FIGURE 1  {out1.relative_to(REPO_ROOT)}")
    print(f"  nyc_taxi decomposition used by final detector: {nyc['decomposition']}")
    print(f"  naive : windows {naive1.windows_hit}/{naive1.n_windows}  flagged {naive1.n_flagged}  "
          f"TP {naive1.true_positives}  FP {naive1.false_positives}  "
          f"precision {naive1.precision:.3f}  recall {naive1.recall:.3f}")
    print(f"  final : windows {final1.windows_hit}/{final1.n_windows}  flagged {final1.n_flagged}  "
          f"TP {final1.true_positives}  FP {final1.false_positives}  "
          f"precision {final1.precision:.3f}  recall {final1.recall:.3f}")

    scores = pd.read_csv(FINAL_SCORES)
    out2, naive_p, final_p = figure_two(scores)
    s = scores.set_index("series")
    print(f"\nFIGURE 2  {out2.relative_to(REPO_ROOT)}  (source: {FINAL_SCORES.relative_to(REPO_ROOT)})")
    print(f"  {'series':16s} {'naive P':>9s} {'final P':>9s}  {'naive R':>9s} {'final R':>9s}  decomposition")
    for k in BAR_ORDER:
        r = s.loc[k]
        print(f"  {SHORT_NAME[k]:16s} {r['naive_P']:9.3f} {r['final_P']:9.3f}  "
              f"{r['naive_R']:9.2f} {r['final_R']:9.2f}  {r['decomposition']}")


if __name__ == "__main__":
    main()
