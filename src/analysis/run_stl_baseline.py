"""Run the STL residual-threshold detector on the six Phase 1 candidate series.

Mirrors `run_naive_baseline.py`: same six series, same k, same time-tolerant
scoring, one overlay plot per series. The difference is that the threshold band
is not flat. It is `trend + seasonal +/- k * resid_std`, so it follows the daily
cycle, and a point is flagged only when it leaves that moving band.

For `ambient_temperature_system_failure` it also prints a check that STL's trend
component has absorbed the multi-month drift and the residual no longer carries
it, since that is the reason the series was chosen.

Usage:
    python -m src.analysis.run_stl_baseline [--k 3.0] [--fill linear|ffill]
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
from src.detectors import stl_threshold  # noqa: E402
from src.evaluation.score import EDGE_TOLERANCE_STEPS, score_series  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "reports" / "figures"
SCORES_PATH = REPO_ROOT / "data" / "processed" / "stl_baseline_scores.csv"

TREND_CHECK_SERIES = "realKnownCause/ambient_temperature_system_failure.csv"


def daily_period(step: pd.Timedelta) -> int:
    return int(round(pd.Timedelta("1D") / step))


def linfit_slope_per_day(t_index: pd.DatetimeIndex, y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on time (per day) and the fraction of variance it explains."""
    x = (t_index - t_index[0]).total_seconds().to_numpy() / 86400.0
    m = np.isfinite(y)
    x, y = x[m], y[m]
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return float(a), float(r2)


def trend_absorption_report(key, series, filled, result) -> None:
    obs = ~filled.to_numpy()
    idx = series.index
    raw = series.to_numpy()
    detrended_only_seasonal = raw - result.seasonal  # what the naive residual would still contain

    raw_slope, raw_r2 = linfit_slope_per_day(idx[obs], raw[obs])
    des_slope, des_r2 = linfit_slope_per_day(idx[obs], detrended_only_seasonal[obs])
    res_slope, res_r2 = linfit_slope_per_day(idx[obs], result.residual[obs])

    span_days = (idx[-1] - idx[0]).total_seconds() / 86400
    print("-" * 100)
    print(f"TREND ABSORPTION CHECK  {key}")
    print(f"  span {span_days:.0f} days, period {result.period}")
    print(f"  STL trend component range : {np.nanmin(result.trend):.2f} .. {np.nanmax(result.trend):.2f}  "
          f"(spread {np.nanmax(result.trend) - np.nanmin(result.trend):.2f})")
    print(f"  raw series linear slope   : {raw_slope:+.4f} per day   (R^2 {raw_r2:.3f})")
    print(f"  seasonal-removed only     : {des_slope:+.4f} per day   (R^2 {des_r2:.3f})   "
          f"<- what the naive baseline still sees")
    print(f"  STL residual linear slope : {res_slope:+.4f} per day   (R^2 {res_r2:.4f})   "
          f"<- should be near zero")
    print(f"  std raw {np.nanstd(raw[obs]):.3f}   std seasonal-removed {np.nanstd(detrended_only_seasonal[obs]):.3f}   "
          f"std STL residual {np.nanstd(result.residual[obs]):.3f}")
    drift_left = abs(res_slope * span_days)
    drift_raw = abs(raw_slope * span_days)
    print(f"  drift over the whole span : raw {drift_raw:.2f} units, left in STL residual {drift_left:.2f} units "
          f"({100 * drift_left / drift_raw:.1f}% of the raw drift)")
    verdict = "ABSORBED" if drift_raw > 0 and drift_left / drift_raw < 0.1 and res_r2 < 0.02 else "NOT fully absorbed"
    print(f"  verdict: trend {verdict}")


def overlay_plot(key, series, filled, windows, result, score, edge_tol, path):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    t = series.index.values
    v = series.to_numpy()
    band_lo = result.fitted + result.lower_resid
    band_hi = result.fitted + result.upper_resid

    ax.fill_between(t, band_lo, band_hi, color="#bbbbbb", alpha=0.45, lw=0,
                    label=f"trend + seasonal +/- {result.k:g}*resid sd", zorder=1)
    ax.plot(t, v, lw=0.5, color="#3b76af", zorder=2)
    for start, end in windows:
        ax.axvspan(start, end, color="red", alpha=0.18, zorder=0)

    fp = score.fp_mask
    tp = score.tp_mask
    ax.scatter(t[fp], v[fp], s=14, color="#d62728", label=f"false positive ({fp.sum()})", zorder=4)
    ax.scatter(t[tp], v[tp], s=14, color="#2ca02c", label=f"true positive ({tp.sum()})", zorder=4)

    prec = f"{score.precision:.3f}" if np.isfinite(score.precision) else "n/a"
    rec = f"{score.recall:.2f}" if np.isfinite(score.recall) else "n/a"
    ax.set_title(
        f"{key}   STL residual threshold k={result.k:g}, period={result.period}   "
        f"flagged {score.n_flagged}   precision {prec}   recall {rec}   "
        f"windows hit {score.windows_hit}/{score.n_windows}   "
        f"(grid {100 * filled.mean():.0f}% filled; band = expected normal)",
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
    parser.add_argument("--k", type=float, default=stl_threshold.DEFAULT_K)
    parser.add_argument("--fill", choices=["linear", "ffill"], default="linear")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    grid_specs = []
    for key in CANDIDATE_SERIES:
        df = load_series(key)
        windows = load_windows(key)
        series, filled, step = resample_to_grid(df, method=args.fill)
        period = daily_period(step)
        edge_tol = EDGE_TOLERANCE_STEPS * step

        result = stl_threshold.decompose_and_detect(series, period=period, filled=filled, k=args.k)
        score = score_series(series.index.values, result.observed_flags, windows, edge_tol, series=key)

        fig_path = FIG_DIR / ("stl_baseline__" + key.replace("/", "__").replace(".csv", "") + ".png")
        overlay_plot(key, series, filled, windows, result, score, edge_tol, fig_path)
        grid_specs.append((key, series, filled, windows, result, score))

        row = score.as_row()
        row.update({
            "sampling_step": str(step),
            "grid_len": len(series),
            "pct_filled": round(100 * float(filled.mean()), 1),
            "stl_period": period,
            "k": args.k,
            "resid_mean": result.resid_mean,
            "resid_std": result.resid_std,
            "fill_method": args.fill,
            "fig": fig_path.name,
        })
        rows.append(row)

        print("=" * 100)
        print(f"{key}")
        print(f"  grid: {len(series)} slots @ {step}   filled by {args.fill}: "
              f"{int(filled.sum())} ({100 * filled.mean():.1f}%)   STL period: {period} "
              f"({len(series) / period:.1f} cycles)")
        print(f"  residual mean={result.resid_mean:.4g}  std={result.resid_std:.4g}  "
              f"flag when |residual| > {args.k:g} * {result.resid_std:.4g} = {args.k * result.resid_std:.4g}")
        print(f"  FLAGGED (observed slots): {score.n_flagged}   "
              f"windows hit: {score.windows_hit}/{score.n_windows}")
        print(f"  true positives:  {score.true_positives}")
        print(f"  false positives: {score.false_positives}")
        prec = f"{score.precision:.3f}" if np.isfinite(score.precision) else "n/a (nothing flagged)"
        rec = f"{score.recall:.3f}" if np.isfinite(score.recall) else "n/a (no windows)"
        print(f"  PRECISION={prec}   RECALL={rec}   FALSE_POSITIVES={score.false_positives}")

        if key == TREND_CHECK_SERIES:
            trend_absorption_report(key, series, filled, result)

    scores = pd.DataFrame(rows)
    scores.to_csv(SCORES_PATH, index=False)

    print("\n" + "=" * 100)
    print("PER-SERIES SUMMARY, STL detector (no aggregation)")
    cols = ["series", "grid_len", "pct_filled", "stl_period", "n_flagged", "n_windows",
            "windows_hit", "true_positives", "false_positives", "precision", "recall"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda x: f"{x:.3f}"):
        print(scores[cols].to_string(index=False))
    print(f"\nwrote {SCORES_PATH.relative_to(REPO_ROOT)}")

    n = len(grid_specs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n))
    for ax, (key, series, filled, windows, result, score) in zip(np.atleast_1d(axes), grid_specs):
        t = series.index.values
        v = series.to_numpy()
        ax.fill_between(t, result.fitted + result.lower_resid, result.fitted + result.upper_resid,
                        color="#bbbbbb", alpha=0.45, lw=0)
        ax.plot(t, v, lw=0.4, color="#3b76af")
        for start, end in windows:
            ax.axvspan(start, end, color="red", alpha=0.18)
        ax.scatter(t[score.fp_mask], v[score.fp_mask], s=8, color="#d62728")
        ax.scatter(t[score.tp_mask], v[score.tp_mask], s=8, color="#2ca02c")
        prec = f"{score.precision:.3f}" if np.isfinite(score.precision) else "n/a"
        rec = f"{score.recall:.2f}" if np.isfinite(score.recall) else "n/a"
        ax.set_title(f"{key}   flagged {score.n_flagged}   FP {score.false_positives}   "
                     f"precision {prec}   recall {rec}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(f"STL residual-threshold baseline, k={args.k:g}  "
                 f"(grey band = trend + seasonal +/- k*resid sd, green = TP, red dot = FP)", fontsize=10)
    fig.tight_layout()
    grid_path = FIG_DIR / "stl_baseline__overview_grid.png"
    fig.savefig(grid_path, dpi=95)
    plt.close(fig)
    print(f"wrote {grid_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
