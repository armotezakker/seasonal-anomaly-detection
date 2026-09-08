"""Phase 1 exploration: rank the real NAB series by daily/weekly seasonality and
plot a shortlist so candidate series can be chosen by eye.

Two outputs:

  1. A table over every real series (all categories except the artificial ones):
     median sampling step, span, and the lag-1-day and lag-1-week autocorrelation
     of the series resampled onto its regular grid. High autocorrelation at the
     daily and weekly lags is a cheap proxy for "has a cycle", though it is also
     inflated by slow trend, so the plots are the real check.

  2. For a hand-picked shortlist, a three-panel figure per series (full series
     with labelled windows shaded, a ~12-day zoom, and the median value by hour
     of day with one line per weekday) plus a single overview grid.

The six series chosen from this exploration are listed in
`src.data.nab_loader.CANDIDATE_SERIES`.

Usage:
    python -m src.analysis.explore_seasonality
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.nab_loader import DATA_DIR, load_series, load_windows  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "reports" / "figures"
SCAN_PATH = REPO_ROOT / "data" / "processed" / "seasonality_scan.csv"

# Series plotted in detail during Phase 1. Wider than the final six on purpose:
# the point was to compare and reject.
SHORTLIST = [
    "realKnownCause/nyc_taxi.csv",
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "realAdExchange/exchange-2_cpm_results.csv",
    "realAdExchange/exchange-2_cpc_results.csv",
    "realTraffic/occupancy_t4013.csv",
    "realTraffic/occupancy_6005.csv",
    "realTraffic/speed_7578.csv",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    "realAWSCloudwatch/grok_asg_anomaly.csv",
    "realTweets/Twitter_volume_AMZN.csv",
    "realTweets/Twitter_volume_GOOG.csv",
]


def on_regular_grid(df: pd.DataFrame) -> tuple[pd.Series, pd.Timedelta]:
    """Resample onto the median step and linearly fill gaps up to one hour."""
    s = df.set_index("timestamp")["value"].sort_index()
    step = pd.Timedelta(s.index.to_series().diff().median())
    limit = max(1, int(pd.Timedelta("1h") / step))
    return s.resample(step).mean().interpolate("linear", limit=limit), step


def autocorr_at(x: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation of x with itself shifted by `lag` samples."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if lag >= len(x):
        return np.nan
    a, b = x[:-lag], x[lag:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a, b = a[m], b[m]
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom else np.nan


def scan() -> pd.DataFrame:
    rows = []
    for cat_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        if cat_dir.name.startswith("artificial"):
            continue
        for csv_path in sorted(glob.glob(str(cat_dir / "*.csv"))):
            key = f"{cat_dir.name}/{Path(csv_path).name}"
            g, step = on_regular_grid(load_series(key))
            per_day = int(round(pd.Timedelta("1D") / step))
            span_days = (g.index[-1] - g.index[0]).total_seconds() / 86400
            rows.append({
                "series": key,
                "step": str(step),
                "span_days": round(span_days, 1),
                "n_grid": len(g),
                "acf_1d": round(autocorr_at(g.values, per_day), 3),
                "acf_7d": round(autocorr_at(g.values, per_day * 7), 3) if span_days > 10 else np.nan,
                "n_windows": len(load_windows(key)),
            })
    return pd.DataFrame(rows).sort_values("acf_1d", ascending=False)


def detail_figure(key: str) -> Path:
    g, step = on_regular_grid(load_series(key))
    per_day = int(round(pd.Timedelta("1D") / step))
    windows = load_windows(key)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))

    ax = axes[0]
    ax.plot(g.index, g.values, lw=0.5)
    for start, end in windows:
        ax.axvspan(start, end, color="red", alpha=0.25)
    span_days = (g.index[-1] - g.index[0]).total_seconds() / 86400
    ax.set_title(f"{key}   |   {len(g)} pts @ {step}   |   {span_days:.0f} days   |   "
                 f"{len(windows)} labelled window(s), shaded", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    i0 = int(len(g) * 0.25)
    zoom = g.iloc[i0:i0 + per_day * 12]
    ax.plot(zoom.index, zoom.values, lw=0.8)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.set_title("~12-day zoom (repeating daily shape? weekday vs weekend?)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    frame = g.to_frame("value")
    frame["dow"] = frame.index.dayofweek
    frame["hod"] = frame.index.hour + frame.index.minute / 60
    profile = frame.groupby(["dow", "hod"])["value"].median().unstack("dow")
    for col, name in zip(range(profile.shape[1]),
                         ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        ax.plot(profile.index, profile.iloc[:, col], lw=1, label=name)
    ax.set_xlabel("hour of day", fontsize=8)
    ax.set_title("median value by hour of day, one line per weekday", fontsize=9)
    ax.legend(fontsize=6, ncol=7, loc="upper center")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / (key.replace("/", "__").replace(".csv", "") + ".png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def overview_grid() -> Path:
    fig, axes = plt.subplots(4, 3, figsize=(15, 11))
    for ax, key in zip(axes.flat, SHORTLIST):
        g, _ = on_regular_grid(load_series(key))
        ax.plot(g.index, g.values, lw=0.4)
        for start, end in load_windows(key):
            ax.axvspan(start, end, color="red", alpha=0.25)
        ax.set_title(key, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(SHORTLIST):]:
        ax.axis("off")
    fig.suptitle("NAB candidate series (full length, red = labelled anomaly window)", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "_overview_grid.png"
    fig.savefig(path, dpi=95)
    plt.close(fig)
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    table = scan()
    table.to_csv(SCAN_PATH, index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 60):
        print(table.to_string(index=False))
    print(f"\nwrote {SCAN_PATH.relative_to(REPO_ROOT)}")

    for key in SHORTLIST:
        print("plotted", detail_figure(key).name)
    print("plotted", overview_grid().name)


if __name__ == "__main__":
    main()
