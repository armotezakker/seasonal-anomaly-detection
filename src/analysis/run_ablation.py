"""Phase 4 ablation: two fixes for the Phase 3 failures, tested one at a time.

Phase 3 diagnosed two problems with STL(daily) + one global residual sigma:
  1. the weekly cycle stayed in the residual, so weekends looked anomalous;
  2. one global sigma is too tight during busy or bursty stretches.

Four configurations, same k=3, same time-tolerant scoring, no per-series tuning:

  a  STL (daily)          + global robust sigma     reference, close to Phase 3
  b  MSTL (daily+weekly)  + global robust sigma     weekly-cycle fix alone
  c  STL (daily)          + local MAD sigma         adaptive-scale fix alone
  d  MSTL (daily+weekly)  + local MAD sigma         both fixes together

The local MAD window is tried at one day and at one week; the better one is
chosen once, across all series, and used for c and d in the headline table.

Every decomposition is computed once per series and reused across the scale
variants, so the only thing that changes between a and c (or b and d) is the
residual scale, and the only thing that changes between a and b (or c and d) is
the decomposition.

Usage:
    python -m src.analysis.run_ablation [--k 3.0]
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
from src.detectors.local_scale import threshold_residual  # noqa: E402
from src.detectors.mstl_threshold import mstl_decompose, seasonal_amplitudes  # noqa: E402
from src.detectors.stl_threshold import stl_decompose  # noqa: E402
from src.evaluation.score import EDGE_TOLERANCE_STEPS, score_series  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "reports" / "figures"
SCORES_PATH = REPO_ROOT / "data" / "processed" / "ablation_scores.csv"
AMP_PATH = REPO_ROOT / "data" / "processed" / "mstl_weekly_component.csv"
NAIVE_RAW_PATH = REPO_ROOT / "data" / "processed" / "naive_baseline_scores.csv"

SPOTLIGHT = ["realKnownCause/nyc_taxi.csv", "realTraffic/occupancy_6005.csv"]
CONFIG_LABELS = {
    "a": "a: STL(daily) + global",
    "b": "b: MSTL(daily+weekly) + global",
    "c": "c: STL(daily) + local MAD",
    "d": "d: MSTL(daily+weekly) + local MAD",
}


def weekend_fp_fraction(index: pd.DatetimeIndex, score) -> float:
    if score.false_positives == 0:
        return float("nan")
    dow = index.dayofweek.to_numpy()
    return float((dow[score.fp_mask] >= 5).mean())


def build(k: float) -> dict:
    naive_raw = pd.read_csv(NAIVE_RAW_PATH).set_index("series")
    per_series = {}
    amp_rows = []

    for key in CANDIDATE_SERIES:
        df = load_series(key)
        windows = load_windows(key)
        series, filled, step = resample_to_grid(df, method="linear")
        observed = ~filled.to_numpy()
        daily = int(round(pd.Timedelta("1D") / step))
        weekly = 7 * daily
        edge_tol = EDGE_TOLERANCE_STEPS * step

        stl_d = stl_decompose(series, period=daily, robust=True)
        mstl_d = mstl_decompose(series, periods=(daily, weekly), robust=True)
        amp = seasonal_amplitudes(mstl_d, observed=observed)

        amp_rows.append({
            "series": key,
            "daily_period": daily,
            "weekly_period": weekly,
            "weeks_of_data": round(len(series) / weekly, 1),
            "daily_std": amp[daily]["std"],
            "weekly_std": amp[weekly]["std"],
            "weekly_over_daily_std": amp[weekly]["std"] / amp[daily]["std"],
            "daily_ptp": amp[daily]["ptp"],
            "weekly_ptp": amp[weekly]["ptp"],
            "stl_resid_std_obs": float(np.std(stl_d.residual[observed])),
            "mstl_resid_std_obs": float(np.std(mstl_d.residual[observed])),
        })

        variants = {
            # a, b: the Phase 3 rule (global mean +/- k*std) on each decomposition
            "a": threshold_residual(stl_d.residual, observed, k=k, scale_kind="global_std"),
            "b": threshold_residual(mstl_d.residual, observed, k=k, scale_kind="global_std"),
            # diagnostic only: swap std for a robust global MAD, still not local
            "a_gmad": threshold_residual(stl_d.residual, observed, k=k, scale_kind="global_mad"),
            "b_gmad": threshold_residual(mstl_d.residual, observed, k=k, scale_kind="global_mad"),
            # c, d: local MAD scale, tried at a one-day and a one-week window
            "c_day": threshold_residual(stl_d.residual, observed, k=k, scale_kind="local", window=daily),
            "c_week": threshold_residual(stl_d.residual, observed, k=k, scale_kind="local", window=weekly),
            "d_day": threshold_residual(mstl_d.residual, observed, k=k, scale_kind="local", window=daily),
            "d_week": threshold_residual(mstl_d.residual, observed, k=k, scale_kind="local", window=weekly),
        }
        scores = {
            name: score_series(series.index.values, v.observed_flags, windows, edge_tol, series=key)
            for name, v in variants.items()
        }

        per_series[key] = {
            "series_obj": series, "filled": filled, "windows": windows,
            "daily": daily, "weekly": weekly, "edge_tol": edge_tol,
            "stl_d": stl_d, "mstl_d": mstl_d,
            "variants": variants, "scores": scores,
            "naive_raw": naive_raw.loc[key],
        }

    return {"per_series": per_series, "amp_rows": amp_rows}


def choose_window(per_series: dict) -> str:
    """Pick 'day' or 'week' for the local MAD window, once, across all series.

    Rule: prefer the window with the lower total false-positive count, provided
    it does not lose recalled windows overall.
    """
    tot = {"day": {"fp": 0, "hits": 0, "win": 0}, "week": {"fp": 0, "hits": 0, "win": 0}}
    for info in per_series.values():
        for w in ("day", "week"):
            s = info["scores"][f"c_{w}"]
            tot[w]["fp"] += s.false_positives
            tot[w]["hits"] += s.windows_hit
            tot[w]["win"] += s.n_windows
    print("\nLOCAL MAD WINDOW CHOICE (config c: STL daily + local MAD, day vs week window)")
    print(f"{'series':52s} {'c_day P/R/FP':>22s} {'c_week P/R/FP':>22s}")
    for key, info in per_series.items():
        cd, cw = info["scores"]["c_day"], info["scores"]["c_week"]
        print(f"{key:52s} "
              f"{cd.precision:6.3f}/{cd.recall:4.2f}/{cd.false_positives:4d}   "
              f"{cw.precision:6.3f}/{cw.recall:4.2f}/{cw.false_positives:4d}")
    print(f"{'TOTAL':52s} {'':>7s}FP={tot['day']['fp']:<4d} hits={tot['day']['hits']}/{tot['day']['win']}   "
          f"    FP={tot['week']['fp']:<4d} hits={tot['week']['hits']}/{tot['week']['win']}")
    win = "day" if (tot["day"]["fp"] <= tot["week"]["fp"] and tot["day"]["hits"] >= tot["week"]["hits"]) else "week"
    if tot["week"]["fp"] < tot["day"]["fp"] and tot["week"]["hits"] >= tot["day"]["hits"]:
        win = "week"
    print(f"  -> chosen local MAD window: ONE {win.upper()}")
    return win


def scores_table(per_series: dict, window_choice: str) -> pd.DataFrame:
    rows = []
    for key, info in per_series.items():
        variant_names = {"a": "a", "b": "b", "c": f"c_{window_choice}", "d": f"d_{window_choice}"}
        nr = info["naive_raw"]
        rows.append({
            "series": key, "config": "naive (Phase 2, raw)",
            "precision": nr["precision"], "recall": nr["recall"],
            "false_positives": int(nr["false_positives"]),
        })
        for cfg, vname in variant_names.items():
            sc = info["scores"][vname]
            rows.append({
                "series": key, "config": CONFIG_LABELS[cfg],
                "precision": sc.precision, "recall": sc.recall,
                "false_positives": sc.false_positives,
                "n_flagged": sc.n_flagged, "true_positives": sc.true_positives,
                "windows_hit": sc.windows_hit, "n_windows": sc.n_windows,
                "weekend_fp_frac": weekend_fp_fraction(info["series_obj"].index, sc),
            })
    return pd.DataFrame(rows)


def overlay_d(key, info, window_choice, path):
    series = info["series_obj"]
    filled = info["filled"]
    windows = info["windows"]
    variant = info["variants"][f"d_{window_choice}"]
    score = info["scores"][f"d_{window_choice}"]
    fitted = info["mstl_d"].fitted
    t = series.index.values
    v = series.to_numpy()

    fig, ax = plt.subplots(figsize=(13, 4.5))
    band_lo = fitted + variant.lower
    band_hi = fitted + variant.upper
    ax.fill_between(t, band_lo, band_hi, color="#bbbbbb", alpha=0.5, lw=0,
                    label=f"MSTL fitted +/- {variant.k:g} * local MAD sigma", zorder=1)
    ax.plot(t, v, lw=0.5, color="#3b76af", zorder=2)
    for s, e in windows:
        ax.axvspan(s, e, color="red", alpha=0.18, zorder=0)
    ax.scatter(t[score.fp_mask], v[score.fp_mask], s=14, color="#d62728",
               label=f"false positive ({score.false_positives})", zorder=4)
    ax.scatter(t[score.tp_mask], v[score.tp_mask], s=14, color="#2ca02c",
               label=f"true positive ({score.true_positives})", zorder=4)
    prec = f"{score.precision:.3f}" if np.isfinite(score.precision) else "n/a"
    rec = f"{score.recall:.2f}" if np.isfinite(score.recall) else "n/a"
    ax.set_title(f"{key}   config d: MSTL(daily+weekly) + local MAD (1 {window_choice})   "
                 f"flagged {score.n_flagged}   precision {prec}   recall {rec}   "
                 f"windows {score.windows_hit}/{score.n_windows}", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def config_ladder_plot(key, info, window_choice, path):
    series = info["series_obj"]
    windows = info["windows"]
    t = series.index.values
    v = series.to_numpy()
    order = [("a", info["stl_d"].fitted, "a"),
             ("b", info["mstl_d"].fitted, "b"),
             ("c", info["stl_d"].fitted, f"c_{window_choice}"),
             ("d", info["mstl_d"].fitted, f"d_{window_choice}")]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax, (cfg, fitted, vname) in zip(axes, order):
        variant = info["variants"][vname]
        score = info["scores"][vname]
        ax.fill_between(t, fitted + variant.lower, fitted + variant.upper,
                        color="#bbbbbb", alpha=0.5, lw=0)
        ax.plot(t, v, lw=0.4, color="#3b76af")
        for s, e in windows:
            ax.axvspan(s, e, color="red", alpha=0.16)
        ax.scatter(t[score.fp_mask], v[score.fp_mask], s=9, color="#d62728")
        ax.scatter(t[score.tp_mask], v[score.tp_mask], s=9, color="#2ca02c")
        prec = f"{score.precision:.3f}" if np.isfinite(score.precision) else "n/a"
        rec = f"{score.recall:.2f}" if np.isfinite(score.recall) else "n/a"
        wk = weekend_fp_fraction(series.index, score)
        wk_s = f"weekend FP {wk:.0%}" if np.isfinite(wk) else "weekend FP n/a"
        ax.set_title(f"{CONFIG_LABELS[cfg]}   P {prec}  R {rec}  FP {score.false_positives}  "
                     f"({score.windows_hit}/{score.n_windows} windows, {wk_s})", fontsize=8)
        ax.tick_params(labelsize=6)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(f"{key}   ablation ladder  (grey = expected band, green = TP, red dot = FP)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=3.0)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    built = build(args.k)
    per_series = built["per_series"]

    amp = pd.DataFrame(built["amp_rows"])
    amp.to_csv(AMP_PATH, index=False)
    print("=" * 110)
    print("1. MSTL WEEKLY COMPONENT: is it real and non-trivial?")
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", lambda x: f"{x:.4g}"):
        print(amp[["series", "daily_period", "weekly_period", "weeks_of_data",
                   "daily_std", "weekly_std", "weekly_over_daily_std",
                   "stl_resid_std_obs", "mstl_resid_std_obs"]].to_string(index=False))
    print("  (weekly_over_daily_std is the weekly component's amplitude as a fraction of the daily one;")
    print("   mstl_resid_std_obs < stl_resid_std_obs means MSTL pulled real structure out of the residual)")

    window_choice = choose_window(per_series)

    table = scores_table(per_series, window_choice)
    table.to_csv(SCORES_PATH, index=False)

    print("\n" + "=" * 110)
    print("3. FOUR-CONFIGURATION COMPARISON (naive baseline shown for reference)")
    for key in CANDIDATE_SERIES:
        sub = table[table["series"] == key]
        print(f"\n{key}")
        for _, r in sub.iterrows():
            extra = ""
            if np.isfinite(r.get("weekend_fp_frac", np.nan)):
                extra = f"   weekendFP={r['weekend_fp_frac']:.0%}"
            fp = int(r["false_positives"])
            print(f"   {r['config']:34s} P={r['precision']:.3f}  R={r['recall']:.3f}  FP={fp:<4d}{extra}")

    print("\n" + "=" * 110)
    print("4a. WEEKEND FALSE POSITIVES: does adding the weekly period (a -> b) cut them?")
    print(f"{'series':52s} {'a FP (wknd%)':>16s} {'b FP (wknd%)':>16s} {'d FP (wknd%)':>16s}")
    for key, info in per_series.items():
        a, b = info["scores"]["a"], info["scores"]["b"]
        d = info["scores"][f"d_{window_choice}"]
        def cell(s):
            wk = weekend_fp_fraction(info["series_obj"].index, s)
            return f"{s.false_positives:4d} ({wk:.0%})" if np.isfinite(wk) else f"{s.false_positives:4d} (n/a)"
        print(f"{key:52s} {cell(a):>16s} {cell(b):>16s} {cell(d):>16s}")

    print("\n4b. OVER-ALERTING CASES: does the local MAD scale (a -> c) cut false positives?")
    print(f"{'series':52s} {'a P/FP':>14s} {'c P/FP':>14s} {'d P/FP':>14s}")
    for key, info in per_series.items():
        a = info["scores"]["a"]
        c = info["scores"][f"c_{window_choice}"]
        d = info["scores"][f"d_{window_choice}"]
        print(f"{key:52s} {a.precision:6.3f}/{a.false_positives:<4d} "
              f"{c.precision:6.3f}/{c.false_positives:<4d} "
              f"{d.precision:6.3f}/{d.false_positives:<4d}")

    print("\n4c. HOW MUCH OF a -> c is 'robust estimator' vs 'local'?")
    print("    a = global std   |   a_gmad = global 1.4826*MAD   |   c = local 1.4826*MAD")
    print(f"{'series':52s} {'a P/FP':>13s} {'a_gmad P/FP':>15s} {'c P/FP':>13s}")
    for key, info in per_series.items():
        a = info["scores"]["a"]
        ag = info["scores"]["a_gmad"]
        c = info["scores"][f"c_{window_choice}"]
        print(f"{key:52s} {a.precision:6.3f}/{a.false_positives:<4d} "
              f"{ag.precision:6.3f}/{ag.false_positives:<5d} "
              f"{c.precision:6.3f}/{c.false_positives:<4d}")

    print("\n" + "=" * 110)
    print("5. IS CONFIG d STILL WORSE THAN THE NAIVE BASELINE ANYWHERE?")
    for key, info in per_series.items():
        d = info["scores"][f"d_{window_choice}"]
        nr = info["naive_raw"]
        dp = d.precision - nr["precision"]
        dr = d.recall - nr["recall"]
        dfp = d.false_positives - int(nr["false_positives"])
        worse = []
        if dp < -1e-9:
            worse.append(f"precision {nr['precision']:.3f} -> {d.precision:.3f}")
        if dr < -1e-9:
            worse.append(f"recall {nr['recall']:.3f} -> {d.recall:.3f}")
        if dfp > 0:
            worse.append(f"FP {int(nr['false_positives'])} -> {d.false_positives}")
        verdict = "WORSE on: " + "; ".join(worse) if worse else "not worse (>= naive on P, R and FP)"
        print(f"   {key:52s} {verdict}")

    print("\n" + "=" * 110)
    print("6. OVERLAY PLOTS (config d, and the a->d ladder) for the spotlight series")
    for key in SPOTLIGHT:
        info = per_series[key]
        stub = key.replace("/", "__").replace(".csv", "")
        p1 = FIG_DIR / f"ablation_d__{stub}.png"
        p2 = FIG_DIR / f"ablation_ladder__{stub}.png"
        overlay_d(key, info, window_choice, p1)
        config_ladder_plot(key, info, window_choice, p2)
        print(f"   wrote {p1.name} and {p2.name}")

    print(f"\nwrote {SCORES_PATH.relative_to(REPO_ROOT)} and {AMP_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
