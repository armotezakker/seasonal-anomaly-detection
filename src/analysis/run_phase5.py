"""Phase 5: recalibrate the local MAD k, set the weekly-cycle sufficiency rule,
build the final recommended detector, and compare it against the naive baseline
and against Phase 4 config b.

Steps:
  1. Local MAD k sweep at k = 6, 7, 8, 9 (STL-daily residual, one-day window,
     uniform k across all six series). Full table per k, then a pick.
  2. Weekly-cycle sufficiency threshold. Route each series to MSTL or STL-daily
     by its number of full weekly cycles; show that 8, 10 and 12 differ only for
     one borderline series and that 10 routes it the better way.
  3. Final detector = src.detectors.seasonal_anomaly.detect (decomposition by
     the rule, residual scale by step 1's winner).
  4. Final vs naive (Phase 2) vs config b (MSTL + global), all six series.
  5. Overlay plots for nyc_taxi and occupancy_6005 under the final detector.

Usage:
    python -m src.analysis.run_phase5
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
from src.detectors import seasonal_anomaly  # noqa: E402
from src.detectors.local_scale import threshold_residual  # noqa: E402
from src.detectors.stl_threshold import stl_decompose  # noqa: E402
from src.evaluation.score import EDGE_TOLERANCE_STEPS, score_series  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "reports" / "figures"
FINAL_SCORES_PATH = REPO_ROOT / "data" / "processed" / "final_detector_scores.csv"
KSWEEP_PATH = REPO_ROOT / "data" / "processed" / "phase5_k_sweep.csv"
NAIVE_PATH = REPO_ROOT / "data" / "processed" / "naive_baseline_scores.csv"
ABLATION_PATH = REPO_ROOT / "data" / "processed" / "ablation_scores.csv"

K_GRID = [6, 7, 8, 9]
WEEKLY_CUTOFFS = [8, 10, 12]
SPOTLIGHT = ["realKnownCause/nyc_taxi.csv", "realTraffic/occupancy_6005.csv"]


def load_grids():
    grids = {}
    for key in CANDIDATE_SERIES:
        df = load_series(key)
        series, filled, step = resample_to_grid(df, method="linear")
        grids[key] = {
            "series": series,
            "filled": filled,
            "step": step,
            "observed": ~filled.to_numpy(),
            "windows": load_windows(key),
            "edge_tol": EDGE_TOLERANCE_STEPS * step,
            "daily": int(round(pd.Timedelta("1D") / step)),
            "weekly": 7 * int(round(pd.Timedelta("1D") / step)),
            "weekly_cycles": len(series) / (7 * int(round(pd.Timedelta("1D") / step))),
        }
    return grids


def k_sweep(grids: dict) -> tuple[pd.DataFrame, int]:
    # STL-daily residual per series, computed once
    stl_resid = {}
    for key, g in grids.items():
        stl_resid[key] = stl_decompose(g["series"], period=g["daily"], robust=True).residual

    rows = []
    # reference: global std, k=3 (Phase 4 config a, same decomposition)
    for key, g in grids.items():
        ref = threshold_residual(stl_resid[key], g["observed"], k=3.0, scale_kind="global_std")
        sc = score_series(g["series"].index.values, ref.observed_flags, g["windows"], g["edge_tol"], key)
        rows.append(dict(scale="global std", k=3, series=key, precision=sc.precision,
                         recall=sc.recall, false_positives=sc.false_positives,
                         windows_hit=sc.windows_hit, n_windows=sc.n_windows))
    for k in K_GRID:
        for key, g in grids.items():
            r = threshold_residual(stl_resid[key], g["observed"], k=float(k),
                                   scale_kind="local", window=g["daily"])
            sc = score_series(g["series"].index.values, r.observed_flags, g["windows"], g["edge_tol"], key)
            rows.append(dict(scale="local MAD", k=k, series=key, precision=sc.precision,
                             recall=sc.recall, false_positives=sc.false_positives,
                             windows_hit=sc.windows_hit, n_windows=sc.n_windows))
    table = pd.DataFrame(rows)

    print("=" * 110)
    print("1. LOCAL MAD k RECALIBRATION (STL-daily residual, one-day window, uniform k)")
    for k in K_GRID:
        sub = table[(table["scale"] == "local MAD") & (table["k"] == k)]
        print(f"\n  local MAD, k={k}")
        print(f"  {'series':52s} {'P':>7s} {'R':>6s} {'FP':>6s}   windows")
        for _, r in sub.iterrows():
            print(f"  {r['series']:52s} {r['precision']:7.3f} {r['recall']:6.3f} {r['false_positives']:6d}   "
                  f"{r['windows_hit']}/{r['n_windows']}")
        print(f"  {'AGGREGATE':52s} {sub['precision'].mean():7.3f} {sub['recall'].mean():6.3f} "
              f"{int(sub['false_positives'].sum()):6d}   {int(sub['windows_hit'].sum())}/{int(sub['n_windows'].sum())}")

    ref = table[table["scale"] == "global std"]
    print(f"\n  reference: global std, k=3 (same decomposition)")
    print(f"  {'AGGREGATE':52s} {ref['precision'].mean():7.3f} {ref['recall'].mean():6.3f} "
          f"{int(ref['false_positives'].sum()):6d}   {int(ref['windows_hit'].sum())}/{int(ref['n_windows'].sum())}")

    # pick: among the k grid, most recalled windows, then fewest false positives, then best mean precision
    best_k, best_key = None, None
    for k in K_GRID:
        sub = table[(table["scale"] == "local MAD") & (table["k"] == k)]
        key_tuple = (int(sub["windows_hit"].sum()), -int(sub["false_positives"].sum()), sub["precision"].mean())
        if best_key is None or key_tuple > best_key:
            best_key, best_k = key_tuple, k
    ref_fp = int(ref["false_positives"].sum())
    best_fp = int(table[(table["scale"] == "local MAD") & (table["k"] == best_k)]["false_positives"].sum())
    print(f"\n  -> best local MAD k among {K_GRID}: k={best_k} "
          f"(total FP {best_fp} vs global-std-k3 total FP {ref_fp})")
    winner = "local_mad_k%d" % best_k if best_fp < ref_fp else "global_std_k3"
    print(f"  -> scale that ships in the final detector: "
          f"{'local MAD k=%d' % best_k if winner.startswith('local') else 'global std, k=3'}")
    table.to_csv(KSWEEP_PATH, index=False)
    return table, best_k, winner


def sufficiency(grids: dict) -> float:
    print("\n" + "=" * 110)
    print("2. WEEKLY-CYCLE SUFFICIENCY THRESHOLD")
    print(f"  {'series':52s} {'weeks':>7s}   route at cutoff 8 / 10 / 12")
    for key, g in grids.items():
        w = g["weekly_cycles"]
        routes = " / ".join("MSTL" if w >= c else "STL " for c in WEEKLY_CUTOFFS)
        print(f"  {key:52s} {w:7.2f}   {routes}")
    print("\n  Only exchange-2_cpm (9.81 weeks) changes across 8 / 10 / 12: cutoff 8 -> MSTL, 10 and 12 -> STL.")
    print("  Phase 4: exchange-2_cpm scored better under STL+global (P 0.206, FP 27) than MSTL+global")
    print("  (P 0.162, FP 31), so routing it to STL is correct. Cutoff 10 (through 12) does that; 8 does not.")
    print("  -> weekly-cycle sufficiency threshold: 10 full weekly cycles")
    excluded = [k for k, g in grids.items() if g["weekly_cycles"] < 10]
    print("  -> routed to daily-only STL (below 10 cycles): ")
    for k in excluded:
        print(f"       {k}  ({grids[k]['weekly_cycles']:.2f} weeks)")
    occ = [k for k in excluded if "occupancy" in k]
    print(f"  -> occupancy pair excluded from MSTL: {occ == [k for k in grids if 'occupancy' in k]}")
    return 10.0


def final_detector(grids: dict, scale_kind: str, k: float) -> pd.DataFrame:
    naive = pd.read_csv(NAIVE_PATH).set_index("series")
    abl = pd.read_csv(ABLATION_PATH)
    config_b = abl[abl["config"].str.startswith("b:")].set_index("series")

    rows = []
    results = {}
    for key, g in grids.items():
        res = seasonal_anomaly.detect(g["series"], g["step"], filled=g["filled"],
                                      k=k, weekly_cycle_min=10.0, scale_kind=scale_kind)
        sc = score_series(g["series"].index.values, res.observed_flags, g["windows"], g["edge_tol"], key)
        results[key] = (res, sc)
        nr, b = naive.loc[key], config_b.loc[key]
        rows.append({
            "series": key,
            "decomposition": res.decomposition_used,
            "weekly_cycles": round(res.weekly_cycles_available, 1),
            "final_P": sc.precision, "final_R": sc.recall, "final_FP": sc.false_positives,
            "final_windows": f"{sc.windows_hit}/{sc.n_windows}",
            "naive_P": nr["precision"], "naive_R": nr["recall"], "naive_FP": int(nr["false_positives"]),
            "b_P": b["precision"], "b_R": b["recall"], "b_FP": int(b["false_positives"]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(FINAL_SCORES_PATH, index=False)

    print("\n" + "=" * 110)
    print("3 & 4. FINAL DETECTOR vs NAIVE (Phase 2) vs CONFIG b (MSTL + global)")
    for _, r in table.iterrows():
        print(f"\n{r['series']}   [{r['decomposition']}, {r['weekly_cycles']} weekly cycles]")
        print(f"   naive   P={r['naive_P']:.3f}  R={r['naive_R']:.3f}  FP={r['naive_FP']}")
        print(f"   config b P={r['b_P']:.3f}  R={r['b_R']:.3f}  FP={r['b_FP']}")
        print(f"   FINAL   P={r['final_P']:.3f}  R={r['final_R']:.3f}  FP={r['final_FP']}  "
              f"windows {r['final_windows']}")

    print("\n" + "-" * 110)
    print("PLAIN VERDICT PER SERIES (final vs naive)")
    for _, r in table.iterrows():
        dP = r["final_P"] - r["naive_P"]
        dR = r["final_R"] - r["naive_R"]
        dFP = r["final_FP"] - r["naive_FP"]
        beats = (dP >= -1e-9 and dR >= -1e-9 and dFP <= 0)
        strictly_beats = beats and (dP > 1e-9 or dR > 1e-9 or dFP < 0)
        if strictly_beats:
            tag = "SOLVED / beats naive"
        elif dR > 1e-9 and dP < -1e-9:
            tag = "IMPROVED not solved (recall up, precision below naive)"
        elif abs(dP) < 1e-9 and abs(dR) < 1e-9 and dFP == 0:
            tag = "matches naive"
        else:
            tag = "NOT beaten: naive still better"
        print(f"   {r['series']:52s} dP={dP:+.3f} dR={dR:+.3f} dFP={dFP:+d}   -> {tag}")

    return table, results


def overlay(key, g, res, sc, path):
    series = g["series"]
    t = series.index.values
    v = series.to_numpy()
    lo = res.fitted + res.centre - res.k * res.scale
    hi = res.fitted + res.centre + res.k * res.scale
    fp = sc.fp_mask
    tp = sc.tp_mask

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.fill_between(t, lo, hi, color="#bbbbbb", alpha=0.5, lw=0,
                    label=f"expected band (fitted +/- {res.k:g} sigma)", zorder=1)
    ax.plot(t, v, lw=0.5, color="#3b76af", zorder=2)
    for s, e in g["windows"]:
        ax.axvspan(s, e, color="red", alpha=0.18, zorder=0)
    ax.scatter(t[fp], v[fp], s=14, color="#d62728", label=f"false positive ({sc.false_positives})", zorder=4)
    ax.scatter(t[tp], v[tp], s=14, color="#2ca02c", label=f"true positive ({sc.true_positives})", zorder=4)
    prec = f"{sc.precision:.3f}" if np.isfinite(sc.precision) else "n/a"
    rec = f"{sc.recall:.2f}" if np.isfinite(sc.recall) else "n/a"
    ax.set_title(f"{key}   FINAL detector [{res.decomposition_used}]   "
                 f"flagged {sc.n_flagged}   precision {prec}   recall {rec}   "
                 f"windows {sc.windows_hit}/{sc.n_windows}", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    grids = load_grids()
    _, best_k, winner = k_sweep(grids)
    sufficiency(grids)

    scale_kind = "global_std"  # set from the k-sweep verdict below
    if winner.startswith("local_mad"):
        scale_kind = "local"
        print(f"\n(NOTE: k-sweep selected local MAD k={best_k}; final detector uses local scale.)")
    k_final = 3.0 if scale_kind == "global_std" else float(best_k)

    table, results = final_detector(grids, scale_kind=scale_kind, k=k_final)

    print("\n" + "=" * 110)
    print("5. FINAL OVERLAY PLOTS")
    for key in SPOTLIGHT:
        res, sc = results[key]
        stub = key.replace("/", "__").replace(".csv", "")
        p = FIG_DIR / f"final__{stub}.png"
        overlay(key, grids[key], res, sc, p)
        print(f"   wrote {p.name}")

    print(f"\nwrote {FINAL_SCORES_PATH.relative_to(REPO_ROOT)}, {KSWEEP_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
