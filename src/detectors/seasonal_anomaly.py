"""The recommended detector, assembled from what the earlier phases established.

Two rules, both fixed and applied identically to every series:

1. Seasonal decomposition by data sufficiency.
   Fit MSTL with a daily and a weekly period only when the series has at least
   `WEEKLY_CYCLE_MIN` full weekly cycles. Otherwise fit daily-only STL.
   Phase 4 showed that MSTL with ~2 weekly cycles fits a one-off anomaly as
   part of the weekly seasonal component (the labelled window on occupancy_6005
   came back with a residual of exactly 0.0) and recall collapsed. A LOESS
   seasonal smoother estimates each weekly phase by smoothing across cycles, so
   with N cycles each seasonal value rests on N points and a single bad week is
   a 1/N contribution. Ten cycles keeps one anomalous week at or below 10% of
   every seasonal estimate, which the robust weighting can then reject. At 7.9
   cycles (Twitter_AMZN) MSTL already gave no residual-variance benefit, and at
   2.3 (the occupancy pair) it destroyed detection. The cutoff is not sharp;
   anything in roughly 10 to 12 routes the six series the same way. Eight is too
   low: it would send exchange-2_cpm (9.8 cycles) to MSTL, where it scores
   slightly worse than under STL.

2. Residual threshold with a global robust scale.
   Flag a residual point when it is more than `K` mean-standard-deviations from
   the mean residual (the Phase 3 rule). Phase 5 tested a local rolling-MAD
   scale at k = 6, 7, 8, 9 to correct for MAD sitting well below the standard
   deviation on these heavy-tailed residuals; even at its best fixed k it was
   still worse in aggregate than the global standard-deviation scale, so the
   global scale is what ships.

`K` and `WEEKLY_CYCLE_MIN` are module constants, not per-series parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.detectors.local_scale import threshold_residual
from src.detectors.mstl_threshold import mstl_decompose
from src.detectors.stl_threshold import stl_decompose

K = 3.0
WEEKLY_CYCLE_MIN = 10.0
SCALE_KIND = "global_std"


@dataclass
class FinalResult:
    decomposition_used: str  # "mstl_daily_weekly" or "stl_daily"
    daily_period: int
    weekly_period: int
    weekly_cycles_available: float
    k: float
    trend: np.ndarray = field(repr=False)
    seasonal_total: np.ndarray = field(repr=False)
    fitted: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)
    centre: np.ndarray = field(repr=False)
    scale: np.ndarray = field(repr=False)
    flags: np.ndarray = field(repr=False)
    observed_flags: np.ndarray = field(repr=False)


def detect(
    series: pd.Series,
    step: pd.Timedelta,
    filled=None,
    k: float = K,
    weekly_cycle_min: float = WEEKLY_CYCLE_MIN,
    scale_kind: str = SCALE_KIND,
) -> FinalResult:
    """Decompose by the sufficiency rule, then threshold the residual.

    Parameters
    ----------
    series : pandas.Series
        Float series on a complete DatetimeIndex at a fixed frequency, no NaN
        (from `nab_loader.resample_to_grid`).
    step : pandas.Timedelta
        The grid sampling interval, used to size the daily and weekly periods.
    filled : bool array or None
        True where a slot was interpolated. Kept out of the residual statistics
        and never flagged.
    """
    if filled is None:
        observed = np.ones(len(series), dtype=bool)
    else:
        observed = ~np.asarray(filled, dtype=bool)

    daily = int(round(pd.Timedelta("1D") / step))
    weekly = 7 * daily
    weekly_cycles = len(series) / weekly

    if weekly_cycles >= weekly_cycle_min:
        decomp = mstl_decompose(series, periods=(daily, weekly), robust=True)
        seasonal_total = decomp.seasonal_total
        which = "mstl_daily_weekly"
    else:
        decomp = stl_decompose(series, period=daily, robust=True)
        seasonal_total = decomp.seasonal
        which = "stl_daily"

    # a local scale needs a window; use one day, matching the Phase 5 sweep
    window = daily if scale_kind == "local" else None
    flags = threshold_residual(
        decomp.residual, observed=observed, k=k, scale_kind=scale_kind, window=window,
    )

    return FinalResult(
        decomposition_used=which,
        daily_period=daily,
        weekly_period=weekly,
        weekly_cycles_available=float(weekly_cycles),
        k=float(k),
        trend=decomp.trend,
        seasonal_total=seasonal_total,
        fitted=decomp.fitted,
        residual=decomp.residual,
        centre=flags.centre,
        scale=flags.scale,
        flags=flags.flags,
        observed_flags=flags.observed_flags,
    )
