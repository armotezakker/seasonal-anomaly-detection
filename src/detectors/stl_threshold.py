"""Seasonal-trend-aware detector: STL decomposition, then k-sigma on the residual.

The series is split by STL (seasonal-trend decomposition using LOESS) into

    observed = trend + seasonal + residual

The trend carries slow drift, the seasonal carries the repeating daily shape,
and the residual is what is left. The same rule as the naive baseline is then
applied, but to the residual instead of the raw value: flag any point whose
residual is more than `k` standard deviations from the mean residual.

Everything the naive baseline gets wrong on a cyclic series comes from measuring
"far from normal" against a single global mean. Here "normal" is the trend plus
the seasonal shape at that point in the cycle, so a recurring daily peak has a
small residual and is not flagged, while a value that is unusual *for its time of
day* stands out.

Constants held fixed across every series (no per-series tuning):
  * k = 3, same as the naive baseline.
  * robust = True, so a large anomaly is downweighted while STL fits the trend
    and seasonal and does not drag them toward itself.
  * the seasonal smoother length is left at statsmodels' default (7).

The residual mean and standard deviation are computed from observed grid slots
only. Slots that were filled by resampling carry interpolated values whose
residual is near zero by construction, and including them would shrink the
standard deviation and make the threshold too tight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

DEFAULT_K = 3.0
ROBUST = True


@dataclass
class STLThresholdResult:
    period: int
    k: float
    robust: bool
    trend: np.ndarray = field(repr=False)
    seasonal: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)
    fitted: np.ndarray = field(repr=False)  # trend + seasonal, the "expected normal" curve
    resid_mean: float
    resid_std: float
    flags: np.ndarray = field(repr=False)  # |resid - mean| > k * std, over the whole grid
    observed_flags: np.ndarray = field(repr=False)  # flags with filled slots forced to False

    @property
    def upper_resid(self) -> float:
        return self.resid_mean + self.k * self.resid_std

    @property
    def lower_resid(self) -> float:
        return self.resid_mean - self.k * self.resid_std


def decompose_and_detect(
    series: pd.Series,
    period: int,
    filled: pd.Series | np.ndarray | None = None,
    k: float = DEFAULT_K,
    robust: bool = ROBUST,
) -> STLThresholdResult:
    """Decompose `series` with STL and flag residual outliers.

    Parameters
    ----------
    series : pandas.Series
        Float series on a complete DatetimeIndex at a fixed frequency, no NaN.
        Use `nab_loader.resample_to_grid` to produce this.
    period : int
        Number of samples in one seasonal cycle. Set it from the series' own
        sampling interval, e.g. 48 for a 30-minute series with a daily cycle.
    filled : bool array or None
        True where the corresponding slot was filled by resampling rather than
        observed. Used to keep interpolated points out of the residual
        statistics and out of the flagged output. None means every slot is
        treated as observed.
    k : float
        Threshold in residual standard deviations. Default 3.0.
    robust : bool
        Passed to STL. Default True.
    """
    values = np.asarray(series, dtype=float)
    if filled is None:
        observed = np.ones(len(values), dtype=bool)
    else:
        observed = ~np.asarray(filled, dtype=bool)

    stl = STL(series, period=period, robust=robust).fit()
    trend = np.asarray(stl.trend, dtype=float)
    seasonal = np.asarray(stl.seasonal, dtype=float)
    residual = np.asarray(stl.resid, dtype=float)
    fitted = trend + seasonal

    resid_obs = residual[observed & np.isfinite(residual)]
    resid_mean = float(np.mean(resid_obs))
    resid_std = float(np.std(resid_obs))

    if not np.isfinite(resid_std) or resid_std == 0.0:
        flags = np.zeros(len(values), dtype=bool)
    else:
        flags = np.abs(residual - resid_mean) > k * resid_std
    flags[~np.isfinite(residual)] = False
    observed_flags = flags & observed

    return STLThresholdResult(
        period=period,
        k=float(k),
        robust=robust,
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        fitted=fitted,
        resid_mean=resid_mean,
        resid_std=resid_std,
        flags=flags,
        observed_flags=observed_flags,
    )
