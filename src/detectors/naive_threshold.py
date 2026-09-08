"""Static global-threshold detector. Intentionally naive baseline.

The rule: take the mean and standard deviation over the whole series, then flag
every point that lies more than `k` standard deviations from that mean. There is
no seasonal awareness and no trend awareness. A daily peak that recurs every day
is, to this detector, just a large value, so it gets flagged every day.

This exists to demonstrate the problem the rest of the project addresses. On a
series with a strong daily or weekly cycle it should over-alert badly, flagging
the tops (and bottoms) of the normal cycle instead of the labelled anomalies. On
a series that is roughly stationary with no cycle it should behave acceptably.

`k = 3` is the default: the textbook "three sigma" rule, which flags about 0.27%
of points if the series were normally distributed. It is exposed as a parameter
because real metric distributions are skewed and heavy-tailed and 3 is not
sacred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_K = 3.0


@dataclass(frozen=True)
class ThresholdFit:
    """The global statistics the detector used, kept for plotting and reporting."""

    mean: float
    std: float
    k: float

    @property
    def lower(self) -> float:
        return self.mean - self.k * self.std

    @property
    def upper(self) -> float:
        return self.mean + self.k * self.std


def fit(values: np.ndarray, k: float = DEFAULT_K) -> ThresholdFit:
    """Compute the global mean and standard deviation, ignoring NaNs."""
    v = np.asarray(values, dtype=float)
    return ThresholdFit(mean=float(np.nanmean(v)), std=float(np.nanstd(v)), k=float(k))


def detect(values: np.ndarray, k: float = DEFAULT_K) -> np.ndarray:
    """Flag points more than `k` global standard deviations from the global mean.

    Parameters
    ----------
    values : array-like of float
        The metric series in time order. NaNs are ignored when computing the
        statistics and are never flagged.
    k : float
        Distance from the mean, in standard deviations, beyond which a point is
        flagged. Default 3.0.

    Returns
    -------
    numpy.ndarray of bool
        True where the point is flagged as anomalous. Same length as `values`.
    """
    v = np.asarray(values, dtype=float)
    stats = fit(v, k=k)
    if not np.isfinite(stats.std) or stats.std == 0.0:
        # A flat or degenerate series has no "far from the mean" points.
        return np.zeros(len(v), dtype=bool)
    flags = np.abs(v - stats.mean) > k * stats.std
    flags[~np.isfinite(v)] = False
    return flags
