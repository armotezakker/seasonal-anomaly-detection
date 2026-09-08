"""Multi-season detector: MSTL with a daily and a weekly period, then a residual
threshold (global or local scale).

Phase 3 found that a single daily STL left the entire weekly cycle in the
residual on nyc_taxi and ambient_temperature, so every weekend day looked
anomalous. MSTL fits more than one seasonal period at once:

    observed = trend + seasonal_daily + seasonal_weekly + residual

so a systematic weekend difference is carried by `seasonal_weekly` and no longer
shows up in the residual.

Both periods are derived from the sampling interval: daily is samples-per-day,
weekly is seven times that. `seasonal_amplitudes` reports how much structure
MSTL actually put in each component, so the weekly term can be checked rather
than assumed.

Held fixed across every series: k = 3, robust = True (passed through to the
inner STL fits), statsmodels' default smoother lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from src.detectors.local_scale import ResidualFlags, threshold_residual

DEFAULT_K = 3.0
ROBUST = True


@dataclass
class MSTLDecomposition:
    periods: tuple[int, ...]
    robust: bool
    trend: np.ndarray = field(repr=False)
    seasonal: dict[int, np.ndarray] = field(repr=False)  # period -> component
    seasonal_total: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)
    fitted: np.ndarray = field(repr=False)  # trend + seasonal_total


def mstl_decompose(
    series: pd.Series, periods, robust: bool = ROBUST
) -> MSTLDecomposition:
    """Run MSTL and return trend, one seasonal array per period, residual, sum."""
    periods = tuple(int(p) for p in periods)
    res = MSTL(series, periods=periods, stl_kwargs={"robust": robust}).fit()

    seasonal_df = res.seasonal
    seasonal: dict[int, np.ndarray] = {}
    if isinstance(seasonal_df, pd.DataFrame):
        # columns are named "seasonal_<period>", in the same order as `periods`
        for period, col in zip(periods, seasonal_df.columns):
            seasonal[period] = seasonal_df[col].to_numpy(dtype=float)
    else:  # single period -> Series
        seasonal[periods[0]] = np.asarray(seasonal_df, dtype=float)

    trend = np.asarray(res.trend, dtype=float)
    residual = np.asarray(res.resid, dtype=float)
    seasonal_total = np.sum(list(seasonal.values()), axis=0)

    return MSTLDecomposition(
        periods=periods,
        robust=robust,
        trend=trend,
        seasonal=seasonal,
        seasonal_total=seasonal_total,
        residual=residual,
        fitted=trend + seasonal_total,
    )


def seasonal_amplitudes(decomp: MSTLDecomposition, observed=None) -> dict:
    """Amplitude of each seasonal component: standard deviation and peak to peak.

    Also returns each component's std as a fraction of the largest component's
    std, so a weekly term that is trivial next to the daily term is obvious.
    """
    if observed is None:
        mask = slice(None)
    else:
        mask = np.asarray(observed, dtype=bool)

    stats = {}
    for period, comp in decomp.seasonal.items():
        c = comp[mask]
        stats[period] = {
            "std": float(np.std(c)),
            "ptp": float(np.ptp(c)),
        }
    max_std = max(s["std"] for s in stats.values()) or 1.0
    for period in stats:
        stats[period]["std_frac_of_largest"] = stats[period]["std"] / max_std
    return stats


@dataclass
class MSTLThresholdResult:
    decomposition: MSTLDecomposition
    residual_flags: ResidualFlags
    k: float

    @property
    def residual(self) -> np.ndarray:
        return self.decomposition.residual

    @property
    def fitted(self) -> np.ndarray:
        return self.decomposition.fitted

    @property
    def observed_flags(self) -> np.ndarray:
        return self.residual_flags.observed_flags


def decompose_and_detect(
    series: pd.Series,
    periods,
    filled=None,
    k: float = DEFAULT_K,
    robust: bool = ROBUST,
    scale_kind: str = "global",
    window: int | None = None,
) -> MSTLThresholdResult:
    """MSTL decomposition followed by a residual threshold.

    `scale_kind` and `window` are passed straight to
    `local_scale.threshold_residual`.
    """
    if filled is None:
        observed = np.ones(len(series), dtype=bool)
    else:
        observed = ~np.asarray(filled, dtype=bool)

    decomp = mstl_decompose(series, periods=periods, robust=robust)
    flags = threshold_residual(
        decomp.residual, observed=observed, k=k, scale_kind=scale_kind, window=window,
    )
    return MSTLThresholdResult(decomposition=decomp, residual_flags=flags, k=float(k))
