"""Sanity checks for the STL residual-threshold detector and the grid resampler."""

import numpy as np
import pandas as pd

from src.data.nab_loader import resample_to_grid
from src.detectors import stl_threshold


def _seasonal_series(days=30, period=24, amp=10.0, trend_per_day=0.5, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = days * period
    idx = pd.date_range("2021-01-01", periods=n, freq="h")
    t = np.arange(n)
    cycle = amp * np.sin(2 * np.pi * t / period)
    trend = trend_per_day * (t / period)
    y = 50.0 + cycle + trend + rng.normal(0, noise, n)
    return pd.Series(y, index=idx), period


def test_recurring_peak_is_not_flagged_but_injected_spike_is():
    s, period = _seasonal_series()
    # a spike at a cycle trough, far from the seasonal shape
    spike_pos = 15 * period + period // 2
    s.iloc[spike_pos] += 40.0
    res = stl_threshold.decompose_and_detect(s, period=period, k=3.0)
    assert res.observed_flags[spike_pos]
    # the ordinary daily maxima must not be flagged
    daily_max_positions = [d * period + period // 4 for d in range(3, 12)]
    assert not res.observed_flags[daily_max_positions].any()


def test_trend_is_absorbed_into_trend_component_not_residual():
    s, period = _seasonal_series(days=60, trend_per_day=1.0, noise=0.2)
    res = stl_threshold.decompose_and_detect(s, period=period, k=3.0)
    x = np.arange(len(s))
    resid_slope = np.polyfit(x, res.residual, 1)[0]
    raw_slope = np.polyfit(x, s.to_numpy(), 1)[0]
    assert abs(resid_slope) < 0.05 * abs(raw_slope)


def test_filled_slots_are_never_flagged():
    s, period = _seasonal_series(days=20)
    filled = np.zeros(len(s), dtype=bool)
    filled[100:110] = True
    s.iloc[105] += 60.0  # a spike sitting on a "filled" slot
    res = stl_threshold.decompose_and_detect(s, period=period, filled=filled, k=3.0)
    assert not res.observed_flags[105]
    assert not res.observed_flags[filled].any()


def test_resample_to_grid_marks_and_fills_gaps():
    idx = pd.to_datetime(
        ["2021-01-01 00:00", "2021-01-01 00:05", "2021-01-01 00:10",
         "2021-01-01 00:30", "2021-01-01 00:35"]  # 00:15..00:25 missing
    )
    df = pd.DataFrame({"timestamp": idx, "value": [1.0, 2.0, 3.0, 6.0, 7.0]})
    series, filled, step = resample_to_grid(df, method="linear")
    assert step == pd.Timedelta("5min")
    assert len(series) == 8
    assert int(filled.sum()) == 3
    assert not series.isna().any()
    # linear fill between value 3.0 (idx 2) and value 6.0 (idx 6): +0.75 per slot
    assert np.isclose(series.iloc[3], 3.75)
    assert np.isclose(series.iloc[4], 4.5)
    assert np.isclose(series.iloc[5], 5.25)


def test_resample_to_grid_ffill_option():
    idx = pd.to_datetime(
        ["2021-01-01 00:00", "2021-01-01 00:05", "2021-01-01 00:20"]
    )
    df = pd.DataFrame({"timestamp": idx, "value": [1.0, 2.0, 5.0]})
    series, filled, _ = resample_to_grid(df, method="ffill")
    assert int(filled.sum()) == 2
    assert series.iloc[2] == 2.0 and series.iloc[3] == 2.0  # carried forward
