"""Checks for the final recommended detector: routing rule and residual flagging."""

import numpy as np
import pandas as pd

from src.detectors import seasonal_anomaly


def _series(days, freq="h", amp_daily=10.0, amp_weekly=4.0, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=days * 24, freq=freq)
    t = np.arange(len(idx))
    daily = amp_daily * np.sin(2 * np.pi * t / 24)
    weekly = amp_weekly * np.sin(2 * np.pi * t / (24 * 7))
    y = 100.0 + daily + weekly + rng.normal(0, noise, len(idx))
    return pd.Series(y, index=idx), pd.Timedelta("1h")


def test_routes_to_mstl_when_enough_weekly_cycles():
    series, step = _series(days=7 * 12)  # 12 weekly cycles
    res = seasonal_anomaly.detect(series, step, weekly_cycle_min=10.0)
    assert res.decomposition_used == "mstl_daily_weekly"


def test_routes_to_stl_when_too_few_weekly_cycles():
    series, step = _series(days=7 * 3)  # 3 weekly cycles
    res = seasonal_anomaly.detect(series, step, weekly_cycle_min=10.0)
    assert res.decomposition_used == "stl_daily"


def test_short_series_flags_the_injected_spike_and_not_the_weekly_peaks():
    series, step = _series(days=7 * 3, seed=1)
    spike = 7 * 3 * 24 // 2
    series.iloc[spike] += 50.0
    res = seasonal_anomaly.detect(series, step, weekly_cycle_min=10.0)
    assert res.decomposition_used == "stl_daily"
    assert res.observed_flags[spike]
    # weekly maxima of the clean signal must stay unflagged
    weekly_peaks = [d * 24 * 7 + 24 * 7 // 4 for d in range(3)]
    assert not res.observed_flags[weekly_peaks].any()


def test_filled_slots_are_not_flagged():
    series, step = _series(days=7 * 12, seed=2)
    filled = np.zeros(len(series), dtype=bool)
    filled[200:210] = True
    series.iloc[205] += 80.0
    res = seasonal_anomaly.detect(series, step, filled=filled, weekly_cycle_min=10.0)
    assert not res.observed_flags[205]
    assert not res.observed_flags[filled].any()


def test_k_and_threshold_are_module_constants():
    assert seasonal_anomaly.K == 3.0
    assert seasonal_anomaly.WEEKLY_CYCLE_MIN == 10.0
