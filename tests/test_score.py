"""Checks for the time-tolerant scoring in src/evaluation/score.py."""

import numpy as np
import pandas as pd

from src.evaluation.score import score_series


def _ts(n, start="2020-01-01", freq="5min"):
    return pd.date_range(start, periods=n, freq=freq)


def test_flag_inside_window_is_true_positive():
    ts = _ts(100)
    flags = np.zeros(100, dtype=bool)
    flags[50] = True
    windows = [(ts[45], ts[55])]
    s = score_series(ts, flags, windows, pd.Timedelta(0))
    assert s.true_positives == 1
    assert s.false_positives == 0
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.windows_hit == 1


def test_flag_outside_every_window_is_false_positive():
    ts = _ts(100)
    flags = np.zeros(100, dtype=bool)
    flags[10] = True
    windows = [(ts[45], ts[55])]
    s = score_series(ts, flags, windows, pd.Timedelta(0))
    assert s.true_positives == 0
    assert s.false_positives == 1
    assert s.precision == 0.0
    assert s.recall == 0.0


def test_edge_tolerance_rescues_a_near_miss():
    ts = _ts(100)  # 5-minute spacing
    flags = np.zeros(100, dtype=bool)
    flags[44] = True  # one step before the window opens at index 45
    windows = [(ts[45], ts[55])]
    assert score_series(ts, flags, windows, pd.Timedelta(0)).false_positives == 1
    rescued = score_series(ts, flags, windows, pd.Timedelta("5min"))
    assert rescued.true_positives == 1
    assert rescued.false_positives == 0


def test_recall_counts_windows_not_points():
    ts = _ts(200)
    flags = np.zeros(200, dtype=bool)
    flags[50] = flags[51] = flags[52] = True  # three flags, all in window A
    windows = [(ts[48], ts[55]), (ts[150], ts[160])]  # window B never hit
    s = score_series(ts, flags, windows, pd.Timedelta(0))
    assert s.windows_hit == 1
    assert s.recall == 0.5
    assert s.true_positives == 3


def test_no_windows_gives_nan_recall_and_all_false_positives():
    ts = _ts(50)
    flags = np.zeros(50, dtype=bool)
    flags[5] = True
    s = score_series(ts, flags, [], pd.Timedelta(0))
    assert np.isnan(s.recall)
    assert s.false_positives == 1
    assert s.precision == 0.0


def test_nothing_flagged_gives_nan_precision():
    ts = _ts(50)
    flags = np.zeros(50, dtype=bool)
    windows = [(ts[10], ts[20])]
    s = score_series(ts, flags, windows, pd.Timedelta(0))
    assert np.isnan(s.precision)
    assert s.recall == 0.0
    assert s.false_positives == 0
