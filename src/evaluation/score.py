"""Time-tolerant scoring of point detections against NAB anomaly windows.

Rules
-----
Each labelled window from `combined_windows.json` is widened by `edge_tolerance`
on both sides. Then:

  * a flagged point inside any widened window is a true positive;
  * a flagged point outside every widened window is a false positive;
  * a window is "recalled" if at least one flagged point lands inside its
    widened interval.

  precision = true positives / all flagged points
  recall    = recalled windows / labelled windows
  false positives = count of flagged points outside every widened window

Relationship to NAB's official scoring
--------------------------------------
This is deliberately simpler than NAB. NAB scores each detection with a sigmoid
that gives more credit to detections early in a window and applies a tunable
false-positive penalty, then sums to one application-weighted number per series.
We do not implement any of that. Plain precision, recall and a false-positive
count are enough to show the naive baseline's over-alerting, and they are easier
to read than a single weighted score. When this project later compares detectors,
the comparison stays on these same three numbers.

The edge tolerance
------------------
`edge_tolerance` is a real wall-clock `Timedelta`, not a count of rows, and it is
meant to be derived from the series cadence by the caller (see
`EDGE_TOLERANCE_STEPS`). Its only job is to forgive a detection that fires just
before a window opens or just after it closes, which is a labelling-precision
issue rather than a real miss. It is intentionally small relative to the windows
themselves (NAB windows are hours to days long).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Caller-side default: widen each window by this many sampling intervals on each
# side. Three steps is 15 minutes on 5-minute data and 3 hours on hourly data;
# both are short next to NAB windows, which run from a few hours to several days.
# Expressing the tolerance in "number of observations" keeps it consistent across
# series sampled at different rates. The caller multiplies this by the series'
# median sampling step and passes the result as `edge_tolerance`.
EDGE_TOLERANCE_STEPS = 3


@dataclass
class SeriesScore:
    """Per-series result. Arrays are aligned to the full input series."""

    series: str
    n_points: int
    n_flagged: int
    n_windows: int
    true_positives: int
    false_positives: int
    windows_hit: int
    precision: float
    recall: float
    tp_mask: np.ndarray = field(repr=False)
    fp_mask: np.ndarray = field(repr=False)
    window_hit: list[bool] = field(repr=False)

    def as_row(self) -> dict:
        """Flat dict of the scalar fields, for building a results table."""
        return {
            "series": self.series,
            "n_points": self.n_points,
            "n_flagged": self.n_flagged,
            "n_windows": self.n_windows,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "windows_hit": self.windows_hit,
            "precision": self.precision,
            "recall": self.recall,
        }


def score_series(
    timestamps,
    flags,
    windows,
    edge_tolerance: pd.Timedelta,
    series: str = "",
) -> SeriesScore:
    """Score one series.

    Parameters
    ----------
    timestamps : array-like of datetime64, length n
        Observation times, aligned with `flags`.
    flags : array-like of bool, length n
        Detector output: True where a point is flagged anomalous.
    windows : list of (start, end)
        Labelled anomaly windows as Timestamp-coercible pairs. May be empty.
    edge_tolerance : pandas.Timedelta
        Amount each window is widened on both sides before matching.
    series : str
        Name carried through into the result for reporting.
    """
    ts = pd.to_datetime(pd.Index(timestamps)).values.astype("datetime64[ns]")
    flags = np.asarray(flags, dtype=bool)
    if len(ts) != len(flags):
        raise ValueError("timestamps and flags must have the same length")

    tol = pd.Timedelta(edge_tolerance).to_timedelta64()
    widened = [
        (pd.Timestamp(start).to_datetime64() - tol, pd.Timestamp(end).to_datetime64() + tol)
        for start, end in windows
    ]

    in_any_window = np.zeros(len(ts), dtype=bool)
    window_hit: list[bool] = []
    for lo, hi in widened:
        inside = (ts >= lo) & (ts <= hi)
        in_any_window |= inside
        window_hit.append(bool((inside & flags).any()))

    tp_mask = flags & in_any_window
    fp_mask = flags & ~in_any_window

    n_flagged = int(flags.sum())
    true_positives = int(tp_mask.sum())
    false_positives = int(fp_mask.sum())
    n_windows = len(windows)
    windows_hit = int(sum(window_hit))

    precision = true_positives / n_flagged if n_flagged else float("nan")
    recall = windows_hit / n_windows if n_windows else float("nan")

    return SeriesScore(
        series=series,
        n_points=len(ts),
        n_flagged=n_flagged,
        n_windows=n_windows,
        true_positives=true_positives,
        false_positives=false_positives,
        windows_hit=windows_hit,
        precision=precision,
        recall=recall,
        tp_mask=tp_mask,
        fp_mask=fp_mask,
        window_hit=window_hit,
    )
