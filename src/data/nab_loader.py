"""Readers for the local NAB data and label files.

This module only parses the CSV and JSON files that `scripts/fetch_data.sh`
pulls into `data/raw/`. It does not import anything from the NAB codebase; the
file formats are simple and are documented inline here.

Data files: two columns, `timestamp,value`. Timestamps are naive local time
formatted `YYYY-MM-DD HH:MM:SS`. One float value per row, in time order.

Label file `combined_windows.json`: a flat object keyed by
`"<category>/<filename>.csv"`. Each value is a list of `[start, end]` timestamp
strings (microsecond precision); each pair is one labelled anomaly window, i.e.
a closed time interval a detector is expected to alert somewhere inside.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_DIR = RAW_DIR / "data"
WINDOWS_PATH = RAW_DIR / "labels" / "combined_windows.json"

# The six series selected in Phase 1. Keys are exactly the label-file keys.
CANDIDATE_SERIES = [
    "realKnownCause/nyc_taxi.csv",
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realAdExchange/exchange-2_cpm_results.csv",
    "realTraffic/occupancy_t4013.csv",
    "realTraffic/occupancy_6005.csv",
    "realTweets/Twitter_volume_AMZN.csv",
]


def load_series(key: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load one series as a DataFrame with `timestamp` (datetime64) and `value` (float).

    Rows are returned in timestamp order with a fresh RangeIndex. No resampling
    and no interpolation: the frame holds exactly the observations in the file,
    which is what a point detector should see.
    """
    path = data_dir / key
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["value"] = df["value"].astype(float)
    return df


def load_windows(key: str, windows_path: Path = WINDOWS_PATH) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the labelled anomaly windows for `key` as (start, end) Timestamp pairs.

    Returns an empty list if the series has no labelled windows.
    """
    with open(windows_path) as fh:
        windows = json.load(fh)
    return [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in windows.get(key, [])]


def sampling_step(df: pd.DataFrame) -> pd.Timedelta:
    """Median spacing between consecutive timestamps.

    Used as the unit for the evaluation edge tolerance so that a tolerance
    expressed in "number of observations" means the same thing across series
    sampled at 5, 10, 30 or 60 minutes.
    """
    step = df["timestamp"].diff().median()
    return pd.Timedelta(step)


def modal_step(df: pd.DataFrame) -> pd.Timedelta:
    """Most common spacing between consecutive timestamps.

    Preferred over the median when a series has many short gaps: the mode is the
    sensor's real reporting interval, which is what the resampling grid should
    use.
    """
    diffs = df["timestamp"].diff().dropna()
    return pd.Timedelta(diffs.value_counts().idxmax())


def resample_to_grid(df: pd.DataFrame, method: str = "linear"):
    """Put a series on a complete grid at its modal sampling interval.

    STL needs a gap-free series on a fixed frequency. This bins every
    observation into a grid slot (`resample(...).mean()`, so no observation is
    dropped even when its timestamp sits slightly off the grid), then fills the
    slots that had no observation.

    Returns
    -------
    series : pandas.Series
        Float series, complete DatetimeIndex at `step`, no NaN, `index.freq` set.
    filled : pandas.Series of bool
        True where the slot held no observation and was filled. Aligned to
        `series`. Callers should exclude these timestamps from detector output
        (nothing was measured there) and compute residual statistics from the
        observed slots only.
    step : pandas.Timedelta
        The modal sampling interval used for the grid.

    Gap filling
    -----------
    `method="linear"` (default) linearly interpolates filled slots.
    `method="ffill"` carries the last observation forward.

    Linear interpolation is the default because forward fill creates flat
    plateaus that STL reads as a genuine low-variance stretch followed by a
    step, which distorts the seasonal and trend fit near every gap. The cost of
    linear interpolation is that a long gap becomes a straight ramp carrying no
    daily cycle, so STL still imposes a seasonal wave on stretches where there
    is no data. This is acceptable only because filled timestamps are barred
    from being flagged; residuals at genuine observations within about one
    period of a long gap are still affected, which is a real limitation on the
    two realTraffic series (roughly half their grid is filled, with one gap of
    about 3.5 days) and a minor one on ambient_temperature (two multi-day gaps).
    The cleaner alternative, splitting a series at long gaps and decomposing
    each segment, fragments the trend estimate and needs every segment to span
    at least two periods; it is left as future work.
    """
    s = df.set_index("timestamp")["value"].sort_index()
    step = modal_step(df)
    on_grid = s.resample(step, origin="start").mean()
    filled = on_grid.isna()
    if method == "linear":
        on_grid = on_grid.interpolate("linear", limit_direction="both")
    elif method == "ffill":
        on_grid = on_grid.ffill().bfill()
    else:
        raise ValueError(f"unknown method: {method!r}")
    on_grid.index.freq = step
    return on_grid.astype(float), filled, step
