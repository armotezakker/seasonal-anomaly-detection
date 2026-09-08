"""Sanity checks for the naive global-threshold detector."""

import numpy as np

from src.detectors import naive_threshold


def test_flags_only_the_far_outlier():
    values = np.concatenate([np.zeros(999), [100.0]])
    flags = naive_threshold.detect(values, k=3.0)
    assert flags[-1]
    assert flags.sum() == 1


def test_k_controls_sensitivity():
    rng = np.random.default_rng(0)
    values = rng.normal(size=5000)
    assert naive_threshold.detect(values, k=2.0).sum() > naive_threshold.detect(values, k=4.0).sum()


def test_flat_series_flags_nothing():
    assert naive_threshold.detect(np.full(100, 7.0), k=3.0).sum() == 0


def test_nans_are_never_flagged():
    values = np.concatenate([np.zeros(50), [np.nan], np.zeros(48), [20.0]])
    flags = naive_threshold.detect(values, k=3.0)
    assert not flags[50]  # the NaN
    assert flags[-1]  # the clear outlier, well beyond 3 sigma here


def test_fit_band_matches_mean_and_std():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fit = naive_threshold.fit(values, k=2.0)
    assert np.isclose(fit.mean, 3.0)
    assert np.isclose(fit.upper, 3.0 + 2.0 * values.std())
    assert np.isclose(fit.lower, 3.0 - 2.0 * values.std())
