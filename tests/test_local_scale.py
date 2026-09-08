"""Checks for the local robust residual scale and the shared residual threshold."""

import numpy as np

from src.detectors.local_scale import (
    rolling_mad_sigma,
    robust_global_scale,
    threshold_residual,
)


def test_rolling_sigma_tracks_a_variance_change():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1, 600)
    loud = rng.normal(0, 5, 600)
    x = np.concatenate([quiet, loud])
    _, sigma = rolling_mad_sigma(x, window=101)
    assert np.nanmedian(sigma[100:500]) < 2.0
    assert np.nanmedian(sigma[700:1100]) > 3.5


def test_current_point_is_excluded_from_its_own_window():
    # window of 5 centred on index 2; neighbours are [0, 0, 100, 100]
    x = np.array([0.0, 0.0, 999.0, 100.0, 100.0])
    centre_excl, _ = rolling_mad_sigma(x, window=5, exclude_center=True)
    centre_incl, _ = rolling_mad_sigma(x, window=5, exclude_center=False)
    assert centre_excl[2] == 50.0  # median of [0, 0, 100, 100], the spike left out
    assert centre_incl[2] == 100.0  # median of [0, 0, 999, 100, 100], the spike counted

    # and a sustained burst cannot pump up the scale it is judged against
    rng = np.random.default_rng(9)
    resid = rng.normal(0, 1, 800)
    resid[400:420] = 15.0
    out = threshold_residual(resid, k=3.0, scale_kind="local", window=101)
    assert out.observed_flags[400:420].mean() > 0.7


def test_filled_slots_do_not_shrink_the_local_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 3, 400)
    observed = np.ones(400, dtype=bool)
    observed[150:250] = False
    x[150:250] = 0.0  # interpolated-looking flat stretch
    _, sigma_with = rolling_mad_sigma(x, window=81, observed=observed)
    _, sigma_without = rolling_mad_sigma(x, window=81, observed=np.ones(400, dtype=bool))
    # ignoring the flat filled stretch keeps the scale near the true 3
    assert np.nanmedian(sigma_with[160:240]) > np.nanmedian(sigma_without[160:240])


def test_robust_global_scale_matches_std_for_gaussian():
    rng = np.random.default_rng(2)
    x = rng.normal(5.0, 2.0, 20000)
    centre, scale = robust_global_scale(x)
    assert abs(centre - 5.0) < 0.1
    assert abs(scale - 2.0) < 0.1


def test_threshold_local_flags_a_spike_in_a_quiet_stretch_not_loud_noise():
    rng = np.random.default_rng(3)
    resid = np.concatenate([rng.normal(0, 1, 500), rng.normal(0, 8, 500)])
    resid[250] = 12.0  # 12 sigma locally in the quiet half
    out = threshold_residual(resid, k=3.0, scale_kind="local", window=101)
    assert out.observed_flags[250]
    # ordinary points in the loud half are within the local band
    assert out.observed_flags[600:900].mean() < 0.05


def test_threshold_global_is_a_flat_band():
    rng = np.random.default_rng(7)
    resid = rng.normal(0, 1, 801)
    resid[400] = 20.0
    for kind in ("global_std", "global_mad"):
        out = threshold_residual(resid, k=3.0, scale_kind=kind)
        assert out.observed_flags[400]
        assert np.allclose(out.scale, out.scale[0])  # one scale everywhere


def test_local_window_required():
    try:
        threshold_residual(np.zeros(10), k=3.0, scale_kind="local")
    except ValueError:
        return
    raise AssertionError("expected ValueError when window is missing for local scale")
