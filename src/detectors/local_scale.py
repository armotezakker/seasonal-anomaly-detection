"""Local robust scale for residuals, and the residual threshold that uses it.

Phase 3 flagged a residual point when it sat more than k standard deviations
from the residual mean, with one mean and one standard deviation for the whole
series. That is too tight during busy or bursty stretches, where the residual
genuinely varies more, and too loose during quiet stretches.

`rolling_mad_sigma` replaces the single number with a local one: inside a window
of roughly one seasonal period it takes the median absolute deviation of the
residual and scales it by 1.4826 (the factor that makes MAD match the standard
deviation for Gaussian data). The point being judged is left out of its own
window, so a spike cannot inflate the scale that is used to test that same
spike. Filled slots are also left out, so interpolated near-zero residuals do
not shrink the scale.

`threshold_residual` is the shared comparison used by both the STL and the MSTL
detectors: flag where `|residual - local centre| > k * local scale`, with a
robust global fallback wherever the local window has too few real points.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

MAD_TO_SIGMA = 1.4826


def _observed_finite(residual, observed):
    r = np.asarray(residual, dtype=float)
    mask = np.isfinite(r)
    if observed is not None:
        mask &= np.asarray(observed, dtype=bool)
    return r[mask]


def global_std_scale(residual, observed=None):
    """Mean and standard deviation over observed points. The Phase 3 rule."""
    r = _observed_finite(residual, observed)
    if r.size == 0:
        return 0.0, 0.0
    return float(np.mean(r)), float(np.std(r))


def robust_global_scale(residual, observed=None):
    """Median and 1.4826 * MAD of the residual over observed points. Scalars."""
    r = _observed_finite(residual, observed)
    if r.size == 0:
        return 0.0, 0.0
    centre = float(np.median(r))
    scale = float(MAD_TO_SIGMA * np.median(np.abs(r - centre)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = float(np.std(r))
    return centre, scale


def rolling_mad_sigma(
    residual,
    window: int,
    observed=None,
    exclude_center: bool = True,
    min_valid_frac: float = 0.25,
    chunk: int = 4096,
):
    """Rolling robust centre and scale of `residual`.

    Parameters
    ----------
    residual : array-like of float
    window : int
        Nominal window length in samples. Forced to the next odd number so the
        window has a single centre sample.
    observed : bool array or None
        False where a slot was filled by resampling; those slots are excluded
        from every window.
    exclude_center : bool
        Leave the centre sample out of its own window. Default True.
    min_valid_frac : float
        If fewer than this fraction of a window's samples are usable, the local
        centre and scale are returned as NaN for that position (the caller
        substitutes a global value).

    Returns
    -------
    centre, scale : numpy.ndarray, numpy.ndarray
        Same length as `residual`. `scale` is 1.4826 * local MAD. Positions with
        too few usable neighbours are NaN in both.
    """
    r = np.asarray(residual, dtype=float)
    n = r.size
    half = int(window) // 2
    w = 2 * half + 1

    usable = np.isfinite(r)
    if observed is not None:
        usable &= np.asarray(observed, dtype=bool)
    r_clean = np.where(usable, r, np.nan)

    pad = np.full(half, np.nan)
    padded = np.concatenate([pad, r_clean, pad])

    centre = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    min_valid = max(3, int(min_valid_frac * w))

    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        win = np.lib.stride_tricks.sliding_window_view(padded, w)[start:stop].copy()
        if exclude_center:
            win[:, half] = np.nan
        valid = np.isfinite(win).sum(axis=1)
        with warnings.catch_warnings():
            # windows that are entirely inside a gap are all-NaN; expected.
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=1)
            win -= med[:, None]
            np.abs(win, out=win)
            mad = np.nanmedian(win, axis=1)

        seg_centre = med
        seg_scale = MAD_TO_SIGMA * mad
        too_few = valid < min_valid
        seg_centre[too_few] = np.nan
        seg_scale[too_few] = np.nan
        centre[start:stop] = seg_centre
        scale[start:stop] = seg_scale

    return centre, scale


@dataclass
class ResidualFlags:
    scale_kind: str  # "global" or "local"
    window: int | None
    k: float
    centre: np.ndarray = field(repr=False)  # per-sample, after fallback
    scale: np.ndarray = field(repr=False)  # per-sample, after fallback
    flags: np.ndarray = field(repr=False)  # over the whole grid
    observed_flags: np.ndarray = field(repr=False)  # filled slots forced False

    @property
    def lower(self) -> np.ndarray:
        return self.centre - self.k * self.scale

    @property
    def upper(self) -> np.ndarray:
        return self.centre + self.k * self.scale


def threshold_residual(
    residual,
    observed=None,
    k: float = 3.0,
    scale_kind: str = "global_std",
    window: int | None = None,
    exclude_center: bool = True,
) -> ResidualFlags:
    """Flag residual points with a global or a local scale.

    scale_kind:
      "global_std"  one mean and one standard deviation for the whole series
                    (this is the Phase 3 rule);
      "global_mad"  one median and one 1.4826 * MAD for the whole series
                    (robust, but still not local);
      "local"       rolling 1.4826 * MAD in `window`, with the current point
                    left out, falling back to "global_mad" wherever the local
                    window is too sparse or flat.
    """
    r = np.asarray(residual, dtype=float)
    n = r.size
    if observed is None:
        observed_mask = np.ones(n, dtype=bool)
    else:
        observed_mask = np.asarray(observed, dtype=bool)

    g_centre, g_scale = robust_global_scale(r, observed_mask)
    fallback_flat = not np.isfinite(g_scale) or g_scale == 0.0

    if scale_kind == "global_std":
        s_centre, s_scale = global_std_scale(r, observed_mask)
        centre = np.full(n, s_centre)
        scale = np.full(n, s_scale)
        fallback_flat = not np.isfinite(s_scale) or s_scale == 0.0
    elif scale_kind == "global_mad":
        centre = np.full(n, g_centre)
        scale = np.full(n, g_scale)
    elif scale_kind == "local":
        if window is None:
            raise ValueError("window is required when scale_kind='local'")
        centre, scale = rolling_mad_sigma(
            r, window=window, observed=observed_mask, exclude_center=exclude_center,
        )
        bad = ~np.isfinite(centre)
        centre[bad] = g_centre
        bad_scale = ~np.isfinite(scale) | (scale <= 0.0)
        scale[bad_scale] = g_scale if g_scale > 0.0 else 1.0
    else:
        raise ValueError(f"unknown scale_kind: {scale_kind!r}")

    if fallback_flat:
        # Degenerate residual (flat series): nothing to flag.
        flags = np.zeros(n, dtype=bool)
    else:
        flags = np.abs(r - centre) > k * scale
    flags[~np.isfinite(r)] = False
    observed_flags = flags & observed_mask

    return ResidualFlags(
        scale_kind=scale_kind,
        window=window,
        k=float(k),
        centre=centre,
        scale=scale,
        flags=flags,
        observed_flags=observed_flags,
    )
