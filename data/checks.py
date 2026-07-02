"""
Runtime guards to ensure CERRA and ERA5 memmaps are correctly oriented and
aligned BEFORE any training happens.

The memmaps carry no latitude metadata, so orientation is verified indirectly:
CERRA and ERA5 cover the same geographic box at the same timestamps, so their
per-row (latitudinal) wind profiles must be positively correlated — and must
correlate *worse* when one is vertically flipped. If CERRA were stored
upside-down relative to ERA5, the flipped correlation would win and the assert
fires.
"""

import numpy as np


def _row_profile(frame: np.ndarray) -> np.ndarray:
    """Mean wind per pixel row -> 1-D latitudinal profile."""
    return np.nanmean(np.asarray(frame, dtype=np.float64), axis=1)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def assert_cerra_era5_aligned(
    cerra_mm,
    era5_mm,
    n_samples: int = 256,
    min_corr:  float = 0.3,
    seed:      int = 0,
    verbose:   bool = True,
    split:     str = "",
) -> tuple[float, float]:
    """
    Assert CERRA is oriented the same way as ERA5 (north-up) and plausibly
    time-aligned. Raises AssertionError otherwise. Returns (corr_asis, corr_flip).
    """
    N = min(len(cerra_mm), len(era5_mm))
    assert N > 0, f"[{split}] Empty memmaps passed to alignment check."
    assert len(cerra_mm) == len(era5_mm), (
        f"[{split}] CERRA and ERA5 have different lengths "
        f"({len(cerra_mm)} vs {len(era5_mm)}) — alignment is broken."
    )

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_samples, N), replace=False)

    era5_h  = int(np.asarray(era5_mm[0]).shape[0])
    cerra_h = int(np.asarray(cerra_mm[0]).shape[0])
    x_c = np.linspace(0.0, 1.0, cerra_h)
    x_e = np.linspace(0.0, 1.0, era5_h)

    corr_asis, corr_flip = [], []
    for i in idx:
        pc = _row_profile(cerra_mm[i])
        pe = _row_profile(era5_mm[i])
        pc_rs = np.interp(x_e, x_c, pc)
        corr_asis.append(_pearson(pc_rs,       pe))
        corr_flip.append(_pearson(pc_rs[::-1], pe))

    ca = float(np.mean(corr_asis))
    cf = float(np.mean(corr_flip))

    if verbose:
        print(f"[orient-check] {split} frames={len(idx)}  "
              f"mean corr as-is={ca:+.3f}  flipped={cf:+.3f}")

    assert ca > cf, (
        f"[{split}] CERRA appears VERTICALLY FLIPPED relative to ERA5 "
        f"(as-is corr {ca:+.3f} <= flipped {cf:+.3f}). "
        f"Do NOT train. Rebuild the memmaps with build_memmap.py."
    )
    assert ca >= min_corr, (
        f"[{split}] CERRA and ERA5 latitudinal profiles barely correlate "
        f"(corr {ca:+.3f} < min_corr {min_corr}). Likely timestamp misalignment "
        f"or wrong crop — inspect build_memmap.py before training."
    )
    if verbose:
        print(f"[orient-check] {split} OK — CERRA is north-up and aligned with ERA5.")
    return ca, cf
