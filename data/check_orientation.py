"""
Orientation diagnostic: raw NetCDF vs processed memmap.

Plots a 2×2 grid:
  Row 0: raw nc crops (ground truth, no manipulation)
  Row 1: memmap data (what the model actually trains on)

Also prints corner latitudes so you can read orientation without guessing.

The memmap stores only wind values (no latitude metadata), so its row
latitudes are recovered *dynamically* by matching the stored frame against the
raw crop and its north-up flip — no hardcoded assumption about whether
build_memmap flipped the data. This means the bottom-right panel tells you the
memmap's TRUE orientation, and will flag it if the flip has not been applied
(e.g. the memmap on disk is stale and needs rebuilding).

Usage
-----
    python check_orientation.py
    python check_orientation.py --idx 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from setup import image_size, dataset_length_2009
from data.build_memmap import (
    _find_cerra_crop,
    ERA5_LAT_MAX, ERA5_LAT_MIN, ERA5_LON_MIN, ERA5_LON_MAX,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx",      type=int, default=100,
                        help="Time index into the 2009 NC files")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out",      default="check_orientation.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    idx = args.idx

    # ── 1. Load raw CERRA nc ───────────────────────────────────────────────────
    cerra_nc = next(Path("data/cerra").glob("cerra_si10_2009.nc"))
    ds_c = xr.open_dataset(cerra_nc)
    lat2d = ds_c["latitude"].values
    lon2d = ds_c["longitude"].values
    y0, x0 = _find_cerra_crop(lat2d, lon2d)

    corner_top    = lat2d[y0,                x0]
    corner_bottom = lat2d[y0 + image_size-1, x0]
    print("─" * 60)
    print("CERRA native grid orientation (from NC file):")
    print(f"  row  0 of crop → lat = {corner_top:.3f}°N")
    print(f"  row {image_size-1} of crop → lat = {corner_bottom:.3f}°N")
    if corner_top > corner_bottom:
        print("  ✓  North-first  (row 0 = top = north).  NO flip needed in build_memmap.")
    else:
        print("  ✗  South-first  (row 0 = top = south).  Flip IS needed in build_memmap.")
    print("─" * 60)

    cerra_raw = ds_c["si10"].values[idx, y0 : y0 + image_size, x0 : x0 + image_size]
    ds_c.close()

    # ── 2. Load raw ERA5 nc ────────────────────────────────────────────────────
    era5_nc = next(Path("data/era5").glob("era5_wind_2009.nc"))
    ds_e = xr.open_dataset(era5_nc)
    region = ds_e.sel(
        latitude  = slice(ERA5_LAT_MAX, ERA5_LAT_MIN),
        longitude = slice(ERA5_LON_MIN,  ERA5_LON_MAX),
    )
    u10 = region["u10"].values[idx].astype("float32")
    v10 = region["v10"].values[idx].astype("float32")
    era5_raw = np.sqrt(u10**2 + v10**2)
    era5_lats = region["latitude"].values    # descending → row 0 = north
    ds_e.close()

    print(f"ERA5 native grid:  row 0 lat={era5_lats[0]:.2f}°N, "
          f"last row lat={era5_lats[-1]:.2f}°N  (should be N→S)")

    # ── 3. Load processed memmaps ──────────────────────────────────────────────
    era5_shape = np.load(data_dir / "era5_shape.npy")
    era5_h, era5_w = int(era5_shape[0]), int(era5_shape[1])

    cerra_mm = np.memmap(data_dir / "cerra_val.npy", dtype="float32", mode="r",
                         shape=(dataset_length_2009, image_size, image_size))
    era5_mm  = np.memmap(data_dir / "era5_val.npy",  dtype="float32", mode="r",
                         shape=(dataset_length_2009, era5_h, era5_w))
    cerra_memmap = cerra_mm[idx]
    era5_memmap  = era5_mm[idx]

    # Latitudes of each raw-crop row (as it comes out of the NC, south-first).
    cerra_raw_lats = lat2d[y0 : y0 + image_size, x0]

    # ── 3b. Recover the memmap's TRUE orientation dynamically ──────────────────
    # No hardcoding: compare the stored frame against the raw crop as-is and the
    # raw crop flipped north-up. Whichever it matches reveals the real row order,
    # and hence the real per-row latitudes to annotate with.
    err_asis    = float(np.nanmean((cerra_memmap - cerra_raw)       ** 2))
    err_flipped = float(np.nanmean((cerra_memmap - cerra_raw[::-1]) ** 2))
    if err_flipped <= err_asis:
        cerra_mm_lats  = cerra_raw_lats[::-1]          # north-up: matches flipped raw
        cerra_mm_state = "NORTH-up ✓ (matches ERA5)"
    else:
        cerra_mm_lats  = cerra_raw_lats                # south-up: flip NOT applied
        cerra_mm_state = "SOUTH-up ✗ (flip missing — rebuild memmap)"

    print(f"[memmap] orientation match: err_asis={err_asis:.4f}  "
          f"err_flipped={err_flipped:.4f}")
    print(f"[memmap] CERRA memmap is {cerra_mm_state}")

    # ── 4. Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Orientation diagnostic  (NC idx={idx})\n"
        "Row axis: pixel row 0 is always at TOP.  "
        "Correct = north (high lat) at top.",
        fontsize=11,
    )

    vmax_cerra = max(cerra_raw.max(), cerra_memmap.max())
    vmax_era5  = max(era5_raw.max(),  era5_memmap.max())

    kw_c = dict(cmap="viridis", vmin=0, vmax=vmax_cerra)
    kw_e = dict(cmap="viridis", vmin=0, vmax=vmax_era5)

    def annotate(ax, img, lats=None):
        """Print lat of first and last pixel row on the image."""
        if lats is not None:
            ax.text(0.01, 0.99, f"row 0 → {lats[0]:.1f}°N",
                    transform=ax.transAxes, va="top", fontsize=8,
                    color="white", bbox=dict(fc="black", alpha=0.5, pad=2))
            ax.text(0.01, 0.01, f"row {img.shape[0]-1} → {lats[-1]:.1f}°N",
                    transform=ax.transAxes, va="bottom", fontsize=8,
                    color="white", bbox=dict(fc="black", alpha=0.5, pad=2))

    # [0,0] ERA5 raw nc
    axes[0, 0].imshow(era5_raw, **kw_e)
    axes[0, 0].set_title("ERA5  –  raw NC\n(row 0 = north, always correct)")
    annotate(axes[0, 0], era5_raw, era5_lats)

    # [0,1] CERRA raw nc (no flip, as it comes out of the file)
    axes[0, 1].imshow(cerra_raw, **kw_c)
    axes[0, 1].set_title("CERRA  –  raw NC (no manipulation)")
    annotate(axes[0, 1], cerra_raw, cerra_raw_lats)

    # [1,0] ERA5 memmap
    axes[1, 0].imshow(era5_memmap, **kw_e)
    axes[1, 0].set_title("ERA5  –  memmap")
    annotate(axes[1, 0], era5_memmap, era5_lats)

    # [1,1] CERRA memmap  — labels derived dynamically from the data above
    axes[1, 1].imshow(cerra_memmap, **kw_c)
    axes[1, 1].set_title(f"CERRA  –  memmap\n({cerra_mm_state})")
    annotate(axes[1, 1], cerra_memmap, cerra_mm_lats)

    for ax in axes.flat:
        ax.set_xlabel("pixel column →")
        ax.set_ylabel("pixel row ↓")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

    fig.colorbar(axes[0, 0].images[0], ax=axes[0], label="ERA5 wind [m/s]",  shrink=0.7)
    fig.colorbar(axes[0, 1].images[0], ax=axes[1], label="CERRA wind [m/s]", shrink=0.7)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved: {args.out}")
    print()
    print("How to read the plot:")
    print("  • ERA5 raw NC (top-left) is the reference — always north at top.")
    print("  • CERRA raw NC (top-right): if its text says row 0 is a SMALLER lat")
    print("    (e.g. 36°N) than the last row, it is south-first → flip IS needed.")
    print("  • CERRA memmap (bottom-right): its label is derived from the actual")
    print("    stored pixels. It should read NORTH-up and look like the vertical")
    print("    mirror of the raw NC panel. If it reads SOUTH-up, the memmap on")
    print("    disk is stale — rerun build_memmap.py to regenerate it.")


if __name__ == "__main__":
    main()