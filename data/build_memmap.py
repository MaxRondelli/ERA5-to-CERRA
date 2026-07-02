"""
Preprocess downloaded NetCDF files into numpy memmap arrays for fast training.

Steps
-----
1. CERRA:
   - Open yearly NetCDF files.
   - Auto-detect a 256×256 crop centred on Italy using 2-D lat/lon arrays.
   - Stack into (N, 256, 256) memmap.
2. ERA5:
   - Open yearly NetCDF files.
   - Crop to a lat/lon bounding box covering the same geographic area.
   - Compute wind speed: sqrt(u10²+v10²).
   - Stack into (N, H_era5, W_era5) memmap.
3. Align both datasets on their common UTC timestamps; write the final
   aligned memmaps that the Dataset classes consume.

Usage
-----
    python build_memmap.py
    python build_memmap.py --cerra_dir data/cerra --era5_dir data/era5 \\
                           --out_dir data/processed --years 2009 2010 2019 2020
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import image_size

# ── Geographic defaults ────────────────────────────────────────────────────────

# Italy centre (used to find the CERRA 256×256 crop)
CENTER_LAT = 42.0
CENTER_LON  = 12.5

# ERA5 lat/lon bounding box (must fully contain the CERRA crop)
ERA5_LAT_MAX = 48.0
ERA5_LAT_MIN = 36.0
ERA5_LON_MIN  = 5.0
ERA5_LON_MAX  = 18.0


# ── CERRA helpers ──────────────────────────────────────────────────────────────

def _find_cerra_crop(lat2d: np.ndarray, lon2d: np.ndarray, size: int = 256):
    """Return (y0, x0) for a `size`×`size` patch centred on Italy."""
    dist   = (lat2d - CENTER_LAT) ** 2 + (lon2d - CENTER_LON) ** 2
    yc, xc = np.unravel_index(dist.argmin(), dist.shape)
    y0     = int(np.clip(yc - size // 2, 0, lat2d.shape[0] - size))
    x0     = int(np.clip(xc - size // 2, 0, lat2d.shape[1] - size))
    return y0, x0


def build_cerra_memmap(
    cerra_dir: str,
    years:     list[int],
    out_path:  str,
) -> tuple[list, int, int]:
    """
    Stack yearly CERRA files into a single (N, 256, 256) memmap.
    Returns (timestamps_list, y0, x0).
    """
    files = sorted(Path(cerra_dir).glob("cerra_si10_*.nc"))
    files = [f for f in files if int(f.stem.split("_")[-1]) in years]
    assert files, f"No CERRA files found in {cerra_dir} for years {years}"

    # Detect crop from first file
    ds0    = xr.open_dataset(files[0])
    lat2d  = ds0["latitude"].values
    lon2d  = ds0["longitude"].values
    y0, x0 = _find_cerra_crop(lat2d, lon2d)
    
    print(f"[cerra] crop origin y0={y0}, x0={x0}  "
            f"lat=[{lat2d[y0, x0]:.2f}, {lat2d[y0+image_size-1, x0+image_size-1]:.2f}]")
    ds0.close()

    all_times, all_data = [], []
    for f in files:
        ds   = xr.open_dataset(f)
        si10 = ds["si10"].values[:, y0 : y0 + image_size, x0 : x0 + image_size]
        # CERRA's native grid is south-to-north (row 0 = south); flip to match
        # ERA5's north-to-south orientation so imshow renders north at the top.
        si10 = si10[:, ::-1, :]
        times = pd.to_datetime(ds["valid_time"].values)
        all_times.extend(times.tolist())
        all_data.append(si10.astype("float32"))
        ds.close()
        print(f"[cerra] loaded {f.name}: {si10.shape}")

    data = np.concatenate(all_data, axis=0)
    N    = data.shape[0]
    mm   = np.memmap(out_path, dtype="float32", mode="w+", shape=(N, image_size, image_size))
    mm[:] = data
    mm.flush()
    print(f"[cerra] wrote {out_path}  shape={mm.shape}  max={data.max():.3f}")
    return all_times, y0, x0


# ── ERA5 helpers ───────────────────────────────────────────────────────────────

def build_era5_memmap(
    era5_dir: str,
    years:    list[int],
    out_path: str,
) -> tuple[list, int, int]:
    """
    Stack yearly ERA5 files into a single (N, H, W) memmap.
    Returns (timestamps_list, H, W) where H×W is the native ERA5 resolution
    for the Italy bounding box (≈ 48×52 at 0.25°).
    """
    files = sorted(Path(era5_dir).glob("era5_wind_*.nc"))
    files = [f for f in files if int(f.stem.split("_")[-1]) in years]
    assert files, f"No ERA5 files found in {era5_dir} for years {years}"

    all_times, all_data = [], []
    H_ref = W_ref = None

    for f in files:
        ds = xr.open_dataset(f)
        # Latitude is descending in ERA5 (north to south)
        region = ds.sel(
            latitude  = slice(ERA5_LAT_MAX, ERA5_LAT_MIN),
            longitude = slice(ERA5_LON_MIN,  ERA5_LON_MAX),
        )
        u10  = region["u10"].values.astype("float32")
        v10  = region["v10"].values.astype("float32")
        speed = np.sqrt(u10 ** 2 + v10 ** 2)

        times = pd.to_datetime(ds["valid_time"].values)
        all_times.extend(times.tolist())
        all_data.append(speed)
        H_ref, W_ref = speed.shape[1], speed.shape[2]
        ds.close()
        print(f"[era5]  loaded {f.name}: {speed.shape}")

    data = np.concatenate(all_data, axis=0)
    N    = data.shape[0]
    mm   = np.memmap(out_path, dtype="float32", mode="w+", shape=(N, H_ref, W_ref))
    mm[:] = data
    mm.flush()
    print(f"[era5]  wrote {out_path}  shape={mm.shape}  max={data.max():.3f}")
    return all_times, H_ref, W_ref


# ── Alignment ──────────────────────────────────────────────────────────────────

def align_datasets(
    cerra_times: list,
    era5_times:  list,
    cerra_raw:   str,
    era5_raw:    str,
    era5_h:      int,
    era5_w:      int,
    out_cerra:   str,
    out_era5:    str,
) -> int:
    """
    Keep only timestamps present in both datasets; write aligned memmaps.
    Returns the number of aligned samples.
    """
    cerra_idx_map = {t: i for i, t in enumerate(cerra_times)}
    era5_idx_map  = {t: i for i, t in enumerate(era5_times)}
    common        = sorted(set(cerra_times) & set(era5_times))
    N             = len(common)
    assert N > 0, "No common timestamps between CERRA and ERA5!"
    print(f"[align] {N} common timestamps")

    c_raw = np.memmap(cerra_raw, dtype="float32", mode="r").reshape(-1, image_size, image_size)
    e_raw = np.memmap(era5_raw,  dtype="float32", mode="r").reshape(-1, era5_h, era5_w)

    c_out = np.memmap(out_cerra, dtype="float32", mode="w+", shape=(N, image_size, image_size))
    e_out = np.memmap(out_era5,  dtype="float32", mode="w+", shape=(N, era5_h, era5_w))

    for i, t in enumerate(common):
        c_out[i] = c_raw[cerra_idx_map[t]]
        e_out[i] = e_raw[era5_idx_map[t]]

    c_out.flush()
    e_out.flush()
    print(f"[align] wrote {out_cerra}  {out_era5}")
    return N


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--cerra_dir",  default=str(_here / "cerra"))
    parser.add_argument("--era5_dir",   default=str(_here / "era5"))
    parser.add_argument("--out_dir",    default=str(_here / "processed"))
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=list(range(2009, 2021)),
    )
    args = parser.parse_args()

    splits = {
        "train": [y for y in args.years if 2010 <= y <= 2019],
        "val":   [y for y in args.years if y == 2009],
        "test":  [y for y in args.years if y == 2020],
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split, years in splits.items():
        if not years:
            print(f"[skip] {split}: no years in range")
            continue
        print(f"\n=== {split.upper()} (years {years}) ===")

        cerra_raw = str(out / f"cerra_raw_{split}.npy")
        era5_raw  = str(out / f"era5_raw_{split}.npy")

        cerra_times, *_ = build_cerra_memmap(args.cerra_dir, years, cerra_raw)
        era5_times, h, w = build_era5_memmap(args.era5_dir, years, era5_raw)
        print(f"ERA5 spatial size: {h}×{w}")

        N = align_datasets(
            cerra_times, era5_times,
            cerra_raw, era5_raw, h, w,
            str(out / f"cerra_{split}.npy"),
            str(out / f"era5_{split}.npy"),
        )
        print(f"[{split}] {N} aligned samples,  ERA5 grid {h}×{w}")

    # Save ERA5 spatial size for the dataset classes
    np.save(str(out / "era5_shape.npy"), np.array([h, w]))
    print("\n[done] all splits built.")


if __name__ == "__main__":
    main()
