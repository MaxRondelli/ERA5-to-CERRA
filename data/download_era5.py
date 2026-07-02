"""
Download ERA5 10 m wind components (u10, v10) from CDS.

The geographic area covers Italy and the surrounding Mediterranean region.
Adjust AREA if you want a different domain.

Usage:
    python download_era5.py
    python download_era5.py --years 2018 2019 --area 48 5 36 18
"""

import argparse
from pathlib import Path
import cdsapi


# [North, West, South, East] — Italy / Mediterranean region
DEFAULT_AREA = [48.0, 5.0, 36.0, 18.0]

def download_era5_wind(
    years:      list[int],
    area:       list[float] = DEFAULT_AREA,
    output_dir: str         = "data/era5",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year in years:
        output_path = f"{output_dir}/era5_wind_{year}.nc"
        if Path(output_path).exists():
            print(f"[skip] {output_path} already exists")
            continue

        print(f"[download] ERA5 {year} → {output_path}")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year":  [str(year)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day":   [f"{d:02d}" for d in range(1, 32)],
                "time": [
                    "00:00", "03:00", "06:00", "09:00",
                    "12:00", "15:00", "18:00", "21:00",
                ],
                "area":            area,
                "data_format":     "netcdf",
                "download_format": "unarchived",
            },
        ).download(output_path)
        print(f"[done]     {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=list(range(2009, 2021)),
    )
    parser.add_argument(
        "--area", type=float, nargs=4,
        default=DEFAULT_AREA,
        metavar=("N", "W", "S", "E"),
    )
    parser.add_argument("--output_dir", default="data/era5")
    args = parser.parse_args()
    download_era5_wind(args.years, args.area, args.output_dir)
