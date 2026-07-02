"""
Download CERRA 10 m wind speed from the Copernicus Climate Data Store (CDS).

Usage:
    python download_cerra.py            # downloads 2009-2020
    python download_cerra.py --years 2018 2019

Requires a ~/.cdsapirc file with your CDS API key.
Dataset: reanalysis-cerra-single-levels (3-hourly, si10)
"""

import argparse
from pathlib import Path
import cdsapi


def download_cerra_wind(years: list[int], output_dir: str = "data/cerra") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year in years:
        output_path = f"{output_dir}/cerra_si10_{year}.nc"
        if Path(output_path).exists():
            print(f"[skip] {output_path} already exists")
            continue

        print(f"[download] CERRA {year} → {output_path}")
        client.retrieve(
            "reanalysis-cerra-single-levels",
            {
                "variable":     ["10m_wind_speed"],
                "level_type":   "surface_or_atmosphere",
                "data_type":    ["reanalysis"],
                "product_type": "analysis",
                "year":         [str(year)],
                "month":        [f"{m:02d}" for m in range(1, 13)],
                "day":          [f"{d:02d}" for d in range(1, 32)],
                "time": [
                    "00:00", "03:00", "06:00", "09:00",
                    "12:00", "15:00", "18:00", "21:00",
                ],
                "data_format": "netcdf",
            },
        ).download(output_path)
        print(f"[done]     {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=list(range(2009, 2021)),
        help="Years to download (default: 2009-2020)"
    )
    parser.add_argument("--output_dir", default="data/cerra")
    args = parser.parse_args()
    download_cerra_wind(args.years, args.output_dir)
