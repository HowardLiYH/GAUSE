#!/usr/bin/env python3
"""
Download REAL NYC Yellow Taxi trip data and aggregate to hourly trip counts.

NO API KEY REQUIRED. The NYC Taxi & Limousine Commission publishes monthly
trip-record parquet files on a public CloudFront bucket.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Files:  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet

This closes the reproducibility gap for data/traffic/nyc_taxi_real_hourly.csv,
which previously had no fetch script. It reproduces the same schema
(timestamp, trip_count, regime) with clock-based (causal) regime labels that
match src/domains/traffic.py::detect_regime.

Usage:
    python scripts/download_real_traffic.py                 # Jan-Apr 2023 (default)
    python scripts/download_real_traffic.py 2023-01 2023-04 # explicit month range
"""

import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests")
    import requests

# pandas needs a parquet engine; install pyarrow if missing.
if importlib.util.find_spec("pyarrow") is None:
    os.system(f"{sys.executable} -m pip install pyarrow")

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path(__file__).parent.parent / "data" / "traffic"
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def months_between(start: str, end: str):
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    out = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def clock_regime(hour: int, weekday: int) -> str:
    """Causal regime: derived from the clock only (matches src/domains/traffic.py)."""
    if weekday >= 5:
        return "weekend"
    if 7 <= hour <= 9:
        return "morning_rush"
    if 17 <= hour <= 19:
        return "evening_rush"
    if 10 <= hour <= 16:
        return "midday"
    if 0 <= hour <= 5:
        return "night"
    return "transition"


def download_month(month: str) -> pd.DataFrame:
    """Download one month and aggregate yellow-taxi pickups to hourly counts."""
    url = f"{BASE_URL}/yellow_tripdata_{month}.parquet"
    local = OUT_DIR / f"_raw_yellow_{month}.parquet"
    if not local.exists():
        print(f"  downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        local.write_bytes(resp.content)
    # Only need the pickup timestamp column.
    df = pd.read_parquet(local, columns=["tpep_pickup_datetime"])
    ts = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce").dropna()
    # Restrict to the target month (files contain a few stray out-of-range rows).
    ts = ts[ts.dt.strftime("%Y-%m") == month]
    hourly = ts.dt.floor("h").value_counts().sort_index()
    out = hourly.reset_index()
    out.columns = ["timestamp", "trip_count"]
    return out


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2023-04"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for month in months_between(start, end):
        try:
            frames.append(download_month(month))
        except (requests.RequestException, OSError, ValueError) as e:
            print(f"  [warn] {month}: {e}")
    if not frames:
        print("No data downloaded.")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Fill any missing hours with 0 so the series is regular.
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full, fill_value=0).rename_axis("timestamp").reset_index()
    df["trip_count"] = df["trip_count"].astype(int)
    df["regime"] = [clock_regime(ts.hour, ts.weekday()) for ts in df["timestamp"]]

    out_path = OUT_DIR / "nyc_taxi_real_hourly.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} hourly rows to {out_path}")
    print(f"  range: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"  trip_count: min={df['trip_count'].min()} max={df['trip_count'].max()} "
          f"mean={df['trip_count'].mean():.0f}")
    print(f"  regimes: {df['regime'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
