#!/usr/bin/env python3
"""
Download the UCI Gas Sensor Array Drift dataset into data/gas_sensor/.

Run this LOCALLY (it needs network; the Cowork sandbox is offline). Two routes:
it first tries the official `ucimlrepo` package, then falls back to a direct
zip download + unzip. After it finishes you'll have:

    data/gas_sensor/batch1.dat ... batch10.dat   (LIBSVM format, class 1..6, 128 feats)

Then run the real experiment:

    python experiments/exp_real_data_gas.py --data data/gas_sensor
    python experiments/exp_robustness_sweep.py

License: CC BY 4.0 (Vergara, A. 2012, UCI ML Repository, DOI 10.24432/C5RP6W).
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "gas_sensor"
ZIP_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"


def via_ucimlrepo():
    """Official package route: writes batch*.dat reconstructed from the dataframe.
    Note: ucimlrepo returns a single concatenated frame; we keep it as one file plus
    the per-batch split if the 'batch' column is present."""
    from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
    ds = fetch_ucirepo(id=224)
    X = ds.data.features
    y = ds.data.targets
    OUT.mkdir(parents=True, exist_ok=True)
    # ucimlrepo flattens batches; write a single libsvm file the loader can read as
    # one "batch" — for the time-ordered batch structure prefer the zip route below.
    import numpy as np
    Xv = np.asarray(X, float)
    yv = np.asarray(y).ravel().astype(int)
    with open(OUT / "batch1.dat", "w") as f:
        for row, lab in zip(Xv, yv):
            f.write(f"{int(lab)} " + " ".join(f"{j+1}:{v:.6f}" for j, v in enumerate(row)) + "\n")
    print(f"[ucimlrepo] wrote {OUT/'batch1.dat'} "
          f"({len(yv)} rows). NOTE: batch/time structure is flattened on this route; "
          f"use the zip route for the true 10-batch ordering.")


def via_zip():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[zip] downloading {ZIP_URL} ...")
    raw = urllib.request.urlopen(ZIP_URL, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    n = 0
    for name in z.namelist():
        base = Path(name).name
        if base.lower().startswith("batch") and base.lower().endswith(".dat"):
            (OUT / base).write_bytes(z.read(name))
            n += 1
            print(f"   extracted {base}")
    if n == 0:
        # some mirrors nest the files; extract all *.dat
        for name in z.namelist():
            if name.lower().endswith(".dat"):
                (OUT / Path(name).name).write_bytes(z.read(name)); n += 1
    print(f"[zip] done: {n} batch files -> {OUT}")


def main():
    try:
        via_zip()
    except Exception as e:
        print(f"[zip] failed ({type(e).__name__}: {e}); trying ucimlrepo ...")
        try:
            via_ucimlrepo()
        except Exception as e2:
            print(f"[ucimlrepo] also failed ({type(e2).__name__}: {e2}).")
            print("\nManual fallback:\n"
                  f"  1) open {ZIP_URL}\n"
                  f"  2) unzip so that {OUT}/batch1.dat ... batch10.dat exist\n"
                  f"  3) python experiments/exp_real_data_gas.py --data {OUT}")
            sys.exit(1)
    print("\nNext:\n  python experiments/exp_real_data_gas.py --data data/gas_sensor")


if __name__ == "__main__":
    main()
