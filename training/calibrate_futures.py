"""Set and check the random sizes used by the many-futures runs (dashboard/lab.py).

Usage (from the repo root):
    python training/calibrate_futures.py                          # measure on the tuning year, try several kick sizes
    python training/calibrate_futures.py --years 2020 2021 --scales 1.0 --save dashboard/futures_check.json

Step 1 measures the model's own one-day miss per field on the chosen years, in "typical day-to-day swings".
Step 2 runs many futures and prints spread / miss per field at days 1, 4 and 8. A value near 1 means the runs
disagree with each other about as much as the plain forecast misses the recorded weather. Below 1 means the runs
are too sure of themselves. Sizes are chosen on the tuning year only; the final-check years are for the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))
import lab  # noqa: E402
from model.net import NKEY, Engine  # noqa: E402

LEADS = (1, 4, 8)
NAMES = ["rain", "temperature", "wind speed", "pressure", "moisture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--runs", type=int, default=24)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    eng = lab.fast_copy(Engine(ROOT / "model"))
    dev = eng.clim.device
    ds = xr.open_dataset(ROOT / "data" / "india_climate_train.nc")
    data = ds["data"].values.astype("f4")
    t = ds["time"].values.astype("datetime64[D]")
    years = t.astype("datetime64[Y]").astype(int) + 1970
    key = ((t - t.astype("datetime64[Y]")).astype(int) + 1) % NKEY
    idx = np.where((years >= min(a.years)) & (years <= max(a.years)))[0]
    wins = [int(i) for i in idx[2:-9:a.stride]]

    # step 1: one-day miss per model field, in typical swings
    sq = torch.zeros(6)
    with torch.no_grad():
        for i in wins:
            hist = lab.start_history(eng, data[i - 2:i + 1], int(key[i]))[None]
            pred = lab.roll(eng, hist, int(key[i]), 1)[0][0]
            truth = eng.encode(torch.from_numpy(data[i + 1]).to(dev), int(key[i + 1]))
            sq += ((pred - truth) ** 2).mean((1, 2)).cpu()
    miss = (sq / len(wins)).sqrt()
    print(f"years {a.years}, {len(wins)} windows, device {dev}")
    print("one-day miss per field (t2m, tp, u10, v10, msl, tcwv):", [round(float(v), 3) for v in miss])
    print("lab.DAY_MISS in use:", lab.DAY_MISS)

    # step 2: spread of the runs against the real miss of the plain forecast
    out = {}
    for sc in a.scales:
        ratio = {d: [] for d in LEADS}
        for i in wins:
            m = lab.futures(eng, data[i - 2:i + 1], int(key[i]), max(LEADS), a.runs, kick_scale=sc)
            truth = lab.display(torch.from_numpy(data[i + 1:i + 1 + max(LEADS)]))
            for d in LEADS:
                spread = m[:, d, :5].var(0).mean((1, 2))
                err = ((m[0, d, :5] - truth[d - 1, :5]) ** 2).mean((1, 2))
                ratio[d].append(torch.stack([spread, err]).numpy())
        row = {}
        for d in LEADS:
            r = np.mean(ratio[d], 0)
            row[d] = [round(float(v), 2) for v in np.sqrt(r[0] / r[1])]
        out[sc] = row
        print(f"kick scale {sc}: " + " | ".join(f"day {d} " + " ".join(f"{n} {v}" for n, v in zip(NAMES, row[d])) for d in LEADS))
    if a.save:
        sc = a.scales[0]
        Path(a.save).write_text(json.dumps({"years": a.years, "windows": len(wins), "runs": a.runs, "kick_scale": sc,
                                            "fields": NAMES, "spread_over_miss": {str(d): out[sc][d] for d in LEADS}}, indent=1))


if __name__ == "__main__":
    main()
