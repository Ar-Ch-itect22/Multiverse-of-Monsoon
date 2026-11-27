"""Fit the fade: how much of the forecast's gap from normal weather to keep at each day ahead, per field.

Usage (from the repo root, after training):
    python training/fit_fade.py              # fits on the tuning year and writes the table into model/norm.npz
    python training/fit_fade.py --strength 0 # removes the fade (keep everything at every day)

Why: a forecast gets less sure with every day. Past a few days part of its gap from normal weather is noise, and
keeping all of it makes the error larger than simply showing normal weather. So the server keeps a share of the gap
that falls with the day ahead. The share for each day and field is the one that gives the smallest error on the
tuning year (2019). The final-check years are never used here. The network's own state is not touched: the fade is
applied only to what is sent out, so rollouts stay exactly as stable as before.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.net import NC, NKEY, NLAT, RAIN, Engine, step  # noqa: E402

DATA = ROOT / "data" / "india_climate_train.nc"
TUNE = 2019
DAYS = 10
SHARES = np.linspace(0.30, 1.0, 29)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength", type=float, default=1.0, help="1 = the fitted fade, 0 = no fade, in between = part of it")
    ap.add_argument("--model-dir", default=str(ROOT / "model"))
    ap.add_argument("--fade-rain", action="store_true", help="also fade rain (off by default, see the note below)")
    a = ap.parse_args()
    eng = Engine(Path(a.model_dir))
    ds = xr.open_dataset(DATA)
    x = torch.from_numpy(ds["data"].values.astype("f4"))
    t = ds["time"].values.astype("datetime64[D]")
    years = t.astype("datetime64[Y]").astype(int) + 1970
    key = torch.from_numpy((((t - t.astype("datetime64[Y]")).astype(int) + 1) % NKEY).astype("int64"))
    z = x.clone()
    z[:, RAIN] = torch.log1p(z[:, RAIN].clamp(min=0))
    cl = eng.clim[key]                                                   # normal weather for every day, model space
    an = (z - cl) / eng.sigma
    floor = -cl[:, RAIN] / eng.sigma[RAIN]
    i = np.where(years == TUNE)[0]
    idx = torch.arange(int(i[0]) + 2, int(i[-1]) - DAYS + 1)
    lat_w = torch.cos(torch.deg2rad(torch.linspace(6.5, 37.5, NLAT)))
    lat_w = (lat_w / lat_w.mean()).view(1, NLAT, 1)

    fade = np.ones((DAYS, NC), "f4")
    with torch.no_grad():
        hist = an[idx[:, None] + torch.arange(-2, 1)]
        for k in range(1, DAYS + 1):
            nxt = step(eng.net, hist, key[idx + k - 1], eng.static, floor[idx + k])
            hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
            for c in range(NC):
                best, best_err = 1.0, None
                for s in SHARES:
                    v = cl[idx + k, c] + float(s) * nxt[:, c] * eng.sigma[c]
                    if c == RAIN:
                        v = torch.expm1(v.clamp(min=0))
                    err = float((((v - x[idx + k, c]) ** 2) * lat_w).mean())
                    if best_err is None or err < best_err:
                        best, best_err = float(s), err
                fade[k - 1, c] = best
    fade = np.minimum.accumulate(fade, axis=0)                           # never keep more on a later day than on an earlier one
    if not a.fade_rain:
        fade[:, RAIN] = 1.0                                              # fading rain cost heavy-rain hits and map sharpness on 2019
    fade = (1.0 - a.strength * (1.0 - fade)).astype("f4")
    names = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
    print(f"fitted on {TUNE}, {len(idx)} windows, strength {a.strength}. Share of the gap kept, per day ahead:")
    print("day   " + "  ".join(f"{n:>5}" for n in names))
    for k in range(DAYS):
        print(f"{k + 1:>3}   " + "  ".join(f"{v:5.2f}" for v in fade[k]))
    d = dict(np.load(Path(a.model_dir) / "norm.npz"))
    d["fade"] = fade
    np.savez(Path(a.model_dir) / "norm.npz", **d)
    print("written to", Path(a.model_dir) / "norm.npz")


if __name__ == "__main__":
    main()
