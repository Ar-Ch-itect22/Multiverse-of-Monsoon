"""Train the VARUNA model on the organizers' dataset.

Usage (from the repo root, GPU recommended):
    python training/train.py                       # all three stages
    python training/train.py --epochs 12 10 6      # epochs for stage A, B, C

Stages: A = one day ahead, B = 4 days in a row, C = 8 days in a row. Training on the model's own multi-day output
is what keeps a 10-day forecast stable. After every epoch the model is scored on the tuning year and the weights
are saved only when the score improves.

Data split: train 2005-2018, tune 2019. The years 2020-2021 are never touched here; training/scoreboard.py uses them.
Writes model/varuna.pt (weights) and model/norm.npz (366-day-or-less table of normal weather, spread, fixed maps).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.net import NC, NKEY, NLAT, NLON, RAIN, LocalNet, step  # noqa: E402

DATA = ROOT / "data" / "india_climate_train.nc"
OUT = ROOT / "model"
TRAIN, TUNE = (2005, 2018), (2019, 2019)
LEADS = (1, 4, 8)


def smooth_circular(a: np.ndarray, half: int = 15) -> np.ndarray:
    out = np.zeros_like(a)
    for k in range(-half, half + 1):
        out += np.roll(a, k, axis=0)
    return out / (2 * half + 1)


def build(width: int, blocks: int):
    ds = xr.open_dataset(DATA)
    x = ds["data"].values.astype("f4")                                   # [T, 6, H, W] physical units
    st = ds["static"].values.astype("f4")                                # [2, H, W] mask, height in metres
    t = ds["time"].values.astype("datetime64[D]")
    years = t.astype("datetime64[Y]").astype(int) + 1970
    key = ((t - t.astype("datetime64[Y]")).astype(int) + 1) % NKEY       # day of year % 365
    z = x.copy()
    z[:, RAIN] = np.log1p(np.clip(z[:, RAIN], 0, None))                  # model space
    tr = (years >= TRAIN[0]) & (years <= TRAIN[1])
    clim = np.stack([z[tr & (key == k)].mean(0) for k in range(NKEY)])
    clim_phys = np.stack([x[tr & (key == k)].mean(0) for k in range(NKEY)])
    clim, clim_phys = smooth_circular(clim).astype("f4"), smooth_circular(clim_phys).astype("f4")
    anom = z - clim[key]
    sigma = anom[tr].std(axis=(0, 2, 3)).astype("f4")
    lat = np.linspace(-1, 1, NLAT, dtype="f4")[:, None].repeat(NLON, 1)
    lon = np.linspace(-1, 1, NLON, dtype="f4")[None, :].repeat(NLAT, 0)
    static = np.stack([st[0], st[1] / 2000.0, lat, lon]).astype("f4")
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "norm.npz", clim=clim, sigma=sigma, static=static, width=width, blocks=blocks)
    return x, z, years, key, clim, clim_phys, sigma, static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs=3, default=[12, 10, 6])
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, z, years, key, clim, clim_phys, sigma, static = build(a.width, a.blocks)
    T = x.shape[0]
    sig = torch.tensor(sigma, device=dev).view(1, NC, 1, 1)
    X = torch.tensor(x, device=dev)                                      # physical truth
    CL = torch.tensor(clim[key], device=dev)                             # [T, 6, H, W] normal for each day, model space
    A = (torch.tensor(z, device=dev) - CL) / sig                         # anomalies
    FLOOR = -CL[:, RAIN] / sig[0, RAIN]                                  # rain anomaly that means zero rain
    CLP = torch.tensor(clim_phys[key], device=dev)                       # physical-space normal, the baseline to beat
    KEY = torch.tensor(key, device=dev)
    STATIC = torch.tensor(static, device=dev)
    lat_w = torch.cos(torch.deg2rad(torch.linspace(6.5, 37.5, NLAT, device=dev)))
    lat_w = (lat_w / lat_w.mean()).view(1, 1, NLAT, 1)

    def idx(rng, horizon, stride=1):
        i = np.where((years >= rng[0]) & (years <= rng[1]))[0]
        return torch.arange(int(i[0]) + 2, int(i[-1]) - horizon + 1, stride, device=dev)

    net = LocalNet(a.width, a.blocks).to(dev)
    print(f"device {dev}  params {sum(p.numel() for p in net.parameters()):,}  reach {net.reach} cells/day", flush=True)
    off = torch.arange(-2, 1, device=dev)

    def rollout(tidx, k_steps, noise=0.0):
        hist = A[tidx[:, None] + off]
        if noise:
            hist = hist + noise * torch.randn_like(hist)
        preds = []
        for k in range(1, k_steps + 1):
            nxt = step(net, hist, KEY[tidx + k - 1], STATIC, FLOOR[tidx + k])
            preds.append(nxt)
            hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
        return preds

    def decode(an, tk):
        zz = CL[tk] + an * sig
        return torch.cat([zz[:, :RAIN], torch.expm1(zz[:, RAIN:RAIN + 1].clamp(min=0)), zz[:, RAIN + 1:]], 1)

    def rmse(p, t_):
        return torch.sqrt((((p - t_) ** 2) * lat_w).mean(dim=(0, 2, 3)))

    @torch.no_grad()
    def evaluate():
        net.eval()
        tidx = idx(TUNE, max(LEADS), stride=2)
        preds = rollout(tidx, max(LEADS))
        ss, gate = [], 0.0
        for d in LEADS:
            tk = tidx + d
            p, tr_ = decode(preds[d - 1], tk), X[tk]
            r, rp, rc = rmse(p, tr_), rmse(X[tidx], tr_), rmse(CLP[tk], tr_)
            ss.append((1 - r / torch.minimum(rp, rc)).mean().item())
            if d == 4:
                gate = (1 - r / rp).mean().item()
        net.train()
        return float(np.mean(ss)), ss, gate

    stages = [("A", 1, 1e-3, 0.02, a.epochs[0]), ("B", 4, 2e-4, 0.0, a.epochs[1]), ("C", 8, 1e-4, 0.0, a.epochs[2])]
    best = -1e9
    t0 = time.time()
    for name, k_steps, lr, noise, epochs in stages:
        if epochs <= 0:
            continue
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        tidx_all = idx(TRAIN, k_steps)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (len(tidx_all) // a.batch + 1))
        for ep in range(1, epochs + 1):
            perm = tidx_all[torch.randperm(len(tidx_all), device=dev)]
            tot, nb = 0.0, 0
            for b in range(0, len(perm), a.batch):
                tidx = perm[b:b + a.batch]
                preds = rollout(tidx, k_steps, noise)
                loss = 0.0
                for k, p in enumerate(preds, start=1):
                    tk = tidx + k
                    tgt = A[tk]
                    w = torch.ones_like(tgt)
                    w[:, RAIN] = 1.0 + 2.0 * (X[tk, RAIN] > 10.0).float()      # heavy rain counts three times
                    mse = (((p - tgt) ** 2) * w * lat_w).mean()
                    zp, zt = CL[tk] + p * sig, CL[tk] + tgt * sig
                    sharp = ((torch.log(zp.flatten(2).std(2) + 1e-6) - torch.log(zt.flatten(2).std(2) + 1e-6)) ** 2).mean()
                    loss = loss + mse + 0.1 * sharp
                loss = loss / k_steps
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item(); nb += 1
            score, ss, gate = evaluate()
            tag = ""
            if score > best:
                best = score
                torch.save({k: v.cpu() for k, v in net.state_dict().items()}, OUT / "varuna.pt")
                tag = "  saved"
            print(f"stage {name} ep {ep:2d}  loss {tot / nb:.4f}  tune skill mean {score:+.4f}  "
                  f"d1 {ss[0]:+.3f} d4 {ss[1]:+.3f} d8 {ss[2]:+.3f}  day4 vs repeat-yesterday {gate:+.3f}  "
                  f"{time.time() - t0:.0f}s{tag}", flush=True)
    print(f"done. best tune skill {best:+.4f}", flush=True)


if __name__ == "__main__":
    main()
