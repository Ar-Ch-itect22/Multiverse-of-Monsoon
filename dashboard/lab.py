"""Model experiments for the dashboard: many futures, and the Butterfly Finder. Not graded.

Everything here calls the same network and the same one-day step as the graded server (model/net.py). Nothing in
this file is imported by serve.py.

Many futures: the start day is never known exactly and the model makes a small error every day. So we run the
model N times. Each run starts from a slightly nudged start day and also gets a small random kick after every
simulated day. The kick for each field is a fixed fraction of the model's own measured one-day miss on the tuning
year (see training/calibrate_futures.py, which also checks that the spread of the runs is about as large as the real
miss). Where the N runs agree we are surer; where they split we are not.

Butterfly Finder: pick an outcome (say rain near one city on day 4). We ask the network, by back-propagation, how
much that outcome changes for a small change in every field of every square of the start day. Then we test the
answer: nudge the most important patch, rerun the model the normal way, and compare the real change with the
predicted one. The idea follows published sensitivity studies on learned weather models (Hakim and Masanam 2023;
Vonich and Hakim 2024).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from model.net import NC, NKEY, NLAT, NLON, RAIN, T2M, T2M_MAX, T2M_MIN, step

FIELD_KEYS = ["tp", "t2m", "wind", "msl", "tcwv", "u10", "v10"]            # display fields, same order as the page
INGREDIENT = ["temperature", "rain", "wind", "wind", "pressure", "moisture"]   # plain name of each model channel
# Sizes are in "typical day-to-day swings" of each field (t2m, tp, u10, v10, msl, tcwv). Set by training/calibrate_futures.py.
START_AMP = 0.10                                                            # nudge to the start day
DAY_MISS = [0.44, 0.65, 0.54, 0.55, 0.47, 0.38]                             # the model's one-day miss, measured on the tuning year
KICK_SCALE = 1.0                                                            # daily kick = KICK_SCALE * DAY_MISS
# "how many futures cross this line": direction and levels per display field
LADDERS = {"tp": (">=", [2.5, 10.0, 25.0, 50.0, 100.0]), "t2m": (">=", [30.0, 35.0, 38.0, 40.0, 42.0]),
           "wind": (">=", [6.0, 8.0, 10.0, 12.0, 15.0]), "msl": ("<=", [1004.0, 1000.0, 996.0, 992.0, 988.0]),
           "tcwv": (">=", [40.0, 50.0, 60.0, 65.0, 70.0]), "u10": (">=", [4.0, 6.0, 8.0, 10.0, 12.0]),
           "v10": (">=", [4.0, 6.0, 8.0, 10.0, 12.0])}
TINY = [0.1, 0.02, 0.04, 0.04, 0.1, 0.04, 0.04]                             # a change smaller than this counts as no change


def display(x: torch.Tensor) -> torch.Tensor:
    """[..., 6, H, W] physical units -> [..., 7, H, W] display units (mm/day, deg C, m/s, hPa, kg/m2, m/s, m/s)."""
    t, r, u, v, p, q = x.unbind(-3)
    return torch.stack([r, t - 273.15, torch.sqrt(u * u + v * v + 1e-12), p / 100.0, q, u, v], -3)


def decode(engine, a: torch.Tensor, key: int) -> torch.Tensor:
    """Anomalies [B, 6, H, W] -> physical units. Same maths as Engine.decode, written so gradients can flow."""
    z = engine.clim[key] + a * engine.sigma
    return torch.cat([z[:, T2M].clamp(T2M_MIN, T2M_MAX)[:, None], torch.expm1(z[:, RAIN].clamp(min=0))[:, None], z[:, 2:]], 1)


def rain_floor(engine, key: int) -> torch.Tensor:
    return (-engine.clim[key, RAIN] / engine.sigma[RAIN])[None]             # [1, H, W]: the anomaly that means zero rain


def fast_copy(engine):
    """A copy of the model on the graphics chip when there is one, for the many-futures runs. The graded path stays on CPU."""
    if not torch.cuda.is_available():
        return engine
    e = copy.copy(engine)
    e.net = copy.deepcopy(engine.net).to("cuda")
    e.clim, e.sigma, e.static = engine.clim.to("cuda"), engine.sigma.to("cuda"), engine.static.to("cuda")
    e.fade = None if engine.fade is None else engine.fade.to("cuda")
    return e


def start_history(engine, init: np.ndarray, key0: int) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(init, dtype="f4")).to(engine.clim.device)
    return torch.stack([engine.encode(x[i], (key0 - 2 + i) % NKEY) for i in range(3)])     # [3, 6, H, W]


def roll(engine, hist: torch.Tensor, key0: int, steps: int, kick=None) -> list:
    """hist [B, 3, 6, H, W] anomalies -> list of anomalies [B, 6, H, W] for days 1..steps.
    kick(k) may return a small random change added after day k (used only by the many-futures runs)."""
    out = []
    for k in range(1, steps + 1):
        key_last = torch.full((hist.shape[0],), (key0 + k - 1) % NKEY, dtype=torch.long, device=hist.device)
        floor = rain_floor(engine, (key0 + k) % NKEY)
        nxt = step(engine.net, hist, key_last, engine.static, floor)
        if kick is not None:
            nxt = nxt + kick(k)
            nxt = torch.cat([nxt[:, :RAIN], torch.maximum(nxt[:, RAIN], floor)[:, None], nxt[:, RAIN + 1:]], 1)
        out.append(nxt)
        hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
    return out


def smooth_noise(n: int, seed: int, passes: int = 3) -> torch.Tensor:
    """[n, 6, H, W] random maps, smooth over a few hundred km, each scaled to a spread of 1. Fixed seed: repeatable."""
    z = torch.randn(n * NC, 1, NLAT, NLON, generator=torch.Generator().manual_seed(seed))
    for _ in range(passes):
        z = F.avg_pool2d(F.pad(z, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    return (z / z.std(dim=(2, 3), keepdim=True)).view(n, NC, NLAT, NLON)


@torch.no_grad()
def futures(engine, init: np.ndarray, key0: int, steps: int, n: int, delta=None, seed: int = 7,
            start_amp: float = START_AMP, kick_scale: float = KICK_SCALE, day_miss=DAY_MISS):
    """N forecasts. Run 0 is the plain forecast: no nudge, no kicks. Returns [n, steps + 1, 7, H, W] display units (CPU).
    With delta (a what-if change to the start day) every run gets the same change, the same nudge and the same kicks,
    so run k with the change can be compared with run k without it."""
    dev = engine.clim.device
    x = np.array(init, dtype="f4", copy=True)
    if delta is not None:
        x[-1] = x[-1] + delta
        x[-1, RAIN] = np.maximum(x[-1, RAIN], 0)
    noise = (smooth_noise(n, seed) * start_amp).to(dev)
    noise[0] = 0.0
    hist = start_history(engine, x, key0)[None] + noise[:, None]            # the same nudge on all three input days
    for i in range(3):
        hist[:, i, RAIN] = torch.maximum(hist[:, i, RAIN], rain_floor(engine, (key0 - 2 + i) % NKEY))
    size = (torch.tensor(day_miss, dtype=torch.float32) * kick_scale).view(1, NC, 1, 1)

    def kick(k):
        z = smooth_noise(n, seed + 1000 * k) * size
        z[0] = 0.0
        return z.to(dev)

    days = [decode(engine, hist[:, -1], key0)]
    for k, a in enumerate(roll(engine, hist, key0, steps, kick), 1):
        # the plain forecast is faded exactly as the graded server fades it; each run keeps its own distance from it
        days.append(decode(engine, engine.faded(a[:1], k) + (a - a[:1]), (key0 + k) % NKEY))
    return display(torch.stack(days, 1)).cpu()


def summarise(m: torch.Tensor) -> dict:
    """m [n, T, 7, H, W] -> per-square statistics across the runs."""
    q = torch.quantile(m, torch.tensor([0.1, 0.9]), dim=0)
    counts = {}
    for fi, key in enumerate(FIELD_KEYS):
        sign, levels = LADDERS[key]
        hit = [(m[:, :, fi] >= lv) if sign == ">=" else (m[:, :, fi] <= lv) for lv in levels]
        counts[key] = torch.stack([h.sum(0) for h in hit]).to(torch.uint8)     # [levels, T, H, W]
    return {"mean": m.mean(0), "spread": m.std(0), "low": q[0], "high": q[1], "counts": counts}


def compare(base: torch.Tensor, changed: torch.Tensor) -> dict:
    """Paired runs (same nudges and kicks, with and without the user's change): in how many does each square go up or down?"""
    d = changed - base
    tiny = torch.tensor(TINY).view(1, 1, 7, 1, 1)
    return {"mean_diff": d.mean(0), "up": (d > tiny).sum(0).to(torch.uint8), "down": (d < -tiny).sum(0).to(torch.uint8)}


def to_physical(engine, last: np.ndarray, da: torch.Tensor):
    """A nudge in anomaly units -> the change in physical units, and the nudge that survives 'rain cannot go below zero'."""
    sig = engine.sigma
    delta, eff = (da * sig).clone(), da.clone()
    z0 = torch.log1p(torch.from_numpy(np.ascontiguousarray(last[RAIN], dtype="f4")).clamp(min=0))
    z1 = (z0 + da[RAIN] * sig[RAIN]).clamp(min=0)
    delta[RAIN] = torch.expm1(z1) - torch.expm1(z0)
    eff[RAIN] = (z1 - z0) / sig[RAIN]
    return delta.numpy().astype("f4"), eff


def finder(engine, init: np.ndarray, sc, day: int, field: str, cells, sign: float = 1.0, amp: float = 0.5) -> dict:
    """Which squares of the start day does the outcome (mean of `field` over `cells` on `day`) depend on? Then prove it.
    `engine` must be the CPU engine: the proof reruns go through the same call the graded server uses."""
    key0 = engine.key_from_sincos(sc)
    fi = FIELD_KEYS.index(field)
    idx = torch.tensor(sorted(set(int(c) for c in cells)), dtype=torch.long)

    def outcome(phys: torch.Tensor) -> torch.Tensor:                        # phys [6, H, W]
        return display(phys)[fi].reshape(-1)[idx].mean()

    hist0 = start_history(engine, init, key0)[None]
    d = torch.zeros(NC, NLAT, NLON, requires_grad=True)
    hist = torch.cat([hist0[:, :2], (hist0[:, 2] + d)[:, None]], 1)
    target = outcome(decode(engine, engine.faded(roll(engine, hist, key0, day)[-1], day), (key0 + day) % NKEY)[0])
    g = torch.autograd.grad(target, d)[0].detach()                          # outcome change per one typical swing, per field, per square
    score = g.pow(2).sum(0).sqrt()
    top = float(score.max())
    out = {"field": field, "day": day, "target_cells": idx.tolist(), "base_value": float(target.detach()), "flat": top <= 1e-9}
    if out["flat"]:
        return out

    patch = score >= 0.35 * top
    if int(patch.sum()) > 60:
        patch = score >= torch.topk(score.flatten(), 60).values[-1]
    da = sign * amp * g / g.abs().max() * patch                             # at most `amp` typical swings, in the most useful direction
    delta, eff = to_physical(engine, init[-1], da)
    control, _ = to_physical(engine, init[-1], torch.roll(da, (NLAT // 2, NLON // 2), (1, 2)))

    with torch.no_grad():
        base = torch.from_numpy(engine.simulate(init, sc, day, None)[-1])
        nudged = torch.from_numpy(engine.simulate(init, sc, day, delta)[-1])
        wrong = torch.from_numpy(engine.simulate(init, sc, day, control)[-1])
        j0 = float(outcome(base))
        share = g.pow(2).sum((1, 2))
        share = share / share.sum()
        names = {}
        for c in range(NC):
            names[INGREDIENT[c]] = names.get(INGREDIENT[c], 0.0) + float(share[c])
        big = np.abs(delta).reshape(NC, -1).max(1)
        ii, jj = np.divmod(idx.numpy(), NLON)
        pi, pj = np.nonzero(patch.numpy())
        out.update({
            "score": (score / top).numpy().astype("f4"), "patch": patch.numpy().astype("f4"),
            "diff": (display(nudged)[fi] - display(base)[fi]).numpy().astype("f4"),
            "base_value": j0, "predicted_change": float((g * eff).sum()),
            "real_change": float(outcome(nudged)) - j0, "wrong_place_change": float(outcome(wrong)) - j0,
            "ingredients": sorted(names.items(), key=lambda kv: -kv[1]),
            "nudge_max": {"temperature": float(big[0]), "rain": float(big[1]), "wind": float(max(big[2], big[3])),
                          "pressure": float(big[4] / 100.0), "moisture": float(big[5])},
            "patch_squares": int(patch.sum()),
            "patch_distance": float(np.hypot(pi.mean() - ii.mean(), pj.mean() - jj.mean())),
        })
    return out
