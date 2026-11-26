"""VARUNA core model: a small local-view network that predicts how tomorrow's weather departs from the seasonal normal.

Design rules (each one exists because of a scoring rule):
  * Local view only. One 3x3 stem plus one 3x3 convolution per block, so information moves at most REACH cells
    per simulated day. No pooling, no attention, no map-wide normalisation: those would spread a local shock
    across the whole map at once.
  * Works on anomalies: the input is (state - normal for that day of year) / spread, and the network predicts the
    change in that anomaly. The output layer starts at zero, so an untrained model is exactly "repeat yesterday".
  * Rain is handled as log(1 + rain) and can never go below zero.
  * Fully deterministic at inference: no dropout, no random numbers.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NC, NLAT, NLON = 6, 32, 36
RAIN, T2M = 1, 0
T2M_MIN, T2M_MAX = 233.15, 333.15      # -40 C .. +60 C, the validity gate
NKEY = 365                              # day-of-year bins: key = day_of_year % 365
N_STATIC = 4                            # land-sea mask, scaled height, lat plane, lon plane
C_IN = 3 * NC + N_STATIC + 2


class PixelNorm(nn.Module):
    """Normalises across channels at each cell separately, so it never mixes information between cells."""

    def __init__(self, c: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        v = x.var(1, keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + 1e-5) * self.g + self.b


class Block(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, padding_mode="replicate")
        self.norm = PixelNorm(c)
        self.up = nn.Conv2d(c, 2 * c, 1)
        self.down = nn.Conv2d(2 * c, c, 1)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(self.norm(self.conv(x)))))


class LocalNet(nn.Module):
    def __init__(self, width: int = 96, blocks: int = 5):
        super().__init__()
        self.stem = nn.Conv2d(C_IN, width, 3, padding=1, padding_mode="replicate")
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, NC, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.reach = 1 + blocks          # cells per simulated day; must stay at or below 8

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def doy_planes(key: torch.Tensor) -> torch.Tensor:
    """key [B] in 0..364 -> [B, 2, H, W] planes holding sin and cos of the day of year."""
    ang = 2 * math.pi * key.float() / 365.0
    sc = torch.stack([torch.sin(ang), torch.cos(ang)], 1)
    return sc[:, :, None, None].expand(-1, -1, NLAT, NLON)


def step(net: LocalNet, hist: torch.Tensor, key_last: torch.Tensor, static: torch.Tensor, rain_floor: torch.Tensor):
    """One simulated day.

    hist       [B, 3, 6, H, W] anomalies, oldest first
    key_last   [B] day-of-year key of the newest day in hist
    static     [4, H, W]
    rain_floor [B, H, W] lowest allowed rain anomaly for the NEXT day (the value that means zero rain)
    returns    [B, 6, H, W] anomaly for the next day
    """
    b = hist.shape[0]
    x = torch.cat([hist.reshape(b, 3 * NC, NLAT, NLON), static.expand(b, -1, -1, -1), doy_planes(key_last)], 1)
    nxt = hist[:, -1] + net(x)
    rain = torch.maximum(nxt[:, RAIN], rain_floor)
    return torch.cat([nxt[:, :RAIN], rain[:, None], nxt[:, RAIN + 1:]], 1)


class Engine:
    """Loads the trained weights and the table of normal weather once, then runs forecasts on the CPU."""

    def __init__(self, folder: Path, threads: int = 4):
        torch.set_num_threads(threads)
        d = np.load(Path(folder) / "norm.npz")
        self.clim = torch.from_numpy(d["clim"])            # [365, 6, H, W] model space (rain as log1p)
        self.sigma = torch.from_numpy(d["sigma"]).view(NC, 1, 1)
        self.static = torch.from_numpy(d["static"])        # [4, H, W]
        # share of the gap from normal kept at each day ahead, per field [days, 6]; absent = keep everything
        self.fade = torch.from_numpy(d["fade"]) if "fade" in d.files else None
        self.net = LocalNet(int(d["width"]), int(d["blocks"]))
        self.net.load_state_dict(torch.load(Path(folder) / "varuna.pt", map_location="cpu"))
        self.net.eval()

    def encode(self, x: torch.Tensor, key: int) -> torch.Tensor:
        z = x.clone()
        z[RAIN] = torch.log1p(z[RAIN].clamp(min=0))
        return (z - self.clim[key]) / self.sigma

    def decode(self, a: torch.Tensor, key: int) -> torch.Tensor:
        z = self.clim[key] + a * self.sigma
        z[RAIN] = torch.expm1(z[RAIN].clamp(min=0))
        z[T2M] = z[T2M].clamp(T2M_MIN, T2M_MAX)
        return z

    def faded(self, a: torch.Tensor, k: int) -> torch.Tensor:
        """Anomaly [..., 6, H, W] for day k ahead -> the share of it that is sent out. A forecast gets less sure with
        every day, so past a few days only part of its gap from normal weather is kept (fitted on the tuning year by
        training/fit_fade.py). Only the output is changed; the state that is rolled forward is not."""
        if self.fade is None or k < 1:
            return a
        return a * self.fade[min(k, self.fade.shape[0]) - 1].view(NC, 1, 1)

    @staticmethod
    def key_from_sincos(doy_sincos) -> int:
        ang = math.atan2(float(doy_sincos[0]), float(doy_sincos[1]))
        return int(round(ang / (2 * math.pi) * 365.0)) % NKEY

    @torch.inference_mode()
    def simulate(self, init_states, doy_sincos, steps: int, perturbation=None) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(init_states, dtype="f4")).clone()
        p = torch.zeros(NC, NLAT, NLON) if perturbation is None else torch.from_numpy(
            np.ascontiguousarray(perturbation, dtype="f4"))
        x[-1] = x[-1] + p                                   # zeros and null follow the same arithmetic path
        x[-1, RAIN] = x[-1, RAIN].clamp(min=0)
        key0 = self.key_from_sincos(doy_sincos)
        hist = torch.stack([self.encode(x[i], (key0 - 2 + i) % NKEY) for i in range(3)])[None]
        out = []
        for k in range(1, int(steps) + 1):
            kn = (key0 + k) % NKEY
            floor = (-self.clim[kn, RAIN] / self.sigma[RAIN])[None]
            nxt = step(self.net, hist, torch.tensor([(key0 + k - 1) % NKEY]), self.static, floor)
            out.append(self.decode(self.faded(nxt[0], k).clone(), kn))
            hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
        res = torch.stack(out)
        return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).numpy()
