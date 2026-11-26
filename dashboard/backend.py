"""VARUNA dashboard backend. Not graded.

Run from the repo root:
    python dashboard/backend.py                 then open http://127.0.0.1:8050

It loads the same model as the graded server (the model/ folder) and runs it live for the map. It never changes
serve.py. Starting weather is read from data/india_climate_train.nc, the organizers' file, which stays out of the repo.

Routes (all under /api, plus the three static files of the page):
    GET  /api/meta    grid, land mask, date range, saved scoreboard result
    POST /api/run     one forecast, and the same forecast again with the user's change applied
    POST /api/proof   speed, zero-change test, repeat test, and the inputs the page needs to test the graded server
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from model.net import NC, NLAT, NLON, RAIN, T2M, T2M_MAX, T2M_MIN, Engine  # noqa: E402

DATA = ROOT / "data" / "india_climate_train.nc"
SCORES = HERE / "scores.json"
MAX_STEPS = 10
GRADED_PORT = 8000
LOCK = threading.Lock()
ENGINE = Engine(ROOT / "model")
PARAMS = int(sum(p.numel() for p in ENGINE.net.parameters()))


class Source:
    """Starting weather and what really happened next, from the organizers' file."""

    def __init__(self, path: Path):
        import xarray as xr
        ds = xr.open_dataset(path)
        self.data = ds["data"].values.astype("f4")                    # [T, 6, H, W] physical units
        self.time = ds["time"].values.astype("datetime64[D]")
        ds.close()

    def index(self, date: str) -> int:
        hit = np.where(self.time == np.datetime64(date, "D"))[0]
        if len(hit) == 0 or hit[0] < 2:
            raise KeyError(date)
        return int(hit[0])

    def get(self, date: str, steps: int):
        i = self.index(date)
        d = self.time[i]
        doy = int((d - d.astype("datetime64[Y]")).astype(int)) + 1
        year = int(d.astype("datetime64[Y]").astype(int)) + 1970
        tag = "train" if year <= 2018 else "tune" if year == 2019 else "unseen"
        return self.data[i - 2:i + 1].copy(), self.data[i + 1:i + 1 + steps].copy(), doy, tag


SRC = Source(DATA) if DATA.exists() else None
app = FastAPI()


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def sincos(doy: int):
    ang = 2.0 * np.pi * doy / 365.0
    return [float(np.sin(ang)), float(np.cos(ang))]


def build_delta(last: np.ndarray, shock: dict) -> np.ndarray:
    """delta = region * (add + mul * today's value), per field. This is the same form as the two official shocks."""
    mask = np.asarray(shock["mask"], "f4").reshape(NLAT, NLON).clip(0, 1)
    add = np.asarray(shock["add"], "f4").reshape(NC, 1, 1)
    mul = np.asarray(shock["mul"], "f4").reshape(NC, 1, 1)
    delta = (mask[None] * (add + mul * last)).astype("f4")
    delta[RAIN] = np.maximum(delta[RAIN], -last[RAIN])                # rain cannot go below zero
    if not np.isfinite(delta).all():
        raise ValueError("change is not a finite number")
    return delta


def fail(msg: str, code: int = 400):
    return JSONResponse(status_code=code, content={"error": msg})


@app.get("/api/meta")
def meta():
    st = ENGINE.static.numpy()
    out = {"nlat": NLAT, "nlon": NLON, "lat0": 6.5, "lon0": 65.5, "mask": b64(st[0]), "height": b64(st[1] * 2000.0),
           "params": PARAMS, "reach": int(ENGINE.net.reach), "max_steps": MAX_STEPS, "data_ok": SRC is not None,
           "scores": json.loads(SCORES.read_text()) if SCORES.exists() else None}
    if SRC is None:
        out["error"] = "Starting weather not found. Put india_climate_train.nc in the data folder and restart."
        return out
    out["date_min"], out["date_max"] = str(SRC.time[2]), str(SRC.time[-1])
    return out


@app.post("/api/run")
async def run(request: Request):
    if SRC is None:
        return fail("no starting weather on disk", 503)
    try:
        body = await request.json()
        date = str(body["date"])
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError("bad date")
        steps = max(1, min(int(body.get("steps", MAX_STEPS)), MAX_STEPS))
        init, truth, doy, tag = SRC.get(date, steps)
        shock = body.get("shock")
        delta = build_delta(init[-1], shock) if shock else None
    except KeyError:
        return fail("that day is not in the data", 404)
    except Exception as e:  # noqa: BLE001
        return fail(f"bad request: {e}")
    sc = sincos(doy)
    with LOCK:
        t0 = time.perf_counter()
        base = ENGINE.simulate(init, sc, steps, None)
        base_ms = (time.perf_counter() - t0) * 1000.0
        shocked, shock_ms = None, None
        if delta is not None:
            t0 = time.perf_counter()
            shocked = ENGINE.simulate(init, sc, steps, delta)
            shock_ms = (time.perf_counter() - t0) * 1000.0
    return {"date": date, "doy": doy, "tag": tag, "steps": steps, "init": b64(init[-1]), "base": b64(base),
            "shocked": None if shocked is None else b64(shocked), "delta": None if delta is None else b64(delta),
            "truth": b64(truth), "truth_days": int(truth.shape[0]), "base_ms": base_ms, "shock_ms": shock_ms}


@app.post("/api/proof")
async def proof(request: Request):
    if SRC is None:
        return fail("no starting weather on disk", 503)
    try:
        body = await request.json()
        init, _, doy, _ = SRC.get(str(body["date"]), MAX_STEPS)
    except Exception:  # noqa: BLE001
        return fail("that day is not in the data", 404)
    sc = sincos(doy)
    with LOCK:
        times, runs = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            runs.append(ENGINE.simulate(init, sc, 10, None))
            times.append((time.perf_counter() - t0) * 1000.0)
        zero = ENGINE.simulate(init, sc, 10, np.zeros((NC, NLAT, NLON), "f4"))
    a = runs[0]
    valid = bool(np.isfinite(a).all() and (a[:, RAIN] >= 0).all()
                 and (a[:, T2M] >= T2M_MIN).all() and (a[:, T2M] <= T2M_MAX).all())
    return {"ms_10_days": float(np.median(times)), "limit_ms": 5000,
            "zero_change_max_diff": float(np.abs(a - zero).max()),
            "repeat_max_diff": float(max(np.abs(a - r).max() for r in runs[1:])),
            "valid_output": valid,
            # the page sends these to the graded server itself and compares the two answers
            "init_states": b64(init), "doy_sincos": sc, "rollout": b64(a), "graded_port": GRADED_PORT}


def page(name: str, kind: str):
    return FileResponse(HERE / name, media_type=kind, headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    return page("index.html", "text/html")


@app.get("/app.js")
def app_js():
    return page("app.js", "application/javascript")


@app.get("/style.css")
def style_css():
    return page("style.css", "text/css")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--graded-port", type=int, default=8000)
    a = ap.parse_args()
    GRADED_PORT = a.graded_port
    print(f"VARUNA dashboard: http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
