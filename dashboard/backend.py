"""VARUNA dashboard backend. Not graded.

Run from the repo root:
    python dashboard/backend.py                 then open http://127.0.0.1:8050

It loads the same model as the graded server (the model/ folder) and runs it live for the map. It never changes
serve.py. Starting weather is read from data/india_climate_train.nc, the organizers' file, which stays out of the repo.

Routes (all under /api, plus the three static files of the page):
    GET  /api/meta    grid, land mask, date range, saved scoreboard result
    POST /api/start   starting weather and the user's change; the page sends these to the graded server itself
    POST /api/run     fallback when the graded server is off: the same forecast run here
    POST /api/proof   speed, zero-change test, repeat test, and the inputs the page needs to test the graded server
    POST /api/futures 50 forecasts from nudged starts: how many cross a line, the range, the spread (see lab.py)
    POST /api/alerts  role alerts: plain rules checked in each of the 50 futures (see alerts.py)
    POST /api/exam    physics exam: live checks of how the model behaves (see exam.py)
    POST /api/finder  Butterfly Finder: what an outcome depends on, with a rerun that tests it (see lab.py)
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
from torch import stack as torch_stack
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import alerts as alert_rules  # noqa: E402
import exam as physics_exam  # noqa: E402
import lab  # noqa: E402
from model.net import NC, NLAT, NLON, RAIN, T2M, T2M_MAX, T2M_MIN, Engine  # noqa: E402

DATA = ROOT / "data" / "india_climate_train.nc"
SCORES = HERE / "scores.json"
CHECK = HERE / "futures_check.json"
MAX_STEPS = 10
N_RUNS = 50
GRADED_PORT = 8000
LOCK = threading.Lock()
ENGINE = Engine(ROOT / "model")
FAST = lab.fast_copy(ENGINE)                     # same weights on the graphics chip when there is one; used only for the 50 futures
CACHE: dict = {}                                 # the last many-futures result, for the per-square chart
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
UNUSUAL = None                                   # (value per day, sorted training values), filled on first use
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


def rain_normal() -> list:
    """Normal rain for each day of the year over the India box (land between 8 and 30 N, 70 and 90 E), from the
    training years, smoothed over a month. The page compares the forecast with it to name the monsoon phase."""
    if "rain_normal" not in CACHE:
        lat, lon = 6.5 + np.arange(NLAT), 65.5 + np.arange(NLON)
        box = ((lat >= 8) & (lat <= 30))[:, None] & ((lon >= 70) & (lon <= 90))[None, :] & (ENGINE.static[0].numpy() >= 0.5)
        years = SRC.time.astype("datetime64[Y]").astype(int) + 1970
        key = ((SRC.time - SRC.time.astype("datetime64[Y]")).astype(int) + 1) % 365
        daily = SRC.data[:, RAIN][:, box].mean(axis=1)
        raw = np.array([daily[(years <= 2018) & (key == k)].mean() for k in range(365)])
        CACHE["rain_normal"] = [float(v) for v in sum(np.roll(raw, s) for s in range(-15, 16)) / 31.0]
    return CACHE["rain_normal"]


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
    out["rain_normal"] = rain_normal()
    return out


@app.post("/api/start")
async def start(request: Request):
    """Starting weather for a day, what really happened next, and the user's change as a plain delta. No model run here:
    the page sends these to the graded server itself."""
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
    return {"date": date, "doy": doy, "tag": tag, "steps": steps, "init_states": b64(init), "doy_sincos": sincos(doy),
            "delta": None if delta is None else b64(delta), "truth": b64(truth), "truth_days": int(truth.shape[0]),
            "graded_port": GRADED_PORT}


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


def b64u8(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a.numpy() if hasattr(a, "numpy") else a, dtype="u1").tobytes()).decode("ascii")


def by_field(t) -> str:
    """[T, 7, H, W] -> base64 of [7, T, H, W], the layout the page reads."""
    return b64(t.permute(1, 0, 2, 3).numpy())


@app.post("/api/futures")
async def futures(request: Request):
    """50 forecasts from nudged starts with daily random kicks. With a change, the same 50 again with the change applied."""
    if SRC is None:
        return fail("no starting weather on disk", 503)
    try:
        body = await request.json()
        init, _, doy, _ = SRC.get(str(body["date"]), MAX_STEPS)
        shock = body.get("shock")
        delta = build_delta(init[-1], shock) if shock else None
    except KeyError:
        return fail("that day is not in the data", 404)
    except Exception as e:  # noqa: BLE001
        return fail(f"bad request: {e}")
    key0 = ENGINE.key_from_sincos(sincos(doy))
    with LOCK:
        t0 = time.perf_counter()
        base = lab.futures(FAST, init, key0, MAX_STEPS, N_RUNS)
        changed = lab.futures(FAST, init, key0, MAX_STEPS, N_RUNS, delta=delta) if delta is not None else None
        ms = (time.perf_counter() - t0) * 1000.0
        s = lab.summarise(base)
        pair = lab.compare(base, changed) if changed is not None else None
        CACHE.update(id=CACHE.get("id", 0) + 1, base=base, changed=changed)
        out_id = CACHE["id"]
    return {"id": out_id, "runs": N_RUNS, "days": MAX_STEPS + 1, "fields": lab.FIELD_KEYS, "ms": ms,
            "device": "graphics chip" if FAST is not ENGINE else "processor",
            "mean": by_field(s["mean"]), "spread": by_field(s["spread"]), "low": by_field(s["low"]), "high": by_field(s["high"]),
            "ladders": {k: {"sign": v[0], "levels": v[1]} for k, v in lab.LADDERS.items()},
            "counts": {k: b64u8(v) for k, v in s["counts"].items()},
            "pair": None if pair is None else {"mean_diff": by_field(pair["mean_diff"]), "up": b64u8(pair["up"].permute(1, 0, 2, 3)),
                                               "down": b64u8(pair["down"].permute(1, 0, 2, 3))},
            "check": json.loads(CHECK.read_text()) if CHECK.exists() else None}


@app.post("/api/futures/place")
async def futures_place(request: Request):
    """Every run's values for one square, for the chart. Uses the last many-futures result."""
    try:
        body = await request.json()
        i, j = int(body["i"]), int(body["j"])
        if not (0 <= i < NLAT and 0 <= j < NLON):
            raise ValueError("square is off the map")
    except Exception as e:  # noqa: BLE001
        return fail(f"bad request: {e}")
    with LOCK:
        if CACHE.get("id") != body.get("id"):
            return fail("that result is no longer held", 409)
        return {"id": CACHE["id"], "base": b64(CACHE["base"][:, :, :, i, j].numpy())}       # [runs, T, 7]


@app.post("/api/alerts")
async def alerts(request: Request):
    """Role alerts: every rule in alerts.py checked in each of the 50 futures."""
    global UNUSUAL
    if SRC is None:
        return fail("no starting weather on disk", 503)
    try:
        body = await request.json()
        date = str(body["date"])
        init, _, doy, _ = SRC.get(date, MAX_STEPS)
    except Exception:  # noqa: BLE001
        return fail("that day is not in the data", 404)
    key0 = ENGINE.key_from_sincos(sincos(doy))
    with LOCK:
        if UNUSUAL is None:
            years = SRC.time.astype("datetime64[Y]").astype(int) + 1970
            key = ((SRC.time - SRC.time.astype("datetime64[Y]")).astype(int) + 1) % 365
            UNUSUAL = alert_rules.unusualness(SRC.data, years, key, ENGINE.clim.numpy(), ENGINE.sigma.numpy().reshape(-1))
        m = lab.futures(FAST, init, key0, MAX_STEPS, N_RUNS)
        normal = torch_stack([ENGINE.clim[(key0 + k) % 365, T2M] - 273.15 for k in range(MAX_STEPS + 1)])
        st = ENGINE.static
        res = alert_rules.evaluate(m, normal, st[0], st[1] * 2000.0)
    rules = {}
    for k, r in alert_rules.RULES.items():
        rules[k] = {"field": r["field"], "level": r["level"], "where": r["where"], "says": r["says"], "source": r["source"], "good": bool(r.get("good")),
                    "count": b64u8(res[k]["count"]), "when": b64u8(res[k]["when"]), "applies": b64u8(res[k]["applies"])}
    rank = float(np.searchsorted(UNUSUAL[1], UNUSUAL[0][SRC.index(date)]) / len(UNUSUAL[1]))
    return {"runs": N_RUNS, "roles": alert_rules.ROLES, "rules": rules, "unusual": rank}


@app.post("/api/exam")
async def exam(request: Request):
    """Physics exam: live checks of how the model behaves on the chosen start day (see exam.py)."""
    if SRC is None:
        return fail("no starting weather on disk", 503)
    try:
        body = await request.json()
        init, _, doy, _ = SRC.get(str(body["date"]), MAX_STEPS)
    except Exception:  # noqa: BLE001
        return fail("that day is not in the data", 404)
    with LOCK:
        t0 = time.perf_counter()
        rows = physics_exam.run(ENGINE, init, sincos(doy))
        ms = (time.perf_counter() - t0) * 1000.0
    return {"rows": rows, "ms": ms}


@app.post("/api/finder")
async def finder(request: Request):
    """Butterfly Finder: which squares of the start day an outcome depends on, plus a rerun that tests the answer."""
    if SRC is None:
        return fail("no starting weather on disk", 503)
    try:
        body = await request.json()
        init, _, doy, _ = SRC.get(str(body["date"]), MAX_STEPS)
        day, field = int(body["day"]), str(body["field"])
        cells = [int(c) for c in body["cells"]]
        sign = -1.0 if float(body.get("sign", 1)) < 0 else 1.0
        if not (1 <= day <= MAX_STEPS) or field not in lab.FIELD_KEYS or not cells or min(cells) < 0 or max(cells) >= NLAT * NLON:
            raise ValueError("day, field or target is out of range")
    except KeyError:
        return fail("that day is not in the data", 404)
    except Exception as e:  # noqa: BLE001
        return fail(f"bad request: {e}")
    with LOCK:
        t0 = time.perf_counter()
        res = lab.finder(ENGINE, init, sincos(doy), day, field, cells, sign)
        res["ms"] = (time.perf_counter() - t0) * 1000.0
    for k in ("score", "patch", "diff"):
        if k in res:
            res[k] = b64(res[k])
    return res


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
    if SRC is not None:                          # warm up the many-futures path so the first click is fast
        threading.Thread(target=lambda: lab.futures(FAST, SRC.data[:3], 1, 1, 2), daemon=True).start()
    print(f"VARUNA dashboard: http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
