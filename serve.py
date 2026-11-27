"""VARUNA server. The judge runs this exact file offline:  python serve.py --port 8000

Only two routes live here, the two that are graded: POST /health and POST /simulate. Everything else (ensembles,
cause tracing, alerts) lives in the separate dashboard backend so this path stays small, fast and deterministic.
Weights and the table of normal weather are loaded once from the model/ folder next to this file.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model.net import NC, NLAT, NLON, Engine

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL_NAME = "varuna-local-v2"
ENGINE = Engine(Path(__file__).resolve().parent / "model")
MAX_STEPS = 366                         # the judge asks for 10; longer runs are allowed up to a year


def _arr(x):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        import base64
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Returns [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    return ENGINE.simulate(init_states, doy_sincos, steps, perturbation)


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0 = time.perf_counter()
    try:
        body = await request.json()
        init = _arr(body["init_states"])
        pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
        sc = [float(v) for v in body["doy_sincos"]]
        steps = int(body["steps"])
        ok = init.shape == (3, NC, NLAT, NLON) and len(sc) == 2 and 1 <= steps <= MAX_STEPS
        ok = ok and (pert is None or pert.shape == (NC, NLAT, NLON))
        ok = ok and bool(np.isfinite(init).all()) and bool(np.isfinite(sc).all())
        ok = ok and (pert is None or bool(np.isfinite(pert).all()))
        if not ok:
            raise ValueError("bad request")
    except Exception:
        return JSONResponse(status_code=422, content={"error": "invalid request: check shapes, steps and finite values"})
    out = simulate(init, sc, steps, pert)
    return {"rollout": np.asarray(out, dtype="f4").tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
