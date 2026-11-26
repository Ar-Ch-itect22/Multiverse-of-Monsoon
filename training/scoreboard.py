"""Local scoreboard: runs the organizers' own scoring functions (monsoon_r1.scoring) on held-out years.

Usage (from the repo root):
    python training/scoreboard.py                 # scores serve.simulate on the final-check years
    python training/scoreboard.py --baselines     # only prints the two simple-guess baselines
    python training/scoreboard.py --years 2019    # score on the tuning year instead

What it reports, mirroring the 70 model points of the automatic grade:
    forecast error (30)   skill vs the better of "repeat yesterday" and "normal for the season", days 1, 4, 8
    heavy rain (15)       CSI at 10 mm/day, same days
    sharpness (10)        day-8 spatial variance ratio per channel
    shocks (15)           warm ocean + monsoon stall + zero shock

Assumption: the hidden grader's exact window list and per-window averaging are not in the kit. Here each window's
skill is the mean over the three lead days and six channels, and the climatology is a smoothed day-of-year mean
built from the training years only. Treat the totals as a close estimate, not the official score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from monsoon_r1 import scoring
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, LEAD_DAYS, SCORE_STEPS, doy_sincos

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "india_climate_train.nc"
STATS = ROOT / "data" / "stats.json"
TRAIN_YEARS = (2005, 2018)
TUNE_YEARS = (2019, 2019)
FINAL_YEARS = (2020, 2021)


def load():
    ds = xr.open_dataset(DATA)
    data = ds["data"].values.astype("f4")                      # [T, 6, 32, 36]
    time = ds["time"].values.astype("datetime64[D]")
    years = time.astype("datetime64[Y]").astype(int) + 1970
    doy = (time - time.astype("datetime64[Y]")).astype(int) + 1  # 1..366
    std = np.array(json.loads(STATS.read_text())["std"], "f4")
    return data, years, doy, std


def climatology(data, years, doy, half_window: int = 15) -> np.ndarray:
    """[367, 6, 32, 36]: mean per day of year over the training years, smoothed over +/- half_window days (index = doy)."""
    m = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])
    raw = np.zeros((367,) + data.shape[1:], "f8")
    for d in range(1, 367):
        sel = m & (doy == d)
        raw[d] = data[sel].mean(axis=0) if sel.any() else np.nan
    raw[366] = np.where(np.isnan(raw[366]), raw[365], raw[366])
    days = raw[1:367]
    out = np.zeros_like(days)
    for k in range(-half_window, half_window + 1):
        out += np.roll(days, k, axis=0)
    clim = np.zeros_like(raw)
    clim[1:367] = out / (2 * half_window + 1)
    return clim.astype("f4")


def windows(years, lo_hi, stride: int):
    """Indices i of the LAST input day such that i-2..i are inputs and i+1..i+SCORE_STEPS are targets, all in range."""
    lo, hi = lo_hi
    idx = np.where((years >= lo) & (years <= hi))[0]
    first, last = idx[0] + 2, idx[-1] - SCORE_STEPS
    return list(range(first, last + 1, stride))


def score(simulate, data, years, doy, std, clim, year_range, stride=7, shock_every=8, verbose=True) -> dict:
    ss_w, csi_w, ratios, gate, lead_ss = [], [], [], [], {d: [] for d in LEAD_DAYS}
    shock_res = {"warm_ocean": [], "monsoon_stall": []}
    null_ok, bad = [], 0
    wins = windows(years, year_range, stride)
    for n, i in enumerate(wins):
        init = data[i - 2:i + 1]
        truth = data[i + 1:i + 1 + SCORE_STEPS]
        sc = doy_sincos(int(doy[i]))
        pred = np.asarray(simulate(init.copy(), sc, SCORE_STEPS, None), "f4")
        if scoring.validity(pred, SCORE_STEPS):
            bad += 1
            ss_w.append(0.0); csi_w.append(0.0)
            continue
        persist = np.repeat(init[-1][None], SCORE_STEPS, axis=0)
        cl = np.stack([clim[int(doy[i + k])] for k in range(1, SCORE_STEPS + 1)])
        ss_leads, csi_leads = [], []
        for d in LEAD_DAYS:
            k = d - 1
            r = scoring.skill_vs_best_ref(pred[k], truth[k], persist[k], cl[k])
            ss_leads.append(np.nanmean(r["ss"])); lead_ss[d].append(np.nanmean(r["ss"]))
            if d == 4:
                gate.append(np.nanmean(r["ss_persist"]))
            cs = scoring.csi_skill(scoring.csi(pred[k, 1], truth[k, 1]), scoring.csi(persist[k, 1], truth[k, 1]),
                                   scoring.csi(cl[k, 1], truth[k, 1]))
            if np.isfinite(cs):
                csi_leads.append(cs)
        ss_w.append(float(np.mean(ss_leads)))
        if csi_leads:
            csi_w.append(float(np.mean(csi_leads)))
        ratios.append(scoring.variance_ratio(pred[7], truth[7]))
        if n % shock_every == 0:
            for name, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
                delta = scoring.make_shock(name, init[-1], std)
                shocked = np.asarray(simulate(init.copy(), sc, SCORE_STEPS, delta), "f4")
                shock_res[name].append(scoring.causality_checks(pred, shocked, box, std))
            zero = np.asarray(simulate(init.copy(), sc, SCORE_STEPS, np.zeros_like(init[-1])), "f4")
            null_ok.append(scoring.null_consistent(pred, zero, std))

    fp = scoring.forecast_points(ss_w, csi_w)
    blur = scoring.anti_blur_points(np.array(ratios)) if ratios else 0.0
    # shock points: average the per-window points, each window scored exactly as the grader scores its one window
    shock_pts = [scoring.causality_points([a, b], ok) for a, b, ok in
                 zip(shock_res["warm_ocean"], shock_res["monsoon_stall"], null_ok)]
    shock = float(np.mean(shock_pts)) if shock_pts else 0.0
    out = {
        "windows": len(wins), "invalid_windows": bad,
        "forecast_error_pts_of_30": round(fp["rmse_pts"], 2), "mean_skill": round(fp["mean_ss"], 4),
        "skill_by_lead": {d: round(float(np.mean(v)), 4) for d, v in lead_ss.items() if v},
        "heavy_rain_pts_of_15": round(fp["csi_pts"], 2), "mean_csi_skill": round(fp["mean_csi_skill"], 4),
        "sharpness_pts_of_10": round(blur, 2),
        "day8_variance_ratio": [round(float(x), 2) for x in np.nanmean(np.array(ratios), axis=0)] if ratios else [],
        "shock_pts_of_15": round(shock, 2),
        "shock_pass_rates": {name: {k: round(float(np.mean([bool(r[k]) for r in res])), 2)
                                    for k in ("nonzero", "localized", "propagates", "stable")}
                             for name, res in shock_res.items() if res},
        "zero_shock_identical": bool(all(null_ok)) if null_ok else None,
        "day4_beats_repeat_yesterday": bool(np.mean(gate) > 0) if gate else None,
        "day4_skill_vs_repeat_yesterday": round(float(np.mean(gate)), 4) if gate else None,
    }
    out["model_pts_of_70"] = round(out["forecast_error_pts_of_30"] + out["heavy_rain_pts_of_15"]
                                   + out["sharpness_pts_of_10"] + out["shock_pts_of_15"], 2)
    if verbose:
        print(json.dumps(out, indent=1))
    return out


def baselines(data, years, doy, clim, year_range, stride=7):
    """Lat-weighted RMSE of the two simple guesses at days 1, 4, 8, per channel, and which one is the bar to beat."""
    wins = windows(years, year_range, stride)
    names = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
    print(f"baselines on {year_range[0]}-{year_range[1]}, {len(wins)} windows")
    for d in LEAD_DAYS:
        p, c, t = [], [], []
        for i in wins:
            t.append(data[i + d]); p.append(data[i]); c.append(clim[int(doy[i + d])])
        rp, rc = scoring.rmse_lat(np.array(p), np.array(t)), scoring.rmse_lat(np.array(c), np.array(t))
        row = "  ".join(f"{n}: {a:.2f}/{b:.2f} {'R' if a < b else 'N'}" for n, a, b in zip(names, rp, rc))
        print(f" day {d}  repeat-yesterday / normal-for-season (letter = the better one)\n   {row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--years", type=int, nargs="+", default=list(FINAL_YEARS))
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--save", default="", help="also write the result to this JSON file (the dashboard reads dashboard/scores.json)")
    a = ap.parse_args()
    yr = (min(a.years), max(a.years))
    data, years, doy, std = load()
    clim = climatology(data, years, doy)
    baselines(data, years, doy, clim, yr, a.stride)
    if not a.baselines:
        sys.path.insert(0, str(ROOT))
        import serve
        print(f"scoring serve.simulate ({serve.MODEL_NAME})")
        res = score(serve.simulate, data, years, doy, std, clim, yr, a.stride)
        if a.save:
            res.update(model_name=serve.MODEL_NAME, years=list(yr), stride=a.stride)
            Path(a.save).write_text(json.dumps(res, indent=1))
