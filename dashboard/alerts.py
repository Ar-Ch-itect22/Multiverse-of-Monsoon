"""Role alerts: plain rules applied to each of the 50 futures. Not graded.

This is the one place where alert thresholds live. Each has its source next to it. A rule is checked in every
future separately, and the page shows in how many of the 50 it fires. That count is a scenario frequency, not a
calibrated probability, and the page says so.

All values are for a square of about 100 km and a whole day, so they are lower than the station thresholds used in
official warnings. These alerts are a regional indicator. They do not replace official warnings.
"""
from __future__ import annotations

import numpy as np
import torch

FIELD = {"tp": 0, "t2m": 1, "wind": 2}          # index into the display fields made by lab.display

RULES = {
    "rain_3d": {"kind": "any_above", "field": "tp", "level": 10.0, "days": (1, 3), "where": "land", "unit": "mm/day",
                "says": "rain of 10 mm or more in a day, within days 1 to 3",
                "source": "10 mm/day on a 100 km square is the heavy-rain line in the organizers' scoring"},
    "rain_road": {"kind": "any_above", "field": "tp", "level": 25.0, "days": (1, 5), "where": "land", "unit": "mm/day",
                  "says": "rain of 25 mm or more in a day, within days 1 to 5",
                  "source": "team choice: 2.5 times the organizers' heavy-rain line"},
    "rain_hills": {"kind": "any_above", "field": "tp", "level": 10.0, "days": (1, 3), "where": "hills", "unit": "mm/day",
                   "says": "rain of 10 mm or more in a day over ground above 1000 m, within days 1 to 3",
                   "source": "organizers' heavy-rain line; 1000 m is a team choice for hill squares"},
    "dry_spell": {"kind": "run_below", "field": "tp", "level": 2.5, "run": 5, "days": (1, 10), "where": "land", "unit": "mm/day",
                  "says": "5 or more days in a row with under 2.5 mm of rain, within the next 10 days",
                  "source": "the India Meteorological Department counts a day as rainy at 2.5 mm or more; 5 days is a team choice"},
    "spray_window": {"kind": "calm_dry", "field": "tp", "level": 2.5, "wind": 5.0, "days": (1, 3), "where": "land", "unit": "mm/day", "good": True,
                     "says": "at least one dry and calm day (under 2.5 mm of rain, wind under 5 m/s) within days 1 to 3",
                     "source": "2.5 mm is the India Meteorological Department's rainy-day line; the 5 m/s wind limit is a team choice"},
    "hot_spell": {"kind": "run_above_normal", "field": "t2m", "level": 3.0, "run": 2, "days": (1, 7), "where": "land", "unit": "deg C above normal",
                  "says": "2 or more days in a row at least 3 deg C above normal for the season, within days 1 to 7",
                  "source": "team choice on the daily mean; official heat-wave rules use the daily maximum at stations"},
    "cold_spell": {"kind": "run_below_normal", "field": "t2m", "level": 3.0, "run": 2, "days": (1, 7), "where": "hills", "unit": "deg C below normal",
                   "says": "2 or more days in a row at least 3 deg C below normal over ground above 1000 m, within days 1 to 7",
                   "source": "team choice on the daily mean"},
    "sea_wind": {"kind": "any_above", "field": "wind", "level": 10.8, "days": (1, 3), "where": "sea", "unit": "m/s",
                 "says": "daily mean wind of 10.8 m/s or more over the sea, within days 1 to 3",
                 "source": "Beaufort force 6 (strong breeze) starts at 10.8 m/s"},
}

ROLES = [
    {"key": "farmer", "name": "Farmer", "rules": ["rain_3d", "dry_spell", "spray_window"],
     "lines": {"rain_3d": "Rain on a cut crop", "dry_spell": "Dry spell after sowing", "spray_window": "A dry, calm day for spraying"},
     "acts": {"rain_3d": "Move the harvested crop under cover before the rain day.", "dry_spell": "Hold sowing, or plan to water the seedbed.",
              "spray_window": "Spray on the dry, calm day."}},
    {"key": "fisher", "name": "Fisher", "rules": ["sea_wind"],
     "lines": {"sea_wind": "Strong wind at sea"}, "acts": {"sea_wind": "Small boats should stay in port on those days."}},
    {"key": "heat", "name": "Heat officer", "rules": ["hot_spell"],
     "lines": {"hot_spell": "Hot spell"}, "acts": {"hot_spell": "Open cooling centres and shift outdoor work hours."}},
    {"key": "hills", "name": "Hill routes", "rules": ["rain_hills", "cold_spell"],
     "lines": {"rain_hills": "Heavy rain in the hills", "cold_spell": "Cold spell in the hills"},
     "acts": {"rain_hills": "Delay travel on hill roads. This model cannot see landslides or cloudbursts.", "cold_spell": "Move herds and travellers lower before the cold days."}},
    {"key": "logistics", "name": "Logistics", "rules": ["rain_road"],
     "lines": {"rain_road": "Very heavy rain on the route"}, "acts": {"rain_road": "Send perishable loads early or by another route."}},
]


def evaluate(m: torch.Tensor, normal_t: torch.Tensor, land: torch.Tensor, height: torch.Tensor) -> dict:
    """m [runs, T, 7, H, W] display units, normal_t [T, H, W] deg C, land [H, W] share of the square that is land,
    height [H, W] metres. For every rule: in how many runs it fires per square, the usual first day, and where it applies.
    A coastal square counts as both land and sea, so a port city gets the land alerts and the sea alerts."""
    on_land, at_sea = land >= 0.2, land <= 0.8
    where = {"land": on_land, "sea": at_sea, "hills": on_land & (height >= 1000.0)}
    out = {}
    for key, r in RULES.items():
        d0, d1 = r["days"]
        x = m[:, d0:d1 + 1, FIELD[r["field"]]]                                   # [runs, days, H, W]
        if r["kind"] == "any_above":
            cond, need = x >= r["level"], 1
        elif r["kind"] == "calm_dry":
            cond, need = (x < r["level"]) & (m[:, d0:d1 + 1, FIELD["wind"]] < r["wind"]), 1
        elif r["kind"] == "run_below":
            cond, need = x < r["level"], r["run"]
        elif r["kind"] == "run_above_normal":
            cond, need = (x - normal_t[d0:d1 + 1]) >= r["level"], r["run"]
        else:
            cond, need = (x - normal_t[d0:d1 + 1]) <= -r["level"], r["run"]
        run = torch.zeros_like(cond[:, 0], dtype=torch.int16)
        first = torch.zeros_like(run)                                             # day the spell starts, 0 = never
        for k in range(cond.shape[1]):
            run = (run + 1) * cond[:, k]
            hit = (run >= need) & (first == 0)
            first = torch.where(hit, torch.full_like(first, d0 + k - need + 1), first)
        fires = first > 0
        count = fires.sum(0)
        when = torch.where(count > 0, (first.sum(0).float() / count.clamp(min=1)).round(), torch.zeros(()))
        ok = where[r["where"]]
        out[key] = {"count": (count * ok).to(torch.uint8).numpy(), "when": (when * ok).to(torch.uint8).numpy(),
                    "applies": ok.to(torch.uint8).numpy()}
    return out


def unusualness(data: np.ndarray, years: np.ndarray, key: np.ndarray, clim: np.ndarray, sigma: np.ndarray, last_train: int = 2018):
    """How far each day's weather sits from normal, as one number per day (root mean square, in typical swings).
    Returns the sorted values for the training years, to rank any start day against."""
    out = np.empty(len(data), "f4")
    for a in range(0, len(data), 512):                                           # in chunks to keep memory low
        z = data[a:a + 512].copy()
        z[:, 1] = np.log1p(np.clip(z[:, 1], 0, None))
        an = (z - clim[key[a:a + 512]]) / sigma.reshape(1, -1, 1, 1)
        out[a:a + 512] = np.sqrt((an ** 2).mean(axis=(1, 2, 3)))
    return out, np.sort(out[years <= last_train])
