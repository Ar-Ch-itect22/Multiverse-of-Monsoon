# SPEC

This file is the contract for every teammate and every AI coding tool. Read it before writing code.

## Pitch sentence

For anyone in India who must act on the weather, VARUNA answers three questions on one map: what is coming for me,
how sure are we, and what does it depend on. Unlike a normal forecast, you can also change the starting weather and
watch what happens.

## Graded core (build first, never break)

- `serve.py` at the repo root serves only `POST /health` and `POST /simulate`. Do not add routes to it.
- `serve.py` may import from `model/` only. Never from `training/` or `dashboard/`.
- No date literals, no weather-service names, no chart or UI libraries in `serve.py` or `model/`.
- The data file stays in `data/`, which git ignores. Never commit a `.nc` file.
- Same input must give exactly the same output. A zero shock must change nothing.
- After any model change run `python training/scoreboard.py` and record the result.
- Scan trap: the organizers' repo scan treats any Python file that names the graded route (the word `simulate` in
  quotes or after a slash) as graded code. Keep that string out of every Python file except `serve.py`. This already
  cost us 8 points once in a local test. Demo dates live in `dashboard/app.js`, never in a Python file.
- Before every push, run the scan on a clean clone and check it says 15:
  `python -c "from pathlib import Path; from monsoon_r1 import repo_scan; print(repo_scan.scan(Path('PATH_TO_CLEAN_CLONE'))['points'])"`

## Model rules

- Local view only: 3x3 convolutions, at most 8 cells of reach per simulated day (currently 6).
- Banned inside the network: attention, pooling, BatchNorm, GroupNorm, InstanceNorm, Fourier layers.
- Predict the change from the seasonal normal. Rain is handled as log(1 + rain) and is never negative.
- Data split: train 2005 to 2018, tune 2019, final check 2020 and 2021. Never tune on the final-check years.

## Dashboard (separate folder, separate backend)

- `dashboard/index.html` plus plain JavaScript and a canvas. No build step, no internet, no map tiles.
- The coastline is drawn from the land-sea mask.
- `dashboard/backend.py` imports `model/` and serves everything that is not graded, on another port.
- Every number on screen comes from a live model run. Nothing is pre-rendered.

## Feature list, in build order

| Tier | Feature | Honest label shown on screen |
| --- | --- | --- |
| 1 | Forecast map with day slider, six fields | Model forecast, about 100 km squares, daily |
| 1 | Paint a shock, split view, difference map | Modelled what-if, not proof of real-world effect |
| 1 | Proof panel: live speed, zero-shock test, measured scores | Scores are from our own scoreboard |
| 2 | 50 universes with "N of 50" counts | Scenario frequency, not a calibrated probability |
| 2 | Noise mask and spread curve on the difference map | Grey means smaller than the model's own noise |
| 2 | Butterfly Finder with predict-then-prove | Model sensitivity, not proven cause |
| 2 | Role alerts: farmer, fisher, heat officer, hill routes, logistics | Regional indicator; follow official warnings |
| 2 | Plain-words "why" sentence and "what changed" list | Every sentence cites a computed number |
| 3 | Wind-aware sea route between two ports | Uses wind only; no waves, tides or currents |
| 3 | Limited crop advisor: harvest rain alert, sowing dry-spell check, spraying window | Thresholds cited from a published source |
| 3 | Physics exam panel | Failed rows are shown, not hidden |
| 3 | Strength sweep, monsoon phase tracker, moisture corridor view | Indicators only |
| 3 | Ventilation (air stagnation) score | Not air quality; no pollution data |
| 3 | 0 to 10 risk scores, ask-anything box with template answers | Weights and answers cite their source |

## Role alerts: one engine, four rules

| Hazard | Rule on the live forecast | Roles that see it |
| --- | --- | --- |
| Heavy rain | Daily rain above a cited threshold in the chosen region, days 1 to 3 | Farmer (harvest), logistics, hill routes |
| Dry spell | Run of days with rain below 2.5 mm in the next 10 days | Farmer (sowing) |
| Heat | Temperature well above the seasonal normal for 2 or more days | Heat officer, hill routes (cold uses the same rule reversed) |
| Strong wind at sea | Wind speed above a cited threshold over sea squares | Fisher, sea route |

Every alert shows the day, the "N of 50" count, and a low-confidence tag when the 50 runs disagree.
A "never seen before" flag appears when the starting weather is far outside the 17 training years.

## Claims we never make

- Cloudbursts, landslides, city-level floods: our squares are about 100 km and one step is a day.
- Tides, waves, crop yield, flood depth, pollution numbers: not in the data.
- A planting calendar from long-term trends: we offer a 10-day dry-spell check instead.
- Any partnership. We say the system is designed to plug into existing government advisory channels.

## Pitch only (not built)

- Data-centre siting from climate averages.
- Seasonal outlooks using ocean-state inputs.

## Working rules

- No new library without posting it in the team chat first.
- Small pull requests, merged every 1 to 2 hours. Never merge a change that breaks the format checker.
- Thresholds live in one config file, each with its source next to it.
