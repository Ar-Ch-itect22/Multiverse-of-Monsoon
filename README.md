# VARUNA: Multiverse of Monsoon (Track 2)

Team Ship Happens. A small weather model for India that you can question: change the starting weather on a map and
watch what the model says happens next.

Work in progress. This file will grow before submission (libraries and licences, AI tools used, final scores).

## Setup

```
python -m venv .venv
.venv\Scripts\activate          (Linux or Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Put the organizers' file `india_climate_train.nc` in a folder named `data/`. It is never committed.

## Run

| What | Command | Then |
| --- | --- | --- |
| Graded server | `python serve.py --port 8000` | `POST /health` and `POST /simulate`, offline, CPU only |
| Dashboard | `python dashboard/backend.py` | open http://127.0.0.1:8050 |
| Train the model | `python training/train.py` | writes `model/varuna.pt` and `model/norm.npz` |
| Score the model | `python training/scoreboard.py --save dashboard/scores.json` | prints our estimate of the 70 model points |
| Check the 50 futures | `python training/calibrate_futures.py --years 2020 2021 --scales 1.0 --save dashboard/futures_check.json` | prints spread against real miss |

The dashboard needs the data file for its starting weather. The graded server does not.

## Layout

| Path | What it is |
| --- | --- |
| `serve.py` | The graded server. Two routes only. Imports only `model/`. |
| `model/` | The network, its weights, and a 365-day table of normal weather. |
| `training/` | Training script and a local scoreboard that uses the organizers' scoring code. |
| `dashboard/` | The map: `index.html`, `app.js`, `style.css`, and `backend.py` on its own port. |
| `SPEC.md` | The rules every teammate and AI tool follows. Read it before writing code. |

## Honest limits

- One square is about 100 km and one step is one day. No cloudbursts, landslides or city floods.
- A what-if shows what the model does, not proof of what the real atmosphere would do.
- Scores shown in the dashboard are from our own scoreboard on 2020 and 2021. They are not the official grade.
