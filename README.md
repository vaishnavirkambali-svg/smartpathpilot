# SmartPathPilot: AI/ML route optimization for Bengaluru

FastAPI backend + Leaflet web app. It ranks routes by predicted travel time, distance and road quality.

## What is real and what is simulated
| Part | Status |
|---|---|
| Road network | **OpenStreetMap** after you run `python osm_ingest.py`. Until then a **demo network** of 21 real Bengaluru junctions with straight-line roads is used. The page badge says which. |
| Traffic: "Simulated profile" | Hand-written hourly pattern with **assumed values**. Not measured. |
| Traffic: "ML prediction" | Gradient-boosting model. On the demo network it is trained on the **simulated** dataset. On OpenStreetMap it needs readings you collect (see step 7). |
| Traffic: "Live (TomTom)" | **Real readings, only with a TomTom key.** Without a key the app falls back to simulated traffic and labels it as a fallback. It never shows simulated numbers as live. |
| Road quality | Demo: simulated pothole scores. OpenStreetMap: a proxy from the road's `surface` tag (OSM has no pothole data). |

## Install and run on Windows (VS Code, PowerShell)
Needs Python 3.10 to 3.13 from python.org (tick "Add python.exe to PATH") and VS Code with the Python extension.

1. Unzip the project, then in VS Code use File > Open Folder and choose it. Open Terminal > New Terminal (PowerShell).
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry. In VS Code, press Ctrl+Shift+P, choose "Python: Select Interpreter" and pick the `.venv` one.
3. Install:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```
4. Test (expect `10 passed`):
   ```
   python -m pytest -q
   ```
5. Run:
   ```
   python -m uvicorn app:app --reload
   ```
   Open http://127.0.0.1:8000 for the app, http://127.0.0.1:8000/docs for the API, and http://127.0.0.1:8000/classic for the original schematic view. The first start trains the model (a few seconds).
6. Real OpenStreetMap roads (needs internet, takes several minutes, run once):
   ```
   python osm_ingest.py
   ```
   Stop and restart the server. The badge changes to "Roads: OpenStreetMap". Delete `data\bengaluru_osm.pkl` to return to the demo network. If the download times out, run it again.
7. Optional live traffic. Get a TomTom key at developer.tomtom.com, then in the same terminal:
   ```
   $env:TOMTOM_KEY="your_key_here"
   python -m uvicorn app:app --reload
   ```
   To collect real traffic history and train on it (check your TomTom daily quota; `collect.py` prints its estimated calls per day):
   ```
   python collect.py --every 30 --n 50
   python osm_ml.py
   ```
   `collect.py` keeps running, so use a second terminal for the server. Run `osm_ml.py` after a few days of data.

## Troubleshooting
- `ModuleNotFoundError`: the virtual environment is not active. Rerun the activate command in step 2.
- Map background is blank but routes appear: map tiles need internet. The app itself runs offline.
- `pip install osmnx` fails: remove the `osmnx` line from `requirements.txt`. The app and tests still run on the demo network, but step 6 will not work.
- Port busy: add `--port 8001` and open that port instead.

## Layout
`app.py` (server) | `core.py` (21-junction features, model, routing) | `osm_service.py` (routing on the map graph) | `osm_api.py` (map endpoints) | `osm_ingest.py` (OpenStreetMap download) | `traffic_live.py` (profile, TomTom) | `osm_ml.py` (ML on collected readings) | `collect.py` | `static/app.html` (main UI) | `static/index.html` (classic UI) | `tests/` | `data/`

## Main endpoints
`GET /api/osm/route` (ranked routes with traffic level, road condition, directions and a `traffic.kind` field saying where the traffic numbers came from) | `GET /api/osm/best-time` | `GET /api/osm/places` | `GET /api/osm/status` | classic: `/api/route`, `/api/best-time`, `/api/hotspots`, `/api/predictions`, `/api/metrics`

---
# Advanced features (multi-stop, road quality, weather, traffic profiles)
Additive: every earlier endpoint and the original page behave as before. Real request/response examples: `API_EXAMPLES.md`.

## New environment variables (all optional)
| Variable | Purpose |
|---|---|
| `TOMTOM_KEY` | Enables the `live` traffic profile (real TomTom readings). Without it `live` is not offered and requesting it returns HTTP 409. |
| `OPENWEATHER_API_KEY` | Enables official weather alerts (OpenWeather One Call 3.0, needs a subscription). Without it alerts are reported as "not configured". |
| `OPEN_METEO_API_KEY` | Only for Open-Meteo commercial plans. The free tier needs no key. |
| `SPP_MAX_STOPS` | Maximum mandatory stops per request (default 20; the request schema caps at 50). |

## Multi-stop routing: `POST /api/osm/route/multi`
Body: `source`, `destination`, `stops[]` (each `{lat, lon, name?}`), `mode` (`optimized` | `strict` | `compare`), `objective` (`fastest` | `shortest`), `road_preference` (`none` | `balanced` | `better_roads`), `traffic_profile` (`free_flow` | `typical` | `peak` | `live`), `hour`, `day`, optional `depart_at` (ISO time, IST if no zone; overrides hour/day and sets the weather forecast time), `weather_assessment` (bool).
- Costs come from Dijkstra on the loaded road graph (never straight-line distance). Source and destination stay fixed.
- Stop order: exact Held-Karp dynamic programming for up to 12 stops; above that, nearest-neighbour + 2-opt + or-opt (deterministic, not guaranteed optimal). `optimization.method` says which ran.
- Duplicate stops (same junction as the source, destination or an earlier stop) are dropped with a warning. Unreachable stops, stops too far from the road network, too many stops, bad coordinates and invalid edge weights return clear 404/422 errors.
- `multistop.py` works on `Graph`, `DiGraph`, `MultiGraph` and `MultiDiGraph` and uses the cheapest valid parallel edge. The loaded OSM graph is a `DiGraph`: `osm_ingest.py` collapses parallel edges at load time, keeping the shortest one.

## Road quality
Source: OpenStreetMap `surface` and `smoothness` tags (kept by `osm_ingest.py`; re-run it to pick up `smoothness`). Score 0 (best) to 1 (worst) from an assumption table in `road_quality.py`, e.g. asphalt 0, paving_stones 0.5, unpaved 0.8, dirt 0.95; smoothness good 0.1, bad 0.75. If both tags exist the worse score is used.
- **Missing tags are neutral**: unknown roads get no penalty and are never assumed poor. Responses report `known_share` and `unknown_share`.
- Edge cost = base cost x (1 + alpha x score). Base cost is travel time (`fastest`) or length (`shortest`). Alpha: `none` 0, `balanced` 0.5, `better_roads` 2.0. Override scores and alphas with `data/road_quality_config.json` (see the docstring in `road_quality.py`). `GET /api/roadquality/config` shows the active values.
- Distance, time and the quality penalty are reported separately; the preference only changes route choice.
- The data is mapper-contributed OSM tags. It is **not** verified pothole or live condition data, and the response says so. On the demo network, quality is simulated and labelled "simulated".
- Existing `/api/osm/route` still uses its older `road_condition` field (where unknown surface defaulted to roughness 2); multi-stop uses the new treatment.

## Traffic profiles (`GET /api/traffic/profiles`)
| Profile | Meaning | Live? |
|---|---|---|
| `free_flow` | Every road at free-flow speed (OSM `maxspeed`, else assumed class defaults; simulated speeds on the demo network). | No |
| `typical` | Simulated/assumed congestion for the chosen hour and day. | No |
| `peak` | The same model at the weekday 9 AM peak. | No |
| `live` | Real TomTom readings on ~50-120 sampled main roads, spread to nearby roads. Only offered when `TOMTOM_KEY` is set and data has arrived; otherwise HTTP 409. | Yes |
Every response carries `traffic.source`, `assumptions`, `limitations` and `verified_live`. Fixed multipliers and OSM default speeds are never called live.

## Weather (`weather_assessment: true`)
Provider: Open-Meteo (no key), sampled at up to 5 points along each route, cached for 15 minutes (failures for 60 s), 6 s timeout. Each sample is labelled `forecast` or `current_model_estimate` (Open-Meteo's "current" is model-based, not a station observation). Reports rain, temperature, wind, gusts, visibility and warnings (heavy rain >= 7.6 mm/h, strong wind, low visibility, thunderstorm, heat). Official alerts only with `OPENWEATHER_API_KEY`.
- Weather does **not** reroute. The planner cannot predict flooding or which road is safest. It reports exposure (estimated rain depth during the trip) and an ETA adjusted by an **assumed** rain slowdown table (1.00 to 1.30, not calibrated for Bengaluru). In `compare` mode it reports which order has lower estimated rain exposure, with a caveat.
- If the provider fails, the route is still returned and the weather block says `available: false` with the reason.

## Run the tests
```
python -m pytest -q
```
Optional browser-logic checks (need Node.js; the server must be running):
```
cd tests
npm install jsdom
$env:BASE="http://127.0.0.1:8000"; node ui_check.js
```

## Manual checklist (website at http://127.0.0.1:8000)
1. Set From = Majestic, To = Whitefield. Add stops Silk Board, Hebbal, Jayanagar in the Multi-stop box.
2. Use the arrows to reorder and the cross to remove a stop.
3. Mode "Compare both", then "Plan multi-stop trip": two cards (optimized, strict), numbered markers, a green/amber/red route, and a line such as "Optimized order saves ... min".
4. Click the Strict card: the map redraws that route; the other stays dashed.
5. Switch Objective to "Shortest distance": distance should not increase.
6. Switch Road quality to "Better roads": the card shows the quality penalty and "Quality known on N% of route"; open "Legs and road-quality source" to read the source note.
7. Open the Traffic profile list: Live is greyed out ("not configured") unless `TOMTOM_KEY` is set. The badge always says NOT LIVE for the other profiles.
8. Tick "Check weather along the route": with internet you see forecast values and warnings; offline you see "Weather unavailable: ..." and the route still appears.
9. Set From and To to the same place with no stops: a clear error message appears.
