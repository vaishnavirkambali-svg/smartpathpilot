# CityPulse and the Gradient Descent Lab

SmartPathPilot now has three pages in one website: **Route Planner** (`/`, unchanged), **CityPulse** (`/citypulse`) and **Gradient Descent Lab** (`/gradient-descent-lab`). The original multi-stop planner and all existing endpoints are untouched.

## 1. Architecture
```
static/app.html (Route Planner)   static/citypulse.html        static/gdlab.html
        |  /api/osm/route                |  /api/citypulse/*          |  /api/gdlab/*
        |  ?use_road_conditions=true     |                            |
   osm_api.py ----------------------> citypulse/api.py            gdlab_api.py -> gdlab.py (NumPy only)
        |                                  |
   osm_service.Router  <-- extra cost --  citypulse/routing.py   citypulse/{store, geo, analysis, risk, ml, demo}.py
   (NetworkX graph, existing)                                     SQLite: data/citypulse.db
```
- **Reused:** the active road graph and `Router` (real OpenStreetMap if downloaded, otherwise the demo network), the Leaflet map, the FastAPI app, scikit-learn (for the optional CityPulse risk model only).
- **Separate by design:** `gdlab.py` imports only NumPy and knows nothing about CityPulse or the route planner (a test enforces this). The traffic `GradientBoostingRegressor` (`core.py`) is untouched. CityPulse's optional model is a logistic regression and is unrelated to both.
- **Persistence:** SQLite via Python's standard library (no new dependency) because the project had no database, only CSV/pickle files. File: `data/citypulse.db` (override with `CITYPULSE_DB`).
- **Segments:** a road segment is an undirected pair of graph nodes, key `min-max`. Reports store only latitude/longitude and are snapped to the nearest segment of the active graph at analysis time (max 0.15 km on OpenStreetMap, 0.5 km on the demo network), so switching graphs never leaves stale ids. Reports not near a mapped road are kept, shown on the map, flagged "not matched", and excluded from segment analysis and routing.

## 2. CityPulse workflow
1. Someone submits a report (type, map location, description, severity, observation time). It is stored as **unverified community information**.
2. A moderator (anyone holding `CITYPULSE_ADMIN_TOKEN`) can mark it verified or rejected. Without the variable, moderation is disabled.
3. The dashboard groups reports by segment, detects **recurring problems**, ranks segments for **maintenance priority**, and shows **risk estimates**.
4. Route Planner can optionally add a cost penalty for **verified** issues (section 7).

## 3. Report schema (`reports` table)
| Column | Meaning |
|---|---|
| `id` | auto-increment |
| `issue_type` | `pothole`, `waterlogging`, `damaged_surface`, `obstruction`, `other` |
| `lat`, `lon` | location; must be finite and inside the Bengaluru bounding box |
| `description` | max 500 chars, control characters stripped |
| `severity` | integer 1-5 or empty (empty is never replaced by a guess in the analysis) |
| `observed_at` | when seen (UTC; naive input is read as India time); not in the future, not older than 5 years |
| `created_at` | when submitted |
| `status` | `unverified` (default), `verified`, `rejected` |
| `reviewed_at` | when the status changed (used to avoid leakage in ML features) |
| `source` | `community_report` or `demo_seed` (**simulated**) |
| `reporter_hash` | salted HMAC of IP + user agent, 16 hex chars. Never returned by the API. No names, emails or raw IPs are stored. |

Submission guards: duplicate = same reporter, same type, within 50 m and 24 h (HTTP 409); at most 20 reports per reporter per hour (HTTP 429). Both are configurable.

## 4. Recurring problems and priority (all thresholds in `citypulse/config.py`, overridable in `data/citypulse_config.json`)
- **Episode:** reports of the same type on one segment, each within 72 h of the previous one. Many reports in one episode are one problem.
- **Independent observation:** one distinct reporter within one episode. The same person repeating adds nothing.
- **Recurring:** at least 2 separate episodes within 180 days, seen by at least 2 different reporters. Rejected reports are ignored.
- **Priority score (0-100)** = 100 x sum of weight x factor, over the factors that are available (weights renormalised):

| Factor | Weight | Value (0-1) |
|---|---|---|
| severity | 0.30 | highest supplied severity / 5 (omitted if nobody supplied one) |
| recency | 0.20 | 0.5 ^ (days since last report / 30) |
| recurrence | 0.20 | (episodes of the most repeated issue - 1) / 3, capped at 1 |
| confidence | 0.15 | 0.5 x share verified + 0.5 x min(1, distinct reporters / 3) |
| impact | 0.15 | road-class proxy (primary 0.85, secondary 0.65, ...). **An assumption, not traffic counts.** Omitted if the class is unknown. |

Every factor, its value, weight, points and explanation is shown in the UI. The score is a **decision-support recommendation for inspection**, not an official or engineering assessment.

## 5. Risk estimates and ML
**Target:** the chance of at least one new report on a segment within the next 14 days.
- **Baseline (always available):** `rate = (episodes in 180 days + 0.1) / (180 + 180)`, `index = 1 - exp(-rate x 14)`. An **uncalibrated heuristic**, shown as such.
- **ML model (optional, `python -m citypulse.train`):** logistic regression on weekly (segment, date) rows. Features: report counts/episodes over 30/90/180 days, days since last report, severity, verified share, road quality score, waterlogging flag, season (sin/cos), road-class proxy. Each prediction is an exact sum of feature contributions in log-odds, shown in the UI.
- **Leakage control:** features at time t use only reports observed and submitted before t, and count a status change only if `reviewed_at <= t`. Splits are chronological (60/20/20) with a 14-day embargo so no training label window overlaps validation or test. Hyperparameter C is chosen on validation; metrics (ROC AUC, average precision, Brier, log loss, base rate) are reported on the untouched test set next to the baseline heuristic.
- **Honesty gates:** with demonstration data the model is labelled **`demo_simulated`** and the metrics describe the simulator, not the real world. A model counts as **validated** only if trained on real reports, with at least 30 training and 15 test positives, test AUC >= 0.65, and better test log loss than the baseline. Demonstration data is excluded from training unless explicitly requested. With too little data, training refuses and says why.
- **Limits:** reports show where people report, not only where problems occur; rainfall is not a feature (no historical rainfall is stored); the model estimates new reports, not specific incidents. On the bundled simulated data the model beats the heuristic on log loss but not clearly on AUC; do not read that as evidence about real roads.

## 6. Data sources and labelling
| Kind | Where it appears | Label |
|---|---|---|
| Community reports | CityPulse DB | "Community report", unverified until moderated |
| Simulated reports | `demo_seed` rows (button "Load simulated demo data") | "SIMULATED demonstration data" everywhere, plus a page banner |
| OpenStreetMap attributes | road names, class, surface | "OpenStreetMap" |
| Model output | risk estimates | "estimate", "heuristic" or "trained on SIMULATED data" |

Nothing is fabricated at runtime: an empty database shows zeros and "no reports yet".

## 7. How CityPulse affects route planning (`/api/osm/route?use_road_conditions=true`, default off)
- Only **verified** reports that are still **active** change a route: potholes/damaged surface 120 days, obstruction 7, other 14, **waterlogging 3**.
- Each adds `penalty_per_severity x severity` (pothole/damage 0.05, waterlogging 0.15, obstruction 0.25, other 0.03; default severity 2 if missing) to a cost factor on that segment, capped at 2.5. **Nothing is ever blocked.**
- The factor changes the planner's internal **cost** only. Displayed distance and time are unchanged.
- Unverified reports are listed as "informational"; rejected, expired and future-dated reports are ignored.
- Predicted risk is **not** used. The configuration flag `use_predicted_risk_in_routing` is off, and even when on it requires a model validated on real data; applying predictions to costs is not implemented in this version.
- If CityPulse is unavailable the route is calculated normally and the response says `available: false`.
- The response states that this is based on stored reports, not a real-time hazard feed, and labels simulated data.

## 8. Gradient Descent Lab (CIE-3)
Model `y_hat_i = w . x_i + b`; loss `J(w,b) = (1/n) sum_i (y_hat_i - y_i)^2`;
`grad_w J = (2/n) X^T (X w + b 1 - y)`, `dJ/db = (2/n) sum_i (y_hat_i - y_i)`;
updates `w <- w - alpha grad_w J`, `b <- b - alpha dJ/db`.
Implemented explicitly in `gdlab.py` with NumPy only (no optimizer library). Features are standardised for training and the coefficients converted back to original units. **Data is synthetic:** `y = 2.2 distance_km + 9.0 traffic_slowdown + 3.0 road_roughness + 5 + noise`, reproducible by seed; no real measured travel times exist in this project.
The lab records the actual loss at every iteration (downsampled to at most 1000 points for the chart), reports initial/final loss, coefficients versus the generator and versus the closed-form least-squares solution (verification only), train/test MSE, RMSE, MAE, R^2, a finite-difference gradient check, and the stability limit `alpha < 2 / lambda_max((2/n) A^T A)` with `A = [X, 1]`. A too-large alpha stops early and is reported as diverged.

**Worked example** (computed by the same code; `GET /api/gdlab/worked-example`): x = [1, 2], y = [2, 3], w = b = 0, alpha = 0.1, n = 2.

| | step 1 | step 2 |
|---|---|---|
| y_hat | [0, 0] | [1.3, 2.1] |
| residuals | [-2, -3] | [-0.7, -0.9] |
| J | 6.5 | 0.65 |
| dJ/dw = (2/2)(1(-2) + 2(-3)) | -8 | -2.5 |
| dJ/db = (2/2)(-2 - 3) | -5 | -1.6 |
| w_new = w - 0.1 dJ/dw | 0.8 | 1.05 |
| b_new = b - 0.1 dJ/db | 0.5 | 0.66 |
| loss after step | 0.65 | 0.07085 |

## 9. Files
**New:** `citypulse/` (`config`, `store`, `geo`, `analysis`, `risk`, `ml`, `routing`, `demo`, `api`, `train`), `gdlab.py`, `gdlab_api.py`, `static/shared.css`, `static/citypulse.html`, `static/gdlab.html`, `tests/conftest.py`, `tests/test_citypulse.py`, `tests/test_gdlab.py`, `tests/ui_check_pages.js`, this document.
**Modified (additive):** `osm_service.py` (optional per-edge cost factor `extra`, internal node path), `osm_api.py` (optional `use_road_conditions`, new `road_conditions` response field; defaults unchanged), `app.py` (two routers, two page routes), `static/app.html` (navigation, "Use CityPulse road conditions" option and display), `README.md`.
