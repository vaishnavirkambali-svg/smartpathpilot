"""ML traffic model trained on REAL readings collected by collect.py.
Train:  python osm_ml.py        (needs data/live_history.csv)"""
import json, os
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
LOG, MODEL = os.path.join(BASE, "data", "live_history.csv"), os.path.join(BASE, "models", "osm_traffic_model.joblib")
HW = ["motorway", "trunk", "primary", "secondary", "tertiary"]

def frame(hour, weekend, hw, lat, lon):
    hw = np.asarray(hw); n = len(hw)
    hour = np.broadcast_to(np.asarray(hour, float), (n,)); wk = np.broadcast_to(np.asarray(weekend, float), (n,))
    d = {"hs": np.sin(2 * np.pi * hour / 24), "hc": np.cos(2 * np.pi * hour / 24), "weekend": wk,
         "lat": np.asarray(lat, float), "lon": np.asarray(lon, float)}
    for h in HW: d["hw_" + h] = (hw == h).astype(float)
    return pd.DataFrame(d)

def train():
    df = pd.read_csv(LOG)
    if len(df) < 500: raise SystemExit(f"Only {len(df)} readings in {LOG}. Let collect.py run longer (aim for thousands).")
    df = df.sort_values("timestamp")
    t = pd.to_datetime(df.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    X = frame((t.dt.hour + t.dt.minute / 60).values, (t.dt.dayofweek >= 5).astype(int).values, df.hw.values, df.lat.values, df.lon.values)
    y = df.slowdown.values; cut = int(len(df) * 0.8)                       # chronological split: test on the newest 20%
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42).fit(X[:cut], y[:cut])
    p = m.predict(X[cut:]); base = np.full(len(p), y[:cut].mean())
    metrics = {"rows": len(df), "ml_mae": round(mean_absolute_error(y[cut:], p), 4), "ml_r2": round(r2_score(y[cut:], p), 4),
               "mean_baseline_mae": round(mean_absolute_error(y[cut:], base), 4)}
    os.makedirs(os.path.dirname(MODEL), exist_ok=True); joblib.dump({"model": m, "columns": list(X.columns), "metrics": metrics}, MODEL)
    print(json.dumps(metrics, indent=2))

_B = None
def model_exists(): return os.path.exists(MODEL)

def predict_edges(R, hour, day):
    """Predicted slowdown for every edge of the router's graph, or None if no model has been trained."""
    global _B
    if _B is None:
        if not os.path.exists(MODEL): return None
        try: _B = joblib.load(MODEL)
        except Exception: return None
    X = frame(hour, int(day == "weekend"), R.hw, [a for a, _ in R.mid_latlon], [b for _, b in R.mid_latlon])[_B["columns"]]
    return np.clip(_B["model"].predict(X), 1.0, None)

if __name__ == "__main__": train()
