"""Traffic for the OSM graph: a time-of-day profile, or live TomTom readings spread over nearby roads.
Set your key first:  export TOMTOM_KEY=...   (check your TomTom plan's daily request quota)."""
import csv, math, os, time
import numpy as np, requests
from scipy.spatial import cKDTree

URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
PEAK = {"motorway": 1.6, "trunk": 1.9, "primary": 2.3, "secondary": 2.1, "tertiary": 1.7}   # ASSUMED peak slowdowns
RANK = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4}
DAMP = {"motorway": 1, "trunk": 1, "primary": 1, "secondary": .9, "tertiary": .7}
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "live_history.csv")

def key(): return os.getenv("TOMTOM_KEY")

def hour_profile(h):
    return min(1.0, max(math.exp(-((h - 9.5) / 1.6) ** 2), math.exp(-((h - 19) / 2.0) ** 2)) + 0.05)

def profile_mult(hw, hour, day, weather):
    m = np.array([1 + (PEAK.get(x, 1.7) - 1) * hour_profile(hour) for x in hw])
    if day == "weekend": m = 1 + (m - 1) * 0.6
    return m * (1.12 if weather == "rain" else 1.0)

def fetch_flow(lat, lon):
    r = requests.get(URL, params={"key": key(), "point": f"{lat},{lon}", "unit": "KMPH"}, timeout=10)
    r.raise_for_status(); d = r.json()["flowSegmentData"]
    return float(d["currentSpeed"]), float(d["freeFlowSpeed"])

def pick_samples(R, n=120):
    """One main road per ~1.7 km grid cell, most important roads first."""
    cells = {}
    for i, hw in enumerate(R.hw):
        if hw not in ("motorway", "trunk", "primary", "secondary"): continue
        c = (round(R.mid[i][0] / 1.7), round(R.mid[i][1] / 1.7))
        if c not in cells or RANK[hw] < RANK[R.hw[cells[c]]]: cells[c] = i
    return sorted(cells.values(), key=lambda i: RANK[R.hw[i]])[:n]

def snapshot(R, samples, pause=0.15):
    rows = []
    for i in samples:
        try: cur, free = fetch_flow(*R.mid_latlon[i])
        except Exception: continue
        rows.append((i, max(1.0, free / max(cur, 1.0)))); time.sleep(pause)
    if rows:                                              # keep every reading: this becomes your real training history
        new = not os.path.exists(LOG); os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["timestamp", "u", "v", "lat", "lon", "hw", "slowdown"])
            for i, r in rows: w.writerow([int(time.time()), *R.edges[i], *R.mid_latlon[i], R.hw[i], round(r, 3)])
    return rows

def spread(R, rows):
    """Every edge gets the slowdown of its 3 nearest measured roads (inverse-distance weighted)."""
    idx = [i for i, _ in rows]; val = np.array([r for _, r in rows])
    d, j = cKDTree(R.mid[idx]).query(R.mid, k=min(3, len(idx)))
    d = np.maximum(d.reshape(len(R.mid), -1), 0.05); j = j.reshape(len(R.mid), -1); w = 1 / d
    r = (val[j] * w).sum(1) / w.sum(1); r[d[:, 0] > 4] = np.median(val)     # >4 km from any reading: city median
    return 1 + (r - 1) * np.array([DAMP.get(h, .7) for h in R.hw])
