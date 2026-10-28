"""SmartPathPilot core: data, traffic features, ML model, graph and multi-factor routing."""
import functools, math, os
import joblib, numpy as np, pandas as pd, networkx as nx
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODEL_PATH = os.path.join(BASE, "models", "traffic_model.joblib")
DEFAULT_WEIGHTS = (0.6, 0.2, 0.2)          # time, distance, road quality

def load_segments(): return pd.read_csv(os.path.join(DATA, "bengaluru_road_segments.csv"))
def load_nodes():    return pd.read_csv(os.path.join(DATA, "bengaluru_nodes.csv"))

def load_places():
    """Junctions of the demo network plus well-known landmarks (approximate coordinates), for search and pickers."""
    out = [{"name": r.location, "lat": r.latitude, "lon": r.longitude, "kind": "junction", "area": ""} for r in load_nodes().itertuples()]
    f = os.path.join(DATA, "landmarks.csv")
    if os.path.exists(f):
        out += [{"name": r.name, "lat": r.lat, "lon": r.lon, "kind": "landmark", "area": r.area} for r in pd.read_csv(f).itertuples()]
    return out

# ---------- rule-based traffic profile (baseline) ----------
def hour_profile(h):
    am = math.exp(-((h - 9.5) / 1.6) ** 2); pm = math.exp(-((h - 19) / 2.0) ** 2)
    return min(1.0, max(am, pm) + 0.05)

def seg_multiplier(r, hour, day="weekday", weather="dry"):
    off = r.offpeak_time_min / r.freeflow_time_min
    pk = r.peak_time_min / r.freeflow_time_min
    m = off + (pk - off) * hour_profile(hour)
    if day == "weekend": m = 1 + (m - 1) * 0.6
    if weather == "rain": m *= 1.25 if r.waterlogging_prone == "Yes" else 1.10
    return m

def rule_multipliers(seg, hour, day, weather):
    return np.array([seg_multiplier(r, hour, day, weather) for r in seg.itertuples()])

# ---------- ML model ----------
def make_features(seg_id, dist, pothole, waterlog, day, hour, weather, seg_ids):
    row = {"hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
           "is_weekend": int(day == "weekend"), "is_rain": int(weather == "rain"),
           "waterlogging": int(waterlog == "Yes"), "pothole": pothole, "distance_km": dist}
    for s in seg_ids: row["seg_" + s] = int(seg_id == s)
    return row

def train(save=True):
    seg = load_segments(); ids = sorted(seg.segment_id)
    hist = pd.read_csv(os.path.join(DATA, "bengaluru_traffic_history.csv"))
    X = pd.DataFrame([make_features(r.segment_id, r.distance_km, r.pothole_score_0to5, r.waterlogging_prone,
                                    r.day_type, r.hour, r.weather, ids) for r in hist.itertuples()])
    y = hist["congestion_multiplier"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    by_id = {r.segment_id: r for r in seg.itertuples()}
    base = [seg_multiplier(by_id[hist.segment_id[i]], hist.hour[i], hist.day_type[i], hist.weather[i]) for i in yte.index]
    metrics = {"model": "GradientBoostingRegressor", "train_rows": len(Xtr), "test_rows": len(Xte),
               "ml_mae": round(mean_absolute_error(yte, pred), 4), "ml_r2": round(r2_score(yte, pred), 4),
               "rule_mae": round(mean_absolute_error(yte, base), 4), "rule_r2": round(r2_score(yte, base), 4)}
    bundle = {"model": model, "columns": list(X.columns), "seg_ids": ids, "metrics": metrics}
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True); joblib.dump(bundle, MODEL_PATH)
    return bundle

@functools.lru_cache(maxsize=1)
def get_bundle():
    if os.path.exists(MODEL_PATH):
        try: return joblib.load(MODEL_PATH)
        except Exception: pass                     # e.g. scikit-learn version mismatch -> retrain
    return train()

def ml_multipliers(bundle, seg, hour, day, weather):
    X = pd.DataFrame([make_features(r.segment_id, r.distance_km, r.pothole_score_0to5, r.waterlogging_prone,
                                    day, hour, weather, bundle["seg_ids"]) for r in seg.itertuples()])[bundle["columns"]]
    return np.clip(bundle["model"].predict(X), 1.0, None)

# ---------- graph + routing ----------
def build_graph(seg, mults, avoid_poor=False, avoid_waterlogged=False):
    G = nx.Graph(); G.add_nodes_from(set(seg.from_location) | set(seg.to_location))
    for r, m in zip(seg.itertuples(), mults):
        if (avoid_poor and r.road_condition == "Poor") or (avoid_waterlogged and r.waterlogging_prone == "Yes"): continue
        t = r.freeflow_time_min * float(m) * (1 + 0.03 * r.pothole_score_0to5)
        G.add_edge(r.from_location, r.to_location, dist=r.distance_km, time=t, pothole=r.pothole_score_0to5,
                   road=r.road_name, seg=r.segment_id, mult=float(m), condition=r.road_condition,
                   waterlogged=r.waterlogging_prone == "Yes")
    return G

def path_detail(G, p):
    es = [G[a][b] for a, b in zip(p, p[1:])]
    d = sum(e["dist"] for e in es); t = sum(e["time"] for e in es)
    rp = sum(e["pothole"] * e["dist"] for e in es) / d / 5
    segs = [{"segment_id": e["seg"], "road": e["road"], "from": a, "to": b, "distance_km": round(e["dist"], 2),
             "time_min": round(e["time"], 1), "traffic_multiplier": round(e["mult"], 2),
             "road_condition": e["condition"], "waterlogging_prone": e["waterlogged"]}
            for (a, b), e in zip(zip(p, p[1:]), es)]
    return d, t, rp, segs

def best_routes(G, src, dst, k=3, weights=DEFAULT_WEIGHTS):
    paths = []
    try:
        for p in nx.shortest_simple_paths(G, src, dst, weight="time"):
            paths.append(p)
            if len(paths) == k: break
        sd = nx.shortest_path(G, src, dst, weight="dist")
    except nx.NetworkXNoPath:
        return []
    if sd not in paths: paths.append(sd)
    rows = [(p, *path_detail(G, p)) for p in paths]
    tmin = min(r[2] for r in rows); dmin = min(r[1] for r in rows)
    wt, wd, wr = weights; ws = (wt + wd + wr) or 1
    out = []
    for p, d, t, rp, segs in rows:
        score = (wt * t / tmin + wd * d / dmin + wr * (1 + rp)) / ws
        out.append({"path": p, "distance_km": round(d, 2), "time_min": round(t, 1), "road_roughness": round(rp, 2),
                    "score": round(score, 3), "is_shortest_distance": p == sd, "segments": segs})
    out.sort(key=lambda x: x["score"])
    for i, r in enumerate(out, 1): r["rank"] = i; r["recommended"] = i == 1
    return out
