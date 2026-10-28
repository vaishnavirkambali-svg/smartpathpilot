"""SmartPathPilot API. Run: python -m uvicorn app:app --reload   (then open http://127.0.0.1:8000)
The map engine (osm_service.Router) is the single routing engine. The original /api/route endpoints use the SAME engine
on the 21-junction simulated network, so website, Android app and the original API give identical results."""
import os
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import core
from osm_api import router as osm_router
from multi_api import router as multi_router
from citypulse.api import router as citypulse_router
from gdlab_api import router as gdlab_router
from osm_service import get_demo_router

app = FastAPI(title="SmartPathPilot API", version="2.0",
              description="Multi-factor route optimisation for Bengaluru. Every response says whether traffic is simulated, ML-predicted or live.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(osm_router)
app.include_router(multi_router)
app.include_router(citypulse_router)
app.include_router(gdlab_router)

SEG, NODES, BUNDLE = core.load_segments(), core.load_nodes(), core.get_bundle()
PLACES = list(NODES.location)
Day = Literal["weekday", "weekend"]; Weather = Literal["dry", "rain"]; Engine = Literal["ml", "rule", "profile"]
KIND = {"ml": "simulated-ml", "rule": "simulated-profile", "profile": "simulated-profile"}
_cache = {}

def all_hours(day, weather, engine):
    key = (day, weather, engine)
    if key not in _cache:
        fn = (lambda h: core.ml_multipliers(BUNDLE, SEG, h, day, weather)) if engine == "ml" else (lambda h: core.rule_multipliers(SEG, h, day, weather))
        _cache[key] = [fn(h) for h in range(24)]
    return _cache[key]

def check_places(*names):
    for n in names:
        if n not in PLACES: raise HTTPException(404, f"Unknown place '{n}'. Use one of: {', '.join(PLACES)}")

@app.get("/api/health")
def health(): return {"status": "ok", "model": BUNDLE["metrics"]["model"], "segments": len(SEG), "places": len(PLACES)}

@app.get("/api/metrics")
def metrics(): return BUNDLE["metrics"]

@app.get("/api/nodes")
def nodes(): return NODES.to_dict("records")

@app.get("/api/segments")
def segments(): return SEG.to_dict("records")

@app.get("/api/predictions")
def predictions(day: Day = "weekday", weather: Weather = "dry", engine: Engine = "ml"):
    """Traffic multiplier (1.0 = free flow) per hour (rows 0-23) and segment (CSV order). Simulated or ML-predicted, never live."""
    e = "ml" if engine == "ml" else "rule"
    return {"engine": engine, "traffic_kind": KIND[engine], "segment_ids": list(SEG.segment_id),
            "multipliers": [[round(float(x), 3) for x in row] for row in all_hours(day, weather, e)]}

def classic_route(R, r, mult, best_cost):
    ids = [R.name2id[n] for n in r["path"]]; segs = []
    for a, b in zip(ids, ids[1:]):
        e = (a, b); d = R.G.edges[e]; i = R.eix[e]; row = SEG.iloc[d["seg"]]
        segs.append({"segment_id": row.segment_id, "road": row.road_name, "from": R.G.nodes[a]["name"], "to": R.G.nodes[b]["name"],
                     "distance_km": round(R.len[i] / 1000, 2), "time_min": round(R.ff[i] * mult[i] * (1 + .03 * R.rough[i]) / 60, 1),
                     "traffic_multiplier": round(float(mult[i]), 2), "road_condition": row.road_condition, "waterlogging_prone": row.waterlogging_prone == "Yes"})
    return {"rank": r["rank"], "recommended": r["recommended"], "path": r["path"], "distance_km": r["distance_km"], "time_min": r["time_min"],
            "road_roughness": r["road_roughness"], "traffic_level": r["traffic_level"], "score": round(r["cost"] / best_cost, 3),
            "is_shortest_distance": r["is_shortest_distance"], "segments": segs}

@app.get("/api/route")
def route(src: str, dst: str, hour: int = Query(9, ge=0, le=23), day: Day = "weekday", weather: Weather = "dry",
          engine: Engine = "ml", w_time: float = Query(0.6, ge=0), w_dist: float = Query(0.2, ge=0),
          w_road: float = Query(0.2, ge=0), avoid_poor: bool = False, avoid_waterlogged: bool = False):
    check_places(src, dst)
    if src == dst: raise HTTPException(422, "Source and destination must be different.")
    R = get_demo_router(); eng = "ml" if engine == "ml" else "profile"
    mult, kind, note = R.multipliers(eng, hour, day, weather)
    rs = R.route(R.name2id[src], R.name2id[dst], mult, (w_time, w_dist, w_road), avoid_poor=avoid_poor, avoid_waterlogged=avoid_waterlogged)
    if not rs: raise HTTPException(404, "No route found with the selected filters.")
    routes = [classic_route(R, r, mult, rs[0]["cost"] or 1) for r in rs]
    sd = next((r for r in routes if r["is_shortest_distance"]), routes[0])
    return {"engine": engine, "traffic": {"kind": kind, "verified_live": False, "note": note}, "src": src, "dst": dst, "hour": hour,
            "day": day, "weather": weather, "recommended": routes[0], "time_saved_vs_shortest_min": round(sd["time_min"] - routes[0]["time_min"], 1), "routes": routes}

@app.get("/api/best-time")
def best_time(src: str, dst: str, day: Day = "weekday", weather: Weather = "dry", engine: Engine = "ml",
              hour: int = Query(9, ge=0, le=23), avoid_poor: bool = False, avoid_waterlogged: bool = False):
    check_places(src, dst)
    if src == dst: raise HTTPException(422, "Source and destination must be different.")
    R = get_demo_router(); eng = "ml" if engine == "ml" else "profile"; times = []
    for h in range(24):
        m, kind, _ = R.multipliers(eng, h, day, weather)
        q = R.quick(R.name2id[src], R.name2id[dst], m, avoid_poor=avoid_poor, avoid_waterlogged=avoid_waterlogged)
        if q is None: raise HTTPException(404, "No route found with the selected filters.")
        times.append(round(q, 1))
    best = min(range(24), key=lambda h: times[h])
    return {"traffic_kind": kind, "times_min": times, "best_hour": best, "chosen_hour": hour, "saving_min": round(times[hour] - times[best], 1)}

@app.get("/api/hotspots")
def hotspots(hour: int = Query(9, ge=0, le=23), day: Day = "weekday", weather: Weather = "dry", engine: Engine = "ml", limit: int = Query(5, ge=1, le=20)):
    e = "ml" if engine == "ml" else "rule"; m = all_hours(day, weather, e)[hour]
    order = sorted(range(len(m)), key=lambda i: -m[i])[:limit]
    return [{"segment_id": SEG.segment_id[i], "road": SEG.road_name[i], "from": SEG.from_location[i], "to": SEG.to_location[i],
             "slowdown_x": round(float(m[i]), 2), "traffic_kind": KIND[engine]} for i in order]

@app.get("/", include_in_schema=False)
def index(): return FileResponse(os.path.join(core.BASE, "static", "app.html"))

@app.get("/classic", include_in_schema=False)
def classic(): return FileResponse(os.path.join(core.BASE, "static", "index.html"))

@app.get("/citypulse", include_in_schema=False)
def citypulse_page(): return FileResponse(os.path.join(core.BASE, "static", "citypulse.html"))

@app.get("/gradient-descent-lab", include_in_schema=False)
def gdlab_page(): return FileResponse(os.path.join(core.BASE, "static", "gdlab.html"))

@app.get("/osm", include_in_schema=False)
def osm_page(): return FileResponse(os.path.join(core.BASE, "static", "app.html"))

app.mount("/static", StaticFiles(directory=os.path.join(core.BASE, "static")), name="static")
