import time
from typing import Literal
import requests
from fastapi import APIRouter, HTTPException, Query
import core, osm_ml, traffic_live as tl
from osm_service import get_router, hotspots as _hotspots

router = APIRouter(prefix="/api/osm", tags=["map"])
Day = Literal["weekday", "weekend"]; Weather = Literal["dry", "rain"]; Engine = Literal["profile", "ml", "live"]
NO_ROUTE = "No route found with these settings. Try turning off the road filters."

@router.get("/status")
def status():
    R = get_router()
    return {"graph_loaded": True, "graph_source": R.source, "nodes": len(R.nodes), "edges": len(R.edges),
            "tomtom_key_set": bool(tl.key()), "ml_available": R.demo or osm_ml.model_exists(), "waterlogging_filter": R.demo}

@router.get("/places")
def places():
    """Junctions and well-known landmarks (approximate coordinates)."""
    return core.load_places()

_cache, _last = {}, [0.0]
def nominatim(q, limit):
    key = (q.lower(), limit)
    if key in _cache: return _cache[key]
    wait = 1.1 - (time.time() - _last[0])                       # Nominatim usage policy: at most 1 request per second
    if wait > 0: time.sleep(wait)
    _last[0] = time.time()
    r = requests.get("https://nominatim.openstreetmap.org/search", timeout=6, headers={"User-Agent": "SmartPathPilot-college-project/1.0"},
                     params={"q": q + ", Bengaluru", "format": "jsonv2", "limit": limit, "viewbox": "77.45,13.20,77.80,12.80", "bounded": 1, "countrycodes": "in"})
    r.raise_for_status()
    out = [{"name": ", ".join(x["display_name"].split(",")[:2]).strip(), "lat": float(x["lat"]), "lon": float(x["lon"]), "kind": "osm-search", "area": ""} for x in r.json()]
    _cache[key] = out; return out

@router.get("/search")
def search(q: str = Query("", max_length=80), limit: int = Query(8, ge=1, le=15), online: bool = True):
    """Destination search: known junctions/landmarks first, then (if the server has internet) OpenStreetMap's Nominatim."""
    P, toks = core.load_places(), q.lower().split()
    if not toks: return {"results": P[:limit], "online": False, "note": ""}
    def score(p):
        hay = (p["name"] + " " + p["area"]).lower()
        if not all(t in hay for t in toks): return None
        return 0 if p["name"].lower().startswith(q.lower().strip()) else 1 if any(w.startswith(toks[0]) for w in hay.split()) else 2
    local = sorted([(score(p), p["name"], p) for p in P if score(p) is not None], key=lambda x: x[:2])
    res, used, note = [x[2] for x in local[:limit]], False, ""
    if online and len(q.strip()) >= 3 and len(res) < limit:
        try: res += nominatim(q.strip(), limit - len(res)); used = True
        except Exception: note = "Online place search is unavailable; showing known landmarks only."
    return {"results": res, "online": used, "note": note}

def endpoints(R, src_lat, src_lon, dst_lat, dst_lon):
    s, ds, si = R.nearest(src_lat, src_lon); t, dt, ti = R.nearest(dst_lat, dst_lon)
    if max(ds, dt) > (15 if R.demo else 5): raise HTTPException(422, "A point is too far from the mapped road network.")
    if s == t: raise HTTPException(422, "Start and end snap to the same junction. Choose points further apart.")
    return s, t, si, ti

@router.get("/route")
def route(src_lat: float, src_lon: float, dst_lat: float, dst_lon: float, hour: int = Query(9, ge=0, le=23),
          day: Day = "weekday", weather: Weather = "dry", engine: Engine = "profile",
          w_time: float = Query(0.6, ge=0), w_dist: float = Query(0.2, ge=0), w_road: float = Query(0.2, ge=0),
          avoid_poor: bool = False, avoid_waterlogged: bool = False, use_road_conditions: bool = False):
    R = get_router(); s, t, si, ti = endpoints(R, src_lat, src_lon, dst_lat, dst_lon)
    mult, kind, note = R.multipliers(engine, hour, day, weather)
    ctx = None
    if use_road_conditions:
        from citypulse import routing as cp_routing           # CityPulse is optional: any failure leaves routing unchanged
        ctx = cp_routing.build(R)
    routes = R.route(s, t, mult, (w_time, w_dist, w_road), avoid_poor=avoid_poor, avoid_waterlogged=avoid_waterlogged, extra=ctx.factors if ctx is not None and ctx.available else None)
    if not routes: raise HTTPException(404, NO_ROUTE)
    for r in routes:
        nodes = r.pop("_nodes", None)
        if ctx is not None and nodes: r["road_conditions"] = ctx.annotate(nodes)
    sd = next((r for r in routes if r["is_shortest_distance"]), routes[0])
    return {"graph_source": R.source, "traffic": {"kind": kind, "verified_live": kind == "live", "note": note},
            "filters": {"avoid_poor": avoid_poor, "avoid_waterlogged": avoid_waterlogged and R.demo, "waterlogging_supported": R.demo},
            "snapped": {"src": si, "dst": ti}, "routes": routes, "road_conditions": ctx.summary() if ctx is not None else {"enabled": False},
            "time_saved_vs_shortest_min": round(sd["time_min"] - routes[0]["time_min"], 1)}

@router.get("/best-time")
def best_time(src_lat: float, src_lon: float, dst_lat: float, dst_lon: float, hour: int = Query(9, ge=0, le=23),
              day: Day = "weekday", weather: Weather = "dry", engine: Engine = "profile",
              w_time: float = Query(0.6, ge=0), w_dist: float = Query(0.2, ge=0), w_road: float = Query(0.2, ge=0),
              avoid_poor: bool = False, avoid_waterlogged: bool = False):
    if engine == "live": raise HTTPException(400, "Live traffic describes the present only. Use the profile or ML engine to compare departure hours.")
    R = get_router(); s, t, _, _ = endpoints(R, src_lat, src_lon, dst_lat, dst_lon); times = []
    for h in range(24):
        m, kind, _ = R.multipliers(engine, h, day, weather)
        q = R.quick(s, t, m, (w_time, w_dist, w_road), avoid_poor, avoid_waterlogged)
        if q is None: raise HTTPException(404, NO_ROUTE)
        times.append(round(q, 1))
    best = min(range(24), key=lambda h: times[h])
    return {"traffic_kind": kind, "times_min": times, "best_hour": best, "chosen_hour": hour, "saving_min": round(times[hour] - times[best], 1)}

@router.get("/hotspots")
def hotspots(hour: int = Query(9, ge=0, le=23), day: Day = "weekday", weather: Weather = "dry", engine: Engine = "profile", limit: int = Query(5, ge=1, le=15)):
    R = get_router(); m, kind, note = R.multipliers(engine, hour, day, weather)
    return {"traffic": {"kind": kind, "verified_live": kind == "live", "note": note}, "roads": _hotspots(R, m, limit)}
