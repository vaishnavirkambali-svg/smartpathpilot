"""Advanced routing endpoints: multi-stop, road-quality preference, weather assessment, traffic profiles.
Existing endpoints (/api/osm/route etc.) are unchanged; everything here is additive."""
import os
from datetime import datetime
from typing import List, Literal, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import multistop, osm_ml, road_quality, traffic_live as tl, weather
from osm_service import LiveUnavailable, get_router

router = APIRouter(tags=["advanced"])
def max_stops(): return int(os.getenv("SPP_MAX_STOPS", "20"))

class Point(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    name: Optional[str] = Field(None, max_length=80)

class MultiRequest(BaseModel):
    source: Point
    destination: Point
    stops: List[Point] = Field(default_factory=list, max_length=50)
    mode: Literal["optimized", "strict", "compare"] = "optimized"
    objective: Literal["fastest", "shortest"] = "fastest"
    road_preference: Literal["none", "balanced", "better_roads"] = "none"
    traffic_profile: Literal["free_flow", "typical", "peak", "live"] = "typical"
    hour: int = Field(9, ge=0, le=23)
    day: Literal["weekday", "weekend"] = "weekday"
    depart_at: Optional[str] = Field(None, description="ISO time (IST if no timezone). Overrides hour/day and sets the weather forecast time.")
    weather_assessment: bool = False

@router.get("/api/traffic/profiles")
def traffic_profiles():
    R = get_router(); live_ok = bool(tl.key())
    out = []
    for p in ("free_flow", "typical", "peak", "live"):
        i = R.profile_info(p); i["available"] = live_ok if p == "live" else True
        if p == "live" and not live_ok: i["reason"] = "TOMTOM_KEY is not set. Live traffic is not configured, so it is not offered."
        out.append(i)
    return {"profiles": out}

@router.get("/api/weather/status")
def weather_status():
    return {"provider": "open-meteo", "key_required": False, "commercial_key_set": bool(os.getenv("OPEN_METEO_API_KEY")),
            "alerts_provider_configured": bool(os.getenv("OPENWEATHER_API_KEY")), "cache_ttl_s": weather.TTL}

@router.get("/api/roadquality/config")
def roadquality_config():
    return {"preferences": road_quality.ALPHA, "surface_scores": road_quality.SURFACE, "smoothness_scores": road_quality.SMOOTHNESS,
            "unknown_policy": "neutral (no penalty); missing tags are never assumed poor", "notes": road_quality.SOURCE_NOTE}

def _snap(R, p, label):
    n, d, info = R.nearest(p.lat, p.lon)
    if d > (15 if R.demo else 5): raise HTTPException(422, f"{label} is too far ({d:.1f} km) from the mapped road network.")
    info["req_lat"], info["req_lon"] = p.lat, p.lon
    return n, info

@router.post("/api/osm/route/multi")
def route_multi(req: MultiRequest):
    R = get_router(); warnings = []
    if len(req.stops) > max_stops(): raise HTTPException(422, f"At most {max_stops()} mandatory stops are supported (got {len(req.stops)}).")
    depart = None
    if req.depart_at:
        try: depart = datetime.fromisoformat(req.depart_at.replace("Z", "+00:00"))
        except ValueError: raise HTTPException(422, "depart_at must be an ISO 8601 time, e.g. 2026-10-15T09:00.")
        if depart.tzinfo: depart = depart.astimezone(weather.IST).replace(tzinfo=None)
    hour, day = (depart.hour, "weekend" if depart.weekday() >= 5 else "weekday") if depart else (req.hour, req.day)
    s_node, s_info = _snap(R, req.source, "Source"); d_node, d_info = _snap(R, req.destination, "Destination")
    nodes, labels, infos, seen = [s_node], [req.source.name or "Source"], [s_info], {s_node, d_node}
    for k, st in enumerate(req.stops, 1):
        label = st.name or f"Stop {k}"; n, info = _snap(R, st, label)
        if n in seen:
            warnings.append(f"{label} was dropped: it snaps to the same junction as the source, destination or an earlier stop."); continue
        seen.add(n); nodes.append(n); labels.append(label); infos.append(info)
    nodes.append(d_node); labels.append(req.destination.name or "Destination"); infos.append(d_info)
    if len(nodes) == 2 and s_node == d_node: raise HTTPException(422, "Source and destination snap to the same junction and there are no stops. Choose points further apart.")
    try: mult, tinfo = R.profile_multipliers(req.traffic_profile, hour, day)
    except LiveUnavailable as e: raise HTTPException(409, f"Live traffic is not available: {e}. Choose free_flow, typical or peak, or configure TOMTOM_KEY.")
    try: res = R.multi(nodes, labels, infos, mult, req.objective, req.road_preference, req.mode, max_stops())
    except multistop.RoutingError as e: raise HTTPException(e.status, str(e))
    notes = ["Distances and times are computed on the road graph, never straight-line distance.",
             "Travel time = free-flow time x traffic slowdown. It excludes the legacy rough-road slowdown used by /api/osm/route, and road quality is reported separately as a penalty.",
             "Road-quality preference changes the route choice only; reported distance and time are physical estimates."]
    if req.traffic_profile != "live": notes.append("Traffic is NOT live: " + tinfo["source"] + ".")
    for r in res.values():
        ws = r.pop("_weather_samples"); r.pop("_cost", None)
        if req.weather_assessment: r["weather"] = weather.assess(ws, depart)
    if req.weather_assessment and len(res) == 2 and all(r["weather"].get("available") for r in res.values()):
        a, b = (res[k]["weather"]["summary"]["estimated_rain_exposure_mm"] for k in ("optimized", "strict"))
        wcmp = {"optimized_rain_exposure_mm": a, "strict_rain_exposure_mm": b, "lower_exposure": "optimized" if a < b else "strict" if b < a else "equal",
                "caveat": "Exposure is estimated from a few sample points. It does not show which road is safer."}
    else: wcmp = None
    out = {"mode": req.mode, "objective": req.objective, "road_preference": req.road_preference, "traffic": tinfo, "graph_source": R.source,
           "depart": {"hour": hour, "day": day, "depart_at": depart.isoformat(timespec="minutes") if depart else None}, "warnings": warnings, "notes": notes,
           "results": res, "result": res.get("optimized") or res["strict"]}
    if len(res) == 2:
        o, s = res["optimized"], res["strict"]
        out["comparison"] = {"same_order": [x["index"] for x in o["order"]] == [x["index"] for x in s["order"]], "time_saved_min": round(s["time_min"] - o["time_min"], 1),
                             "distance_saved_km": round(s["distance_km"] - o["distance_km"], 2), "weather": wcmp}
    return out
