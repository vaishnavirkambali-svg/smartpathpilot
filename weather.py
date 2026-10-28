"""Weather along a route from Open-Meteo (no API key needed for non-commercial use).
Honesty rules: values come only from the provider response; nothing is invented. 'current' from Open-Meteo is a
MODEL-BASED estimate of present conditions (not a station observation), so it is labelled that way. Open-Meteo has no
official alerts; alerts need an optional OpenWeather key (OPENWEATHER_API_KEY). Env vars: OPEN_METEO_API_KEY (commercial plan, optional)."""
import os, threading, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

IST = ZoneInfo("Asia/Kolkata"); TTL = 900; FAIL_TTL = 60; TIMEOUT = 6
_cache, _lock = {}, threading.Lock()

class WeatherUnavailable(Exception): pass

# ASSUMED thresholds / factors (documented in README). Not calibrated for Bengaluru.
HEAVY_RAIN, MODERATE_RAIN, STRONG_WIND, STRONG_GUST, LOW_VIS, VERY_LOW_VIS, HOT = 7.6, 2.5, 40.0, 60.0, 1000.0, 200.0, 38.0
RAIN_SLOWDOWN = [(0.1, 1.00), (2.5, 1.05), (7.6, 1.10), (15.0, 1.20), (float("inf"), 1.30)]   # (mm/h upper bound, assumed time factor)
CODES = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
         61: "light rain", 63: "rain", 65: "heavy rain", 80: "light showers", 81: "showers", 82: "violent showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}

def now_ist(): return datetime.now(IST).replace(tzinfo=None)

def _http_get(url, params):
    r = requests.get(url, params=params, timeout=TIMEOUT); r.raise_for_status(); return r.json()

def _parse(j):
    h = j["hourly"]; t = [datetime.fromisoformat(x) for x in h["time"]]
    hours = {t[i].replace(minute=0, second=0, microsecond=0): {k: h[k][i] for k in ("temperature_2m", "precipitation", "precipitation_probability", "visibility", "wind_speed_10m", "wind_gusts_10m", "weather_code")} for i in range(len(t))}
    c = j.get("current"); cur = None
    if c: cur = {"time": datetime.fromisoformat(c["time"]), **{k: c.get(k) for k in ("temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "weather_code")}}
    return {"hours": hours, "current": cur}

def fetch_point(lat, lon):
    key = (round(lat, 2), round(lon, 2)); t = time.time()
    with _lock:
        hit = _cache.get(("w",) + key)
        if hit and hit[1] is None and t - hit[0] < FAIL_TTL: raise WeatherUnavailable("Open-Meteo recently failed for this location; not retrying yet.")
        if hit and hit[1] is not None and t - hit[0] < TTL: return hit[1]
    apikey = os.getenv("OPEN_METEO_API_KEY")
    params = {"latitude": key[0], "longitude": key[1], "timezone": "Asia/Kolkata", "forecast_days": 3, "wind_speed_unit": "kmh",
              "current": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,weather_code",
              "hourly": "temperature_2m,precipitation,precipitation_probability,visibility,wind_speed_10m,wind_gusts_10m,weather_code"}
    url = "https://api.open-meteo.com/v1/forecast"
    if apikey: url, params["apikey"] = "https://customer-api.open-meteo.com/v1/forecast", apikey
    try: data = _parse(_http_get(url, params))
    except Exception as e:
        with _lock: _cache[("w",) + key] = (t, None)                      # remember the failure briefly so a down provider is not hammered
        raise WeatherUnavailable(f"Open-Meteo request failed ({type(e).__name__}).") from e
    with _lock: _cache[("w",) + key] = (t, data)
    return data

def fetch_alerts(lat, lon):
    k = os.getenv("OPENWEATHER_API_KEY")
    if not k: return {"available": False, "reason": "No alert provider configured. Set OPENWEATHER_API_KEY to enable official alerts (OpenWeather One Call 3.0, subscription required). Open-Meteo provides no alerts."}
    ck = ("a", round(lat, 1), round(lon, 1)); t = time.time()
    with _lock:
        hit = _cache.get(ck)
        if hit and t - hit[0] < TTL: return hit[1]
    try: j = _http_get("https://api.openweathermap.org/data/3.0/onecall", {"lat": lat, "lon": lon, "exclude": "minutely,hourly,daily", "appid": k})
    except Exception as e: return {"available": False, "reason": f"Alert request failed ({type(e).__name__})."}
    out = {"available": True, "provider": "openweathermap", "alerts": [{"event": a.get("event"), "sender": a.get("sender_name"), "start": a.get("start"), "end": a.get("end"), "description": (a.get("description") or "")[:300]} for a in j.get("alerts", [])]}
    with _lock: _cache[ck] = (t, out)
    return out

def rain_factor(mm_h): return next(f for ub, f in RAIN_SLOWDOWN if mm_h < ub)

def _sample(p, when, now, data):
    cur = data.get("current")
    if cur and abs((when - now).total_seconds()) <= 1800:
        kind, d = "current_model_estimate", cur; prob = None; vis = None
        # 'current' has no visibility/probability: take them from the matching hour when available
        hr = data["hours"].get(now.replace(minute=0, second=0, microsecond=0)) or {}
        prob, vis = hr.get("precipitation_probability"), hr.get("visibility")
        v = {"temperature_c": d["temperature_2m"], "precipitation_mm_h": d["precipitation"], "precipitation_probability_pct": prob, "wind_kmh": d["wind_speed_10m"], "gust_kmh": d["wind_gusts_10m"], "visibility_m": vis, "weather_code": d["weather_code"]}
    else:
        hr = data["hours"].get((when + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0))
        if hr is None: return None
        kind = "forecast"
        v = {"temperature_c": hr["temperature_2m"], "precipitation_mm_h": hr["precipitation"], "precipitation_probability_pct": hr["precipitation_probability"], "wind_kmh": hr["wind_speed_10m"], "gust_kmh": hr["wind_gusts_10m"], "visibility_m": hr["visibility"], "weather_code": hr["weather_code"]}
    v["summary"] = CODES.get(v["weather_code"], "unknown conditions")
    return {"lat": p["lat"], "lon": p["lon"], "offset_min": round(p["offset_min"]), "time": when.isoformat(timespec="minutes"), "type": kind, **v}

def _warn(rows):
    w = []; num = lambda r, k: r[k] if isinstance(r.get(k), (int, float)) else None
    def add(level, typ, text): 
        if not any(x["type"] == typ and x["level"] == level for x in w): w.append({"level": level, "type": typ, "text": text})
    for r in rows:
        p, pp, wd, g, vis, tc, code = (num(r, k) for k in ("precipitation_mm_h", "precipitation_probability_pct", "wind_kmh", "gust_kmh", "visibility_m", "temperature_c", "weather_code"))
        if p is not None and p >= HEAVY_RAIN: add("severe", "heavy_rain", f"Heavy rain ({p:.1f} mm/h) forecast near {r['lat']:.3f},{r['lon']:.3f} at {r['time']}.")
        elif p is not None and p >= MODERATE_RAIN: add("warning", "moderate_rain", f"Moderate rain ({p:.1f} mm/h) expected along the route.")
        elif pp is not None and pp >= 70: add("info", "rain_likely", f"Rain is likely ({pp:.0f}% chance) along the route.")
        if (wd is not None and wd >= STRONG_WIND) or (g is not None and g >= STRONG_GUST): add("warning", "strong_wind", f"Strong wind ({max(x for x in (wd, g) if x is not None):.0f} km/h) expected.")
        if vis is not None and vis < VERY_LOW_VIS: add("severe", "very_low_visibility", f"Very low visibility ({vis:.0f} m).")
        elif vis is not None and vis < LOW_VIS: add("warning", "low_visibility", f"Low visibility ({vis:.0f} m).")
        if tc is not None and tc >= HOT: add("warning", "high_temperature", f"High temperature ({tc:.0f} \u00b0C).")
        if code in (95, 96, 99): add("severe", "thunderstorm", "Thunderstorm expected along the route.")
    return w

def assess(points, depart=None, now=None):
    """points: [{lat, lon, offset_min}] along the route. Returns a dict; never raises for provider problems."""
    now = now or now_ist(); depart = depart or now; rows, errors = [], []
    for p in points:
        try: data = fetch_point(p["lat"], p["lon"])
        except WeatherUnavailable as e:
            errors.append(str(e))
            if not rows: break                                            # provider looks down: do not wait for every remaining point
            continue
        s = _sample(p, depart + timedelta(minutes=p["offset_min"]), now, data)
        if s is None: errors.append("Forecast does not cover the requested time (provider forecasts about 3 days ahead).")
        else: rows.append(s)
    if not rows:
        return {"available": False, "provider": "open-meteo", "reason": errors[0] if errors else "No weather points to assess.", "warnings": [], "samples": []}
    rows.sort(key=lambda r: r["offset_min"]); pr = [r["precipitation_mm_h"] or 0.0 for r in rows]
    exposure = sum((pr[i] + pr[i + 1]) / 2 * ((rows[i + 1]["offset_min"] - rows[i]["offset_min"]) / 60) for i in range(len(rows) - 1))
    mean_p = sum(pr) / len(pr); vals = lambda k: [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    mid = rows[len(rows) // 2]; alerts = fetch_alerts(mid["lat"], mid["lon"])
    warnings = _warn(rows)
    if alerts.get("available"): warnings += [{"level": "severe", "type": "official_alert", "text": f"{a['event']} ({a['sender']})"} for a in alerts["alerts"]]
    return {"available": True, "provider": "open-meteo", "depart_at": depart.isoformat(timespec="minutes"), "samples": rows, "warnings": warnings, "alerts": alerts,
            "summary": {"max_precipitation_mm_h": max(pr), "max_wind_kmh": max(vals("wind_kmh"), default=None), "min_visibility_m": min(vals("visibility_m"), default=None),
                        "max_temperature_c": max(vals("temperature_c"), default=None), "estimated_rain_exposure_mm": round(exposure, 2)},
            "time_adjustment": {"factor": rain_factor(mean_p), "basis": "ASSUMED rain slowdown table (not calibrated for Bengaluru); applied to the estimate only, the route itself is not changed by weather."},
            "partial": bool(errors), "errors": errors[:3],
            "limitations": ["Weather is sampled at a few points along the route, not on every road.", "Forecasts are model output, not observations; 'current' is a model-based estimate.",
                            "The planner cannot predict flooding or which road will be safest; weather is reported, not used to reroute."]}
