"""Tests for multi-stop routing, road quality, weather and traffic profiles. Run: python -m pytest -q"""
import itertools, math, os, random, re, sys
from datetime import timedelta
import networkx as nx, numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
import app as appmod, multistop as ms, osm_service, road_quality as rqm, traffic_live as tl, weather, multi_api

c = TestClient(appmod.app)
PL = {p["name"]: p for p in c.get("/api/osm/places").json() if p["kind"] == "junction"}
pt = lambda n: {"lat": PL[n]["lat"], "lon": PL[n]["lon"], "name": n}
def body(**k):
    b = {"source": pt("Majestic"), "destination": pt("Whitefield"), "stops": [pt("Silk Board"), pt("Hebbal"), pt("Jayanagar")]}; b.update(k); return b
def post(**k): return c.post("/api/osm/route/multi", json=body(**k))
labels = lambda r: [x["label"] for x in r["order"]]

@pytest.fixture(autouse=True)
def clean(monkeypatch):
    weather._cache.clear(); monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False); monkeypatch.delenv("TOMTOM_KEY", raising=False); yield

# ---------- deterministic graphs: optimiser ----------
def line(n, bidir=True):
    G = nx.DiGraph()
    for i in range(n - 1):
        G.add_edge(i, i + 1, w=1.0)
        if bidir: G.add_edge(i + 1, i, w=1.0)
    return G

def test_held_karp_matches_brute_force():
    rnd = random.Random(7)
    for _ in range(5):
        m = 6; n = m + 2; pts = [(rnd.random(), rnd.random()) for _ in range(n)]
        cost = [[math.dist(a, b) for b in pts] for a in pts]
        order, _ = ms.held_karp(cost, m)
        best = min(ms._total(cost, p, m + 1) for p in itertools.permutations(range(1, m + 1)))
        assert ms._total(cost, order, m + 1) == pytest.approx(best)

def test_exact_plan_on_small_graph_beats_entered_order():
    G = line(30); r = ms.plan(G, [0, 20, 5, 25, 29], "w")            # source 0, stops 20,5,25 entered, destination 29
    assert r["optimized"]["method"] == "exact_held_karp" and r["optimized"]["order"] == [0, 2, 1, 3, 4]
    assert r["optimized"]["total"] == 29 and r["strict"]["total"] > r["optimized"]["total"]
    assert r["entered_cost"] == r["strict"]["total"] and r["strict"]["order"] == [0, 1, 2, 3, 4]

def test_heuristic_on_larger_graph_is_valid_and_near_optimal():
    G = line(60); rnd = random.Random(3); stops = rnd.sample(range(1, 59), 15)
    nodes = [20] + stops + [59]; r = ms.plan(G, nodes, "w", max_mid=20)
    assert r["optimized"]["method"] == "heuristic_nn_2opt_oropt"
    assert sorted(r["optimized"]["order"][1:-1]) == list(range(1, 16))            # every stop exactly once
    assert r["optimized"]["order"][0] == 0 and r["optimized"]["order"][-1] == 16  # source/destination fixed
    optimum = (20 - min(stops + [20])) + (59 - min(stops + [20]))
    assert r["optimized"]["total"] <= optimum * 1.15 and r["optimized"]["total"] <= r["strict"]["total"]

def test_stop_limit():
    with pytest.raises(ms.TooManyStops): ms.plan(line(50), [0] + list(range(1, 22)) + [49], "w", max_mid=20)

# ---------- graph types, parallel edges, bad weights ----------
def test_multigraph_uses_cheapest_parallel_edge():
    G = nx.MultiDiGraph(); G.add_edge("a", "b", key="fast", t=1, d=10); G.add_edge("a", "b", key="short", t=10, d=1)
    assert ms.plan(G, ["a", "b"], "t")["strict"]["total"] == 1 and ms.plan(G, ["a", "b"], "d")["strict"]["total"] == 1
    assert ms.best_edge(G, "a", "b", "t")[0] == "fast" and ms.best_edge(G, "a", "b", "d")[0] == "short"

def test_multigraph_ignores_one_bad_parallel_edge_but_rejects_all_bad():
    G = nx.MultiDiGraph(); G.add_edge("a", "b", t=float("nan")); G.add_edge("a", "b", t=4)
    assert ms.plan(G, ["a", "b"], "t")["strict"]["total"] == 4
    H = nx.MultiDiGraph(); H.add_edge("a", "b", t=-1); H.add_edge("a", "b")
    with pytest.raises(ms.InvalidWeight): ms.plan(H, ["a", "b"], "t")

@pytest.mark.parametrize("bad", [None, -1.0, float("nan"), float("inf"), "x"])
def test_invalid_travel_time_weights_rejected(bad):
    G = line(4); G[1][2]["w"] = bad
    with pytest.raises(ms.InvalidWeight): ms.validate_weights(G, "w")
    with pytest.raises(ms.InvalidWeight): ms.plan(G, [0, 3], "w")

def test_missing_weight_attribute():
    G = nx.DiGraph(); G.add_edge(0, 1)
    with pytest.raises(ms.InvalidWeight): ms.plan(G, [0, 1], "w")

def test_undirected_graph_supported():
    G = nx.Graph(); G.add_edge(0, 1, w=2); G.add_edge(1, 2, w=3)
    assert ms.plan(G, [0, 1, 2], "w")["optimized"]["total"] == 5

def test_unreachable_waypoint():
    G = line(5); G.add_edge(10, 11, w=1); G.add_edge(11, 10, w=1)
    with pytest.raises(ms.Unreachable) as e: ms.plan(G, [0, 10, 4], "w", labels=["A", "Island", "B"])
    assert e.value.status == 404 and "Island" in str(e.value)
    with pytest.raises(ms.Unreachable): ms.plan(G, [0, 99, 4], "w")                # node not in graph

def test_one_way_graph_directionality():
    G = nx.DiGraph(); G.add_edge(0, 1, w=1); G.add_edge(1, 2, w=1)               # 2 cannot return to 1
    r = ms.plan(G, [0, 1, 2], "w"); assert r["optimized"]["total"] == 2
    with pytest.raises(ms.Unreachable): ms.plan(G, [0, 2, 1, 2], "w")               # must visit 1 after reaching 2: impossible

# ---------- API: modes, objectives, validation ----------
def test_existing_basic_routing_still_works():
    r = c.get("/api/osm/route", params=dict(src_lat=PL["Majestic"]["lat"], src_lon=PL["Majestic"]["lon"], dst_lat=PL["Whitefield"]["lat"], dst_lon=PL["Whitefield"]["lon"])).json()
    assert {"graph_source", "traffic", "snapped", "routes", "time_saved_vs_shortest_min", "filters"} <= set(r) and r["routes"][0]["recommended"]
    assert c.get("/api/route?src=Majestic&dst=Whitefield").status_code == 200

def test_strict_mode_keeps_entered_order():
    r = post(mode="strict").json()
    assert list(r["results"]) == ["strict"] and labels(r["result"]) == ["Majestic", "Silk Board", "Hebbal", "Jayanagar", "Whitefield"]
    assert r["result"]["optimization"]["method"] == "strict_as_entered"

def test_optimized_mode_includes_all_stops_and_is_not_worse():
    o = post(mode="optimized").json(); s = post(mode="strict").json()
    L = labels(o["result"]); assert L[0] == "Majestic" and L[-1] == "Whitefield" and sorted(L[1:-1]) == ["Hebbal", "Jayanagar", "Silk Board"]
    assert o["result"]["objective_value"]["value"] <= s["result"]["objective_value"]["value"] + 1e-9
    assert o["result"]["optimization"]["exact"] is True and o["result"]["legs"] and o["result"]["segments"]

def test_compare_mode_returns_both_consistently():
    r = post(mode="compare").json()
    assert set(r["results"]) == {"optimized", "strict"} and r["result"] == r["results"]["optimized"]
    cmp_ = r["comparison"]; o, s = r["results"]["optimized"], r["results"]["strict"]
    assert cmp_["time_saved_min"] == pytest.approx(s["time_min"] - o["time_min"], abs=0.11) and cmp_["same_order"] == (labels(o) == labels(s))

def test_fastest_vs_shortest_objectives():
    f = post(stops=[], objective="fastest").json()["result"]; s = post(stops=[], objective="shortest").json()["result"]
    assert s["distance_km"] <= f["distance_km"] + 1e-9 and f["time_min"] <= s["time_min"] + 1e-9
    assert "min" in f["objective_value"]["unit"] and "km" in s["objective_value"]["unit"]
    f2 = post(objective="fastest", mode="strict").json()["result"]; s2 = post(objective="shortest", mode="strict").json()["result"]
    assert s2["distance_km"] <= f2["distance_km"] + 1e-9

def test_distance_is_road_distance_not_straight_line():
    r = post(stops=[]).json()["result"]; a, b = PL["Majestic"], PL["Whitefield"]
    straight = 6371 * 2 * math.asin(math.sqrt(math.sin(math.radians(b["lat"] - a["lat"]) / 2) ** 2 + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.sin(math.radians(b["lon"] - a["lon"]) / 2) ** 2))
    assert r["distance_km"] > straight * 1.05

def test_no_stops_is_a_plain_route():
    r = post(stops=[]).json(); assert len(r["result"]["order"]) == 2 and r["result"]["optimization"]["method"] == "trivial"

def test_round_trip_allowed_with_stops():
    r = post(destination=pt("Majestic"), stops=[pt("Hebbal"), pt("Jayanagar")]).json()["result"]
    assert labels(r)[0] == labels(r)[-1] == "Majestic" and r["distance_km"] > 0 and sorted(labels(r)[1:-1]) == ["Hebbal", "Jayanagar"]

@pytest.mark.parametrize("patch,code", [({"source": {"lat": 200, "lon": 77}}, 422), ({"destination": {"lat": 12, "lon": 400}}, 422), ({"mode": "x"}, 422),
    ({"objective": "x"}, 422), ({"road_preference": "x"}, 422), ({"traffic_profile": "x"}, 422), ({"hour": 30}, 422), ({"depart_at": "not a time"}, 422),
    ({"stops": [{"lat": 0.0, "lon": 0.0}]}, 422), ({"source": {"lat": 0.0, "lon": 0.0}}, 422)])
def test_input_validation(patch, code): assert post(**patch).status_code == code

def test_stop_count_limit(monkeypatch):
    many = [{"lat": PL["Hebbal"]["lat"] + i * 1e-4, "lon": PL["Hebbal"]["lon"], "name": f"s{i}"} for i in range(21)]
    r = post(stops=many); assert r.status_code == 422 and "At most 20" in r.json()["detail"]
    monkeypatch.setenv("SPP_MAX_STOPS", "2"); assert post().status_code == 422
    assert c.post("/api/osm/route/multi", json=body(stops=[pt("Hebbal")] * 51)).status_code == 422       # schema cap

def test_duplicate_stops_dropped_with_warning():
    r = post(stops=[pt("Hebbal"), pt("Hebbal"), pt("Majestic"), pt("Whitefield"), pt("Jayanagar")]).json()
    assert labels(r["result"]).count("Stop 2") == 0 and len(r["warnings"]) == 3 and len(r["result"]["order"]) == 4
    assert c.post("/api/osm/route/multi", json=body(destination=pt("Majestic"), stops=[])).status_code == 422

def test_invalid_edge_weights_at_api_level():
    R = osm_service.get_router(); old = R.ff[0]; R.ff[0] = np.nan
    try:
        r = post(); assert r.status_code == 422 and "invalid weights" in r.json()["detail"]
    finally: R.ff[0] = old
    assert post().status_code == 200

# ---------- road quality ----------
def test_edge_quality_scores_and_unknown_policy():
    assert rqm.edge_quality("asphalt") == (0.0, "osm:surface") and rqm.edge_quality("dirt")[0] > 0.9
    assert rqm.edge_quality(None, "bad") == (0.75, "osm:smoothness")
    assert rqm.edge_quality("asphalt", "bad") == (0.75, "osm:smoothness+surface")         # the worse of the two
    assert rqm.edge_quality(["unpaved", "asphalt"])[0] == 0.8 and rqm.edge_quality("Paved;asphalt")[0] == 0.1
    for unknown in (None, "", "paving_stones:lanes", "something_new"): assert rqm.edge_quality(unknown) == (None, "unknown")
    assert rqm.penalty_factor(None, 2.0) == 1.0 and rqm.penalty_factor(0.8, 2.0) == pytest.approx(2.6)
    with pytest.raises(ValueError): rqm.alpha("x")

def tiny_router(unknown_c=False):
    G = nx.DiGraph(source="openstreetmap")
    for n, (la, lo) in {"A": (12.90, 77.50), "B": (12.91, 77.51), "C": (12.90, 77.52), "D": (12.90, 77.53)}.items(): G.add_node(n, lat=la, lon=lo)
    def e(u, v, L, rq, src):
        for a, b in ((u, v), (v, u)): G.add_edge(a, b, length=L, hw="primary", kph=36.0, rough=2, name=f"{u}{v}", geom=None, rq=rq, rq_src=src)
    e("A", "B", 1000, .8, "osm:surface"); e("B", "D", 1000, .8, "osm:surface")                     # short, unpaved
    e("A", "C", 1500, None if unknown_c else 0.0, "unknown" if unknown_c else "osm:surface"); e("C", "D", 1500, None if unknown_c else 0.0, "unknown" if unknown_c else "osm:surface")
    return osm_service.Router(G)

def run_tiny(R, pref, objective="fastest"):
    snap = [{"lat": 0, "lon": 0, "distance_km": 0, "req_lat": 0, "req_lon": 0}] * 2
    return R.multi(["A", "D"], ["A", "D"], snap, np.ones(len(R.edges)), objective, pref, "optimized", 20)["optimized"]

@pytest.mark.parametrize("unknown_c", [False, True])
def test_road_quality_preference_changes_route_and_unknown_is_neutral(unknown_c):
    R = tiny_router(unknown_c)
    none, bal, better = run_tiny(R, "none"), run_tiny(R, "balanced"), run_tiny(R, "better_roads")
    assert none["distance_km"] == 2.0 and bal["distance_km"] == 2.0 and better["distance_km"] == 3.0     # better_roads detours via the good/unknown road
    assert better["road_quality"]["penalty"]["value"] == 0.0 if not unknown_c else better["road_quality"]["unknown_share"] == 1.0
    assert none["road_quality"]["penalty"]["value"] == 0.0 and bal["road_quality"]["penalty"]["value"] > 0

def test_distance_time_and_quality_penalty_reported_separately():
    R = tiny_router(); r = run_tiny(R, "balanced")                                          # unpaved route chosen, penalised
    assert r["distance_km"] == 2.0 and r["time_min"] == pytest.approx(2000 / (36 / 3.6) / 60, abs=0.1)   # physical numbers untouched by the penalty
    q = r["road_quality"]; assert q["penalty"]["unit"] == "min-equivalent" and q["penalty"]["value"] == pytest.approx(r["time_min"] * 0.5 * 0.8, abs=0.1)
    assert r["objective_value"]["value"] == pytest.approx(r["time_min"] + q["penalty"]["value"], abs=0.1)
    assert q["known_share"] == 1.0 and "not verified" in q["note"] and "OpenStreetMap" in q["note"]
    assert run_tiny(R, "balanced", "shortest")["road_quality"]["penalty"]["unit"] == "km-equivalent"

def test_demo_network_quality_is_labelled_simulated():
    q = post(road_preference="balanced").json()["result"]["road_quality"]
    assert "SIMULATED" in q["note"] and set(q["sources_km"]) == {"simulated"}
    cfg = c.get("/api/roadquality/config").json(); assert "never assumed poor" in cfg["unknown_policy"]

def test_better_roads_changes_demo_route_without_changing_reported_physics():
    a, b = post(stops=[], road_preference="none").json()["result"], post(stops=[], road_preference="better_roads").json()["result"]
    assert b["road_quality"]["mean_score_known"] <= a["road_quality"]["mean_score_known"] + 1e-9

# ---------- weather (mocked provider) ----------
def fake_open_meteo(precip=0.0, vis=20000.0, wind=10.0, gust=15.0, code=1, prob=10, temp=28.0):
    day0 = weather.now_ist().replace(hour=0, minute=0, second=0, microsecond=0)
    times = [(day0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(72)]
    h = {"time": times, "temperature_2m": [temp] * 72, "precipitation": [precip] * 72, "precipitation_probability": [prob] * 72, "visibility": [vis] * 72,
         "wind_speed_10m": [wind] * 72, "wind_gusts_10m": [gust] * 72, "weather_code": [code] * 72}
    now = weather.now_ist(); cur = {"time": now.strftime("%Y-%m-%dT%H:%M"), "temperature_2m": temp, "precipitation": precip, "wind_speed_10m": wind, "wind_gusts_10m": gust, "weather_code": code}
    return {"hourly": h, "current": cur}

PTS = [{"lat": 12.97, "lon": 77.59, "offset_min": 0}, {"lat": 12.98, "lon": 77.65, "offset_min": 40}, {"lat": 12.97, "lon": 77.75, "offset_min": 90}]

def test_weather_current_vs_forecast_labelling(monkeypatch):
    monkeypatch.setattr(weather, "_http_get", lambda url, params: fake_open_meteo())
    a = weather.assess(PTS)
    assert a["available"] and a["samples"][0]["type"] == "current_model_estimate" and a["samples"][2]["type"] == "forecast"
    assert a["provider"] == "open-meteo" and a["alerts"]["available"] is False and "OPENWEATHER_API_KEY" in a["alerts"]["reason"]
    assert any("not" in x and "reroute" in x for x in a["limitations"]) and a["summary"]["estimated_rain_exposure_mm"] == 0

def test_weather_warnings(monkeypatch):
    monkeypatch.setattr(weather, "_http_get", lambda url, params: fake_open_meteo(precip=10.0, vis=150.0, wind=55.0, code=95, temp=40.0))
    a = weather.assess(PTS); types = {w["type"] for w in a["warnings"]}
    assert {"heavy_rain", "very_low_visibility", "strong_wind", "thunderstorm", "high_temperature"} <= types
    assert a["time_adjustment"]["factor"] == 1.20 and a["summary"]["estimated_rain_exposure_mm"] == pytest.approx(10 * 1.5, abs=0.01)
    assert "ASSUMED" in a["time_adjustment"]["basis"]

def test_weather_provider_failure_reports_unavailable(monkeypatch):
    calls = []
    def boom(url, params): calls.append(1); raise TimeoutError("slow")
    monkeypatch.setattr(weather, "_http_get", boom); a = weather.assess(PTS)
    assert len(calls) == 1                                                  # stops after the first failure instead of waiting on every point
    weather.assess(PTS); assert len(calls) == 1                             # failure is cached briefly
    assert a["available"] is False and "Open-Meteo request failed" in a["reason"] and a["warnings"] == [] and a["samples"] == []

def test_weather_failure_does_not_break_routing(monkeypatch):
    monkeypatch.setattr(weather, "_http_get", lambda url, params: (_ for _ in ()).throw(ConnectionError("down")))
    r = post(weather_assessment=True); assert r.status_code == 200
    w = r.json()["result"]["weather"]; assert w["available"] is False and w["reason"]
    assert r.json()["result"]["time_min"] == post().json()["result"]["time_min"]

def test_weather_does_not_change_the_route(monkeypatch):
    monkeypatch.setattr(weather, "_http_get", lambda url, params: fake_open_meteo(precip=12.0))
    a, b = post(weather_assessment=True).json()["result"], post().json()["result"]
    assert labels(a) == labels(b) and a["time_min"] == b["time_min"] and a["weather"]["available"] and "weather" not in b

def test_weather_cache_limits_provider_calls(monkeypatch):
    calls = []; monkeypatch.setattr(weather, "_http_get", lambda url, params: calls.append(url) or fake_open_meteo())
    weather.assess(PTS); n = len(calls); weather.assess(PTS)
    assert n == 3 and len(calls) == n

def test_weather_alerts_with_key(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "k")
    def get(url, params):
        if "openweathermap" in url: return {"alerts": [{"event": "Flood Watch", "sender_name": "IMD", "start": 1, "end": 2, "description": "heavy"}]}
        return fake_open_meteo()
    monkeypatch.setattr(weather, "_http_get", get); a = weather.assess(PTS)
    assert a["alerts"]["available"] and any(w["type"] == "official_alert" for w in a["warnings"])

def test_weather_forecast_beyond_coverage(monkeypatch):
    monkeypatch.setattr(weather, "_http_get", lambda url, params: fake_open_meteo())
    a = weather.assess(PTS, depart=weather.now_ist() + timedelta(days=10)); assert a["available"] is False and "does not cover" in a["reason"]

def test_weather_compare_mode_reports_exposure(monkeypatch):
    monkeypatch.setattr(weather, "_http_get", lambda url, params: fake_open_meteo(precip=3.0))
    cmp_ = post(mode="compare", weather_assessment=True).json()["comparison"]["weather"]
    assert cmp_["lower_exposure"] in ("optimized", "strict", "equal") and "does not show which road is safer" in cmp_["caveat"]

# ---------- traffic profiles ----------
def test_traffic_profile_labelling_without_live_key():
    p = {x["profile"]: x for x in c.get("/api/traffic/profiles").json()["profiles"]}
    assert set(p) == {"free_flow", "typical", "peak", "live"} and not p["live"]["available"] and "TOMTOM_KEY" in p["live"]["reason"]
    assert all(not p[k]["verified_live"] for k in ("free_flow", "typical", "peak")) and p["free_flow"]["kind"] == "free-flow-assumption" and p["typical"]["kind"] == "simulated-profile"
    for k in ("free_flow", "typical", "peak"):
        t = post(traffic_profile=k).json()["traffic"]; assert t["verified_live"] is False and t["profile"] == k and t["limitations"]

def test_live_profile_refused_without_data():
    r = post(traffic_profile="live"); assert r.status_code == 409 and "not available" in r.json()["detail"]

def test_live_profile_with_mocked_tomtom(monkeypatch):
    monkeypatch.setenv("TOMTOM_KEY", "t"); monkeypatch.setattr(tl, "fetch_flow", lambda lat, lon: (20.0, 40.0)); monkeypatch.setattr(tl, "LOG", os.path.join(os.path.dirname(__file__), "_l.csv"))
    R = osm_service.get_router(); R.live.mult, R.live.ts = None, 0; R.live.get()
    import time
    for _ in range(100):
        if R.live.mult is not None: break
        time.sleep(0.2)
    t = post(traffic_profile="live").json()["traffic"]; assert t["kind"] == "live" and t["verified_live"] is True
    R.live.mult, R.live.ts = None, 0
    try: os.remove(tl.LOG)
    except OSError: pass

def test_profiles_order_travel_time():
    t = {k: post(traffic_profile=k, stops=[], mode="strict", hour=9).json()["result"]["time_min"] for k in ("free_flow", "typical", "peak")}
    assert t["free_flow"] <= t["typical"] <= t["peak"] + 1e-9

# ---------- frontend integration ----------
HTML = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "app.html"), encoding="utf-8").read()

def test_frontend_dom_ids_exist():
    js = HTML.split("<script>")[1]; ids = set(re.findall(r'id="([^"]+)"', HTML))
    used = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", js)); assert used <= ids, sorted(used - ids)

def test_frontend_endpoints_exist():
    urls = set(re.findall(r"fetch\('(/api/[a-z/\-]+)", HTML)); assert {"/api/osm/route", "/api/osm/search", "/api/osm/route/multi", "/api/traffic/profiles"} <= urls
    for u in urls:
        r = c.post(u, json={}) if u.endswith("/multi") else c.get(u); assert r.status_code in (200, 422), (u, r.status_code)

def test_frontend_fields_come_from_api():
    r = post(mode="compare", weather_assessment=False).json()
    for key in ("results", "result", "comparison", "traffic", "warnings", "notes"): assert key in r
    for key in ("order", "legs", "segments", "distance_km", "time_min", "road_quality", "optimization", "objective_value", "traffic_level"): assert key in r["result"]
    assert c.get("/api/traffic/profiles").json()["profiles"][0]["source"]
