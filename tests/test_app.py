"""Run:  python -m pytest -q   (from the project folder)"""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
import app as appmod, osm_service, traffic_live as tl

c = TestClient(appmod.app)
A = dict(src_lat=12.9767, src_lon=77.5713, dst_lat=12.9698, dst_lon=77.7500)      # Majestic -> Whitefield

def test_pages_and_static():
    for u in ["/", "/classic", "/osm", "/static/vendor/leaflet.js", "/docs"]:
        assert c.get(u).status_code == 200, u

def test_classic_api():
    assert c.get("/api/health").json()["status"] == "ok"
    for e in ("ml", "rule"):
        d = c.get(f"/api/route?src=Majestic&dst=Whitefield&engine={e}").json()
        assert d["recommended"]["time_min"] > 0 and len(d["routes"]) >= 2
    assert len(c.get("/api/best-time?src=Majestic&dst=Hebbal").json()["times_min"]) == 24
    assert len(c.get("/api/hotspots?limit=3").json()) == 3
    assert c.get("/api/route?src=Nowhere&dst=Hebbal").status_code == 404
    assert c.get("/api/route?src=Majestic&dst=Majestic").status_code == 422
    assert len(c.get("/api/predictions").json()["multipliers"]) == 24

def test_map_status_and_places():
    s = c.get("/api/osm/status").json()
    assert s["graph_loaded"] and s["graph_source"] in ("demo", "openstreetmap")
    P = c.get("/api/osm/places").json(); assert len(P) >= 40 and sum(p["kind"] == "junction" for p in P) == 21

@pytest.mark.parametrize("engine", ["profile", "ml"])
def test_map_route(engine):
    d = c.get("/api/osm/route", params=dict(A, engine=engine, hour=9)).json()
    r = d["routes"]
    assert d["traffic"]["verified_live"] is False and d["traffic"]["kind"].startswith(("simulated", "ml"))
    assert r[0]["recommended"] and len(r) >= 2 and r[0]["cost"] <= r[1]["cost"]
    assert r[0]["traffic_level"] in ("Free", "Moderate", "Heavy", "Severe") and r[0]["segments"] and r[0]["steps"]
    assert sum(1 for x in r if x["is_shortest_distance"]) == 1
    assert d["snapped"]["src"]["name"] == "Majestic"

def test_traffic_varies_with_conditions():
    t = lambda **k: c.get("/api/osm/route", params=dict(A, **k)).json()["routes"][0]["time_min"]
    assert t(hour=9) > t(hour=3)                       # rush hour is slower than 3 AM
    assert t(hour=9, weather="rain") >= t(hour=9)
    assert t(hour=9, day="weekend") <= t(hour=9)

def test_best_time_and_validation():
    d = c.get("/api/osm/best-time", params=dict(A, engine="profile", hour=9)).json()
    assert len(d["times_min"]) == 24 and d["times_min"][d["best_hour"]] == min(d["times_min"])
    assert c.get("/api/osm/best-time", params=dict(A, engine="live")).status_code == 400
    assert c.get("/api/osm/route", params=dict(A, dst_lat=A["src_lat"], dst_lon=A["src_lon"])).status_code == 422
    assert c.get("/api/osm/route", params=dict(A, src_lat=40, src_lon=-100)).status_code == 422

def test_live_is_never_faked(monkeypatch):
    monkeypatch.delenv("TOMTOM_KEY", raising=False)
    d = c.get("/api/osm/route", params=dict(A, engine="live")).json()
    assert d["traffic"]["kind"] == "simulated-fallback" and d["traffic"]["verified_live"] is False and "TOMTOM_KEY" in d["traffic"]["note"]

def test_live_path_with_mocked_tomtom(monkeypatch):
    monkeypatch.setenv("TOMTOM_KEY", "test"); monkeypatch.setattr(tl, "fetch_flow", lambda lat, lon: (20.0, 40.0))
    monkeypatch.setattr(tl, "LOG", os.path.join(os.path.dirname(__file__), "_live_test.csv"))
    R = osm_service.get_router(); R.live.mult, R.live.ts = None, 0
    R.live.get()
    for _ in range(100):
        if R.live.mult is not None: break
        import time; time.sleep(0.2)
    d = c.get("/api/osm/route", params=dict(A, engine="live")).json()
    assert d["traffic"]["kind"] == "live" and d["traffic"]["verified_live"] is True
    R.live.mult, R.live.ts = None, 0
    try: os.remove(tl.LOG)
    except OSError: pass

def test_osmnx_conversion(tmp_path):
    pytest.importorskip("osmnx"); import osm_ingest; ox = osm_ingest.configure()
    xml = """<?xml version='1.0' encoding='UTF-8'?><osm version='0.6'>
    <node id='1' lat='12.9700' lon='77.5900'/><node id='2' lat='12.9700' lon='77.6000'/><node id='3' lat='12.9800' lon='77.6000'/><node id='4' lat='12.9800' lon='77.5900'/>
    <way id='10'><nd ref='1'/><nd ref='2'/><tag k='highway' v='primary'/><tag k='maxspeed' v='50'/><tag k='surface' v='asphalt'/><tag k='name' v='Test Road'/></way>
    <way id='11'><nd ref='2'/><nd ref='3'/><tag k='highway' v='tertiary'/><tag k='surface' v='unpaved'/></way>
    <way id='12'><nd ref='3'/><nd ref='4'/><tag k='highway' v='secondary'/></way>
    <way id='13'><nd ref='4'/><nd ref='1'/><tag k='highway' v='secondary'/></way></osm>"""
    p = tmp_path / "t.osm"; p.write_text(xml)
    G = osm_ingest.from_osmnx(ox.graph_from_xml(str(p), simplify=False))
    assert G.graph["source"] == "openstreetmap" and G.number_of_edges() >= 4
    E = {d["name"]: d for _, _, d in G.edges(data=True)}
    assert E["Test Road"]["kph"] == 50 and E["Test Road"]["rough"] == 1 and E["Test Road"]["hw"] == "primary"
    assert any(d["rough"] == 4 and d["kph"] == 30 for d in E.values())            # unpaved tertiary -> rough, default speed
    R = osm_service.Router(G); n1, _, _ = R.nearest(12.9700, 77.5900); n3, _, _ = R.nearest(12.9800, 77.6000)
    m, kind, _ = R.multipliers("profile", 9, "weekday", "dry")
    assert R.route(n1, n3, m)[0]["distance_km"] > 0 and kind == "simulated-profile"


def test_search():
    r = c.get("/api/osm/search", params=dict(q="whitef", online=False)).json()
    assert r["results"][0]["name"] == "Whitefield" and r["online"] is False
    assert any(p["name"] == "Cubbon Park" for p in c.get("/api/osm/search", params=dict(q="cubbon", online=False)).json()["results"])
    assert len(c.get("/api/osm/search", params=dict(q="", online=False)).json()["results"]) == 8
    assert c.get("/api/osm/search", params=dict(q="zzzzqq", online=False)).json()["results"] == []

def test_search_online_failure_is_graceful(monkeypatch):
    import osm_api
    def boom(q, n): raise RuntimeError("offline")
    monkeypatch.setattr(osm_api, "nominatim", boom)
    r = c.get("/api/osm/search", params=dict(q="some new cafe")).json()
    assert r["online"] is False and "unavailable" in r["note"]

def test_search_online_results_are_merged(monkeypatch):
    import osm_api
    monkeypatch.setattr(osm_api, "nominatim", lambda q, n: [{"name": "Test Cafe, Indiranagar", "lat": 12.97, "lon": 77.64, "kind": "osm-search", "area": ""}])
    r = c.get("/api/osm/search", params=dict(q="test cafe")).json()
    assert r["online"] is True and r["results"][-1]["kind"] == "osm-search"

def test_filters_and_hotspots():
    base = c.get("/api/osm/route", params=dict(A, hour=9)).json()
    assert any(s["road"] == "Whitefield Main Rd / ITPL Rd" for s in base["routes"][0]["steps"])
    f = c.get("/api/osm/route", params=dict(A, hour=9, avoid_poor="true")).json()          # avoids the Poor Whitefield Main Rd
    assert f["filters"]["avoid_poor"] and not any(s["road"] == "Whitefield Main Rd / ITPL Rd" for r in f["routes"] for s in r["steps"])
    assert f["routes"][0]["time_min"] >= base["routes"][0]["time_min"]
    r = c.get("/api/osm/route", params=dict(A, avoid_waterlogged="true"))                  # every road into Whitefield is waterlogging-prone here
    assert r.status_code == 404 and "filters" in r.json()["detail"]
    j = c.get("/api/osm/route", params=dict(src_lat=12.9767, src_lon=77.5713, dst_lat=12.9250, dst_lon=77.5938, avoid_waterlogged="true")).json()
    assert j["filters"]["avoid_waterlogged"] and j["routes"]
    assert c.get("/api/route?src=Majestic&dst=Whitefield&avoid_waterlogged=true").status_code == 404
    h = c.get("/api/osm/hotspots", params=dict(hour=18, limit=3)).json()
    assert len(h["roads"]) == 3 and h["roads"][0]["slowdown_x"] >= h["roads"][1]["slowdown_x"] and h["traffic"]["verified_live"] is False

def test_provenance_on_every_traffic_response():
    assert c.get("/api/route?src=Majestic&dst=Whitefield").json()["traffic"]["verified_live"] is False
    assert c.get("/api/predictions?engine=rule").json()["traffic_kind"] == "simulated-profile"
    assert c.get("/api/hotspots").json()[0]["traffic_kind"] == "simulated-ml"
    assert c.get("/api/best-time?src=Majestic&dst=Hebbal").json()["traffic_kind"] == "simulated-ml"

@pytest.mark.parametrize("classic_engine,map_engine", [("rule", "profile"), ("ml", "ml")])
def test_classic_and_map_engines_agree(classic_engine, map_engine):
    """Both APIs use one routing engine, so every pair must match."""
    pl = {p["name"]: p for p in c.get("/api/osm/places").json() if p["kind"] == "junction"}
    names = list(pl)
    for s, t in [(a, b) for a in names for b in names if a != b][::7]:
        a = c.get(f"/api/route?src={s}&dst={t}&engine={classic_engine}&hour=9").json()["recommended"]
        b = c.get("/api/osm/route", params=dict(src_lat=pl[s]["lat"], src_lon=pl[s]["lon"], dst_lat=pl[t]["lat"], dst_lon=pl[t]["lon"], engine=map_engine, hour=9)).json()["routes"][0]
        assert a["path"] == b["path"] and a["time_min"] == b["time_min"] and a["distance_km"] == b["distance_km"], (s, t)
