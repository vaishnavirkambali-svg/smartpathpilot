"""CityPulse: reports, recurrence, priority, risk, ML, and route-cost integration. Run: python -m pytest -q"""
import dataclasses, math, os, re, sqlite3
from datetime import timedelta
import networkx as nx, numpy as np, pytest
from fastapi.testclient import TestClient
import app as appmod, osm_service
from citypulse import analysis, api as cpapi, demo, geo, ml, risk, routing, store
from citypulse.config import get_config

c = TestClient(appmod.app)
NOW = store.utcnow()
PL = {p["name"]: p for p in c.get("/api/osm/places").json() if p["kind"] == "junction"}
OK = {"issue_type": "pothole", "lat": PL["Majestic"]["lat"], "lon": PL["Majestic"]["lon"], "description": "deep pothole", "severity": 3}
def post(**kw): return c.post("/api/citypulse/reports", json={**OK, **kw})

def rep(i, t="pothole", days=1.0, sev=3, status="verified", reporter=None, lat=12.91, lon=77.51, source="community_report", created_days=None, reviewed_days=None):
    obs = NOW - timedelta(days=days); cr = NOW - timedelta(days=created_days if created_days is not None else days)
    rv = (NOW - timedelta(days=reviewed_days if reviewed_days is not None else days)) if status != "unverified" else None
    return {"id": i, "issue_type": t, "lat": lat, "lon": lon, "severity": sev, "observed_at": obs, "created_at": cr, "status": status, "reviewed_at": rv, "source": source, "reporter": reporter or f"r{i}"}
snap = lambda reps, key="1-2": {r["id"]: (key, (1, 2), 0.0) for r in reps}

# ---------- validation and persistence ----------
def test_valid_report_is_stored_as_unverified_community_data():
    r = post(); assert r.status_code == 201; j = r.json()
    rep_ = j["report"]; assert rep_["status"] == "unverified" and rep_["source"] == "community_report" and rep_["simulated"] is False and "unverified" in j["message"]
    assert "reporter_hash" not in rep_ and "reporter" not in rep_ and j["matched_segment"]

@pytest.mark.parametrize("patch,field", [({"issue_type": "ufo"}, "issue_type"), ({"issue_type": None}, "issue_type"), ({"lat": 40.0}, "location"), ({"lon": 0}, "location"),
    ({"lat": "abc"}, "location"), ({"lat": True}, "location"), ({"lat": 1e308}, "location"), ({"severity": 9}, "severity"), ({"severity": 0}, "severity"), ({"severity": "3"}, "severity"),
    ({"severity": 2.5}, "severity"), ({"observed_at": "not a date"}, "observed_at"), ({"observed_at": (NOW + timedelta(days=2)).isoformat()}, "observed_at"),
    ({"observed_at": (NOW - timedelta(days=4000)).isoformat()}, "observed_at")])
def test_report_validation_errors(patch, field):
    r = post(**patch)
    assert r.status_code == 422 and field in r.json()["errors"], r.text
    assert c.get("/api/citypulse/summary").json()["total_reports"] == 0                      # nothing stored on failure

def test_non_finite_numbers_in_raw_json_are_rejected():
    for bad in ("NaN", "Infinity", "-Infinity"):
        r = c.post("/api/citypulse/reports", content='{"issue_type":"pothole","lat":%s,"lon":77.59}' % bad, headers={"Content-Type": "application/json"})
        assert r.status_code == 422 and "location" in r.json()["errors"]
    assert c.get("/api/citypulse/summary").json()["total_reports"] == 0

def test_severity_optional_description_sanitised_and_capped():
    r = post(severity=None, description="a\x00b\x07" + "x" * 700).json()["report"]
    assert r["severity"] is None and len(r["description"]) == 500 and "\x00" not in r["description"]

def test_persistence_survives_reconnect_and_stores_no_raw_ip():
    rid = post().json()["report"]["id"]
    assert store.get(rid)["issue_type"] == "pothole" and os.path.exists(store.db_path())
    con = sqlite3.connect(store.db_path()); rows = con.execute("SELECT * FROM reports").fetchall(); con.close()
    assert len(rows) == 1 and "testclient" not in " ".join(str(x) for x in rows[0]) and re.fullmatch(r"[0-9a-f]{16}", rows[0][-1])
    assert c.get(f"/api/citypulse/reports/{rid}").json()["id"] == rid and c.get("/api/citypulse/reports/999").status_code == 404

def test_duplicate_reports_by_same_reporter_rejected():
    assert post().status_code == 201
    d = post(); assert d.status_code == 409 and "duplicate" in d.json()["errors"]
    assert post(lat=OK["lat"] + 0.001).status_code == 201                                     # ~110 m away: a different spot
    assert post(issue_type="waterlogging").status_code == 201                                 # different issue type
    assert post(observed_at=(NOW - timedelta(days=3)).isoformat()).status_code == 201         # observed on another day
    assert len(store.list_reports()) == 4

def test_rate_limit(monkeypatch):
    monkeypatch.setattr(get_config(), "max_reports_per_hour", 3)
    codes = [post(lat=OK["lat"] + 0.002 * i).status_code for i in range(5)]
    assert codes == [201, 201, 201, 429, 429]

def test_moderation_requires_token(monkeypatch):
    rid = post().json()["report"]["id"]; url = f"/api/citypulse/reports/{rid}/status"
    assert c.patch(url, json={"status": "verified"}).status_code == 403                       # no token configured: moderation disabled
    monkeypatch.setenv("CITYPULSE_ADMIN_TOKEN", "tok")
    assert c.patch(url, json={"status": "verified"}, headers={"X-Admin-Token": "bad"}).status_code == 403
    assert c.patch(url, json={"status": "bogus"}, headers={"X-Admin-Token": "tok"}).status_code == 422
    assert c.patch("/api/citypulse/reports/999/status", json={"status": "verified"}, headers={"X-Admin-Token": "tok"}).status_code == 404
    j = c.patch(url, json={"status": "verified"}, headers={"X-Admin-Token": "tok"}).json(); assert j["status"] == "verified" and j["reviewed_at"]

def test_unmatched_report_is_flagged_and_excluded_from_segments():
    r = post(lat=13.28, lon=77.35).json(); assert r["matched_segment"] is None and r["warnings"] and r["report"]["segment"] is None
    s = c.get("/api/citypulse/summary").json(); assert s["total_reports"] == 1 and s["unmatched_reports"] == 1 and c.get("/api/citypulse/segments").json()["segments"] == []

# ---------- recurrence ----------
CFG = get_config()
def one(reps): a, _ = analysis.analyse(reps, snap(reps), {"1-2": {"road": "X", "hw": "primary"}}, NOW, CFG); return a[0] if a else None

def test_repeats_within_one_episode_count_once():
    s = one([rep(1, days=5, reporter="a"), rep(2, days=4.9, reporter="a"), rep(3, days=4.8, reporter="a")])
    p = s["issues"]["pothole"]; assert p["raw_reports"] == 3 and p["episodes"] == 1 and p["independent_observations"] == 1 and not s["recurring"]

def test_independent_reporters_in_one_episode_corroborate_but_are_not_recurrence():
    s = one([rep(1, days=5, reporter="a"), rep(2, days=4.9, reporter="b")])
    assert s["issues"]["pothole"]["independent_observations"] == 2 and s["issues"]["pothole"]["episodes"] == 1 and not s["recurring"]

def test_recurring_needs_separate_episodes_and_distinct_reporters():
    two_people = one([rep(1, days=40, reporter="a"), rep(2, days=5, reporter="b")]); assert two_people["recurring"][0]["episodes"] == 2
    one_person = one([rep(1, days=40, reporter="a"), rep(2, days=5, reporter="a")])
    assert one_person["issues"]["pothole"]["episodes"] == 2 and one_person["recurring"] == []             # one reporter alone cannot create recurrence

def test_rejected_and_old_reports_ignored_and_types_are_separate():
    assert one([rep(1, days=40, status="rejected"), rep(2, days=5, status="rejected")]) is None
    assert one([rep(1, days=400, reporter="a"), rep(2, days=5, reporter="b")])["recurring"] == []         # first is outside the 180-day lookback
    mixed = one([rep(1, "pothole", days=40, reporter="a"), rep(2, "waterlogging", days=5, reporter="b")]); assert mixed["recurring"] == []

def test_recurrence_thresholds_are_configurable():
    reps = [rep(1, days=40, reporter="a"), rep(2, days=20, reporter="b"), rep(3, days=2, reporter="c")]
    strict = dataclasses.replace(CFG, min_episodes=4)
    a, _ = analysis.analyse(reps, snap(reps), {}, NOW, strict); assert a[0]["recurring"] == []
    assert len(one(reps)["recurring"]) == 1
    long_gap = dataclasses.replace(CFG, episode_gap_hours=24 * 60)                                       # everything now one episode
    a, _ = analysis.analyse(reps, snap(reps), {}, NOW, long_gap); assert a[0]["issues"]["pothole"]["episodes"] == 1 and "separate episodes" in analysis.recurrence_rule(CFG)

# ---------- priority ranking ----------
def test_priority_score_matches_hand_calculation_and_factors_sum_to_score():
    reps = [rep(1, days=30, sev=4, reporter="a", status="verified"), rep(2, days=0, sev=2, reporter="b", status="unverified")]
    p = one(reps)["priority"]; f = {x["name"]: x for x in p["factors"]}
    assert f["severity"]["value"] == 0.8 and f["recency"]["value"] == pytest.approx(1.0) and f["recurrence"]["value"] == pytest.approx(1 / 3, abs=1e-3)
    assert f["confidence"]["value"] == pytest.approx(0.5 * 0.5 + 0.5 * (2 / 3), abs=1e-3) and f["impact"]["value"] == 0.85
    assert p["score"] == pytest.approx(72.17, abs=0.2) and p["score"] == pytest.approx(sum(x["points"] for x in p["factors"]), abs=1e-9)
    assert sum(x["weight"] for x in p["factors"]) == pytest.approx(1.0, abs=0.003) and "not an official" in p["label"]

def test_missing_severity_and_missing_class_are_omitted_not_invented():
    reps = [rep(1, days=30, sev=None, reporter="a"), rep(2, days=1, sev=None, reporter="b")]
    a, _ = analysis.analyse(reps, snap(reps), {"1-2": {"road": "X"}}, NOW, CFG); p = a[0]["priority"]
    assert {"severity", "impact"} == set(p["not_available"]) and "severity" not in {x["name"] for x in p["factors"]}
    assert sum(x["weight"] for x in p["factors"]) == pytest.approx(1.0, abs=0.003) and 0 < p["score"] <= 100

def test_ranking_order_monotonic_in_severity_and_recency():
    A = [rep(1, days=2, sev=5, reporter="a"), rep(2, days=1, sev=5, reporter="b")]; B = [rep(3, days=2, sev=1, reporter="a"), rep(4, days=1, sev=1, reporter="b")]
    C = [rep(5, days=150, sev=5, reporter="a"), rep(6, days=140, sev=5, reporter="b")]
    reps = A + B + C; sn = {**snap(A, "1-2"), **snap(B, "3-4"), **snap(C, "5-6")}
    segs, _ = analysis.analyse(reps, sn, {}, NOW, CFG); keys = [s["key"] for s in segs]
    assert keys == ["1-2", "5-6", "3-4"] or keys == ["1-2", "3-4", "5-6"]; assert segs[0]["key"] == "1-2" and [s["rank"] for s in segs] == [1, 2, 3]
    sc = {s["key"]: s["priority"]["score"] for s in segs}; assert sc["1-2"] > sc["3-4"] and sc["1-2"] > sc["5-6"]      # more severe and more recent ranks higher

# ---------- baseline risk ----------
def test_baseline_risk_formula():
    cfg = CFG; none = risk.baseline([], NOW, cfg); assert none["index"] == pytest.approx(1 - math.exp(-(0.1 / 360) * 14), abs=1e-3) and none["episodes_in_lookback"] == 0
    more = risk.baseline([rep(1, days=40, reporter="a"), rep(2, days=5, reporter="b")], NOW, cfg)
    assert more["episodes_in_lookback"] == 2 and more["index"] > none["index"] and more["index"] == pytest.approx(1 - math.exp(-(2.1 / 360) * 14), abs=1e-3)
    assert "Not a validated probability" in more["caveat"] and more["kind"] == "baseline-heuristic" and len(more["explanation"]) == 3

# ---------- geometry ----------
def tiny_router(source="openstreetmap"):
    G = nx.DiGraph(source=source); xy = {"A": (12.90, 77.50), "B": (12.92, 77.52), "C": (12.88, 77.52), "D": (12.90, 77.54)}
    for n, (la, lo) in xy.items(): G.add_node(n, lat=la, lon=lo)
    for u, v, L in (("A", "B", 1000), ("B", "D", 1000), ("A", "C", 1050), ("C", "D", 1050)):
        for a, b in ((u, v), (v, u)): G.add_edge(a, b, length=L, hw="primary", kph=36.0, rough=1, name=f"{u}{v} road", geom=None, rq=0.2, rq_src="osm:surface")
    return osm_service.Router(G)

def test_snapping_to_segments():
    R = tiny_router(); s = geo.nearest_segment(R, 12.91, 77.51)                                  # midpoint of A-B
    assert s[0] == "A-B" and s[2] < 0.01 and geo.seg_key("B", "A") == "A-B"
    assert geo.nearest_segment(R, 12.91, 77.5101)[0] == "A-B" and geo.nearest_segment(R, 12.95, 77.60) is None   # far from every road

# ---------- CityPulse -> route costs ----------
def run_route(R, ctx=None):
    mult, _, _ = R.multipliers("profile", 9, "weekday", "dry")
    return R.route("A", "D", mult, extra=ctx.factors if ctx is not None else None), mult
def via(route): return "B" if "AB" in " ".join(s["road"] for s in route["steps"]) or "BD" in " ".join(s["road"] for s in route["steps"]) else "C"

def test_verified_severe_issue_moves_the_recommendation():
    R = tiny_router(); base, _ = run_route(R); assert via(base[0]) == "B"
    ctx = routing.build(R, NOW, CFG, reports=[rep(1, "obstruction", days=1, sev=5, status="verified")])
    assert ctx.n_verified == 1 and ctx.factors.max() == pytest.approx(1 + 0.25 * 5)
    with_c, _ = run_route(R, ctx); assert via(with_c[0]) == "C" and with_c[0]["cost"] >= base[0]["cost"] - 1e-9
    note = ctx.annotate(["A", "B", "D"]); assert note["cost_factor_applied"] and note["verified_issues"][0]["cost_factor"] == 2.25 and note["max_cost_factor"] == 2.25

def test_physical_time_and_distance_are_unchanged_by_conditions():
    R = tiny_router(); base, _ = run_route(R); ctx = routing.build(R, NOW, CFG, reports=[rep(1, "obstruction", sev=5)]); wc, _ = run_route(R, ctx)
    b = next(r for r in base if via(r) == "B"); w = next(r for r in wc if via(r) == "B")
    assert (b["time_min"], b["distance_km"]) == (w["time_min"], w["distance_km"]) and w["cost"] > b["cost"]

@pytest.mark.parametrize("r", [rep(1, "obstruction", sev=5, status="unverified"), rep(1, "obstruction", sev=5, status="rejected"), rep(1, "waterlogging", days=10, sev=5), rep(1, "pothole", days=200, sev=5),
                               rep(1, "obstruction", days=-2, sev=5)])
def test_unverified_rejected_expired_and_future_reports_never_change_a_route(r):
    R = tiny_router(); ctx = routing.build(R, NOW, CFG, reports=[r]); assert ctx.n_verified == 0 and np.all(ctx.factors == 1.0)
    assert via(run_route(R, ctx)[0][0]) == "B"

def test_unverified_reports_are_informational_only():
    R = tiny_router(); ctx = routing.build(R, NOW, CFG, reports=[rep(1, "obstruction", sev=5, status="unverified")])
    a = ctx.annotate(["A", "B", "D"]); assert a["verified_issues"] == [] and len(a["unverified_nearby"]) == 1 and not a["cost_factor_applied"] and ctx.n_unverified == 1

def test_penalty_is_capped_and_never_blocks_a_road():
    R = tiny_router(); ctx = routing.build(R, NOW, CFG, reports=[rep(i, "obstruction", sev=5, reporter=f"x{i}") for i in range(1, 31)])
    assert ctx.factors.max() == CFG.max_cost_factor and np.all(np.isfinite(ctx.factors))
    mult, _, _ = R.multipliers("profile", 9, "weekday", "dry"); assert R.route("A", "B", mult, extra=ctx.factors)                  # still routable

def test_missing_severity_uses_documented_default_for_verified_only():
    R = tiny_router(); ctx = routing.build(R, NOW, CFG, reports=[rep(1, "pothole", sev=None)]); assert ctx.factors.max() == pytest.approx(1 + 0.05 * CFG.default_routing_severity)

def test_predicted_risk_never_used_for_routing(monkeypatch):
    R = tiny_router(); ctx = routing.build(R, NOW, CFG, reports=[]); assert ctx.predicted_used is False and any("Predicted risk is not used" in n for n in ctx.notes)
    monkeypatch.setattr(CFG, "use_predicted_risk_in_routing", True)
    assert ml.usable_in_routing(None, CFG)[0] is False
    assert ml.usable_in_routing({"trained_on_simulated": True, "validated": False, "validation_notes": []}, CFG)[0] is False
    assert ml.usable_in_routing({"trained_on_simulated": False, "validated": False, "validation_notes": ["x"]}, CFG)[0] is False
    assert ml.usable_in_routing({"trained_on_simulated": False, "validated": True, "validation_notes": []}, CFG)[0] is True
    assert routing.build(R, NOW, CFG, reports=[]).predicted_used is False                  # policy would allow it, but applying predictions is not implemented

def test_routing_works_when_citypulse_data_is_unavailable(monkeypatch):
    R = tiny_router(); monkeypatch.setattr(store, "for_analysis", lambda: (_ for _ in ()).throw(sqlite3.OperationalError("db gone")))
    ctx = routing.build(R, NOW, CFG); assert ctx.available is False and np.all(ctx.factors == 1.0) and "unavailable" in ctx.summary()["data_mode"]
    r = c.get("/api/osm/route", params=dict(src_lat=PL["Majestic"]["lat"], src_lon=PL["Majestic"]["lon"], dst_lat=PL["Whitefield"]["lat"], dst_lon=PL["Whitefield"]["lon"], use_road_conditions="true"))
    assert r.status_code == 200 and r.json()["road_conditions"]["available"] is False and r.json()["routes"]

def route_api(**kw):
    return c.get("/api/osm/route", params=dict(src_lat=PL["Majestic"]["lat"], src_lon=PL["Majestic"]["lon"], dst_lat=PL["Whitefield"]["lat"], dst_lon=PL["Whitefield"]["lon"], **kw)).json()

def test_default_route_api_is_unchanged_without_the_option():
    d = route_api(); assert d["road_conditions"] == {"enabled": False} and all("road_conditions" not in r and "_nodes" not in r for r in d["routes"])

def test_route_api_with_simulated_conditions_is_labelled_and_consistent():
    c.post("/api/citypulse/demo/seed"); a, b = route_api(), route_api(use_road_conditions="true"); rc = b["road_conditions"]
    assert rc["enabled"] and rc["available"] and "simulated" in rc["data_mode"] and rc["verified_issues_used"] > 0 and rc["predicted_risk_used"] is False
    assert any("not a real-time hazard feed" in n for n in rc["notes"]) and all("_nodes" not in r and "road_conditions" in r for r in b["routes"])
    assert any(i["simulated"] for r in b["routes"] for i in r["road_conditions"]["verified_issues"] + r["road_conditions"]["unverified_nearby"])
    same = [x for x in b["routes"] if x["distance_km"] == a["routes"][0]["distance_km"] and x["traffic_level"] == a["routes"][0]["traffic_level"]]
    if same: assert same[0]["time_min"] == a["routes"][0]["time_min"]                          # displayed time is not altered by the penalty
    assert b["routes"][0]["cost"] >= a["routes"][0]["cost"] - 1e-6

# ---------- API: summary, demo data, ranking, risk ----------
def test_empty_summary_has_no_fabricated_numbers():
    s = c.get("/api/citypulse/summary").json()
    assert s["data_mode"] == "empty" and s["total_reports"] == 0 and s["verified_reports"] == 0 and s["recurring_locations"] == 0 and s["high_priority_segments"] == 0 and s["predicted_risks"]["available"] is False
    assert c.get("/api/citypulse/priority").json()["ranking"] == [] and c.get("/api/citypulse/risk").json()["risks"] == []

def test_demo_data_is_labelled_simulated_everywhere_and_removable():
    assert post().status_code == 201
    j = c.post("/api/citypulse/demo/seed").json(); assert j["simulated"] and j["source"] == "demo_seed" and j["inserted"] > 500
    assert c.post("/api/citypulse/demo/seed").status_code == 409 and c.post("/api/citypulse/demo/seed?replace=true").status_code == 200
    s = c.get("/api/citypulse/summary").json()
    assert s["data_mode"] == "mixed" and s["demo_data_present"] and any("SIMULATED" in n for n in s["notes"]) and sum(s["by_status"].values()) == s["total_reports"]
    assert s["by_source"]["community_report"] == 1 and s["verified_reports"] == s["by_status"]["verified"]
    rs = c.get("/api/citypulse/reports?source=demo_seed&limit=50").json()["reports"]; assert rs and all(r["simulated"] for r in rs)
    assert all(p["simulated"] for p in c.get("/api/citypulse/priority?limit=5").json()["ranking"] if p["simulated"] or True) and c.delete("/api/citypulse/demo").json()["deleted"] == j["inserted"]
    s2 = c.get("/api/citypulse/summary").json(); assert s2["data_mode"] == "real" and s2["total_reports"] == 1                    # real reports untouched

def test_priority_endpoint_ranking_and_transparency():
    c.post("/api/citypulse/demo/seed"); j = c.get("/api/citypulse/priority?limit=10").json(); r = j["ranking"]
    assert len(r) == 10 and [x["priority"]["score"] for x in r] == sorted((x["priority"]["score"] for x in r), reverse=True) and [x["rank"] for x in r] == list(range(1, 11))
    for x in r: assert abs(sum(f["points"] for f in x["priority"]["factors"]) - x["priority"]["score"]) < 1e-6 and 0 <= x["priority"]["score"] <= 100
    assert "Not an official" in j["disclaimer"]
    cfg = c.get("/api/citypulse/config").json(); assert "separate episodes" in cfg["recurrence_rule"] and cfg["config"]["min_episodes"] == 2
    rec = c.get("/api/citypulse/segments?recurring_only=true").json()["segments"]; assert rec and all(s["recurring"] for s in rec)

def test_risk_endpoint_labels_baseline_vs_ml(monkeypatch):
    c.post("/api/citypulse/demo/seed"); r = c.get("/api/citypulse/risk?limit=5").json(); assert r["ml_available"] is False
    assert all(x["ml"] is None and x["baseline"]["kind"] == "baseline-heuristic" and "Not a validated" in x["baseline"]["caveat"] for x in r["risks"])
    assert c.get("/api/citypulse/ml/status").json()["trained"] is False and "No model" in c.get("/api/citypulse/ml/status").json()["reason"]
    assert c.post("/api/citypulse/ml/train?include_demo=true").status_code == 403
    monkeypatch.setenv("CITYPULSE_ADMIN_TOKEN", "tok"); H = {"X-Admin-Token": "tok"}
    real_only = c.post("/api/citypulse/ml/train", headers=H).json(); assert real_only["status"] == "insufficient_data" and c.get("/api/citypulse/ml/status").json()["trained"] is False
    t = c.post("/api/citypulse/ml/train?include_demo=true", headers=H).json(); assert t["status"] == "demo_simulated" and t["validated"] is False and t["trained_on_simulated"] is True
    st_ = c.get("/api/citypulse/ml/status").json(); assert st_["trained"] and st_["trained_on_simulated"] and any("SIMULATED" in n for n in st_["validation_notes"])
    r2 = c.get("/api/citypulse/risk?limit=3").json(); assert r2["ml_available"] and all(x["ml"]["kind"] == "ml-demo-simulated" and "simulated" in x["ml"]["caveat"] for x in r2["risks"])
    assert c.get("/api/citypulse/summary").json()["predicted_risks"]["kind"] == "ml-demo-simulated"
    assert c.get("/api/citypulse/risk?horizon_days=7&limit=2").json()["risks"][0]["ml"] is None            # model horizon is fixed; no silent mismatch

# ---------- ML module ----------
@pytest.fixture
def seeded():
    c.post("/api/citypulse/demo/seed"); return cpapi.compute()

def test_features_use_only_information_available_at_t():
    attrs = {"hw": "primary", "rq": 0.3}; t = NOW - timedelta(days=5)
    reps = [rep(1, days=40, reporter="a", status="unverified"), rep(2, days=20, reporter="b", status="verified", reviewed_days=19)]
    base = ml.features_at(reps, t, attrs)
    future = rep(9, days=-0.0, reporter="z"); future["observed_at"] = t + timedelta(days=1); future["created_at"] = t + timedelta(days=1)
    late_entry = rep(10, days=7, reporter="y", created_days=2)                              # observed before t, but only submitted after t
    assert ml.features_at(reps + [future, late_entry], t, attrs) == base
    reviewed_later = [dict(r) for r in reps]; reviewed_later[1]["reviewed_at"] = t + timedelta(days=1)         # verification happened after t
    assert base["verified_share_90"] == 0.5 and ml.features_at(reviewed_later, t, attrs)["verified_share_90"] == 0.0
    rejected_later = [dict(reps[0], status="rejected", reviewed_at=t + timedelta(days=1))]            # rejected only AFTER t, so still counted at t
    assert ml.features_at(rejected_later, t, attrs)["n_90"] == 1                              # rejection not yet known at t

def test_label_window_and_chronological_split_with_embargo():
    t = NOW - timedelta(days=30); H = 14
    inside = rep(1, days=20); outside = rep(2, days=10); rej = rep(3, days=25, status="rejected")
    assert ml.label_at([inside], t, H) == 1 and ml.label_at([outside], t, H) == 0 and ml.label_at([rej], t, H) == 0
    dates = [NOW - timedelta(days=7 * k) for k in range(60, 0, -1)]; tr, va, te, t1, t2 = ml.split_dates(dates, H); Hd = timedelta(days=H)
    T, V, E = ([d for d in dates if f(d)] for f in (tr, va, te))
    assert max(T) + Hd < min(V) and max(V) + Hd < min(E) and not set(T) & set(V) and not set(V) & set(E) and T and V and E
    assert min(E) >= t2 and abs(len(T) / len(dates) - 0.6) < 0.1

def test_training_refuses_when_data_is_insufficient():
    cfg = get_config()
    assert ml.train([], {}, {}, NOW, cfg, save=False)["status"] == "insufficient_data"
    few = [rep(i, days=30 - i, reporter=f"r{i}") for i in range(1, 6)]
    b = ml.train(few, snap(few), {"1-2": {"hw": "primary"}}, NOW, cfg, include_demo=True, save=False)
    assert b["status"] == "insufficient_data" and "too little time" in b["reason"] and b["validated"] is False

def test_demo_model_is_labelled_simulated_and_evaluated_on_held_out_data(seeded):
    R, cfg, now, reports, snaps, attrs, segs, _ = seeded
    assert ml.train(reports, snaps, attrs, now, cfg, include_demo=False, save=False)["status"] == "insufficient_data"       # demo data excluded by default
    b = ml.train(reports, snaps, attrs, now, cfg, include_demo=True, save=False)
    assert b["status"] == "demo_simulated" and b["validated"] is False and b["trained_on_simulated"] and any("SIMULATED" in n for n in b["validation_notes"])
    m = b["metrics"]; assert m["rows"]["train"] > m["rows"]["test"] > 50 and m["split_dates"]["embargo_days"] == cfg.risk_horizon_days
    for k in ("roc_auc", "average_precision", "brier", "log_loss", "base_rate", "positives"): assert k in m["test_model"] and k in m["test_baseline_heuristic"]
    assert 0.5 < m["test_model"]["roc_auc"] <= 1.0 and 0 < m["test_model"]["base_rate"] < 1 and any("Weather" in x for x in b["limitations"])

def test_prediction_contributions_are_exact_and_probability_is_valid(seeded):
    R, cfg, now, reports, snaps, attrs, segs, _ = seeded; b = ml.train(reports, snaps, attrs, now, cfg, include_demo=True, save=False)
    by = {}
    for r in reports:
        s = snaps.get(r["id"])
        if s: by.setdefault(s[0], []).append(r)
    for key, reps in list(by.items())[:5]:
        f = ml.features_at(reps, now, attrs[key], cfg); p = ml.predict(b, f)
        assert 0 < p["probability"] < 1 and p["intercept_logodds"] + sum(x["contribution_logodds"] for x in p["contributions"]) == pytest.approx(p["logit"], abs=2e-3)
        x = np.array([[f[n] for n in b["features"]]]); assert p["probability"] == pytest.approx(b["model"].predict_proba(b["scaler"].transform(x))[0, 1], abs=1e-9)

def test_risk_responses_never_present_predictions_as_incidents():
    c.post("/api/citypulse/demo/seed"); j = c.get("/api/citypulse/risk").json(); assert "not confirmed incidents" in j["note"]
    p = c.get("/api/citypulse/summary").json()["predicted_risks"]; assert "not confirmed incidents" in p["label"] and "uncalibrated" in p["label"]

# ---------- pages ----------
def test_new_pages_are_served_and_linked_from_every_page():
    for u in ("/", "/citypulse", "/gradient-descent-lab", "/classic"):
        assert c.get(u).status_code == 200
    for u in ("/", "/citypulse", "/gradient-descent-lab"):
        t = c.get(u).text; assert 'href="/citypulse"' in t and 'href="/gradient-descent-lab"' in t and 'href="/"' in t

@pytest.mark.parametrize("page", ["app", "citypulse", "gdlab"])
def test_page_dom_ids_and_script_declarations(page):
    html = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", f"{page}.html"), encoding="utf-8").read(); js = html.split("<script>")[-1]
    ids = set(re.findall(r'id="([^"]+)"', html)); used = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", js)); assert used <= ids, sorted(used - ids)
    for name in ("esc", "st", "api"): assert len(re.findall(rf"\bconst {name}=", js)) <= 1, f"{name} declared twice"          # a redeclaration would break the whole page

def test_citypulse_page_endpoints_exist():
    html = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "citypulse.html"), encoding="utf-8").read()
    for path in ("/api/citypulse/summary", "/api/citypulse/reports", "/api/citypulse/segments", "/api/citypulse/risk", "/api/citypulse/ml/status", "/api/citypulse/demo/seed"): assert path in html
    for path in ("summary", "reports?limit=10", "segments?limit=10", "risk?limit=10", "ml/status", "config", "priority"): assert c.get("/api/citypulse/" + path).status_code == 200
