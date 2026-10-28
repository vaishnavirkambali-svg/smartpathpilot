"""CityPulse HTTP API (FastAPI router, mounted by app.py). Follows the existing /api/* conventions."""
import hmac, os
from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Body, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from osm_service import get_router
from . import analysis, demo, geo, ml, risk, store
from .config import ISSUE_TYPES, STATUSES, as_dict, get_config

router = APIRouter(prefix="/api/citypulse", tags=["citypulse"])

def _err(e): return JSONResponse(status_code=e.status, content={"detail": str(e), "errors": e.errors})

def _admin(token):
    expected = os.environ.get("CITYPULSE_ADMIN_TOKEN")
    if not expected: raise HTTPException(403, "Moderation is disabled. Set the CITYPULSE_ADMIN_TOKEN environment variable to enable verify/reject and model training.")
    if not token or not hmac.compare_digest(token, expected): raise HTTPException(403, "Invalid admin token.")

def compute(now=None):
    R, cfg, now = get_router(), get_config(), now or store.utcnow()
    reports = store.for_analysis(); snaps = geo.snap_all(R, reports); attrs = {}
    for s in snaps.values():
        if s and s[0] not in attrs: attrs[s[0]] = geo.segment_info(R, s[1])
    segs, unmatched = analysis.analyse(reports, snaps, attrs, now, cfg)
    return R, cfg, now, reports, snaps, attrs, segs, unmatched

def _public(r, snaps=None, R=None):
    out = {**r, "simulated": r["source"] == "demo_seed"}
    if snaps is not None:
        s = snaps.get(r["id"]); out["segment"] = {"key": s[0], "road": R.G.edges[s[1]]["name"], "snap_distance_km": round(s[2], 3)} if s else None
    return out

def _risk_rows(now, cfg, reports, snaps, attrs, horizon=None):
    by = {}
    for r in reports:
        s = snaps.get(r["id"])
        if s: by.setdefault(s[0], []).append(r)
    bundle = ml.load_bundle(); rows = []
    for key, reps in by.items():
        b = risk.baseline(reps, now, cfg, horizon); row = {"key": key, "road": attrs[key]["road"], "midpoint": attrs[key]["midpoint"], "baseline": b, "ml": None}
        if bundle and (horizon in (None, bundle["horizon_days"])):
            p = ml.predict(bundle, ml.features_at(reps, now, attrs[key], cfg))
            row["ml"] = {"kind": "ml-demo-simulated" if bundle["trained_on_simulated"] else "ml-validated" if bundle["validated"] else "ml-unvalidated", "probability": round(p["probability"], 3),
                         "intercept_logodds": p["intercept_logodds"], "top_contributions": p["contributions"][:5], "horizon_days": bundle["horizon_days"],
                         "caveat": "Trained on simulated demonstration data; not validated for real roads." if bundle["trained_on_simulated"] else "See /api/citypulse/ml/status for metrics and limits."}
        row["score"] = row["ml"]["probability"] if row["ml"] else b["index"]; rows.append(row)
    return sorted(rows, key=lambda r: -r["score"]), bundle

@router.get("/config")
def config():
    c = get_config()
    return {"config": as_dict(c), "recurrence_rule": analysis.recurrence_rule(c), "priority_method": "score = 100 x sum(w_i x factor_i) over the available factors, weights renormalised; every factor is listed with its points. Decision support only.",
            "routing_policy": "Only verified, still-active reports add a capped cost factor; unverified reports and predictions never change a route.", "issue_types": ISSUE_TYPES, "statuses": STATUSES}

@router.get("/summary")
def summary():
    R, cfg, now, reports, snaps, attrs, segs, unmatched = compute()
    n_demo = sum(r["source"] == "demo_seed" for r in reports); mode = "empty" if not reports else "demonstration" if n_demo == len(reports) else "real" if not n_demo else "mixed"
    rows, bundle = _risk_rows(now, cfg, reports, snaps, attrs)
    if not rows: pr = {"available": False, "reason": "No reports on mapped roads yet, so there is no history to estimate risk from."}
    else:
        kind = (rows[0]["ml"]["kind"] if rows[0]["ml"] else "baseline-heuristic")
        pr = {"available": True, "kind": kind, "count": sum(r["score"] >= cfg.risk_elevated_threshold for r in rows), "threshold": cfg.risk_elevated_threshold, "horizon_days": cfg.risk_horizon_days,
              "label": "Elevated-risk segments (estimate, not confirmed incidents)" + (" - model trained on simulated data" if kind == "ml-demo-simulated" else " - uncalibrated heuristic" if kind == "baseline-heuristic" else "")}
    return {"data_mode": mode, "demo_data_present": n_demo > 0, "graph_source": R.source, "total_reports": len(reports), "verified_reports": sum(r["status"] == "verified" for r in reports),
            "by_status": {s: sum(r["status"] == s for r in reports) for s in STATUSES}, "by_source": {s: sum(r["source"] == s for r in reports) for s in {r["source"] for r in reports}},
            "recurring_locations": sum(bool(s["recurring"]) for s in segs), "recurring_issue_groups": sum(len(s["recurring"]) for s in segs),
            "high_priority_segments": sum(s["priority"]["score"] >= cfg.high_priority_score for s in segs), "high_priority_threshold": cfg.high_priority_score,
            "unmatched_reports": unmatched, "predicted_risks": pr,
            "notes": (["Demonstration data is SIMULATED, not real road conditions."] if n_demo else []) + (["No reports yet."] if not reports else [])}

@router.get("/reports")
def reports(status: Optional[str] = None, issue_type: Optional[str] = None, source: Optional[str] = None, limit: int = Query(1000, ge=1, le=5000)):
    R = get_router(); rs = store.list_reports(status, issue_type, source, limit); snaps = geo.snap_all(R, rs)
    return {"reports": [_public(r, snaps, R) for r in rs]}

@router.get("/reports/{rid}")
def report(rid: int):
    r = store.get(rid)
    if not r: raise HTTPException(404, "Report not found.")
    R = get_router(); return _public(r, geo.snap_all(R, [r]), R)

@router.post("/reports", status_code=201)
def submit(request: Request, payload: dict = Body(...)):
    try:
        ip = request.client.host if request.client else "unknown"
        clean = store.validate({**payload, "source": "community_report", "reporter_hash": store.reporter_hash(ip, request.headers.get("user-agent", ""))})
        rid = store.insert(clean)
    except store.ReportError as e: return _err(e)
    R = get_router(); s = geo.nearest_segment(R, clean["lat"], clean["lon"])
    r = store.get(rid)
    return {"report": _public(r, geo.snap_all(R, [r]), R), "message": "Report received. It is unverified community information until a moderator verifies it.",
            "matched_segment": s[0] if s else None, "warnings": [] if s else ["The location is not close to a mapped road segment, so it cannot affect segment analysis or routing."]}

@router.patch("/reports/{rid}/status")
def moderate(rid: int, payload: dict = Body(...), x_admin_token: Optional[str] = Header(None)):
    _admin(x_admin_token)
    try: return _public(store.set_status(rid, payload.get("status")))
    except store.ReportError as e: return _err(e)

@router.get("/segments")
def segments(limit: int = Query(200, ge=1, le=1000), recurring_only: bool = False):
    R, cfg, now, reports, snaps, attrs, segs, unmatched = compute()
    if recurring_only: segs = [s for s in segs if s["recurring"]]
    return {"segments": segs[:limit], "unmatched_reports": unmatched, "recurrence_rule": analysis.recurrence_rule(cfg)}

@router.get("/priority")
def priority(limit: int = Query(20, ge=1, le=200)):
    R, cfg, now, reports, snaps, attrs, segs, unmatched = compute()
    return {"ranking": [{k: s[k] for k in ("rank", "key", "road", "hw", "midpoint", "raw_reports", "independent_observations", "recurring", "priority", "simulated", "last_observed")} for s in segs[:limit]],
            "disclaimer": "Decision-support recommendation for further inspection. Not an official municipal decision or a verified engineering assessment.",
            "weights": cfg.weights, "high_priority_threshold": cfg.high_priority_score}

@router.get("/risk")
def risk_endpoint(limit: int = Query(30, ge=1, le=300), horizon_days: Optional[int] = Query(None, ge=1, le=90)):
    R, cfg, now, reports, snaps, attrs, segs, unmatched = compute()
    rows, bundle = _risk_rows(now, cfg, reports, snaps, attrs, horizon_days)
    return {"risks": rows[:limit], "horizon_days": horizon_days or cfg.risk_horizon_days, "ml_available": bundle is not None,
            "note": "Estimates of the chance of a new report within the horizon. They are not confirmed incidents and must not be shown as such."}

@router.get("/ml/status")
def ml_status():
    b = ml.load_bundle()
    if not b: return {"trained": False, "reason": "No model trained. The baseline heuristic is used. Train with `python -m citypulse.train` once enough reports exist.", "gates": {"min_train_positives": get_config().min_train_positives, "min_test_positives": get_config().min_test_positives, "min_test_auc": get_config().min_test_auc}}
    return {"trained": True, **{k: b[k] for k in ("status", "validated", "validation_notes", "trained_on_simulated", "metrics", "horizon_days", "trained_at", "limitations", "coefficients")}}

@router.post("/ml/train")
def ml_train(include_demo: bool = False, x_admin_token: Optional[str] = Header(None)):
    _admin(x_admin_token)
    R, cfg, now, reports, snaps, attrs, segs, unmatched = compute(); b = ml.train(reports, snaps, attrs, now, cfg, include_demo=include_demo)
    return {k: b[k] for k in b if k not in ("model", "scaler")}

@router.post("/demo/seed")
def demo_seed(replace: bool = False):
    try: return demo.seed(get_router(), replace=replace)
    except store.ReportError as e: return _err(e)

@router.delete("/demo")
def demo_delete(): return {"deleted": store.delete_source("demo_seed")}
