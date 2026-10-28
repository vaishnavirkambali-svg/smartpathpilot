"""Turn reliable road-condition information into route-cost factors for the existing planner. Policy (all configurable):
- Only VERIFIED reports change route cost. Unverified reports are shown as information and never change a route.
- A report counts only while 'active' (potholes 120 d, waterlogging 3 d, obstruction 7 d, ...). Old reports expire.
- Nothing is ever blocked: factor = 1 + sum(per-severity penalty), capped (default 2.5). Physical distance/time are unchanged;
  the factor only adds to the planner's cost, like the road-quality penalty.
- Predicted risk is NOT used unless enabled AND the ML model is validated on real data (see ml.usable_in_routing).
This is not a real-time hazard feed: it reflects what has been reported and verified in the CityPulse database."""
from datetime import timedelta
import numpy as np
from . import geo, ml, store
from .config import get_config

class Context:
    def __init__(s, n_edges): s.factors = np.ones(n_edges); s.by_key = {}; s.available = True; s.notes = []; s.predicted_used = False; s.n_verified = 0; s.n_unverified = 0; s.simulated = False; s.error = None; s.n_pen_segments = 0
    def annotate(s, nodes):
        keys = {geo.seg_key(a, b) for a, b in zip(nodes, nodes[1:])}; ver, unv, seen = [], [], set()
        for k in keys:
            for i in s.by_key.get(k, []):
                if i["id"] not in seen: seen.add(i["id"]); (ver if i["status"] == "verified" else unv).append(i)
        return {"verified_issues": sorted(ver, key=lambda i: -(i["cost_factor"])), "unverified_nearby": sorted(unv, key=lambda i: i["id"]),
                "cost_factor_applied": bool(ver), "max_cost_factor": max((i["cost_factor"] for i in ver), default=1.0)}
    def summary(s):
        mode = "unavailable" if not s.available else "none" if not (s.n_verified or s.n_unverified) else "demonstration (simulated) data" if s.simulated else "community reports"
        return {"enabled": True, "available": s.available, "data_mode": mode, "verified_issues_used": s.n_verified, "unverified_issues_informational": s.n_unverified,
                "segments_penalised": s.n_pen_segments, "predicted_risk_used": s.predicted_used, "notes": s.notes + ["Based on reports stored in CityPulse, not a real-time hazard feed."] , "error": s.error}

def build(R, now=None, cfg=None, reports=None):
    """Context for router R. Never raises: on any problem the planner continues unchanged and the context says why."""
    cfg, now = cfg or get_config(), now or store.utcnow(); ctx = Context(len(R.edges))
    try: reports = reports if reports is not None else store.for_analysis()
    except Exception as e:
        ctx.available, ctx.error = False, f"CityPulse database unavailable ({type(e).__name__})"; ctx.notes.append("Route calculated without road-condition data."); return ctx
    snaps = geo.snap_all(R, reports); pen = {}
    for r in reports:
        s = snaps.get(r["id"])
        if s is None or r["status"] == "rejected": continue
        if not (now - timedelta(days=cfg.active_days[r["issue_type"]]) <= r["observed_at"] <= now): continue
        sev = r["severity"] if r["severity"] is not None else cfg.default_routing_severity
        f = 1 + cfg.penalty_per_severity[r["issue_type"]] * sev if r["status"] == "verified" else 1.0
        sim = r["source"] == "demo_seed"
        ctx.by_key.setdefault(s[0], []).append({"id": r["id"], "issue_type": r["issue_type"], "severity": r["severity"], "status": r["status"], "observed_at": r["observed_at"].isoformat() + "Z",
                                                "road": R.G.edges[s[1]]["name"], "source": r["source"], "simulated": sim, "cost_factor": round(f, 3)})
        if r["status"] == "verified": pen[s[0]] = pen.get(s[0], 0.0) + (f - 1); ctx.n_verified += 1; ctx.simulated |= sim
        else: ctx.n_unverified += 1
    for i, (u, v) in enumerate(R.edges): ctx.factors[i] = min(cfg.max_cost_factor, 1 + pen.get(geo.seg_key(u, v), 0.0))
    ctx.n_pen_segments = len(pen)
    ok, why = ml.usable_in_routing(ml.load_bundle(), cfg)
    ctx.notes.append(("Predicted risk is not used in routing: " + why + ".") if not ok else
                     "A validated model exists, but applying predicted risk to route costs is not implemented in this version; only verified reports are used.")
    ctx.predicted_used = False
    return ctx
