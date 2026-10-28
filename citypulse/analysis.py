"""Recurring-problem detection and the transparent maintenance-priority score. Pure functions: no database, no web code.

Terms
- episode: consecutive reports of the same issue type on one segment, each within `episode_gap_hours` of the previous one.
  Many reports inside one episode are ONE problem, not many.
- independent observation: one distinct reporter within one episode. The same person reporting twice adds nothing.
- recurring: a segment+type with >= min_episodes separate episodes inside the lookback window, seen by >= min_distinct_reporters
  different reporters overall.
The priority score is a decision-support recommendation for inspection. It is NOT an official or engineering assessment."""
import math
from collections import defaultdict
from datetime import timedelta
from .config import HW_IMPACT, get_config

def active_reports(reports, now, cfg):
    lo = now - timedelta(days=cfg.lookback_days)
    return [r for r in reports if r["status"] != "rejected" and lo <= r["observed_at"] <= now]

def group_episodes(reps, gap_hours):
    """Split one segment+type's reports into episodes. Returns a list of lists, each sorted by observation time."""
    out, last = [], None
    for r in sorted(reps, key=lambda r: (r["observed_at"], r["id"])):
        if last is not None and (r["observed_at"] - last).total_seconds() <= gap_hours * 3600: out[-1].append(r)
        else: out.append([r])
        last = r["observed_at"]
    return out

def analyse_issue_type(reps, cfg):
    eps = group_episodes(reps, cfg.episode_gap_hours)
    reporters = {r["reporter"] for r in reps}
    obs = sum(len({r["reporter"] for r in e}) for e in eps)
    sev = [r["severity"] for r in reps if r["severity"] is not None]
    return {"raw_reports": len(reps), "episodes": len(eps), "independent_observations": obs, "distinct_reporters": len(reporters),
            "verified_reports": sum(r["status"] == "verified" for r in reps), "max_severity": max(sev) if sev else None,
            "last_observed": max(r["observed_at"] for r in reps), "first_observed": min(r["observed_at"] for r in reps)}

def is_recurring(a, cfg): return a["episodes"] >= cfg.min_episodes and a["distinct_reporters"] >= cfg.min_distinct_reporters

def recurrence_rule(cfg):
    return (f"At least {cfg.min_episodes} separate episodes (reports more than {cfg.episode_gap_hours:g} h apart) of the same issue type on the same "
            f"road segment within {cfg.lookback_days} days, reported by at least {cfg.min_distinct_reporters} different reporters. Repeat reports by one person "
            f"or within one episode count once.")

def priority(by_type, reps, now, attrs, cfg):
    """Weighted score 0-100 with every factor shown. Unavailable factors are omitted and the remaining weights renormalised."""
    sev = [r["severity"] for r in reps if r["severity"] is not None]
    last = max(r["observed_at"] for r in reps); days = max(0.0, (now - last).total_seconds() / 86400)
    top_eps = max(a["episodes"] for a in by_type.values()); reporters = len({r["reporter"] for r in reps})
    ver = sum(r["status"] == "verified" for r in reps) / len(reps)
    f = {}
    if sev: f["severity"] = (max(sev) / 5, f"highest reported severity {max(sev)}/5 ({len(sev)} report(s) supplied a severity)")
    f["recency"] = (0.5 ** (days / cfg.recency_half_life_days), f"last observed {days:.1f} days ago (half-life {cfg.recency_half_life_days:g} d)")
    f["recurrence"] = (min(1.0, (top_eps - 1) / max(1, cfg.recurrence_cap_episodes - 1)), f"{top_eps} separate episode(s) for the most repeated issue (full marks at {cfg.recurrence_cap_episodes})")
    f["confidence"] = (0.5 * ver + 0.5 * min(1.0, reporters / 3), f"{ver:.0%} of reports verified; {reporters} distinct reporter(s) (full marks at 3)")
    hw = (attrs or {}).get("hw")
    if hw is not None: f["impact"] = (HW_IMPACT.get(hw, 0.2), f"road class '{hw}' used as a proxy for commuter exposure (an assumption, not traffic counts)")
    wsum = sum(cfg.weights[k] for k in f); rows = []
    for k, (v, why) in f.items():
        w = cfg.weights[k] / wsum; rows.append({"name": k, "value": round(v, 3), "weight": round(w, 3), "points": round(100 * w * v, 1), "explanation": why})
    missing = [k for k in cfg.weights if k not in f]
    return {"score": round(sum(r["points"] for r in rows), 1), "factors": rows, "not_available": missing,
            "label": "Decision-support recommendation for inspection, not an official or engineering assessment."}

def analyse(reports, snaps, seg_attrs, now, cfg=None):
    """reports: dicts from store.for_analysis(); snaps: {report_id: (key, edge, km) or None}; seg_attrs: {key: info}.
    Returns (list of per-segment analyses sorted by priority, unmatched report count)."""
    cfg = cfg or get_config(); by_seg, unmatched = defaultdict(list), 0
    for r in active_reports(reports, now, cfg):
        s = snaps.get(r["id"])
        if s is None: unmatched += 1
        else: by_seg[s[0]].append(r)
    out = []
    for key, reps in by_seg.items():
        types = defaultdict(list)
        for r in reps: types[r["issue_type"]].append(r)
        by_type = {t: analyse_issue_type(v, cfg) for t, v in types.items()}
        attrs = seg_attrs.get(key, {})
        out.append({"key": key, "road": attrs.get("road"), "hw": attrs.get("hw"), "midpoint": attrs.get("midpoint"), "geometry": attrs.get("geometry"),
                    "raw_reports": len(reps), "independent_observations": sum(a["independent_observations"] for a in by_type.values()),
                    "statuses": {s: sum(r["status"] == s for r in reps) for s in ("verified", "unverified")},
                    "data_sources": {s: sum(r["source"] == s for r in reps) for s in {r["source"] for r in reps}},
                    "simulated": any(r["source"] == "demo_seed" for r in reps),
                    "issues": {t: {**{k: v for k, v in a.items() if k not in ("last_observed", "first_observed")}, "last_observed": a["last_observed"].isoformat() + "Z"} for t, a in by_type.items()},
                    "recurring": [{"issue_type": t, "episodes": a["episodes"], "distinct_reporters": a["distinct_reporters"]} for t, a in by_type.items() if is_recurring(a, cfg)],
                    "priority": priority(by_type, reps, now, attrs, cfg), "last_observed": max(r["observed_at"] for r in reps).isoformat() + "Z"})
    out.sort(key=lambda s: (-s["priority"]["score"], s["key"]))
    for i, s in enumerate(out, 1): s["rank"] = i
    return out, unmatched
