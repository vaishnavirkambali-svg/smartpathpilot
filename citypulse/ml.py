"""Optional ML road-risk model (logistic regression, so every prediction is an exact sum of feature contributions).

Target: 1 if the segment gets at least one new non-rejected report with observed_at in [t, t + horizon) , else 0.
Rows: one per (segment, weekly snapshot date t).
Leakage control: features at t use only reports with observed_at < t AND created_at <= t, and a report counts as
verified/rejected only if reviewed_at <= t. Splits are chronological with a `horizon` embargo so no training label window
overlaps the validation/test period.   train: d < t60 - H | validation: t60 <= d < t80 - H | test: d >= t80.
Honesty: if the training data include simulated reports the model is labelled 'demo_simulated' and its metrics describe the
simulator, not the real world. Only a model trained on real reports that passes the configured gates is 'validated'."""
import math, os
from datetime import timedelta
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from .analysis import group_episodes
from .config import BASE, HW_IMPACT, get_config

def model_path(): return os.environ.get("CITYPULSE_MODEL", os.path.join(BASE, "models", "citypulse_risk.joblib"))
FEATURES = ["n_30", "n_90", "ep_90", "ep_180", "days_since_last", "max_sev_90", "has_sev_90", "verified_share_90", "rq", "rq_known",
            "waterlogging_prone", "season_sin", "season_cos", "road_class_proxy"]
LIMITS = ["Trained on report history, which reflects where people report problems, not only where problems occur.",
          "Class balance and report volume change over time; performance on the next period is not guaranteed.",
          "Predicts the chance of a new report within the horizon, not the occurrence of a specific incident.",
          "Weather and rainfall are not model inputs because no historical rainfall data is stored."]

def status_at(r, t): return r["status"] if r["reviewed_at"] is not None and r["reviewed_at"] <= t else "unverified"

def features_at(reps, t, attrs, cfg=None):
    """Features for ONE segment at time t from `reps` (all reports of that segment). Uses only information available at t."""
    cfg = cfg or get_config()
    known = [r for r in reps if r["observed_at"] < t and r["created_at"] <= t and status_at(r, t) != "rejected"]
    win = lambda d: [r for r in known if r["observed_at"] >= t - timedelta(days=d)]
    def eps(ws):
        by = {}
        for r in ws: by.setdefault(r["issue_type"], []).append(r)
        return sum(len(group_episodes(v, cfg.episode_gap_hours)) for v in by.values())
    w30, w90, w180 = win(30), win(90), win(180)
    last = max((r["observed_at"] for r in known), default=None)
    sev = [r["severity"] for r in w90 if r["severity"] is not None]; rq = attrs.get("rq"); doy = t.timetuple().tm_yday
    return {"n_30": len(w30), "n_90": len(w90), "ep_90": eps(w90), "ep_180": eps(w180), "days_since_last": min(365.0, (t - last).total_seconds() / 86400) if last else 365.0,
            "max_sev_90": max(sev) if sev else 0, "has_sev_90": int(bool(sev)), "verified_share_90": (sum(status_at(r, t) == "verified" for r in w90) / len(w90)) if w90 else 0.0,
            "rq": rq if rq is not None else 0.0, "rq_known": int(rq is not None), "waterlogging_prone": int(bool(attrs.get("waterlogging_prone"))),
            "season_sin": math.sin(2 * math.pi * doy / 365.25), "season_cos": math.cos(2 * math.pi * doy / 365.25), "road_class_proxy": HW_IMPACT.get(attrs.get("hw"), 0.2)}

def label_at(reps, t, horizon):
    end = t + timedelta(days=horizon)
    return int(any(t <= r["observed_at"] < end and r["status"] != "rejected" for r in reps))

def snapshot_dates(first, end, horizon, cfg, step=7):
    t = (first + timedelta(days=cfg.lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0); out = []
    while t + timedelta(days=horizon) <= end: out.append(t); t += timedelta(days=step)
    return out

def baseline_p(ep_180, cfg, horizon): return 1 - math.exp(-((ep_180 + cfg.risk_prior_episodes) / (cfg.lookback_days + cfg.risk_prior_days)) * horizon)

def _metrics(y, p):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "average_precision": round(float(average_precision_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 4),
            "log_loss": round(float(log_loss(y, p)), 4), "base_rate": round(float(y.mean()), 4), "n": int(len(y)), "positives": int(y.sum())}

def split_dates(dates, horizon):
    """Chronological split with an embargo of `horizon` days. Returns (train_mask_fn, val_fn, test_fn, t60, t80)."""
    ds = sorted(dates); t1, t2 = ds[int(0.6 * (len(ds) - 1))], ds[int(0.8 * (len(ds) - 1))]; H = timedelta(days=horizon)
    return (lambda d: d < t1 - H), (lambda d: t1 <= d < t2 - H), (lambda d: d >= t2), t1, t2

def train(reports, snaps, attrs, now, cfg=None, include_demo=False, horizon=None, save=True):
    """reports: store.for_analysis() rows; snaps: {id: (key, edge, km)|None}; attrs: {key: segment info}. Returns a bundle dict (always has 'status')."""
    cfg = cfg or get_config(); H = horizon or cfg.risk_horizon_days
    rs = [r for r in reports if (include_demo or r["source"] != "demo_seed") and snaps.get(r["id"])]
    demo_used = any(r["source"] == "demo_seed" for r in rs)
    def fail(reason): return {"status": "insufficient_data", "validated": False, "reason": reason, "trained_on_simulated": demo_used, "limitations": LIMITS}
    if not rs: return fail("No matched reports" + ("" if include_demo else " (demonstration data is excluded unless include_demo=true)") + ".")
    by_seg = {}
    for r in rs: by_seg.setdefault(snaps[r["id"]][0], []).append(r)
    first, last = min(r["observed_at"] for r in rs), min(now, max(r["observed_at"] for r in rs))
    dates = snapshot_dates(first, last, H, cfg)
    if len(dates) < 20: return fail(f"History spans too little time: need at least {cfg.lookback_days + H + 140} days between the first and last report (have {(last - first).days}).")
    rows = [(t, features_at(reps, t, attrs[k], cfg), label_at(reps, t, H)) for k, reps in by_seg.items() for t in dates]
    tr, va, te, t1, t2 = split_dates(dates, H)
    parts = {n: [x for x in rows if f(x[0])] for n, f in (("train", tr), ("validation", va), ("test", te))}
    for n, p in parts.items():
        ys = {x[2] for x in p}
        if len(p) < 50 or ys != {0, 1}: return fail(f"The {n} split has {len(p)} rows and classes {sorted(ys)}; both classes and at least 50 rows are required.")
    mat = lambda p: (np.array([[x[1][f] for f in FEATURES] for x in p]), np.array([x[2] for x in p]))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = mat(parts["train"]), mat(parts["validation"]), mat(parts["test"])
    sc = StandardScaler().fit(Xtr); best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr); ll = log_loss(yva, np.clip(m.predict_proba(sc.transform(Xva))[:, 1], 1e-6, 1 - 1e-6))
        if best is None or ll < best[0]: best = (ll, C, m)
    _, C, model = best; pte = model.predict_proba(sc.transform(Xte))[:, 1]
    base_te = [baseline_p(x[1]["ep_180"], cfg, H) for x in parts["test"]]
    metrics = {"horizon_days": H, "chosen_C": C, "rows": {n: len(p) for n, p in parts.items()}, "split_dates": {"validation_starts": t1.date().isoformat(), "test_starts": t2.date().isoformat(), "embargo_days": H},
               "test_model": _metrics(yte, pte), "test_baseline_heuristic": _metrics(yte, base_te), "positives_train": int(ytr.sum())}
    if demo_used: status, validated, why = "demo_simulated", False, ["Trained on SIMULATED demonstration reports: the metrics describe the simulator, not the real world."]
    else:
        why = []
        if ytr.sum() < cfg.min_train_positives: why.append(f"only {int(ytr.sum())} training positives (need {cfg.min_train_positives})")
        if yte.sum() < cfg.min_test_positives: why.append(f"only {int(yte.sum())} test positives (need {cfg.min_test_positives})")
        if metrics["test_model"]["roc_auc"] < cfg.min_test_auc: why.append(f"test AUC {metrics['test_model']['roc_auc']} below {cfg.min_test_auc}")
        if metrics["test_model"]["log_loss"] >= metrics["test_baseline_heuristic"]["log_loss"]: why.append("does not beat the baseline heuristic on test log loss")
        validated = not why; status = "trained_real"
    bundle = {"status": status, "validated": validated, "validation_notes": why, "trained_on_simulated": demo_used, "model": model, "scaler": sc, "features": FEATURES, "metrics": metrics,
              "horizon_days": H, "trained_at": now.isoformat() + "Z", "limitations": LIMITS, "coefficients": {f: round(float(c), 4) for f, c in zip(FEATURES, model.coef_[0])}}
    if save:
        os.makedirs(os.path.dirname(model_path()), exist_ok=True); joblib.dump(bundle, model_path())
    return bundle

def predict(bundle, feats):
    """Probability and exact per-feature contributions in log-odds relative to an average training segment."""
    x = np.array([[feats[f] for f in bundle["features"]]], float); z = bundle["scaler"].transform(x)[0]
    coef, b = bundle["model"].coef_[0], float(bundle["model"].intercept_[0]); logit = b + float(coef @ z)
    contrib = sorted(({"feature": f, "value": round(float(v), 3), "contribution_logodds": round(float(c * zi), 4)} for f, v, c, zi in zip(bundle["features"], x[0], coef, z)), key=lambda d: -abs(d["contribution_logodds"]))
    return {"probability": 1 / (1 + math.exp(-logit)), "intercept_logodds": round(b, 4), "logit": logit, "contributions": contrib}

_cached = [None, None]
def load_bundle():
    p = model_path()
    if not os.path.exists(p): return None
    mt = os.path.getmtime(p)
    if _cached[0] != (p, mt):
        try: _cached[1] = joblib.load(p)
        except Exception: _cached[1] = None
        _cached[0] = (p, mt)
    return _cached[1]

def usable_in_routing(bundle, cfg=None):
    """Predicted risk may touch route costs only if enabled in config AND the model is trained on real data and validated."""
    cfg = cfg or get_config()
    if not cfg.use_predicted_risk_in_routing: return False, "disabled in configuration (use_predicted_risk_in_routing = false)"
    if bundle is None: return False, "no trained model"
    if bundle["trained_on_simulated"]: return False, "model was trained on simulated demonstration data"
    if not bundle["validated"]: return False, "model has not passed the validation gates: " + "; ".join(bundle["validation_notes"])
    return True, "validated on real data"
