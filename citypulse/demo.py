"""SIMULATED demonstration reports. Every row is stored with source='demo_seed' and is labelled simulated everywhere it is shown.
Generator (seeded, reproducible): per segment and day, new problem episodes start with a probability that depends on the
segment's road-quality score, its waterlogging flag and a monsoon season factor (Jun-Sep x4, Oct-Nov x2, else x0.3).
Each episode produces 1-3 reports by different synthetic reporters within 36 h. ~30% are marked verified, ~4% rejected.
Because the simulator is known, a model trained on it only shows the pipeline works; it says nothing about real roads."""
import random
from datetime import timedelta
from . import geo, store
from .config import ISSUE_TYPES

def season(month): return 4.0 if 6 <= month <= 9 else 2.0 if month in (10, 11) else 0.3

def _point(line, rng):
    i = rng.randrange(len(line) - 1); t = rng.random(); a, b = line[i], line[i + 1]
    return a[0] + t * (b[0] - a[0]) + rng.uniform(-2e-4, 2e-4), a[1] + t * (b[1] - a[1]) + rng.uniform(-2e-4, 2e-4)

def generate(R, n_segments=36, days=540, seed=42, now=None, intensity=0.6):
    rng, now = random.Random(seed), (now or store.utcnow()).replace(minute=0, second=0, microsecond=0)
    canon = {}
    for e in R.edges: canon.setdefault(geo.seg_key(*e), e)
    keys = sorted(canon); chosen = sorted(rng.sample(keys, min(n_segments, len(keys))))
    rows = []
    for key in chosen:
        info = geo.segment_info(R, canon[key]); rq = info["rq"] if info["rq"] is not None else 0.3; wl = info["waterlogging_prone"]
        for d in range(days, 0, -1):
            day = now - timedelta(days=d)
            lam0 = {"pothole": 0.004 + 0.03 * rq, "waterlogging": season(day.month) * (0.003 + (0.02 if wl else 0)), "damaged_surface": 0.002 + 0.01 * rq, "obstruction": 0.001, "other": 0.0005}
            lam = {t: v * intensity for t, v in lam0.items()}
            for t in ISSUE_TYPES:
                if rng.random() >= lam[t]: continue
                start = day + timedelta(hours=rng.uniform(0, 24))
                for k in range(rng.randint(1, 3)):
                    obs = min(now, start + timedelta(hours=rng.uniform(0, 36))); lat, lon = _point(info["geometry"], rng)
                    sev = max(1, min(5, round(1 + 4 * (0.3 * rq + 0.7 * rng.random())))) if rng.random() < 0.8 else None
                    u = rng.random(); status = "verified" if u < 0.30 and (now - obs).days >= 2 else "rejected" if u > 0.96 else "unverified"
                    rows.append({"issue_type": t, "lat": lat, "lon": lon, "description": "Simulated demonstration report", "severity": sev, "observed_at": obs,
                                 "created_at": obs, "status": status, "reviewed_at": obs + timedelta(days=rng.uniform(1, 5)) if status != "unverified" else None,
                                 "source": "demo_seed", "reporter_hash": f"demo-{rng.randrange(400):03d}"})
    return rows

def seed(R, replace=False, **kw):
    existing = sum(r["n"] for r in store.counts() if r["source"] == "demo_seed")
    if existing and not replace: raise store.ReportError({"demo": f"{existing} demonstration reports already exist; use replace=true"}, 409)
    if existing: store.delete_source("demo_seed")
    rows = generate(R, **kw); rows = [r for r in rows if r["reviewed_at"] is None or r["reviewed_at"] <= store.utcnow()]
    return {"inserted": store.insert_many(rows), "simulated": True, "source": "demo_seed"}
