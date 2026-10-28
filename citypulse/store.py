"""SQLite persistence for road-issue reports (standard library; file data/citypulse.db, override with CITYPULSE_DB).
Privacy: no names, emails or raw IPs are stored. Each report keeps only a salted hash of (IP + user-agent) so repeated
reports by one person are not counted as independent evidence. That hash is never returned by the API."""
import hashlib, hmac, math, os, re, secrets, sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from .config import BASE, ISSUE_TYPES, SOURCES, STATUSES, get_config

IST = ZoneInfo("Asia/Kolkata")
def db_path(): return os.environ.get("CITYPULSE_DB", os.path.join(BASE, "data", "citypulse.db"))
def utcnow(): return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)

class ReportError(Exception):
    """Validation failure. `errors` maps field name -> message; `status` is the HTTP code to use."""
    def __init__(self, errors, status=422): super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items())); self.errors, self.status = errors, status

SCHEMA = """CREATE TABLE IF NOT EXISTS reports(
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 issue_type TEXT NOT NULL CHECK (issue_type IN ('pothole','waterlogging','damaged_surface','obstruction','other')),
 lat REAL NOT NULL, lon REAL NOT NULL, description TEXT NOT NULL DEFAULT '',
 severity INTEGER CHECK (severity IS NULL OR severity BETWEEN 1 AND 5),
 observed_at TEXT NOT NULL, created_at TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'unverified' CHECK (status IN ('unverified','verified','rejected')),
 reviewed_at TEXT, source TEXT NOT NULL, reporter_hash TEXT);
CREATE INDEX IF NOT EXISTS ix_obs ON reports(observed_at); CREATE INDEX IF NOT EXISTS ix_src ON reports(source);"""

def connect():
    p = db_path(); os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    c = sqlite3.connect(p, timeout=10); c.row_factory = sqlite3.Row; c.executescript(SCHEMA); return c

def iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
def parse_ts(s):
    """ISO 8601 -> naive UTC datetime. Naive input is read as India Standard Time."""
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    d = d.replace(tzinfo=IST) if d.tzinfo is None else d
    return d.astimezone(timezone.utc).replace(tzinfo=None)

def _salt():
    s = os.environ.get("CITYPULSE_SALT")
    if s: return s.encode()
    f = os.path.join(BASE, "data", ".citypulse_salt")
    if not os.path.exists(f):
        os.makedirs(os.path.dirname(f), exist_ok=True); open(f, "w").write(secrets.token_hex(16))
    return open(f).read().strip().encode()

def reporter_hash(ip, user_agent):
    return hmac.new(_salt(), f"{ip}|{user_agent}".encode(), hashlib.sha256).hexdigest()[:16]

def haversine_m(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c); h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))

def clean_text(s): return re.sub(r"[\x00-\x1f\x7f]", " ", str(s or "")).strip()[:500]

def validate(p, now=None):
    """Validate and normalise a report payload (dict). Returns a clean dict or raises ReportError."""
    cfg, now, e = get_config(), now or utcnow(), {}
    t = p.get("issue_type")
    if t not in ISSUE_TYPES: e["issue_type"] = f"must be one of {', '.join(ISSUE_TYPES)}"
    lat, lon = p.get("lat"), p.get("lon")
    ok = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (lat, lon))
    if not ok: e["location"] = "lat and lon must be finite numbers"
    elif not (cfg.bbox[0] <= lat <= cfg.bbox[1] and cfg.bbox[2] <= lon <= cfg.bbox[3]): e["location"] = "outside the supported Bengaluru area"
    sev = p.get("severity")
    if sev is not None and (isinstance(sev, bool) or not isinstance(sev, int) or not 1 <= sev <= 5): e["severity"] = "must be an integer from 1 (minor) to 5 (severe), or omitted"
    obs = None
    try:
        obs = parse_ts(p["observed_at"]) if p.get("observed_at") else now
        if obs > now + timedelta(minutes=5): e["observed_at"] = "cannot be in the future"
        elif obs < now - timedelta(days=cfg.max_age_days): e["observed_at"] = f"cannot be older than {cfg.max_age_days} days"
    except (ValueError, TypeError): e["observed_at"] = "must be an ISO 8601 time, e.g. 2026-10-20T08:30"
    src = p.get("source", "community_report")
    if src not in SOURCES: e["source"] = f"must be one of {', '.join(SOURCES)}"
    if e: raise ReportError(e)
    return {"issue_type": t, "lat": float(lat), "lon": float(lon), "description": clean_text(p.get("description")), "severity": sev,
            "observed_at": obs, "source": src, "reporter_hash": p.get("reporter_hash")}

PUBLIC = "id, issue_type, lat, lon, description, severity, observed_at, created_at, status, reviewed_at, source"

def insert(r, now=None, check_duplicates=True, status="unverified", reviewed_at=None, created_at=None):
    """Persist a validated report. Raises ReportError(409) for a duplicate, ReportError(429) when rate-limited."""
    cfg, now = get_config(), now or utcnow(); created = created_at or now
    with connect() as c:
        if check_duplicates and r["reporter_hash"]:
            since = iso(now - timedelta(hours=1))
            n = c.execute("SELECT COUNT(*) FROM reports WHERE reporter_hash=? AND created_at>=?", (r["reporter_hash"], since)).fetchone()[0]
            if n >= cfg.max_reports_per_hour: raise ReportError({"rate_limit": f"at most {cfg.max_reports_per_hour} reports per hour"}, 429)
            lo = iso(r["observed_at"] - timedelta(hours=cfg.dup_window_hours)); hi = iso(r["observed_at"] + timedelta(hours=cfg.dup_window_hours))
            for row in c.execute("SELECT id, lat, lon FROM reports WHERE reporter_hash=? AND issue_type=? AND status!='rejected' AND observed_at BETWEEN ? AND ?", (r["reporter_hash"], r["issue_type"], lo, hi)):
                if haversine_m(r["lat"], r["lon"], row["lat"], row["lon"]) <= cfg.dup_radius_m:
                    raise ReportError({"duplicate": f"you already reported this issue nearby (report #{row['id']})"}, 409)
        cur = c.execute("INSERT INTO reports(issue_type,lat,lon,description,severity,observed_at,created_at,status,reviewed_at,source,reporter_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (r["issue_type"], r["lat"], r["lon"], r["description"], r["severity"], iso(r["observed_at"]), iso(created), status, iso(reviewed_at) if reviewed_at else None, r["source"], r["reporter_hash"]))
        return cur.lastrowid

def get(rid):
    with connect() as c:
        row = c.execute(f"SELECT {PUBLIC} FROM reports WHERE id=?", (rid,)).fetchone()
    return dict(row) if row else None

def list_reports(status=None, issue_type=None, source=None, limit=1000):
    q, a = f"SELECT {PUBLIC} FROM reports WHERE 1=1", []
    for col, v in (("status", status), ("issue_type", issue_type), ("source", source)):
        if v: q += f" AND {col}=?"; a.append(v)
    with connect() as c: return [dict(r) for r in c.execute(q + " ORDER BY observed_at DESC LIMIT ?", a + [limit])]

def for_analysis():
    """All reports with parsed times and the reporter hash (internal use only; never expose)."""
    with connect() as c:
        rows = c.execute("SELECT id, issue_type, lat, lon, severity, observed_at, created_at, status, reviewed_at, source, reporter_hash FROM reports").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("observed_at", "created_at", "reviewed_at"): d[k] = parse_ts(d[k]) if d[k] else None
        d["reporter"] = d.pop("reporter_hash") or f"anon-{d['id']}"
        out.append(d)
    return out

def set_status(rid, status, now=None):
    if status not in STATUSES: raise ReportError({"status": f"must be one of {', '.join(STATUSES)}"})
    with connect() as c:
        n = c.execute("UPDATE reports SET status=?, reviewed_at=? WHERE id=?", (status, iso(now or utcnow()), rid)).rowcount
    if not n: raise ReportError({"id": "report not found"}, 404)
    return get(rid)

def delete_source(source):
    with connect() as c: return c.execute("DELETE FROM reports WHERE source=?", (source,)).rowcount

def counts():
    with connect() as c:
        rows = c.execute("SELECT status, source, COUNT(*) n FROM reports GROUP BY status, source").fetchall()
    return [dict(r) for r in rows]

def insert_many(rows):
    """Bulk insert for demonstration data. rows: dicts with the validated fields plus status, reviewed_at, created_at."""
    with connect() as c:
        c.executemany("INSERT INTO reports(issue_type,lat,lon,description,severity,observed_at,created_at,status,reviewed_at,source,reporter_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      [(r["issue_type"], r["lat"], r["lon"], r["description"], r["severity"], iso(r["observed_at"]), iso(r["created_at"]), r["status"], iso(r["reviewed_at"]) if r["reviewed_at"] else None, r["source"], r["reporter_hash"]) for r in rows])
    return len(rows)
