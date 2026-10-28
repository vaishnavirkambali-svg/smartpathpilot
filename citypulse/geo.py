"""Link report coordinates to road segments of the ACTIVE road graph (real OSM if downloaded, otherwise the demo network).
A segment is an undirected pair of graph nodes, keyed 'min-max'. Reports are snapped at analysis time, so switching graph
never leaves stale segment ids behind."""
import numpy as np
from .config import get_config

def seg_key(u, v): a, b = sorted((u, v)); return f"{a}-{b}"

def _pt_seg_km(p, a, b):
    ab = b - a; L2 = float(ab @ ab)
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))

def edge_polyline(R, e):
    d = R.G.edges[e]
    return d["geom"] or [(R.G.nodes[e[0]]["lat"], R.G.nodes[e[0]]["lon"]), (R.G.nodes[e[1]]["lat"], R.G.nodes[e[1]]["lon"])]

def nearest_segment(R, lat, lon, max_km=None):
    """-> (key, (u, v), distance_km) or None if no road is within max_km. Uses the true polyline when geometry exists."""
    cfg = get_config(); max_km = max_km if max_km is not None else (cfg.max_snap_km_demo if R.demo else cfg.max_snap_km_osm)
    p = np.array([lat * 111.0, lon * 108.0])
    _, ni = R.tree.query(p, k=min(3, len(R.nodes)))
    cand = set()
    for i in np.atleast_1d(ni):
        n = R.nodes[i]
        cand.update((n, v) for v in R.G.successors(n)); cand.update((u, n) for u in R.G.predecessors(n))
    if not hasattr(R, "_etree"):
        from scipy.spatial import cKDTree; R._etree = cKDTree(R.mid)
    _, mi = R._etree.query(p, k=min(8, len(R.edges))); cand.update(R.edges[i] for i in np.atleast_1d(mi))
    best = None
    for e in cand:
        pts = np.array([[a * 111.0, b * 108.0] for a, b in edge_polyline(R, e)])
        d = min(_pt_seg_km(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        if best is None or d < best[2]: best = (seg_key(*e), e, d)
    return best if best and best[2] <= max_km else None

def segment_info(R, e):
    d = R.G.edges[e]; line = edge_polyline(R, e); mid = line[len(line) // 2]
    i = R.eix[e]
    return {"key": seg_key(*e), "road": d["name"], "hw": d["hw"], "rq": None if np.isnan(R.rq[i]) else float(R.rq[i]), "rq_src": R.rq_src[i],
            "waterlogging_prone": bool(R.wl[i]), "midpoint": [float(mid[0]), float(mid[1])], "geometry": [[float(a), float(b)] for a, b in line]}

_cache = {}
def snap_all(R, reports):
    """{report_id: (key, edge, dist_km) or None}, memoised per graph and location."""
    sig = (R.source, len(R.edges), id(R)); out = {}
    for r in reports:
        k = (sig, round(r["lat"], 6), round(r["lon"], 6))
        if k not in _cache:
            if len(_cache) > 50000: _cache.clear()
            _cache[k] = nearest_segment(R, r["lat"], r["lon"])
        out[r["id"]] = _cache[k]
    return out
