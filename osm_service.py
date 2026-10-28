"""Routing on the road graph: real OpenStreetMap if downloaded, otherwise the bundled demo network."""
import os, pickle, threading, time
import networkx as nx, numpy as np
from scipy.spatial import cKDTree
import core, multistop, osm_ml, road_quality, traffic_live as tl

PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bengaluru_osm.pkl")
LEVELS = ["Free", "Moderate", "Heavy", "Severe"]
def level(m): return int(np.digitize(m, [1.3, 1.8, 2.4]))
def road_label(r): return "Good" if r < .25 else "Fair" if r < .5 else "Poor"

class LiveUnavailable(Exception): pass

class Router:
    def __init__(s, G):
        s.G, s.edges, s.nodes = G, list(G.edges), list(G.nodes)
        s.source = G.graph.get("source", "openstreetmap"); s.demo = s.source == "demo"
        s.tree = cKDTree([[G.nodes[n]["lat"] * 111, G.nodes[n]["lon"] * 108] for n in s.nodes])
        D = [G.edges[e] for e in s.edges]
        s.hw = [d["hw"] for d in D]; s.len = np.array([d["length"] for d in D])
        s.kph = np.array([d["kph"] for d in D]); s.rough = np.array([d["rough"] for d in D], float)
        s.ff = s.len / (s.kph / 3.6); s.eix = {e: i for i, e in enumerate(s.edges)}
        s.seg_idx = np.array([d.get("seg", -1) for d in D]); s.wl = np.array([bool(d.get("wl", False)) for d in D]); s._gv = {}
        s.rq = np.array([np.nan if d.get("rq") is None else float(d["rq"]) for d in D]); s.rq_src = [d.get("rq_src", "unknown") for d in D]
        s.name2id = {G.nodes[n]["name"]: n for n in s.nodes if G.nodes[n].get("name")}
        s.mid_latlon = [((G.nodes[u]["lat"] + G.nodes[v]["lat"]) / 2, (G.nodes[u]["lon"] + G.nodes[v]["lon"]) / 2) for u, v in s.edges]
        s.mid = np.array([[a * 111, b * 108] for a, b in s.mid_latlon])
        s.SEG = core.load_segments() if s.demo else None
        s.live = Live(s)

    def nearest(s, lat, lon):
        d, i = s.tree.query([lat * 111, lon * 108]); n = s.nodes[i]
        return n, float(d), {"lat": s.G.nodes[n]["lat"], "lon": s.G.nodes[n]["lon"], "name": s.G.nodes[n].get("name"), "distance_km": round(float(d), 2)}

    def _profile(s, hour, day, weather):
        if s.demo: return core.rule_multipliers(s.SEG, hour, day, weather)[s.seg_idx]
        return tl.profile_mult(s.hw, hour, day, weather)

    def _ml(s, hour, day, weather):
        if s.demo: return core.ml_multipliers(core.get_bundle(), s.SEG, hour, day, weather)[s.seg_idx]
        return osm_ml.predict_edges(s, hour, day)

    def multipliers(s, engine, hour, day, weather):
        """Returns (per-edge slowdown, kind, note). `kind` says exactly where the numbers came from."""
        note = ""
        if engine == "live":
            m, n = s.live.get()
            if m is not None: return m, "live", n
            note = f"Live data unavailable ({n}); showing simulated traffic instead"
        elif engine == "ml":
            m = s._ml(hour, day, weather)
            if m is not None: return m, ("simulated-ml" if s.demo else "ml-live-history"), ""
            note = "ML model not trained yet; showing the simulated profile instead"
        return s._profile(hour, day, weather), ("simulated-profile" if engine == "profile" else "simulated-fallback"), note

    def _cost(s, mult, w, extra=None):
        tsec = s.ff * mult * (1 + 0.03 * s.rough); wt, wd, wr = w; ws = (wt + wd + wr) or 1
        cost = (wt * tsec + wd * s.len / 8.33 + wr * tsec * s.rough / 5) / ws        # seconds-equivalent
        return tsec, (cost if extra is None else cost * extra)      # extra: optional per-edge factor (CityPulse road conditions)

    def _graph(s, avoid_poor=False, avoid_waterlogged=False):
        """The graph, optionally without poor-surface (roughness >= 4) or waterlogging-prone roads."""
        if not (avoid_poor or avoid_waterlogged): return s.G
        k = (avoid_poor, avoid_waterlogged)
        if k not in s._gv:
            s._gv[k] = s.G.edge_subgraph([e for e, i in s.eix.items() if not ((avoid_poor and s.rough[i] >= 4) or (avoid_waterlogged and s.wl[i]))])
        return s._gv[k]

    # ---------- traffic profiles ----------
    def profile_info(s, profile):
        if profile == "free_flow":
            return dict(profile=profile, kind="free-flow-assumption", verified_live=False,
                        source="Simulated demo free-flow speeds" if s.demo else "OSM maxspeed tag where present, otherwise ASSUMED urban speeds by road class",
                        assumptions="No congestion: every road at its free-flow speed.", limitations="Real travel times will be longer, especially at peak hours.")
        if profile in ("typical", "peak"):
            return dict(profile=profile, kind="simulated-profile", verified_live=False,
                        source="Simulated hourly congestion profile (demo dataset)" if s.demo else "ASSUMED peak slowdown by road class x hourly curve",
                        assumptions=("Typical slowdown for the chosen hour and day." if profile == "typical" else "Weekday 9 AM peak profile."),
                        limitations="Not measured traffic. Values are assumptions, not observations.")
        return dict(profile="live", kind="live", verified_live=True, source="TomTom Flow Segment Data (real readings)",
                    assumptions="Readings on a sample of main roads are spread to nearby roads (inverse-distance weighting).",
                    limitations="Minor roads are approximated. Describes the present moment only.")

    def profile_multipliers(s, profile, hour, day):
        """-> (per-edge slowdown, info). Raises LiveUnavailable if live data is not genuinely available."""
        info = s.profile_info(profile)
        if profile == "free_flow": return np.ones(len(s.edges)), info
        if profile == "typical": return s._profile(hour, day, "dry"), info
        if profile == "peak": return s._profile(9, "weekday", "dry"), info
        m, note = s.live.get()
        if m is None: raise LiveUnavailable(note)
        info["note"] = note; return m, info

    # ---------- multi-stop ----------
    def multi(s, nodes, labels, snapped, mult, objective, road_pref, mode, max_mid):
        a = road_quality.alpha(road_pref)
        base = s.ff * mult if objective == "fastest" else s.len.astype(float)
        q = np.where(np.isnan(s.rq), 0.0, s.rq); wc = base * (1 + a * q)
        if not (np.isfinite(wc).all() and (wc >= 0).all()): raise multistop.InvalidWeight("Some edges have missing or invalid weights (check speeds and lengths).")
        cd = dict(zip(s.edges, wc)); res = multistop.plan(s.G, nodes, lambda u, v, d: cd[(u, v)], labels=labels, max_mid=max_mid)
        keys = ["optimized"] if mode == "optimized" else ["strict"] if mode == "strict" else ["optimized", "strict"]
        return {k: s._describe_multi(res[k], res["entered_cost"], labels, snapped, mult, objective, a, base, q) for k in keys}

    def _describe_multi(s, r, entered_cost, labels, snapped, mult, objective, a, base, q):
        G = s.G; legs, segs, edges = [], [], []
        for i, j, path, _ in r["legs"]:
            es = list(zip(path, path[1:])); ix = [s.eix[e] for e in es]; edges += es
            legs.append({"from": labels[i], "to": labels[j], "distance_km": round(sum(s.len[k] for k in ix) / 1000, 2), "time_min": round(sum(s.ff[k] * mult[k] for k in ix) / 60, 1)})
            for e, k in zip(es, ix):
                d = G.edges[e]; lv = int(np.digitize(mult[k], [1.3, 1.8, 2.4]))
                g = d["geom"] or [(G.nodes[e[0]]["lat"], G.nodes[e[0]]["lon"]), (G.nodes[e[1]]["lat"], G.nodes[e[1]]["lon"])]
                if segs and segs[-1]["level"] == lv: segs[-1]["geometry"] += g[1:]
                else: segs.append({"level": lv, "geometry": list(g)})
        ix = np.array([s.eix[e] for e in edges], int); L = s.len[ix]; tot = float(L.sum()) or 1.0
        time_s = s.ff[ix] * mult[ix]; known = ~np.isnan(s.rq[ix]); kl = float(L[known].sum())
        srcs = {}
        for k, l in zip(ix, L): srcs[s.rq_src[k]] = srcs.get(s.rq_src[k], 0.0) + float(l)
        pen = float((base[ix] * a * q[ix]).sum()); fast = objective == "fastest"
        avg = float((mult[ix] * s.ff[ix]).sum() / s.ff[ix].sum()) if len(ix) else 1.0
        osm = any(k.startswith("osm") for k in srcs); sim = "simulated" in srcs; unk = "unknown" in srcs
        note = " ".join(([road_quality.SOURCE_NOTE["osm"]] if osm else []) + ([road_quality.SOURCE_NOTE["simulated"]] if sim else []) + ([road_quality.SOURCE_NOTE["unknown"]] if unk else []))
        cum_t = np.cumsum(time_s); node_seq = [edges[0][0]] + [e[1] for e in edges] if edges else []
        samples, seen = [], set()
        for f in (0, .25, .5, .75, 1):
            pos = 0 if f == 0 or not len(cum_t) else int(np.searchsorted(cum_t, f * cum_t[-1])) + 1
            n = G.nodes[node_seq[min(pos, len(node_seq) - 1)]]; key = (round(n["lat"], 2), round(n["lon"], 2))
            if key not in seen: seen.add(key); samples.append({"lat": n["lat"], "lon": n["lon"], "offset_min": float(cum_t[pos - 1]) / 60 if pos else 0.0})
        n_last = len(labels) - 1
        order = [{"index": i, "role": "source" if k == 0 else "destination" if i == n_last else "stop", "label": labels[i], "lat": snapped[i]["req_lat"], "lon": snapped[i]["req_lon"],
                  "snapped": {"lat": snapped[i]["lat"], "lon": snapped[i]["lon"], "distance_km": snapped[i]["distance_km"]}} for k, i in enumerate(r["order"])]
        conv = (lambda v: round(v / 60, 2)) if fast else (lambda v: round(v / 1000, 3))
        return {"order": order, "legs": legs, "distance_km": round(tot / 1000, 2), "time_min": round(float(time_s.sum()) / 60, 1),
                "objective_value": {"value": round(r["total"] / 60, 2) if fast else round(r["total"] / 1000, 3), "unit": "min (travel time + road-quality penalty)" if fast else "km (distance + road-quality penalty)"},
                "avg_slowdown": round(avg, 2), "traffic_level": LEVELS[level(avg)], "segments": [{"level": x["level"], "geometry": [[round(p, 5), round(c, 5)] for p, c in x["geometry"]]} for x in segs],
                "road_quality": {"preference_alpha": a, "known_share": round(kl / tot, 3), "unknown_share": round(1 - kl / tot, 3),
                                 "mean_score_known": round(float((s.rq[ix][known] * L[known]).sum() / kl), 3) if kl else None,
                                 "penalty": {"value": round(pen / 60, 2) if fast else round(pen / 1000, 3), "unit": "min-equivalent" if fast else "km-equivalent"},
                                 "sources_km": {k: round(v / 1000, 2) for k, v in srcs.items()}, "note": note},
                "optimization": {"method": r["method"], "exact": r["method"] in ("exact_held_karp", "trivial"), "states_evaluated": r["evaluated"], "entered_order_objective": conv(entered_cost),
                                 "improvement_vs_entered_order_pct": round(100 * (entered_cost - r["total"]) / entered_cost, 2) if entered_cost > 0 else 0.0},
                "_weather_samples": samples, "_cost": r["total"]}

    def quick(s, src, dst, mult, w=(0.6, 0.2, 0.2), avoid_poor=False, avoid_waterlogged=False):
        """Travel time (min) of the best route, without alternatives. Used for the best-time chart."""
        tsec, cost = s._cost(mult, w); cd, td = dict(zip(s.edges, cost)), dict(zip(s.edges, tsec))
        try: p = nx.shortest_path(s._graph(avoid_poor, avoid_waterlogged), src, dst, weight=lambda u, v, d: cd[(u, v)])
        except (nx.NetworkXNoPath, nx.NodeNotFound): return None
        return sum(td[e] for e in zip(p, p[1:])) / 60

    def route(s, src, dst, mult, w=(0.6, 0.2, 0.2), k=3, avoid_poor=False, avoid_waterlogged=False, extra=None):
        try: return s._route(src, dst, mult, w, k, s._graph(avoid_poor, avoid_waterlogged), extra)
        except (nx.NetworkXNoPath, nx.NodeNotFound): return []

    def _route(s, src, dst, mult, w, k, G, extra=None):
        tsec, cost = s._cost(mult, w, extra)
        cd, td = dict(zip(s.edges, cost)), dict(zip(s.edges, tsec)); lv = np.digitize(mult, [1.3, 1.8, 2.4]); pen, paths = {}, []
        for _ in range(k + 3):                                  # penalise used roads to find alternatives
            p = nx.shortest_path(G, src, dst, weight=lambda u, v, d: cd[(u, v)] * pen.get((u, v), 1.0))
            for e in zip(p, p[1:]): pen[e] = pen.get(e, 1.0) * 1.5
            if p not in paths: paths.append(p)
            if len(paths) == k: break
        sd = nx.shortest_path(G, src, dst, weight="length")
        if sd not in paths: paths.append(sd)
        out = []
        for p in paths:
            es = list(zip(p, p[1:])); L = sum(G.edges[e]["length"] for e in es)
            rp = sum(G.edges[e]["rough"] * G.edges[e]["length"] for e in es) / L / 5
            segs, steps = [], []
            for e in es:
                d = G.edges[e]; i = s.eix[e]
                g = d["geom"] or [(G.nodes[e[0]]["lat"], G.nodes[e[0]]["lon"]), (G.nodes[e[1]]["lat"], G.nodes[e[1]]["lon"])]
                if segs and segs[-1]["level"] == int(lv[i]): segs[-1]["geometry"] += g[1:]
                else: segs.append({"level": int(lv[i]), "geometry": list(g)})
                if steps and steps[-1]["road"] == d["name"]: steps[-1]["m"] += d["length"]; steps[-1]["s"] += td[e]
                else: steps.append({"road": d["name"], "m": d["length"], "s": td[e]})
            ffs = sum(s.ff[s.eix[e]] for e in es); avg = sum(mult[s.eix[e]] * s.ff[s.eix[e]] for e in es) / ffs
            out.append({"_nodes": p, "distance_km": round(L / 1000, 2), "time_min": round(sum(td[e] for e in es) / 60, 1),
                        "road_roughness": round(rp, 2), "road_condition": road_label(rp), "avg_slowdown": round(float(avg), 2),
                        "traffic_level": LEVELS[level(avg)], "cost": sum(cd[e] for e in es), "path": [s.G.nodes[n].get("name") for n in p] if s.demo else None, "is_shortest_distance": p == sd,
                        "segments": [{"level": x["level"], "geometry": [[round(a, 5), round(b, 5)] for a, b in x["geometry"]]} for x in segs],
                        "steps": [{"road": t["road"], "distance_km": round(t["m"] / 1000, 2), "time_min": round(t["s"] / 60, 1)} for t in steps]})
        out.sort(key=lambda r: r["cost"])
        for i, r in enumerate(out, 1): r["rank"] = i; r["recommended"] = i == 1; r["cost"] = round(r["cost"] / 60, 1)
        return out

class Live:
    """Refreshes TomTom readings in a background thread so requests never wait on the API."""
    def __init__(s, R): s.R, s.mult, s.ts, s.n, s.busy, s.samples = R, None, 0, 0, False, None
    def _refresh(s):
        try:
            s.samples = s.samples or tl.pick_samples(s.R)
            rows = tl.snapshot(s.R, s.samples)
            if rows: s.mult, s.ts, s.n = tl.spread(s.R, rows), time.time(), len(rows)
        finally: s.busy = False
    def get(s, max_age=600):
        if not tl.key(): return None, "TOMTOM_KEY not set"
        if time.time() - s.ts > max_age and not s.busy:
            s.busy = True; threading.Thread(target=s._refresh, daemon=True).start()
        if s.mult is None: return None, "fetching live data, try again in a minute"
        return s.mult, f"TomTom live, {s.n} measured roads, {int(time.time() - s.ts)} s old"

def hotspots(R, mult, n=5):
    seen, out = set(), []
    for i in np.argsort(-mult):
        d = R.G.edges[R.edges[i]]; key = d["name"] if d["name"] != d["hw"] else R.edges[i]
        if key in seen: continue
        seen.add(key); out.append({"road": d["name"], "slowdown_x": round(float(mult[i]), 2), "level": LEVELS[level(mult[i])]})
        if len(out) == n: break
    return out

_D = None
def get_demo_router():
    """Always the 21-junction simulated network (used by the original /api/route endpoints)."""
    global _D
    if _D is None:
        import osm_ingest; _D = Router(osm_ingest.build_demo())
    return _D

_R = None
def get_router(reload=False):
    global _R
    if _R is None or reload:
        if os.path.exists(PKL): G = pickle.load(open(PKL, "rb"))
        else: return get_demo_router()
        _R = Router(G)
    return _R
