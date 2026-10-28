"""Download Bengaluru's main roads from OpenStreetMap and save a routing graph.
Run once (needs internet, takes a few minutes):  python osm_ingest.py
"""
import argparse, os, pickle
import networkx as nx
import road_quality

PLACE = "Bengaluru, Karnataka, India"
FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link"]'
DEFAULT_KPH = {"motorway": 70, "trunk": 55, "primary": 40, "secondary": 35, "tertiary": 30}   # assumed urban free-flow speeds
ROUGH = {"asphalt": 1, "concrete": 1, "paved": 1, "concrete:plates": 2, "paving_stones": 3, "sett": 3,
         "cobblestone": 3, "compacted": 3, "gravel": 4, "unpaved": 4, "dirt": 5, "ground": 5}  # 0 smooth .. 5 rough, from OSM 'surface'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bengaluru_osm.pkl")

def first(x): return x[0] if isinstance(x, list) else x

def speed_kph(d):
    try: return float(str(first(d.get("maxspeed"))).split()[0])
    except ValueError: return DEFAULT_KPH.get(str(first(d.get("highway"))).replace("_link", ""), 30)

def from_osmnx(M):
    """Convert an OSMnx graph into the compact routing graph used by the app."""
    G = nx.DiGraph(source="openstreetmap")
    for n, d in M.nodes(data=True): G.add_node(n, lat=d["y"], lon=d["x"])
    for u, v, d in M.edges(data=True):
        L = float(d["length"])
        if G.has_edge(u, v) and G[u][v]["length"] <= L: continue          # keep the shortest parallel edge
        hw = str(first(d.get("highway"))).replace("_link", "")
        geom = [(y, x) for x, y in d["geometry"].coords] if "geometry" in d else None
        su, sm = first(d.get("surface")), first(d.get("smoothness")); rq, rq_src = road_quality.edge_quality(su, sm)
        G.add_edge(u, v, length=L, hw=hw, kph=speed_kph(d), rough=ROUGH.get(str(su), 2),     # 'rough' is legacy (unknown -> 2); use rq for new features
                   name=str(first(d.get("name")) or hw), geom=geom, surface=su, smoothness=sm, rq=rq, rq_src=rq_src)
    H = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy(); H.graph["source"] = "openstreetmap"
    return H

def configure():
    """OSMnx drops the 'surface' tag unless asked to keep it; we need it for the road-quality proxy."""
    import osmnx as ox
    for tag in ("surface", "smoothness"):
        if tag not in ox.settings.useful_tags_way: ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + [tag]
    return ox

def build(place=PLACE):
    ox = configure()                                    # imported lazily so the demo graph works without osmnx
    return from_osmnx(ox.graph_from_place(place, custom_filter=FILTER, simplify=True))

def build_demo():
    """Offline fallback: the 21-junction SIMULATED network as a routing graph (straight-line roads, not real OSM geometry)."""
    import pandas as pd
    d = os.path.join(os.path.dirname(OUT))
    nodes, seg = pd.read_csv(os.path.join(d, "bengaluru_nodes.csv")), pd.read_csv(os.path.join(d, "bengaluru_road_segments.csv"))
    G = nx.DiGraph(source="demo")
    for r in nodes.itertuples(): G.add_node(r.node_id, lat=r.latitude, lon=r.longitude, name=r.location)
    for k, r in enumerate(seg.itertuples()):
        v = r.freeflow_speed_kmph; hw = "trunk" if v >= 55 else "primary" if v >= 45 else "secondary" if v >= 38 else "tertiary"
        for a, b in ((r.from_id, r.to_id), (r.to_id, r.from_id)):
            G.add_edge(a, b, length=r.distance_km * 1000, hw=hw, kph=float(v), rough=float(r.pothole_score_0to5),
                       name=r.road_name, geom=None, seg=k, wl=r.waterlogging_prone == "Yes",
                       rq=float(r.pothole_score_0to5) / 5, rq_src="simulated")
    return G

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download Bengaluru roads from OpenStreetMap")
    ap.add_argument("--place", default=PLACE)
    a = ap.parse_args()
    G = build(a.place)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); pickle.dump(G, open(OUT, "wb"))
    print(f"Saved {G.number_of_nodes()} nodes, {G.number_of_edges()} edges to {OUT}")
