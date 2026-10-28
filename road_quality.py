"""Road-quality scoring from OpenStreetMap tags.

Score: 0 = best ... 1 = worst, or None = UNKNOWN. A missing tag is never treated as a poor road: unknown edges get no penalty.
These scores are an ASSUMPTION TABLE (how rough each OSM surface/smoothness value usually is), not measurements of real
conditions such as potholes. Override them with data/road_quality_config.json:
  {"surface": {"asphalt": 0.0}, "smoothness": {"bad": 0.8}, "alpha": {"balanced": 0.5, "better_roads": 2.0}}
"""
import json, os

SURFACE = {"asphalt": 0.0, "concrete": 0.0, "paved": 0.1, "concrete:plates": 0.25, "concrete:lanes": 0.25,
           "paving_stones": 0.5, "sett": 0.6, "cobblestone": 0.7, "compacted": 0.6, "fine_gravel": 0.7,
           "gravel": 0.8, "unpaved": 0.8, "dirt": 0.95, "ground": 0.95, "sand": 0.95, "mud": 1.0}
SMOOTHNESS = {"excellent": 0.0, "good": 0.1, "intermediate": 0.4, "bad": 0.75, "very_bad": 0.9,
              "horrible": 1.0, "very_horrible": 1.0, "impassable": 1.0}
# Penalty strength: edge cost = base cost * (1 + alpha * score). base cost is travel time (fastest) or length (shortest).
ALPHA = {"none": 0.0, "balanced": 0.5, "better_roads": 2.0}
PREFERENCES = tuple(ALPHA)

_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "road_quality_config.json")
if os.path.exists(_cfg):
    _c = json.load(open(_cfg))
    SURFACE.update(_c.get("surface", {})); SMOOTHNESS.update(_c.get("smoothness", {})); ALPHA.update(_c.get("alpha", {}))

def _norm(v):
    if isinstance(v, (list, tuple)): v = v[0] if v else None
    if v is None: return None
    v = str(v).strip().lower().split(";")[0]
    return v or None

def edge_quality(surface=None, smoothness=None):
    """-> (score or None, source). If both tags exist the worse (higher) score is used."""
    a, b = SMOOTHNESS.get(_norm(smoothness)), SURFACE.get(_norm(surface))
    if a is not None and b is not None: return max(a, b), "osm:smoothness+surface"
    if a is not None: return a, "osm:smoothness"
    if b is not None: return b, "osm:surface"
    return None, "unknown"

def alpha(pref):
    if pref not in ALPHA: raise ValueError(f"road_preference must be one of {PREFERENCES}")
    return float(ALPHA[pref])

def penalty_factor(score, a):
    """Multiplier applied to an edge's base cost. Unknown quality -> 1.0 (no penalty)."""
    return 1.0 if score is None else 1.0 + a * score

SOURCE_NOTE = {"osm": "Road quality comes from OpenStreetMap surface/smoothness tags. These are mapper-contributed and not verified on the ground; they are not pothole or live condition data.",
               "simulated": "Road quality in the demo network is SIMULATED, not real.",
               "unknown": "No quality tag on some roads; they were treated as neutral (not poor)."}
