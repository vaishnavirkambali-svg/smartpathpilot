"""All CityPulse thresholds in one place. Override any value with data/citypulse_config.json (nested dicts are merged)."""
import copy, json, os
from dataclasses import asdict, dataclass, field

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSUE_TYPES = ("pothole", "waterlogging", "damaged_surface", "obstruction", "other")
STATUSES = ("unverified", "verified", "rejected")
SOURCES = ("community_report", "demo_seed")          # demo_seed = SIMULATED data
HW_IMPACT = {"motorway": 1.0, "trunk": 1.0, "primary": 0.85, "secondary": 0.65, "tertiary": 0.4}   # road-class proxy, an assumption

@dataclass
class Config:
    # --- recurrence ---
    episode_gap_hours: float = 72.0        # reports on the same segment+type closer than this belong to one episode
    lookback_days: int = 180
    min_episodes: int = 2                  # recurring = at least this many separate episodes ...
    min_distinct_reporters: int = 2        # ... seen by at least this many different reporters
    # --- submission guards ---
    dup_window_hours: float = 24.0
    dup_radius_m: float = 50.0
    max_reports_per_hour: int = 20         # per anonymous reporter hash
    max_age_days: int = 1825
    bbox: tuple = (12.70, 13.30, 77.30, 77.90)       # lat_min, lat_max, lon_min, lon_max (Bengaluru region)
    max_snap_km_osm: float = 0.15
    max_snap_km_demo: float = 0.5
    # --- maintenance priority ---
    weights: dict = field(default_factory=lambda: {"severity": 0.30, "recency": 0.20, "recurrence": 0.20, "confidence": 0.15, "impact": 0.15})
    recency_half_life_days: float = 30.0
    recurrence_cap_episodes: int = 4
    high_priority_score: float = 60.0
    # --- risk ---
    risk_horizon_days: int = 14
    risk_prior_episodes: float = 0.1
    risk_prior_days: float = 180.0
    risk_elevated_threshold: float = 0.30
    # --- ML gate ---
    min_train_positives: int = 30
    min_test_positives: int = 15
    min_test_auc: float = 0.65
    # --- routing policy ---
    active_days: dict = field(default_factory=lambda: {"pothole": 120, "damaged_surface": 120, "waterlogging": 3, "obstruction": 7, "other": 14})
    penalty_per_severity: dict = field(default_factory=lambda: {"pothole": 0.05, "damaged_surface": 0.05, "waterlogging": 0.15, "obstruction": 0.25, "other": 0.03})
    default_routing_severity: int = 2      # used only when a verified report has no severity
    max_cost_factor: float = 2.5
    use_predicted_risk_in_routing: bool = False       # still blocked unless the ML model is validated on real data
    predicted_risk_cost: float = 0.10

_cfg = None
def get_config(reload=False):
    global _cfg
    if _cfg is None or reload:
        c = Config(); p = os.environ.get("CITYPULSE_CONFIG", os.path.join(BASE, "data", "citypulse_config.json"))
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                if hasattr(c, k): setattr(c, k, {**getattr(c, k), **v} if isinstance(v, dict) else (tuple(v) if k == "bbox" else v))
        _cfg = c
    return _cfg

def as_dict(c=None): return copy.deepcopy(asdict(c or get_config()))
