"""Baseline road-risk index. Always available, needs no training, fully explainable.

Target: chance of at least one NEW report episode on a segment within the next `horizon` days.
Method: smoothed historical episode rate  rate = (episodes_in_lookback + prior_episodes) / (lookback_days + prior_days);
index = 1 - exp(-rate * horizon). It is an UNCALIBRATED heuristic, not a validated probability. A trained model (ml.py) can
supplement it only when the data supports that."""
import math
from .analysis import active_reports, group_episodes
from .config import get_config

def baseline(reps, now, cfg=None, horizon=None):
    """reps: all reports of ONE segment (any status). Returns the index and the arithmetic behind it."""
    cfg = cfg or get_config(); horizon = horizon or cfg.risk_horizon_days
    live = active_reports(reps, now, cfg); eps = 0
    by_type = {}
    for r in live: by_type.setdefault(r["issue_type"], []).append(r)
    eps = sum(len(group_episodes(v, cfg.episode_gap_hours)) for v in by_type.values())
    rate = (eps + cfg.risk_prior_episodes) / (cfg.lookback_days + cfg.risk_prior_days)
    p = 1 - math.exp(-rate * horizon)
    return {"kind": "baseline-heuristic", "index": round(p, 3), "horizon_days": horizon, "episodes_in_lookback": eps, "episode_rate_per_day": round(rate, 5),
            "explanation": [f"{eps} separate report episode(s) in the last {cfg.lookback_days} days",
                            f"smoothed rate = ({eps} + {cfg.risk_prior_episodes:g}) / ({cfg.lookback_days} + {cfg.risk_prior_days:g}) = {rate:.4f} episodes/day",
                            f"index = 1 - exp(-rate x {horizon} days) = {p:.2f}"],
            "caveat": "Uncalibrated heuristic from report history. Not a validated probability and not a forecast of any specific incident."}
