"""Train the optional risk model from stored reports.   python -m citypulse.train [--include-demo]"""
import argparse, json
from . import api, ml

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--include-demo", action="store_true", help="also use SIMULATED demonstration reports (model is then labelled demo)")
    a = ap.parse_args(); R, cfg, now, reports, snaps, attrs, segs, _ = api.compute()
    b = ml.train(reports, snaps, attrs, now, cfg, include_demo=a.include_demo)
    print(json.dumps({k: v for k, v in b.items() if k not in ("model", "scaler")}, indent=2, default=str))
