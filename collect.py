"""Log live TomTom traffic on a schedule. Run:  python collect.py --every 30 --n 50
Needs TOMTOM_KEY and the OSM graph (python osm_ingest.py). Readings go to data/live_history.csv."""
import argparse, time
import osm_service as sv, traffic_live as tl

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=30, help="minutes between snapshots")
ap.add_argument("--n", type=int, default=50, help="roads measured per snapshot")
a = ap.parse_args()
if not tl.key(): raise SystemExit("Set TOMTOM_KEY first.")
R = sv.get_router()
if R is None: raise SystemExit("Run python osm_ingest.py first.")
samples = tl.pick_samples(R, a.n)
print(f"{len(samples)} roads x {1440 // a.every} snapshots/day = about {len(samples) * (1440 // a.every)} API calls/day. Check your TomTom quota.")
while True:
    rows = tl.snapshot(R, samples)
    print(time.strftime("%Y-%m-%d %H:%M"), len(rows), "readings saved", flush=True)
    time.sleep(a.every * 60)
