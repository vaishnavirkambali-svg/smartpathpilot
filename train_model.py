"""Train the traffic model and save models/traffic_model.joblib. Run: python train_model.py"""
import json, core
b = core.train()
print(json.dumps(b["metrics"], indent=2))
