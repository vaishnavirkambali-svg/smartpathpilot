"""API for the Gradient Descent Lab page. Independent of CityPulse and of the route planner's ML."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import gdlab

router = APIRouter(prefix="/api/gdlab", tags=["gradient-descent-lab"])

class RunRequest(BaseModel):
    alpha: float = Field(0.1, gt=0, le=gdlab.LIMITS["alpha_max"], description="learning rate")
    iterations: int = Field(200, ge=1, le=gdlab.LIMITS["iterations_max"])
    n_samples: int = Field(200, ge=gdlab.LIMITS["n_min"], le=gdlab.LIMITS["n_max"])
    noise: float = Field(2.0, ge=0, le=gdlab.LIMITS["noise_max"], description="standard deviation of the synthetic noise (minutes)")
    seed: int = Field(42, ge=0, le=10 ** 9)

@router.get("/info")
def info():
    return {"title": "Gradient Descent Lab (CIE-3)", "independent_of": ["CityPulse", "GradientBoostingRegressor used by the route planner"], "limits": gdlab.LIMITS, "features": gdlab.FEATURES,
            "equations": {"model": "y_hat_i = w . x_i + b", "loss": "J(w, b) = (1/n) * sum_i (y_hat_i - y_i)^2", "grad_w": "dJ/dw = (2/n) * X^T (X w + b 1 - y)",
                          "grad_b": "dJ/db = (2/n) * sum_i (y_hat_i - y_i)", "update_w": "w_(t+1) = w_t - alpha * dJ/dw", "update_b": "b_(t+1) = b_t - alpha * dJ/db"},
            "data": "SYNTHETIC and reproducible. y = 2.2*distance_km + 9.0*traffic_slowdown + 3.0*road_roughness + 5 + noise."}

@router.post("/run")
def run(req: RunRequest):
    try: return gdlab.run_experiment(req.alpha, req.iterations, req.n_samples, req.noise, req.seed)
    except ValueError as e: raise HTTPException(422, str(e))

@router.get("/worked-example")
def worked(): return gdlab.worked_example()
