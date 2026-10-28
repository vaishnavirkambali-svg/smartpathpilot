"""Gradient Descent Lab: gradients, updates, loss tracking, stability, validation. Run: python -m pytest -q"""
import ast, json, os
import numpy as np, pytest
from fastapi.testclient import TestClient
import app as appmod, gdlab

c = TestClient(appmod.app)
rng = np.random.default_rng(0)

def test_analytic_gradients_match_finite_differences():
    for _ in range(5):
        X, y = rng.normal(size=(40, 3)), rng.normal(size=40); w, b = rng.normal(size=3), float(rng.normal())
        gw, gb = gdlab.gradients(X, y, w, b); nw, nb = gdlab.numerical_gradients(X, y, w, b)
        assert np.allclose(gw, nw, rtol=1e-6, atol=1e-8) and gb == pytest.approx(nb, rel=1e-6, abs=1e-8)

def test_gradient_formulas_by_hand():
    X, y = np.array([[1.0], [2.0]]), np.array([2.0, 3.0]); w, b = np.zeros(1), 0.0
    assert gdlab.loss(X, y, w, b) == pytest.approx(6.5)                       # (4 + 9) / 2
    gw, gb = gdlab.gradients(X, y, w, b); assert gw[0] == pytest.approx(-8.0) and gb == pytest.approx(-5.0)   # (2/2)(1*-2 + 2*-3), (2/2)(-5)
    w2, b2 = gdlab.gd_step(X, y, w, b, 0.1); assert w2[0] == pytest.approx(0.8) and b2 == pytest.approx(0.5)
    assert gdlab.loss(X, y, w2, b2) == pytest.approx(0.65)                    # residuals [-0.7, -0.9]

def test_update_rule_is_exactly_w_minus_alpha_grad():
    X, y = rng.normal(size=(25, 2)), rng.normal(size=25); w, b = rng.normal(size=2), 0.3; a = 0.07
    gw = (2 / 25) * X.T @ (X @ w + b - y); gb = (2 / 25) * np.sum(X @ w + b - y)
    w2, b2 = gdlab.gd_step(X, y, w, b, a); assert np.allclose(w2, w - a * gw) and b2 == pytest.approx(b - a * gb)

def test_worked_example_numbers():
    s = gdlab.worked_example(0.1, 2)["steps"]
    assert (s[0]["loss"], s[0]["grad_w"], s[0]["grad_b"], s[0]["w_new"], s[0]["b_new"], s[0]["loss_after"]) == pytest.approx((6.5, -8, -5, 0.8, 0.5, 0.65))
    assert (s[1]["grad_w"], s[1]["grad_b"], s[1]["w_new"], s[1]["b_new"], s[1]["loss_after"]) == pytest.approx((-2.5, -1.6, 1.05, 0.66, 0.07085))
    assert c.get("/api/gdlab/worked-example").json()["steps"][0]["w_new"] == pytest.approx(0.8)

def standardized(n=200, seed=1):
    X, y = gdlab.make_dataset(n, 2.0, seed); mu, sd = gdlab.standardize_fit(X); return (X - mu) / sd, y, mu, sd, X

def test_loss_tracking_and_monotone_decrease():
    Xs, y, *_ = standardized(); w, b, h, div = gdlab.gradient_descent(Xs, y, 0.1, 300)
    assert len(h) == 301 and not div and h[0] == pytest.approx(gdlab.loss(Xs, y, np.zeros(3), 0.0)) and h[-1] == pytest.approx(gdlab.loss(Xs, y, w, b))
    assert all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)) and h[-1] < 0.01 * h[0]

def test_converges_to_least_squares_solution():
    Xs, y, mu, sd, X = standardized(); w, b, h, _ = gdlab.gradient_descent(Xs, y, 0.1, 3000)
    sol = np.linalg.lstsq(np.column_stack([Xs, np.ones(len(Xs))]), y, rcond=None)[0]
    assert np.allclose(np.append(w, b), sol, atol=1e-6)
    wo, bo = gdlab.to_original_units(w, b, mu, sd); assert np.allclose(X @ wo + bo, Xs @ w + b)       # unit conversion keeps predictions

def test_stability_threshold():
    Xs, y, *_ = standardized(); amax = gdlab.max_stable_alpha(Xs)
    assert gdlab.gradient_descent(Xs, y, 0.9 * amax, 500)[3] is False
    w, b, h, div = gdlab.gradient_descent(Xs, y, 1.5 * amax, 500)
    assert div and np.all(np.isfinite(w)) and np.isfinite(b) and all(np.isfinite(h))               # stops early, keeps last finite values

def test_numerical_stability_extreme_scales():
    X = np.column_stack([np.linspace(0, 1e6, 50), np.linspace(0, 1e-6, 50)]); y = X[:, 0] * 1e-3
    w, b, h, div = gdlab.gradient_descent(X, y, 1.0, 100)                                           # raw, unscaled, deliberately unstable
    assert div or np.all(np.isfinite(h)); assert np.all(np.isfinite(w))

def test_reproducible_and_seed_sensitive():
    a, b = gdlab.run_experiment(0.1, 100, 100, 1.0, 7), gdlab.run_experiment(0.1, 100, 100, 1.0, 7)
    assert json.dumps(a) == json.dumps(b) and json.dumps(gdlab.run_experiment(0.1, 100, 100, 1.0, 8)) != json.dumps(a)

def test_experiment_output_complete_and_json_safe():
    r = gdlab.run_experiment(0.1, 300, 200, 2.0, 42); json.dumps(r, allow_nan=False)
    assert r["initial_loss"] > r["final_loss"] and r["history"]["loss"][0] == r["initial_loss"] and r["history"]["loss"][-1] == r["final_loss"]
    assert r["verification"]["gradient_check_max_relative_error"] < 1e-5 and r["metrics"]["test"]["r2"] > 0.95 and "SYNTHETIC" in r["dataset"]["label"]
    assert r["verification"]["max_abs_weight_gap_to_closed_form"] < 1e-6                           # GD reaches the least-squares optimum

def test_recovers_true_coefficients_exactly_without_noise():
    r = gdlab.run_experiment(0.1, 3000, 200, 0.0, 42); c_ = r["coefficients"]["original_units"]
    assert np.allclose(c_["w"], gdlab.TRUE_W, atol=1e-4) and c_["b"] == pytest.approx(gdlab.TRUE_B, abs=1e-3) and r["final_loss"] < 1e-8

def test_history_is_downsampled():
    r = gdlab.run_experiment(0.05, 5000, 200, 2.0, 1); assert len(r["history"]["loss"]) <= 1000 and r["iterations_run"] == 5000 and r["history"]["iteration"][-1] == 5000

def test_metrics_formulas():
    y = np.array([1.0, 2.0, 3.0]); assert gdlab.regression_metrics(y, y)["r2"] == 1.0
    m = gdlab.regression_metrics(y, y + 1); assert m["mse"] == 1 and m["rmse"] == 1 and m["mae"] == 1 and gdlab.regression_metrics(np.ones(3), np.ones(3))["r2"] is None

@pytest.mark.parametrize("kw", [dict(alpha=0), dict(alpha=-1), dict(alpha=float("nan")), dict(alpha=9), dict(iterations=0), dict(iterations=10 ** 6), dict(n_samples=5), dict(noise=-1), dict(seed=-1)])
def test_parameter_validation(kw):
    args = dict(alpha=0.1, iterations=100, n_samples=100, noise=1.0, seed=1); args.update(kw)
    with pytest.raises(ValueError): gdlab.run_experiment(**args)

def test_api_run_and_validation():
    r = c.post("/api/gdlab/run", json={"alpha": 0.1, "iterations": 150}).json(); assert r["iterations_run"] == 150 and r["history"]["loss"]
    for bad in ({"alpha": 0}, {"alpha": 99}, {"iterations": 0}, {"iterations": 99999}, {"n_samples": 3}, {"noise": -2}, {"alpha": "x"}): assert c.post("/api/gdlab/run", json=bad).status_code == 422
    d = c.post("/api/gdlab/run", json={"alpha": 1.0, "iterations": 100}).json(); assert d["diverged"] is True and d["stability"]["alpha_is_stable"] is False
    assert "J(w, b)" in c.get("/api/gdlab/info").json()["equations"]["loss"]

def _imports(path):
    tree = ast.parse(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)).read())
    return {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}

def test_lab_uses_only_numpy_and_is_independent():
    assert _imports("gdlab.py") == {"numpy"}                                                       # no sklearn, scipy.optimize, autograd ...
    assert _imports("gdlab_api.py") <= {"fastapi", "pydantic", "gdlab"}                           # nothing from CityPulse or the route planner
