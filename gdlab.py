"""Gradient Descent Lab (CIE-3 Calculus). Explicit gradient descent in NumPy for linear regression with a mean-squared-error loss.
No optimizer library is used for the training loop (no SGDRegressor, no scipy.optimize, no autograd).
Independent of CityPulse and of the GradientBoostingRegressor used by the route planner.

Model            y_hat_i = w . x_i + b
Objective        J(w, b) = (1/n) sum_i (y_hat_i - y_i)^2
Gradients        dJ/dw = (2/n) X^T (X w + b 1 - y)          dJ/db = (2/n) sum_i (y_hat_i - y_i)
Updates          w <- w - alpha dJ/dw                         b <- b - alpha dJ/db
Data: SYNTHETIC and reproducible (seeded). No real measured travel-time data exists in this project."""
import numpy as np

FEATURES = ["distance_km", "traffic_slowdown", "road_roughness"]
TRUE_W = np.array([2.2, 9.0, 3.0]); TRUE_B = 5.0          # minutes per km, per unit slowdown, per unit roughness; base minutes
LIMITS = {"alpha_max": 5.0, "iterations_max": 5000, "n_min": 30, "n_max": 2000, "noise_max": 20.0}

def make_dataset(n=200, noise=2.0, seed=42):
    """SYNTHETIC travel times: y = TRUE_W . x + TRUE_B + Normal(0, noise)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(2, 25, n), rng.uniform(1.0, 2.6, n), rng.uniform(0, 1, n)])
    return X, X @ TRUE_W + TRUE_B + rng.normal(0, noise, n)

def split(X, y, test_frac=0.2, seed=42):
    idx = np.random.default_rng(seed).permutation(len(y)); k = int(round(len(y) * (1 - test_frac)))
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]

def loss(X, y, w, b):
    r = X @ w + b - y
    return float(np.mean(r ** 2))

def gradients(X, y, w, b):
    """Analytic gradients of the MSE loss: (dJ/dw, dJ/db)."""
    n = len(y); r = X @ w + b - y
    return (2.0 / n) * (X.T @ r), (2.0 / n) * float(r.sum())

def gd_step(X, y, w, b, alpha):
    gw, gb = gradients(X, y, w, b)
    return w - alpha * gw, b - alpha * gb

def numerical_gradients(X, y, w, b, eps=1e-6):
    """Central finite differences, used only to check the analytic gradients."""
    gw = np.zeros_like(w)
    for j in range(len(w)):
        d = np.zeros_like(w); d[j] = eps
        gw[j] = (loss(X, y, w + d, b) - loss(X, y, w - d, b)) / (2 * eps)
    return gw, (loss(X, y, w, b + eps) - loss(X, y, w, b - eps)) / (2 * eps)

def gradient_descent(X, y, alpha, iterations, w0=None, b0=0.0, blow_up=1e12):
    """Batch gradient descent. Returns (w, b, history, diverged). history[t] is the loss before update t+1 (history[0] = initial loss).
    Stops early and keeps the last finite parameters if the loss becomes non-finite or explodes."""
    if not alpha > 0 or not np.isfinite(alpha): raise ValueError("alpha must be a positive finite number")
    if iterations < 1: raise ValueError("iterations must be at least 1")
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, float); b = float(b0); hist = [loss(X, y, w, b)]; diverged = False
    for _ in range(int(iterations)):
        w2, b2 = gd_step(X, y, w, b, alpha); L = loss(X, y, w2, b2)
        if not np.isfinite(L) or L > blow_up * (1.0 + hist[0]): diverged = True; break
        w, b = w2, b2; hist.append(L)
    return w, b, hist, diverged

def standardize_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0); return mu, np.where(sd == 0, 1.0, sd)

def to_original_units(ws, bs, mu, sd):
    """w_s . (x - mu)/sd + b_s  =  (w_s/sd) . x + (b_s - sum(w_s mu / sd))"""
    return ws / sd, float(bs - np.sum(ws * mu / sd))

def regression_metrics(y, yhat):
    r = yhat - y; ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"mse": float(np.mean(r ** 2)), "rmse": float(np.sqrt(np.mean(r ** 2))), "mae": float(np.mean(np.abs(r))), "r2": None if ss_tot == 0 else float(1 - np.sum(r ** 2) / ss_tot), "n": int(len(y))}

def max_stable_alpha(Xs):
    """Largest alpha for which gradient descent converges on this quadratic loss: 2 / lambda_max(Hessian), Hessian = (2/n) A^T A with A = [X, 1]."""
    A = np.column_stack([Xs, np.ones(len(Xs))]); return float(2.0 / np.linalg.eigvalsh((2.0 / len(A)) * (A.T @ A)).max())

def validate_params(alpha, iterations, n_samples, noise, seed):
    L = LIMITS
    if not (isinstance(alpha, (int, float)) and np.isfinite(alpha) and 0 < alpha <= L["alpha_max"]): raise ValueError(f"alpha must be in (0, {L['alpha_max']}]")
    if not (isinstance(iterations, int) and 1 <= iterations <= L["iterations_max"]): raise ValueError(f"iterations must be an integer from 1 to {L['iterations_max']}")
    if not (isinstance(n_samples, int) and L["n_min"] <= n_samples <= L["n_max"]): raise ValueError(f"n_samples must be an integer from {L['n_min']} to {L['n_max']}")
    if not (isinstance(noise, (int, float)) and np.isfinite(noise) and 0 <= noise <= L["noise_max"]): raise ValueError(f"noise must be in [0, {L['noise_max']}]")
    if not (isinstance(seed, int) and 0 <= seed <= 10 ** 9): raise ValueError("seed must be an integer from 0 to 1000000000")

def run_experiment(alpha=0.1, iterations=200, n_samples=200, noise=2.0, seed=42):
    validate_params(alpha, iterations, n_samples, noise, seed)
    X, y = make_dataset(n_samples, noise, seed); Xtr, ytr, Xte, yte = split(X, y, 0.2, seed)
    mu, sd = standardize_fit(Xtr); Xs = (Xtr - mu) / sd
    ws, bs, hist, diverged = gradient_descent(Xs, ytr, alpha, iterations)
    w, b = to_original_units(ws, bs, mu, sd)
    A = np.column_stack([Xs, np.ones(len(Xs))]); sol = np.linalg.lstsq(A, ytr, rcond=None)[0]      # closed-form reference, for verification only
    w_ols, b_ols = to_original_units(sol[:-1], float(sol[-1]), mu, sd)
    rng = np.random.default_rng(seed + 1); wc, bc = rng.normal(0, 0.5, Xs.shape[1]), float(rng.normal(0, 0.5))
    ga, gba = gradients(Xs, ytr, wc, bc); gn, gbn = numerical_gradients(Xs, ytr, wc, bc)
    rel = float(max(np.max(np.abs(ga - gn) / np.maximum(1e-12, np.abs(gn))), abs(gba - gbn) / max(1e-12, abs(gbn))))
    gw, gb = gradients(Xs, ytr, ws, bs)
    idx = np.unique(np.linspace(0, len(hist) - 1, min(1000, len(hist))).astype(int))
    amax = max_stable_alpha(Xs); ref_loss = loss(Xs, ytr, sol[:-1], float(sol[-1]))
    return {"parameters": {"alpha": alpha, "iterations": iterations, "n_samples": n_samples, "noise": noise, "seed": seed},
            "dataset": {"label": "SYNTHETIC data generated from a known linear rule plus Gaussian noise (reproducible by seed). Not measured travel times.", "features": FEATURES,
                        "n_train": len(ytr), "n_test": len(yte), "generator": {"true_w": TRUE_W.tolist(), "true_b": TRUE_B, "noise_sd": noise}, "standardized_for_training": True},
            "history": {"iteration": idx.tolist(), "loss": [hist[i] for i in idx]}, "iterations_run": len(hist) - 1, "diverged": diverged,
            "initial_loss": hist[0], "final_loss": hist[-1], "loss_reduction_pct": float(100 * (hist[0] - hist[-1]) / hist[0]) if hist[0] > 0 else 0.0,
            "final_gradient_norm": float(np.sqrt(np.sum(gw ** 2) + gb ** 2)),
            "coefficients": {"features": FEATURES, "standardized": {"w": ws.tolist(), "b": bs}, "original_units": {"w": w.tolist(), "b": b},
                             "true": {"w": TRUE_W.tolist(), "b": TRUE_B}, "closed_form_reference": {"w": w_ols.tolist(), "b": b_ols}},
            "metrics": {"train": regression_metrics(ytr, Xtr @ w + b), "test": regression_metrics(yte, Xte @ w + b)},
            "verification": {"gradient_check_max_relative_error": rel, "closed_form_loss": ref_loss, "loss_gap_to_closed_form": hist[-1] - ref_loss,
                             "max_abs_weight_gap_to_closed_form": float(np.max(np.abs(np.append(w - w_ols, b - b_ols)))), "note": "The closed-form least-squares solution is a reference only; it is not used to train."},
            "stability": {"max_stable_alpha": amax, "alpha_is_stable": alpha < amax, "note": "Gradient descent on this quadratic loss converges only if alpha < 2 / lambda_max(Hessian)."}}

def worked_example(alpha=0.1, steps=2):
    """Tiny two-point example (x = [1, 2], y = [2, 3], w = b = 0) computed with the same functions as the lab."""
    X, y = np.array([[1.0], [2.0]]), np.array([2.0, 3.0]); w, b = np.zeros(1), 0.0; out = []
    for t in range(steps):
        yhat = X @ w + b; r = yhat - y; gw, gb = gradients(X, y, w, b); w2, b2 = gd_step(X, y, w, b, alpha)
        out.append({"step": t + 1, "w": float(w[0]), "b": float(b), "y_hat": yhat.tolist(), "residuals": r.tolist(), "loss": loss(X, y, w, b),
                    "grad_w": float(gw[0]), "grad_b": gb, "w_new": float(w2[0]), "b_new": float(b2), "loss_after": loss(X, y, w2, b2)})
        w, b = w2, b2
    return {"x": [1.0, 2.0], "y": [2.0, 3.0], "alpha": alpha, "steps": out}
