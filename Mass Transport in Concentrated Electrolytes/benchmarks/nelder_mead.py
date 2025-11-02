import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from scipy.optimize import minimize, Bounds

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from solver import loss_func
from utils import load_c_data, interp_c_data


# ----------------------- Config -----------------------

@dataclass
class NMConfig:
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]   # ((p0_min, p0_max), (p1_min, p1_max))
    x0: Optional[Tuple[float, float]] = None                  # initial guess in parameter space
    target_loss: Optional[float] = 0.02676                    # early stop at/under this loss; set None to disable
    maxiter: int = 500
    maxfev: Optional[int] = None
    xatol: float = 1e-8
    fatol: float = 1e-10
    adaptive: bool = True
    penalty: float = 1e12                                     # used if solver returns NaN/Inf
    l2_lambda: float = 5e-3                                   # regularization


# -------------------- Objective in parameter space --------------------

def make_objective(c_data_interp, time_indices, space_indices,
                   c_avg, N, L, dt, t_in_minutes,
                   l2_lambda=5e-3, penalty=1e12,
                   counters: Optional[Dict[str, Any]] = None):
    """
    Returns f(p): objective in PARAMETER space p=[p0, p1].
    - Calls the loss_func and adds L2 regularization
    - Returns 'penalty' on any failure / non-finite
    - Tracks best-so-far in `counters`
    """
    if counters is None:
        counters = {}
    counters.setdefault("n_eval", 0)
    counters.setdefault("best_f", np.inf)
    counters.setdefault("best_p", None)

    def f_p(p: np.ndarray) -> float:
        p0, p1 = float(p[0]), float(p[1])
        counters["n_eval"] += 1
        try:
            base = loss_func(
                p0, p1,
                c_data_interp, time_indices, space_indices,
                c_avg, N, L, dt, t_in_minutes
            )
            f = float(base + l2_lambda * (p0**2 + p1**2))
            if not np.isfinite(f):
                f = penalty
        except Exception:
            f = penalty
        if f < counters["best_f"]:
            counters["best_f"] = f
            counters["best_p"] = (p0, p1)
        return f

    return f_p


# -------------------- Nelder–Mead runner --------------------

def run_nelder_mead(cfg: NMConfig, *, verbose: True):
    c_avg = 1.87
    N = 50
    L = 0.3
    dt = 0.1

    c_data = load_c_data()
    c_data_interp, x_interp, time_indices, space_indices = interp_c_data(c_data, N, L)

    num_train = 9
    t_in_minutes = time_indices[num_train - 1].item() / 600

    # Objective (parameter space)
    counters: Dict[str, Any] = {}
    f_p = make_objective(
        c_data_interp[:num_train], time_indices[:num_train], space_indices,
        c_avg, N, L, dt, t_in_minutes,
        l2_lambda=cfg.l2_lambda, penalty=cfg.penalty,
        counters=counters
    )

    # Bounds & x0
    (p0_min, p0_max), (p1_min, p1_max) = cfg.bounds
    bnds = Bounds([p0_min, p1_min], [p0_max, p1_max])
    if cfg.x0 is None:
        x0 = np.array([(p0_min + p0_max)/2, (p1_min + p1_max)/2], float)
    else:
        x0 = np.clip(np.array(cfg.x0, float), bnds.lb, bnds.ub)

    # Callback: print best-so-far each iter; stop early at target
    stopped = {"early": False}
    it = {"k": 0}
    t0 = time.perf_counter()

    def cb(xk: np.ndarray):
        it["k"] += 1
        best_p = counters.get("best_p", (np.nan, np.nan))
        best_f = counters.get("best_f", np.inf)
        if verbose:
            print(f"[iter {it['k']:03d}] p0={best_p[0]:.6g}, p1={best_p[1]:.6g}, loss={best_f:.6g}")
        if cfg.target_loss is not None and best_f <= cfg.target_loss:
            stopped["early"] = True
            raise StopIteration("Target loss reached")

    # Run Nelder–Mead with bounds (SciPy clips simplex to the box)
    options = dict(maxiter=cfg.maxiter, maxfev=cfg.maxfev, xatol=cfg.xatol,
                   fatol=cfg.fatol, adaptive=cfg.adaptive, return_all=False, disp=False)

    res = minimize(f_p, x0, method="Nelder-Mead", bounds=bnds, callback=cb, options=options)

    # Decide the stop reason and iteration count
    if stopped["early"]:
        stop_reason = "target"
        nit = it["k"]
    else:
        stop_reason = getattr(res, "message", "completed")
        nit = getattr(res, "nit", it["k"])

    wall = time.perf_counter() - t0

    best_p0, best_p1 = counters.get("best_p", (np.nan, np.nan))
    best_f = counters.get("best_f", np.inf)

    if verbose:
        print("\n== Nelder–Mead finished ==")
        print(f"Initial guess: {cfg.x0}")
        print(f"Reason     : {stop_reason}")
        print(f"Best (p0,p1): ({best_p0:.8f}, {best_p1:.8f})")
        print(f"Best loss  : {best_f:.8e}")
        print(f"Iters/Evals: {nit} / {counters['n_eval']}")
        print(f"Wall time  : {wall:.3f}s")

    return dict(
        p0=float(best_p0), p1=float(best_p1), loss=float(best_f),
        iterations=int(getattr(res, "nit", it["k"])),
        evaluations=int(counters["n_eval"]),
        time_sec=float(wall),
        stop=stop_reason,
    )


# ------------------------------ Main ------------------------------

if __name__ == "__main__":
    cfg = NMConfig(
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        x0=(-1.0, -2.0),
        target_loss=0.02676,
        maxiter=500,
        xatol=1e-8,
        fatol=1e-10,
        adaptive=True,
        penalty=1e12,
        l2_lambda=5e-3,
    )
    result = run_nelder_mead(cfg, verbose=True)
    
    print("\nBest parameters and loss:")
    print(f"p0 = {result['p0']:.10f}, p1 = {result['p1']:.10f}, loss = {result['loss']:.10e}")
