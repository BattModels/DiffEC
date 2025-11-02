import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from dataclasses import dataclass
from typing import Tuple, Optional

from sko.PSO import PSO  # scikit-opt PSO  (docs: scikit-opt.github.io)

from solver import loss_func
from utils import load_c_data, interp_c_data


# -------------------------- Config --------------------------

@dataclass
class PSOConfig:
    ftarget: Optional[float]
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]  # ((p0_min, p0_max), (p1_min, p1_max))
    popsize: int = 40
    max_iter: int = 200
    w: float = 0.7           # inertia weight
    c1: float = 1.5          # cognitive
    c2: float = 1.5          # social
    penalty: float = 1e12    # penalty if objective returns NaN/Inf or raises
    seed: Optional[int] = 0  # reproducibility (via NumPy global seed)


# ------------------------ Objective wrap ------------------------

def make_objective(c_data_interp, time_indices, space_indices, c_avg, N, L, dt, t_in_minutes, l2_lambda=5e-3, penalty=1e12):
    """Return a PSO-compatible objective: f(x)->scalar with x=[p0,p1]. NaN-safe."""
    def objective_vec(x):
        p0, p1 = float(x[0]), float(x[1])
        try:
            base = loss_func(
                p0, p1,
                c_data_interp, time_indices, space_indices,
                c_avg, N, L, dt, t_in_minutes
            )
            val = float(base + l2_lambda * (p0**2 + p1**2))
            return val if np.isfinite(val) else penalty
        except Exception:
            return penalty
    return objective_vec


# ----------------------- Live-printing PSO -----------------------

class LivePSO(PSO):
    """
    Drop-in PSO that prints best (p0, p1, loss) after each iteration.
    Also records history via scikit-opt's recorder so we can plot later.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_count = 0

    def run(self, max_iter=None, verbose=True):
        if max_iter is None:
            max_iter = self.max_iter
        if not getattr(self, "record_mode", False):
            self.record_mode = True  # keep per-iteration history

        t0 = time.perf_counter()
        for it in range(max_iter):
            self.update_V()     # velocities
            self.update_X()     # positions (respecting lb/ub)
            self.cal_y()        # evaluate objective
            self.eval_count += self.pop
            self.update_pbest()
            self.update_gbest()
            if self.record_mode:
                self.recorder()

            bx = np.array(self.gbest_x, dtype=float).ravel()
            by = np.asarray(self.gbest_y).reshape(-1)[0]

            if verbose:
                print(f"[iter {it+1:03d}] p0={bx[0]:.6g}, p1={bx[1]:.6g}, loss={by:.6g}")

            ftarget = getattr(self, "ftarget", None)
            if ftarget is not None and by <= ftarget:
                if verbose:
                    print(f"Reached target loss {ftarget} at iteration {it+1}. Stopping early.")
                break

        self.runtime_ = time.perf_counter() - t0
        return np.array(self.gbest_x).ravel(), np.asarray(self.gbest_y).reshape(-1)[0]


# ----------------------- Run PSO on the problem -----------------------

def run_pso(cfg: PSOConfig, *, plot: bool = True, verbose: bool = True):
    # Reproducibility (scikit-opt uses NumPy RNG)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    # Problem setup
    c_avg = 1.87
    N = 50
    L = 0.3       # cm
    dt = 0.1      # s

    c_data = load_c_data()
    c_data_interp, x_interp, time_indices, space_indices = interp_c_data(c_data, N, L)

    num_train = 9
    t_in_minutes = time_indices[num_train - 1].item() / 600

    # Objective
    objective = make_objective(
        c_data_interp[:num_train],
        time_indices[:num_train],
        space_indices,
        c_avg, N, L, dt, t_in_minutes,
        l2_lambda=5e-3,
        penalty=cfg.penalty
    )

    # Bounds
    (p0_min, p0_max), (p1_min, p1_max) = cfg.bounds
    lb = [p0_min, p1_min]
    ub = [p0_max, p1_max]

    # Build PSO
    pso = LivePSO(
        func=objective,
        n_dim=2,
        pop=cfg.popsize,
        max_iter=cfg.max_iter,
        lb=lb,
        ub=ub,
        w=cfg.w,
        c1=cfg.c1,
        c2=cfg.c2,
    )

    # Attach target to the instance for early stop
    pso.ftarget = cfg.ftarget

    # Results
    best_x, best_y = pso.run(verbose=verbose)
    best_p0, best_p1 = float(best_x[0]), float(best_x[1])

    if verbose:
        print("\n== PSO finished ==")
        print(f"Seed: {cfg.seed}")
        print(f"Best (p0, p1): ({best_p0:.8f}, {best_p1:.8f})")
        print(f"Best loss    : {best_y:.8e}")
        print("Objective evaluations:", pso.eval_count)
        print(f"Max iterations   : {cfg.max_iter}")
        print(f"Total time   : {getattr(pso, 'runtime_', float('nan')):.3f}s")

    # Optional convergence plot
    Y = np.asarray(pso.record_value.get("Y", None), dtype=float)
    Y = np.squeeze(Y)
    y_iter_best = np.min(Y, axis=1)
    # Best-so-far (monotone nonincreasing)
    y_cum_best = np.minimum.accumulate(y_iter_best)
    
    if plot and Y.size:
        plt.figure()
        plt.plot(y_cum_best, lw=1.6)
        plt.xlabel("Iteration")
        plt.ylabel("Best-so-far loss")
        plt.title("PSO convergence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("pso_convergence.png")

    return {
        "p0": best_p0,
        "p1": best_p1,
        "loss": float(best_y),
        "time_sec": getattr(pso, "runtime_", float("nan")),
        "hit_target": (cfg.ftarget is not None) and (best_y <= cfg.ftarget),
    }


# ------------------------------ Main ------------------------------

if __name__ == "__main__":
    cfg = PSOConfig(
        ftarget=0.02676,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        popsize=40,
        max_iter=200,
        w=0.7, c1=1.5, c2=1.5,
        penalty=1e12,
        seed=0,
    )
    res = run_pso(cfg, plot=True, verbose=True)

    print("\nBest parameters and loss:")
    print(f"p0 = {res['p0']:.10f}, p1 = {res['p1']:.10f}, loss = {res['loss']:.10e}")
    print("Hit target:", res["hit_target"])
