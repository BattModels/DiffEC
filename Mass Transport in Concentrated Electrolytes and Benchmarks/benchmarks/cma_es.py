import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cma  # pycma

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from solver import loss_func
from utils import load_c_data, interp_c_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ----------------------- Config -----------------------

@dataclass
class CMAConfig:
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]  # ((p0_min, p0_max), (p1_min, p1_max))
    x0: Optional[Tuple[float, float]] = None   # initial mean; default = box center
    sigma0: Optional[float] = None             # initial global step size; default = 0.3 * box_diag
    popsize: Optional[int] = None              # default: cma's recommendation
    ftarget: Optional[float] = None            # stop when loss <= ftarget
    maxiter: Optional[int] = 200               # iteration cap
    maxfevals: Optional[int] = None            # evaluation cap (total objective calls)
    tolx: Optional[float] = 1e-8               # stop when step size becomes tiny
    tolfun: Optional[float] = 1e-12            # stop when f value changes become tiny
    tolstagnation: Optional[int] = 50          # stop after so many iters with little progress
    seed: int = 0 
    penalty: float = 1e12                      # NaN/Inf handling, large penalty if objective is invalid
    l2_lambda: float = 5e-3                    # lambda for L2 penalty

    # plotting options
    out_dir: str = "cma_frames"                # where to save per-iteration pngs
    plot_each_iter: bool = True
    plot_mahal_radius: float = 1.0             # 1.0 => 1σ ellipse (Mahalanobis radius=1). Use np.sqrt(2.279) ≈ 1.5096 for ~68% mass in 2D.


# ----------------------- Optimizer -----------------------

class CMAESOptimization:
    def __init__(self, cfg: CMAConfig, objective_callable):
        self.cfg = cfg
        self.objective = objective_callable  # expects (p0, p1) -> float
        self.history: List[Tuple[float, float, float]] = []  # (p0, p1, loss)

        (p0_min, p0_max), (p1_min, p1_max) = cfg.bounds
        if cfg.x0 is None:
            self.x0 = np.array([(p0_min + p0_max) / 2.0, (p1_min + p1_max) / 2.0], dtype=float)
        else:
            self.x0 = np.array(cfg.x0, dtype=float)

        # Default sigma0 ~ 30% of diagonal of the box
        if cfg.sigma0 is None:
            box_diag = np.hypot(p0_max - p0_min, p1_max - p1_min)
            self.sigma0 = 0.3 * box_diag
        else:
            self.sigma0 = float(cfg.sigma0)

        self.lower = np.array([p0_min, p1_min], dtype=float)
        self.upper = np.array([p0_max, p1_max], dtype=float)

        if self.cfg.plot_each_iter:
            os.makedirs(self.cfg.out_dir, exist_ok=True)

    def _safe_objective(self, x: np.ndarray) -> float:
        """Wrap the solver loss; return a large finite penalty on failure."""
        p0, p1 = float(x[0]), float(x[1])
        try:
            val = float(self.objective(p0, p1))
            if not np.isfinite(val):
                return self.cfg.penalty
            return val
        except Exception:
            return self.cfg.penalty

    # Covariance from pycma state
    def _current_covariance(self, es) -> np.ndarray:
        """Return Σ = σ^2 * C as a 2x2 covariance for plotting."""
        # Try direct 'C'; otherwise fall back to eigen parts
        try:
            C = np.asarray(es.C)
            if C.ndim != 2:
                raise ValueError
        except Exception:
            B = np.asarray(es.B)
            D = np.asarray(es.D)  # vector of sqrt(eigenvalues of C)
            if D.ndim == 1:
                D2 = np.diag(D**2)
            else:
                D2 = D**2
            C = B @ D2 @ B.T
        return (es.sigma ** 2) * C

    # Ellipse points for a given Mahalanobis radius r
    def _ellipse_points(self, mean: np.ndarray, cov: np.ndarray, r: float = 1.0, n: int = 256) -> np.ndarray:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-16, None)
        radii = np.sqrt(vals) * r  # std along principal axes times r
        theta = np.linspace(0, 2*np.pi, n)
        circle = np.vstack([np.cos(theta), np.sin(theta)])       # (2,n)
        ellipse = (vecs @ np.diag(radii)) @ circle               # (2,n)
        return (ellipse.T + mean.reshape(1, 2))                  # (n,2)

    # Per-iteration plot
    def _plot_iteration(self, k: int, es, X: List[np.ndarray], f_vals: List[float]):
        X = np.asarray(X, dtype=float)
        mean = np.asarray(es.mean, dtype=float)
        cov = self._current_covariance(es)
        E = self._ellipse_points(mean, cov, r=self.cfg.plot_mahal_radius)

        (p0_min, p0_max), (p1_min, p1_max) = self.cfg.bounds

        fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
        ax.scatter(X[:, 0], X[:, 1], s=28, alpha=0.9, label=f"samples (n={len(X)})", edgecolors="none")
        ax.plot(mean[0], mean[1], marker="x", markersize=8, label="mean")
        ax.plot(E[:, 0], E[:, 1], lw=2, label=f"{self.cfg.plot_mahal_radius}σ ellipse")

        # Bounds box
        ax.add_patch(Rectangle((p0_min, p1_min), p0_max - p0_min, p1_max - p1_min,
                               fill=False, linestyle="--", linewidth=1))
        ax.set_xlim(p0_min, p0_max)
        ax.set_ylim(p1_min, p1_max)
        ax.set_aspect("equal", adjustable="box")

        bestf = np.min(f_vals) if len(f_vals) else np.nan
        ax.set_title(f"iter {k} | σ={es.sigma:.3g} | best f={bestf:.3g}")
        ax.set_xlabel("p0")
        ax.set_ylabel("p1")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        fname = os.path.join(self.cfg.out_dir, f"iter_{k:04d}.png")
        fig.savefig(fname)
        plt.close(fig)

    def run(self, verbose: bool = True):
        # CMA options
        opts = {
            "bounds": [self.lower.tolist(), self.upper.tolist()],
            "seed": self.cfg.seed,
            "maxiter": self.cfg.maxiter,
            "maxfevals": self.cfg.maxfevals,
            "verb_disp": 1 if verbose else 0,
        }
        if self.cfg.popsize is not None:        opts["popsize"] = self.cfg.popsize
        if self.cfg.ftarget is not None:        opts["ftarget"] = self.cfg.ftarget
        if self.cfg.tolx is not None:           opts["tolx"] = self.cfg.tolx
        if self.cfg.tolfun is not None:         opts["tolfun"] = self.cfg.tolfun
        if self.cfg.tolstagnation is not None:  opts["tolstagnation"] = self.cfg.tolstagnation

        es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, opts)

        t0 = time.perf_counter()
        eval_counter = 0
        k_iter = 0

        while not es.stop():
            # Ask for a population of candidates
            X = es.ask()
            # Evaluate all candidates
            f_vals = []
            for x in X:
                f = self._safe_objective(np.asarray(x, dtype=float))
                f_vals.append(f)
                eval_counter += 1
                self.history.append((float(x[0]), float(x[1]), float(f)))
                if verbose:
                    print(f"[{eval_counter:04d}] p0={x[0]:.6g}, p1={x[1]:.6g}, loss={f:.6g}")

                # Plot BEFORE updating (visualizes the distribution that generated X)
                if self.cfg.plot_each_iter:
                    try:
                        self._plot_iteration(k_iter, es, X, f_vals)
                    except Exception as e:
                        if verbose:
                            print(f"(plot warning) iteration {k_iter}: {e}")

            # Tell CMA-ES the fitness values
            es.tell(X, f_vals)
            if verbose:
                es.disp()
            k_iter += 1

        total_time = time.perf_counter() - t0
        res = es.result  # cma.evolution_strategy.CMAEvolutionStrategyResult

        best_p0, best_p1 = float(res.xbest[0]), float(res.xbest[1])
        best_loss = float(res.fbest)

        if verbose:
            print("\n== CMA-ES finished ==")
            print(f"Initial guess for sample mean: {cfg.x0}")
            print("Stop conditions:", es.stop())
            print(f"Best (p0, p1): ({best_p0:.8f}, {best_p1:.8f})")
            print(f"Best loss    : {best_loss:.8e}")
            print(f"Evaluations  : {res.evaluations}")
            print(f"Iterations   : {res.iterations}")
            print(f"Total time   : {total_time:.3f}s")

        return {
            "p0": best_p0,
            "p1": best_p1,
            "loss": best_loss,
            "evaluations": int(res.evaluations),
            "iterations": int(res.iterations),
            "stop": es.stop(),
            "time_sec": total_time,
            "history": np.array(self.history, dtype=float),  # shape (n_evals, 3)
        }

if __name__ == "__main__":
    # Problem setup
    c_avg = 1.87
    N = 50
    L = 0.3                 # cm
    dt = 0.1                # s

    # Load experimental data
    c_data = load_c_data()
    # Interpolate experimental data
    c_data_interp, x_interp, time_indices, space_indices = interp_c_data(c_data, N, L)

    # Use first 'num_train' snapshots
    num_train = 9
    t_in_minutes = time_indices[num_train - 1].item() / 600

    # Objective wrapper
    def objective(p0: float, p1: float) -> float:
        lbd = 0.005
        base = loss_func(
            p0, p1,
            c_data_interp[:num_train], time_indices[:num_train], space_indices,
            c_avg, N, L, dt, t_in_minutes
        )
        return float(base + lbd * (p0**2 + p1**2))

    # CMA-ES configuration
    cfg = CMAConfig(
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        x0=(-3.0, -3.0),            # default: box center
        sigma0=None,                # default: 0.3 * box diagonal
        popsize=None,               # use CMA default
        ftarget=0.02676,            # target loss value
        maxiter=200,
        maxfevals=None,
        tolx=1e-6,
        tolfun=1e-6,
        tolstagnation=50,
        seed=0,
        penalty=1e12,
        l2_lambda=5e-3,
        out_dir="cma_frames",
        plot_each_iter=True,
        plot_mahal_radius=1.0
    )

    opt = CMAESOptimization(cfg, objective)
    result = opt.run(verbose=True)
    
    print("\nBest parameters and loss:")
    print(f"p0 = {result['p0']:.10f}, p1 = {result['p1']:.10f}, loss = {result['loss']:.10e}")
    print(f"Stopped after {result['evaluations']} evals, {result['iterations']} iterations. Reason(s): {result['stop']}")
