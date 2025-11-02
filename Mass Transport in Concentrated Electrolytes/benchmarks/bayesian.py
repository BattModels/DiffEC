import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time

import jax
import jax.numpy as jnp
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from solver import loss_func
from utils import load_c_data, interp_c_data


@dataclass
class BOConfig:
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]  # ((p0_min, p0_max), (p1_min, p1_max))
    target_loss: float = 0.02676
    n_init: int = 10            # number of initial random samples (recorded even if invalid)
    n_iter: int = 30            # number of BO iterations
    n_samples: int = 500        # number of random candidates per iteration
    xi: float = 0.5             # ξ in the acquisition (Eq. [8]), 0..1
    random_state: int = 0
    nan_penalty: Optional[float] = None   # if None, computed from target_loss as max(100*target_loss, 1.0)
    nan_alpha: float = 1e-2               # GP per-point noise for penalized (invalid) observations
    valid_alpha: float = 1e-10            # GP per-point noise for valid observations


class BayesianOptimization:
    """
    Implements the BO loop from Mistry et al. (PCCP 2024), using:
      - GP surrogate on Δ(p0, p1)
      - Acquisition:  A = (σ / RMS(σ))^ξ * (min(Δ_k) / μ)^((1-ξ))  (Eq. [8])

    The invalid/NaN objective evaluations are not discarded.
    Instead, it records them as (X, y = nan_penalty) with a larger per-point
    noise (alpha) so the GP learns that region is bad without overfitting.
    """

    def __init__(self, cfg: BOConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_state)

        # Cache bounds for scaling
        (p0_min, p0_max), (p1_min, p1_max) = self.cfg.bounds
        self.lb = np.array([p0_min, p1_min], dtype=float)
        self.ub = np.array([p0_max, p1_max], dtype=float)
        self.span = self.ub - self.lb

        # Robust kernel: (C * Matern) + White noise
        kernel = (
            ConstantKernel(1.0, (1e-6, 1e6))
            * Matern(nu=2.5, length_scale=[1.0, 1.0])
            + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-16, 1e-2))
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=cfg.random_state,
            alpha=1e-10  # tiny jitter for stability, will be overridden per-fit with per-point alphas
        )

        self.X = None  # (n,2) array of [p0, p1]
        self.y = None  # (n,)   array of Δ
        self._alphas = None  # (n,) per-point alphas

        # Penalty used when objective returns NaN/inf
        if self.cfg.nan_penalty is None:
            self.nan_penalty = max(100.0 * self.cfg.target_loss, 1.0)
        else:
            self.nan_penalty = float(self.cfg.nan_penalty)

    # ----------------- utilities -----------------

    def _to_unit(self, X: np.ndarray) -> np.ndarray:
        """Scale parameters to [0,1]^2 for the GP."""
        return (X - self.lb) / self.span

    def _random_in_bounds(self, n: int) -> np.ndarray:
        (p0_min, p0_max), (p1_min, p1_max) = self.cfg.bounds
        P0 = self.rng.uniform(p0_min, p0_max, size=n)
        P1 = self.rng.uniform(p1_min, p1_max, size=n)
        return np.column_stack([P0, P1])

    def _eval_with_penalty(self, objective: Callable[[float, float], float], p0: float, p1: float):
        """
        Evaluate the objective. If it returns NaN/inf, replace with nan_penalty.
        Returns (y_value, is_valid_bool).
        """
        y = objective(float(p0), float(p1))
        valid = np.isfinite(y)
        return (float(y) if valid else self.nan_penalty), bool(valid)

    def _acquisition(self, mu: np.ndarray, sigma: np.ndarray, best_y: float) -> np.ndarray:
        """
        A = (σ / RMS(σ))^ξ * (best_y / μ)^((1-ξ))
        Guard: μ must be positive (Δ > 0). If μ≤0 from GP artifacts, clamp.
        """
        mu_safe = np.maximum(mu, 1e-16)
        rms_sigma = np.sqrt(np.mean(sigma**2)) + 1e-16
        A = (sigma / rms_sigma) ** self.cfg.xi * (best_y / mu_safe) ** (1.0 - self.cfg.xi)
        return A

    # ----------------- main loop -----------------

    def run(self, objective: Callable[[float, float], float]):
        # Initial random design (record all points; penalize invalid)
        self.X, self.y, alphas = [], [], []
        t0 = time.perf_counter()
        cands = self._random_in_bounds(self.cfg.n_init)
        for cand in cands:
            y_val, ok = self._eval_with_penalty(objective, cand[0], cand[1])
            self.X.append(cand)
            self.y.append(y_val)
            alphas.append(self.cfg.valid_alpha if ok else self.cfg.nan_alpha)
        t1 = time.perf_counter()
        print("Initial guesses time:", t1 - t0)

        self.X = np.vstack(self.X)
        self.y = np.array(self.y, dtype=float)
        self._alphas = np.array(alphas, dtype=float)

        # BO iterations
        for k in range(self.cfg.n_iter):
            # Early stopping
            best_y = float(np.min(self.y))
            if best_y < self.cfg.target_loss:
                print(f"Converged at iteration {self.cfg.n_init + k}, loss = {best_y}")
                break

            iter_start_time = time.perf_counter()

            # Fit GP with per-point alphas
            self.gp.alpha = self._alphas
            self.gp.fit(self._to_unit(self.X), self.y)

            # Sample candidates, compute acquisition, pick the best
            C = self._random_in_bounds(self.cfg.n_samples)
            mu, sigma = self.gp.predict(self._to_unit(C), return_std=True)
            A = self._acquisition(mu, sigma, best_y=best_y)
            j = int(np.argmax(A))
            p0_new, p1_new = C[j]

            # Evaluate once; penalize if invalid
            y_new, ok = self._eval_with_penalty(objective, p0_new, p1_new)
            alpha_new = self.cfg.valid_alpha if ok else self.cfg.nan_alpha

            # Record
            self.X = np.vstack([self.X, [p0_new, p1_new]])
            self.y = np.append(self.y, y_new)
            self._alphas = np.append(self._alphas, alpha_new)

            iter_end_time = time.perf_counter()
            iter_time = iter_end_time - iter_start_time

            print(
                f"[Iteration {k+1:03d}] p0={p0_new:.6g}, p1={p1_new:.6g}, "
                f"loss={y_new:.6g} ({'valid' if ok else 'penalized'}), iter_time={iter_time:.6g}"
            )

        best_idx = int(np.argmin(self.y))
        best = self.X[best_idx]
        return {
            "p0": float(best[0]),
            "p1": float(best[1]),
            "loss": float(self.y[best_idx]),
            "X": self.X,
            "y": self.y,
        }


if __name__ == "__main__":
    # Simulation parameters
    c_avg = 1.87
    N = 50
    L = 0.3                 # Length in cm
    dt = 0.1                # Time step in seconds

    # Load experimental data
    c_data = load_c_data()

    # Interpolate experimental data
    c_data_interp, x_interp, time_indices, space_indices = interp_c_data(c_data, N, L)

    num_train = 9
    t_in_minutes = time_indices[num_train - 1].item() / 600

    # RMSE loss between experiment and simulation + L2 regularization
    def objective(p0: float, p1: float) -> float:
        lbd = 0.005
        return float(
            loss_func(
                p0, p1,
                c_data_interp[:num_train],
                time_indices[:num_train],
                space_indices,
                c_avg, N, L, dt, t_in_minutes
            )
            + lbd * (p0**2 + p1**2)
        )

    cfg = BOConfig(
        bounds=((-4, 4), (-4, 4)),
        target_loss=0.02676,
        n_init=10,
        n_iter=1000,
        n_samples=1000,
        xi=0.0,
        random_state=0,
        nan_penalty=1.0,
        nan_alpha=1e-2,
        valid_alpha=1e-10,
    )
    bo = BayesianOptimization(cfg)

    print(cfg)
    print(str(bo.gp.kernel))

    start_time = time.perf_counter()
    result = bo.run(objective)
    end_time = time.perf_counter()

    print("Best (p0, p1):", (result["p0"], result["p1"]))
    print("Best loss Δ:", result["loss"])
    print("Optimization time:", end_time - start_time)