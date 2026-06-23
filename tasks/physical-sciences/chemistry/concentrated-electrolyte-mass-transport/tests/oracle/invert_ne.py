"""Lab-frame Nernst-Planck inversion (formal definition of t⁺⁰_NE).

Per ``docs/plan/oracle-spec.md`` §3, ``t⁺⁰_NE(c)`` is the transference
number that would be inferred from operando ``c_data`` under classical
Nernst-Planck (``v₀ ≡ 0``), holding the oracle's true ``D(c)`` fixed.
This module is the *canonical* implementation: it runs once per case at
generation time, and the result is stored in ``truth.npz`` as the
ground truth used by the verifier for the regime check.

The inversion is solved with BFGS via :mod:`jaxopt` against the held-out
moving-frame ``c_data`` (already noisy at this point — the lab-frame
inversion sees exactly what an honest agent would see). Light Tikhonov
regularization on the 50-point ``tp0_NE`` ansatz keeps the unidentified
end-point parameters from oscillating without biasing the bulk.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import config

from .solver import CaseInputs, PropFn, simulate

config.update("jax_enable_x64", True)


CurrentFn = Callable[[jax.Array], jax.Array]


def invert(
    c_data: np.ndarray,
    t_indices: np.ndarray,
    D_fn: PropFn,
    factor_fn: PropFn,
    V_bar_fn: PropFn,
    i_fn: CurrentFn,
    case: CaseInputs,
    c_grid: np.ndarray,
    *,
    lambda_reg: float = 1e-4,
    init_tp0: float = 0.3,
    maxiter: int = 200,
    verbose: bool = False,
) -> np.ndarray:
    """BFGS inversion for ``t⁺⁰_NE`` at the 50 ``c_grid`` points.

    Parameters
    ----------
    c_data
        Operando salt-concentration data (mol/L) on the case's
        ``(Nt_out, Nx)`` reporting grid. Matches the array later written
        to ``data.h5`` (noise included).
    t_indices
        Indices into the internal time grid (``arange(num_steps + 1)``)
        at which ``c_data`` was sampled. Used to align the lab-frame
        simulation with ``c_data`` inside the loss.
    D_fn, factor_fn, V_bar_fn, i_fn
        Pure JAX material-property and current functions in the oracle
        solver's internal cgs convention (see ``oracle/solver.py``).
        ``D_fn`` is the oracle's *true* diffusivity — this is the
        canonical definition: NE inversion holds ``D`` fixed and lets
        ``t⁺⁰_NE`` absorb the missing convective transport.
    case
        Per-case ``CaseInputs`` (geometry, dt, num_steps, c_init).
    c_grid
        50 concentration points (mol/L) at which the agent and the
        oracle both report ``t⁺⁰`` and ``t⁺⁰_NE``.
    lambda_reg
        Coefficient on ``Σ (Δparams)²`` Tikhonov term. Default keeps
        end-point oscillation tame without biasing the bulk fit. Tune
        in ``case_gen/configs/`` if a case proves under-regularized.
    init_tp0
        Constant initial guess across all 50 points. Default ``0.3``
        matches the published BFGS reference's typical seed.
    maxiter
        Hard cap on BFGS iterations.
    verbose
        If True, print iteration loss.

    Returns
    -------
    tp0_NE : np.ndarray, shape ``(50,)``
        Optimal lab-frame transference number at ``c_grid`` points.
    """
    c_data_j = jnp.asarray(c_data, dtype=jnp.float64)
    t_idx_j = jnp.asarray(t_indices, dtype=jnp.int64)
    c_grid_j = jnp.asarray(c_grid, dtype=jnp.float64)
    lam = jnp.asarray(lambda_reg, dtype=jnp.float64)
    # Normalize the data term so lambda_reg's effect is independent of
    # absolute c-magnitude. Matches the reference solver's lab-frame
    # inversion for cross-implementation regime stability.
    c_norm_sq = jnp.maximum(jnp.mean(c_data_j ** 2), 1.0)

    def _loss(params: jax.Array) -> jax.Array:
        def tp0_ne_fn(c: jax.Array) -> jax.Array:
            return jnp.interp(c, c_grid_j, params)

        _, _, c_hist, _ = simulate(
            D_fn, tp0_ne_fn, factor_fn, V_bar_fn, i_fn, case, lab_frame=True
        )
        c_sim = c_hist[t_idx_j, :]
        data_term = jnp.mean((c_sim - c_data_j) ** 2) / c_norm_sq
        reg_term = jnp.sum(jnp.diff(params) ** 2)
        return data_term + lam * reg_term

    init_params = jnp.full((c_grid.size,), init_tp0, dtype=jnp.float64)

    if verbose:
        l0 = float(_loss(init_params))
        print(f"  invert_ne: initial loss = {l0:.6e}")

    solver = jaxopt.ScipyMinimize(
        fun=_loss, method="L-BFGS-B", maxiter=maxiter, tol=1e-9,
    )
    result = solver.run(init_params)
    params_opt = np.asarray(result.params, dtype=np.float64)

    if verbose:
        l_final = float(_loss(jnp.asarray(params_opt)))
        print(f"  invert_ne: final   loss = {l_final:.6e}, "
              f"n_iter ≤ {maxiter}, params range = "
              f"[{params_opt.min():.4f}, {params_opt.max():.4f}]")

    return params_opt
