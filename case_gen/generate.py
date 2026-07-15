"""Case generator.

Reads ``case_gen/configs/case_X.yaml``, drives the held-out oracle
under ``tasks/.../tests/oracle/``, and emits:

* ``tasks/.../environment/data/cases/case_X/{data.h5, params.json, formalism.md}``
  — agent-visible bundle.
* ``tasks/.../tests/oracle_truth/case_X/truth.npz``
  — held-out ground truth for the verifier.

Usage::

    uv run python -m case_gen.generate --case case_1
    uv run python -m case_gen.generate --all

Determinism: every RNG draws from ``numpy.random.default_rng(seed)`` where
``seed`` lives in the YAML. Re-running the same config on the same machine
must produce byte-identical artifacts (oracle-spec.md §7).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax import config

config.update("jax_enable_x64", True)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    REPO_ROOT
    / "tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"
)

sys.path.insert(0, str(TASK_ROOT / "tests"))

from oracle.flux import FARADAY, decompose  # noqa: E402
from oracle.invert_ne import invert  # noqa: E402
from oracle.solver import CaseInputs, interp_fn, simulate  # noqa: E402

from case_gen.writers import (  # noqa: E402
    write_data_h5,
    write_formalism_md,
    write_params_json,
    write_truth_npz,
)


# --------------------------------------------------------------- material helpers


def _rho(c: np.ndarray | float, a: float, b: float, d: float) -> Any:
    return a + b * c + d * c * c


def _drhodc(c: np.ndarray | float, b: float, d: float) -> Any:
    return b + 2.0 * d * c


def factor_cgs(c, *, a: float, b: float, d: float, M0: float, M: float):
    """``(1 − d ln c₀ / d ln c)`` at salt concentration c (mol/L).

    Matches ``Mass Transport in.../solver.py``'s ``factor``. ``c0`` here is
    the *local* solvent concentration (mol/L), derived from the
    solution-density polynomial — distinct from the proposal's ``c0``
    (pure-solvent reference).
    """
    rho_v = _rho(c, a, b, d)
    drhodc = _drhodc(c, b, d)
    c0_local = 1.0 / M0 * (rho_v * 1e3 - M * c)
    V0_bar = M0 / (rho_v - c * drhodc)
    return 1.0 / (c0_local * V0_bar) * 1e3


def V_bar_cgs_at(c: float, *, a: float, b: float, d: float, M: float) -> float:
    """Salt partial molar volume (cm³/mol) at concentration c (mol/L)."""
    rho_v = _rho(c, a, b, d)
    drhodc = _drhodc(c, b, d)
    return float((M - drhodc * 1e3) / (rho_v - c * drhodc))


# ------------------------------------------------------------ current schedule


def build_current_schedule(cur_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_s, i_A_per_cm2)`` knot arrays for the configured schedule."""
    if cur_cfg["schedule"] != "ramp_hold_ramp":
        raise ValueError(f"unknown current schedule: {cur_cfg['schedule']}")
    ru = float(cur_cfg["ramp_up_s"])
    hd = float(cur_cfg["hold_s"])
    rd = float(cur_cfg["ramp_down_s"])
    peak = float(cur_cfg["peak_A_per_cm2"])
    t = np.array([0.0, ru, ru + hd, ru + hd + rd], dtype=np.float64)
    i = np.array([0.0, peak, peak, 0.0], dtype=np.float64)
    return t, i


# ----------------------------------------------------------------- main driver


def generate_case(config_path: Path, verbose: bool = True) -> None:
    """Read a case YAML, run the full pipeline, write all four artifacts."""
    cfg_bytes = config_path.read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    config_hash = hashlib.sha256(cfg_bytes).hexdigest()
    case_id = cfg["case_id"]

    if verbose:
        print(f"\n=== Generating {case_id} ===")
        print(f"config: {config_path}")
        print(f"config_hash: {config_hash[:16]}…")

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    # ------------------------------------------------- material setup
    mat = cfg["material"]
    a, b, d = (mat["rho_polynomial"][k] for k in ("a", "b", "d"))
    M0 = float(mat["M0_g_per_mol"])
    M = float(mat["M_g_per_mol"])
    T_temp = float(mat["T_temp_K"])

    # Pure-solvent reference (proposal's c0): solvent concentration when c=0,
    # i.e. density / molar mass of the solvent. mol/L.
    c0_pure = float(_rho(0.0, a, b, d) / M0 * 1000.0)

    # Constant V_bar. Default = evaluated at c_avg from the rho polynomial;
    # may be overridden in YAML via ``material.V_bar_si`` (m³/mol). Override
    # is the knob for controlling convective coupling strength: smaller
    # V_bar shrinks v₀ and brings tp0_NE closer to tp0 (useful for
    # case 1's NE-valid target).
    c_avg = 0.5 * (float(cfg["c_grid"]["c_min"]) + float(cfg["c_grid"]["c_max"]))
    if "V_bar_si" in mat:
        V_bar_scalar_si = float(mat["V_bar_si"])
        V_bar_scalar_cgs = V_bar_scalar_si * 1e6
    else:
        V_bar_scalar_cgs = V_bar_cgs_at(c_avg, a=a, b=b, d=d, M=M)
        V_bar_scalar_si = V_bar_scalar_cgs * 1e-6

    # Dense factor table (shared between agent-facing params.json and the
    # oracle solver — same tabulation everywhere keeps the agent's view
    # consistent with what the oracle integrated against).
    c_max_table = max(
        1.5 * float(cfg["c_grid"]["c_max"]),
        2.0 * float(cfg["initial"]["c_init"]),
    )
    factor_c = np.linspace(0.0, c_max_table, 201, dtype=np.float64)
    factor_v = factor_cgs(factor_c, a=a, b=b, d=d, M0=M0, M=M)

    # ------------------------------------------------- oracle functions (JAX)
    D_c = np.asarray(mat["D_table_cgs"]["c_mol_per_L"], dtype=np.float64)
    D_v = np.asarray(mat["D_table_cgs"]["D_cm2_per_s"], dtype=np.float64)
    tp0_c = np.asarray(mat["tp0_table"]["c_mol_per_L"], dtype=np.float64)
    tp0_v = np.asarray(mat["tp0_table"]["tp0"], dtype=np.float64)

    D_fn = interp_fn(D_c, D_v)
    tp0_fn = interp_fn(tp0_c, tp0_v)
    factor_fn_jax = interp_fn(factor_c, factor_v)

    def V_bar_fn(c: jax.Array) -> jax.Array:
        return jnp.full_like(c, V_bar_scalar_cgs)

    i_t_knots, i_amp_knots = build_current_schedule(cfg["current"])
    i_t_jnp = jnp.asarray(i_t_knots)
    i_amp_jnp = jnp.asarray(i_amp_knots)

    def i_fn(t: jax.Array) -> jax.Array:
        return jnp.interp(t, i_t_jnp, i_amp_jnp)

    # ------------------------------------------------- geometry / time
    geom = cfg["geometry"]
    L_cm = float(geom["L_cm"])
    Nx = int(geom["Nx"])
    dx_cm = L_cm / Nx

    tcfg = cfg["time"]
    T_s = float(tcfg["T_s"])
    dt_s = float(tcfg["dt_s"])
    Nt_out = int(tcfg["Nt_out"])
    num_steps = int(round(T_s / dt_s))

    c_init = float(cfg["initial"]["c_init"])
    case = CaseInputs(N=Nx, L=L_cm, dt=dt_s, num_steps=num_steps, c_init=c_init)

    # ------------------------------------------------- moving-frame forward
    if verbose:
        print(f"forward solve: Nx={Nx}, L={L_cm} cm, dt={dt_s} s, "
              f"num_steps={num_steps}")
    t0 = time.perf_counter()
    x_cm, t_internal, c_hist, v0_hist = simulate(
        D_fn, tp0_fn, factor_fn_jax, V_bar_fn, i_fn, case, lab_frame=False,
    )
    c_hist = np.asarray(c_hist.block_until_ready())
    v0_hist = np.asarray(v0_hist.block_until_ready())
    if verbose:
        print(f"  simulate done in {time.perf_counter() - t0:.1f}s "
              f"(c_hist {c_hist.shape}, v0_hist {v0_hist.shape})")

    x_m = np.asarray(x_cm) * 1e-2
    t_internal = np.asarray(t_internal)

    t_indices = np.linspace(0, num_steps, Nt_out).astype(np.int64)
    v0_center = 0.5 * (v0_hist[:, :-1] + v0_hist[:, 1:])  # (num_steps+1, Nx)

    c_sim_out = c_hist[t_indices, :]                       # (Nt_out, Nx), mol/L
    v_sim_out_cm_per_s = v0_center[t_indices, :]
    v_sim_out_nm_per_s = v_sim_out_cm_per_s * 1e7

    # ------------------------------------------------- noise
    noise = cfg["noise"]
    sigma_c = float(noise["c_sigma_mol_per_L"])
    sigma_v = float(noise["v_sigma_rel"]) * float(np.max(np.abs(v_sim_out_nm_per_s)))
    c_data = c_sim_out + rng.normal(0.0, sigma_c, c_sim_out.shape)
    v_data = v_sim_out_nm_per_s + rng.normal(0.0, sigma_v, v_sim_out_nm_per_s.shape)
    if verbose:
        print(f"noise: sigma_c={sigma_c:.4g} mol/L, "
              f"sigma_v={sigma_v:.4g} nm/s ({100*float(noise['v_sigma_rel'])}% of max|v|)")

    # ------------------------------------------------- c_grid + oracle values
    cgrid_cfg = cfg["c_grid"]
    c_grid = np.linspace(
        float(cgrid_cfg["c_min"]), float(cgrid_cfg["c_max"]),
        int(cgrid_cfg["n"]), dtype=np.float64,
    )
    D_oracle_cgs = np.asarray(D_fn(jnp.asarray(c_grid)))
    tp0_oracle = np.asarray(tp0_fn(jnp.asarray(c_grid)))
    D_oracle_si = D_oracle_cgs * 1e-4

    # ------------------------------------------------- NE inversion
    if verbose:
        print("invert_ne: BFGS (lab-frame, holding D = D_oracle fixed)…")
    t0 = time.perf_counter()
    tp0_NE_oracle = invert(
        c_data=c_data,
        t_indices=t_indices,
        D_fn=D_fn,
        factor_fn=factor_fn_jax,
        V_bar_fn=V_bar_fn,
        i_fn=i_fn,
        case=case,
        c_grid=c_grid,
        lambda_reg=float(cfg["invert_ne"]["lambda_reg"]),
        init_tp0=float(cfg["invert_ne"]["init_tp0"]),
        maxiter=int(cfg["invert_ne"]["maxiter"]),
        verbose=verbose,
    )
    if verbose:
        print(f"  invert_ne done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------- regime labels
    regime = np.array(
        [_classify(float(tp0_oracle[i]), float(tp0_NE_oracle[i]))
         for i in range(c_grid.size)],
        dtype="U16",
    )
    if verbose:
        from collections import Counter
        print(f"regime distribution: {dict(Counter(regime.tolist()))}")

    # ------------------------------------------------- flux samples
    flux_cfg = cfg["flux_samples"]
    n_samples = int(flux_cfg["n"])
    L_m = L_cm * 1e-2
    x_min_m = float(flux_cfg["x_min_frac"]) * L_m
    x_max_m = float(flux_cfg["x_max_frac"]) * L_m
    t_min_s = float(flux_cfg["t_min_frac"]) * T_s
    t_max_s = float(flux_cfg["t_max_frac"]) * T_s
    flux_x = rng.uniform(x_min_m, x_max_m, size=n_samples)
    flux_t = rng.uniform(t_min_s, t_max_s, size=n_samples)

    flux_oracle, flux_total = _compute_flux_oracle(
        flux_x=flux_x, flux_t=flux_t,
        c_hist=c_hist, v0_center=v0_center,
        x_m=x_m, dx_cm=dx_cm, dt_s=dt_s, num_steps=num_steps,
        D_fn=D_fn, tp0_fn=tp0_fn, factor_fn_jax=factor_fn_jax,
        i_t_knots=i_t_knots, i_amp_knots=i_amp_knots,
    )

    # ------------------------------------------------- write all artifacts
    env_dir = TASK_ROOT / "environment/data/cases" / case_id
    truth_dir = TASK_ROOT / "tests/oracle_truth" / case_id
    if verbose:
        print(f"writing → {env_dir}/")
        print(f"writing → {truth_dir}/")

    write_data_h5(
        env_dir / "data.h5",
        c_data=c_data, v_data=v_data,
        x_m=x_m, t_s=t_internal[t_indices],
    )
    write_formalism_md(env_dir / "formalism.md")

    # Convert current knots to A/m² for the agent.
    i_amp_si = (i_amp_knots * 1e4).tolist()
    params = {
        "case_id": case_id,
        "L": L_m,
        "T": T_s,
        "Nt": Nt_out,
        "Nx": Nx,
        "i_app": [[float(t), float(i)] for t, i in zip(i_t_knots, i_amp_si)],
        "c0": c0_pure,
        "V_bar": V_bar_scalar_si,
        "T_temp": T_temp,
        "c_init": c_init,
        "c_grid": c_grid.tolist(),
        "flux_samples": [[float(x), float(t)] for x, t in zip(flux_x, flux_t)],
        "factor_table": {
            "c_mol_per_L": factor_c.tolist(),
            "factor": factor_v.tolist(),
        },
    }
    write_params_json(env_dir / "params.json", params)

    write_truth_npz(
        truth_dir / "truth.npz",
        D_oracle=D_oracle_si,
        tp0_oracle=tp0_oracle,
        tp0_NE_oracle=tp0_NE_oracle,
        regime=regime,
        c_grid=c_grid,
        c_data=c_data,
        v_data=v_data,
        flux_oracle=flux_oracle,
        flux_total=flux_total,
        flux_x=flux_x,
        flux_t=flux_t,
        # Resim fields for verifier check #6.
        L_cm=L_cm,
        Nx=Nx,
        dt_s=dt_s,
        num_steps=num_steps,
        c_init_mol_per_L=c_init,
        V_bar_cm3_per_mol=V_bar_scalar_cgs,
        t_indices=t_indices,
        factor_c_mol_per_L=factor_c,
        factor_v=factor_v,
        i_t_knots_s=i_t_knots,
        i_amp_A_per_cm2=i_amp_knots,
        seed=seed,
        config_hash=config_hash,
    )

    if verbose:
        print(f"=== {case_id} OK ===\n")


# ------------------------------------------------------------ regime / flux helpers


def _classify(tp0: float, tp0_ne: float) -> str:
    if np.sign(tp0) != np.sign(tp0_ne):
        return "NE_wrong_sign"
    if abs(tp0 - tp0_ne) >= 0.05:
        return "NE_deviates"
    return "NE_valid"


def _compute_flux_oracle(
    *,
    flux_x: np.ndarray, flux_t: np.ndarray,
    c_hist: np.ndarray, v0_center: np.ndarray,
    x_m: np.ndarray, dx_cm: float, dt_s: float, num_steps: int,
    D_fn, tp0_fn, factor_fn_jax,
    i_t_knots: np.ndarray, i_amp_knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the canonical flux decomposition at the 10 samples.

    Returns ``(flux_oracle[10, 3], flux_total[10])`` in SI
    ``mol / (m² s)``. Columns of ``flux_oracle`` are ``[J_diff, J_mig, J_conv]``.
    """
    Nx = c_hist.shape[1]
    L_m = x_m[-1] + (x_m[1] - x_m[0]) / 2.0
    n = flux_x.size

    J = np.zeros((n, 3), dtype=np.float64)
    Jt = np.zeros((n,), dtype=np.float64)
    for k in range(n):
        # Nearest internal index. Cell centers at (i + 0.5) * dx_m.
        dx_m = dx_cm * 1e-2
        x_idx = int(round(flux_x[k] / dx_m - 0.5))
        # Clamp away from boundaries to keep the centered dcdx in-bounds.
        x_idx = max(1, min(Nx - 2, x_idx))
        t_idx = int(round(flux_t[k] / dt_s))
        t_idx = max(0, min(num_steps, t_idx))

        c_loc = float(c_hist[t_idx, x_idx])              # mol/L
        v0_loc_cgs = float(v0_center[t_idx, x_idx])       # cm/s
        # centered dcdx in mol/L per cm
        dcdx_cgs = float(
            (c_hist[t_idx, x_idx + 1] - c_hist[t_idx, x_idx - 1]) / (2.0 * dx_cm)
        )
        i_loc_cgs = float(np.interp(t_idx * dt_s, i_t_knots, i_amp_knots))

        D_loc_cgs = float(D_fn(jnp.asarray(c_loc)))
        tp0_loc = float(tp0_fn(jnp.asarray(c_loc)))
        factor_loc = float(factor_fn_jax(jnp.asarray(c_loc)))

        # cgs → SI
        c_si = c_loc * 1e3                                # mol/L → mol/m³
        v0_si = v0_loc_cgs * 1e-2                         # cm/s → m/s
        i_si = i_loc_cgs * 1e4                            # A/cm² → A/m²
        D_si = D_loc_cgs * 1e-4                           # cm²/s → m²/s
        dcdx_si = dcdx_cgs * 1e5                          # (mol/L)/cm → mol/m⁴

        fl = decompose(
            D=D_si, tp0=tp0_loc, factor=factor_loc,
            c=c_si, dcdx=dcdx_si, v0=v0_si, i_t=i_si,
        )
        J[k, 0] = float(fl.J_diff)
        J[k, 1] = float(fl.J_mig)
        J[k, 2] = float(fl.J_conv)
        Jt[k] = float(fl.J_total)

    return J, Jt


# ------------------------------------------------------------ CLI


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate TB-Science case bundles.")
    p.add_argument("--case", type=str, default=None,
                   help="case id, e.g. case_1; mutually exclusive with --all")
    p.add_argument("--all", action="store_true",
                   help="generate every case_*.yaml under case_gen/configs/")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _resolve_configs(args: argparse.Namespace) -> list[Path]:
    cfg_dir = REPO_ROOT / "case_gen/configs"
    if args.all:
        return sorted(cfg_dir.glob("case_*.yaml"))
    if not args.case:
        raise SystemExit("specify --case CASE_ID or --all")
    p = cfg_dir / f"{args.case}.yaml"
    if not p.exists():
        raise SystemExit(f"no such config: {p}")
    return [p]


def main() -> int:
    args = _parse_args()
    cfgs = _resolve_configs(args)
    for c in cfgs:
        generate_case(c, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
