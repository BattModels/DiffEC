"""Smoke test for the moving-frame oracle solver (ADR-0002).

Reproduces the published Steinrück-2020 fit by re-running the
generalized solver under ``tests/oracle/solver.py`` with the
parameterization from BattModels/DiffEC
``Mass Transport in Concentrated Electrolytes and Benchmarks/solver.py``.

Acceptance criteria
-------------------
Primary:   max |c_oracle - c_published| / c_published < 1e-3 at the
           9 sampled time indices, all 50 cells. This compares our
           solver against the published solver's own saved output,
           catching numerics regressions (not fit residuals).

Secondary: max |c_oracle - c_experimental_interp| / c_experimental < 0.05
           at the 9 experimental time points. The published fit had
           loss ~0.027 against the experimental data, so we expect our
           solver to land within a few percent.

Usage
-----
    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Resolve the oracle package without packaging the task tree.
REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"
sys.path.insert(0, str(TASK_ROOT / "tests"))

from oracle.solver import CaseInputs, simulate  # noqa: E402

PUBLISHED = REPO_ROOT / "Mass Transport in Concentrated Electrolytes and Benchmarks"

# --- Published material polynomials -----------------------------------

M0 = 44.05                  # g/mol, solvent (PEO repeat unit)
M = 6.94 + 280.15           # g/mol, LiTFSI salt


def rho(c):
    return 1.123276 + 0.106822 * c + 0.007606 * c ** 2          # g/cm^3


def drhodc(c):
    return 0.106822 + 2 * 0.007606 * c


def V_bar(c):
    return (M - drhodc(c) * 1e3) / (rho(c) - c * drhodc(c))     # cm^3/mol


def factor(c):
    rho_v = rho(c)
    c0 = 1.0 / M0 * (rho_v * 1e3 - M * c)                       # mol/L solvent
    V0_bar = M0 / (rho_v - c * drhodc(c))                       # cm^3/mol solvent
    return 1.0 / (c0 * V0_bar) * 1e3


# --- Published CSV inputs ---------------------------------------------


def csv_to_numpy(path: Path) -> np.ndarray:
    rows = []
    with path.open() as fh:
        for row in csv.reader(fh):
            rows.append([float(x.strip()) for x in row])
    return np.asarray(rows)


def load_published_inputs():
    cd = csv_to_numpy(PUBLISHED / "data/current_density.csv")
    dr = csv_to_numpy(PUBLISHED / "data/D_tp0_relation.csv")
    return cd, dr


# --- Smoke test --------------------------------------------------------


def main() -> int:
    current_density = csv_to_numpy(PUBLISHED / "data/current_density.csv")
    D_relation = csv_to_numpy(PUBLISHED / "data/D_tp0_relation.csv")

    I_xp = jnp.asarray(current_density[:, 0])           # hours
    I_fp = jnp.asarray(current_density[:, 1])           # mA/cm^2
    D_xp = jnp.asarray(D_relation[:, 0])                # mol/L
    D_fp = jnp.asarray(D_relation[:, 1])                # ~ cm^2/s scale

    # Published fitted parameters (loaded from saved history).
    p0_history = np.load(PUBLISHED / "results/p0_history.npy")
    p1_history = np.load(PUBLISHED / "results/p1_history.npy")
    p0 = float(p0_history[-1])
    p1 = float(p1_history[-1])
    print(f"Published fitted (p0, p1) = ({p0:.6f}, {p1:.6f})")

    c_avg = 1.87  # solver.py global

    def tp0_fn(c):
        return p0 + p1 * (c - c_avg) / c_avg

    def factor_fn(c):
        return factor(c)

    def V_bar_fn(c):
        return V_bar(c)

    def D_fn(c):
        relation_coef = jnp.interp(c, D_xp, D_fp)
        return (1.0 - tp0_fn(c)) * relation_coef / factor_fn(c)

    def i_fn(t):
        # t in seconds; published xp is in hours; output in A/cm^2.
        return jnp.interp(t / 3600.0, I_xp, I_fp) * 1e-3

    # Geometry / time matches solver.py defaults.
    N = 50
    L = 0.3                       # cm
    dt = 0.1                      # s
    t_in_minutes = 1100
    t_end = t_in_minutes * 60
    num_steps = int(t_end / dt)   # 660000
    case = CaseInputs(N=N, L=L, dt=dt, num_steps=num_steps, c_init=c_avg)

    print(f"Running oracle: N={N}, L={L}cm, dt={dt}s, num_steps={num_steps} "
          f"(t_end={t_end}s).")
    print("Compiling lax.scan + integrating...")
    import time
    t0 = time.perf_counter()
    x, t, c_hist, v0_hist = simulate(
        D_fn, tp0_fn, factor_fn, V_bar_fn, i_fn, case,
    )
    c_hist = np.asarray(c_hist.block_until_ready())
    v0_hist = np.asarray(v0_hist.block_until_ready())
    t1 = time.perf_counter()
    print(f"Done in {t1 - t0:.1f}s. c_hist shape={c_hist.shape}, "
          f"v0_hist shape={v0_hist.shape}.")

    # --- Primary check: match published c_sim.npy at the 9 time indices.
    time_indices = np.array(
        [20, 151, 314, 412, 511, 609, 707, 871, 1100], dtype=np.int64
    ) * 600                                # step index in dt=0.1s
    c_published = np.load(PUBLISHED / "results/c_sim.npy")        # (9, 50)
    c_oracle = c_hist[time_indices, :]                            # (9, 50)
    rel_err = np.abs(c_oracle - c_published) / np.abs(c_published)
    print(f"Primary (vs published c_sim.npy): "
          f"max rel err = {rel_err.max():.3e}, "
          f"mean rel err = {rel_err.mean():.3e}")
    primary_pass = rel_err.max() < 1e-3

    # --- Secondary check: vs experimental data, interpolated onto our mesh.
    filenames = [
        "t=0020.csv", "t=0151.csv", "t=0314.csv", "t=0412.csv",
        "t=0511.csv", "t=0609.csv", "t=0707.csv", "t=0871.csv", "t=1100.csv",
    ]
    x_mm = 10 * (np.arange(N) + 0.5) * (L / N)  # cm -> mm
    cond = (x_mm >= 0.6) & (x_mm <= 2.4)
    rel_errs_exp = []
    for j, fname in enumerate(filenames):
        raw = csv_to_numpy(PUBLISHED / "data" / fname)
        c_interp = np.interp(x_mm[cond], raw[:, 0], raw[:, 1])
        c_sim_window = c_oracle[j, cond]
        rel_err = np.abs(c_sim_window - c_interp) / np.abs(c_interp)
        rel_errs_exp.append(rel_err.max())
    rel_errs_exp = np.asarray(rel_errs_exp)
    print(f"Secondary (vs experimental): "
          f"max per-time rel err = {rel_errs_exp.max():.3e}, "
          f"mean = {rel_errs_exp.mean():.3e}")
    secondary_pass = rel_errs_exp.max() < 0.05

    print()
    print(f"PRIMARY:   {'PASS' if primary_pass else 'FAIL'}  "
          f"(<1e-3 rel err vs published c_sim.npy)")
    print(f"SECONDARY: {'PASS' if secondary_pass else 'FAIL'}  "
          f"(<5% rel err vs experimental c_data at all 9 times)")

    return 0 if (primary_pass and secondary_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
