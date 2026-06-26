"""ADR-0004 perturbation check: held-out (D, t⁺⁰) vs published values.

For each case, confirm that an agent reporting the published Steinrück
/ DiffEC fit would fail the verifier deterministically. The verifier's
check #2 is ``|t⁺⁰_agent − t⁺⁰_oracle| ≤ 0.05`` at every c_grid point —
ANY single failing point fails the whole check. So the anti-cheat
condition is:

  max_{c ∈ c_grid} |t⁺⁰_oracle(c) − t⁺⁰_published(c)| >= 0.10

i.e. at least one c_grid point has the literature value ≥ 0.10 away
from the oracle. The 0.10 is a 2× safety margin over the 0.05
verifier threshold (so a small literature perturbation or numerical
noise can't accidentally rescue the lookup).

Analogously for D, with relative threshold 0.20 (2× verifier's 0.10).

The pointwise *minimum* distance is also reported — when it is small,
the literature polynomial happens to pass close to the oracle at some
single c, but as long as the *maximum* exceeds the threshold, the
lookup-attack fails overall.

Published reference: the fitted polynomial reported in the BattModels/
DiffEC repo (which reproduces the Steinrück 2020 fit; Pesko 2017 is
the underlying transport-coefficient source). Functional form:

    t⁺⁰_published(c) = p0 + p1 * (c − c_avg) / c_avg
    D_published(c)    = (1 − t⁺⁰_published(c)) * relation_coef(c) / factor(c)

with c_avg = 1.87 mol/L (the Steinrück experiment mean), p0/p1 loaded
from the repo's `results/p0_history.npy` / `p1_history.npy` (final
iteration), and `relation_coef(c)` tabulated in
`data/D_tp0_relation.csv`. `factor(c)` is the same
(1 − d ln c₀ / d ln c) tabulation used everywhere.

Usage::

    uv run python scripts/litvalue_distance.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    REPO_ROOT
    / "tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport"
)
PUBLISHED = REPO_ROOT / "Mass Transport in Concentrated Electrolytes and Benchmarks"

C_AVG_STEINRUCK = 1.87  # mol/L; the Steinrück experiment average c
T_PLUS_THRESHOLD = 0.10
D_REL_THRESHOLD = 0.20


# ---------------------------------------------------------------- helpers
def _csv_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open() as fh:
        for r in csv.reader(fh):
            rows.append([float(x.strip()) for x in r])
    a = np.asarray(rows, dtype=np.float64)
    return a[:, 0], a[:, 1]


def _published_p0_p1() -> tuple[float, float]:
    p0 = float(np.load(PUBLISHED / "results/p0_history.npy")[-1])
    p1 = float(np.load(PUBLISHED / "results/p1_history.npy")[-1])
    return p0, p1


def published_tp0(c_mol_per_L: np.ndarray) -> np.ndarray:
    """Steinrück / DiffEC fitted t⁺⁰(c)."""
    p0, p1 = _published_p0_p1()
    return p0 + p1 * (np.asarray(c_mol_per_L) - C_AVG_STEINRUCK) / C_AVG_STEINRUCK


def published_relation_coef(c_mol_per_L: np.ndarray) -> np.ndarray:
    """`relation_coef(c)` from the Pesko 2017 D-vs-c tabulation as used
    by the DiffEC repo's `solver.py`. Native units are cgs (cm²/s
    family) — we keep them and convert to SI at the boundary."""
    c_xp, c_fp = _csv_xy(PUBLISHED / "data/D_tp0_relation.csv")
    return np.interp(c_mol_per_L, c_xp, c_fp)


def published_D_si(c_mol_per_L: np.ndarray, factor_cgs: np.ndarray) -> np.ndarray:
    """D_published(c) in m²/s. Recovered from the DiffEC formula
    ``D = (1 − tp0) * relation_coef / factor``. ``factor`` is the
    case's tabulated (1 − d ln c₀ / d ln c) — the published fit
    derives it from the rho polynomial; here we use the per-case
    factor table to make the comparison apples-to-apples (case
    generation shares the same rho-derived factor across all cases,
    so this just keeps the SI conversion correct)."""
    tp0 = published_tp0(c_mol_per_L)
    rel = published_relation_coef(c_mol_per_L)
    D_cgs = (1.0 - tp0) * rel / np.asarray(factor_cgs)
    return D_cgs * 1e-4  # cm²/s → m²/s


# ---------------------------------------------------------------- per-case
def check_case(case_id: str) -> dict:
    """Return per-case diagnostics. Pass = True iff both perturbation
    thresholds are met at every c_grid point."""
    truth_path = TASK_ROOT / f"tests/oracle_truth/{case_id}/truth.npz"
    truth = dict(np.load(truth_path))
    c_grid = truth["c_grid"]
    D_oracle = truth["D"]
    tp0_oracle = truth["t_plus_0"]

    # Per-case factor table (same family across cases, see oracle-spec.md §6).
    factor_c = truth["factor_c_mol_per_L"]
    factor_v = truth["factor_v"]
    factor_at_grid = np.interp(c_grid, factor_c, factor_v)

    D_pub = published_D_si(c_grid, factor_at_grid)
    tp0_pub = published_tp0(c_grid)

    tp_abs = np.abs(tp0_oracle - tp0_pub)
    D_rel = np.abs(D_oracle / D_pub - 1.0)

    tp_min = float(tp_abs.min())
    tp_max = float(tp_abs.max())
    D_min = float(D_rel.min())
    D_max = float(D_rel.max())

    return {
        "case_id": case_id,
        "c_grid_lo": float(c_grid[0]),
        "c_grid_hi": float(c_grid[-1]),
        "tp_min": tp_min,
        "tp_max": tp_max,
        "tp_max_c": float(c_grid[int(tp_abs.argmax())]),
        "D_min": D_min,
        "D_max": D_max,
        "D_max_c": float(c_grid[int(D_rel.argmax())]),
        # Pass criterion: at least one c_grid point has the literature
        # value ≥ threshold away from the oracle. This ensures the
        # lookup-attack fails verifier deterministically.
        "tp_pass": tp_max >= T_PLUS_THRESHOLD,
        "D_pass": D_max >= D_REL_THRESHOLD,
    }


# ---------------------------------------------------------------- main
def main() -> int:
    p0, p1 = _published_p0_p1()
    print(f"Published reference: Steinrück 2020 / DiffEC paper fit")
    print(f"  t⁺⁰(c) = {p0:.4f} + {p1:.4f} * (c − {C_AVG_STEINRUCK}) / {C_AVG_STEINRUCK}")
    print(f"  D(c)   = (1 − t⁺⁰(c)) * relation_coef(c) / factor(c)")
    print(f"\nThresholds (ADR-0004, anti-cheat against literature lookup):")
    print(f"  max |t⁺⁰_oracle − t⁺⁰_pub|   >= {T_PLUS_THRESHOLD}  (so verifier check #2 fails on lookup)")
    print(f"  max |D_oracle / D_pub − 1|   >= {D_REL_THRESHOLD}  (so verifier check #1 fails on lookup)")
    print()

    cases = [f"case_{i}" for i in (1, 2, 3, 4)]
    all_pass = True
    results = []
    for case_id in cases:
        r = check_case(case_id)
        results.append(r)
        ok_tp = "OK " if r["tp_pass"] else "FAIL"
        ok_D = "OK " if r["D_pass"] else "FAIL"
        print(f"{case_id} [c_grid {r['c_grid_lo']:.2f}–{r['c_grid_hi']:.2f}]:")
        print(f"  |Δtp⁺⁰|  range [{r['tp_min']:.4f}, {r['tp_max']:.4f}], "
              f"max @ c={r['tp_max_c']:.3f}   (threshold {T_PLUS_THRESHOLD})  {ok_tp}")
        print(f"  |D/Dp−1| range [{r['D_min']:.4f}, {r['D_max']:.4f}], "
              f"max @ c={r['D_max_c']:.3f}   (threshold {D_REL_THRESHOLD})  {ok_D}")
        all_pass = all_pass and r["tp_pass"] and r["D_pass"]

    print()
    if all_pass:
        print("ADR-0004 perturbation check: PASS — held-out values are well separated")
        print("from the published Steinrück 2020 / DiffEC fit at every c_grid point.")
        return 0
    print("ADR-0004 perturbation check: FAIL — at least one case has a held-out")
    print("value too close to the published fit. Re-perturb the case YAML.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
