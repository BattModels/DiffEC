# Architecture

> The Harbor task format is confirmed in `harbor-task-format.md` (refreshed
> 2026-06-22). The tree below uses the upstream-mandated layout. Paths
> shown are the **upstream PR target paths** — during development we work
> on the same tree inside this repo, ready to lift into the upstream
> `tasks/` directory unchanged.

## Target tree

Upstream path: `tasks/physical-sciences/chemistry/<task-name>/` (kebab-case
name to be locked; candidates in `harbor-task-format.md`).

```
tasks/physical-sciences/chemistry/<task-name>/
├── instruction.md                  # agent prompt; absolute paths, ends with deadline + no-cheat line
├── task.toml                       # Harbor manifest, schema_version = "1.0"
│
├── environment/                    # AGENT container build context
│   ├── Dockerfile                  # ubuntu:24.04 + Python + JAX + h5py + pyyaml + numpy + scipy
│   └── data/
│       └── cases/
│           ├── case_1/{data.h5, params.json, formalism.md}    # bundled, agent-visible
│           ├── case_2/ …
│           ├── case_3/ …
│           └── case_4/ …
│
├── solution/                       # reference solution (Oracle agent only)
│   ├── solve.sh                    # entrypoint, run from /root/
│   ├── reference_solver.py         # differentiable JAX moving-frame solver + multi-start BFGS
│   ├── lab_frame_solver.py         # used to produce t⁺⁰_NE
│   └── parameterize.py             # D(c), t⁺⁰(c) parameterization
│
└── tests/                          # VERIFIER container build context (separate from agent)
    ├── Dockerfile                  # COPY . /tests/ ; pre-install pytest + JAX + oracle deps
    ├── test.sh                     # pytest wrapper → /logs/verifier/reward.txt
    ├── test_outputs.py             # the 5 checks + self-consistency
    ├── conftest.py                 # case parameterization + agent-output loader
    ├── oracle/                     # Python package — moving-frame solver, flux decomposition, NE inversion
    │   ├── __init__.py
    │   ├── solver.py               # moving-frame finite-volume forward simulator (JAX)
    │   ├── flux.py                 # canonical J_diff/J_mig/J_conv decomposition (ADR-0003)
    │   └── invert_ne.py            # NE-inversion definition of t⁺⁰_NE
    └── oracle_truth/               # held-out ground truth, baked into the verifier image
        ├── case_1/truth.npz        # D_oracle, t⁺⁰_oracle, t⁺⁰_NE_oracle, regime, v field, fluxes, seed, config hash
        ├── case_2/ …
        ├── case_3/ …
        └── case_4/ …
```

## Out-of-tree development tooling

Lives at the repo root, not shipped to upstream. Used for case generation
and pre-PR validation:

```
case_gen/                           # dev-only; not in the upstream PR
├── configs/
│   ├── case_1.yaml                 # NE-valid
│   ├── case_2.yaml                 # NE-deviates
│   ├── case_3.yaml                 # NE-wrong-sign
│   └── case_4.yaml                 # multi-modal
├── generate.py                     # configs → environment/data/cases/ + tests/oracle_truth/
└── writers.py                      # data.h5 / params.json / formalism.md emitters

scripts/
├── litvalue_distance.py            # ADR-0004 perturbation check
├── smoke_test.py                   # ADR-0002 cross-check vs published Steinrück fit
└── pilot/                          # frontier-agent pilot driver (ADR-0008)
```

`case_gen/` writes directly into the two upstream-shipped trees:
`tasks/.../environment/data/cases/` and `tasks/.../tests/oracle_truth/`.
That keeps the upstream tree the single source of truth for what gets
PR'd.

## Module responsibilities

### `tests/oracle/` (held-out Python package)
The forward model. Pure JAX, fully `jit`-able, no I/O. Lives inside
`tests/` so it ships only in the verifier container, not the agent's.
- `solver.simulate(D_fn, tp0_fn, ic, bc, t_grid, x_grid, seed) → (c, v0)` — moving-frame finite-volume forward simulator.
- `solver.simulate_ne(D_fn, tp0_NE_fn, ic, bc, …) → c` — lab-frame variant with `v₀ ≡ 0`, used both for case generation's NE field and for any verifier check that needs the lab-frame counterfactual.
- `invert_ne.invert(c_data, D_fn) → tp0_NE_fn` — the canonical definition of `t⁺⁰_NE` (see `oracle-spec.md` §3).
- `flux.decompose(D_fn, tp0_fn, c, v0, i, x, t, factor) → (J_diff, J_mig, J_conv)` — single source of truth for the decomposition convention (ADR-0003). Imported by `case_gen/` and `tests/test_outputs.py`; the agent must match the formulas in `instruction.md`.

The exact same Python package is imported by `case_gen/` to generate
ground truth and by `tests/test_outputs.py` to verify — so the
self-consistency check (#6) cannot disagree with case generation by
construction.

### `case_gen/` (out-of-tree dev tooling)
Pure orchestration. Reads a YAML config, drives `tests/oracle/`, writes
two outputs:
- bundled agent inputs at `tasks/.../environment/data/cases/case_X/{data.h5, params.json, formalism.md}`
- held-out ground truth at `tasks/.../tests/oracle_truth/case_X/truth.npz`

`generate.py --all` is idempotent and seeded.

### `tasks/.../tests/` (verifier)
`test_outputs.py` is the pytest verifier. Each test:
1. Loads `oracle_truth/case_X/truth.npz` from `/tests/oracle_truth/`.
2. Loads `results/case_X/transport.json` (transferred from the agent per the `artifacts` list in `task.toml`).
3. Applies one check class.

Parameterization: `@pytest.mark.parametrize("case", ["case_1", …, "case_4"])`.

The self-consistency check (#6) is the only one that re-invokes
`tests/oracle/solver.py`. `test.sh` wraps pytest and writes a single
integer to `/logs/verifier/reward.txt` (1 = all 20 checks pass, 0 = any
fail). See `harbor-task-format.md` for the wrapper pattern.

### `tasks/.../solution/` (reference solution)
Ships in the task tree. `solve.sh` is Harbor's Oracle-agent entrypoint:
```bash
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python "$DIR/reference_solver.py" --cases /root/data/cases --out /root/results
```
The Python implementation is the differentiable JAX moving-frame solver
+ multi-start BFGS. It must:
- run end-to-end on a fresh checkout in under 1 h,
- pass all 5 checks on all 4 cases with the margin recorded in `docs/progress/key-facts.md`,
- be the proof that the task is solvable as designed (`harbor run -a oracle` must return reward = 1).

It also serves as the reference implementation that frontier agents will
be measured against in the pilot run.

## Data flow

```
case_gen/configs/case_X.yaml
        │
        ▼
case_gen/generate.py ──► tests/oracle/solver.simulate ──► c, v
                    │
                    ├── tests/oracle/noise.add ────► c_data, v_data
                    │                                  │
                    │                                  ▼
                    │      tasks/.../environment/data/cases/case_X/{data.h5, params.json, formalism.md}
                    │
                    └── tests/oracle/invert_ne.invert(c_data, D_oracle) ──► t⁺⁰_NE_oracle
                                                                             │
                          ┌──────────────────────────────────────────────────┘
                          ▼
                  tasks/.../tests/oracle_truth/case_X/truth.npz
                  { D_oracle, tp0_oracle, tp0_NE_oracle,
                    regime, v_field, flux_components, seed, config_hash }


At runtime, inside Harbor's containers:

  AGENT container (built from environment/Dockerfile)
  ───────────────────────────────────────────────────
  /root/data/cases/case_X/{data.h5, params.json, formalism.md}    ← input
  /root/results/case_X/transport.json                              ← output

           │ (Oracle mode: /solution/solve.sh runs reference_solver.py)
           │ (Real agents: their own code)
           ▼
  Harbor transfers /root/results/case_X/transport.json (declared in `artifacts`)
  to the verifier container.
           │
           ▼
  VERIFIER container (built from tests/Dockerfile)
  ────────────────────────────────────────────────
  /tests/oracle_truth/case_X/truth.npz                             ← held-out ground truth
  /root/results/case_X/transport.json                              ← agent's output (mounted)
  /tests/oracle/…                                                  ← oracle solver for check #6
           │
           ▼
  test.sh → pytest test_outputs.py → /logs/verifier/reward.txt (0|1)
```

## Invariants

- **Frame convention** is set in `tests/oracle/solver.py` once; both case generation and the verifier's self-consistency check invoke the same function. Disagreement is impossible by construction.
- **Flux decomposition convention** is set in `tests/oracle/flux.py` once; both case generation (to fill `truth.npz`) and the verifier import it. The agent must match the formulas as written in `instruction.md`; the reference solution under `solution/` is a separate clean-room implementation to prove that's possible.
- **No oracle leak.**
  - `tasks/.../environment/` (the agent's container build context) contains *only* `data/cases/` and the `Dockerfile`. Per the Harbor CONTRIBUTING.md: "Do not copy solution or test files into the container."
  - `tasks/.../tests/` (verifier container) contains `oracle/`, `oracle_truth/`, and the test scripts — these never reach the agent. Harbor enforces container isolation; we double-check at PR time by greplng `environment/` for any string that appears only in `tests/oracle_truth/`.
  - `tasks/.../solution/` only runs under Harbor's Oracle agent. Real agents never see it.
- **Reproducibility.** Every `truth.npz` records the `config_hash` and `seed`. Re-running `case_gen/generate.py --case case_X` with the same config must produce a byte-identical artifact. Pre-PR CI in `docs/plan/build-and-run.md` enforces this.

## Open questions (resolve before locking)

1. **Final `<task-name>` (kebab-case).** Candidates in `harbor-task-format.md`: `concentrated-electrolyte-mass-transport`, `diffec-mass-transport`, `newman-inversion-from-operando`. Decide and record in an ADR before laying down the `tasks/` directory.
2. **Verifier-image size.** Baking JAX + the held-out solver + `oracle_truth/` into `tests/Dockerfile` will be a couple hundred MB. Acceptable per the upstream norm (the example image is similar). Confirm under `harbor run` timing.
3. **`allow_internet` in `[environment]`.** Setting `true` lets the agent install extra packages but also lets it call out to the web. We need agents to be able to install (the existing TB-Science example uses `allow_internet = true`), and per ADR-0004 the held-out parameters resist literature-lookup attacks anyway — confirm but tentatively keep `true`.
