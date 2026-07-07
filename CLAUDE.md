# DiffEC → Terminal-Bench Science Task

> Contribute the "Differentiable modeling of concentrated-electrolyte mass
> transport from operando profiles" benchmark to Terminal-Bench Science.
> Build the oracle, 4 hidden cases, the verifier, and a reference solution;
> submit as a PR to `harbor-framework/terminal-bench-science`.

## Documentation Rule

**Keep CLAUDE.md condensed.** Detailed plans, oracle/verifier specs, case
designs, progress logs, and HPC reference material go in `docs/`:

- `docs/code_style.md` — behavioral guidelines to reduce common LLM coding mistakes (auto-imported in *Code Style & Conventions* below)
- `docs/proposal/` — accepted proposal package (do not edit; treat as the contract)
  - `proposal.md`, `formalism.md` — what we promised to TB-Science
  - `review_llm.md`, `review_expert.md`, `decision.md` — review trail
- `docs/plan/` — design documents and implementation plans
  - `plan/architecture.md` — target tree (upstream Harbor paths), module responsibilities, data flow
  - `plan/decisions.md` — ADR-style log of design decisions + rationale
  - `plan/build-and-run.md` — env setup (dev + Harbor containers), oracle smoke test, case regeneration, verifier run, `harbor` CLI validation
  - `plan/harbor-task-format.md` — pinned Harbor task format spec (refreshed 2026-06-22)
  - `plan/case-design.md` — the 4 cases: D(c), t⁺⁰(c), concentration ranges, noise, intended failure modes
  - `plan/oracle-spec.md` — moving-frame PDE solver spec (frame, BCs, IC, sign conventions, numerical scheme)
  - `plan/verifier-spec.md` — 5 checks, tolerance feasibility argument, anti-cheat self-consistency, `test.sh`/`reward.txt` wrapper
- `docs/progress/` — what we tried, outcomes, lessons learned
  - `progress/key-facts.md` — non-obvious gotchas (moving vs lab frame, sign of v₀, t⁺⁰_NE inversion, case_4 basin-trap-intent postmortem)
- `docs/session.md` — running experience notebook (gitignored, updated every work chunk)
- `docs/hpc/artemis.md` — Artemis cluster + Slurm reference (oracle generation may use a CPU node)

**Always document new experiments or plans in the appropriate `docs/` subfolder
before or after implementation.** Never let CLAUDE.md grow into a status report —
drop a pointer here, write the detail in `docs/`.

## Project Goal

Ship a complete, mergeable Harbor task that:

1. Bundles 4 oracle-generated cases of operando-style `c(x, t)` and `v(x, t)`
   data from a symmetric Li | electrolyte | Li cell.
2. Asks the agent to recover the concentration-dependent transport properties
   `D(c)` and `t⁺⁰(c)` (and the Nernst-Einstein-equivalent `t⁺⁰_NE(c)`,
   regime labels, predicted velocity field, and cation flux decomposition)
   under Newman's concentrated-solution theory.
3. Verifies the agent's output with a deterministic pytest verifier
   (tolerances + categorical pattern + physics-consistency + self-consistency).
4. Ships a reference agent solution that reliably passes the verifier within
   the 5–10 min/case CPU budget.

**Success criteria:**
- Reference solution passes all 7 verifier checks on all 4 cases on a fresh
  4–8 CPU machine in under 1 hour total wall time (locally: 28/28 in ~11 min).
- Frontier-agent pilot run (Claude Opus 4.7, GPT-5, Gemini 2.5) returns a
  solve rate in the proposal's 10–20 % target band.
- PR opens cleanly against `harbor-framework/terminal-bench-science/main`,
  honors the Harbor task format, passes upstream CI.
- **Deadline:** August 17, 2026 (from `docs/proposal/decision.md`).

## The Task (one-screen summary)

> Authoritative spec is `docs/proposal/formalism.md`. Read it before coding.

### Inputs (given to the agent, per case)
- `data.h5` — `c_data[Nt, Nx]` (mol/L), `v_data[Nt, Nx]` (nm/s), `x[Nx]` (m), `t[Nt]` (s); `Nx ≈ 100`, `Nt ≈ 50`.
- `params.json` — `L`, `T`, `i_app(t)`, `c0`, `V_bar`, `T_temp`, `c_grid[50]`, `flux_samples[10][2]`, plus a tabulated `(1 − d ln c₀ / d ln c)` factor and `c_init`.
- `formalism.md` — moving-frame PDE spec, regime rule, output schema.

### Output (agent writes, per case)
`results/case_X/transport.json` with: `D[50]`, `t_plus_0[50]`, `t_plus_0_NE[50]`, `regime[50]`, `v_pred[Nt][Nx]`, `flux_decomposition[10]`.

### Verifier (deterministic pytest, 5 checks)
1. `|D_agent − D_oracle| / D_oracle ≤ 0.10` at every c_grid point.
2. `|t⁺⁰_agent − t⁺⁰_oracle| ≤ 0.05` at every c_grid point.
3. Exact match on the 200 regime labels.
4. `||v_pred − v_data||₂ / max|v_data| ≤ 0.15`.
5. Flux decomposition at 10 points: each component within `0.15 · |J_total_oracle|`.
6. Self-consistency: verifier reruns its own moving-frame solver from the
   agent's reported `(D, t⁺⁰)` and rechecks #4. Catches "right answer,
   wrong physics" and parameter-lookup cheats.

### The 4 cases (intended failure modes)
- **Case 1 (NE-valid):** weak ion-solvent correlation; `t⁺⁰ > 0` everywhere; lab-frame agents pass.
- **Case 2 (NE-deviates):** moderate correlation; signs agree but magnitudes diverge; lab-frame agents fail check #1 or #5.
- **Case 3 (NE-wrong-sign):** Steinrück-2020-like sign flip at high c; lab-frame agents fail catastrophically; the headline case.
- **Case 4 (NE-wrong-sign, high c):** deeply negative `t⁺⁰` at higher concentration (c_grid ≈ [2.93, 3.20], `t⁺⁰` ∈ [-0.20, -0.18]). Companion to case 3; lab-frame agents fail catastrophically on checks #2 and #6. Originally scoped as a multi-modal basin trap but the v-data-weighted joint inverse reliably finds the correct basin from a literature-prior init.

## Design Constraints

- **Authoritative reference:** `Mass Transport in Concentrated Electrolytes and Benchmarks/solver.py` (the published moving-frame finite-volume + JAX solver). Treat the oracle as a parameterized generalization of that solver, not a rewrite from scratch.
- **Harbor task format:** the upstream repo's task spec is the boundary. Pull the latest format from `https://github.com/harbor-framework/terminal-bench-science` and `harborframework.com/docs/task-format` *before* finalizing the tree layout; record findings in `docs/plan/harbor-task-format.md`.
- **Compute budget for agents:** 4–8 CPU cores, 8–16 GB RAM, no GPU, 5–10 min/case, < 1 h total. Oracle generation, verifier, and reference solution must all live inside this envelope (the verifier itself is much tighter).
- **Python only.** No Julia, no MATLAB, no proprietary licenses. Standard scientific stack (NumPy/SciPy/JAX).
- **Hidden ground truth.** Oracle `D(c)` and `t⁺⁰(c)` parameter functions and noise seeds must never be checked in alongside the bundled cases. Separate the `oracle/` directory (hidden, ships with the verifier) from `cases/` (visible, ships with the agent).
- **Determinism.** Case generation is seeded; the verifier is seeded; both produce byte-identical outputs across re-runs on the same machine.
- **Anti-cheat.** No literature values for this exact system can pass: the held-out parameter functions and noise seeds must be perturbed from any published values (Pesko 2017, Steinrück 2020, the DiffEC repo's PEO-LiTFSI fit).

## Architecture

> Full target tree and abstractions: `docs/plan/architecture.md`.
> Harbor task format pinned in `docs/plan/harbor-task-format.md`.

We develop directly at the upstream PR path inside this repo (ADR-0009):

```
tasks/physical-sciences/chemistry/<task-name>/
├── instruction.md            # agent prompt
├── task.toml                 # schema_version = "1.0"
├── environment/              # agent container
│   ├── Dockerfile
│   └── data/cases/case_{1..4}/{data.h5, params.json, formalism.md}
├── solution/                 # reference solution (Oracle-agent only)
│   ├── solve.sh
│   └── reference_solver.py + helpers
└── tests/                    # verifier container (separate; environment_mode = "separate")
    ├── Dockerfile
    ├── test.sh               # pytest → /logs/verifier/reward.txt
    ├── test_outputs.py
    ├── oracle/               # held-out moving-frame solver + flux + NE inversion
    └── oracle_truth/case_{1..4}/truth.npz
```

Out-of-tree dev tooling at the repo root (not shipped upstream):

- `case_gen/` — config-driven case generator. Writes directly into `tasks/.../environment/data/cases/` and `tasks/.../tests/oracle_truth/`.
- `scripts/smoke_test.py` — ADR-0002 cross-check (reproduce the published Steinrück fit).
- `scripts/pilot/` — ADR-0008 frontier-agent pilot driver, built on the `harbor` CLI.

**Design principle:** every case is fully reproduced by `(case_config.yaml, seed)`.
The oracle forward solver (under `tests/oracle/`) is a pure function
`(params, config, seed) → (c, v, truth)`. The verifier is a pure
function `(agent_output, truth) → pass/fail breakdown`. Both are seeded
and reproducible. The same `tests/oracle/` Python package is imported by
`case_gen/` (to generate ground truth) and by the verifier (to grade) —
disagreement between case generation and self-consistency check is
impossible by construction.

## Build & Run

> Full setup, oracle smoke test, case regeneration, and verifier run:
> `docs/plan/build-and-run.md`.

**Dev environment: `uv` only. Harbor task containers: built by `harbor`,
pinned to the same JAX/NumPy versions as `uv.lock`.**

Let `TASK = tasks/physical-sciences/chemistry/<task-name>`.

```bash
# One-time dev setup
uv sync

# Regenerate all 4 cases from configs (deterministic, ~5 min on a laptop)
uv run python -m case_gen.generate --all

# Run the verifier locally against the reference solution (must pass cleanly)
uv run python "$TASK/solution/reference_solver.py" \
    --cases "$TASK/environment/data/cases" --out ./_local_results
RESULTS_DIR=./_local_results uv run pytest "$TASK/tests/test_outputs.py" -v

# The real pre-PR check (containerized; what upstream CI runs)
harbor run -p "$TASK" -a oracle           # must return reward = 1
```

For the frontier-agent pilot run, two entry points depending on
scope:

- **Mock exam** (single-trial-per-agent sanity check, run BEFORE
  the pilot): `mock_exam/README.md`. Verifies that Harbor's
  `environment_mode="separate"` container isolation actually keeps
  our reference solution, oracle, docs, and CLAUDE.md out of the
  agent's view — a real cold-start attempt.
- **Full pilot** (10-trial × 3-agent solve-rate measurement per
  ADR-0008): `docs/laptop-runbook.md` Part 2 is the authoritative
  driver; `scripts/pilot/README.md` is the per-script reference.
  Both cover subscription auth (Claude Max/Pro, ChatGPT Plus/Pro,
  Gemini free tier) + paid-API-key fallback.

`docs/plan/build-and-run.md` §7 has historical context.

## Code Style & Conventions

@docs/code_style.md

Project-specific additions:
- **Config-driven everything.** Each case's true `D(c)`, `t⁺⁰(c)`, current schedule, concentration range, noise, and seed live in a single YAML under `tb_sci_task/case_gen/configs/`. No hard-coded constants in `oracle/`.
- **Pure-Python physics.** `oracle/` is testable in isolation: import, call, get arrays back. No CLI parsing, no file I/O hidden inside.
- **Sign and frame conventions** are written *once*, in `docs/plan/oracle-spec.md`, and quoted verbatim into `formalism.md`. The verifier's check #5 (flux decomposition) only works if the agent's convention matches the oracle's — pin it down up front, never reinterpret later.
- **Type hints** on all public functions; document array shapes and units in docstrings.
- **No silent coercion at boundaries.** If the agent's `transport.json` is missing a field, has the wrong shape, or has NaNs, the verifier reports a structured failure — never falls back to defaults.
- **Determinism.** Every RNG is seeded from the case config. Re-running case generation produces byte-identical `data.h5`.
- **Tolerance feasibility is a precondition, not an aspiration.** Before locking the case design, demonstrate the reference solution passes the verifier with margin. Document margins in `docs/progress/key-facts.md`.
- **Code comments explain *why*, not *what*.** Update `docs/` in the same commit as the code change.

## HPC Interaction

Mostly local. The agent-side budget is laptop-scale by design.

- Case generation and reference-solution smoke tests run locally.
- Frontier-agent pilot runs (Claude Opus 4.7, GPT-5, Gemini 2.5) may use
  `venkvis-cpu` with `--qos=venkvis-short`; one node per case is enough.
- See `docs/hpc/artemis.md` for the cluster reference if needed.

## Workflow

- Work on `feat/tb-sci-task`; small, focused commits.
- Harbor task format is pinned in `docs/plan/harbor-task-format.md` (refreshed 2026-06-22). Refresh it again before opening the PR — if upstream `CONTRIBUTING.md` HEAD has moved, reconcile.
- Major design choices (case parameterization, noise level, verifier tolerance band, anti-cheat strategy) get a short ADR in `docs/plan/decisions.md` *before* implementation.
- When uncertain about the **physics** (frame, sign of `v₀`, definition of `t⁺⁰_NE`, flux decomposition convention), check `Mass Transport in.../solver.py` and Chen et al. 2026 first; ask if still ambiguous.
- When uncertain about the **task contract** (what the agent sees vs. what the verifier holds), default to the most restrictive reading of `docs/proposal/formalism.md`.
- Before any commit that touches `tests/oracle/` or `tests/test_outputs.py`, regenerate cases and run the verifier locally against the reference solution.
- Before opening the PR, `harbor run -p "$TASK" -a oracle` must return reward = 1.

## Out of Scope

- Anything outside the 4 bundled cases: real experimental data fitting, new chemistries, new physics regimes.
- Porting the existing `solver.py` to a different language or framework.
- Web UI, visualization beyond diagnostic plots.
- Optimizing the agent's solution beyond "convincingly passes the verifier".
- Soliciting agent submissions or grading them ourselves — that's TB-Science's job once the task is merged.

## Definition of Done

The task is mergeable when:
1. The local verifier passes on all 4 cases against the reference solution, on a fresh checkout, in under 1 hour on 4–8 CPU cores: `RESULTS_DIR=./_local_results uv run pytest "$TASK/tests/test_outputs.py" -v` returns 28/28 green (7 active checks × 4 cases).
2. `harbor run -p "$TASK" -a oracle` returns reward = 1.
3. `docs/plan/case-design.md` documents each case's true parameter functions, noise level, concentration range, and the failure mode it is designed to catch — and the reference-solution margins on each check are recorded in `docs/progress/key-facts.md`.
4. The bundled `environment/data/cases/case_X/` directories contain only what the agent should see; no leaked ground-truth files. The verifier image holds `tests/oracle_truth/` and `tests/oracle/` but never reaches the agent (Harbor's separate-container enforcement).
5. The held-out oracle parameters are demonstrably different from any published values for the PEO-LiTFSI system (Steinrück 2020, Pesko 2017, DiffEC paper, DiffEC repo results).
6. The frontier-agent pilot returns solve-rate evidence consistent with the proposal's 10–20 % band, recorded in `docs/progress/pilot_run.md`. **Interim status** (2026-07-06): direct-CLI mock pilot on Artemis ran 11 Opus + 2 Sonnet + 1 gpt-5.5 attempts, with 0 solves across all attempts (Wilson 95 % CI 0.0-25.9 % on N=11 Opus), mean check-pass 45–57 % depending on subset. This is consistent with the 10–20 % band but does not confirm it; the Harbor-driven 3-agent × 10-trial run from `docs/laptop-runbook.md` Part 2 remains the authoritative check.
7. PR title `[TASK: Chemistry] …` references discussion #335; body includes `harbor analyze` output and an Oracle-pass screenshot; upstream CI (static checks, rubric review, execution checks, three-pass human review) is green.

## Experience Notebook

`docs/session.md` is the running experience notebook — treat it as authoritative
session state. Sessions can die unexpectedly; a fresh Claude Code instance must
be able to resume cold by reading that file alone + `git status`.

**Update it when you:** finish a coherent chunk of work; make a non-obvious
decision; launch long-running runs; observe anomalies (even if unresolved).

**Each entry covers:** date + chunk name; actions taken; results (numbers beat
prose); decisions / rationale; open items.

A crash-recovery checklist lives at the bottom of `docs/session.md`.
`docs/session.md` is gitignored — write freely.
