# Build & Run

> Single source of truth for environment setup, oracle smoke test, case
> regeneration, verifier run, Harbor validation, and the frontier-agent
> pilot procedure. Task tree paths follow ADR-0009 (we develop at the
> upstream PR path inside this repo).
>
> Let `TASK = tasks/physical-sciences/chemistry/<task-name>` throughout.

## 1. Environment

Two environments:

**Dev environment** (this repo's `uv`): for `case_gen/`, the smoke test,
and local pytest runs.

```bash
# One-time setup
uv sync

# Sanity-check the interpreter and JAX
uv run python -c "import jax; jax.config.update('jax_enable_x64', True); \
    print(jax.devices(), jax.numpy.array([1.0]).dtype)"
# expect: [CpuDevice(id=0)] float64
```

Dependencies (dev `pyproject.toml`):

- `numpy`, `scipy`, `h5py`, `pyyaml`
- `jax`, `jaxopt`
- `pytest`, `pytest-xdist` (parallelize cases)
- `matplotlib` (diagnostic plots only; not required by the verifier)

Pin versions in `uv.lock`. Do not bump JAX without rerunning the oracle
smoke test (ADR-0002 — solver numerics are tied to a specific JAX version).

**Harbor task environments** (containers): the agent and verifier
containers each have their own `Dockerfile` (`$TASK/environment/Dockerfile`
and `$TASK/tests/Dockerfile`). These are built by Harbor, not by `uv`.
Pin the same JAX/NumPy/SciPy versions across all three (dev `uv`, agent
image, verifier image) — drift will make `harbor run -a oracle` reward
inconsistent with local pytest results.

**Harbor CLI** (for pre-PR validation): install per upstream Harbor docs.
Not pinned in `uv.lock` — it's an external dev tool.

## 2. Oracle smoke test

Before trusting the oracle, reproduce the published Steinrück-2020 fit
from the public DiffEC repo using the **same** parameters as
`Mass Transport in Concentrated Electrolytes and Benchmarks/`.

```bash
uv run python scripts/smoke_test.py
# expect: max relative error vs published c_data < 5% at all 9 time points
```

This is the cross-check called out in ADR-0002. If it fails, the oracle
has a bug; do not generate cases until it passes.

## 3. Regenerate cases

Cases are deterministic from `(config.yaml, seed)`. The generator writes
directly into the upstream-shipped task tree.

```bash
# Regenerate all 4 cases (configs + seeds → environment/data/cases + tests/oracle_truth)
uv run python -m case_gen.generate --all

# Regenerate a single case (faster iteration during case design)
uv run python -m case_gen.generate --case case_3
```

Expected runtime: ~30 s per case on 8 CPU cores. Outputs:

- `$TASK/environment/data/cases/case_X/{data.h5, params.json, formalism.md}` (agent-visible)
- `$TASK/tests/oracle_truth/case_X/truth.npz` (verifier-only)

Both directories are tracked by git (ADR-0007). After regeneration,
`git diff --stat` should be empty unless `case_gen/configs/case_X.yaml`
changed.

## 4. Run the reference solution (locally, without Harbor)

For fast iteration during reference-solution development:

```bash
# Solve all 4 cases using the same Python code that solve.sh will invoke
uv run python "$TASK/solution/reference_solver.py" \
    --cases "$TASK/environment/data/cases" --out ./_local_results

# Solve a single case
uv run python "$TASK/solution/reference_solver.py" \
    --cases "$TASK/environment/data/cases" --case case_3 --out ./_local_results
```

Expected runtime: 3–7 min per case on 8 CPU cores (well inside the 5–10
min/case budget). Writes `./_local_results/case_X/transport.json`.

`./_local_results/` is gitignored — it's a scratch directory for local
iteration, not the canonical output location (which is `/root/results/`
inside the agent container).

## 5. Run the verifier (locally, without Harbor)

```bash
# Point the verifier at the local results directory
RESULTS_DIR=./_local_results uv run pytest "$TASK/tests/test_outputs.py" -v

# Single check, single case (fast iteration)
RESULTS_DIR=./_local_results uv run pytest \
    "$TASK/tests/test_outputs.py::test_D" -v -k case_3
```

`test_outputs.py` reads the agent's output from `$RESULTS_DIR` (defaults
to `/root/results/` inside the container, but local runs override via
env var). Expected result: **all 20 tests pass**, with margins recorded
in `docs/progress/key-facts.md`. If any check fails on the reference
solution, the case design or the verifier tolerance is mis-calibrated
(ADR-0005); fix before continuing.

## 6. Run under Harbor (the real pre-PR validation)

This is what upstream CI runs. Both containers are built; the agent runs
under the Oracle to invoke `solution/solve.sh`; the verifier container
grades the output via the `artifacts` declared in `task.toml`.

```bash
# Oracle must achieve reward = 1
harbor run -p "$TASK" -a oracle

# Static rubric check (LLM judge against rubrics/task-implementation.toml)
harbor check -r rubrics/task-implementation.toml \
    -m anthropic/claude-opus-4-8 "$TASK"

# Interactive debugging (drops you into the agent container)
harbor tasks start-env -p "$TASK" -e docker -a -i

# Pilot a real agent (one of: claude-opus-4-7, gpt-5, gemini-2.5)
harbor run -p "$TASK" -a <agent> -m <provider/model>

# Aggregate failure analysis (used in the PR body)
harbor analyze "$TASK"
```

If `harbor run -a oracle` does not return reward = 1, the task is
broken and we do not open the PR.

## 7. Frontier-agent pilot

> Procedure for the empirical solve-rate measurement promised in the
> proposal. Run *after* the verifier passes on the reference solution
> AND `harbor run -a oracle` passes.

Use Harbor's own runner (it knows how to mount the task in the right
container, route traffic, enforce timeouts, and write `reward.txt`).
Loop over models and seeds:

```bash
for model in anthropic/claude-opus-4-7 openai/gpt-5 google/gemini-2.5-pro; do
  for seed in 0 1 2 3 4; do
    harbor run -p "$TASK" -a <agent-template> -m "$model" \
        --seed "$seed" --out "docs/progress/pilot/${model//\//_}_seed${seed}.json"
  done
done

# Aggregate
uv run python scripts/pilot/aggregate.py docs/progress/pilot/ \
    > docs/progress/pilot_run.md
```

For each (agent, case, seed) record:

- per-check pass/fail (from `reward.json` / Harbor logs)
- wall-time per case
- token count and cost
- failure mode classification (lab-frame? wrong basin? missing decomposition?)

Aggregate solve rate is reported in `docs/progress/pilot_run.md`.
Target band per ADR-0008: **10–20 %**.

## 8. Submitting the PR

```bash
# Final pre-PR smoke test on a fresh checkout
git clean -xdf && uv sync
uv run python -m case_gen.generate --all

# Local sanity
RESULTS_DIR=./_local_results uv run pytest "$TASK/tests/test_outputs.py" -v

# Real Harbor validation (Oracle must return reward = 1)
harbor run -p "$TASK" -a oracle
harbor check -r rubrics/task-implementation.toml \
    -m anthropic/claude-opus-4-8 "$TASK"
```

PR title (per upstream CONTRIBUTING.md):
`[TASK: Chemistry] Differentiable modeling of concentrated-electrolyte mass transport from operando profiles`

PR body must include:
- Link to the approved proposal discussion (`#335`) — the
  `check-task-proposal-link` static check blocks merge if absent or if
  the proposal lacks the `proposal-approved` label.
- Output of `harbor analyze "$TASK"` (failure-mode summary from pilot).
- Screenshots / log excerpts showing the Oracle pass.

Open the PR against `harbor-framework/terminal-bench-science:main`.

## 9. Crash recovery

If a long-running pilot or case-gen run dies, see the recovery checklist
at the bottom of `docs/session.md`. Key invariants:

- `case_gen/generate.py` is idempotent; rerun freely.
- The verifier reads only from `tests/oracle_truth/` and the agent's
  `results/` directory; safe to rerun.
- `solution/reference_solver.py` writes atomically (write-then-rename)
  to avoid half-written `transport.json` files. If you see a malformed
  JSON, delete it and rerun.
