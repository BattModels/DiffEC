# Design Decisions (ADR log)

> Light ADR format. Each entry: **Context → Decision → Consequences**. When a
> decision is overturned, append a new ADR; never edit history.

---

## ADR-0001 — Adopt Harbor task format as the boundary contract

**Status:** Accepted (2026-06-22); refreshed (2026-06-22)

**Context.** The accepted proposal commits us to submit a PR against
`harbor-framework/terminal-bench-science`. The repo expects a specific task
layout (see `harbor-task-format.md`). Diverging from it would mean a rejected
PR.

**Decision.** Treat the upstream task format as load-bearing. We pinned the
format from upstream `CONTRIBUTING.md`, `harborframework.com/docs/task-format`,
and the example task
`tasks/physical-sciences/chemistry/geometric-pharmacophore-alignment/` on
2026-06-22 (see `harbor-task-format.md`). The working tree mirrors the
upstream layout 1:1 so we can lift it into `tasks/physical-sciences/chemistry/<task-name>/`
unchanged.

Concretely:
- **`schema_version = "1.0"`** in `task.toml`, matching the TB-Science
  example — *not* the `"1.3"` shown on `harborframework.com`, which
  describes a newer Harbor that the TB-Science repo isn't on.
- Two separate containers: `environment/Dockerfile` (agent) and
  `tests/Dockerfile` (verifier). `environment_mode = "separate"`.
- Agent reads from `/root/data/`, writes to `/root/results/`. Output paths
  declared in `artifacts = […]`.
- Verifier signals via `/logs/verifier/reward.txt` (single integer 0 or 1).
- Reference solution lives in-tree under `solution/`, gated by Harbor's
  Oracle agent.

**Consequences.** No "looks right but doesn't match upstream" risk. The
oracle solver package moves under `tests/oracle/` so it ships with the
verifier image; case ground truth ships at `tests/oracle_truth/`. Refresh
`harbor-task-format.md` once more before opening the PR — if upstream
`CONTRIBUTING.md` HEAD has moved, reconcile.

---

## ADR-0002 — Adapt, don't rewrite, the existing DiffEC moving-frame solver

**Status:** Accepted (2026-06-22)

**Context.** `Mass Transport in Concentrated Electrolytes and Benchmarks/solver.py`
is the published, peer-reviewed forward simulator that produced the figures
in Chen et al. 2026. It is JAX-native, `jit`-compatible, and already
implements the moving-frame finite-volume discretization the task requires.

**Decision.** The oracle's forward solver is a parameterized generalization
of `solver.py`, not a fresh implementation. We lift the discretization,
the volume-average / interface stencil, and the sign conventions verbatim;
we generalize:
- `D(c)` from the hard-coded `D_xp/D_fp` table to a configurable functional form,
- `t⁺⁰(c)` from the published linear polynomial to a configurable form,
- the current schedule, BCs, IC concentration, and `(1 − d ln c₀/d ln c)` factor to per-case inputs.

**Consequences.** Faster path to a working oracle; lower risk of "right
equation, wrong sign" bugs that would invalidate the verifier; the case
generation inherits the solver's already-validated numerics. Downside: we
inherit any latent bugs in `solver.py` too — we cross-check by reproducing
the Steinrück 2020 fit from the public DiffEC results as part of the
oracle smoke test (see `build-and-run.md`).

---

## ADR-0003 — Single source of truth for the flux decomposition convention

**Status:** Accepted (2026-06-22)

**Context.** Reviewer feedback (`docs/proposal/review_llm.md`, "Well-Specified")
flagged that two reasonable implementers can produce different but
internally-consistent splits of the cation flux into diffusion / migration /
convection. The verifier's check #5 (flux decomposition) only works if the
agent's convention matches the oracle's. The proposal's flux definitions
(`formalism.md` §3.4) need to be unambiguous.

**Decision.** `oracle/flux.py` defines the canonical decomposition as a
single Python function imported by both the case generator and the verifier.
`formalism.md` (shipped to the agent) restates the three formulas verbatim:
```
J_diff(x, t) = −D(c) (1 − d ln c₀ / d ln c) ∂c/∂x
J_mig (x, t) =  t⁺⁰(c) i(t) / F
J_conv(x, t) =  c v₀
```
with the moving-frame sign of `v₀` defined relative to the deposition
electrode at `x = 0`.

**Consequences.** Removes the specification risk the LLM reviewer flagged.
The agent has a single unambiguous rule to match. Cost: the agent must
read `formalism.md` carefully — which is the task, not a bug.

---

## ADR-0004 — Held-out parameters must be perturbed from any public values

**Status:** Accepted (2026-06-22)

**Context.** The reviewer flagged the risk that an agent with internet
access could lift `D(c)` and `t⁺⁰(c)` from the public DiffEC paper or
the Pesko 2017 / Steinrück 2020 literature and pass the verifier without
performing the inversion. The proposal's anti-cheat check (#6) catches
this only when the lifted parameters fail to reproduce `v_data` —
which happens for lab-frame parameters but not necessarily for
moving-frame literature values.

**Decision.** The 4 held-out `(D_oracle, t⁺⁰_oracle)` functions are each
constructed as a deliberate perturbation of literature values:
- different functional family (e.g., piecewise-cubic vs the published linear `t⁺⁰`),
- different concentration range (perturbed `c_avg`, `c_init`, current schedule),
- different magnitudes (target `|t⁺⁰_agent − t⁺⁰_oracle| ≤ 0.05` — so the held-out value must be at least 0.10 away from any published value).
Each case's `config.yaml` documents the perturbation versus the closest public reference.

**Consequences.** "Look up the answer" no longer suffices. The agent must
run the inversion. We pay a one-time cost: each case's design needs an
explicit lit-value-distance check.

---

## ADR-0005 — Tolerance feasibility is a precondition, not a hope

**Status:** Accepted (2026-06-22)

**Context.** Both reviewers flagged the 10 % relative tolerance on `D(c)`
and the 0.05 absolute tolerance on `t⁺⁰` as stringent — feasibility under
the injected measurement noise is not obvious.

**Decision.** Before locking any case design, the reference solution must
pass the verifier with **at least 50 % margin on the worst point of the
worst case** (e.g., max `|D_ref − D_oracle| / D_oracle ≤ 0.05` when the
threshold is 0.10). Margins per check per case are tabulated in
`docs/progress/key-facts.md` and re-measured on every case-design change.

**Consequences.** Eliminates the "intended solution can't reliably pass"
risk the reviewer flagged. Forces us to calibrate noise levels against
the inversion's information content rather than picking a number out of
the air.

---

## ADR-0006 — Python + JAX only

**Status:** Accepted (2026-06-22)

**Context.** Agents run under a 4–8 CPU / 8–16 GB budget with no GPU.
The published reference is JAX. PyTorch is an option; pure-NumPy/SciPy
is an option. Mixing languages adds packaging friction.

**Decision.** The oracle, case generator, verifier, and reference solution
are pure Python on the NumPy/SciPy/JAX stack. JAX runs CPU-only
(`JAX_PLATFORMS=cpu`) and `jax_enable_x64=True`, matching the published
implementation. No PyTorch, no Julia, no Numba.

**Consequences.** A single, well-known environment (`uv sync` reproduces).
Float64 + JAX gives us auto-diff for the reference solution and exactness
for the verifier. The agent is free to use whatever they want — this
decision constrains *our* code, not theirs.

---

## ADR-0007 — Bundled `.h5` files committed to git, not regenerated on agent start

**Status:** Accepted (2026-06-22; confirmed against upstream large-file policy)

**Context.** Each case is ~10–25 MB of HDF5; total ~50–100 MB. Options:
(a) commit the `.h5` files directly, (b) commit only `case_gen/configs/`
and let the verifier regenerate on first run, (c) git-LFS.

**Decision.** Option (a) — commit the `.h5` files directly under
`tasks/.../environment/data/cases/`. The agent should see the data, not
the generator. Regenerating on first run leaks the existence of `case_gen/`
to a curious agent and risks RNG drift across machines. Git-LFS adds
operational complexity for a 50–100 MB asset.

The upstream CONTRIBUTING.md large-file policy reads: *"Files >100MB
should not be committed — host the data on Hugging Face and use download
scripts in your task."* Our per-file size (~10–25 MB) is well under that
threshold, so direct commit is sanctioned.

**Consequences.** Repo size grows by ~100 MB once, all from
`environment/data/`. The `tests/oracle_truth/` directory adds another
~5–20 MB. If any single file ever exceeds 100 MB, switch to the
HuggingFace + download pattern from CONTRIBUTING.md.

---

## ADR-0009 — Develop in-repo at the upstream PR path

**Status:** Accepted (2026-06-22)

**Context.** With ADR-0001 pinning us to the Harbor task format and
ADR-0007 committing to direct-commit bundles, we have to decide where in
*this* repo the upstream-shipped files live during development.

**Decision.** Develop directly under
`tasks/physical-sciences/chemistry/<task-name>/` inside this repo, with
the exact tree that will land in the upstream PR. Out-of-tree dev tooling
(`case_gen/`, `scripts/`, `docs/`) stays at the repo root and is *not*
copied across in the PR — only `tasks/physical-sciences/chemistry/<task-name>/`
goes upstream.

Rationale: avoids a confusing "rename + move at PR time" step,
keeps the upstream paths visible during development, and makes
`harbor run -p tasks/physical-sciences/chemistry/<task-name>` work in
this repo without any path translation.

**Consequences.** The README / proposal docs / scripts at the repo root
look unrelated to the upstream `tasks/` tree at a glance; we explain
this in `CLAUDE.md` and the eventual PR description. Anyone reading the
upstream PR sees only the task tree, which is the right view for them.


## ADR-0010 — Pin Harbor task `schema_version = "1.0"`

**Status:** Accepted (2026-06-22)

**Context.** The Harbor docs site (`harborframework.com/docs/task-format`)
describes `schema_version = "1.3"` with richer fields (`network_mode`,
`allowed_hosts`, `[environment.tpu]`, `[[environment.mcp_servers]]`).
The TB-Science example
`tasks/physical-sciences/chemistry/geometric-pharmacophore-alignment/task.toml`
uses `schema_version = "1.0"` with the older `allow_internet = true`
form. The two don't validate to the same schema.

**Decision.** Match the TB-Science example (`"1.0"`). The upstream CI
validates against whatever `harbor check` accepts at the TB-Science
repo's HEAD, not against the newer Harbor docs.

**Consequences.** We can't use newer manifest fields (e.g.,
`allowed_hosts`). If the upstream repo upgrades its `harbor` pin
between now and PR submission, we re-pin in this ADR and re-validate
locally with the upgraded `harbor check`.


## ADR-0008 — Frontier-agent pilot before PR submission

**Status:** Accepted (2026-06-22)

**Context.** The proposal commits to a "10–20 % solve rate" empirical
target with Claude Opus 4.7, GPT-5, and Gemini 2.5. The reviewer's
"Solvable" concern (and the difficulty calibration) hinges on this number.

**Decision.** Before opening the PR, run each frontier agent against the
final cases. Record solve rate, time-to-first-pass, and per-check failure
modes in `docs/progress/pilot_run.md`. If solve rate is < 5 % or > 30 %,
reopen the case design.

**Consequences.** Adds 1–2 weeks of pilot time. In exchange we ship a
task whose difficulty is empirically calibrated, not guessed — exactly
what the TB-Science maintainers asked for.
