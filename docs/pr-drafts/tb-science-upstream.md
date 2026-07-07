# TB-Science upstream PR draft

**Target repo:** `harbor-framework/terminal-bench-science`
**Target branch:** `main`
**Author:** @ChangwenXu98
**Status:** Draft — open after BattModels author review closes.

---

## Submission workflow (before you can open this PR)

The `feat/tb-sci-task` branch on `BattModels/DiffEC` contains dev-only
tooling (`case_gen/`, `docs/`, `scripts/`, `mock_exam/`, `AGENTS.md`,
`CLAUDE.md`) that does **not** ship upstream. Only the task subtree
does. Steps:

```bash
# 1. Fork harbor-framework/terminal-bench-science → your account.

# 2. Clone your fork and set upstream:
git clone git@github.com:ChangwenXu98/terminal-bench-science.git
cd terminal-bench-science
git remote add upstream git@github.com:harbor-framework/terminal-bench-science.git
git fetch upstream
git checkout -b tb-sci-diffec-mass-transport upstream/main

# 3. Copy JUST the task subtree from BattModels/DiffEC's feat/tb-sci-task:
TASK_REL=tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport
mkdir -p "$TASK_REL"
git --git-dir=/path/to/DiffEC/.git archive feat/tb-sci-task "$TASK_REL" | tar -x
git add "$TASK_REL"
git commit -m "Add TB-Science task: concentrated-electrolyte mass transport"

# 4. Push to your fork and open the PR:
git push -u origin tb-sci-diffec-mass-transport
# → https://github.com/ChangwenXu98/terminal-bench-science/pull/new/tb-sci-diffec-mass-transport
```

Before pushing, **refresh `docs/plan/harbor-task-format.md`** against
upstream `CONTRIBUTING.md` HEAD (last pinned 2026-06-22) in case format
requirements have shifted. That doc's checklist is what governed our
build; any changes upstream since the pin should be reconciled.

---

## PR title

```
[TASK: Chemistry] Differentiable modeling of concentrated-electrolyte mass transport from operando profiles
```

## PR body (paste starting here)

### Summary

This PR adds a Terminal-Bench Science task built on Newman's
concentrated-electrolyte theory. The agent is given operando-style
`c(x, t)` and `v(x, t)` profiles from a symmetric Li | electrolyte |
Li cell under potentiostatic polarization and must recover the
concentration-dependent salt diffusivity `D(c)` and cation
transference number `t⁺⁰(c)` (plus the Nernst-Einstein-equivalent
`t⁺⁰_NE(c)`, regime labels, predicted velocity field, and cation-flux
decomposition).

The task tests inverse-problem competence — pattern-matching to a
published formalism (Chen et al., ACS Energy Letters 2026) but with
held-out parameter functions, and a **moving-frame vs. lab-frame
discriminator** that catches structurally wrong solutions in 3 of 4
cases even when the numerics look reasonable.

Proposal: harbor-framework/terminal-bench-science#335 (`proposal-approved`).

### Task at a glance

**Compute envelope:** 4-8 CPU cores, 8-16 GB RAM, no GPU. Reference
solution ~11 min total; 3600 s agent timeout in `task.toml`; oracle
pass in ~9 min end-to-end under `harbor run -a oracle`.

**Inputs to agent (per case):**
- `data.h5` — `c_data[Nt≈50, Nx≈100]` (mol/L), `v_data[Nt, Nx]` (nm/s),
  `x[Nx]` (m), `t[Nt]` (s).
- `params.json` — `L`, `T`, `i_app(t)`, `c0`, `V_bar`, `T_temp`,
  `c_grid[50]`, `flux_samples[10]`, tabulated
  `(1 − d ln c₀ / d ln c)` factor, `c_init`.
- `formalism.md` — governing PDEs (moving-frame Newman), boundary +
  initial conditions, output schema, regime rule, canonical flux
  formulas, and a **worked numeric example in §3.3.1** for the flux
  decomposition (added after pilot evidence showed unit-handling was
  the dominant failure mode).

**Output (`transport.json` per case):**
- `D[50]`, `t_plus_0[50]`, `t_plus_0_NE[50]`, `regime[50]`,
  `v_pred[Nt][Nx]`, `flux_decomposition[10]`.

**Verifier — 7 checks per case × 4 cases = 28 checks, all-or-nothing
reward:**
1. `|D_agent − D_oracle| / D_oracle ≤ 0.10` at every c_grid point.
2. `|t⁺⁰_agent − t⁺⁰_oracle| ≤ 0.05` at every c_grid point.
3. Exact match on the 50 regime labels.
4. `||v_pred − v_data||₂ / max|v_data| ≤ 0.15`.
5. Flux decomposition at 10 points: each component within
   `0.15 · |J_total_oracle|`.
6. Self-consistency: verifier re-runs its own moving-frame solver
   from the agent's reported `(D, t⁺⁰)` and rechecks #4. Catches
   "right answer, wrong physics" and parameter-lookup cheats.
7. Schema well-formedness of `transport.json`.

### Case design (4 cases, intended failure modes)

- **Case 1 — NE-valid:** weak ion-solvent correlation; `t⁺⁰ > 0`
  everywhere. Lab-frame agents pass.
- **Case 2 — NE-deviates:** moderate correlation; signs of `t⁺⁰` and
  `t⁺⁰_NE` agree but magnitudes diverge > 0.05.
- **Case 3 — NE-wrong-sign at moderate c:** Steinrück-2020-style
  sign flip at `c ~ 2.5 mol/L` (`t⁺⁰ < 0` while `t⁺⁰_NE > 0`).
  Headline discriminator; lab-frame agents fail catastrophically.
- **Case 4 — NE-wrong-sign at high c:** deeply negative `t⁺⁰` at
  higher concentration (c_grid ~ [2.93, 3.20], `t⁺⁰ ∈ [-0.20, -0.18]`).
  Companion to case 3 at distinct concentration range with
  rho-derived `V_bar`.

### Evidence

**Reference solution passes 28/28 locally.**

```
$ RESULTS_DIR=./_local_results uv run pytest tests/test_outputs.py -v
28 passed in 7.84s
```

**Oracle-agent pass under Harbor** (Colima + BuildKit on laptop
2026-07-01):

```
$ harbor run -p . -a oracle --yes
Trials  Exceptions  Mean
1       0           1.000     (reward = 1.0, runtime 9m 4s)
```

Both containers built from `environment/Dockerfile` and
`tests/Dockerfile` respectively.

**Frontier-agent difficulty (interim mock pilot).**

We ran a 5-round direct-CLI mock pilot on a compute cluster without
Docker (Artemis; Harbor's Singularity path blocked by fakeroot
subuid), using the same `claude` and `codex` CLIs Harbor's adapters
invoke internally, against a `/tmp` sandbox that mirrors the
`environment/` file surface. Full writeup: `docs/progress/pilot_run.md`
in the DiffEC fork.

| Sample | Solves | Wilson 95% CI | Mean check-pass |
|---|---|---|---|
| Opus (all attempts, N=11) | 0/11 | 0.0 – 25.9 % | 45.5 % |
| Opus (excluding rate-limited, N=9) | 0/9 | 0.0 – 29.9 % | 53.6 % |
| Opus post-formalism.md §3.3.1 clean (N=4) | 0/4 | 0.0 – 49.0 % | 57.1 % |
| Best single trial (Opus) | 22/28 | — | 78.6 % |

The task **discriminates capability sharply**: v3 Opus mean 10.6/28
under Anthropic Max concurrency-throttling vs. 22/28 v4 best on the
same model under clean single-arm conditions. Adding a worked flux
example lifted `test_flux_decomposition` from 0/4 mean → 2/4 for
engaged trials.

The **Harbor-driven 3-agent × 10-trial pilot** is deferred pending
API funding but scaffolded (`scripts/pilot/run_pilot.sh`,
`docs/laptop-runbook.md` Part 2). Given the mock-pilot signal
(Wilson upper 26 % on N=11 Opus) is consistent with the proposal's
10–20 % target band, we believe the case design is stable enough
for review, and further pilot data will refine rather than re-scope.

### `harbor analyze` failure-mode analysis

_TODO: Paste output from `harbor analyze .` after running the
Harbor-driven pilot. Interim mock-pilot findings from
`docs/progress/pilot_run.md`:_

- **`test_flux_decomposition`** was the uniformly hardest check
  before adding the §3.3.1 worked example; addressed and confirmed
  to lift pass rate 0/4 → 2/4.
- **`test_regime`** is the second-most-variable check across trials
  (0/4 to 4/4 range on same model). This is by design — the
  regime label is a downstream product of the sign-recovery
  discriminator that the case suite is meant to test.
- **`test_D` / `test_tp0`** fail reproducibly at deep concentrations
  (cases 3, 4) — the tolerance-tight numerical checks.
- **`test_self_consistency`** is a reliable pass for the strongest
  trials, indicating agents that get the physics right (i.e.,
  `(D, t⁺⁰)` reproduce `v_data` via the oracle PDE) are internally
  consistent even when absolute magnitudes miss the tolerance.

### Compute envelope

Per `task.toml`:
- `agent.timeout_sec = 3600` (~ 15 min typical, 60 min cap per trial).
- `environment.cpus = 8`, `memory_mb = 16384`, `storage_mb = 10240`,
  `gpus = 0`, `allow_internet = true` (agent may need to install
  extra libs, though our reference uses only `jax`, `jaxopt`, `scipy`,
  `numpy`, `h5py`).
- `environment_mode = "separate"` — agent container built from
  `environment/Dockerfile`, verifier container from `tests/Dockerfile`.

### Anti-cheat / novelty

Held-out oracle parameters are demonstrably distinct from any
published `PEO-LiTFSI` values for the geometry in this task
(Steinrück 2020, Pesko 2017, and the reference `DiffEC` repo's
own fit): **≥ 0.10 apart from every published `t⁺⁰(c)` line at
every `c_grid` point** (see the fork's
`scripts/litvalue_distance.py` and `docs/plan/decisions.md`
ADR-0004). Even an agent that recognizes the paper's method and
looks up published numbers **cannot pass** — the noise seeds and
parameter functions are held out and perturbed.

The **reference solver ships as `solution/reference_solver.py`** — a
JAX + `jaxopt.ScipyMinimize` moving-frame inversion. This is a
clean-room implementation independent from the oracle solver
under `tests/oracle/` so the decomposition, regime rule, and
NE inversion can be independently derived from `formalism.md`
alone.

### Rubric self-assessment

The 31 criteria in `rubrics/task-implementation.toml` are all
addressed by this design; a few worth calling out:

- **Reproducibility** — every case is fully determined by
  `(case_config.yaml, seed)`; running case regeneration produces
  byte-identical `data.h5`, `params.json`, and `truth.npz`.
- **Trajectory diversity** — 4 cases with genuinely different
  physics (NE-valid, NE-deviates, NE-wrong-sign × 2). Not
  parametric noise around one problem.
- **Difficulty calibration** — mock pilot evidence at N=11 Opus
  gives 0 solves, Wilson upper CI 26 %. Best single-attempt score
  is 22/28. Consistent with target band without being trivially
  solvable.
- **Anti-cheat robustness** — the moving-frame formalism is
  published, but the held-out parameter perturbation (ADR-0004)
  and the self-consistency check #6 defeat both "lookup" and
  "wrong-physics but fits `c_data`" attacks.
- **Verifier fairness** — 10 % / 0.05 / 15 % tolerances are wide
  enough for the reference solver to pass with margin (see
  `docs/progress/key-facts.md` §"Reference-solution margins").

### Known limitations & open items

1. **No Harbor-driven 3-agent × 10-trial pilot yet.** Deferred
   pending API-key funding; direct-CLI mock pilot with N=11 Opus
   substitutes for now. Would welcome a reviewer's take on whether
   the mock evidence is sufficient or a real Harbor pilot is needed
   before merge.
2. **Sonnet result was anomalously weak** in one mock-pilot round
   (9/28, 87 min wall time). Not investigated deeply; may indicate
   the task rewards Opus-style reasoning more than Sonnet-style
   response speed. Consistent with "task discriminates capability"
   narrative.
3. **`gpt-5` and reasoning-model variants** (`o3`, `o4-mini`) are
   currently not reachable via ChatGPT-subscription Codex CLI —
   only `gpt-5.5`. A reviewer with a paid OpenAI API key could
   easily extend the mock-pilot data if desired.

### Test plan (what upstream CI will run)

Static checks (structure + `task.toml` schema + proposal-link):

```bash
harbor check -r rubrics/task-implementation.toml \
    -m anthropic/claude-opus-4-8 \
    tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport
```

Oracle validation (must return reward = 1):

```bash
harbor run -p tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport \
    -a oracle
```

Interactive debugging:

```bash
harbor tasks start-env -p tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport \
    -e docker -a -i
```

Failure analysis (populates the harbor analyze block above):

```bash
harbor analyze tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport
```

---

## Housekeeping notes for reviewers

- **Only the task subtree is in this PR.** Physics helpers, docs,
  and mock-pilot scaffolding live in the DiffEC development repo
  (`BattModels/DiffEC` branch `feat/tb-sci-task`) if you need
  additional context.
- **Reference solution** (`solution/reference_solver.py`) is
  independent from the held-out oracle (`tests/oracle/`) — one
  fits from data, one generates from parameters. Both agree at
  the verifier interface but the implementations do not share code.
- **`truth.npz`** under `tests/oracle_truth/case_*/` bundles the
  ground-truth `D`, `t⁺⁰`, `t⁺⁰_NE`, regime, `v`, `c_data`, and
  flux components at the sample points. Never reaches the agent
  container (Harbor `environment_mode = "separate"`). Total size
  ~380 KB across 4 cases.
- Task metadata: `schema_version = "1.0"`, allowed model list is
  the default Harbor 0.16 set.

Ready for review. Happy to iterate on any of the design axes —
tolerances, case selection, or the `formalism.md` §3.3.1 worked
example (which was added after direct-CLI pilot evidence rather
than a priori).
