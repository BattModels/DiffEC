# Harbor Task Format

> Refreshed **2026-06-22** from:
> - `https://github.com/harbor-framework/terminal-bench-science/blob/main/CONTRIBUTING.md` (main)
> - `https://harborframework.com/docs/task-format`
> - Example task: `tasks/physical-sciences/chemistry/geometric-pharmacophore-alignment/`
>
> Re-checked **2026-06-27** against:
> - `harbor` CLI 0.16.0 (installed via `uv tool install harbor`).
> - Upstream `geometric-pharmacophore-alignment/task.toml` at HEAD (unchanged
>   from 2026-06-22 reference; still uses `schema_version = "1.0"` and
>   `allow_internet = true`).
>
> Our `task.toml` parses successfully under Harbor 0.16.0's `TaskConfig`
> pydantic schema (one `DeprecationWarning` for `allow_internet`, which is
> auto-migrated to `network_mode = "public"` — identical to upstream's
> example, so no action needed for now). ADR-0010 (pin `schema_version =
> "1.0"`) holds.
>
> Refresh again before opening the PR. If the upstream `CONTRIBUTING.md` HEAD
> moves, update this file *and* the related ADRs.

## Tree (this is what we ship)

Tasks live at `tasks/<domain>/<field>/<task-name>/` in the upstream repo,
kebab-case throughout. For our submission:

```
tasks/physical-sciences/chemistry/<task-name>/
├── instruction.md                  # what the agent sees (our formalism.md, restated for the harness)
├── task.toml                       # manifest (schema_version = "1.0")
│
├── environment/                    # AGENT container build context
│   ├── Dockerfile                  # installs agent-side deps (Python + JAX + ...)
│   └── data/                       # files copied into the agent's /root/data/ at build time
│       ├── cases/case_1/{data.h5, params.json, formalism.md}
│       ├── cases/case_2/…
│       ├── cases/case_3/…
│       └── cases/case_4/…
│
├── solution/                       # reference solution (run by the Oracle agent only)
│   ├── solve.sh                    # entrypoint; mounted at /solution/ inside agent container
│   └── <our reference solver scripts>
│
└── tests/                          # VERIFIER container build context (separate from agent)
    ├── Dockerfile                  # installs verifier-side deps + COPY . /tests/
    ├── test.sh                     # runs pytest, writes /logs/verifier/reward.txt
    ├── test_outputs.py             # the actual pytest checks
    └── oracle_truth/case_X/truth.npz   # held-out ground truth, baked into the verifier image
```

**Authoritative facts about this layout (from upstream CONTRIBUTING.md):**
- Naming: kebab-case for `<task-name>`. Domains: `life-sciences`,
  `physical-sciences`, `earth-sciences`, `mathematical-sciences`,
  `other-sciences`.
- `environment/Dockerfile` "do[es] not copy solution or test files into
  the container."
- `tests/Dockerfile` "must pre-install all test dependencies (no runtime
  installs)" and `COPY . /tests/`.
- "The verifier runs in its own clean container, completely isolated
  from the agent."

## `task.toml` (manifest)

The upstream example (`geometric-pharmacophore-alignment/task.toml`) uses
`schema_version = "1.0"`. Match that — *not* the `"1.3"` shown on
`harborframework.com/docs/task-format`. The TB-Science repo is on the
older schema; align with what the upstream CI validates.

Concrete starting template (fill in once case design is locked):

```toml
schema_version = "1.0"

artifacts = [
    "/root/results/case_1/transport.json",
    "/root/results/case_2/transport.json",
    "/root/results/case_3/transport.json",
    "/root/results/case_4/transport.json",
]

[metadata]
author_name = "Changwen Xu"
author_email = "changwex@umich.edu"
author_organization = "University of Michigan"
difficulty_explanation = "..."        # see proposal §Complexity
solution_explanation = "..."          # see oracle-spec.md + reference solution
verification_explanation = "..."      # see verifier-spec.md (5 checks + self-consistency)
domain = "physical-sciences"
field = "chemistry"
subfield = "electrochemistry"
tags = ["electrochemistry", "concentrated-electrolyte", "inverse-problem", "differentiable-simulation", "jax"]
expert_time_estimate_hours = 4.0      # proposal's "3-5 hours of focused work"

[verifier]
timeout_sec = 600.0
environment_mode = "separate"

[agent]
timeout_sec = 3600.0                  # 1 hour total = proposal's budget for 4 cases

[environment]
build_timeout_sec = 600.0
cpus = 8                              # proposal: 4-8 cores
memory_mb = 16384                     # proposal: 8-16 GB
storage_mb = 10240
gpus = 0
allow_internet = true                 # proposal allows JAX install etc.
```

Fields to confirm against the latest upstream example before submitting:
- Whether `allow_internet` is the right key (a newer Harbor version uses
  `network_mode = "public"|"allowlist"|"no-network"` and `allowed_hosts`).
  Trust the upstream example, not `harborframework.com`.
- Whether `relevant_experience` is required (CONTRIBUTING.md mentions it
  but `geometric-pharmacophore-alignment` omits it). Include defensively.

## Agent working-directory contract

(From the example's Dockerfiles + Harbor docs.)

- Agent's working directory is `/root/` (the `geometric-pharmacophore-alignment`
  example uses `/root/data/` for inputs and `/root/results/` for outputs;
  these come from `mkdir -p /root/data /root/results` in the agent
  `Dockerfile`).
- Input bundle: `/root/data/cases/case_X/{data.h5, params.json, formalism.md}`.
  Created at agent-image build time by copying `environment/data/cases/`.
- Output bundle (what the agent must write): `/root/results/case_X/transport.json`.
- Reference solution: mounted at `/solution/` (only when running as the
  Oracle agent). `solve.sh` is invoked from there.
- Tests: mounted at `/tests/` (the verifier container is separate;
  artifacts are transferred from agent → verifier per the `artifacts`
  list).

The `instruction.md` we ship must say absolute paths only — per the
CONTRIBUTING guide: *"Always use absolute paths (e.g., `/root/output.txt`)
not relative paths."* And it must end with: *"You have X seconds to
complete this task. Do not cheat by using online solutions or hints
specific to this task."*

## Verifier contract

`tests/test.sh` is the verifier entrypoint. The upstream example
pattern is:

```bash
mkdir -p /logs/verifier
pytest /tests/test_outputs.py --ctrf /logs/verifier/pytest-ctrf.json \
    > /logs/verifier/pytest.log 2>&1
status=$?
if [ "$status" -eq 0 ]; then echo 1 > /logs/verifier/reward.txt
else echo 0 > /logs/verifier/reward.txt
fi
exit $status
```

Our `tests/test_outputs.py` is the pytest verifier sketched in
`docs/plan/verifier-spec.md`. The `reward.txt` is a **single integer
(0 or 1)** in the TB-Science convention — partial credit via
`reward.json` is allowed by Harbor but the upstream example uses the
all-or-nothing form, and the proposal commits to "all five checks must
succeed for all four cases."

Note for our setup: the verifier container has its own `Dockerfile`;
that's where we install JAX + pytest + the oracle solver code so the
self-consistency check (#6) can re-run our solver on the agent's
reported parameters. The held-out `oracle_truth/case_X/truth.npz` lives
inside `tests/` and is baked into the verifier image via `COPY . /tests/`.

## Resource-budget contract

`[environment]` fields are declarations to the provider. Harbor itself
doesn't dictate enforcement; providers apply per their policy.

Three separate timeout budgets (do not conflate):
- `[environment] build_timeout_sec` — container build (default 600).
- `[agent] timeout_sec` — wall-time the agent has to produce outputs.
- `[verifier] timeout_sec` — wall-time the verifier has to grade.

For us: `[agent] timeout_sec = 3600` (1 h, matches the proposal's
"< 1 hour total"); `[verifier] timeout_sec = 600` (the verifier's most
expensive check, #6 self-consistency, re-runs the forward solver four
times — that's ~5–7 min total at our oracle step size).

## Reference-solution policy

In-tree, under `solution/`. The Oracle agent runs `bash /solution/solve.sh`
from `/root/`. Non-script files (Python modules, configs) are allowed
under `solution/`. Solutions are technically optional but
*required for Oracle sanity checks* — which is the gate the upstream CI
runs (`harbor run -p <task> -a oracle` must achieve 100 % reward).

For us this means: `solution/solve.sh` reads `/root/data/cases/case_X/`,
writes `/root/results/case_X/transport.json` for each of the 4 cases,
exits 0. Implementation lives in Python alongside `solve.sh`.

## Large-file policy

> "Files >100MB should not be committed — host the data on Hugging Face
> and use download scripts in your task, or contact project maintainers."

Our bundled cases are ~50–100 MB total (per the proposal). Per-file we
should stay well under 100 MB (each `data.h5` is ~10–25 MB), so direct
commit is fine. If any single artifact balloons over 100 MB,
HuggingFace-host it and download in the agent `Dockerfile`.

This is more permissive than what ADR-0007 anticipated; we can keep
the "commit `.h5` directly" decision.

## Local validation (pre-PR)

Per CONTRIBUTING.md:

```bash
# Static rubric check (uses harbor CLI)
harbor check -r rubrics/task-implementation.toml \
    -m anthropic/claude-opus-4-8 \
    tasks/physical-sciences/chemistry/<task-name>

# Oracle must achieve 100% reward
harbor run -p tasks/physical-sciences/chemistry/<task-name> -a oracle

# Interactive debugging
harbor tasks start-env -p tasks/physical-sciences/chemistry/<task-name> -e docker -a -i

# Test with a real agent
harbor run -p tasks/physical-sciences/chemistry/<task-name> \
    -a <agent> -m <provider/model>

# Failure analysis for the PR body
harbor analyze tasks/physical-sciences/chemistry/<task-name>
```

Add to `docs/plan/build-and-run.md`: install `harbor` CLI per its own
docs; we don't pin it in our `uv.lock` since it's an upstream dev tool.

## PR contract

- **Title format:** `[TASK: <Scientific field>] Task title`. For us:
  `[TASK: Chemistry] Differentiable modeling of concentrated-electrolyte mass transport from operando profiles`.
- **Body must include** a link to the approved proposal discussion
  (`#335`) — the static check `check-task-proposal-link` blocks merge
  if the linked proposal does not carry the `proposal-approved` label.
- **Body should include** failure-mode analysis from `harbor analyze`
  (this is what our `docs/progress/pilot_run.md` aggregates).
- **Body should include** screenshots / evidence of an end-to-end
  Oracle pass.

## Upstream CI on the PR

Three layers, all blocking:
1. Static checks (structure, metadata, formatting, proposal-link check).
2. Implementation-rubric review via `harbor check` (31 evaluation criteria).
3. Execution checks (similarity, Docker build, oracle/nop validation).

Plus three-pass human review (1st reviewer DRI matched to field, 2nd
independent, 3rd bar-raiser).

## Deadline & cadence

PRs due **August 17, 2026** (matches `docs/proposal/decision.md`).
Review and iteration are post-deadline but no new PRs after.
"Most tasks require a few rounds of feedback" — start early.

## Action items now that the format is pinned

- [ ] `docs/plan/architecture.md` working tree → switch the working
      paths from our placeholder `tb_sci_task/` to
      `tasks/physical-sciences/chemistry/<task-name>/` and update the
      module layout: `oracle/` becomes a Python package inside
      `tests/` (so its code is baked into the verifier image), and
      `reference_solution/` becomes `solution/`.
- [ ] `docs/plan/decisions.md` ADR-0001 → close out: format confirmed,
      tree alignment in progress; ADR-0007 → confirm bundled-`.h5`
      decision survives (it does, < 100 MB per file).
- [ ] `docs/plan/build-and-run.md` → add `harbor` CLI install +
      `harbor run -a oracle` to the pre-PR procedure.
- [ ] `docs/plan/verifier-spec.md` → confirm the `reward.txt`
      single-integer convention (all-or-nothing per the upstream
      example) and the `test.sh` wrapper around pytest.
- [ ] Decide a kebab-case `<task-name>`. Candidates: `concentrated-electrolyte-mass-transport`,
      `diffec-mass-transport`, `newman-inversion-from-operando`.
- [ ] Sketch `task.toml` once the case design is locked.
