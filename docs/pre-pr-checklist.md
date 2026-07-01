# Pre-PR Checklist

Run this before opening the upstream PR to
`harbor-framework/terminal-bench-science`. Every check must pass;
the only manual action is reverting the two PILOT-ONLY
`docker_image` lines that were added in commit `e12cff9` for the
laptop pilot.

## When to use

- After the frontier-agent pilot on the laptop has finished
  (`docs/laptop-runbook.md` Part 2 complete).
- After `docs/progress/pilot_run.md` has been committed with the
  aggregate solve rate.
- Immediately before opening the PR — a green audit here is the
  go-signal for PR submission.

## Step 1 — Automated audit (pre-revert)

```bash
bash scripts/pre_pr_audit.sh
```

Runs 14 checks. Expected result at this point: **FAILED with 1
check blocked** — check #8 (`PILOT-ONLY docker_image lines`) will
flag the two lines in `task.toml` from commit `e12cff9`. All 13
other checks should pass. If any of the 13 fail, stop and fix
those first before proceeding.

What each check covers (in case you need to triage a failure):

| # | Check | Failure means |
|---|---|---|
| 1 | Subtree inventory | file count outside 20–60; unexpected additions/deletions |
| 2 | 100 MB size limit | some file is too large for upstream |
| 3 | Agent view is clean | `oracle/` or `oracle_truth/` leaked into `environment/` |
| 4 | Verifier has oracle + truth | one of the required files is missing |
| 5 | Solution has entrypoint + helpers | `solve.sh` or a `.py` under `solution/` is missing |
| 6 | No stray untracked files | ungitignored files that would ship if `git add`-ed |
| 7 | `formalism.md` doesn't leak numbers | oracle truth values in the agent-facing spec |
| 8 | PILOT-ONLY `docker_image` reverted | (expected FAIL until Step 2 below) |
| 9 | `task.toml` validates | Harbor 0.16 schema rejects it |
| 10 | Cross-refs (solve.sh ↔ artifacts, test.sh reward) | pathnames don't match between files |
| 11 | `reference_solver.py` CLI matches `solve.sh` | `--cases` / `--out` mismatch |
| 12 | No `:latest` in Dockerfiles | image is unpinned; reproducibility risk |
| 13 | No hardcoded HPC/laptop paths | shipped file references our filesystem |
| 14 | No TODO/FIXME/XXX in shipped files | unresolved code marker |

## Step 2 — Manual revert of PILOT-ONLY lines

Open `tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/task.toml`.

**Delete these six lines from the `[environment]` block** (currently around lines 37-40):

```toml
# PILOT-ONLY: points Singularity at a prebuilt GHCR image so the Artemis
# pilot doesn't need Docker. Personal namespace — REVERT before the upstream
# PR (upstream builds from environment/Dockerfile). See docs/pre-pr-runbook.md.
docker_image = "ghcr.io/changwenxu98/diffec-env:v1"
```

**Delete this whole section** (currently around lines 48-50):

```toml
[verifier.environment]
# PILOT-ONLY (private image; Artemis authenticates to GHCR). REVERT before PR.
docker_image = "ghcr.io/changwenxu98/diffec-tests:v1"
```

Result: `[environment]` reverts to `build_timeout_sec` /
`cpus` / `memory_mb` / `storage_mb` / `gpus` / `allow_internet`
only. The `[verifier.environment]` section disappears entirely.
The final `task.toml` matches upstream's
`geometric-pharmacophore-alignment` example schema-for-schema.

Verify with grep:

```bash
grep -c '^docker_image'         tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/task.toml   # expect 0
grep -c '^\[verifier\.environment\]' tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/task.toml  # expect 0
```

Both should print `0`.

## Step 3 — Automated audit (post-revert)

```bash
bash scripts/pre_pr_audit.sh
```

Expected result now: **AUDIT PASSED — subtree is ready for the
upstream PR.** All 14 checks green.

## Step 4 — Commit the revert

```bash
git add tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/task.toml
git commit -m "Revert PILOT-ONLY docker_image lines for upstream PR

The two docker_image lines and the [verifier.environment] section
added in e12cff9 for the Artemis Singularity pilot pointed at the
personal ghcr.io/changwenxu98 namespace. Remove them so the upstream
task builds from environment/Dockerfile + tests/Dockerfile per the
Harbor default. The images remain on GHCR (private) as a fallback
if we ever run the Singularity pilot again."
git push
```

## Step 5 — Sanity: verifier still passes 28/28 locally

```bash
TASK=tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport
RESULTS_DIR=./_local_results TRUTH_DIR="$TASK/tests/oracle_truth" \
    uv run pytest "$TASK/tests/test_outputs.py"
# Expect: 28 passed
```

If this fails, something else broke — investigate before opening
the PR.

## Step 6 — PR submission (see CLAUDE.md DoD item #7)

PR target: `harbor-framework/terminal-bench-science` (a separate
repo, not `BattModels/DiffEC`). This is a one-time copy of the
`tasks/.../concentrated-electrolyte-mass-transport/` subtree per
ADR-0009 — not a git-merge relationship.

PR conventions (from CLAUDE.md and upstream `CONTRIBUTING.md`):

- **Title:** `[TASK: Chemistry] Concentrated-electrolyte mass transport (Newman inverse problem)`
- **Body:** should reference **discussion #335** and include:
  - Overview + failure-mode taxonomy
  - Reference-solver margins table (from `docs/progress/key-facts.md`)
  - Frontier-agent pilot solve rate (from `docs/progress/pilot_run.md`)
  - `harbor analyze` output on the shipped subtree (upstream review artifact)
  - Screenshot / block-quote of oracle-agent `reward = 1` from the smoke test

A ready-to-paste PR body can be drafted at `docs/pr-description.md`
(not yet written; separate task).

## When to re-run this checklist

- Any time `task.toml`, `tests/`, `solution/`, `environment/`, or
  `instruction.md` is touched.
- Before any PR to upstream — not just the initial one.
- After merging upstream changes (if the review requests
  modifications).

## Related docs

- `docs/pre-pr-runbook.md` — how to get to the point of running
  this checklist (build images, run pilot, etc.).
- `docs/laptop-runbook.md` — laptop-side smoke test + pilot
  procedure a Claude Code session can drive.
- `docs/plan/decisions.md` — ADRs that constrain what does/doesn't
  ship (esp. 0001, 0004, 0007, 0009, 0010).
- `docs/plan/harbor-task-format.md` — Harbor task format pin;
  refresh before PR if upstream `CONTRIBUTING.md` HEAD has moved.
- `CLAUDE.md` §"Definition of Done" — the full 7-item merge
  contract.
