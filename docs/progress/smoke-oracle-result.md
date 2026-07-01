# Oracle smoke test — DoD #2 pass

## Result

Ran on laptop (Changwen's MacBook Pro, Colima + BuildKit) 2026-07-01:

```
Trials  Exceptions  Mean
1       0           1.000     (reward = 1.0, runtime 9m 4s)
```

`Mean = 1.000` → `reward = 1` → the Oracle-agent trajectory
(agent = our `solution/solve.sh` running `reference_solver.py`
inside a Docker container built from `tasks/.../environment/Dockerfile`,
then verified by the container built from `tasks/.../tests/Dockerfile`)
returned all-green on all 28 checks across the 4 cases.

This satisfies **CLAUDE.md Definition of Done item #2**:

> `harbor run -p "$TASK" -a oracle` returns reward = 1.

Runtime 9m 4s end-to-end (image builds + agent execution + verifier
run) is comfortably under the 1-hour `agent.timeout_sec` budget in
`task.toml`.

## Environment (host)

- **Machine:** Apple Silicon MacBook Pro
- **Container runtime:** Colima (Docker daemon in a Linux VM) + BuildKit
- **Docker CLI:** homebrew `docker` (not Docker Desktop) — see
  "Runbook gaps" below
- **Harbor CLI:** 0.16.x (`uv tool install harbor`)

Prior state on the laptop from the earlier build+push work (commit
`e12cff9`): the two images
`ghcr.io/changwenxu98/diffec-{env,tests}:v1` are still up on GHCR
(private). They were NOT used for the smoke test — Harbor's Docker
env preferred to build from `Dockerfile` locally because
`docker_image` was left committed in `task.toml`. See the runbook
deviation notes below.

## Runbook gaps discovered (three, all fixed in the runbook for the pilot)

The laptop-runbook v1 (commit `2c15ecb`) assumed a Docker Desktop
default install. Homebrew's `docker` CLI ships no plugins, so a
BuildKit-based `harbor` invocation needed extras before the first
`docker build` would run:

1. **Missing `docker compose` plugin** — Harbor's Docker env
   invokes `docker compose up …` behind the scenes. Symptom:
   `unknown flag: --project-name`. Fix: `brew install
   docker-compose` (or install the docker-compose-plugin), then
   symlink into `~/.docker/cli-plugins/`.
2. **Missing `docker buildx` plugin** — the legacy builder can't
   resolve multi-stage `FROM ubuntu:24.04`. Symptom: `Error: no
   build stage in current context`. Fix: `brew install
   docker-buildx`, same symlink treatment. Both images then built
   cleanly on Colima's BuildKit.
3. **Colima VM under-provisioned** — the default Colima VM ships
   with 4 CPU / 6 GB; `task.toml` requests 8 CPU / 16 GB. Symptom:
   Docker refused container create with the requested resource
   limit. Workaround for the smoke test: temporarily lowered
   `task.toml` to `cpus=4 / memory_mb=8192`, ran, then restored
   via `git checkout -- task.toml`. Durable fix for the pilot:
   `colima stop && colima start --cpu 8 --memory 16` before
   running Part 2.

## Runbook deviations (both faithful to intent, fully reversed)

- **Step 3 stash was a no-op** — `git stash push task.toml` was
  supposed to remove the two `docker_image` lines, but they're
  committed (from `e12cff9`), so stashing didn't touch them. Ran
  an in-place edit to delete the lines, then restored with
  `git checkout -- tasks/.../task.toml` after the run.
- The temporary Colima-fit CPU/memory reduction described above.

Working tree ended clean after both restorations — `task.toml`
back to `cpus=8 / memory_mb=16384` with both `docker_image` lines
intact (they're still needed for the fallback singularity path if
subuid ever gets enabled on Artemis).

## What lives on the laptop only (not committed)

- `_local_jobs/smoke-oracle*/` — full Harbor job dir with
  `result.json`, per-trial logs, artifacts. Gitignored per the
  runbook. Kept on the laptop for post-hoc inspection.
- `docs/session.md` (gitignored, laptop-local) — running experience
  notebook, has the full timeline of the three env-gap fixes.

## Reference

Full procedure lived in `docs/laptop-runbook.md` (Part 1). The three
runbook gaps above have been folded into an updated version so the
same friction doesn't repeat when someone runs Part 2 (the full
frontier-agent pilot).
