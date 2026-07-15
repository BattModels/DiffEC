# Pre-PR Runbook: smoke test + frontier-agent pilot

> For machine-to-machine sync (laptop ↔ Artemis) and the
> `origin` / `upstream` remote convention, see `docs/dev-setup.md`.


The two remaining Definition-of-Done items (containerized Harbor oracle
smoke test, frontier-agent pilot) can't be done on the Artemis HPC node
alone — Harbor needs to build container images, and Artemis has no
Docker. This runbook splits the work between **your laptop** (Docker)
and **Artemis** (Singularity). Total: ~30 min of laptop time + ~8 hours
of Artemis wall time + ~$200–500 API spend.

## Can I do everything from a Claude Code session on my laptop?

**Yes.** Open Claude Code on your laptop in this repo (or a fresh
clone of the `feat/tb-sci-task` branch), and it can drive every step
below — running shell commands, editing `task.toml`, committing,
pushing. The Artemis portion is then a single `ssh` + `srun` away.

## Laptop prerequisites

| What | Required | Why |
|---|---|---|
| OS | macOS, Linux, or Windows + WSL2 | Anything that runs Docker. |
| `docker` | Docker Desktop or Docker Engine | Build the two task images. |
| Free disk | ~6 GB | base image + JAX/JAXopt layers + push cache. |
| Free RAM during build | ~4 GB | mostly for pip installing JAX. |
| `gh` CLI | recommended | Auto-handles GHCR auth. Otherwise need a PAT (steps below). |
| GitHub account | yes | host the GHCR images for free. |
| Internet | yes | pulls `ubuntu:24.04`, pushes ~1 GB to ghcr.io. |
| Claude Code (optional) | yes if you want my help | runs on macOS / Linux / Windows. |

Docker Desktop is free for personal use and for orgs with < 250 employees and
< $10 M revenue. If you can't install Docker Desktop on your laptop,
GitHub Codespaces is a free fallback (free monthly quota for personal
accounts) — Codespaces ships with Docker pre-installed.

Test the basics in a terminal on your laptop:

```bash
docker --version       # 20+ is fine
docker run --rm hello-world
gh auth status         # or: have a PAT with write:packages ready
```

If any of those fail, fix them before moving on.

## Phase 1 — Laptop: build and push the two images (~30 min, $0)

```bash
# Clone or pull the feature branch on your laptop:
git clone https://github.com/<your-org-or-fork>/DiffEC.git
cd DiffEC
git checkout feat/tb-sci-task

# Authenticate with GHCR (one of these):
gh auth login                          # interactive, easiest
# OR:
export GHCR_TOKEN=ghp_xxxxxxxxxxxxxxx   # PAT with 'write:packages' scope

# Build + push. Substitute YOUR lowercase GitHub handle.
GHCR_USER=<your-github-handle> bash scripts/build_and_push.sh
```

The script will:

1. `docker login ghcr.io` using `gh auth token` or `$GHCR_TOKEN`.
2. `docker build` the agent image from `tasks/.../environment/Dockerfile`
   → tag `ghcr.io/<you>/diffec-env:v1`. ~3-5 min on a modern laptop;
   most time is JAX pip install.
3. `docker build` the verifier image from `tasks/.../tests/Dockerfile`
   → tag `ghcr.io/<you>/diffec-tests:v1`. Similar time.
4. `docker push` both. ~1-2 min depending on uplink.
5. Print exactly which two lines to paste into `task.toml`.

If the build fails on your laptop's architecture (e.g., Apple Silicon),
the script already uses `--platform linux/amd64` which slows the build
slightly but produces images that run on the cluster.

## Phase 2 — Keep the images PRIVATE (anti-cheat) — do NOT make public

**Decision (2026-06-29):** both images stay **private**. The
`diffec-tests` image is built with `COPY . /tests/`, so it contains the
held-out ground truth (`tests/oracle/` solver + `tests/oracle_truth/case_*/truth.npz`).
Making it public on GHCR would publish the benchmark answers to anyone who
knows the URL — directly breaking the project's anti-cheat invariant
("hidden ground truth must never ship alongside the cases"). So we leave
visibility private and have Singularity on Artemis authenticate instead.

There is nothing to do on the laptop for this phase. Authentication is
set up on Artemis in Phase 4 via a `read:packages` PAT and the
`SINGULARITY_DOCKER_USERNAME` / `SINGULARITY_DOCKER_PASSWORD` env vars.

Optional: confirm the images are present (and private) in your GHCR:

```bash
gh api /user/packages/container/diffec-env   --jq '.visibility'   # -> "private"
gh api /user/packages/container/diffec-tests --jq '.visibility'   # -> "private"
```

## Phase 3 — Laptop: edit `task.toml`, commit, push (5 min, $0)

Edit `tasks/physical-sciences/chemistry/concentrated-electrolyte-transport/task.toml`.

Add **one line** at the top of the existing `[environment]` block:

```toml
[environment]
docker_image = "ghcr.io/<your-handle>/diffec-env:v1"   # ← add this
build_timeout_sec = 600.0
cpus = 8
memory_mb = 16384
storage_mb = 10240
gpus = 0
allow_internet = true
```

And add **a new section** at the bottom of the file:

```toml
[verifier.environment]
docker_image = "ghcr.io/<your-handle>/diffec-tests:v1"
```

(The Dockerfile-based fields are still respected by Harbor's default
Docker env mode upstream — `docker_image` is an *additional* hint that
Singularity uses. Both modes coexist.)

Commit and push:

```bash
git add tasks/physical-sciences/chemistry/concentrated-electrolyte-transport/task.toml
git commit -m "Point task.toml at GHCR images for singularity env"
git push origin feat/tb-sci-task
```

## Phase 4 — Laptop: smoke test + pilot (~8 h wall, $200–500)

> **Why the laptop instead of Artemis?** Empirically confirmed
> 2026-06-30: Harbor's `--env singularity` hardcodes
> `singularity exec --fakeroot`, and Artemis users have no subuid
> mappings → the FastAPI server inside the container can never
> start (`could not use fakeroot: no valid mapping entry`).
> See `docs/hpc/artemis.md` §"Harbor's --env singularity is BLOCKED".
> The Singularity path stays scaffolded in this repo
> (`scripts/pilot/run_pilot_singularity.sh`, `docs/hpc/artemis.md`)
> in case subuid gets enabled on the cluster — but for now, run on
> the laptop with `--env docker`, which Just Works.

On your laptop (same machine that ran Phase 1 — Docker is already
installed):

```bash
cd ~/DiffEC                       # or wherever your clone lives
git checkout feat/tb-sci-task
git pull --ff-only

uv tool install harbor            # one-time, user-local
docker --version                   # confirm Docker is running

# API keys (gitignored — never committed).
cat > scripts/pilot/.env <<EOF
ANTHROPIC_API_KEY=sk-ant-…
OPENAI_API_KEY=sk-…
GEMINI_API_KEY=AIza…
EOF
chmod 600 scripts/pilot/.env

# === SMOKE TEST FIRST (~15 min, free) ===
# This uses Docker directly so no GHCR auth is needed — Harbor builds
# both images locally from environment/Dockerfile + tests/Dockerfile.
# (You can leave the docker_image = "ghcr.io/…" lines in task.toml or
# delete them; Docker env prefers the local Dockerfile build either way.)
harbor run \
    --path tasks/physical-sciences/chemistry/concentrated-electrolyte-transport \
    --env docker \
    --agent oracle \
    --yes
# Expect reward = 1. If it doesn't:
#   - check docker is running (docker ps)
#   - check disk space (the build is ~6 GB)
#   - check Dockerfile parses (docker build tasks/.../environment)
# Don't proceed to the full pilot until the smoke test passes.

# === FULL PILOT (3 agents × 10 trials, ~8 h, $200–500) ===
# scripts/pilot/run_pilot.sh defaults to --env docker. Optional: run in
# tmux/screen so it survives terminal disconnects.
bash scripts/pilot/run_pilot.sh

# === AGGREGATE ===
uv run python scripts/pilot/aggregate.py \
    --jobs jobs --out docs/progress/pilot_run.md

# === COMMIT RESULTS ===
git add docs/progress/pilot_run.md
git commit -m "Frontier-agent pilot results"
git push
```

After this, both Definition-of-Done items are met. The PR is unblocked.

## Failure / recovery cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `gh: command not found` (laptop) | gh CLI missing | `brew install gh` (macOS) / `apt install gh` (Linux) / download from cli.github.com |
| `docker: command not found` (laptop) | Docker missing | install Docker Desktop |
| `denied: requested access to the resource is denied` on push | Not authenticated or wrong scope | rerun `gh auth login` ensuring `write:packages` scope, OR generate a PAT with that scope |
| Singularity pull `unauthorized` | GHCR auth not set / PAT lacks read:packages | export `SINGULARITY_DOCKER_USERNAME` + `SINGULARITY_DOCKER_PASSWORD` (read:packages PAT); on Apptainer use the `APPTAINER_` names |
| `harbor` exits with "no `docker_image` field" | task.toml not updated or not pushed | redo Phase 3, push, then `git pull` on Artemis |
| Singularity cache permission errors | `$SCRATCH_DIR` not writable | `mkdir -p /tmp/sing_cache && export SINGULARITY_CACHE=/tmp/sing_cache` |
| Pilot exits with API auth error | API key wrong or env file not loaded | check `scripts/pilot/.env` is present and not empty; re-run |
| Pilot trial timeouts | Agent stuck | increase `AGENT_TIMEOUT_MULT=1.5 bash scripts/pilot/run_pilot_singularity.sh` |
| Cost too high | Cut attempts | `N_ATTEMPTS=5 bash scripts/pilot/run_pilot_singularity.sh` (looser CI, half cost) |

## Where this fits in CLAUDE.md's Definition of Done

This runbook closes DoD items 2 (`harbor run -a oracle` reward = 1) and
6 (frontier-agent pilot 10–20 % solve rate). Item 7 (the actual PR) is
unblocked once these complete.
