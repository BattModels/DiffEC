# Pre-PR Runbook: smoke test + frontier-agent pilot

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

## Phase 2 — Laptop: make the images public (1 min, $0)

GHCR packages default to private. Singularity on Artemis won't
authenticate, so the images need to be public.

```bash
# With gh CLI (run on the laptop):
gh api --method PATCH /user/packages/container/diffec-env/visibility   -f visibility=public
gh api --method PATCH /user/packages/container/diffec-tests/visibility -f visibility=public
```

Or in a browser:

1. Open <https://github.com/your-handle?tab=packages>
2. Click `diffec-env` → **Package settings** → **Danger Zone** → **Change visibility** → **Public** → type name → confirm
3. Repeat for `diffec-tests`

Verify public anonymously from the laptop terminal:

```bash
docker logout ghcr.io
docker pull ghcr.io/<you>/diffec-env:v1    # should succeed without auth
docker pull ghcr.io/<you>/diffec-tests:v1
```

## Phase 3 — Laptop: edit `task.toml`, commit, push (5 min, $0)

Edit `tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/task.toml`.

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
git add tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/task.toml
git commit -m "Point task.toml at GHCR images for singularity env"
git push origin feat/tb-sci-task
```

## Phase 4 — Artemis: smoke test + pilot (~8 h wall, $200–500)

SSH to Artemis from your laptop (`ssh changwex@artemis-login.engin.umich.edu`
or whatever your endpoint is). Then on Artemis:

```bash
cd /nfs/turbo/coe-venkvis/$USER/projects/DiffEC
git pull

module load singularity/4.4.1
uv tool install harbor              # one-time, user-local

# API keys (gitignored — never committed).
cat > scripts/pilot/.env <<EOF
ANTHROPIC_API_KEY=sk-ant-…
OPENAI_API_KEY=sk-…
GEMINI_API_KEY=AIza…
EOF
chmod 600 scripts/pilot/.env

# Request an interactive compute node (12 h to cover the pilot).
srun -p venkvis-cpu -N 1 --cpus-per-task=16 --mem=32G \
     -t 12:00:00 --pty bash
```

When the compute-node shell appears:

```bash
module load singularity/4.4.1
cd /nfs/turbo/coe-venkvis/$USER/projects/DiffEC

# === SMOKE TEST FIRST — verify singularity + images work (~15 min, free) ===
harbor run --path tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport \
           --env singularity \
           --agent oracle \
           --yes
# Should report reward = 1 at the end. If it doesn't:
#   - check images are public (Phase 2)
#   - check task.toml has docker_image lines pushed to the branch (Phase 3)
#   - check singularity cache is writable: ls -la $SCRATCH_DIR/singularity_cache
# Don't proceed to the full pilot until the smoke test passes.

# === FULL PILOT (3 agents × 10 trials, ~8 h, $200–500) ===
bash scripts/pilot/run_pilot_singularity.sh

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
| Singularity pull `unauthorized` | Image still private | redo Phase 2 (visibility = public) |
| `harbor` exits with "no `docker_image` field" | task.toml not updated or not pushed | redo Phase 3, push, then `git pull` on Artemis |
| Singularity cache permission errors | `$SCRATCH_DIR` not writable | `mkdir -p /tmp/sing_cache && export SINGULARITY_CACHE=/tmp/sing_cache` |
| Pilot exits with API auth error | API key wrong or env file not loaded | check `scripts/pilot/.env` is present and not empty; re-run |
| Pilot trial timeouts | Agent stuck | increase `AGENT_TIMEOUT_MULT=1.5 bash scripts/pilot/run_pilot_singularity.sh` |
| Cost too high | Cut attempts | `N_ATTEMPTS=5 bash scripts/pilot/run_pilot_singularity.sh` (looser CI, half cost) |

## Where this fits in CLAUDE.md's Definition of Done

This runbook closes DoD items 2 (`harbor run -a oracle` reward = 1) and
6 (frontier-agent pilot 10–20 % solve rate). Item 7 (the actual PR) is
unblocked once these complete.
