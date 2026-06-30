## Artemis Cluster Overview

33 nodes total: 25 CPU, 3 large-memory, 3 H100 GPU, 2 A100 GPU.

### Partitions

| Partition | Wall Time | Nodes | CPUs | RAM | GPUs | Notes |
|-----------|-----------|-------|------|-----|------|-------|
| `venkvis-cpu` | 48h | 25 | 96c (EPYC 9654) | 368 GB | — | Default for DFT |
| `venkvis-largemem` | 48h | 3 | 96c (EPYC 9654) | 768 GB | — | Large-memory jobs |
| `venkvis-a100` | 8h | 2 | 32c (EPYC 7513) | 512 GB | 4× A100 80GB | GPU compute |
| `venkvis-h100` | 8h | 3 | 96c (EPYC 9654) | 368 GB | 4× H100 80GB | GPU compute (fastest) |
| `venkvis-debug` | 30m | 4 max | varies | varies | varies | Quick tests |

### QOS (Quality of Service)

| QOS | Wall Time | Resources | Notes |
|-----|-----------|-----------|-------|
| `venkvis-short` | 4h | 2 nodes reserved from `venkvis-cpu` | Fast-turnaround CPU jobs and interactive sessions |

The general `venkvis-cpu` partition allows wall times up to 48h, so short jobs and interactive sessions can queue behind long-running ones. The `venkvis-short` QOS reserves two `venkvis-cpu` nodes for short jobs to avoid this.

- sbatch: `#SBATCH --qos=venkvis-short`
- interactive: `srun --qos=venkvis-short ...`
- Open OnDemand: select `venkvis-short` from the QOS dropdown.

Jobs requesting more than 4h wall time under this QOS are **rejected** — size accordingly. Omitting the QOS runs the job on the general `venkvis-cpu` pool as before; existing workflows are unaffected.

### Storage

| Tier | Path | Capacity | Notes |
|------|------|----------|-------|
| Turbo | `/nfs/turbo/coe-venkvis/` | 10 TB (500 GB fair share) | Persistent, backed up |
| Scratch | `/scratch/venkvis_root/venkvis/` | 10 TB (500 GB fair share) | **60-day auto-purge** |
| Home | `/home/<user>` | 80 GB | User home |
| Node Local | `/tmp` | 1.9 TB NVMe | Ephemeral, fast I/O |

## Containers (Singularity, no Docker)

Artemis has **no native Docker**, but does have **Singularity 4.4.1**
(module-loadable) and an installed `podman` (rootless mode unavailable
because `/etc/subuid` lacks user entries).

```bash
module load singularity/4.4.1
singularity --version            # singularity-ce version 4.4.1
```

Capabilities checked 2026-06-29:

| Capability | Available? | Notes |
|---|---|---|
| `singularity exec docker://image` | ✓ | pulls any Docker image from a registry, converts to `.sif`, executes. |
| `singularity build .sif file.def` (from a Singularity recipe) | ⚠ (no fakeroot) | `--fakeroot` errors with `no valid mapping entry found for $USER` because the cluster has no subuid mappings. |
| `singularity build --remote` | ✓ (with Sylabs account) | Builds remotely via Sylabs Cloud. Free signup. |
| Pre-built Docker images from public registries (ghcr.io, dockerhub) | ✓ | Anonymous pull works for public images. |

**Implication for Harbor.** The Harbor framework's `--env singularity`
mode (NVIDIA-contributed; see
`harbor/environments/singularity/singularity.py`) requires a *pre-built*
Docker image referenced by `docker_image = "..."` in `task.toml`. The
canonical workflow for our task:

1. Build the agent + verifier Docker images **once** on a Docker host
   (laptop or GitHub Codespaces).
2. Push them to `ghcr.io/<your-handle>/diffec-{env,tests}:vN` and
   **keep both private** — the verifier image bundles the held-out
   ground truth and publishing it would leak the benchmark answers.
3. Add `docker_image = "..."` lines to `task.toml`.
4. On Artemis, `module load singularity/4.4.1`, export
   `SINGULARITY_DOCKER_USERNAME` + `SINGULARITY_DOCKER_PASSWORD`
   (a `read:packages` GitHub PAT) so singularity can pull, and run
   the pilot via `scripts/pilot/run_pilot_singularity.sh`.

The helper `scripts/build_and_push.sh` handles steps 1-2 end-to-end.

## Writing a SLURM Submission Script

Create a bash script with `#SBATCH` directives. Example for a GPU job:

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --partition=venkvis-h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err

# Activate Python environment
source /path/to/venv/bin/activate

# Export any needed API keys
export HF_TOKEN="..."
export MP_API_KEY="..."

# Run your computation
python my_script.py --arg1 value1 --format json > results.json

echo "Done: $(date)"
```

For CPU jobs, remove `--gres` and use `--partition=venkvis-cpu`. For short CPU jobs (≤4h), add `--qos=venkvis-short` to use the reserved fast-lane nodes.

Key `#SBATCH` directives:
- `--partition=<name>` — which queue (see table above)
- `--qos=<name>` — quality of service (e.g. `venkvis-short` for ≤4h CPU jobs)
- `--gres=gpu:<N>` — request N GPUs (GPU partitions only)
- `--time=HH:MM:SS` — wall time limit
- `--mem=<N>G` — memory per node
- `--cpus-per-task=<N>` — CPU cores
- `--output=<path>` / `--error=<path>` — stdout/stderr files (`%j` = job ID)
- `--array=0-9` — submit a job array (10 tasks)

## Submitting Jobs

```bash
# Submit a script
sbatch submit.sh

# Submit with partition override
sbatch --partition=venkvis-h100 submit.sh

# Submit with QOS (short-job fast lane, ≤4h CPU)
sbatch --qos=venkvis-short submit.sh

# Submit with dependency (run after job 12345 completes)
sbatch --dependency=afterok:12345 next_step.sh
```

Output: `Submitted batch job 12345`

## Checking Job Status

```bash
# Check your running/pending jobs
squeue -u $USER

# Check a specific job
squeue -j 12345

# Check a specific partition
squeue -p venkvis-h100

# Detailed job info (including completed jobs)
sacct -j 12345 --format=JobID,State,Elapsed,ExitCode,NodeList,MaxRSS

# Check estimated start time for pending job
squeue -j 12345 --start
```

Key job states: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`.

## Retrieving Results

After a job completes, results are wherever your script wrote them:

```bash
# Check if job finished
sacct -j 12345 --format=JobID,State,Elapsed,ExitCode --noheader

# Read stdout/stderr
cat slurm-12345.out
cat slurm-12345.err

# Read structured results (if your script wrote JSON)
cat results.json | python3 -m json.tool
```

## Cancelling Jobs

```bash
# Cancel a specific job
scancel 12345

# Cancel all your jobs
scancel -u $USER

# Cancel all pending jobs
scancel -u $USER --state=PENDING
```

## Interactive GPU Sessions

For quick debugging or running with GPU access:

```bash
# Interactive shell with H100 GPU (up to 8 hours)
srun -N 1 -n 1 -p venkvis-h100 --gres=gpu:h100:1 --mem=32G -t 04:00:00 --pty bash

# Interactive shell with A100 GPU
srun -N 1 -n 1 -p venkvis-a100 --gres=gpu:a100:1 --mem=32G -t 04:00:00 --pty bash

# Quick debug session (30 min max, fastest scheduling)
srun --partition=venkvis-debug --nodes=1 --gres=gpu:h100:1 --mem=2G --time=30 --pty bash

# Interactive CPU session on the short-job fast lane (≤4h)
srun -p venkvis-cpu --qos=venkvis-short --mem=8G -t 02:00:00 --pty bash
```

## Running the Harbor pilot on Artemis

Step-by-step procedure for the frontier-agent pilot (ADR-0008). The
**full end-to-end procedure** (laptop build → Artemis pilot) lives in
`docs/pre-pr-runbook.md` and supersedes this section if they ever
disagree. See also `scripts/pilot/README.md` for the agent / cost
rationale.

**Off-cluster prep (one-time, ~30 min, on a Docker host):**

```bash
# 1. Have docker + gh CLI installed and authenticated.
docker --version
gh auth status

# 2. Build the two images and push to ghcr.io. Both stay PRIVATE
#    (diffec-tests bundles the held-out truth via COPY . /tests/ —
#    publishing it would leak the benchmark answers).
GHCR_USER=<your-github-handle> bash scripts/build_and_push.sh

# 3. Add the two docker_image lines to task.toml (the script prints them).
#    Commit and push. Mark them PILOT-ONLY (revert before upstream PR).
```

**On Artemis (the actual pilot):**

```bash
# 1. Sync the repo to Artemis if not already.
cd /nfs/turbo/coe-venkvis/$USER/projects/DiffEC
git pull

# 2. Load singularity + ensure harbor CLI is on PATH.
module load singularity/4.4.1
uv tool install harbor      # one-time, user-local

# 3. Put API keys in scripts/pilot/.env (gitignored).
cat > scripts/pilot/.env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
EOF
chmod 600 scripts/pilot/.env

# 4. Request a long-running interactive CPU node (the pilot needs
#    ~8h walltime, so use the general venkvis-cpu, not venkvis-short).
srun -p venkvis-cpu -N 1 --cpus-per-task=16 --mem=32G \
     -t 12:00:00 --pty bash
# (Wait for allocation. Once you have a shell on the compute node:)

module load singularity/4.4.1
cd /nfs/turbo/coe-venkvis/$USER/projects/DiffEC

# 5. Authenticate to GHCR so singularity can pull the private images.
#    Create a classic PAT with read:packages scope at
#    https://github.com/settings/tokens, then:
export SINGULARITY_DOCKER_USERNAME=<your-github-handle>
export SINGULARITY_DOCKER_PASSWORD=ghp_xxxxxxxx
# (If the cluster ships Apptainer rather than Singularity, use the
#  APPTAINER_DOCKER_USERNAME / APPTAINER_DOCKER_PASSWORD names.)

# 6. Run the pilot.
bash scripts/pilot/run_pilot_singularity.sh

# 7. Aggregate.
uv run python scripts/pilot/aggregate.py \
    --jobs jobs --out docs/progress/pilot_run.md
```

Singularity caches `.sif` files under `$SCRATCH_DIR/singularity_cache`
(override with `SINGULARITY_CACHE`). The cache survives across runs but
gets scratch-purged after 60 days — that's fine, the .sif is just a
local mirror of the GHCR image.

If you want to run the pilot as a batch job instead of interactively,
wrap the `run_pilot_singularity.sh` call in an `sbatch` script with
`--partition=venkvis-cpu --time=12:00:00` (the pilot is mostly idle
wait on agent execution, so 16 cores is plenty).

## Safety Rules

- **Never submit from inside a compute node** — check with `echo $SLURM_JOB_ID` (should be empty on login node)
- **Never install packages globally** — always use a virtualenv
- **Write large temporary data to `/scratch/`**, not `/nfs/turbo/` or `/home/`
- **Respect wall time limits** — GPU partitions have 8h max, CPU has 48h, `venkvis-short` QOS has 4h
- Jobs inherit environment variables from the submitting shell by default