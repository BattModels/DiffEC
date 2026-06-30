#!/bin/bash
# Singularity variant of run_pilot.sh for HPC clusters (e.g. Artemis)
# that have Singularity/Apptainer but no Docker.
#
# Prerequisites:
#   1. `scripts/build_and_push.sh` already ran ONCE on a Docker host;
#      task.toml now has `docker_image = "ghcr.io/…"` lines for both
#      [environment] and [verifier.environment].
#   2. The GHCR images are PRIVATE (they hold the held-out truth), so
#      export GHCR credentials before running so Singularity can pull:
#         export SINGULARITY_DOCKER_USERNAME=<ghcr-handle>
#         export SINGULARITY_DOCKER_PASSWORD=<read:packages PAT>
#      (Apptainer: use the APPTAINER_DOCKER_USERNAME/PASSWORD names.)
#   3. `harbor` 0.16+ installed: `uv tool install harbor`.
#   4. `singularity` available: on Artemis, `module load singularity/4.4.1`.
#   5. scripts/pilot/.env contains the API keys.
#
# Usage:
#     module load singularity/4.4.1   # Artemis-specific
#     bash scripts/pilot/run_pilot_singularity.sh

set -euo pipefail

# --- Tunables --------------------------------------------------------
N_ATTEMPTS="${N_ATTEMPTS:-10}"
N_CONCURRENT="${N_CONCURRENT:-2}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
JOBS_DIR="${JOBS_DIR:-jobs}"
ENV_FILE="${ENV_FILE:-scripts/pilot/.env}"
# Default cache on Artemis scratch (60-day auto-purge OK; .sif files are
# regenerated on first pull). Override for other clusters.
SINGULARITY_CACHE="${SINGULARITY_CACHE:-${SCRATCH_DIR:-/scratch/venkvis_root/venkvis/$USER}/singularity_cache}"

# Comment out lines below to skip an agent.
AGENTS=(
    "claude-code:claude-opus-4-7:opus47"
    "codex:gpt-5:gpt5"
    "gemini-cli:gemini-2.5-pro:gemini25"
)

TASK="tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport"

# --- Pre-flight ------------------------------------------------------
command -v harbor       >/dev/null || { echo "ERROR: harbor CLI not in PATH"; exit 2; }
command -v singularity  >/dev/null || {
    echo "ERROR: singularity not in PATH."
    echo "       On Artemis: module load singularity/4.4.1"
    exit 2
}
[ -d "$TASK" ] || { echo "ERROR: task dir not found: $TASK"; exit 2; }

# Sanity-check: task.toml must have docker_image set or Singularity bails.
if ! grep -q '^docker_image\s*=' "$TASK/task.toml" 2>/dev/null; then
    echo "ERROR: $TASK/task.toml has no 'docker_image = ...' line."
    echo "       Run scripts/build_and_push.sh first, then add the lines"
    echo "       printed at the end of that script."
    exit 2
fi

# Images are private — warn early if GHCR credentials aren't exported.
if [ -z "${SINGULARITY_DOCKER_USERNAME:-}${APPTAINER_DOCKER_USERNAME:-}" ]; then
    echo "WARNING: neither SINGULARITY_DOCKER_USERNAME nor APPTAINER_DOCKER_USERNAME"
    echo "         is set. The GHCR images are private; the pull will fail with"
    echo "         'unauthorized'. Export a read:packages PAT first (see header)."
fi

mkdir -p "$JOBS_DIR" "$SINGULARITY_CACHE"
export SINGULARITY_CACHEDIR="$SINGULARITY_CACHE"

ENV_FILE_FLAG=""
if [ -f "$ENV_FILE" ]; then
    ENV_FILE_FLAG="--env-file $ENV_FILE"
    echo "Using env file: $ENV_FILE"
else
    echo "No env file at $ENV_FILE — relying on shell-exported API keys."
fi

# --- Run -------------------------------------------------------------
TS="$(date -u +%Y%m%d-%H%M%S)"

for spec in "${AGENTS[@]}"; do
    IFS=":" read -r agent model short <<< "$spec"
    job_name="pilot-${short}__${TS}"
    echo
    echo "=== $agent : $model — $N_ATTEMPTS attempts (singularity) → $JOBS_DIR/$job_name/ ==="
    set +e
    harbor run \
        --path "$TASK" \
        --env singularity \
        --ek "singularity_image_cache_dir=$SINGULARITY_CACHE" \
        --agent "$agent" \
        --model "$model" \
        --n-attempts "$N_ATTEMPTS" \
        --n-concurrent "$N_CONCURRENT" \
        --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT" \
        --jobs-dir "$JOBS_DIR" \
        --job-name "$job_name" \
        --quiet \
        $ENV_FILE_FLAG \
        --yes
    ec=$?
    set -e
    echo "=== $agent done (exit $ec) ==="
done

echo
echo "Pilot complete. Aggregate with:"
echo "  uv run python scripts/pilot/aggregate.py --jobs $JOBS_DIR --out docs/progress/pilot_run.md"
