#!/bin/bash
# Frontier-agent pilot driver (ADR-0008). Invokes `harbor run` once per
# agent with N_ATTEMPTS trials. Outputs to jobs/pilot-<agent>__<timestamp>/.
# See README.md for prerequisites (Docker, harbor 0.16+, .env auth).
# The three agents (claude-code, codex, gemini-cli) each accept either
# a paid API key or the user's existing subscription auth — see
# README.md §"Pre-flight" for env vars and docs/laptop-runbook.md
# Step 6b for the one-time host setup.

set -euo pipefail

# --- Tunables --------------------------------------------------------
N_ATTEMPTS="${N_ATTEMPTS:-10}"
N_CONCURRENT="${N_CONCURRENT:-2}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
JOBS_DIR="${JOBS_DIR:-jobs}"
ENV_FILE="${ENV_FILE:-scripts/pilot/.env}"

# Comment out lines below to skip an agent.
AGENTS=(
    "claude-code:claude-opus-4-7:opus47"
    "codex:gpt-5:gpt5"
    "gemini-cli:gemini-2.5-pro:gemini25"
)

TASK="tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"

# --- Pre-flight ------------------------------------------------------
command -v harbor >/dev/null || { echo "ERROR: harbor CLI not in PATH"; exit 2; }
command -v docker >/dev/null || { echo "ERROR: docker not in PATH"; exit 2; }
[ -d "$TASK" ] || { echo "ERROR: task dir not found: $TASK"; exit 2; }
mkdir -p "$JOBS_DIR"

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
    echo "=== $agent : $model — $N_ATTEMPTS attempts → $JOBS_DIR/$job_name/ ==="
    # `harbor run` exit code is non-zero if ANY trial errors (distinct
    # from "verifier returned 0"). Don't `set -e` on this; just log.
    set +e
    harbor run \
        --path "$TASK" \
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
