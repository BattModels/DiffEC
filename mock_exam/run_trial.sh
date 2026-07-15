#!/bin/bash
# Run one mock-exam trial against a single frontier agent, using
# Harbor's isolated Docker containers. See mock_exam/README.md for
# context; docs/laptop-runbook.md Step 6b for auth setup.
#
# Usage:
#   bash mock_exam/run_trial.sh <agent> <model>
#
# Example:
#   bash mock_exam/run_trial.sh claude-code claude-opus-4-7
#   bash mock_exam/run_trial.sh codex       gpt-5
#   bash mock_exam/run_trial.sh gemini-cli  gemini-2.5-pro

set -euo pipefail

AGENT="${1:-}"
MODEL="${2:-}"

if [ -z "$AGENT" ] || [ -z "$MODEL" ]; then
    cat <<EOF >&2
usage: $0 <agent> <model>

supported agents (built into Harbor 0.16):
  claude-code   e.g., claude-opus-4-7 or claude-sonnet-4-6
  codex         e.g., gpt-5
  gemini-cli    e.g., gemini-2.5-pro
EOF
    exit 2
fi

TASK="tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"
JOBS_DIR="mock_exam/results"
ENV_FILE="scripts/pilot/.env"

TS="$(date -u +%Y%m%d-%H%M%S)"
short="$(echo "$AGENT" | tr -d '-_')"
JOB="mock-${short}__${TS}"

[ -d "$TASK" ] || { echo "ERROR: task dir not found: $TASK — run from repo root."; exit 2; }
command -v harbor >/dev/null || { echo "ERROR: harbor CLI not in PATH — 'uv tool install harbor'"; exit 2; }
command -v docker >/dev/null || { echo "ERROR: docker not in PATH — start Docker Desktop / Colima"; exit 2; }
mkdir -p "$JOBS_DIR"

ENV_FLAG=""
if [ -f "$ENV_FILE" ]; then
    ENV_FLAG="--env-file $ENV_FILE"
    echo "Using env file: $ENV_FILE"
else
    echo "WARNING: no $ENV_FILE — relying on shell-exported auth env vars."
    echo "         See docs/laptop-runbook.md Step 6b before proceeding."
fi

echo
echo "=== $AGENT : $MODEL — 1 attempt → $JOBS_DIR/$JOB/ ==="
echo "  This is a single-trial cold attempt. The agent container is built"
echo "  from tasks/.../environment/Dockerfile and sees only the four case"
echo "  bundles + instruction.md. Nothing else in this repo is inside the"
echo "  container. See mock_exam/show_agent_view.sh for the exact surface."
echo

harbor run \
    --path "$TASK" \
    --agent "$AGENT" \
    --model "$MODEL" \
    --n-attempts 1 \
    --n-concurrent 1 \
    --jobs-dir "$JOBS_DIR" \
    --job-name "$JOB" \
    $ENV_FLAG \
    --yes

echo
echo "=== done: $JOBS_DIR/$JOB/ ==="
echo "  reward.txt:            $(cat "$JOBS_DIR/$JOB"/*/reward.txt 2>/dev/null || echo '(not found)')"
echo "  agent transcript:      $JOBS_DIR/$JOB/*/agent/session.log"
echo "  verifier pytest log:   $JOBS_DIR/$JOB/*/verifier/pytest.log"
