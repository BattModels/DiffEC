#!/bin/bash
# Print the exact file surface an agent's Docker container will see
# when it attempts our TB-Science task. Read-only, offline.
#
# The claim (see mock_exam/README.md): Harbor's environment_mode="separate"
# gives the agent a container built ONLY from tasks/.../environment/ +
# instruction.md; nothing else in this repo reaches the agent.
#
# This script proves that by listing exactly what's inside those paths.

set -euo pipefail

TASK="tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"

if [ ! -d "$TASK" ]; then
    echo "ERROR: task dir not found at $TASK — run from repo root." >&2
    exit 2
fi

banner() {
    printf '\n=== %s ===\n' "$1"
}

banner "1. Task prompt the agent receives (instruction.md)"
echo "path: $TASK/instruction.md"
wc -l "$TASK/instruction.md"
echo "first 40 lines:"
head -40 "$TASK/instruction.md" | sed 's/^/  /'
echo "..."

banner "2. environment/ subtree — everything COPY'd into the agent container"
find "$TASK/environment" -type f | sort | sed 's/^/  /'

banner "3. environment/Dockerfile — the container recipe (reveals installed tools)"
sed 's/^/  /' "$TASK/environment/Dockerfile"

banner "4. task.toml — isolation-relevant settings"
grep -E "^(schema_version|environment_mode|\[environment\]|\[verifier|allow_)" \
    "$TASK/task.toml" | sed 's/^/  /'

banner "5. WHAT THE AGENT DOES NOT SEE (verify by inspection)"
cat <<'MSG'
  Nothing outside tasks/.../{environment,instruction.md} is inside the
  agent image. In particular NONE of the following reach the agent:

    tasks/.../solution/          — reference solver (would give the answer)
    tasks/.../tests/             — oracle + truth + verifier
    tasks/.../tests/oracle_truth/ — ground-truth D, t+0, regime, v arrays
    case_gen/                    — case configs with true D(c), t+0(c) params
    docs/                        — proposal, plans, key facts, session log
    scripts/                     — pilot/audit tooling
    CLAUDE.md, AGENTS.md         — assistant briefs
    _local_results/, jobs/       — prior local outputs

  These live only on the host filesystem and inside our own containers
  (verifier image for tests/, dev environment for case_gen/). The
  agent has no path to reach them.

  Adapter-side confirmation (2026-07-01 audit of
  ~/.local/share/uv/tools/harbor/lib/python3.13/site-packages/harbor/
  agents/installed/claude_code.py lines 1213-1229 + 1410-1421):
    - claude-code adapter does NOT bind-mount host ~/.claude
    - it does NOT read host CLAUDE.md at the invocation dir
    - it does NOT read the user's Claude Code project memory
    - `skills_dir` is copied only if task.toml sets it; ours does not
MSG

banner "6. Everything else in this repo (visible to you, invisible to the agent)"
echo "  Repo root non-agent-visible files:"
ls -1 . | grep -vE '^\.' | grep -v "^tasks$" | sed 's/^/    /'
echo "  Inside tasks/.../ (non-agent-visible siblings):"
ls -1 "$TASK" | grep -vE "^(environment|instruction\.md|task\.toml)$" | sed 's/^/    /'

echo
echo "OK. This is the mock-exam surface. Nothing else reaches the agent."
