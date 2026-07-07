# Mock exam — frontier agents attempt the task cold, before TB-Science

This directory is a **pre-submission mock exam**. Before we submit the
task to `harbor-framework/terminal-bench-science`, we want to confirm
that frontier agents (Claude Opus 4.7 via Claude Code, GPT-5 via
Codex, Gemini 2.5 via Gemini CLI) can attempt it as if it were live —
seeing only what a real TB-Science evaluator would see.

## What the agent sees (and only that)

Harbor's `environment_mode = "separate"` in `task.toml` splits the
task into two Docker images:

- **Agent image** (built from `tasks/…/environment/Dockerfile`).
  The agent container has: `instruction.md`, and everything the
  Dockerfile COPYs from `environment/` — i.e., the four case bundles
  `case_X/{data.h5, params.json, formalism.md}` under `/root/data/`.
  Nothing else.
- **Verifier image** (built from `tasks/…/tests/Dockerfile`).
  Holds the oracle + truth + pytest checks. Never reachable from
  the agent container.

**What the agent does NOT see** — the rest of this repo. `solution/`
(reference solver), `tests/` (oracle + truth), `case_gen/` (case
generator + configs with hidden params), `docs/` (design docs,
proposal, key facts, session notebook), `CLAUDE.md`, `AGENTS.md`,
`scripts/pilot/`, and everything else at the repo root — none of
this is inside either container image.

We verified this is enforced by:

1. Harbor packages tasks by `COPY environment/ …` in the agent
   Dockerfile — no `..`/parent-directory reads possible.
2. The `claude-code` adapter does not bind-mount `~/.claude` from
   host, does not read the user's project memory, and does not
   read the invocation directory's `CLAUDE.md`. (Confirmed by
   auditing `harbor/agents/installed/claude_code.py` lines
   1213-1229 + 1410-1421 on 2026-07-01.)
3. `task.toml` does not set `skills_dir`, so no host paths are
   passed through the adapter.

## How to run the mock exam

**Prerequisites.** Same as the full pilot — see
`docs/laptop-runbook.md` Part 2 Step 6b for auth setup
(subscription or paid API key per agent, one-time). This mock
exam re-uses `scripts/pilot/.env`.

**1. Show the agent's view first** (~5 s):

```bash
bash mock_exam/show_agent_view.sh
```

This prints the exact file surface that ends up inside the agent
container. Skim it — confirm the list contains only what an outside
evaluator should see.

**2. Optional — build the agent image and shell into it** (~5 min,
~2 GB):

```bash
bash mock_exam/inspect_container.sh
```

Starts a bash shell inside the freshly built agent image. Poke
around `/root/`, `/opt/`, `$PATH` — this is bit-for-bit what the
frontier agent's shell sees.

**3. Run one mock trial per agent** (~30 min each, sequential):

```bash
bash mock_exam/run_trial.sh claude-code claude-opus-4-7
bash mock_exam/run_trial.sh codex       gpt-5
bash mock_exam/run_trial.sh gemini-cli  gemini-2.5-pro
```

Each trial: single attempt, ~30 min budget. Output goes to
`mock_exam/results/mock-<agent>__<timestamp>/`. The final
`reward.txt` will be `1` if the agent's `transport.json` passes
all 7 verifier checks × 4 cases, `0` otherwise.

## When to run this vs. the full pilot

- **Mock exam** (this dir) — single trial per agent, verifies the
  auth path and container isolation work end-to-end. Do this
  before the pilot, and revisit any time we change the case
  bundles or the instruction.
- **Full pilot** (`scripts/pilot/run_pilot.sh`) — 10 trials × 3
  agents, produces the solve-rate distribution ADR-0008 asks for.
  Only run after mock trials confirm the loop works.

Once all three mock trials complete (reward = 0 or 1 with no
crashes), we've evidence the mock exam is legitimate and can
proceed to the full pilot.

## What to look at after a trial

Under `mock_exam/results/mock-<agent>__<ts>/`:

- `reward.txt` — 1 (all-green) or 0.
- `verifier/pytest.log` — which of the 7 checks × 4 cases failed
  and how far off the agent was.
- `agent/session.log` — the agent's transcript.

Anything surprising there (agent asked to read `docs/`, tried to
reach the host filesystem, referenced private context) — flag it,
because it means the isolation model has a hole we haven't
plugged.
