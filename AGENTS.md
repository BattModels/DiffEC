# DiffEC → Terminal-Bench Science task

**For any assistant opening this repo, `CLAUDE.md` is the authoritative
project brief.** It covers goal, task summary, architecture, code
style, workflow, and Definition of Done. Read it first.

This file exists so Codex CLI (which reads `AGENTS.md` by default)
sees the same context Claude Code does. Content is intentionally
kept slim to avoid the two files drifting.

## What you're most likely being asked to do

If the user has opened you on the laptop, they almost certainly want
to drive one of two things:

1. **Mock exam** (`mock_exam/README.md`) — pre-submission sanity
   check. Runs the same 3 frontier agents (Claude Opus via Claude
   Code, GPT-5 via Codex, Gemini 2.5 via Gemini CLI) for **1 trial
   each**, using Harbor's `environment_mode="separate"` container
   isolation to guarantee the agent sees only what a real TB-Science
   evaluator would see. Do this BEFORE the full pilot. `bash
   mock_exam/show_agent_view.sh` prints the exact file surface the
   agent sees; `bash mock_exam/inspect_container.sh` starts an
   interactive shell inside the agent image so you can confirm by
   hand.
2. **Full pilot** (`scripts/pilot/README.md` + `docs/laptop-runbook.md`
   Part 2) — 10 trials × 3 agents to measure solve-rate distribution
   per ADR-0008. Only after mock trials confirm the loop works.

Auth setup is shared between both (see `docs/laptop-runbook.md`
Step 6b): subscription (Claude Max/Pro, ChatGPT Plus/Pro, Gemini
free tier) or paid API key.

If the user asks for anything else (case designs, verifier, oracle,
PR prep), route to `CLAUDE.md` → `docs/plan/` and follow that.

## Non-negotiable guardrails

- All pushes go to `origin` (= `ChangwenXu98/DiffEC`, personal fork).
  `upstream` (= `BattModels/DiffEC`) is fetch-only. `docs/dev-setup.md`
  has the remote topology.
- Do NOT modify `docs/proposal/` — that's the frozen TB-Science
  contract.
- Do NOT commit `scripts/pilot/.env`, `jobs/`, or `_local_jobs/`
  (already `.gitignore`d). Auth files live in `$HOME`, not in the
  repo — see `docs/laptop-runbook.md` §"What NOT to do".
- Do NOT run the full pilot without explicit user go-ahead. Even in
  subscription mode it burns quota.
- Do NOT revert the PILOT-ONLY `docker_image` lines in `task.toml`
  without an accompanying PR-open action (they're intentionally
  present for the pilot; `scripts/pre_pr_audit.sh` check #8
  enforces reverting them before the upstream PR).

## Session hygiene

- The running experience notebook is `docs/session.md` (gitignored).
  Update it when you finish a chunk of work or make a non-obvious
  decision, so the next session can resume cold.
- The pre-PR audit script is `scripts/pre_pr_audit.sh` — run it
  before proposing any push that touches `tasks/…/chemistry/…/`.
