# DiffEC → Terminal-Bench Science task

**For any assistant opening this repo, `CLAUDE.md` is the authoritative
project brief.** It covers goal, task summary, architecture, code
style, workflow, and Definition of Done. Read it first.

This file exists so Codex CLI (which reads `AGENTS.md` by default)
sees the same context Claude Code does. Content is intentionally
kept slim to avoid the two files drifting.

## What you're most likely being asked to do

If the user has opened you on the laptop, they almost certainly want
to drive the frontier-agent pilot (ADR-0008). The self-contained
runbook is:

- **`docs/laptop-runbook.md`** — laptop-side end-to-end procedure.
  Part 1 is the Docker + Harbor smoke test (DoD #2, already passed
  on 2026-07-01). Part 2 is the 3-agent × 10-trial pilot with a
  Step 6b for one-time auth setup. Both parts have failure/recovery
  tables in the appendix.
- **`scripts/pilot/README.md`** — per-script reference for the pilot
  driver. Covers subscription auth (Claude Max/Pro, ChatGPT Plus/Pro,
  Gemini free tier) and paid-API-key fallback contracts per adapter.

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
