# Laptop runbook — drive Harbor smoke test + pilot

**Audience.** A Claude Code (or human) agent working in this repo on
the laptop. Everything below runs on the laptop; nothing needs Artemis.

**Scope right now.** Complete the Harbor **oracle smoke test** only
(Definition-of-Done item #2 in `CLAUDE.md`). **Do NOT run the full
frontier-agent pilot** — the LLM API keys are not ready yet. The pilot
section at the bottom of this doc is for LATER, once the user
explicitly says the keys are ready.

**Why the laptop and not Artemis?** Harbor 0.16 requires either a
running Docker daemon (`--env docker`) or Singularity with
`--fakeroot` (`--env singularity`). Artemis has no Docker, and its
Singularity works but Harbor's fakeroot-based container startup is
blocked on Artemis by missing subuid mappings (empirically confirmed
2026-06-30). Details in `docs/pre-pr-runbook.md` §"Why the laptop"
and `docs/hpc/artemis.md` §"Harbor's --env singularity is BLOCKED".

**Repo topology.** `origin` should point at
`git@github.com:ChangwenXu98/DiffEC.git` (personal fork).
`upstream` = `git@github.com:BattModels/DiffEC.git` (paper source
repo; fetch-only). All pushes go to `origin`. If your local clone
is set up differently, follow `docs/dev-setup.md` first.

---

## Part 1 — Smoke test (do this now, ~30 min, $0)

### Step 1. Preflight (5 min)

```bash
# Adjust path if needed.
cd ~/DiffEC
git checkout feat/tb-sci-task
git pull --ff-only
```

Verify the environment:

```bash
docker --version && docker info | head -3   # Docker daemon reachable
uv --version                                  # uv installed
```

**If `docker info` errors with "Cannot connect to the Docker daemon":**
open Docker Desktop and wait for the whale icon to stop animating.
Grant any first-launch permission dialogs (network / privileged
helper). Retry `docker info` — proceed only when it succeeds.

**If `uv` is missing:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# then reopen the shell.
```

### Step 2. Install harbor CLI (~1 min, one-time per laptop)

```bash
uv tool install harbor
harbor --version                              # expect 0.16.x
```

### Step 3. Prepare `task.toml` for local Dockerfile build (~1 min)

`task.toml` currently has two `docker_image = "ghcr.io/…"` lines
(from commit `e12cff9`) that point at private GHCR images. They
were added for the (now-blocked) Artemis Singularity path. For the
smoke test we want Harbor to build from `environment/Dockerfile`
and `tests/Dockerfile` directly — no GHCR pull, no auth.

**Recommended:** stash the lines for the duration of the smoke test
(they'll come back with `git stash pop`):

```bash
git stash push -m "PILOT-ONLY docker_image lines (temp for smoke test)" \
    tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/task.toml

# Verify:
grep -c '^docker_image' \
    tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/task.toml
# Expected output: 0
```

If it prints anything other than `0`, the stash didn't take —
inspect the file manually and remove any `docker_image = "…"` lines
before proceeding.

### Step 4. Run the oracle smoke test (~15 min, free)

```bash
harbor run \
    --path tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport \
    --env docker \
    --agent oracle \
    --yes
```

**Expected output** (last few lines):

```
Trials  Exceptions  Mean
1       0           1.000
```

`Mean = 1.000` is the pass. This satisfies CLAUDE.md **DoD item #2**.

**If Mean = 0.000 or Exceptions ≥ 1**, do NOT proceed. Save the last
50 lines of output:

```bash
harbor run … 2>&1 | tee /tmp/smoke-output.log
tail -50 /tmp/smoke-output.log
```

Common causes and fixes:

| Symptom | Cause | Fix |
|---|---|---|
| "Cannot connect to the Docker daemon" | Docker Desktop not running | Open it, wait for whale icon, retry |
| "no space left on device" | <6 GB free disk | Free space or move Docker's root |
| Build error on `RUN uv pip install …` | Slow network / uv mirror flake | Retry once; if repeated, check `uv pip install jax` works locally |
| `docker_image` referenced but not pullable | Step 3 didn't take | Re-run the `grep -c '^docker_image'` check |

If the fix isn't obvious, stop here and report the tail of the log to
the user.

### Step 5. Record the smoke-test outcome

If the smoke test passed:

```bash
# Restore the PILOT-ONLY docker_image lines (they're still valuable
# for the fallback Singularity path if subuid ever lands on Artemis).
git stash pop

# Working tree should be clean now:
git status
```

Then **stop here** and tell the user:

> Smoke test PASSED. `harbor run --env docker --agent oracle` returned
> Mean = 1.000. DoD #2 is satisfied. Ready to run the full pilot when
> API keys are available (see Part 2 below).

If the smoke test FAILED after basic troubleshooting, restore the
task.toml (`git stash pop`) and report the failure to the user with
the last 50 lines of the smoke-test log.

### Step 6. STOP. Wait for the user.

**Do NOT proceed to Part 2. The full pilot burns real money (~$200–500)
and requires LLM API keys that are not ready yet.** Only proceed when
the user explicitly says something like "API keys are ready, run the
pilot" and confirms the three keys are in `scripts/pilot/.env`.

---

## Part 2 — Full pilot (LATER; gated on user go-ahead + API keys)

**Prerequisites for this part:**

- Part 1 (smoke test) completed successfully.
- User has confirmed API keys are ready.
- `scripts/pilot/.env` exists with three lines:
  ```
  ANTHROPIC_API_KEY=sk-ant-...
  OPENAI_API_KEY=sk-...
  GEMINI_API_KEY=AIza...
  ```
  with mode `600` (`chmod 600 scripts/pilot/.env`).

Confirm all three before starting:

```bash
[ -f scripts/pilot/.env ] || { echo "MISSING .env — STOP"; exit 1; }
[ "$(stat -c '%a' scripts/pilot/.env 2>/dev/null || stat -f '%p' scripts/pilot/.env | tail -c 4)" = "600" ] \
    || echo "WARNING: .env should be chmod 600"
grep -c '^ANTHROPIC_API_KEY=' scripts/pilot/.env    # expect 1
grep -c '^OPENAI_API_KEY='    scripts/pilot/.env    # expect 1
grep -c '^GEMINI_API_KEY='    scripts/pilot/.env    # expect 1
```

If anything is missing, STOP and ask the user.

### Step 7. Launch the pilot in tmux (~8 h wall, $200–500)

Run inside `tmux` (or `screen`) so it survives terminal disconnects:

```bash
# Optional cap for a first sanity run: N_ATTEMPTS=3 halves cost + time
# but widens the solve-rate CI. Ask the user which to use if unsure.

tmux new -s diffec-pilot
cd ~/DiffEC
bash scripts/pilot/run_pilot.sh

# Detach with Ctrl-b d  (leaves tmux running in background).
# Reattach later with: tmux attach -t diffec-pilot
```

The pilot runs 3 agents × 10 trials each, 2 concurrent. Per-trial cap
is 1 hour (`agent.timeout_sec = 3600` in `task.toml`); most trials
finish in 5–30 min. Total wall time typically 5–8 hours.

Monitor disk usage: `jobs/pilot-*/` can consume several GB total.
Ensure ~10 GB free before starting.

### Step 8. Aggregate and commit (~5 min)

When the pilot finishes:

```bash
uv run python scripts/pilot/aggregate.py \
    --jobs jobs --out docs/progress/pilot_run.md

# Peek at the result:
head -40 docs/progress/pilot_run.md
```

Report the aggregate solve rate to the user. The ADR-0008 acceptance
band is **10–20 %**. `aggregate.py` prints "within band" / "BELOW" /
"ABOVE" for you.

Then commit and push:

```bash
git add docs/progress/pilot_run.md
git commit -m "Frontier-agent pilot results — ADR-0008"
git push
```

This satisfies CLAUDE.md **DoD item #6**.

### Step 9. STOP. Tell the user the outcome.

Do NOT touch case designs, thresholds, or the PR description on your
own after the pilot completes. Report the results and wait for
direction. The user will decide next steps based on the solve-rate
number.

---

## Appendix — Full failure/recovery table

| Symptom | Cause | Fix |
|---|---|---|
| `gh: command not found` | gh CLI missing | `brew install gh` (macOS) / `apt install gh` (Linux). Not required unless the user asks for GHCR ops. |
| `docker: command not found` | Docker not installed | Install Docker Desktop. If not permitted on this laptop, path A of this runbook cannot proceed — report to user. |
| `harbor: command not found` after install | uv tool bin not on PATH | Run `uv tool update-shell` and reopen shell, or use `~/.local/bin/harbor` directly |
| Smoke test hangs > 30 min | Slow network or huge JAX download | Wait longer (up to 60 min); if truly stuck, `Ctrl-C` and retry |
| Smoke test Mean = 0.000 | Verifier failed; the reference solution didn't produce a valid `transport.json` | Read `jobs/*/…/results.json` and `verifier/pytest.log`; report specifics to user |
| Pilot exits with API auth error | Wrong or expired API key | Verify `scripts/pilot/.env` values with the user; keys are provider-specific format |
| Pilot exits with rate-limit error | Provider rate limits | Lower `N_CONCURRENT` to 1 in `scripts/pilot/run_pilot.sh` env; retry |
| Cost exceeds budget mid-pilot | 3-agent × 10-trial default too much | Cancel with `tmux attach -t diffec-pilot`, `Ctrl-C`; user can then rerun with `N_ATTEMPTS=3` |

## Appendix — What NOT to do

- Do NOT modify `case_gen/configs/case_*.yaml`, `tasks/.../tests/`,
  `tasks/.../solution/`, or any oracle/verifier code without user
  approval. The case designs are calibrated and locked.
- Do NOT push to `upstream` (BattModels/DiffEC). All pushes go to
  `origin` (your fork).
- Do NOT commit `scripts/pilot/.env`, `jobs/`, or `_local_jobs/`
  (already `.gitignore`d).
- Do NOT run the full pilot without explicit user go-ahead. It
  spends real money.
