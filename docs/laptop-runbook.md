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
docker compose version                       # Compose plugin present (Harbor needs it)
docker buildx version                        # BuildKit plugin present (Harbor uses it)
uv --version                                  # uv installed
```

**If `docker info` errors with "Cannot connect to the Docker daemon":**
open Docker Desktop and wait for the whale icon to stop animating.
Grant any first-launch permission dialogs (network / privileged
helper). Retry `docker info` — proceed only when it succeeds.

**If `docker compose version` errors "docker: 'compose' is not a docker
command":** you're using Homebrew's `docker` CLI (no plugins). Install
the two plugins and symlink them:

```bash
brew install docker-compose docker-buildx
mkdir -p ~/.docker/cli-plugins
ln -sfn "$(brew --prefix)/opt/docker-compose/bin/docker-compose" \
        ~/.docker/cli-plugins/docker-compose
ln -sfn "$(brew --prefix)/opt/docker-buildx/bin/docker-buildx" \
        ~/.docker/cli-plugins/docker-buildx
docker compose version    # confirm now works
docker buildx version
```

(Empirically necessary on `brew` Docker CLI, 2026-07-01. Docker
Desktop bundles both.)

**Colima VM sizing** — if you use Colima as the Docker runtime, its
default is 4 CPU / 6 GB, but `task.toml` requests **8 CPU / 16 GB**.
Container creation will fail with a resource-limit rejection unless
the VM is resized:

```bash
# Check current allocation:
colima list

# If cpu < 8 or memory < 16:
colima stop
colima start --cpu 8 --memory 16 --disk 40   # 40 GB disk is plenty for the pilot
```

(Docker Desktop users: check **Preferences → Resources** and set CPUs
≥ 8, Memory ≥ 16 GB.)

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
(from commit `e12cff9`, currently committed to the branch) that
point at private GHCR images. They were added for the (now-blocked)
Artemis Singularity path. For the smoke test we want Harbor to
build from `environment/Dockerfile` and `tests/Dockerfile` directly
— no GHCR pull, no auth.

Because the lines are *committed* (not a working-tree modification),
`git stash push` is a no-op. Use in-place edit + `git checkout` to
restore afterward.

```bash
TASK=tasks/physical-sciences/chemistry/concentrated-electrolyte-transport

# Delete the two docker_image lines and the [verifier.environment]
# section. Edit with your editor of choice, or use this sed one-liner
# (macOS BSD sed syntax; on Linux drop the '' after -i):
sed -i '' -e '/^# PILOT-ONLY/,/^docker_image/d' \
          -e '/^\[verifier\.environment\]/,/^docker_image/d' "$TASK/task.toml"

# Verify:
grep -c '^docker_image' "$TASK/task.toml"        # expect 0
grep -c '^\[verifier\.environment\]' "$TASK/task.toml"  # expect 0
```

If either grep prints > 0, open `$TASK/task.toml` in an editor and
remove the offending lines manually.

**Remember to restore after the smoke test** (Step 5) with
`git checkout -- $TASK/task.toml` — a single command that puts the
file back to its committed state, `docker_image` lines intact.

### Step 4. Run the oracle smoke test (~15 min, free)

```bash
harbor run \
    --path tasks/physical-sciences/chemistry/concentrated-electrolyte-transport \
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
git checkout -- tasks/physical-sciences/chemistry/concentrated-electrolyte-transport/task.toml

# Working tree should be clean now:
git status
```

Then **stop here** and tell the user:

> Smoke test PASSED. `harbor run --env docker --agent oracle` returned
> Mean = 1.000. DoD #2 is satisfied. Ready to run the full pilot when
> API keys are available (see Part 2 below).

If the smoke test FAILED after basic troubleshooting, restore
task.toml (`git checkout -- tasks/.../task.toml`) and report the
failure to the user with the last 50 lines of the smoke-test log.

### Step 6. STOP. Wait for the user.

**Do NOT proceed to Part 2. The full pilot burns real money (~$200–500)
and requires LLM API keys that are not ready yet.** Only proceed when
the user explicitly says something like "API keys are ready, run the
pilot" and confirms the three keys are in `scripts/pilot/.env`.

---

## Part 2 — Full pilot (LATER; gated on user go-ahead + auth setup)

**Prerequisites for this part:**

- Part 1 (smoke test) completed successfully.
- User has confirmed auth is ready (subscription OR paid API key
  per agent — see Step 6b below).
- `scripts/pilot/.env` exists and is `chmod 600`.

The full auth contract is documented in `scripts/pilot/README.md`
§"Pre-flight". Summary: each of the three agents accepts either a
paid API key or the user's existing subscription (Claude Max/Pro,
ChatGPT Plus/Pro, Gemini free tier). You can mix modes per agent —
each Harbor adapter picks its highest-priority available auth.

### Step 6b — one-time auth setup on this laptop (~10 min, once)

**Mode A — subscription auth (recommended, ~$0 marginal).**

Prep the host CLIs first (all three save state into `$HOME`):

```bash
# 1. Claude Code — prints an sk-ant-oat01-... OAuth token to stdout.
#    You paste this token into .env below (adapter has no ~/.claude mount).
claude setup-token
# 2. Codex — browser flow to ChatGPT Plus/Pro.
codex login
test -f ~/.codex/auth.json && echo "codex auth OK"
# 3. Gemini CLI — start interactively, pick "Login with Google".
gemini    # then quit after login completes
test -f ~/.gemini/oauth_creds.json && echo "gemini auth OK"
```

Then write `scripts/pilot/.env`:

```
CLAUDE_CODE_OAUTH_TOKEN=<paste token from step 1>
CLAUDE_FORCE_OAUTH=1
CODEX_FORCE_AUTH_JSON=1
GEMINI_FORCE_OAUTH=1
```

**Mode B — paid API keys (fallback, ~$200–500 pilot).**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

Get keys from [Anthropic Console](https://console.anthropic.com/),
[OpenAI Platform](https://platform.openai.com/api-keys),
[Google AI Studio](https://aistudio.google.com/apikey).

**Mode C — mix.** Combine A and B, e.g., subscription for Claude/Gemini,
paid key for Codex if your ChatGPT plan is too rate-limited.

Then lock down the file:

```bash
chmod 600 scripts/pilot/.env
```

Verify auth resolves for each agent before launching (regardless of mode):

```bash
[ -f scripts/pilot/.env ] || { echo "MISSING .env — STOP"; exit 1; }

# Perms — GNU stat first, BSD stat fallback (macOS).
perms=$(stat -c '%a' scripts/pilot/.env 2>/dev/null || stat -f '%A' scripts/pilot/.env)
[ "$perms" = "600" ] || echo "WARNING: .env should be chmod 600 (found $perms)"

# Each agent needs at least one of its two auth env vars set in .env.
# Mode A subscription requires the host auth file to also exist.
grep -qE '^(CLAUDE_CODE_OAUTH_TOKEN|ANTHROPIC_API_KEY)=' scripts/pilot/.env \
    || echo "MISSING claude-code auth"
grep -qE '^(CODEX_FORCE_AUTH_JSON|OPENAI_API_KEY)='       scripts/pilot/.env \
    || echo "MISSING codex auth"
grep -qE '^(GEMINI_FORCE_OAUTH|GEMINI_API_KEY)='          scripts/pilot/.env \
    || echo "MISSING gemini auth"

# Host-file existence check for subscription mode:
grep -q '^CODEX_FORCE_AUTH_JSON=1'  scripts/pilot/.env \
    && ! [ -f ~/.codex/auth.json ]        && echo "MISSING ~/.codex/auth.json"
grep -q '^GEMINI_FORCE_OAUTH=1'     scripts/pilot/.env \
    && ! [ -f ~/.gemini/oauth_creds.json ] && echo "MISSING ~/.gemini/oauth_creds.json"
```

If any check prints `MISSING`, STOP and either re-run the login flow
in Step 6b Mode A, or fall back to Mode B for that agent.

### Step 6c — mock exam first (~30 min per agent, one trial each)

**Do NOT skip this before running Step 7.** The mock exam is a
pre-submission sanity check: it runs the same agent × task loop as
the full pilot but with only 1 trial per agent, and it lets you
visually inspect what the agent's container sees so you trust the
downstream solve-rate numbers.

The self-contained runbook lives in `mock_exam/README.md`. Summary:

```bash
# 1. Print the exact file surface the agent sees (offline, ~5 sec):
bash mock_exam/show_agent_view.sh

# 2. Optional — build the agent image and shell into it (~5 min):
bash mock_exam/inspect_container.sh    # exit when done

# 3. Run one trial per agent (~30 min each, sequential):
bash mock_exam/run_trial.sh claude-code claude-opus-4-7
bash mock_exam/run_trial.sh codex       gpt-5
bash mock_exam/run_trial.sh gemini-cli  gemini-2.5-pro
```

Read `reward.txt` for each result. Any of {0, 1} is a valid outcome
for the mock — we're checking that the loop runs end-to-end, not
the solve rate. What matters: **no crashes on setup**, **no evidence
of context leak** (check `agent/session.log` for references to files
we didn't ship in `environment/`).

If any of the three crashes, fix that before Step 7 — a crash in
the mock will also crash 10× in the pilot.

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
| Pilot exits with API auth error (paid mode) | Wrong or expired API key | Verify `scripts/pilot/.env` values with the user; keys are provider-specific format |
| Pilot exits with auth error on `claude-code` (subscription mode) | `CLAUDE_CODE_OAUTH_TOKEN` expired or malformed | Re-run `claude setup-token` on host, replace the token in `.env`, retry |
| Pilot exits with auth error on `codex` (subscription mode) | `~/.codex/auth.json` missing or ChatGPT session expired | Re-run `codex login` on host; adapter re-uploads the fresh file on next `harbor run` |
| Pilot exits with auth error on `gemini-cli` (subscription mode) | `~/.gemini/oauth_creds.json` missing or expired | Re-run `gemini` login flow on host; adapter re-uploads on next `harbor run` |
| Pilot exits with rate-limit error | Provider rate limits (esp. ChatGPT Plus) | Lower `N_CONCURRENT` to 1 in `scripts/pilot/run_pilot.sh` env; if a specific agent throttles, comment it out of the `AGENTS` array and rerun; retry |
| Cost exceeds budget mid-pilot | 3-agent × 10-trial default too much | Cancel with `tmux attach -t diffec-pilot`, `Ctrl-C`; user can then rerun with `N_ATTEMPTS=3` |

## Appendix — What NOT to do

- Do NOT modify `case_gen/configs/case_*.yaml`, `tasks/.../tests/`,
  `tasks/.../solution/`, or any oracle/verifier code without user
  approval. The case designs are calibrated and locked.
- Do NOT push to `upstream` (BattModels/DiffEC). All pushes go to
  `origin` (your fork).
- Do NOT commit `scripts/pilot/.env`, `jobs/`, or `_local_jobs/`
  (already `.gitignore`d).
- Do NOT copy `~/.codex/auth.json` or `~/.gemini/oauth_creds.json`
  into the repo. The Harbor adapters *upload* them from `$HOME`
  directly; there is no need to stage them anywhere else. Leaving
  auth files inside the repo tree risks committing bearer tokens.
- Do NOT run the full pilot without explicit user go-ahead. It
  may cost real money (paid API keys) or burn subscription quota.
