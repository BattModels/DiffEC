# Frontier-agent pilot driver (ADR-0008)

> **For the full end-to-end procedure** (laptop image build → Artemis
> singularity pilot) see `docs/pre-pr-runbook.md`. This file is
> per-script documentation only.


Runs the 3 frontier agents (Claude Opus 4.7, GPT-5, Gemini 2.5) against
the bundled task and aggregates the solve rate + per-check failure
modes. Per ADR-0008, the proposal commits to a 10–20 % aggregate solve
rate; if the pilot lands < 5 % or > 30 % the case design should be
re-opened.

This directory holds **scaffolding only** — the actual pilot runs need
Docker (so `harbor` can build agent/verifier containers) and one of
two auth modes per provider (subscription OR paid API key). Run on a
Docker-capable host, not on the HPC node we develop on.

## Pre-flight

**Auth mode (pick one — subscription is cheaper).**

The three agents (`claude-code`, `codex`, `gemini-cli`) each accept
either paid API keys OR the user's existing subscription auth
(Claude Max/Pro, ChatGPT Plus/Pro, Google Gemini free tier). We
audited the three Harbor 0.16 adapters (`harbor/agents/installed/{claude_code,codex,gemini_cli}.py`)
and they all support both paths. Subscription mode tells each adapter
to *copy* a locally-saved auth file / OAuth token into the sandbox
container.

Per-adapter contract:

| Adapter | Subscription env vars | Paid env vars | Prereq on host |
| --- | --- | --- | --- |
| `claude-code` | `CLAUDE_CODE_OAUTH_TOKEN` + `CLAUDE_FORCE_OAUTH=1` | `ANTHROPIC_API_KEY` | Run `claude setup-token` once, paste output into `.env` |
| `codex` | `CODEX_FORCE_AUTH_JSON=1` | `OPENAI_API_KEY` | Run `codex login` once so `~/.codex/auth.json` exists |
| `gemini-cli` | `GEMINI_FORCE_OAUTH=1` | `GEMINI_API_KEY` | Run `gemini` once + "Login with Google" so `~/.gemini/oauth_creds.json` exists |

**Subscription-auth `.env` (recommended, ~$0):**

```
CLAUDE_CODE_OAUTH_TOKEN=<from `claude setup-token`>
CLAUDE_FORCE_OAUTH=1
CODEX_FORCE_AUTH_JSON=1
GEMINI_FORCE_OAUTH=1
```

**Paid-API-key `.env` (fallback, ~$200–500 for a 10-trial pilot):**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

You can mix modes per-agent (e.g., Claude via subscription, Codex via
paid key). Each adapter resolves its highest-priority available auth.

**Other prerequisites:**

- **Docker**: required for `harbor run`. Install via [docker.com](https://docs.docker.com/get-docker/).
- **`harbor` CLI**: `uv tool install harbor` (we tested with 0.16.0).
- **`.env` file mode**: `chmod 600 scripts/pilot/.env` — it holds bearer tokens.

Quick sanity:

```bash
harbor --version                          # >= 0.16.0
docker --version                          # any recent
[ -f scripts/pilot/.env ] && \
  grep -Ec '^(CLAUDE_|CODEX_|GEMINI_|ANTHROPIC_|OPENAI_)' scripts/pilot/.env   # >= 3
```

For the full end-to-end laptop procedure (Docker plugin install,
Colima sizing, tmux launch, aggregate + commit), see
`docs/laptop-runbook.md` Part 2. That file is the authoritative
runbook; this README is the per-script reference.

Before this pilot, do the **mock exam** (`mock_exam/README.md`) —
single-trial-per-agent sanity check with a visible agent-view
inspector. That confirms Harbor's container isolation actually
keeps our reference solver / oracle / docs / CLAUDE.md out of the
agent's view, and that auth + container-build works end-to-end.
Only run this pilot after the mock trials complete without crashes.

## Run the pilot

```bash
# From repo root:
bash scripts/pilot/run_pilot.sh
```

The script invokes `harbor run` once per agent with `--n-attempts $N_ATTEMPTS`
(default 10). Each attempt is a full trial (agent attempts all 4 cases;
verifier scores; produces one reward = 0 or 1). Outputs land in
`jobs/pilot-<agent>__<timestamp>/`.

Tune by editing the script header:

| Variable | Default | What it does |
| --- | --- | --- |
| `N_ATTEMPTS` | 10 | trials per agent (more = tighter solve-rate CI, more $) |
| `N_CONCURRENT` | 2 | concurrent trials (caps API rate, RAM) |
| `AGENT_TIMEOUT_MULT` | 1.0 | `task.toml` agent.timeout_sec multiplier |
| `JOBS_DIR` | `jobs/` | output dir |
| Agents in `AGENTS` array | all 3 | comment out to skip an agent |

## Aggregate results

```bash
uv run python scripts/pilot/aggregate.py --jobs jobs --out docs/progress/pilot_run.md
```

Reads every `jobs/pilot-*/` directory, parses per-trial reward.txt and
pytest-ctrf.json (downloaded as artifacts), computes:

- **Solve rate per agent** (fraction of trials with reward = 1).
- **Per-check failure breakdown** (which of the 28 verifier checks
  fails most often per agent).
- **Wall time per trial** (median, p90).

Writes the result table into `docs/progress/pilot_run.md`. Manually
review and add narrative notes there before committing.

## Cost estimate (rough)

Per trial: ~1 hour of agent execution (per `task.toml`
`agent.timeout_sec = 3600`). For 10 trials × 3 agents = 30 trials =
~30 agent-hours.

**Subscription auth (recommended):** ~$0 marginal for our volume if
you already have Claude Max ($100/mo) + Gemini free tier ($0)
+ optionally ChatGPT Plus ($20/mo) or Pro ($200/mo). Watch for
rate-limit throttling on Plus during the 30-trial run — drop
`N_CONCURRENT` to 1 or run agents sequentially if Codex throttles.

**Paid API keys:** ~$200–500 for a 10-trial pilot:
- Claude Opus 4.7: $10–25/trial → $100–250 for 10 trials.
- GPT-5: $5–15/trial → $50–150.
- Gemini 2.5: $2–8/trial → $20–80.

Plus container compute (Docker on the host machine — free if local,
else cloud spend). The 10–20 % target band is detectable at N ≥ 10
(Wilson 95 % CI ~ ±15 pp); reduce `N_ATTEMPTS` for a first-pass sanity.

## Notes

- The `oracle` agent (Harbor's built-in reference) is NOT in this
  pilot — it's used for ADR-DoD item #2 (`harbor run -p $TASK -a oracle`)
  separately. Oracle uses our `solution/solve.sh` and should return
  reward = 1 deterministically.
- The pilot does NOT exercise multi-step rewards or per-step
  artifacts — our task is single-step (one verifier run after all
  4 cases). `n_attempts` controls trial repetition; each trial is
  independent.
- If a single agent fails catastrophically (e.g., always crashes during
  setup), `aggregate.py` flags it separately from "completed but
  failed verifier".
- **Record auth mode in the writeup.** When editing
  `docs/progress/pilot_run.md` after aggregation, note which agent
  used which mode (subscription vs paid API). TB-Science reviewers
  may correlate solve rate with rate-limit or session-timeout
  artifacts, so we should not obscure this.
