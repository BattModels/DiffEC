# Frontier-agent pilot driver (ADR-0008)

Runs the 3 frontier agents (Claude Opus 4.7, GPT-5, Gemini 2.5) against
the bundled task and aggregates the solve rate + per-check failure
modes. Per ADR-0008, the proposal commits to a 10–20 % aggregate solve
rate; if the pilot lands < 5 % or > 30 % the case design should be
re-opened.

This directory holds **scaffolding only** — the actual pilot runs need
Docker (so `harbor` can build agent/verifier containers) and live
API keys for each provider. Run on a Docker-capable host, not on the
HPC node we develop on.

## Pre-flight

- **Docker**: required for `harbor run`. Install via [docker.com](https://docs.docker.com/get-docker/).
- **`harbor` CLI**: `uv tool install harbor` (we tested with 0.16.0).
- **API keys**: at least one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GEMINI_API_KEY` exported, depending on which agents you run.
  Put them in `.env` alongside this README (gitignored) so
  `run_pilot.sh` picks them up via `--env-file`.

Quick sanity:

```bash
harbor --version          # >= 0.16.0
docker --version          # any recent
echo $ANTHROPIC_API_KEY   # not empty
```

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

Roughly:
- Claude Opus 4.7: $10–25/trial → $100–250 for 10 trials.
- GPT-5: $5–15/trial → $50–150.
- Gemini 2.5: $2–8/trial → $20–80.

Plus container compute (Docker on the host machine — free if local,
else cloud spend).

**Total pilot budget: $200–500** depending on which models and how
many attempts. Adjust `N_ATTEMPTS` if budget is tight; the 10–20 %
band is detectable at N ≥ 10 (Wilson 95 % CI ~ ±15 percentage points).

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
