# Frontier-agent pilot run (ADR-0008)

> _Placeholder._ This file gets overwritten by
> `scripts/pilot/aggregate.py` after the pilot runs on a Docker host.
> See `scripts/pilot/README.md` for the run procedure.

## Why

Per ADR-0008, before opening the PR we need to confirm the task lands
in the proposal's **10–20 % aggregate solve rate** band across the 3
frontier agents:

- Claude Opus 4.7 (`claude-code` + `claude-opus-4-7`)
- GPT-5 (`codex` + `gpt-5`)
- Gemini 2.5 (`gemini-cli` + `gemini-2.5-pro`)

If aggregate solve rate is < 5 % or > 30 %, the case design needs to
be re-opened.

## How to run (on a Docker host)

```bash
# 1. Install harbor + ensure docker daemon is running.
uv tool install harbor             # we tested 0.16.0
docker --version                    # any recent

# 2. Put API keys in scripts/pilot/.env (gitignored):
cat > scripts/pilot/.env <<EOF
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
EOF

# 3. Run the pilot (default: 10 trials per agent, 2 concurrent).
bash scripts/pilot/run_pilot.sh

# 4. Aggregate.
uv run python scripts/pilot/aggregate.py \
    --jobs jobs --out docs/progress/pilot_run.md
```

Total wall time: 30 trials × ~30 min/trial / 2 concurrent ≈ 8 hours.
Budget: $200–500 depending on which models hit the 1-hour cap.

## Open questions before locking the case design

- **If aggregate < 5 %**: cases are too hard. Most likely culprit is
  case_3 / case_4's negative-tp⁺⁰ regime — most frontier agents may
  default to a lab-frame-style positive parameterization. Consider
  adding a hint in `instruction.md` that t⁺⁰ may be negative.
- **If aggregate > 30 %**: cases are too easy. Most likely culprit is
  case_1 (NE-valid) being passable by lab-frame agents — that's by
  design but lifts the aggregate. Consider weighting the discriminator
  cases (2, 3, 4) more heavily, or making case_1 stricter.

## Results

_Will be populated by `aggregate.py` after the pilot completes._
