# Frontier-agent pilot — direct-CLI mock exam

> **Status:** in progress, mock-CLI variant (Artemis). Full Harbor-driven
> pilot from `docs/laptop-runbook.md` Part 2 is a separate follow-up on
> the laptop. This document records what we ran on Artemis using the
> `claude` and `codex` CLIs directly against a staged sandbox.

## Why this exists

Per ADR-0008 and CLAUDE.md Definition-of-Done item #6, before opening
the PR we need evidence that frontier agents solve at 10–20 % rate on
the current case bundles. If < 5 % or > 30 %, we re-scope.

The authoritative pilot is the Harbor-driven 3-agent × 10-trial run
in `scripts/pilot/run_pilot.sh`, which requires a Docker daemon.
Artemis has no Docker (see `docs/hpc/artemis.md`), so instead of
blocking on the laptop, we ran a **direct-CLI mock exam** using the
same CLIs Harbor's adapters invoke internally (`claude` and `codex`),
against a sandbox that mirrors the container's file surface.

The mock exam gives us solve-rate signal now; the Harbor pilot on
the laptop remains the last-mile verification before the PR.

## Isolation model (mock exam)

Harbor's `environment_mode = "separate"` in `task.toml` guarantees
the agent container sees only `environment/`. Our mock reproduces
that by staging at `/tmp/mock_pilot_YYYY-MM-DD/` with **only**:

- `instruction.md` (paths rewritten from `/root/…` to
  `/tmp/mock_pilot_.../`)
- `data/cases/case_{1..4}/{data.h5, params.json, formalism.md}`
- `venv/` — `uv venv` with `jax[cpu]`, `jaxopt`, `scipy`, `numpy`,
  `h5py` pre-installed (float64 enabled). No internet at runtime.
- Per-agent `trial_*_run/` and `results_*/` output dirs (only the
  agent's own dirs are relevant per the prompt).

The agent prompt (`agent_prompt.md`) explicitly forbids:
- Reads outside `/tmp/mock_pilot_.../`.
- Reads in other trials' or agents' `trial_*_run/` / `results_*/`.
- Internet access.
- Reads of `~/.claude`, `~/.codex`, `~/.config`, or any global config.

Post-run isolation attestation: for each transcript we grep for
`{/nfs/, /home/changwex, coe-venkvis, DiffEC, CLAUDE.md, AGENTS.md,
oracle_truth, reference_solver, solution/, case_gen, docs/proposal,
docs/plan}`. Any hit is either the echoed prompt itself (containing
those tokens in the "do not read" list) or a genuine leak attempt.
All runs to date are clean.

**Weaker than Harbor's container-level isolation** — a determined
adversarial agent could disobey the prompt and read the parent
directory (`/nfs/turbo/…/DiffEC/`). None have. The full Harbor
pilot on the laptop will remove this residual risk.

## Auth (both providers subscription, $0)

- **Anthropic** (`claude` CLI): `env -u ANTHROPIC_API_KEY` before invoking
  claude, so the CLI falls back to `~/.claude/.credentials.json`
  (Max/Pro OAuth). This is a real fix — the parent Claude Code
  session's inherited `ANTHROPIC_API_KEY` was claiming precedence
  and forcing claude onto a depleted API-key account.
- **OpenAI** (`codex` CLI): `~/.codex/auth.json` (ChatGPT Plus/Pro
  OAuth) unmodified.

Note: `scripts/pilot/run_pilot.sh` should adopt the same
`env -u ANTHROPIC_API_KEY` prefix when we do the laptop pilot.

## Runs

### v1 — 2-arm smoke (2026-07-02)

Confirmed the mock-exam loop works end-to-end.

| Arm | Model | Wall | Verifier | Notes |
|---|---|---|---|---|
| `claude_run` | claude default (unspecified) | 28 min | **10/28** | Regime labels all `NE_deviates` (missed the wrong-sign signal); self-consistency 1/4 |
| `codex_run` | gpt-5.5 default | 14 min | **13/28** | Self-consistency 4/4 (its (D, t⁺⁰) fit reproduces v_data via oracle PDE) |

### v2 — 4-arm capability sweep (2026-07-02)

Anthropic best + good and OpenAI best + good.

| Arm | Model | Wall | Verifier | Notes |
|---|---|---|---|---|
| `opus_run` | `--model opus` | 41 min | **21/28** | Best single-attempt score across everything |
| `sonnet_run` | `--model sonnet` | 87 min | 9/28 | Anomalous — worse than default claude at 3× the wall time |
| `gpt55_run` | `-m gpt-5.5` | 5 min | 13/28 | Deterministic — same 13/28 as v1 |
| `gpt5_run` | `-m gpt-5` | **12 s (failed)** | — | Codex 400: `"The 'gpt-5' model is not supported when using Codex with a ChatGPT account."` Subscription tier doesn't expose gpt-5. |

**v2 per-check breakdown:**

```
test                        opus       sonnet     gpt-5.5
schema                       4/4        4/4        4/4
D                            2/4        0/4        1/4
tp0                          3/4        0/4        0/4
regime                       3/4        0/4        0/4
velocity_rmse                4/4        4/4        4/4
flux_decomposition           1/4        0/4        0/4
self_consistency             4/4        1/4        4/4
─────────────────────────────────────────────────────────
TOTAL / 28                 21/28      9/28       13/28
```

### v5 — Opus × 4 sequential to reach N=10 (2026-07-06)

Sequential single-arm run through 4 more Opus trials in the same
`/tmp/mock_pilot_2026-07-06/` sandbox (post-§3.3.1). Sequential
because v3's parallel batching visibly hurt results.

| Arm | Wall | Exit | Outputs | Verifier | Notes |
|---|---|---|---|---|---|
| `trial_opus_2` | 26 min | 0 clean | 4/4 | **19/28** | 2/4 flux, 3/4 regime, 4/4 self-consistency |
| `trial_opus_3` | 52 min | 0 clean | 4/4 | 9/28 | 0/4 flux, 0/4 regime, 1/4 self-consistency — worst of the clean arms |
| `trial_opus_4` | 118 min | 0 clean | 4/4 | 14/28 | 0/4 flux, 3/4 regime — regime OK but flux regressed |
| `trial_opus_5` | 127 min | 1 (rate-limit) | 0/4 | 0/28 | Hit Max wall at launch: `"out of extra usage · resets 9:40pm (America/Detroit)"`. Different reset window than v3's 4:20am — Max appears to enforce a rolling cap that drained mid-sequence after ~4.5 h cumulative activity. |

**v4 + v5 per-check (5 arms attempted, 4 valid — opus_5 rate-limited):**

```
test                v4_opus_1 v5_opus_2 v5_opus_3 v5_opus_4 v5_opus_5
schema                  4/4       4/4       4/4       4/4       0/4
D                       2/4       1/4       0/4       1/4       0/4
tp0                     3/4       1/4       0/4       1/4       0/4
regime                  3/4       3/4       0/4       3/4       0/4
velocity_rmse           4/4       4/4       4/4       4/4       0/4
flux_decomposition      2/4       2/4       0/4       0/4       0/4   ← stochastic
self_consistency        4/4       4/4       1/4       1/4       0/4
────────────────────────────────────────────────────────────────────
TOTAL / 28             22/28    19/28     9/28     14/28      0/28
```

**§3.3.1 lift is real but stochastic.** 2 of 4 valid post-§3.3.1
arms (v4_opus_1, v5_opus_2) got `flux_decomposition` 2/4 —
predictably cases 2 and 4. The other 2 valid arms (v5_opus_3,
v5_opus_4) got 0/4. Session logs indicate the 2/4 arms explicitly
referenced §3.3.1 while the 0/4 arms fell back to their own
scratch computation. The worked example helps agents that engage
with it, but doesn't force engagement.

**Isolation:** all 4 arms clean on transcript grep.

Archived at `_local_jobs/mock_pilot_2026-07-06/` (3.5 MB now covers
v4 + v5).

### v4 — Opus × 1 after §3.3.1 worked example added (2026-07-06)

Single-arm rerun to test whether adding a worked flux-decomposition
example to `formalism.md` (commit `9acb775`) lifts Opus's
`test_flux_decomposition` pass rate above 0/4.

| Arm | Model | Wall | Exit | Outputs | Verifier | Notes |
|---|---|---|---|---|---|---|
| `trial_opus_1` | `--model opus` | 29 min | 0 clean | 4/4 | **22/28** | New best. `test_flux_decomposition` 2/4 (up from 0/4 v3 mean). |

**v4 per-check (compared to v2 Opus and v3 mean):**

```
test                        v4     v2 Opus    v3 mean
schema                     4/4       4/4       3.4/4
D                          2/4       2/4       0.4/4
tp0                        3/4       3/4       0.6/4
regime                     3/4       3/4       1.8/4
velocity_rmse              4/4       4/4       3.4/4
flux_decomposition        2/4       1/4       0.0/4   ← the intended lift
self_consistency          4/4       4/4       1.0/4
─────────────────────────────────────────────────────
TOTAL / 28                22/28    21/28      10.6/28
```

**Which flux samples now pass:** case_2 and case_4 pass check #5;
case_1 and case_3 still fail. The lift is real but partial —
worked example helps agents get the units right when they engage,
but doesn't force it.

**Session-log evidence the fix landed.** Opus explicitly cited
"the exact formulas from §3.3" when describing its flux
decomposition step. Its case_4 t⁺⁰ ≈ −0.18 matches the oracle
truth (−0.18 to −0.20 across c_grid[2.93, 3.20]); its case_3
t⁺⁰ ≈ −0.37 is closer to truth (~−0.20) than any v3 arm (all of
which reported t⁺⁰ ≈ −0.8).

**Isolation:** clean transcript grep (no repo mentions).

Archived to `_local_jobs/mock_pilot_2026-07-06/` (568 KB).

### v3 — Opus × 5 + o3 (2026-07-05)

5-trial mini-pilot on the best-performing arm (Opus) to estimate the
Anthropic-frontier solve rate + retry OpenAI-good with `o3` after
gpt-5 was rejected.

Finished on 2026-07-06. Anthropic Max hit its usage limit during the
run — 3/5 Opus arms received an `"out of extra usage · resets 4:20am
America/Detroit"` message. Two of those arms (`opus_1`, `opus_2`) had
already written all 4 output files before the CLI errored on its
final message and are usable for grading. `opus_4` was cut off with
only 1/4 outputs and is unusable. `opus_3` and `opus_5` completed
cleanly.

| Arm | Model | Wall | Exit | Outputs | Verifier | Notes |
|---|---|---|---|---|---|---|
| `trial_opus_1` | `--model opus` | 34 min | 1 (rate-limited) | 4/4 | **14/28** | Rate-limit at end but wrote outputs; 4/4 regime! (only arm to nail regime) |
| `trial_opus_2` | `--model opus` | 42 min | 1 (rate-limited) | 4/4 | **14/28** | Rate-limit at end but wrote outputs |
| `trial_opus_3` | `--model opus` | 29 min | 0 clean | 4/4 | 9/28 | Clean; regime 0/4 (all wrong) |
| `trial_opus_4` | `--model opus` | 36 min | 1 (rate-limited) | 1/4 | 3/28 | Partial; only case_1 written before rate-limit |
| `trial_opus_5` | `--model opus` | 22 min | 0 clean | 4/4 | 13/28 | Clean, fastest; all cases labeled `NE_wrong_sign` (wrong for case_1, case_2) |
| `trial_o3_1` | `-m o3` | **44 s (failed)** | 1 | 0/4 | — | Same 400 as gpt-5 |

**OpenAI subscription-tier constraint (confirmed 2026-07-05):** the
user's ChatGPT plan exposes only `gpt-5.5` via Codex CLI. Probed
`gpt-5`, `gpt-5-mini`, `o3`, `o4-mini` — all return the same 400
`"model is not supported when using Codex with a ChatGPT account"`.
Adding a second OpenAI arm (per the original v3 design) would require
a paid OpenAI API-key path with billing. Deferred — v2's `gpt-5.5`
13/28 result stands as the OpenAI baseline.

**v3 per-check breakdown (Opus × 5):**

```
test                    opus_1   opus_2   opus_3   opus_4   opus_5
schema                    4/4      4/4      4/4      1/4      4/4
D                         0/4      1/4      0/4      0/4      1/4
tp0                       1/4      1/4      0/4      0/4      1/4
regime                    4/4      3/4      0/4      0/4      2/4
velocity_rmse             4/4      4/4      4/4      1/4      4/4
flux_decomposition        0/4      0/4      0/4      0/4      0/4
self_consistency          1/4      1/4      1/4      1/4      1/4
─────────────────────────────────────────────────────────────────
TOTAL / 28              14/28    14/28     9/28     3/28    13/28
```

**Isolation:** all 5 Opus arms clean on the transcript grep sweep
(no references to `coe-venkvis`, `DiffEC`, `CLAUDE.md`, `AGENTS.md`,
`oracle_truth`, `reference_solver`, `case_gen`, `docs/proposal`, or
`docs/plan`). Cross-trial reads (`results_opus_M/` from `opus_N`)
also absent.

## Solve-rate estimates

Combining v3 (N=5) with v2's Opus run gives 6 independent Opus
trials against these case bundles.

| Sample | Solves | Point | Wilson 95 % CI | Mean check-pass rate |
|---|---|---|---|---|
| v3 all 5 (incl. partial) | 0/5 | 0 % | 0.0 – 43.4 % | 37.9 % |
| v3 clean 4 (excl. opus_4 partial) | 0/4 | 0 % | 0.0 – 49.0 % | 44.6 % |
| v2 + v3 combined (6 Opus) | 0/6 | 0 % | 0.0 – 39.0 % | 44.0 % |
| v2 + v3 + v4 (7 Opus) | 0/7 | 0 % | 0.0 – 35.4 % | 47.4 % |
| v2 + v3 + v4 + v5 (11 Opus) | 0/11 | 0 % | 0.0 – 25.9 % | 45.5 % |
| Above, **excluding rate-limited** (9 Opus) | 0/9 | 0 % | 0.0 – 29.9 % | 53.6 % |
| **Post-§3.3.1 clean subset** (4 Opus: v4_opus_1 + v5_opus_{2,3,4}) | 0/4 | 0 % | 0.0 – 49.0 % | 57.1 % |

**Reading the numbers.** Solve rate is 0/N across all Opus samples;
Wilson upper bound at N=6 is ~39 %, which is consistent with
either the ADR-0008 10–20 % target band or "too hard". The mean
check-pass rate (~44 %) is well above the ~35 % floor implied by
schema+velocity passing "for free", showing Opus does make real
progress on the physics — just not to the 28/28 all-green bar.

At N=6 we cannot distinguish "in-band difficulty (10–20 % solve)"
from "too hard (< 5 %)" with confidence. To tighten:
- Full Harbor pilot on the laptop (10 trials × Opus alone would
  narrow the Wilson CI to ~ ±20 pp at 0/10).
- Or accept the current signal and lean on the check-level
  narrative — "Opus scores 44 % of individual checks but never
  clears the 28/28 bar" is defensible.

## Concurrency caveat

**Anthropic Max concurrency-throttling degrades results.** Running 5
Opus arms in parallel produced:
- Wall times 22–42 min vs v2's single-arm 41 min (some faster, some
  slower — likely uneven request scheduling).
- 3/5 arms hit `"out of extra usage"` before finishing the transcript.
- v2's clean-single-arm 21/28 remains the best-observed Opus score
  by a wide margin.

If we want a defensible solve-rate estimate for TB-Science, the
right move is single-arm-at-a-time on the laptop pilot, not parallel
batching against a Max subscription.

## Regime observations

Across the 5 v3 Opus trials, `regime` was highly variable:

| Arm | case_1 (NE_valid) | case_2 (NE_deviates) | case_3 (NE_wrong_sign) | case_4 (NE_wrong_sign) |
|---|---|---|---|---|
| opus_1 | ✓ | ✓ | ✓ | ✓ (**4/4 perfect**) |
| opus_2 | ✓ | ✗ | ✓ | ✓ |
| opus_3 | ✗ | ✗ | ✗ | ✗ (**all wrong**) |
| opus_5 | ✗ (called NE_wrong_sign) | ✗ (called NE_wrong_sign) | ✓ | ✓ |

The best regime scorer (opus_1) got 4/4 despite being rate-limited.
Regime is orthogonal to the tolerance-tight numerical checks — the
model can nail sign structure without matching absolute values.

Even opus_1's 14/28 total was pulled down by the tolerance-tight
D and tp0 checks (0/4 D, 1/4 tp0), which look nearly insensitive
to model quality across the 5 trials.

## Observations

**The task discriminates model capability sharply.**
v2 spread: Opus 21/28 vs Sonnet 9/28 vs gpt-5.5 13/28 vs claude
default 10/28. That's a wider capability spread than a random
task would produce.

**Check-level failure modes we've seen consistently:**

- `test_schema` — universally passed. The schema in `instruction.md`
  is unambiguous.
- `test_velocity_rmse` — universally passed. The 15% tolerance
  is generous; even a middling `(D, t⁺⁰)` fit reproduces `v_data`.
- `test_self_consistency` — Opus and gpt-5.5 pass 4/4; Sonnet
  and claude-default pass 1/4. This is a strong signal of whether
  the agent's inverse solution is internally physical.
- `test_D`, `test_tp0` — the tolerance-tight checks (10% relative
  and 0.05 absolute). Fail most often at deep concentrations
  (case_3 and case_4). Opus is the only arm to pass these on any
  case.
- `test_regime` — the discriminator. Only Opus's 3/4 result and
  gpt-5.5's near-misses on cases 3/4 show the sign-flip is
  recoverable from the data.
- `test_flux_decomposition` — hardest for everyone. Even Opus
  only got 1/4. Suggests convention differences vs the oracle's
  exact formula; worth flagging as a place `formalism.md` may
  benefit from a worked numeric example.

**Sonnet was anomalously slow (87 min) AND worse.** Possibly it
chose a bad numerical basin and thrashed on tool calls. Full
transcript at `_local_jobs/mock_exam_2026-07-02/sonnet_run/` for
postmortem.

## Reproducibility

To rerun the mock exam on a fresh /tmp:

```bash
# 1. Stage sandbox (adjust date)
MOCK=/tmp/mock_pilot_YYYY-MM-DD
mkdir -p $MOCK/data/cases
TASK=tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport
for c in case_{1..4}; do cp -r $TASK/environment/data/cases/$c $MOCK/data/cases/; done
sed "s|/root/|$MOCK/|g" $TASK/instruction.md > $MOCK/instruction.md

# 2. Preinstall physics libs (float64)
uv venv $MOCK/venv --python 3.11 --quiet
uv pip install --python $MOCK/venv/bin/python --quiet "jax[cpu]" jaxopt scipy numpy h5py

# 3. Write agent_prompt.md (see this doc §"Isolation model" — the
#    prompt lists strict sandbox rules and points at $MOCK/venv/bin/python).

# 4. Per arm: launch.sh runs the CLI headless with subscription auth.
#    Anthropic: env -u ANTHROPIC_API_KEY timeout 7200 claude -p <prompt> --model opus --permission-mode bypassPermissions --add-dir $MOCK
#    OpenAI:    timeout 7200 codex exec <prompt> -m <model> -C $MOCK --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check

# 5. Grade: TRUTH_DIR=$PWD/$TASK/tests/oracle_truth
#          RESULTS_DIR=$MOCK/results_<arm> uv run pytest $TASK/tests/test_outputs.py -v
```

Full raw transcripts + outputs preserved at
`_local_jobs/mock_exam_2026-07-02/` and `_local_jobs/mock_pilot_2026-07-05/`
(both gitignored).

## Caveats + open items

1. **Direct-CLI ≠ Harbor.** Harbor bundles a prompt shim, its own
   tool-use scaffolding differences, and its own environment
   container Dockerfile with `RUN uv pip install …`. The mock exam
   approximates but is not identical. The final Harbor pilot on the
   laptop (`docs/laptop-runbook.md` Part 2) is the last-mile check.
2. **Anthropic Max concurrency-throttling is real.** 5 concurrent
   Opus arms in v3 all completed, but 3/5 hit the "out of extra
   usage" message before finishing cleanly. The clean single-Opus
   run in v2 (21/28) stayed above all v3 arms. Any future N > 3
   Opus batch should be sequential, not parallel, on the same
   subscription.
3. **N still small.** 6 total Opus trials (v2 + v3) gives Wilson
   95% CI 0–39%. Cannot yet distinguish "in-band" (10–20%) from
   "too hard" (< 5%). A single-arm N=10 laptop pilot would give
   ~ ±20 pp precision.
4. **Sonnet and claude-default were anomalously weak** (v2). Worth
   pulling transcripts to see whether tool-use scaffolding is the
   issue or if the model itself struggles with the physics.
5. **OpenAI second arm needs paid API-key path.** ChatGPT
   subscription exposes only `gpt-5.5` via Codex CLI (probed
   gpt-5, gpt-5-mini, o3, o4-mini — all 400).
6. **Flux decomposition was uniformly hard through v3** (0/4 across
   all 5 v3 Opus arms, 1/4 for v2 opus at best). Worked example
   added to `formalism.md` §3.3.1 in commit `9acb775`. **v4
   confirmed the fix landed**: single Opus arm scored
   `test_flux_decomposition` 2/4 (case_2 and case_4 passing), and
   overall score of 22/28 — a new best. Case_1 and case_3 still
   fail check #5, so the lift is real but partial; agents that
   engage with §3.3.1 get units right on some samples but the
   worked example doesn't force it universally.

## Next steps

- **N = 11 Opus attempts (9 usable) gives 0 solves with Wilson 95%
  CI 0-30%.** Still cannot exclude "in-band" (10-20%) vs "too hard"
  (< 5%). Two ways forward: (a) accept 0-30% CI as evidence the task
  is real and discriminating without hitting solve — mean check-pass
  rate 53.6% is a meaningful narrative for reviewers; or
  (b) run 10 more sequential Opus trials on the laptop (paid API-key
  path to skip Max quota) to tighten CI to ~ ±15 pp.
- **Update CLAUDE.md DoD item #6 status** to note this mock-CLI pilot
  as an interim step; the Harbor laptop pilot remains the
  authoritative check.
- **Start writing the PR body's "empirical solve-rate validation"
  section.** Current evidence supports "the task is real, discriminates
  frontier-model capability, but no arm hit 28/28 in 8+ attempts
  (v1+v2+v3 combined)."
- Optionally probe **whether relaxing `test_flux_decomposition`
  tolerance from 15% to 20%** would move Opus toward the target
  band — this is the only tolerance-tight check that's uniformly
  0/4 for Opus.
- Run the full Harbor pilot on the laptop
  (`docs/laptop-runbook.md` Part 2) as the last-mile verification
  before the PR.
