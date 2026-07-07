# TB-Science task contribution — DiffEC author review

**Target:** `BattModels/DiffEC` — `feat/tb-sci-task` → `main`
**Author:** @ChangwenXu98
**Status:** Ready for author review before submitting upstream to
`harbor-framework/terminal-bench-science`.

---

## Summary

This branch adds a Terminal-Bench Science (TB-Science) task
contribution built on the DiffEC method: **"Differentiable modeling
of concentrated-electrolyte mass transport from operando profiles."**

The agent is given four bundled cases of operando-style `c(x, t)`
and `v(x, t)` profiles from a symmetric Li | electrolyte | Li cell,
and must recover the concentration-dependent salt diffusivity
`D(c)` and cation transference number `t⁺⁰(c)` (plus the
Nernst-Einstein-equivalent `t⁺⁰_NE(c)`, regime labels, predicted
velocity field, and a cation-flux decomposition) under Newman's
concentrated-solution theory.

The contribution lives entirely under
`tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/`.
The rest of the branch is dev-only tooling (`case_gen/`, `scripts/`,
`docs/`, `mock_exam/`, `AGENTS.md`, `CLAUDE.md`) that won't reach
the TB-Science upstream PR.

## What we ask reviewers to check

1. **Case design.** Do the 4 cases exercise the failure modes we
   want (NE-valid, NE-deviates, NE-wrong-sign × 2)? Are the
   held-out `D(c)` and `t⁺⁰(c)` parameter functions physically
   defensible while distinct from any published values
   (Steinrück 2020, Pesko 2017, this repo's PEO-LiTFSI fit)?
   → `docs/plan/case-design.md`, `case_gen/configs/case_*.yaml`.
2. **Oracle.** Does the moving-frame finite-volume + JAX solver
   under `tests/oracle/` faithfully generalize the published
   `Mass Transport in Concentrated Electrolytes and Benchmarks/solver.py`?
   → `docs/plan/oracle-spec.md`, `tests/oracle/`.
3. **Verifier tolerances.** Are the 7 checks and their tolerances
   (10 % D, 0.05 t⁺⁰, exact regime, 15 % v_pred, 15 % flux
   components, self-consistency) the right band for the intended
   difficulty? The reference solver clears them with margin;
   frontier agents don't (see pilot section).
   → `docs/plan/verifier-spec.md`, `tests/test_outputs.py`.
4. **Physics spec (`formalism.md`).** §3.3.1 has a worked
   flux-decomposition numeric example added after mock-pilot
   evidence showed unit-handling was the dominant failure mode.
   Does the example clarify the convention without leaking the
   oracle numbers?
   → `tasks/…/environment/data/cases/case_1/formalism.md` (all
   4 cases carry the identical spec; single source in
   `case_gen/writers.py::FORMALISM_MD`).
5. **Anti-cheat perturbation.** Held-out oracle parameters are
   demonstrably ≥ 0.10 apart from Steinrück / DiffEC fit at every
   `c_grid` point (ADR-0004).
   → `scripts/litvalue_distance.py`, `docs/plan/decisions.md`
   §ADR-0004.

## What's in the shipped task subtree

```
tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/
├── instruction.md            # agent prompt
├── task.toml                 # Harbor 0.16 schema_version = "1.0"
├── environment/              # agent container
│   ├── Dockerfile
│   └── data/cases/case_{1..4}/{data.h5, params.json, formalism.md}
├── solution/                 # reference solution (Oracle-agent only)
│   ├── solve.sh
│   ├── reference_solver.py
│   └── helpers
└── tests/                    # verifier container (environment_mode = "separate")
    ├── Dockerfile
    ├── test.sh               # pytest → /logs/verifier/reward.txt
    ├── test_outputs.py
    ├── oracle/               # held-out moving-frame solver + flux + NE inversion
    └── oracle_truth/case_{1..4}/truth.npz
```

## Evidence

**DoD #1 — reference verifier passes 28/28 locally.**
Fresh checkout, `RESULTS_DIR=./_local_results uv run pytest
"$TASK/tests/test_outputs.py" -v --tb=no`:

```
28 passed in 7.84s
```

(7 active checks × 4 cases). Wall time for reference solver on
4–8 CPUs: ~11 minutes total.

**DoD #2 — `harbor run -p "$TASK" -a oracle` returns reward = 1.**
Verified on laptop (Colima + BuildKit) 2026-07-01: `Trials=1,
Exceptions=0, Mean=1.000` in 9m 4s wall time. See
`docs/progress/smoke-oracle-result.md`.

**DoD #6 — frontier-agent pilot evidence (interim, mock-CLI).**
Five rounds of direct-CLI mock pilot on Artemis (no Docker; sandbox
isolation approximated by staged `/tmp` + prompt rules +
transcript-grep audit). Full writeup in `docs/progress/pilot_run.md`.
Headline:

| Sample | Solves | Wilson 95% CI | Mean check-pass |
|---|---|---|---|
| All Opus attempts (N=11) | 0/11 | 0.0 – 25.9 % | 45.5 % |
| Excluding rate-limited (N=9) | 0/9 | 0.0 – 29.9 % | 53.6 % |
| Post-§3.3.1 clean (N=4) | 0/4 | 0.0 – 49.0 % | 57.1 % |
| Best single attempt | **22/28** (Opus, v4) | | |

The task clearly discriminates capability (v3 Opus mean 10.6 vs v4
best 22 — 2× swing on same model, same sandbox, just clearer spec
+ no concurrency contention). Adding a worked flux-decomposition
example to `formalism.md` §3.3.1 lifted `test_flux_decomposition`
pass rate from 0/4 (v3 mean) to 2/4 (v4/v5 subset) — stochastic but
real. No arm hit 28/28 in 9 clean attempts. Consistent with the
proposal's 10-20 % solve-rate target band; cannot yet exclude
"too hard" (< 5 %) at this N.

**Anti-cheat isolation attestation.** All 14 mock-pilot transcripts
clean on grep sweep — no references to `CLAUDE.md`, `oracle_truth`,
`reference_solver`, `case_gen`, `docs/proposal`, `docs/plan`,
`DiffEC`, `coe-venkvis`, or any repo path. Cross-trial reads also
absent.

## Development record

- `docs/proposal/` — accepted proposal package (proposal.md,
  formalism.md, review_llm.md, review_expert.md, decision.md).
  Frozen contract; unchanged since proposal acceptance.
- `docs/plan/` — 7 design docs + 10 ADRs.
- `docs/progress/` — key-facts, smoke-oracle-result, pilot_run.
- `docs/session.md` — running experience notebook (gitignored,
  laptop-local; a fresh Claude Code / Codex session can cold-start
  from `CLAUDE.md` / `AGENTS.md`).
- `case_gen/` — deterministic config-driven case generator. Each
  case is fully reproduced by `(case_config.yaml, seed)`; the
  pre-PR audit (`scripts/pre_pr_audit.sh`) runs 14 checks and
  currently returns **AUDIT PASSED**.

## Deferred / not done

1. **Harbor-driven pilot on the laptop** (`docs/laptop-runbook.md`
   Part 2). Planned but deferred: pending API reimbursement and
   the tradeoff between the mock-CLI evidence (already gathered)
   and burning ~ $200-500 on paid API keys for a 3-agent × 10-trial
   Harbor pilot. `scripts/pilot/*` is scaffolded and ready to run.
2. **Tolerance-relaxation decision.** v4 best 22/28 is 6 checks
   from a solve — relaxing D 10 %→12 % or t⁺⁰ 0.05→0.07 would tip
   that one arm into solve. Deferred pending real Harbor pilot
   data; current evidence (mock-CLI) is arguably not strong enough
   to justify changing case design.
3. **TB-Science upstream PR body.** Once BattModels authors sign
   off on this branch, the actual submission to
   `harbor-framework/terminal-bench-science` will follow the
   template in `docs/pre-pr-checklist.md`.

## Related

- Proposal thread: `docs/proposal/decision.md` (accepted 2026-05).
- Original paper repo (`DiffEC` core code): unchanged; this branch
  only *uses* the moving-frame formalism, doesn't modify the core.
- Related discussion: TB-Science
  [discussion #335](https://github.com/harbor-framework/terminal-bench-science/discussions/335).

## Housekeeping notes for review

- **Branch state:** `feat/tb-sci-task` HEAD = `0fb1cd2` (revert of
  PILOT-ONLY docker_image lines) — this is what upstream CI sees.
  Prior commits `e12cff9`, `731f765`, `c751b45` in the branch
  history added and later reverted GHCR image pointers used during
  Artemis Singularity probes; the final HEAD has none of them.
- **Anti-cheat:** `tests/oracle_truth/case_*/truth.npz` and
  `case_gen/configs/*.yaml` (with hidden parameter functions) live
  in the branch but never reach the agent's container by design
  (Harbor `environment_mode = "separate"`; the agent image is
  built from `environment/Dockerfile` only). The mock pilot
  confirmed this isolation held under the direct-CLI substitute
  as well.
- **File count:** `git diff --stat main...feat/tb-sci-task` totals
  the added task plus the dev tooling. The task subtree itself is
  well under the 100 MB upstream limit; largest file is
  `data.h5` (~ 85 KB per case).

## Commit graph (recent tip)

```
0fb1cd2 task.toml: revert PILOT-ONLY docker_image lines for upstream PR
4466dd0 CLAUDE.md DoD #6: note mock pilot as interim, Harbor still authoritative
78352f5 pilot_run.md: v5 adds 4 sequential Opus, reaches N=10 (9 usable)
de68f2b pilot_run.md: v4 confirms formalism.md §3.3.1 lifts flux-decomp pass rate
7e4cbc8 pilot_run.md: link formalism.md worked example to its motivating finding
9acb775 formalism.md: add a worked flux-decomposition example
ecd5a00 Populate pilot_run.md with real 3-round mock-CLI pilot data
294891a Add mock_exam/ — pre-submission cold-start test for frontier agents
6bee447 Pilot README + laptop-runbook: subscription-auth path + AGENTS.md
997300b Rename case_4: multi-modal basin trap -> NE-wrong-sign at high c
```

Full history: `git log main..feat/tb-sci-task --oneline`.

## Test plan for reviewers

```bash
# 1. Clone and check out the branch
git checkout feat/tb-sci-task
uv sync

# 2. Regenerate all 4 cases from configs (deterministic, ~5 min)
uv run python -m case_gen.generate --all

# 3. Verifier against the reference solution (must be 28 passed)
TASK=tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport
uv run python "$TASK/solution/reference_solver.py" \
    --cases "$TASK/environment/data/cases" --out ./_local_results
RESULTS_DIR=./_local_results TRUTH_DIR="$PWD/$TASK/tests/oracle_truth" \
    uv run pytest "$TASK/tests/test_outputs.py" -v

# 4. Pre-PR audit (14 checks; last one only trips on unstaged mods)
bash scripts/pre_pr_audit.sh

# 5. (Optional, on a Docker host) Harbor oracle smoke test
harbor run -p "$TASK" -a oracle           # expect reward = 1
```
