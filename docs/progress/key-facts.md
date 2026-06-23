# Key Facts & Gotchas

> Non-obvious traps. Add to this file whenever you spend more than 30 minutes
> figuring something out, or whenever a code reviewer would have to re-derive
> something from scratch.

## Physics & conventions

### Moving frame vs lab frame is *the* discriminator
The whole task is built around this distinction. Concretely:

- The proposal's eqs 7-8 are written in the moving electrolyte frame
  (co-moving with the deposition electrode at `x = 0`).
- A lab-frame Nernst-Planck implementation drops the `c v₀` term in the
  cation flux and sets `v₀ ≡ 0`. It will still fit `c_data` plausibly
  — but it will predict the wrong `v_data` and the wrong flux
  decomposition.
- The verifier's checks #4 (velocity RMSE), #5 (flux decomposition),
  and especially #6 (self-consistency re-running the moving-frame
  solver from the agent's parameters) are designed to catch this
  exact failure.

### Sign of `v₀`
`v₀ > 0` means the solvent is moving in the `+x` direction, i.e. away
from the plating electrode at `x = 0`. The published `solver.py` uses
this convention. Do not "fix" any sign without simultaneously updating
`oracle/flux.py`, `formalism.md`, and every test that compares `v_pred`
to `v_data`.

### `t⁺⁰_NE` is a derived quantity, not a parameter
`t⁺⁰_NE(c)` is *defined* as the result of a lab-frame inversion
(`oracle/invert_ne.py`). It is not an independent function the case
designer picks. Two consequences:

1. The 4 cases' "regime intent" is set by the *relationship* between
   the chosen `t⁺⁰(c)` and the `t⁺⁰_NE(c)` that falls out of the
   lab-frame inversion on the same `c_data` — not by independently
   prescribing both.
2. When tuning a case, change `t⁺⁰(c)` (and possibly the noise level
   or current schedule), then re-run the NE inversion to see where
   `t⁺⁰_NE` ends up. Iterate until the regime labels distribute as
   intended.

### Flux decomposition convention is ADR-locked
Per ADR-0003 and `docs/plan/oracle-spec.md` §4. The canonical formulas:

```
J_diff = − D(c) · (1 − d ln c₀ / d ln c) · ∂c/∂x
J_mig  =   t⁺⁰(c) · i(t) / F
J_conv =   c · v₀
```

Both the case generator and the verifier import `oracle/flux.py`. The
agent reads `formalism.md` and matches. Any "I think the more natural
convention is..." reasoning belongs in a new ADR or nowhere.

## Numerics

### JAX float64 is non-optional
`solver.py` and the README both call this out. Float32 silently fails
at high concentration. Set `jax.config.update("jax_enable_x64", True)`
*and* set `JAX_PLATFORMS=cpu` *before* any JAX import. The case
generator and the verifier both enforce this in their entrypoint.

### `lax.scan`, not Python for-loops
Both the oracle solver and the reference solution time-step inside a
`jax.lax.scan` (this is what `solver.py` does). Unrolling the loop in
Python with `jax.grad` blows up the trace; the proposal calls this out
as one of the listed failure modes for agents.

### Cumulative-sum velocity update
The `v₀` update in `solver.py` is a cumulative sum over interior faces
with `v₀[0] ← 0` from the moving-boundary condition. Naive recomputation
of `v₀` from a discretized version of `∂v₀/∂x = V̄ ∂F/∂x` from scratch
will get the boundary value wrong and silently introduce a constant
offset to the entire velocity field. Match `solver.py`'s
`update_solvent_vel` exactly.

### Published-fit residual bounds the achievable case tolerance
The published Steinrück-2020 fit (BFGS to convergence with the published
2-parameter `tp0` polynomial and the `D(c) = (1-tp0)·relation_coef/factor`
ansatz) reaches an experimental-data fit residual of **~6 % mean,
~9 % worst-point** across the 9 sampled times × 50 cells (measured in
`scripts/smoke_test.py`, 2026-06-23). That residual is dominated by the
experimental noise plus the limitations of the published 2-parameter
ansatz, *not* by any solver defect — `scripts/smoke_test.py` shows the
solver itself reproduces the published `c_sim.npy` to ~3e-16 relative
error (machine identity).

Implications:
- The oracle-generated cases have synthetic noise instead of
  experimental noise; we control noise level, so the equivalent
  fit-residual budget for case generation can be tighter than 6 %.
- But any noise level we choose puts a *floor* on the achievable
  `D(c)` / `t⁺⁰(c)` recovery error. ADR-0005's "50 % margin on the
  worst point" check has to be calibrated against this floor — pick
  noise so that the floor is well below 0.05 absolute on `t⁺⁰` and
  5 % relative on `D`.

### Forward Euler is sufficient at the published `Δt`
`solver.py` uses first-order explicit Euler with `dt = 0.1 s`. This is
stable for the published parameter range and the case-generation
parameter range we plan. If you bump `dt` above ~0.5 s, expect
instability at high `i`. Keep `dt` per case in `case_gen/configs/`.

## Optimization

### NE inversion is information-limited at low current / low SNR

First end-to-end smoke (2026-06-23, case_1 draft YAML) showed BFGS on
the 50-point `tp0_NE` ansatz moved only ~0.05 from initialization
(loss 4.01e-5 → 3.71e-5) under σ_c = 0.006 mol/L and peak `i = 4 A/m²`.
The data fit was already at the noise floor (σ_c² = 3.6e-5), so there
was little extra information for BFGS to extract about the *shape* of
`tp0_NE(c)`. Consequences:

- At low SNR, `tp0_NE` collapses toward the spatially-averaged init
  value; the inversion underfits the tails.
- For an NE-valid case, this is harmless *if* the true `tp0(c)` is
  also nearly constant (within 0.05 of the same average). It bites
  when the true `tp0(c)` varies more than ~0.05 across `c_grid`.
- Calibration knobs (later, when reference solver lands): flatten true
  `tp0(c)` shape for the easy case; or boost peak current; or reduce
  noise σ. ADR-0005's 50%-margin audit will force one of these.

### Multi-modal basin separation in Case 4
Case 4 is intentionally constructed so a single-start optimization
from a uniform-`t⁺⁰` initial condition lands in a positive-`t⁺⁰` basin
that fits `c_data` to within a small but nonzero residual, but predicts
the wrong `v_data` (and therefore fails checks #4 and #5). The "right"
basin requires either physical intuition (`t⁺⁰ < 0` at high c) or
multi-start with seeds that probe negative values.

**Calibration target:** with single-start BFGS from `t⁺⁰ ≡ 0.3`,
the wrong basin should be hit ≥ 70 % of the time across 10 RNG seeds.
This is what lets us hit the 10–20 % solve-rate band.

### Tolerance feasibility checks come *before* parameter lock-in
Per ADR-0005, the reference solution must clear each verifier check
with ≥ 50 % margin. If a case fails this audit, do not "tighten the
case to make it more discriminating" — relax the noise or the
parameter contrast until margin is recovered, then re-test the
failure-mode discriminator.

### `jaxopt.ScipyMinimize(BFGS)` is the reference choice
What `bfgs.py` uses in the published code. Easy to drive from a
multi-start wrapper. Forward-mode value-and-grad (the
`make_value_and_grad_forward` helper in `bfgs.py`) is fast for the
2-parameter polynomial; for the 50-point free parameterization the
reference solution will need either reverse-mode grad or a smaller
parameterization (cubic spline with K knots, K ≪ 50). Decide and
record once we get to the reference solution.

## Anti-cheat

### Don't ship anything in `cases/` that isn't in the spec
The agent's prompt is exactly `cases/case_X/{data.h5, params.json, formalism.md}`.
Nothing else. Pre-PR CI (and any local smoke test) should grep `cases/`
for stray files (e.g., accidental `truth.npz` copies, debug dumps,
`__pycache__`). If you find one, treat it as a leak and re-generate.

### Held-out values must be ≥ 0.10 from published values for `t⁺⁰`
ADR-0004. Specifically: at every `c_grid[i]` point in every case,
`|t⁺⁰_oracle[i] − t⁺⁰_published(c_grid[i])| ≥ 0.10` for every
published reference (Pesko 2017 polynomial, DiffEC paper's fitted
polynomial, Steinrück 2020-derived values). This is checked
automatically by `case_gen/litvalue_distance.py` and recorded per
case in this file.

### `D(c)` perturbation budget
Analogously: `|D_oracle / D_published − 1| ≥ 0.20` at every c_grid
point. Larger margin because `D(c)` is the easier of the two to look
up.

## Reference-solution margins (to be filled in after first calibration)

> Updated whenever case configs or the reference solution change.
> Required by ADR-0005's audit mode.

| Case | Check #1 (D rel err) | Check #2 (t⁺⁰ abs err) | Check #4 (v RMSE) | Check #6 (self-consistency) |
| --- | --- | --- | --- | --- |
| case_1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| case_2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| case_3 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| case_4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Thresholds: 0.10 / 0.05 / 0.15 / 0.15 respectively. Target margin: 50 %.

## Frontier-agent pilot facts (to be filled in after pilot)

| Agent | Case 1 | Case 2 | Case 3 | Case 4 | Aggregate |
| --- | --- | --- | --- | --- | --- |
| Claude Opus 4.7 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| GPT-5 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Gemini 2.5 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Target aggregate band per ADR-0008: **10–20 %**.
Failure-mode classification per agent goes in `docs/progress/pilot_run.md`.
