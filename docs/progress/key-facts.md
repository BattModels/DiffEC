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

### Velocity closure drops `c v₀` — no `(1 + c V̄)` denominator (ADR-0012)
The flux that drives the *velocity* update is evaluated at `v₀ = 0`
(`solver.py:196` / `pde.py:161` pass `jnp.zeros_like(v0)`), so
`v₀ = V̄[D φ c_x + (1 − t⁺⁰) i/F]` — **not** the self-consistent
`… / (1 + c V̄)`. This is the published DiffEC operator split. The
distinction is invisible at low c (dilute case_1: `c V̄ ≈ 0.02–0.04`,
~2–4 % velocity effect) but decisive at high c (case_4: `c V̄ ≈ 0.4`,
~27–29 % — well past the 15 % v-tolerance). It was **undocumented** until
PR #584's technical reviewer caught it (2026-07-30); `formalism.md` §2
now states the closure + closed form explicitly. Any forward model that
solves the coupled flux/velocity pair self-consistently (keeping `c v₀`)
will fail the concentrated cases on checks #1/#2/#4/#6. NB `formalism.md`'s
IC was already denominator-free, so it always matched the oracle — only
the second PDE was wrong.

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

### Case 4's basin trap didn't materialize (2026-06-25)
Case 4 was originally scoped as a multi-modal basin trap: single-start
BFGS from a literature-prior `t⁺⁰ ≈ +0.30` init would land in a
positive-`t⁺⁰` basin fitting `c_data` plausibly but mispredicting
`v_data`. In practice, our reference solver's **v-data-weighted
joint inverse** breaks the (D, t⁺⁰) degeneracy well before BFGS can
settle in a plausible-but-wrong basin — from a +0.30 init, joint c+v
optimization reliably lands in the correct deeply-negative basin.

Reframed as "NE-wrong-sign at high c" (companion to case_3 at
different concentration + realistic V_bar); the case still catches
lab-frame agents catastrophically on check #2 and #6 but doesn't
exercise basin-trap behavior. See `docs/plan/case-design.md` §"Case 4".

A real basin-trap design would require weakening the `v_data` weight
in the loss or introducing a genuinely pathological non-monotone
`t⁺⁰(c)` shape; deferred to a future case if the pilot shows
current cases are too easy in aggregate.

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

### Check #6 strength depends on V_bar magnitude

Anti-cheat sanity run for case_1 (`V_bar = 5e-5` m³/mol — the weak
override that makes case_1 robustly NE-valid):

| Submission                              | rmse / max | Pass check #6? |
| --------------------------------------- | ---------- | -------------- |
| Honest agent                            | 0.042      | yes            |
| Cheat: `tp0 = 0`                        | 0.310      | no             |
| Cheat: `tp0 = 0.15` constant            | 0.159      | no             |
| Cheat: `tp0 = oracle's t⁺⁰_NE` (lab-frame fit) | 0.046 | **yes (!)**    |

Case 1 is the NE-valid calibration case by design — lab-frame agents
*should* pass it, so this is expected. Two important consequences:

1. **Check #6 alone is not a moving-frame discriminator on case 1.**
   The discriminator is the *full* set of checks across all 4 cases.
   For case 1, checks #1/#2/#3 do the discrimination; #6 catches only
   garbage (`tp0 = 0`) and crude constants.
2. **Cases 2-4 must use realistic V_bar** (≥ 1e-4 m³/mol) so that the
   lab-frame cheat fails check #6. Specifically, the gap between
   moving-frame v₀ and lab-frame v₀ has to exceed the 0.15 RMSE
   threshold. Verify per case before locking that YAML — log the
   "lab-frame cheat rmse/max" alongside the honest-agent margin so
   the discrimination is documented.

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

| Case | Check #1 (D rel err) | Check #2 (t⁺⁰ abs err) | Check #4 (v RMSE) | Check #5 (flux worst) | Check #6 (self-consistency) | Check #3 (regime) |
| --- | --- | --- | --- | --- | --- | --- |
| case_1 | 0.017 / 0.10 (83 %) | 0.013 / 0.05 (74 %) | 0.042 / 0.15 (72 %) | 0.0009 / 0.15 (99 %) | 0.042 / 0.15 (72 %) | 50/50 ✓ |
| case_2 | 0.024 / 0.10 (77 %) | 0.010 / 0.05 (81 %) | 0.042 / 0.15 (72 %) | 0.0003 / 0.15 (100 %) | 0.042 / 0.15 (72 %) | 50/50 ✓ |
| case_3 | 0.016 / 0.10 (84 %) | 0.008 / 0.05 (84 %) | 0.043 / 0.15 (72 %) | 0.0192 / 0.15 (87 %) | 0.050 / 0.15 (66 %) | 50/50 ✓ |
| case_4 | 0.026 / 0.10 (74 %) | 0.011 / 0.05 (78 %) | 0.043 / 0.15 (71 %) | 0.0080 / 0.15 (95 %) | 0.046 / 0.15 (70 %) | 50/50 ✓ |
| case_2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| case_3 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| case_4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Thresholds: 0.10 / 0.05 / 0.15 / 0.15 / 0.15 respectively. Target margin: 50 %.

Cases 1-4: all continuous checks (#1, #2, #4, #5, #6) meet ADR-0005's
50 %-margin precondition, and the categorical regime check (#3)
passes exact-match (50/50) on all 4 cases — **28/28 total**.

Calibration moves that mattered:
- `LAMBDA_SMOOTH_D` lowered from 1e-2 → 1e-4 in `reference_solver.py`
  (2026-06-24) — the 10-knot D parameterization was over-smoothed
  into a near-constant fit, masking the linear D-slope c_data encodes.
- `DT_INV_S` lowered from 0.5 → 0.1 in `reference_solver.py`
  (2026-06-24) — agent's NE inversion now runs at the same dt as
  oracle's, tightening agreement on boundary param values
  (was causing borderline regime label flips in case_2).
- c_grid narrowed per case to where polarization gives good data
  density (case_1: [0.55, 0.75]; case_2: [1.13, 1.49]; case_3:
  [2.20, 2.80]). Outside these ranges, D and t⁺⁰ are poorly
  constrained and the endpoint c_grid points have noisy fits →
  margin drop.
- Reference solver's knot range tracks **c_data**, not c_grid
  (added in case_3 calibration, 2026-06-25). The forward PDE
  evaluates D(c) and tp0(c) at every cell, with c often spanning
  far outside the narrow c_grid. Knots only over c_grid leaves
  the agent's model flat-extrapolated outside, so c_sim can't
  match c_data near electrodes and the joint inverse misfits
  catastrophically. Cases 1 and 2 also improved with this fix
  (case_2's D margin 53 % → 77 %).
- case_2 uses `V_bar_si = 2.5e-4` override (~1.7× rho-derived)
  to push the moving-frame v₀ enough that the lab-frame NE
  inversion's |tp0 − tp0_NE| gap is robustly > 0.07 across
  c_grid, avoiding borderline regime labels.

Wall-time cost of `DT_INV_S = 0.1`: per-case joint inverse grew from
~30s to ~140s. Total per case ~3 min — still under the 5-10 min
budget. NE inversion grew from ~5s to ~25s.

### case_3 regime debt: RESOLVED via cubic NE ansatz (2026-06-26)

After perturbing case_3's `tp⁺⁰(c)` by -0.12 to satisfy ADR-0004,
the original 50-point free-knot lab-frame NE inversion produced
`tp⁺⁰_NE(c)` with zero crossings inside c_grid. Oracle and agent
BFGS placed the crossings at slightly different c values (~1 c_grid
index off), flipping 1-2 regime labels at each crossing. Six
iterations of c_grid narrowing + lambda_reg adjustment couldn't
eliminate the drift.

**Fix (2026-06-26):** replaced the 50-point free-knot ansatz with a
**cubic polynomial** in normalized concentration in both
`oracle/invert_ne.py` and `solution/lab_frame_solver.py`:

```
tp⁺⁰_NE(c) = a₀ + a₁ u + a₂ u² + a₃ u³,   u = (c − c_avg) / c_scale
```

The 4-parameter polynomial fit lands at the same smooth optimum in
both implementations — zero crossings (if any) at reproducible c
values. Also switched `jaxopt.ScipyMinimize` → `jaxopt.LBFGS`
(pure JAX): `scipy.minimize`'s line search aborted immediately with
NaN on this loss for cases 3, 4 (lab-frame simulation can blow up
under bad trial steps for high-degree polynomial); `LBFGS`'s
built-in line search handles it gracefully.

Result: all 4 cases now pass 28/28 with regime labels matching
50/50 exactly between oracle and agent inversions.

## Frontier-agent pilot facts (to be filled in after pilot)

| Agent | Case 1 | Case 2 | Case 3 | Case 4 | Aggregate |
| --- | --- | --- | --- | --- | --- |
| Claude Opus 4.7 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| GPT-5 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Gemini 2.5 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Target aggregate band per ADR-0008: **10–20 %**.
Failure-mode classification per agent goes in `docs/progress/pilot_run.md`.

## ADR-0015 sensitivity check margins (2026-09-24, v0.3)

Check #7 (`test_sensitivity`) is truth-free, so it has two distinct
margins. (a) *Implementation* margin — reference (clean-room `pde.py`,
`jax.value_and_grad`) vs the verifier's `oracle.sensitivity` at the same
knot values, ratio of deviation to the 0.25·max|block| gate: ≤ 0.034
(dQ_dlnD, case 3) and ≤ 4e-3 on every other block; Q_pol agreement ≤ 2e-3
of its 10 % gate. (b) *Discretization* margin — the irreducible drift a
correct solver on a different grid carries, measured on the oracle at
N=100/dt=0.1 vs N=200/dt=0.05 (`docs/plan/_sens_exp/calib_final.json`):

| Case | dQ_dlnD | dQ_dtp0 | dQ_di | dQ_dc0 |
|---|---|---|---|---|
| 1 | 0.025 | 0.008 | 0.018 | 0.001 |
| 2 | 0.044 | 0.012 | 0.021 | 0.003 |
| 3 | 0.042 | 0.009 | 0.016 | 0.002 |
| 4 | 0.111 | 0.025 | 0.019 | 0.006 |

→ the 0.25 gate holds 2.25× margin on the worst block (case 4 ln D) and
≥ 8× elsewhere. Base-point rule (knot model must reproduce `Q_data` to
10 %): reference ratios 0.27 / 0.20 / 0.02 / 0.12 (cases 1–4); case 1's
2.7 % deviation is dominated by the noise on `Q_data` itself
(σ_c·√2/|Q| ≈ 2.6 %), i.e. the 10 % gate is ≈ 3.8σ. Existing checks
unchanged (table below). 36/36 green; reference runtime +1.1–1.6 s/case.

### Gotcha: narrow hats on c_grid are not a gradable basis for D

D enters through ∂/∂x(D φ ∂c/∂x). A hat perturbation 0.0075 mol/L wide
(one c_grid spacing) produces mesh-dependent flux kinks: pointwise
D-sensitivities drift 26–70 % under grid refinement, while t⁺⁰ (a source
term) drifts ≤ 2 %. That is why the sensitivity contract lives on 12
wide knots over the *whole observed* c-range and grades one integrated
scalar, not the field Jacobian (`docs/plan/gradient-hardening-proposal.md` §3).

### Gotcha: truth-based grading of sensitivities is impossible

A solution sitting exactly on the D/t⁺⁰ gate edge (D×1.08, t⁺⁰+0.04)
shifts dQ_dlnD by 27–41 % relative to truth. Check #7 therefore
recomputes everything at the *agent's* knot values; the tolerance covers
discretization only. Consequence for future edits: never add a
sensitivity tolerance that references `truth.npz` values.

### Gotcha: velocity is a poor QoI in this cell

`v₀` at the x = L face is identically 0 (both electrode faces carry the
same flux i/F, and v₀ = V̄·(i/F − F_face)); mid-cell velocity's t⁺⁰
gradient is the analytic direct term −V̄ i/F·h_k(c_mid) with no PDE
content, and its D gradient is ill-conditioned (133 % shift at the gate
edge). Polarization `Q` is the QoI; velocity stays graded as a field.

### Gotcha: the current-memory kernel must vanish after t_qoi

`dQ_di[k]` for hats entirely after `t_qoi` is exactly 0 (causality). The
hat straddling `t_qoi` is non-zero. A submission with non-zero entries
beyond the plateau has differentiated the wrong quantity — the block
tolerance catches it because the pre-plateau entries are O(1e-3–1e-2).

## ADR-0013 hardening margins (2026-09-09, post case_4 redesign)

Reference-solution worst-point ratios vs tolerance (ratio < 1 = pass;
32/32 green; `_local_results` from the λ-ladder reference):

| Case | D (÷0.10) | t⁺⁰ (÷0.05) | graded regime miss | v-RMSE (÷0.15) | flux (÷0.15) | honest #6 |
|---|---|---|---|---|---|---|
| 1 | 0.17 | 0.26 | 0/50 | 0.28 | 0.01 | 0.042 |
| 2 | 0.22 | 0.20 | 0/50 | 0.28 | 0.00 | 0.042 |
| 3 | 0.22 | 0.09 | 0/50 | 0.28 | 0.13 | 0.057 |
| 4 | 0.39 | 0.42 | 0/42 | 0.29 | 0.03 | 0.047 |

All ≥ 2.3× margin. Lab-frame cheat and literature-lookup catches
re-verified (see case-design.md tables). Case_4 numbers are for the
c_grid extended to 3.30 (mock-trial-driven, see below); its t⁺⁰ worst
point (0.021 abs) is the new high-c endpoint.

### Gotcha: a global t⁺⁰ smoothness weight cannot serve all cases

`LAMBDA_SMOOTH_TP = 1e-3` (v0.1 reference) flattens case_4's steep
`t⁺⁰(c)` transition — recovered slope biased ~25 % low, D up to 12 % off
at the high-c tail → reference FAILED checks #1/#2 on the redesigned
case. But dropping to 1e-5 globally makes case_1's weakly-constrained
`t⁺⁰` oscillate (worst error 0.060 > 0.05 gate): its wiggle is
data-irrelevant (misfit changes < 0.05 % across λ), which is exactly why
only the prior can suppress it. Fix: per-case λ selection from the
ladder (1e-3, 1e-5), keeping the largest λ whose pure data misfit is
within `DATAFIT_TOL = 1.002` of the best rung. Selected: cases 1–2 →
1e-3, cases 3–4 → 1e-5. Deterministic; ~2× joint-inverse cost
(~2 min/case, still ≪ budget).

### Gotcha: `case_gen/writers.py::FORMALISM_MD` must be kept in sync

The shipped `formalism.md` files were patched directly during v0.1
review (velocity closure, §3.2 rule, §5 RMS wording) without updating
the template — a regeneration would have silently reverted three
reviewer fixes. Synced 2026-09-09 (`553c517`). Any future spec edit goes
through the template, then `generate --all`.

### Regime mask (ADR-0013)

`regime_graded = |t⁺⁰_oracle| ≥ 0.03` stored per case in `truth.npz`;
verifier grades labels only there (graded counts 50/50/50/42 = 192/200).
δ = 0.03 is ~2.7× the reference's t⁺⁰ error near the crossing (~0.011;
its global worst 0.021 sits at the far c_grid endpoint, not near c*).
The masked-band width is set by the truth slope at the crossing
(−1.0 /(mol/L) → ~8 masked points at Δc = 0.0076); a shallower crossing
would mask proportionally more.

### Mock-trial-driven c_grid extension to 3.30 (2026-09-09)

Cold containerized mock trial (harbor, subscription auth, full
isolation): claude-opus-5 xhigh PASSED the [2.93, 3.20] hardened task
(reward 1.0, 27 min, ~29 agent turns). Its method: BIC-selected
polynomial joint inversion + σ-weighted c+v loss — a genuinely correct
solve; it located c* to 0.001 and hit all 39 graded labels. BUT its
t⁺⁰ error grew monotonically into the sparse high-c tail (0.028@3.13 →
0.037@3.20 = 1.35× margin; D 0.053@3.20), the signature of
parsimony-prior slope shrinkage the truth's post-crossing steepness is
designed to punish. Extending c_max 3.20 → 3.30 (still inside the
data-visited range, c_data max ≈ 4.27) puts that solution class at
~0.052 > 0.05 at the new endpoint while the reference sits at 0.021
(2.4× margin). Trial artifacts:
`mock_exam/results/mock-opus5-hardened__20260909-204211/` (gitignored).

### Confirmation mock trial on the 3.30 task (2026-09-09): FAIL, as designed

Same Opus-5-xhigh config vs the extended band: **reward 0.0, 31/32** —
sole fail `test_D[case_4]` rel_err 0.1003 at idx 0 (c = 2.93, low-c
graded edge; data-bulk region, reference margin 2.5× there). Cause:
Chebyshev global-polynomial ansatz with Δχ²-based order selection →
model bias ~10 % at the band edge (agent's own 1σ ≈ 2.7 % — bias, not
noise, binds). All graded labels correct; its label-split disagreement
with truth sits entirely inside the |t⁺⁰| < 0.03 mask. Agent spent 3.7 M
input tokens / $4.58 / ~75 min vs 27 min for the 3.20-version pass.
Mock evidence: 1/2 Opus-5-xhigh solves on the hardened task.
