# Proposal: graded sensitivity (gradient) deliverables as v0.3 hardening

**Status:** ACCEPTED as ADR-0015 (2026-09-24, user approval; implemented
the same day). This file keeps the rationale, experiments and rejected
variants; the binding decision text is ADR-0015 in `decisions.md`, the
contract is formalism §3.4, the check is `tests/test_outputs.py::test_sensitivity`.

**Origin.** Reviewer AaronFeller (PR #584): pass rate too high. ADR-0013
(L1+L2) moved Opus-5-xhigh to ~1/2 on n=2 mock trials; ADR-0014 closed the
information-starvation lever ("future difficulty must come from new physics
scope"). Haotian Chen (DiffEC first author) suggested forcing gradient
evaluation. This document turns that hint into a concrete, gradable,
method-agnostic deliverable.

## 1. Evidence: what agents actually build

Both cold Opus-5-xhigh mock trials (`mock_exam/results/mock-opus5-hardened*`)
and the passing reviewer trial built **pure NumPy forward solvers driven by
`scipy.optimize.least_squares`** — finite-difference Jacobians, no JAX, no
autodiff, no adjoint. Grep of both transcripts: `least_squares(` 11–12 hits,
`import jax` / `jaxopt` / `jax.grad` 0 hits. The "differentiable" in the
task title is currently not exercised at all: at this problem size (100 cells
× 11 000 steps, ≤ 20 fit parameters) a finite-difference Jacobian costs a
few seconds per iteration and is perfectly accurate.

Consequence: **we cannot force autodiff by cost or accuracy.** Measured
today on the held-out oracle (case 4, N=100): one jitted forward solve
0.26 s; 100 JVP directions 27 s; central finite differences (ε = 10⁻³)
agree with AD to 10⁻⁵ – 10⁻² relative. Any gradient quantity we ask for can
be produced by finite differences on a NumPy solver within budget, as long
as the parameter dimension stays O(10–100).

So the honest framing (the one the user already proposed) is the only
defensible one: **make the gradient a scientific deliverable with physical
meaning, graded deterministically**, not a method requirement. This is new
physics scope (sensitivity / identifiability analysis), which is exactly
where ADR-0014 said further difficulty must come from, and it is the literal
subject of the proposal's title ("Differentiable modeling …").

## 2. What the gradient tells us (why it is physics, not numerics)

For the moving-frame model the Jacobian of the observables with respect to
the transport functions answers three questions the current task never
asks:

1. **Which concentration window controls cell behaviour.** The gradient of
   an engineering quantity of interest (QoI) — e.g. end-of-hold salt
   polarization Δc = c(L, T_hold) − c(0, T_hold) — with respect to
   ln D(c) and t⁺⁰(c) is a *kernel in c*: it is non-zero only over the
   concentrations the profile actually visits, and its sign structure
   encodes the moving-frame coupling (raising t⁺⁰ lowers migration-driven
   polarization but also changes the convective flux through
   (1 − t⁺⁰) i/F in v₀). Measured at case 4 truth: ∂Δc/∂t⁺⁰ is O(0.2–0.7)
   mol/L per unit t⁺⁰, ∂Δc/∂ln D is O(0.4–0.6) mol/L at the band ends and
   ~0 in the interior of the narrow c_grid.
2. **Which measurement carries the information** (the paper's central
   claim). The Fisher-information split between the c and v channels,
   f_v(c_j) = I_v / (I_c + I_v), measured at truth with the bundled noise
   levels: **velocity carries ~70 % of the information on t⁺⁰** (median
   f_v = 0.68 case 4, 0.71 case 1; range 0.10–0.92) and, less obviously,
   ~70 % of the information on D as well. This is the quantitative form of
   "t⁺⁰ < 0 is invisible to a concentration-only, lab-frame analysis".
3. **Where the recovered functions are actually determined.** The RMS
   sensitivity of the observables to each node is the identifiability
   profile that motivated the maintainer note on the sparse high-c knots
   and the reference's λ-ladder. Reported by the agent, it becomes part of
   the scientific answer instead of a hidden calibration concern.

## 3. Feasibility experiment (2026-09-22, scratchpad `exp_sens.py`)

Setup: held-out oracle at case truth; perturbation basis = piecewise-linear
hats on the **50-point c_grid** (Δc ≈ 0.0075 mol/L), clamped at the ends
(`jnp.interp` semantics, identical to check #6's model); ln D perturbed
relatively, t⁺⁰ absolutely; full Jacobian by 100 JVPs. Three robustness
tests, deviation = max|Δ| / max|reference| over the block of nodes:

| quantity | refine N=100→200, dt/2 (case 4 / case 1) | truth → gate-edge params (D×1.08, t⁺⁰+0.04) |
|---|---|---|
| full Jacobian ∂c/∂θ, pointwise | 3 % (ends) but 70 % interior / 35 % | 7 % / 100 % |
| full Jacobian ∂v/∂θ, pointwise | 94 % / 73 % | 100 % / 126 % |
| ∂Δc_pol/∂ln D | 0.6 % ends, 6 % interior / 29 % | 6 % / 70 % |
| ∂Δc_pol/∂t⁺⁰ | 1 % / 5 % | 4 % / 11 % |
| kernel of c at 10 flux points, K_c | 2 % (lnD) 7 % (t⁺⁰) / 1 % 2 % | 25–43 % / 17–21 % |
| kernel of v at 10 flux points, K_v | 3–4 % / 1 % | 19–26 % / 5–23 % |
| RMS sensitivity per node, t⁺⁰ block | 1–2 % / 1 % | 2–3 % / 2–5 % |
| RMS sensitivity per node, ln D block | 26–42 % interior / 22–59 % | 19–24 % / 31–76 % |
| info fraction f_v, max abs dev | 0.25 / (NaN nodes) | 0.09 / — |

Timing (oracle, per case): forward 0.26 s; 100 JVPs 27 s at N=100, 87 s at
N=200. A verifier that recomputes 20–30 directions per case adds well under
a minute total — inside the 600 s verifier timeout.

Three lessons that fix the design:

- **Narrow hats on c_grid are the wrong basis for D.** D enters through
  ∂/∂x(D ∂c/∂x); a 0.0075 mol/L-wide bump produces mesh-dependent flux
  kinks, so pointwise D-sensitivities are not discretization-robust (26–70 %
  drift under refinement). t⁺⁰ enters as a smooth source term and is robust.
  The basis must be **wide knots spanning the visited concentration range**
  (like the reference's 10-knot parameterization). Pointwise full Jacobians
  are out; **integrated quantities** (QoI gradients, kernels at sample
  points, RMS profiles) are in.
- **Grade truth-free.** Sensitivities evaluated at the truth vs. at a
  solution sitting exactly on the D/t⁺⁰ gate edge differ by 20–40 % for the
  kernels. Grading against truth-evaluated sensitivities would need
  tolerances that wide, i.e. it would test nothing. Instead the verifier
  **recomputes the sensitivities of the canonical model at the agent's own
  reported parameters** by AD through the held-out solver, and compares.
  This is the differential analogue of check #6: it tests that the agent
  can differentiate the stated physics, with no oracle-specific curve
  involved, and its tolerance covers only discretization error (~1–7 %
  measured → 20 % gate gives ≥3× margin, same policy as ADR-0005).
- **Clamped endpoints dominate on narrow c_grid.** For cases 3/4 the
  c_grid window is a small sub-range of the data (case 4: c_grid
  [2.93, 3.30] vs c_data [2.03, 4.39]); the end nodes of a clamped basis own
  the whole exterior and swamp the interior. The wide-knot basis on the full
  visited range removes this artefact.

## 3b. Second experiment (2026-09-24): adjoint gradients of a scalar QoI

Rationale: DiffEC itself only differentiates the fit loss (`bfgs.py`, two
parameters), which is ≈ 0 at convergence and ungradeable. The property a
differentiable solver uniquely provides is **reverse mode**: the gradient of
one scalar physical quantity with respect to *all* inputs in one backward
pass. The deliverable should have that shape.

Setup (`_sens_exp/exp_qoi.py`, `exp_qoi2.py`; results `qoi_case_{1,3,4}.json`,
`qoi2_case_4.json`): scalar QoI **Q = c(x_last, t_q) − c(x_first, t_q)**
(end-of-plateau polarization on the bundled x grid, t_q = end of the
constant-current hold = 1000 s). Inputs: ln D and t⁺⁰ as clamped hats on
**12 wide knots** uniformly spanning [min c_data, max c_data]; the current
program i(t) as hats on the 50-point bundled time grid (amplitude 1 A/m²);
the initial state c(x, 0) per data cell. 174 inputs total, `jax.jacrev`.

| block | refine N=100→200, dt/2 (cases 1 / 3 / 4) | truth → gate-edge (D×1.08, t⁺⁰+0.04) |
|---|---|---|
| ∂Q/∂ln D knots | 3 % / 4 % / 9 % | 27 % / 28 % / 41 % |
| ∂Q/∂t⁺⁰ knots | 1 % / 1 % / 2 % | 8 % / 10 % / 10 % |
| ∂Q/∂i(t_k) memory kernel | 1 % / 2 % / 2 % | 7 % / 2 % / 4 % |
| ∂Q/∂c₀(x_i) adjoint state | 0.1 % / 0.2 % / 0.3 % | 4 % / 3 % / 3 % |

Cost on the held-out oracle: forward 0.2 s; **all 174 sensitivities in one
`jacrev` call, 0.5 s per case** (1.8 s at N=200). By finite differences:
174 forward solves ≈ 40 s in jitted JAX, ≈ 3–5 min per case for a NumPy
solver of the kind both mock agents wrote.

Physics of the case-4 truth gradients (sanity anchors an agent can use):
∂Q/∂ln D > 0 at every knot (more D, less polarization); ∂Q/∂t⁺⁰ > 0 and
peaked at c ≈ 2.9–3.1 (the bulk concentration); memory kernel negative,
growing toward t_q, **exactly zero after t_q** (causality); adjoint state
antisymmetric across the cell with sum −0.90 (non-zero because D, t⁺⁰
depend on c).

Rejected QoIs (tested): solvent velocity at the far electrode is identically
zero by the flux BC (v₀ face at x=L = V̄·(i/F − F_right) = 0); mid-cell
velocity and mid-cell solvent displacement have a t⁺⁰ gradient that is just
the analytic direct term −V̄ i/F·h_k(c_mid) (no PDE content) and a D gradient
that is ill-conditioned (26 % refinement drift, 133 % at the gate edge).
Velocity stays graded as a field (checks #4/#6); polarization is the QoI.

## 4. Proposed deliverable (`transport.json` additions) — revised 2026-09-24

`params.json` gains `c_knots[12]` (uniform over [min c_data, max c_data])
and `t_qoi_s` (end of the current plateau). `formalism.md` gains §3.4: the
canonical knot model (piecewise-linear in c on `c_knots`, constant outside
— the verifier's existing interpolation rule, stated for the extended
grid), the perturbation bases (δ ln D = ε h_k(c), δ t⁺⁰ = ε h_k(c),
δ i = ε·1 A/m²·h_k(t) on the bundled `t`, δ c₀ = ε at cell i), the QoI
definition and sign, units of every array.

New output fields:

| field | shape | graded | meaning |
|---|---|---|---|
| `D_knots`, `t_plus_0_knots` | [12], [12] | consistency (reproduce `D`/`t_plus_0` on c_grid to 5 %; finite; D > 0) | the agent's full recovered functions — base point of the sensitivities |
| `sensitivity.Q_pol` | scalar | 20 % vs verifier recomputation | Q at the agent's parameters |
| `sensitivity.dQ_dlnD` | [12] | 20 % of max\|block\| | which concentration window's D controls polarization |
| `sensitivity.dQ_dtp0` | [12] | 20 % | moving-frame migration/convection control |
| `sensitivity.dQ_di` | [Nt] | 20 % | current-program memory kernel (mol/L per A/m²) |
| `sensitivity.dQ_dc0` | [Nx] | 20 % | adjoint state at t = 0 |

Check #9 (truth-free): rebuild the knot model from `D_knots`/`t_plus_0_knots`,
`jax.jacrev` through the held-out solver, compare block-wise with
max|agent − verifier| ≤ 0.25·max|verifier| (gate widened from 0.20 after
the final-definition calibration `_sens_exp/calib_final.json`: ln D drift
11 % on case 4 → 2.25× margin; other blocks ≤ 3 % → ≥ 8×).

Difficulty knob: the memory-kernel resolution. On the 50-point bundled grid
FD costs 174 solves per case; on a 200-point canonical grid it costs 324,
i.e. ~20–40 min of pure compute across 4 cases for a NumPy solver, while
reverse mode is unchanged. Start with the bundled grid; raise only if a
fresh mock trial shows FD agents pass with time to spare.

Anti-cheat unchanged from §4-old: depends on the agent's own parameters,
cannot be looked up or copied from data, causality/sign structure makes
zero-filled or guessed arrays fail by construction.

Superseded: the (a)/(b)/(c) menu of the 2026-09-22 draft. (b) point kernels
and (c1) RMS profiles remain valid fallbacks; (c2) info fraction and the
σ publication question are dropped from the baseline.

## 5. Where the difficulty comes from (and where it does not)

Not from computation: FD over 24 directions × 4 cases costs a NumPy solver
~1–3 min. The agent is free to use FD; JAX makes it ~seconds. Expected
sources of failure, in order of likelihood:

1. **Total vs. partial derivative.** Agents that compute v₀ from the
   closed form after solving for c will be tempted to differentiate the
   closed form at fixed c (partial), whereas the graded quantity is the
   total derivative through the coupled c–v₀ evolution. Same class of trap
   as the ADR-0012 closure — but this time the spec states it once and the
   check is truth-free, so it is fair.
2. **Basis / normalization slips:** ln D vs D, hat support and clamped
   ends, the QoI sign convention (Δc = c(L) − c(0)), nm/s vs m/s in the v
   kernels, evaluation at `t_hold` vs `T`.
3. **Time pressure**, secondary: the two mock trials used 27 and ~50 min
   of the 60 min budget before this deliverable existed. Adding a
   sensitivity pass with FD is cheap in CPU but not in agent turns.
4. **Model-order interaction:** an agent whose internal model is a global
   cubic must still evaluate the canonical knot model's sensitivities, not
   its own — the spec has to be explicit that (a)–(c) are properties of the
   knot model built from the reported knot values.

Honest expectation: this is a **new capability axis**, not a wall. A top
agent that already built a correct joint inverse will most likely pass it
too on a good day; the gain is (i) a wider contract surface on which
correct-method attempts can slip, (ii) the differentiable-modeling content
of the title finally being graded, (iii) a stronger difficulty story for
reviewers than more noise or compute ("the task now asks for the sensitivity
analysis a domain scientist would publish"). We should not promise the
reviewer a pass-rate number before a fresh mock trial.

## 6. Calibration plan (precondition for ADR-0015, per ADR-0005)

1. Re-run `exp_sens.py` with the **12-knot basis over [min c_data,
   max c_data]** on all 4 cases: refine test (N=200, dt/2) and clean-room
   check (reference `solution/pde.py` vs. held-out oracle at the same knot
   values). Go criterion: every graded block drifts ≤ 7 % so a 20 % gate has
   ≥3× margin; else widen the basis or drop the offending block.
2. Add `jax.jacfwd` sensitivities to the reference solver (its `pde.py` is
   JAX already) and confirm the reference passes the new check with the
   same margin on all 4 cases; time it (expected +10 s/case).
3. Implement `tests/oracle/sensitivity.py` + `test_sensitivity` +
   generator/writer updates (`c_knots`, formalism §3.4, instruction check
   list), regenerate cases (params.json changes → data.h5 unchanged), rerun
   the full anti-cheat matrix, `harbor run -a oracle` → reward 1.
4. One cold Opus-5-xhigh mock trial (same config as Sep 9) to measure
   effect and time usage; then `/run` request.

Rough effort: 1 day for steps 1–2, 1–2 days for step 3 including docs,
1 day for the mock trial. Fits before the v0.2 window closes on
2026-10-05 only if the design is fixed this week.

## 7. Open questions for Haotian / the user

- Which QoIs does DiffEC itself use for sensitivity plots (polarization,
  limiting current, time-to-steady-state)? Matching the paper's framing
  makes the deliverable read as "the analysis the method was built for".
- Is the velocity-information fraction (c2) worth publishing σ_c, σ_v in
  `params.json`? It is the most quotable insight ("~70 % of the t⁺⁰
  information is in v") but adds a noise-model statement to the contract.
- Knot count: 12 uniform knots over the visited range is a guess; the
  reference uses 10. Fewer knots → more robust, less resolution.

## 8. Alternatives considered and not recommended

- **Gradient of the misfit at the reported solution:** ≈ 0 for any
  converged fit, ungradeable.
- **Optimizer-trajectory reporting** (loss/gradient history): not
  deterministic across methods, not physics.
- **Posterior 1σ error bars on D, t⁺⁰** (Gauss-Newton covariance): the
  strongest scientific deliverable but needs a canonical prior/λ to keep
  (JᵀJ + R)⁻¹ well-posed at sparse knots — a large new contract surface
  with ADR-0012-class ambiguity risk. Revisit only if (a)–(c) prove too
  easy.
- **High-dimensional influence maps** (∂θ̂/∂v_data per pixel — genuinely
  reverse-mode-only): physically the nicest "which measurement matters"
  quantity, but for a 24-parameter knot model it reduces to
  (JᵀWJ)⁻¹JᵀW, again FD-able, and it inherits the prior problem above.
