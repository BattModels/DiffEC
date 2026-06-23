# Case Design

> Each case is one row of the discriminator. They share geometry, current
> schedule family, and noise model; they differ in the true `(D(c), t⁺⁰(c))`
> functions and the concentration range, which together set the regime.
>
> The held-out parameter functions live in `tb_sci_task/case_gen/configs/case_X.yaml`
> and are summarized — but not pinned to numerical values — here. Pinned
> values land after ADR-0005 tolerance calibration.

## Shared template

All cases use the same cell geometry and forward-model envelope:

- **Geometry:** symmetric Li | electrolyte | Li, `L ≈ 3 mm`, 1D, `Nx ≈ 100` uniform cells.
- **Time:** `T ≈ 1100 s` (matched to the Steinrück 2020 experiment family); `Nt ≈ 50` reporting times.
- **Current schedule `i(t)`:** ramp + hold + ramp-down, peak `i ≈ 0.04 A/cm²`. Case-specific perturbations only.
- **`(1 − d ln c₀ / d ln c)` factor:** tabulated per case from the case's `rho(c)` polynomial (same family as the published solver, with case-specific coefficients).
- **c_grid:** 50 uniform points spanning each case's `[c_min, c_max]`.
- **flux_samples:** 10 `(x, t)` pairs spread across the bulk (`0.2 L ≤ x ≤ 0.8 L`) and middle of the time window (`0.3 T ≤ t ≤ 0.9 T`). The case generator picks them deterministically from `seed`.
- **Noise model (oracle):**
  - `c_data ← c_sim + N(0, σ_c)`, `σ_c` calibrated so the inverse problem
    is well-posed under ADR-0005's 50 % margin requirement (target `σ_c / c_avg ≈ 1–2 %`).
  - `v_data ← v_sim + N(0, σ_v)`, `σ_v` calibrated similarly (target
    `σ_v / max|v| ≈ 5 %`).
  - Seeded; same seed → same noise instantiation.

## Case 1 — NE-valid (the easy one)

**Regime intent.** Dilute / weakly correlated electrolyte where the
Nernst-Planck approximation is essentially correct. Lab-frame agents pass.

**Concentration range.** `c ∈ [0.2, 1.0] mol/L`. Far from any sign flip.

**True `t⁺⁰(c)`.** Smooth, monotone in `[0.20, 0.45]`. Positive everywhere.
Slope chosen so that `|t⁺⁰ − t⁺⁰_NE| < 0.05` at every c_grid point.

**True `D(c)`.** Smoothly varying, no kinks; magnitudes around
`5e-7 cm²/s` (PEO-LiTFSI-ish but perturbed per ADR-0004).

**Designed failure mode caught.** None — this is the calibration case.
An agent that *fails* case 1 either has the wrong physics, the wrong
output schema, or NaN'd. It validates that the easy regime is reachable.

**Expected regime labels.** `[NE_valid] × 50`.

---

## Case 2 — NE-deviates (moderate solvent motion)

**Regime intent.** Moderate ion-solvent correlation: lab-frame and
moving-frame inversions both produce positive `t⁺⁰`, but with magnitudes
that diverge enough to fail check #1 or #2 for a lab-frame agent.

**Concentration range.** `c ∈ [0.5, 2.0] mol/L`.

**True `t⁺⁰(c)`.** Positive but decreasing; `t⁺⁰ ∈ [0.15, 0.35]`. The
lab-frame inversion (`t⁺⁰_NE`) overshoots upward, so
`t⁺⁰_NE − t⁺⁰ ≈ 0.10–0.20`.

**True `D(c)`.** Bowl-shaped with a minimum near `c ≈ 1.3 M`.

**Designed failure mode caught.** Lab-frame Nernst-Planck agents:
they recover `t⁺⁰_NE` rather than `t⁺⁰` and fail check #2 (absolute
tolerance 0.05) and check #5 (under-predicted convective contribution).

**Expected regime labels.** `[NE_deviates] × 50`.

---

## Case 3 — NE-wrong-sign (the headline case)

**Regime intent.** Reproduce the Steinrück-2020-like negative-`t⁺⁰`
phenomenon at high salt concentration. Lab-frame agents recover only
positive `t⁺⁰_NE` and fail catastrophically.

**Concentration range.** `c ∈ [0.5, 3.0] mol/L`. Spans the sign flip.

**True `t⁺⁰(c)`.** Starts positive (~0.3 at `c = 0.5 M`), crosses zero
near `c ≈ 1.8 M`, reaches ≈ `−0.3` at `c = 3.0 M`. Held-out value at the
crossover is perturbed from any published value (ADR-0004) — different
crossover concentration than Steinrück 2020, different magnitude than
the DiffEC fit.

**True `D(c)`.** Monotonically decreasing from `~1e-6` to `~3e-7 cm²/s`.

**Designed failure mode caught.**
- Lab-frame agents: recover `t⁺⁰_NE > 0` everywhere → fail check #2 wherever the true sign is negative; fail check #3 (regime labels include `NE_wrong_sign`) at 200 categorical points; fail check #5 (convective contribution under-predicted exactly where it is largest).
- Agents that use a single-mode parameterization (e.g., a linear `t⁺⁰(c)` like the published solver) but get the slope wrong: fail check #2 in the high-c tail.

**Expected regime labels.** Roughly `[NE_valid, NE_deviates, NE_wrong_sign]`
distributed across the c_grid; exact partition depends on the locked
parameter functions and is recorded once they're pinned.

---

## Case 4 — Multi-modal (the basin trap)

**Regime intent.** A loss landscape with two well-separated basins:
a "physically correct" basin with negative `t⁺⁰` at high c, and a
"plausible-looking" basin with positive `t⁺⁰` everywhere that fits
`c_data` reasonably well but predicts the wrong `v_data`. Single-start
optimization lands in the wrong basin a substantial fraction of the time.

**Concentration range.** `c ∈ [1.0, 3.5] mol/L`. Centered in the
concentrated regime where multi-modality is intrinsic.

**True `t⁺⁰(c)`.** Strongly non-monotone: positive at low c, dips
sharply negative around `c ≈ 2.5 M`, recovers slightly at the tail.
Constructed so that the lab-frame-style positive-`t⁺⁰` fit produces a
local minimum of the moving-frame loss within ~1 order of magnitude of
the global minimum — close enough to trap single-start optimizers.

**True `D(c)`.** Has a shoulder at `c ≈ 2 M`; this gives the
multi-modality enough room to live without making either basin trivially
distinguishable from `c_data` alone.

**Designed failure mode caught.**
- Single-start BFGS / Adam from any of: `t⁺⁰ ≈ 0.3` initial, `t⁺⁰ ≈ 0` initial, "literature-prior" initial — lands in the positive-basin and fails checks #2, #4, #5 in the high-c half.
- Agents that *try* multi-start but with too few seeds (≤ 3) may still miss the true basin a substantial fraction of the time, contributing to the 10–20 % solve-rate band.

**Expected regime labels.** Mix of `NE_deviates` and `NE_wrong_sign` at
the high-c end. Pinned after parameter lock-in.

---

## Discriminator summary

| Failure mode | Case 1 | Case 2 | Case 3 | Case 4 |
| --- | :-: | :-: | :-: | :-: |
| Wrong output schema | ✗ | ✗ | ✗ | ✗ |
| Lab-frame Nernst-Planck | ✓ (passes) | ✗ (fails #2/#5) | ✗ (fails #2/#3/#5) | ✗ (fails #5) |
| Single-mode `t⁺⁰` ansatz | ✓ | ~ | ✗ (sign-flip tail) | ✗ (non-monotonic) |
| Single-start optimization | ✓ | ✓ | ✓ | ✗ (basin trap) |
| Correct moving-frame + multi-start | ✓ | ✓ | ✓ | ✓ |

A passing submission needs all four ✓ in the bottom row.

## Open

- **Pin parameter functions** once ADR-0005 tolerance calibration is run.
  Record final coefficients, noise σ, and reference-solution margins in
  `docs/progress/key-facts.md`.
- **Sanity-check basin separation in Case 4** by running multi-start
  BFGS from a grid of initial conditions and verifying the wrong-basin
  hit rate is high enough to limit pure single-start agents to the
  10–20 % solve-rate band.
- **Confirm noise model** is rich enough that a literature-value lookup
  (Pesko 2017 D(c), public DiffEC `t⁺⁰` polynomial) fails check #1 / #2
  by at least 2× the threshold.
