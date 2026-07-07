# Case Design

> Final calibrated values pinned 2026-06-26. All four cases pass 28/28
> verifier checks with ≥ 50 % margin (ADR-0005), 50/50 exact regime
> match (ADR-0005), and ADR-0004 anti-cheat against the published
> Steinrück 2020 / DiffEC paper fit. The original design intent — four
> failure-mode discriminators across `NE_valid`, `NE_deviates`,
> `NE_wrong_sign`, and the basin-trap idea — is preserved where
> achievable. Calibration deltas vs the initial design are recorded in
> "Calibration notes" at the bottom.

## Shared template

All cases use the same cell geometry, time horizon, and noise model
family:

- **Geometry:** symmetric Li | electrolyte | Li, `L = 3 mm`, 1-D,
  `Nx = 100` uniform finite-volume cells.
- **Time:** `T = 1100 s`, internal forward-Euler step `dt = 0.1 s`,
  `Nt = 50` reporting times (uniformly subsampled including endpoints).
- **Current schedule `i(t)`:** ramp 0 → peak over 100 s, hold for 900 s,
  ramp peak → 0 over 100 s. Peak per case in the table below.
- **`(1 − d ln c₀ / d ln c)` factor:** tabulated per case at 201
  points spanning `[0, max(2.5 · c_max, 2 · c_init)] mol/L` from the
  per-case `rho(c) = a + b·c + d·c²` polynomial. Same `(a, b, d)`
  across all 4 cases (the published PEO-LiTFSI density polynomial).
- **`V̄`:** scalar per case. Default = the `rho(c)`-derived value
  evaluated at `c_avg = mean(c_grid)`. Cases 1 and 2 use a
  `V_bar_si` override to tune the strength of convective coupling
  (see per-case notes).
- **c_grid:** 50 uniformly-spaced points spanning each case's
  `[c_min, c_max]`. Final narrowed bounds per case below — narrower
  than the originally-proposed design ranges because the lessons
  from calibration showed that c_grid points outside the
  data-supported (or label-stable) region are unrecoverable and
  noisily-labeled (see "Calibration notes").
- **flux_samples:** 10 deterministic `(x_k, t_k)` coordinates drawn
  uniformly from `[0.2 L, 0.8 L] × [0.3 T, 0.9 T]` using
  `numpy.random.default_rng(seed)`.
- **Noise model:**
  - `c_data ← c_sim + N(0, σ_c)`, additive Gaussian, σ per case.
  - `v_data ← v_sim + N(0, σ_v)` with `σ_v = 0.05 · max|v_sim|`
    (5 % of peak).
- **Determinism:** `seed` per case feeds `numpy.random.default_rng`.
  Re-running case generation produces byte-identical `data.h5`,
  `params.json`, `formalism.md`, `truth.npz`.

## Pinned case parameters (2026-06-26)

| Parameter | case_1 (NE-valid) | case_2 (NE-deviates) | case_3 (NE-wrong-sign) | case_4 (NE-wrong-sign, high c) |
| --- | --- | --- | --- | --- |
| seed | 12345 | 23456 | 34567 | 45678 |
| `c_init` (mol/L) | 0.6 | 1.25 | 2.5 | 3.0 |
| `c_grid` (mol/L) | [0.45, 0.75] | [1.13, 1.49] | [2.42, 2.69] | [2.93, 3.20] |
| peak current (A/m²) | 4 | 16 | 48 | 32 |
| `V̄` (m³/mol) | 5×10⁻⁵ (override) | 2.5×10⁻⁴ (override) | ~1.42×10⁻⁴ (rho-derived) | ~1.32×10⁻⁴ (rho-derived) |
| σ_c (mol/L) | 0.006 (~1 %) | 0.012 (~1 %) | 0.015 (~0.6 %) | 0.018 (~0.6 %) |
| `invert_ne.init_tp0` | 0.10 | 0.30 | 0.30 | 0.30 |
| `invert_ne.lambda_reg` | 1×10⁻³ | 1×10⁻³ | 1×10⁻³ | 1×10⁻³ |
| Realized true `t⁺⁰` range | [0.096, 0.104] | [0.215, 0.263] | [-0.408, -0.346] | [-0.200, -0.180] |
| Realized lab-frame `t⁺⁰_NE` range | [0.117, 0.119] | [0.365, 0.457] | [0.024, 0.051] | [0.208, 0.220] |
| Realized regime distribution | 50× NE_valid | 50× NE_deviates | 50× NE_wrong_sign | 50× NE_wrong_sign |

`t⁺⁰(c)` and `D(c)` tables per case are the YAML truth tables in
`case_gen/configs/case_X.yaml`; pinned values reproduced under each
case below.

## Case 1 — NE-valid (calibration case)

**Regime intent.** Weak convective coupling. `|t⁺⁰ − t⁺⁰_NE| < 0.05`
at every c_grid point. Lab-frame Nernst-Planck agents pass cleanly —
this case validates that the easy regime is achievable end-to-end.

**Final design (`case_1.yaml`).** `c_init = 0.6`, c_grid [0.45, 0.75],
peak current 4 A/m², `V̄_override = 5×10⁻⁵ m³/mol` (~1/3 of the
rho-derived value, to suppress `v₀` enough that the lab-frame NE
inversion lands within 0.05 of true `t⁺⁰` everywhere).

True material functions (interpolated linearly between knots):

```
c (mol/L):   0.0    0.1    0.2    0.4    0.6    0.8    1.0    1.2
t⁺⁰:         0.120  0.115  0.110  0.105  0.100  0.095  0.090  0.085
D (cm²/s):   6.5e-7 6.4e-7 6.0e-7 5.5e-7 5.0e-7 4.5e-7 4.0e-7 3.8e-7
```

`t⁺⁰` is near-flat at 0.10 (small monotone decrease, spread 0.04 across
the whole table) — *not* the originally-proposed 0.30 mean. Shifted
down per ADR-0004: at c = 0.6 the published Steinrück fit gives
`t⁺⁰ ≈ 0.30`, so a 0.30 oracle would have been indistinguishable from
a literature lookup at mid-c_grid. The 0.10 mean is uniformly
≥ 0.10 below the Steinrück line across c_grid.

**Designed failure mode caught.** None — case 1 is the calibration
case. An agent that *fails* case 1 has the wrong output schema, the
wrong physics (NaN-producing), or both.

**Lab-frame anti-cheat sanity.** Lab-frame "cheat" (reporting `t⁺⁰ =
oracle's t⁺⁰_NE`) gives check #6 RMSE/max = 0.045 and check #2 worst
= 0.021 — **both below threshold; the cheat PASSES**. This is by
design: the V̄ override deliberately makes moving-frame and lab-frame
numerically indistinguishable in this case. Cases 2/3/4 do the
moving-frame discrimination.

---

## Case 2 — NE-deviates (moderate convection)

**Regime intent.** Moderate convective coupling. Both moving-frame and
lab-frame inversions produce positive `t⁺⁰`, but with magnitudes that
diverge enough (|gap| ≥ 0.07) that a lab-frame agent fails check #2.

**Final design (`case_2.yaml`).** `c_init = 1.25`, c_grid [1.13, 1.49],
peak current 16 A/m² (4× case_1), `V̄_override = 2.5×10⁻⁴ m³/mol`
(~1.7× rho-derived). The V̄ override pushes the lab-frame `|t⁺⁰ −
t⁺⁰_NE|` gap robustly above 0.07 across c_grid (avoids borderline
regime labels).

True material functions:

```
c (mol/L):   0.0    0.3    0.7    1.0    1.3    1.6    2.0    2.5
t⁺⁰:         0.40   0.37   0.32   0.28   0.24   0.20   0.15   0.12
D (cm²/s):   4.5e-7 4.0e-7 3.4e-7 3.0e-7 2.8e-7 3.0e-7 3.5e-7 4.0e-7
```

`t⁺⁰` is monotone-decreasing positive; `D(c)` is bowl-shaped with
minimum near c ≈ 1.3 (tests the agent's ability to fit non-monotone
material functions).

**Designed failure mode caught.** Lab-frame Nernst-Planck agents:
they recover `t⁺⁰_NE` ∈ [0.37, 0.46] instead of true `t⁺⁰` ∈ [0.22,
0.26] → max |Δt⁺⁰| = 0.21 ≫ 0.05 → fail check #2 catastrophically.
Self-consistency #6 also fails (RMSE/max 0.21 > 0.15).

**Expected regime labels.** All 50 NE_deviates.

---

## Case 3 — NE-wrong-sign (headline case)

**Regime intent.** Reproduce the Steinrück-2020-style negative-`t⁺⁰`
phenomenon at high salt concentration. Lab-frame inversion recovers
positive `t⁺⁰_NE` while the true `t⁺⁰` is deeply negative — opposite
signs → NE_wrong_sign.

**Final design (`case_3.yaml`).** `c_init = 2.5`, c_grid [2.42, 2.69]
(narrowed from initial [2.20, 2.80] — see "Calibration notes"), peak
current 48 A/m², `V̄` rho-derived (~1.42×10⁻⁴ m³/mol).

True material functions:

```
c (mol/L):   0.0    0.5    1.0    1.5    2.0    2.5    2.5(rep)   3.0    3.5
t⁺⁰:         0.28   0.18   0.08  -0.07  -0.22  -0.37    (—)      -0.47  -0.52
D (cm²/s):   1.2e-6 1.0e-6 7.0e-7 5.0e-7 4.0e-7 3.5e-7   (—)     3.0e-7 2.7e-7
```

`t⁺⁰(c)` crosses zero somewhere near c ≈ 1.6 mol/L (outside c_grid —
the crossing is in the truth-table polynomial but the *c_grid* lives
entirely in the deep-negative region). True `t⁺⁰` in c_grid spans
[-0.41, -0.35]. `D(c)` monotonically decreases from ~1.2 ×10⁻⁶ to
~2.7×10⁻⁷ cm²/s.

`t⁺⁰` table shifted -0.12 uniformly from the initial design to clear
ADR-0004: max |Δt⁺⁰_lit| = 0.17 (≥ 2× the verifier threshold).

**Designed failure mode caught.**

- **Lab-frame agents** report `t⁺⁰_NE` ∈ [0.024, 0.051] (positive)
  instead of true `t⁺⁰` ∈ [-0.41, -0.35] (negative) → check #2 worst
  = 0.43 = 9× threshold, check #6 RMSE/max = 0.23 = 1.5× threshold.
  Catastrophic catch.
- **Agents using a single-mode positive parameterization** (e.g.
  the published linear `t⁺⁰` ansatz constrained > 0) fail check #2
  at every c_grid point.

**Expected regime labels.** All 50 NE_wrong_sign.

---

## Case 4 — NE-wrong-sign at high concentration

**Regime intent.** Companion to case_3 at a different concentration
regime. True `t⁺⁰(c)` is deeply negative throughout c_grid
([-0.20, -0.18]); lab-frame agents (which structurally recover
positive `t⁺⁰_NE`) fail catastrophically on check #2
(`|t⁺⁰_lab − t⁺⁰_true|` worst = 0.42, 8× threshold) and check #6
(self-consistency RMSE/max = 0.26, 1.7× threshold). Uses realistic
(rho-derived) `V̄` instead of the case_3 override, so it's the
"cleanest" NE_wrong_sign test of the four.

**Original design intent (dropped).** The case was originally scoped
as a "multi-modal basin trap": a loss landscape with two
well-separated basins where single-start BFGS from a literature-prior
`t⁺⁰ ≈ +0.30` init would land in the wrong (positive) basin a
substantial fraction of the time.

**Why it was renamed.** The reference solver's v-data-weighted joint
inverse reliably finds the correct negative basin from the same
+0.30 init — the `v_data` term breaks the (D, t⁺⁰) degeneracy
before BFGS can settle in a plausible-but-wrong positive basin.
The trap does not materialize with our loss. A real basin trap
would require either a weaker `v_data` weight or a more intricate
non-monotone `t⁺⁰(c)` shape; the reframing keeps the case honest
about what it currently tests. Documented in
`docs/session.md` (2026-06-25 entry) with the empirical result.

**Final design (`case_4.yaml`).** `c_init = 3.0`, c_grid [2.93, 3.20]
(narrowed from initial [2.70, 3.30]), peak current 32 A/m², `V̄`
rho-derived (~1.32×10⁻⁴ m³/mol).

True material functions:

```
c (mol/L):   0.0    0.5    1.0    1.5    2.0    2.5    3.0    3.5    4.0
t⁺⁰:         0.40   0.30   0.20   0.05  -0.05  -0.15  -0.20  -0.15  -0.10
D (cm²/s):   1.5e-6 1.2e-6 9.0e-7 6.5e-7 4.5e-7 3.8e-7 3.6e-7 3.0e-7 2.5e-7
```

`t⁺⁰(c)` is non-monotone (positive at low c, dips to -0.20 around
c ≈ 3.0, recovers slightly at very high c). `D(c)` has a shoulder/
plateau near c = 3.0 — designed so that the 10-knot agent fit
matches the truth's piecewise-linear `D` to within 5 % across the
narrow c_grid.

True `t⁺⁰` in c_grid spans [-0.20, -0.18] — clearly negative for
the NE_wrong_sign discriminator.

**Designed failure mode caught.** Lab-frame agents (same as case_3):
they recover `t⁺⁰_NE` ≈ +0.21 instead of true -0.19 → check #2
worst = 0.42 = 8× threshold, check #6 RMSE/max = 0.26 = 1.7×
threshold.

**Expected regime labels.** All 50 NE_wrong_sign.

---

## Discriminator matrix (post-calibration)

| Failure mode | case_1 | case_2 | case_3 | case_4 |
| --- | :-: | :-: | :-: | :-: |
| Wrong output schema (NaN, missing fields) | ✗ | ✗ | ✗ | ✗ |
| Lab-frame Nernst-Planck (tp⁺⁰ = tp⁺⁰_NE) | ✓ pass | ✗ #2/#6 fail | ✗ #2/#6 fail | ✗ #2/#6 fail |
| Constant positive t⁺⁰ (literature prior) | ~depends | ✗ #2 fail | ✗ #2 cat. fail | ✗ #2 cat. fail |
| Literature-lookup (Steinrück fit) | ✗ #2 fail (margin 0.22) | ✗ #2 fail (margin 0.15) | ✗ #2 fail (margin 0.17) | ✗ #2 fail (margin 0.19) |
| Honest moving-frame + joint c+v fit | ✓ | ✓ | ✓ | ✓ |

A passing submission needs ✓ in the bottom row across all 4 cases.

## Reference-solution margins (final)

Recorded in `docs/progress/key-facts.md`. All cases ≥ 50 % margin on
continuous checks; all cases 50/50 regime match.

## Lab-frame anti-cheat sanity (final)

Computed by running the held-out `oracle/solver.simulate` in moving
frame with `D = D_oracle` and `t⁺⁰ = t⁺⁰_NE_oracle` (the "lab-frame
agent's submission"), then comparing the resulting `v₀` to `v_data`:

| Case | honest #6 | lab-frame cheat #6 | lab-frame cheat #2 worst |
| --- | --- | --- | --- |
| case_1 | 0.042 | 0.045 (pass — by design) | 0.021 (pass — by design) |
| case_2 | 0.042 | **0.209 (catch)** | **0.214 = 4× threshold** |
| case_3 | 0.057 | **0.225 (catch)** | **0.432 = 9× threshold** |
| case_4 | 0.044 | **0.259 (catch)** | **0.416 = 8× threshold** |

## Calibration notes (changes from the initial design)

The original design proposed wider c_grids spanning the regime
transition zones (e.g., case_3's [0.5, 3.0] spanning the t⁺⁰ sign
flip). Calibration showed:

1. **c_grid points outside the data-supported range** are not
   recoverable from the bundled experiment. They just measure the
   agent's extrapolation prior. Resolution: narrow c_grid per case
   to the actually-explored band (`c_min`/`c_max` ≈ c_init ± 1× the
   realized polarization).
2. **c_grid points spanning regime transitions** (where lab-frame
   `t⁺⁰_NE` crosses zero or where `|t⁺⁰ − t⁺⁰_NE|` hovers near
   0.05) produce label flips between oracle and agent BFGS runs.
   Resolution: (i) narrow c_grid to a single-regime band per case;
   (ii) use a cubic-polynomial ansatz for `t⁺⁰_NE` in both
   `oracle/invert_ne.py` and `solution/lab_frame_solver.py` so the
   crossings (when any) land at reproducible c values
   (commit `cc2d45f`).
3. **ADR-0004 anti-cheat** (held-out values ≥ 0.10 from published
   Steinrück fit) required uniform shifts in cases 1, 3, 4 — see
   per-case notes.
4. **V̄ overrides** (cases 1 and 2) — the rho-derived V̄ produced
   convective coupling that was either too weak (case 1, couldn't
   achieve robust NE_valid) or too strong / not strong enough (case
   2, borderline regime labels). Per-case V̄ scalars are a
   physically-defensible knob: real concentrated-electrolyte V̄
   values span 1–3 ×10⁻⁴ m³/mol depending on the salt + solvent
   combination, and the case YAML documents the choice.

Full calibration journey in `docs/session.md` (2026-06-23 through
2026-06-26 entries) and `docs/progress/key-facts.md`.
