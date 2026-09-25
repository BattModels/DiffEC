# Case Design

> Final calibrated values pinned 2026-06-26; **case_4 redesigned
> 2026-09-09 per ADR-0013** (v0.2 hardening: `t⁺⁰` sign crossing inside
> the graded band + truth-side regime mask). All four cases pass 32/32
> verifier checks with ≥ 2.7× margin on every continuous check, exact
> regime match at every graded point (192 of 200 labels graded), and
> ADR-0004 anti-cheat against the published Steinrück 2020 / DiffEC
> paper fit. Calibration deltas vs the initial design are recorded in
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
- **Sensitivity contract (ADR-0015, formalism §3.4):** `c_knots` = 12
  uniform knots over the noisy observed range `[min c_data, max c_data]`;
  `t_qoi` = last bundled time sample inside the current plateau
  (`t[44] = 987.7 s` for all four cases). The graded QoI is the
  polarization `Q = c(x[Nx−1], t_qoi) − c(x[0], t_qoi)`; the noiseless
  `Q_sim` must match the measured `Q_data` to 5 % at generation time
  (verifier gate 10 %). Per case (`c_knots` range, `Q_sim`, `Q_data`,
  mol/L): case 1 [0.435, 0.772], −0.323, −0.314; case 2 [0.705, 1.799],
  −1.071, −1.052; case 3 [0.630, 5.003], −4.308, −4.309; case 4
  [2.033, 4.391], −2.339, −2.324. `data.h5` is unchanged by ADR-0015;
  only `params.json`, `formalism.md`, `truth.npz` gained fields.

## Pinned case parameters (2026-06-26)

| Parameter | case_1 (NE-valid) | case_2 (NE-deviates) | case_3 (NE-wrong-sign) | case_4 (NE-transition) |
| --- | --- | --- | --- | --- |
| seed | 12345 | 23456 | 34567 | 45678 |
| `c_init` (mol/L) | 0.6 | 1.25 | 2.5 | 3.0 |
| `c_grid` (mol/L) | [0.45, 0.75] | [1.13, 1.49] | [2.42, 2.69] | [2.93, 3.30] |
| peak current (A/m²) | 4 | 16 | 48 | 32 |
| `V̄` (m³/mol) | 5×10⁻⁵ (override) | 2.5×10⁻⁴ (override) | ~1.42×10⁻⁴ (rho-derived) | ~1.32×10⁻⁴ (rho-derived) |
| σ_c (mol/L) | 0.006 (~1 %) | 0.012 (~1 %) | 0.015 (~0.6 %) | 0.018 (~0.6 %) |
| `invert_ne.init_tp0` | 0.10 | 0.30 | 0.30 | 0.30 |
| `invert_ne.lambda_reg` | 1×10⁻³ | 1×10⁻³ | 1×10⁻³ | 1×10⁻³ |
| Realized true `t⁺⁰` range | [0.096, 0.104] | [0.215, 0.263] | [-0.408, -0.346] | [-0.280, +0.090], crossing c*≈3.02 |
| Realized lab-frame `t⁺⁰_NE` range | [0.117, 0.119] | [0.365, 0.457] | [0.024, 0.051] | [0.153, 0.375] |
| Realized regime distribution | 50× NE_valid | 50× NE_deviates | 50× NE_wrong_sign | 12× NE_deviates + 38× NE_wrong_sign |
| Regime-graded points (ADR-0013 mask) | 50/50 | 50/50 | 50/50 | 42/50 (8 masked at the crossing) |

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

## Case 4 — NE-transition (sign crossing inside the graded band)

> Redesigned 2026-09-09 per **ADR-0013** (v0.2 hardening). The v1
> design ("NE-wrong-sign at high c", `t⁺⁰` ∈ [-0.20, -0.18] across
> c_grid) duplicated case_3's discriminator and let agents collect all
> 50 labels for free once the case-level regime was identified —
> Opus-5 passed the task 3/3 in reviewer trials. v1 history (including
> the dropped basin-trap intent, `docs/session.md` 2026-06-25) is
> preserved in git and in the ADR.

**Regime intent.** The true `t⁺⁰(c)` crosses zero *inside* c_grid at
c* ≈ 3.02 mol/L (between grid points 16 and 17) while the emergent
lab-frame `t⁺⁰_NE` stays robustly positive ([0.153, 0.375]) across the
band. The 50 labels therefore split into an `NE_deviates` block
(idx 0–11, `t⁺⁰ > 0`, gap to `t⁺⁰_NE` ≥ 0.25) and an `NE_wrong_sign`
block (idx 12–49). Recovering the block boundary requires locating the
sign crossing to ~±0.03 mol/L — pointwise `t⁺⁰` accuracy ~0.03 near
c*, deliberately **sharper than check #2's 0.05 gate**. The NE
inversion and the local `t⁺⁰` shape are load-bearing; labels are no
longer implied by the case's overall regime.

**Label-robustness mask (ADR-0013).** The verifier grades labels only
where `|t⁺⁰_oracle| ≥ 0.03` (`regime_graded` in `truth.npz`; rule
quoted abstractly in `formalism.md` §4 without revealing which points
are masked). For this case 8 points around the crossing are masked →
42/50 graded. δ = 0.03 ≈ 2.7× the reference's worst-point `t⁺⁰` error,
so the reference's labels are sign-safe with margin. Cases 1–3 are
unaffected (their min `|t⁺⁰|` ≥ 0.085).

**Final design (`case_4.yaml`).** `c_init = 3.0` (just below c*),
c_grid [2.93, 3.30] (c_max extended from 3.20 on 2026-09-09 — see the
mock-trial note below), peak current 32 A/m², `V̄` rho-derived
(~1.32×10⁻⁴ m³/mol). Unchanged from v1 except the `t⁺⁰` table.

True material functions:

```
c (mol/L):   0.0    0.5    1.0    1.5    2.0    2.5    2.85   3.35   3.6    4.0
t⁺⁰:         0.42   0.38   0.33   0.28   0.24   0.20   0.17  -0.33  -0.38  -0.42
D (cm²/s):   1.5e-6 1.2e-6 9.0e-7 6.5e-7 4.5e-7 3.8e-7 3.6e-7(at 3.0) 3.0e-7(3.5) 2.5e-7(4.0)
```

(`D` keeps the v1 table on knots 0.0–4.0.) `t⁺⁰(c)` declines gently to
a knee at c = 2.85 (below the graded band), then drops linearly at
slope −1.0 /(mol/L) through zero at c* = 3.02, leveling off above
3.35. On c_grid the truth runs from +0.090 down to −0.280. The steep
slope keeps the masked band narrow (|t⁺⁰| < 0.03 spans ~0.06 mol/L
≈ 8 grid points at the widened Δc).

**Designed failure modes caught.**

- **Lab-frame agents:** recover `t⁺⁰_NE` ∈ [+0.15, +0.38] instead of
  the true [+0.09, −0.28] → check #2 worst = 0.43 = 8.7× threshold,
  check #6 RMSE/max = 0.252 = 1.7× threshold, and **all 42 graded
  labels wrong**.
- **Case-constant labelers** (agents that classify the whole case from
  its dominant regime, without a pointwise NE comparison): at best 30
  of 42 graded labels — check #3 fails.
- **Over-smoothed inversions:** a strong smoothness prior biases the
  recovered slope near c* (the reference's own λ = 10⁻³ rung shrinks it
  ~25 %, shifting the recovered crossing and D by up to 12 % at the
  high-c tail). Passing requires letting the data set the local slope —
  the reference does this via data-driven λ selection (see below).

**Expected regime labels.** 12× NE_deviates + 38× NE_wrong_sign
(42 graded: 12 NE_deviates + 30 NE_wrong_sign).

**Mock-trial evidence for the 3.30 extension (2026-09-09).** A cold
containerized Opus-5-xhigh mock trial passed the [2.93, 3.20] version
(reward 1.0, 27 min) with its case_4 t⁺⁰ error growing monotonically
into the sparse tail — 0.037 at 3.20 (1.35× margin), the classic
parsimony-prior slope shrinkage; near the crossing it was ~0.005 and it
located c* to 0.001. Extending c_max to 3.30 puts that same solution
class at ~0.052 > 0.05 at the new endpoint (fail), while the reference
(data-driven λ selection) sits at 0.021 there (2.4× margin). Anti-cheat
re-verified on the wider band (lit min-distance 0.12; cheat table
above).

---

## Discriminator matrix (post-calibration)

| Failure mode | case_1 | case_2 | case_3 | case_4 |
| --- | :-: | :-: | :-: | :-: |
| Wrong output schema (NaN, missing fields) | ✗ | ✗ | ✗ | ✗ |
| Lab-frame Nernst-Planck (tp⁺⁰ = tp⁺⁰_NE) | ✓ pass | ✗ #2/#6 fail | ✗ #2/#6 fail | ✗ #2/#3/#6 fail |
| Constant positive t⁺⁰ (literature prior) | ~depends | ✗ #2 fail | ✗ #2 cat. fail | ✗ #2 cat. fail |
| Literature-lookup (Steinrück fit) | ✗ #2 fail (margin 0.15) | ✗ #2 fail (margin 0.11) | ✗ #2 fail (margin 0.16) | ✗ #2 fail (margin 0.19) |
| Case-constant regime labels (no pointwise NE comparison) | ✓ pass | ✓ pass | ✓ pass | ✗ #3 fail (≥11 of 39 graded wrong) |
| Over-smoothed t⁺⁰ inversion (strong global prior) | ✓ pass | ✓ pass | ✓ pass | ✗ #1/#2 fail near c* and at the high-c tail |
| Honest moving-frame + joint c+v fit | ✓ | ✓ | ✓ | ✓ |

(Literature-lookup "margin" = the 2026-09-09 `scripts/litvalue_distance.py`
minimum pointwise `|Δt⁺⁰|` vs the published fit — every c_grid point sits
at least that far away, all beyond the 0.05 verifier gate; the audit's
pass condition — max distance ≥ 0.10 — holds with ≥ 2× headroom
everywhere.)

A passing submission needs ✓ in the bottom row across all 4 cases.

## Reference-solution margins (final)

Recorded in `docs/progress/key-facts.md`. All cases ≥ 50 % margin on
continuous checks; all cases 50/50 regime match.

## Lab-frame anti-cheat sanity (final)

Computed by running the held-out `oracle/solver.simulate` in moving
frame with `D = D_oracle` and `t⁺⁰ = t⁺⁰_NE_oracle` (the "lab-frame
agent's submission"), then comparing the resulting `v₀` to `v_data`:

| Case | honest #6 | lab-frame cheat #6 | lab-frame cheat #2 worst | cheat graded-label mismatches |
| --- | --- | --- | --- | --- |
| case_1 | 0.042 | 0.045 (pass — by design) | 0.021 (pass — by design) | 0/50 (pass — by design) |
| case_2 | 0.042 | **0.209 (catch)** | **0.214 = 4× threshold** | **50/50** |
| case_3 | 0.057 | **0.225 (catch)** | **0.432 = 9× threshold** | **50/50** |
| case_4 | 0.047 | **0.252 (catch)** | **0.433 = 8.7× threshold** | **42/42** |

(Re-measured 2026-09-09 after the ADR-0013 case_4 redesign; the cheat's
labels are all `NE_valid` since its `t⁺⁰` equals its `t⁺⁰_NE`.)

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
   **ADR-0013 amendment (2026-09-09):** resolution (i) is deliberately
   reversed for case_4 — its c_grid now spans the *true* `t⁺⁰` sign
   crossing, and label determinism near the crossing is handled by the
   truth-side `regime_graded` mask (`|t⁺⁰_oracle| < 0.03` ungraded)
   instead of by narrowing the band. Note the flip risk being masked is
   different from 2026-06: the fragile quantity here is the agent's
   `t⁺⁰` sign near c*, not the `t⁺⁰_NE` crossing (which stays far
   outside the band; `t⁺⁰_NE` ∈ [0.22, 0.38] on c_grid).
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
