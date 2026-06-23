# Mass transport in a concentrated electrolyte: problem specification

You are given operando profiles of salt concentration `c(x, t)` and solvent velocity `v(x, t)` measured in a symmetric cell under potentiostatic polarization. Your task is to recover the concentration-dependent salt diffusivity `D(c)` and cation transference number with respect to solvent motion `t⁺⁰(c)` that produced these profiles under Newman's concentrated-solution theory. This is an inverse problem: forward simulation maps `(D(c), t⁺⁰(c))` to predicted `c_sim(x, t)` and `v_sim(x, t)`; you must invert.

## 1. Geometry and inputs

Each case directory `cases/case_{1..4}/` contains:

- `data.h5` — HDF5 file with datasets:
  - `c_data[Nt, Nx]`: salt concentration in mol/L, on a uniform 1D spatial grid of `Nx ≈ 100` points spanning `x ∈ [0, L]`, sampled at `Nt ≈ 50` time points spanning `t ∈ [0, T]`
  - `v_data[Nt, Nx]`: solvent velocity in nm/s on the same grid
  - `x[Nx]`: spatial coordinates (m)
  - `t[Nt]`: time coordinates (s)
- `params.json` — case-specific parameters:
  - `L`: cell length (m)
  - `T`: total polarization time (s)
  - `i_app(t)`: applied current density profile (A/m²), provided as `[t_k, i_k]` pairs interpolated linearly
  - `c0`: solvent concentration in the pure-solvent limit (mol/L)
  - `V_bar`: salt partial molar volume (m³/mol)
  - `T_temp`: temperature (K)
  - `c_grid[50]`: the canonical concentration grid (mol/L) at which you must report D, t⁺⁰, t⁺⁰_NE, and regime
  - `flux_samples[10][2]`: 10 `(x_k, t_k)` coordinates at which to report the flux decomposition

The data is anchored to realistic operando measurements (Steinrück et al., Energy Environ. Sci. 2020), but the underlying transport-property functions and noise seeds are oracle-generated and held out.

## 2. Governing physics

The salt concentration and solvent velocity evolve in the moving electrolyte frame according to Newman's concentrated-solution theory (Chen et al., ACS Energy Letters 2026, eqs 7-8):

```
∂c/∂t  =  ∂/∂x [ D(c) (1 − d ln c₀ / d ln c) ∂c/∂x  −  t⁺⁰(c) i/F  −  c v₀ ]

∂v₀/∂x =  V̄ ∂/∂x [ D(c) (1 − d ln c₀ / d ln c) ∂c/∂x  −  t⁺⁰(c) i/F  −  c v₀ ]
```

where `F` is Faraday's constant. The factor `(1 − d ln c₀ / d ln c)` accounts for solvent-concentration variation; it is supplied per case in `params.json` as a tabulated function of c.

**Boundary conditions**: At the electrolyte/electrode interfaces (x = 0 and x = L), the cation flux equals i_app(t) / F (Li deposition at one electrode, stripping at the other). The solvent velocity at the moving Li interface is set by the volumetric balance of deposition/stripping.

**Initial conditions**: Uniform salt concentration c(x, 0) = c_init (provided per case), zero solvent velocity.

## 3. What you must produce

For each case, write `results/case_X/transport.json` with the following schema. All quantities are floats unless noted.

```json
{
  "case_id": "case_1",
  "c_grid": [0.5, 0.55, ..., 3.0],          // copy of params.json c_grid
  "D":           [...],                      // 50 values, D(c_grid[i]), units m²/s
  "t_plus_0":    [...],                      // 50 values, t⁺⁰(c_grid[i])
  "t_plus_0_NE": [...],                      // 50 values, t⁺⁰_NE(c_grid[i])
  "regime":      [...],                      // 50 strings, see Section 4
  "v_pred": [[...], [...], ...],             // [Nt][Nx] velocity prediction
  "flux_decomposition": [
    {"x": 0.001, "t": 600.0,
     "J_diff": ..., "J_mig": ..., "J_conv": ...},
    ...                                       // 10 entries, one per flux sampling point
  ]
}
```

### 3.1 Canonical parameterization

You report `D(c)` and `t⁺⁰(c)` only at the 50 c_grid points. Linear interpolation between grid points is the evaluation rule used by the verifier. You may internally parameterize however you wish (free per-point values, polynomial, spline, neural network, ...); only the values on c_grid are evaluated.

### 3.2 The Nernst-Einstein-equivalent transference number

`t⁺⁰_NE(c)` is the transference number that *would* be inferred from the same operando data under classical Nernst-Planck theory — that is, ignoring solvent motion entirely. Concretely:

```
t⁺⁰_NE(c) = (transference number that minimizes ||c_sim − c_data|| under the model
             ∂c/∂t = ∂/∂x [ D(c) ∂c/∂x − t⁺⁰_NE(c) i/F ],  v₀ ≡ 0 )
```

You compute this by running a second inversion in the laboratory frame with `v₀` forced to zero. The contrast between `t⁺⁰` (full theory) and `t⁺⁰_NE` (no solvent motion) is the central physical quantity of the task: in the dilute limit they agree; at high concentration they can have opposite signs.

### 3.3 Velocity prediction

`v_pred[i][j]` is your forward-simulated solvent velocity at `(x[j], t[i])` from your recovered `D(c)` and `t⁺⁰(c)`. The verifier compares this to `v_data` as a physics-consistency check.

### 3.4 Flux decomposition

At each of the 10 sampling points `(x_k, t_k)` listed in `params.json`, compute the cation-flux components (units: mol/(m²·s)):

```
J_diff(x, t) = −D(c) (1 − d ln c₀ / d ln c) (∂c/∂x)
J_mig (x, t) =  t⁺⁰(c) i(t) / F
J_conv(x, t) =  c v₀
```

All three are evaluated using your recovered `D(c)` and `t⁺⁰(c)` and your simulated `c(x, t)`, `v₀(x, t)` at the requested `(x_k, t_k)`. The total cation flux is the sum of the three.

## 4. Regime classification rule

For each c in c_grid, classify the regime as:

| Condition                                                        | Label              |
| ---------------------------------------------------------------- | ------------------ |
| sign(t⁺⁰(c)) ≠ sign(t⁺⁰_NE(c))                                  | `NE_wrong_sign`    |
| sign(t⁺⁰(c)) = sign(t⁺⁰_NE(c)) and |t⁺⁰(c) − t⁺⁰_NE(c)| ≥ 0.05  | `NE_deviates`      |
| |t⁺⁰(c) − t⁺⁰_NE(c)| < 0.05                                     | `NE_valid`         |

Compute this mechanically from your reported t⁺⁰ and t⁺⁰_NE.

## 5. How you are evaluated

The verifier (pytest, deterministic, no LLM judges) checks five quantities per case:

1. **Transport parameters** at all 50 c_grid points:
   - `|D_you(c) − D_oracle(c)| / D_oracle(c) ≤ 0.10`
   - `|t⁺⁰_you(c) − t⁺⁰_oracle(c)| ≤ 0.05`
2. **Regime classification** at all 50 c_grid points: exact match to oracle.
3. **Velocity prediction RMSE**: `||v_pred − v_data||₂ / max|v_data| ≤ 0.15`.
4. **Flux decomposition** at all 10 sampling points: `|J_X_you − J_X_oracle| / |J_total_oracle| ≤ 0.15` for each X ∈ {diff, mig, conv}.
5. **Self-consistency**: the verifier runs its own moving-frame PDE solver from your reported D(c), t⁺⁰(c) and checks that the resulting velocity field satisfies #3. This catches cases where the reported parameters are plausible but were not derived by actually inverting the data.

To pass, all five checks must succeed for all four cases.

## 6. Compute budget and methodology

You have a budget of 5-10 minutes per case on 4-8 CPU cores. Total across 4 cases: under 1 hour. No GPU is required.

You may use any numerical approach: differentiable JAX or PyTorch simulators, gradient-free optimizers, surrogate models, neural networks, hand-tuned procedures, anything. The benchmark does not prescribe a method. The compute budget and case design are calibrated so that approaches that are too slow (e.g., gradient-free optimization on the full moving-frame PDE) or too brittle (e.g., single-start fits on the multi-modal case) are unlikely to pass.

## 7. References

- Chen, H. et al. *Differentiable Electrochemistry.* ACS Energy Letters 2026, 11, 2005-2018. (Eqs 7-8 are the governing PDEs.)
- Steinrück, H.-G. et al. *Concentration and velocity profiles in a polymeric lithium-ion battery electrolyte.* Energy Environ. Sci. 2020, 13, 4312-4321. (Operando measurement methodology.)
- BattModels/DiffEC repository: methodological background.