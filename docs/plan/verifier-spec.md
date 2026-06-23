# Verifier Spec

> The verifier is a deterministic pytest. It loads the agent's
> `/root/results/case_X/transport.json` (transferred from the agent
> container per the `artifacts` list in `task.toml`), the held-out
> `/tests/oracle_truth/case_X/truth.npz`, and (for check #6) re-invokes
> the oracle solver under `/tests/oracle/`. No LLM judges, no tolerance
> ambiguity.
>
> The verifier runs in its own container (`environment_mode = "separate"`),
> built from `tasks/.../tests/Dockerfile`. See `harbor-task-format.md`
> for the wrapper contract.

## Wrapper (`test.sh`)

Per the upstream TB-Science convention, `tests/test.sh` is a thin shell
wrapper that runs pytest and translates the exit code into Harbor's
reward signal:

```bash
#!/bin/bash
mkdir -p /logs/verifier
pytest /tests/test_outputs.py \
    --ctrf /logs/verifier/pytest-ctrf.json \
    > /logs/verifier/pytest.log 2>&1
status=$?
if [ "$status" -eq 0 ]; then echo 1 > /logs/verifier/reward.txt
else echo 0 > /logs/verifier/reward.txt
fi
exit $status
```

All-or-nothing reward (single integer, `0` or `1`) per the upstream
example pattern. Per-case / per-check breakdown still goes into the
CTRF JSON and `pytest.log` for debugging — Harbor surfaces those to
the reviewer.

## 0. Schema and sanity (`tests/test_schema.py`)

Run first. If schema fails, the rest are not informative.

Per case:
- Required top-level keys present: `case_id`, `c_grid`, `D`, `t_plus_0`, `t_plus_0_NE`, `regime`, `v_pred`, `flux_decomposition`.
- `c_grid`, `D`, `t_plus_0`, `t_plus_0_NE`, `regime` are length-50.
- `v_pred` shape is `(Nt, Nx)` matching `data.h5`.
- `flux_decomposition` is length-10; each entry has `x`, `t`, `J_diff`, `J_mig`, `J_conv`.
- `regime` entries ∈ `{"NE_valid", "NE_deviates", "NE_wrong_sign"}`.
- No NaN, no inf, no `null`. All numeric fields are JSON numbers.
- `case_id` matches the expected case label.
- `c_grid` matches `params.json` `c_grid` byte-for-byte (no resampling tricks).

Failure on schema → all downstream checks are reported as "blocked", not "failed".

## 1. Diffusion-coefficient check (`tests/test_parameters.py::test_D`)

For each of the 50 c_grid points:

```
|D_agent[i] − D_oracle[i]| / D_oracle[i] ≤ 0.10
```

Report: index of worst point, achieved relative error there, summary of distribution. Threshold: 0.10 per the proposal.

**Feasibility:** ADR-0005 — the reference solution must clear this with
≥ 50 % margin (i.e., worst-point relative error ≤ 0.05). Recorded in
`docs/progress/key-facts.md`.

## 2. Transference-number check (`tests/test_parameters.py::test_tp0`)

For each of the 50 c_grid points:

```
|t⁺⁰_agent[i] − t⁺⁰_oracle[i]| ≤ 0.05
```

Absolute tolerance, not relative — `t⁺⁰` straddles zero in cases 3 and 4
so relative tolerance is undefined. Same 50 %-margin precondition.

## 3. Regime classification check (`tests/test_regime.py`)

For each c_grid point, the agent's `regime[i]` must equal the oracle's
regime label computed mechanically per `formalism.md` §4:

```
NE_wrong_sign  if sign(t⁺⁰[i]) ≠ sign(t⁺⁰_NE[i])
NE_deviates    if signs agree and |t⁺⁰[i] − t⁺⁰_NE[i]| ≥ 0.05
NE_valid       if |t⁺⁰[i] − t⁺⁰_NE[i]| < 0.05
```

The oracle's `regime_oracle[50]` is computed from
`(t⁺⁰_oracle, t⁺⁰_NE_oracle)` once at case-generation time and stored in
`truth.npz`. The verifier checks **exact match** at all 50 points per
case (200 across the 4 cases).

This is the categorical-pattern discriminator the proposal calls out:
under any reasonable null model, 200 ternary labels are essentially
impossible to satisfy by luck.

## 4. Velocity-field RMSE (`tests/test_velocity.py`)

```
||v_pred − v_data||₂ / max|v_data| ≤ 0.15
```

`v_pred` is the agent's forward-simulated solvent velocity on the same
`(x, t)` grid as `v_data` (which is `data.h5`'s `v_data`, i.e. noisy
oracle output). RMS L₂ norm over all `Nt × Nx` points.

`max|v_data|` uses the bundled noisy `v_data`; this is a fixed scalar
per case and is recorded in `truth.npz` for the verifier to read
directly (no recomputation that could drift).

## 5. Flux decomposition (`tests/test_flux.py`)

For each of the 10 `(x_k, t_k)` sampling points, for each
`X ∈ {diff, mig, conv}`:

```
|J_X_agent − J_X_oracle| / |J_total_oracle| ≤ 0.15
```

`J_X_oracle` and `J_total_oracle` are computed from the oracle's
`(D, t⁺⁰, c, v₀)` at the same `(x_k, t_k)` using `oracle/flux.py`
(ADR-0003), and cached in `truth.npz`.

Note the normalization: each component is normalized by `|J_total|`,
not by `|J_X|`. This makes the check robust when one component is near
zero (no division-by-tiny blowup) but still penalizes wrong attribution
when one mechanism dominates the total. The proposal's reasoning is
that an under-predicted convective contribution at high salt
concentration is exactly the "right answer wrong physics" failure mode
to catch.

## 6. Self-consistency / anti-cheat (`tests/test_self_consistency.py`)

Pipeline:

1. Load `D_agent[50]`, `t⁺⁰_agent[50]` from `transport.json`.
2. Build piecewise-linear interpolants on `c_grid`.
3. Re-run `oracle/solver.py` from the case's `params.json` BC/IC and
   the agent's interpolated `(D, t⁺⁰)`.
4. Extract `v₀_sim_from_agent_params`.
5. Compare to `v_data` from `data.h5` with the same tolerance as check #4:

```
||v₀_sim − v_data||₂ / max|v_data| ≤ 0.15
```

This catches:
- Agents who report parameters lifted from literature without doing the inversion (their parameters won't reproduce the case's actual `v_data`).
- Agents whose physics is wrong but whose reported `v_pred` happens to match `v_data` by virtue of being fit *directly* to it.
- Right-answer-wrong-physics agents whose `D` and `t⁺⁰` happen to land near the oracle but whose underlying inversion was lab-frame.

The verifier runs the **same** `oracle/solver.py` that generated the
case; the agent's reported parameters either reproduce `v_data` or they
don't. No room for argument.

## 7. Edge cases the verifier handles explicitly

- **Missing `results/case_X/transport.json`:** that case's checks are reported as "no submission" (distinct from "failed").
- **Truncated / malformed JSON:** schema check fails; rest blocked.
- **Agent reports `D = 0` or `D` very close to zero:** check #1 division-by-zero guard reports a structured error, doesn't raise.
- **`t⁺⁰_NE` exactly zero at a point:** sign comparison is well-defined (sign(0) := the oracle's sign at that point, so the agent doesn't get a free pass on the discontinuity).
- **`v_pred` mass per case > some sane RAM threshold:** capped at the bundled `(Nt, Nx)` shape; over-large arrays caught by the schema check.
- **Out-of-distribution c_grid points:** the verifier interpolates the agent's `D` and `t⁺⁰` linearly between c_grid points (the canonical evaluation rule per `formalism.md` §3.1). It never extrapolates; if a case's PDE evaluates `D(c)` at a c outside `[c_min, c_max]` of the c_grid, that's a case-design bug, fixed in `case_gen/configs/`.

## 8. Reporting

The verifier's pytest output is intentionally verbose:

- For continuous checks: worst-point index, achieved value, threshold, margin.
- For the regime check: count and indices of mis-classified points, broken out by `(oracle_label → agent_label)` confusion matrix.
- For self-consistency: achieved RMSE vs threshold, alongside #4's achieved RMSE for direct comparison.

This is also written to `results/_verifier_report.json` so the
frontier-agent pilot can aggregate failure modes per agent.

## 9. Tolerance feasibility audit

ADR-0005 requires the reference solution to clear each check with
≥ 50 % margin. The audit runs as part of the verifier in `--audit` mode:

```bash
uv run pytest tb_sci_task/tests/ -v --audit
```

In audit mode, every check additionally asserts the margin condition.
This catches the case where the reference solution barely passes — a
warning sign that an honest frontier agent will fail more often than
the 10–20 % target band.

## 10. Pytest configuration notes

- `pytest.ini` pins `-p no:cacheprovider` to avoid the verifier picking up stale state on rerun.
- `pytest-xdist` is supported; the four cases run in parallel.
- The verifier is `JAX_PLATFORMS=cpu` (matches the agent's environment).
- `conftest.py` exposes fixtures: `case_id`, `transport_json`, `truth_npz`, `params_json`, `data_h5`.
- The agent's output directory is `/root/results/` inside Harbor's container; for local iteration the verifier honors `RESULTS_DIR` env var (see `build-and-run.md` §5).

## 11. Reward signal contract

The verifier writes exactly one of these to `/logs/verifier/`:

- `reward.txt` — single integer `1` (all 20 tests passed) or `0` (any failed).
- `pytest-ctrf.json` — full CTRF report, per-test pass/fail, used by Harbor and reviewers for debugging.

The `test.sh` wrapper above is the source of truth for how exit status
becomes the reward. Do not add `reward.json` (some Harbor tasks use it
for partial-credit floats; the TB-Science example uses `reward.txt`,
and the proposal commits to all-or-nothing — "to pass, all five checks
must succeed for all four cases").
