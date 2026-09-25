# Design Decisions (ADR log)

> Light ADR format. Each entry: **Context → Decision → Consequences**. When a
> decision is overturned, append a new ADR; never edit history.

---

## ADR-0001 — Adopt Harbor task format as the boundary contract

**Status:** Accepted (2026-06-22); refreshed (2026-06-22)

**Context.** The accepted proposal commits us to submit a PR against
`harbor-framework/terminal-bench-science`. The repo expects a specific task
layout (see `harbor-task-format.md`). Diverging from it would mean a rejected
PR.

**Decision.** Treat the upstream task format as load-bearing. We pinned the
format from upstream `CONTRIBUTING.md`, `harborframework.com/docs/task-format`,
and the example task
`tasks/physical-sciences/chemistry/geometric-pharmacophore-alignment/` on
2026-06-22 (see `harbor-task-format.md`). The working tree mirrors the
upstream layout 1:1 so we can lift it into `tasks/physical-sciences/chemistry/<task-name>/`
unchanged.

Concretely:
- **`schema_version = "1.0"`** in `task.toml`, matching the TB-Science
  example — *not* the `"1.3"` shown on `harborframework.com`, which
  describes a newer Harbor that the TB-Science repo isn't on.
- Two separate containers: `environment/Dockerfile` (agent) and
  `tests/Dockerfile` (verifier). `environment_mode = "separate"`.
- Agent reads from `/root/data/`, writes to `/root/results/`. Output paths
  declared in `artifacts = […]`.
- Verifier signals via `/logs/verifier/reward.txt` (single integer 0 or 1).
- Reference solution lives in-tree under `solution/`, gated by Harbor's
  Oracle agent.

**Consequences.** No "looks right but doesn't match upstream" risk. The
oracle solver package moves under `tests/oracle/` so it ships with the
verifier image; case ground truth ships at `tests/oracle_truth/`. Refresh
`harbor-task-format.md` once more before opening the PR — if upstream
`CONTRIBUTING.md` HEAD has moved, reconcile.

---

## ADR-0002 — Adapt, don't rewrite, the existing DiffEC moving-frame solver

**Status:** Accepted (2026-06-22)

**Context.** `Mass Transport in Concentrated Electrolytes and Benchmarks/solver.py`
is the published, peer-reviewed forward simulator that produced the figures
in Chen et al. 2026. It is JAX-native, `jit`-compatible, and already
implements the moving-frame finite-volume discretization the task requires.

**Decision.** The oracle's forward solver is a parameterized generalization
of `solver.py`, not a fresh implementation. We lift the discretization,
the volume-average / interface stencil, and the sign conventions verbatim;
we generalize:
- `D(c)` from the hard-coded `D_xp/D_fp` table to a configurable functional form,
- `t⁺⁰(c)` from the published linear polynomial to a configurable form,
- the current schedule, BCs, IC concentration, and `(1 − d ln c₀/d ln c)` factor to per-case inputs.

**Consequences.** Faster path to a working oracle; lower risk of "right
equation, wrong sign" bugs that would invalidate the verifier; the case
generation inherits the solver's already-validated numerics. Downside: we
inherit any latent bugs in `solver.py` too — we cross-check by reproducing
the Steinrück 2020 fit from the public DiffEC results as part of the
oracle smoke test (see `build-and-run.md`).

---

## ADR-0003 — Single source of truth for the flux decomposition convention

**Status:** Accepted (2026-06-22)

**Context.** Reviewer feedback (`docs/proposal/review_llm.md`, "Well-Specified")
flagged that two reasonable implementers can produce different but
internally-consistent splits of the cation flux into diffusion / migration /
convection. The verifier's check #5 (flux decomposition) only works if the
agent's convention matches the oracle's. The proposal's flux definitions
(`formalism.md` §3.4) need to be unambiguous.

**Decision.** `oracle/flux.py` defines the canonical decomposition as a
single Python function imported by both the case generator and the verifier.
`formalism.md` (shipped to the agent) restates the three formulas verbatim:
```
J_diff(x, t) = −D(c) (1 − d ln c₀ / d ln c) ∂c/∂x
J_mig (x, t) =  t⁺⁰(c) i(t) / F
J_conv(x, t) =  c v₀
```
with the moving-frame sign of `v₀` defined relative to the deposition
electrode at `x = 0`.

**Consequences.** Removes the specification risk the LLM reviewer flagged.
The agent has a single unambiguous rule to match. Cost: the agent must
read `formalism.md` carefully — which is the task, not a bug.

---

## ADR-0004 — Held-out parameters must be perturbed from any public values

**Status:** Accepted (2026-06-22)

**Context.** The reviewer flagged the risk that an agent with internet
access could lift `D(c)` and `t⁺⁰(c)` from the public DiffEC paper or
the Pesko 2017 / Steinrück 2020 literature and pass the verifier without
performing the inversion. The proposal's anti-cheat check (#6) catches
this only when the lifted parameters fail to reproduce `v_data` —
which happens for lab-frame parameters but not necessarily for
moving-frame literature values.

**Decision.** The 4 held-out `(D_oracle, t⁺⁰_oracle)` functions are each
constructed as a deliberate perturbation of literature values:
- different functional family (e.g., piecewise-cubic vs the published linear `t⁺⁰`),
- different concentration range (perturbed `c_avg`, `c_init`, current schedule),
- different magnitudes (target `|t⁺⁰_agent − t⁺⁰_oracle| ≤ 0.05` — so the held-out value must be at least 0.10 away from any published value).
Each case's `config.yaml` documents the perturbation versus the closest public reference.

**Consequences.** "Look up the answer" no longer suffices. The agent must
run the inversion. We pay a one-time cost: each case's design needs an
explicit lit-value-distance check.

---

## ADR-0005 — Tolerance feasibility is a precondition, not a hope

**Status:** Accepted (2026-06-22)

**Context.** Both reviewers flagged the 10 % relative tolerance on `D(c)`
and the 0.05 absolute tolerance on `t⁺⁰` as stringent — feasibility under
the injected measurement noise is not obvious.

**Decision.** Before locking any case design, the reference solution must
pass the verifier with **at least 50 % margin on the worst point of the
worst case** (e.g., max `|D_ref − D_oracle| / D_oracle ≤ 0.05` when the
threshold is 0.10). Margins per check per case are tabulated in
`docs/progress/key-facts.md` and re-measured on every case-design change.

**Consequences.** Eliminates the "intended solution can't reliably pass"
risk the reviewer flagged. Forces us to calibrate noise levels against
the inversion's information content rather than picking a number out of
the air.

---

## ADR-0006 — Python + JAX only

**Status:** Accepted (2026-06-22)

**Context.** Agents run under a 4–8 CPU / 8–16 GB budget with no GPU.
The published reference is JAX. PyTorch is an option; pure-NumPy/SciPy
is an option. Mixing languages adds packaging friction.

**Decision.** The oracle, case generator, verifier, and reference solution
are pure Python on the NumPy/SciPy/JAX stack. JAX runs CPU-only
(`JAX_PLATFORMS=cpu`) and `jax_enable_x64=True`, matching the published
implementation. No PyTorch, no Julia, no Numba.

**Consequences.** A single, well-known environment (`uv sync` reproduces).
Float64 + JAX gives us auto-diff for the reference solution and exactness
for the verifier. The agent is free to use whatever they want — this
decision constrains *our* code, not theirs.

---

## ADR-0007 — Bundled `.h5` files committed to git, not regenerated on agent start

**Status:** Accepted (2026-06-22; confirmed against upstream large-file policy)

**Context.** Each case is ~10–25 MB of HDF5; total ~50–100 MB. Options:
(a) commit the `.h5` files directly, (b) commit only `case_gen/configs/`
and let the verifier regenerate on first run, (c) git-LFS.

**Decision.** Option (a) — commit the `.h5` files directly under
`tasks/.../environment/data/cases/`. The agent should see the data, not
the generator. Regenerating on first run leaks the existence of `case_gen/`
to a curious agent and risks RNG drift across machines. Git-LFS adds
operational complexity for a 50–100 MB asset.

The upstream CONTRIBUTING.md large-file policy reads: *"Files >100MB
should not be committed — host the data on Hugging Face and use download
scripts in your task."* Our per-file size (~10–25 MB) is well under that
threshold, so direct commit is sanctioned.

**Consequences.** Repo size grows by ~100 MB once, all from
`environment/data/`. The `tests/oracle_truth/` directory adds another
~5–20 MB. If any single file ever exceeds 100 MB, switch to the
HuggingFace + download pattern from CONTRIBUTING.md.

---

## ADR-0009 — Develop in-repo at the upstream PR path

**Status:** Accepted (2026-06-22)

**Context.** With ADR-0001 pinning us to the Harbor task format and
ADR-0007 committing to direct-commit bundles, we have to decide where in
*this* repo the upstream-shipped files live during development.

**Decision.** Develop directly under
`tasks/physical-sciences/chemistry/<task-name>/` inside this repo, with
the exact tree that will land in the upstream PR. Out-of-tree dev tooling
(`case_gen/`, `scripts/`, `docs/`) stays at the repo root and is *not*
copied across in the PR — only `tasks/physical-sciences/chemistry/<task-name>/`
goes upstream.

Rationale: avoids a confusing "rename + move at PR time" step,
keeps the upstream paths visible during development, and makes
`harbor run -p tasks/physical-sciences/chemistry/<task-name>` work in
this repo without any path translation.

**Consequences.** The README / proposal docs / scripts at the repo root
look unrelated to the upstream `tasks/` tree at a glance; we explain
this in `CLAUDE.md` and the eventual PR description. Anyone reading the
upstream PR sees only the task tree, which is the right view for them.


## ADR-0010 — Pin Harbor task `schema_version = "1.0"`

**Status:** Accepted (2026-06-22)

**Context.** The Harbor docs site (`harborframework.com/docs/task-format`)
describes `schema_version = "1.3"` with richer fields (`network_mode`,
`allowed_hosts`, `[environment.tpu]`, `[[environment.mcp_servers]]`).
The TB-Science example
`tasks/physical-sciences/chemistry/geometric-pharmacophore-alignment/task.toml`
uses `schema_version = "1.0"` with the older `allow_internet = true`
form. The two don't validate to the same schema.

**Decision.** Match the TB-Science example (`"1.0"`). The upstream CI
validates against whatever `harbor check` accepts at the TB-Science
repo's HEAD, not against the newer Harbor docs.

**Consequences.** We can't use newer manifest fields (e.g.,
`allowed_hosts`). If the upstream repo upgrades its `harbor` pin
between now and PR submission, we re-pin in this ADR and re-validate
locally with the upgraded `harbor check`.


## ADR-0011 — Lock task name as `concentrated-electrolyte-transport`

**Status:** Accepted (2026-06-23)

**Context.** Harbor task names are kebab-case under
`tasks/<domain>/<field>/<task-name>/`. We brainstormed three candidates
in `harbor-task-format.md`:
- `concentrated-electrolyte-transport`
- `diffec-mass-transport`
- `newman-inversion-from-operando`

**Decision.** `concentrated-electrolyte-transport`.

**Rationale.**
- Most directly describes *what the task is about* (the physics regime),
  not what tools solve it (`diffec-`) or what theory is invoked
  (`newman-`). A future reader scanning `tasks/physical-sciences/chemistry/`
  can guess the content from the directory name alone.
- Avoids embedding a framework name (DiffEC) into a public benchmark.
  The benchmark should outlive any specific reference implementation.
- "Newman inversion" is jargon-precise but narrower than the task: the
  task also covers regime classification, flux decomposition, and the
  NE-equivalent comparison, none of which are "the Newman inversion" by
  themselves.

Full upstream path: `tasks/physical-sciences/chemistry/concentrated-electrolyte-transport/`.

**Consequences.** Final paths are now pinned. Replace the
`<task-name>` placeholder in `harbor-task-format.md`, `architecture.md`,
`build-and-run.md`, and `CLAUDE.md` with the concrete name (or leave
the `$TASK` shorthand pointing here).

---

## ADR-0008 — Frontier-agent pilot before PR submission

**Status:** Accepted (2026-06-22)

**Context.** The proposal commits to a "10–20 % solve rate" empirical
target with Claude Opus 4.7, GPT-5, and Gemini 2.5. The reviewer's
"Solvable" concern (and the difficulty calibration) hinges on this number.

**Decision.** Before opening the PR, run each frontier agent against the
final cases. Record solve rate, time-to-first-pass, and per-check failure
modes in `docs/progress/pilot_run.md`. If solve rate is < 5 % or > 30 %,
reopen the case design.

**Consequences.** Adds 1–2 weeks of pilot time. In exchange we ship a
task whose difficulty is empirically calibrated, not guessed — exactly
what the TB-Science maintainers asked for.

---

## ADR-0012 — Document the velocity operator-split + NE fitting rule in `formalism.md`

**Status:** Accepted (2026-07-30)

**Context.** PR #584's technical reviewer (AllenGrahamHart, via a local
GPT analysis) found that `formalism.md`'s written governing equations do
not match the oracle's forward model, in two places:

1. **Velocity closure.** `formalism.md` §2 wrote the solvent-velocity PDE
   with the same bracket as the concentration PDE — including the `c v₀`
   convection term. Solving that pair self-consistently implies
   `v₀ = V̄[D φ c_x + (1 − t⁺⁰) i/F] / (1 + c V̄)`. But `solver.py:196`
   (and the reference `pde.py:161`) compute the velocity increment from
   `_flux(..., jnp.zeros_like(v0), ...)` — the flux with `v₀ = 0` — giving
   the same expression **without** the `(1 + c V̄)` denominator. This is
   the published DiffEC operator-split convention, inherited verbatim, but
   it was never stated in the task. The gap is `O(c V̄)`: ~2–4 % in the
   dilute cases (passes anyway) but ~27–29 % in case_4 (`c V̄ ≈ 0.4`) —
   far outside the 15 % velocity tolerance. An agent that faithfully
   implements the written equations is thus systematically rejected on the
   concentrated cases; the one passing trial only succeeded by empirically
   reverse-engineering the hidden convention. Notably `formalism.md`'s own
   IC (`v₀ = V̄(1 − t⁺⁰)i/F`, no denominator) already matched the oracle —
   only the second PDE was inconsistent.
2. **NE counterfactual.** §3.2's lab-frame constraint dropped the
   thermodynamic factor `(1 − d ln c₀ / d ln c)`, but `invert_ne.py`
   (via `simulate(lab_frame=True)`) retains it; the ansatz/fitting rule
   (cubic in normalized c + light Tikhonov, λ=1e-3) was also unstated, and
   the verifier grades only the regime label, not the `t⁺⁰_NE` number.

**Decision.** Adopt the reviewer's route 2 — **make the spec match the data,
docs-only, no physics/data/truth change.** (Route 1, regenerating the data
with a self-consistent-denominator solver, was rejected: it contradicts the
project contract that the oracle is "a parameterized generalization of
[DiffEC `solver.py`], not a rewrite," and is far heavier.) Specifically:

- `formalism.md` §2 (all 4 cases): velocity PDE bracket drops `c v₀`; added
  an explicit "Velocity closure" note + the closed form + an explicit
  statement that there is **no** `(1 + c V̄)` denominator.
- `formalism.md` §3.2: NE constraint retains the thermodynamic factor;
  admissible function class (cubic in `u = (c − c̄)/c_scale`) + regularized
  LSQ fitting rule (λ=1e-3 on higher-order terms) stated in full.
- `oracle-spec.md` §2/§3 updated to match (also fixed a stale "50-knot"
  NE-ansatz description that predated the cubic switch).

**Consequences.** No code, `data.h5`, or `truth.npz` change → the reference
solution still passes 28/28 unchanged (re-verified 2026-07-30). The task's
difficulty story shifts slightly: solving no longer requires reverse-
engineering an undocumented convention, but the core ill-conditioned coupled
D(c)/t⁺⁰(c) inversion (incl. negative t⁺⁰) remains — the reviewer agreed the
inverse problem is "legitimate and challenging" once documented.

## ADR-0013 — v0.2 hardening: de-scaffolded spec + mixed-regime case_4

**Status:** Accepted (2026-09-09; user approval of the L1+L2 package;
reviewer mandate AaronFeller 2026-08-21: "pass rate seems a bit high …
you may need to harden the task, if possible")

**Context.** Since the ADR-0012 velocity-closure fix, `claude-opus-5`
(`reasoning_effort=xhigh`) has passed 3/3 reviewer `/run`s (Aug 12/14/21),
each with ~4× margin on the D/t⁺⁰ gates in ~half the time budget, while
Fable 5.1 reports 52.6 % on TB-Science 0.1 overall. The 2026-08-21 noise
experiment (`_harden_results/`) closed the margin-based levers empirically:
at 4× noise the reference solution itself fails (case_1 t⁺⁰ ratio 1.01,
case_4 D 1.18), and a top agent that finds the correct method matches the
reference's ~0.02 accuracy — so no noise level or tolerance separates a
correct agent from the oracle. Hardening must make the correct method
harder to find or execute, while the reference still passes
deterministically with ≥2× margin.

**Decision.** Two measures for v0.2 (deadline 2026-10-05):

1. **L1 — De-scaffold the agent-facing spec.** Remove *guidance* content,
   keep every *contract* element verbatim:
   - `formalism.md` (template `case_gen/writers.py::FORMALISM_MD`): delete
     §3.3.1 entirely (worked flux example, sign-convention bullets,
     common-pitfalls list). Units remain fully declared in §1/§3.3;
     recognizing the conversions becomes the agent's job again.
   - `instruction.md`: delete the method-advice paragraph ("You may use
     any numerical approach … too slow … too brittle. Choose
     accordingly.") — it explicitly warns about single-start optimization
     on multi-modal landscapes, i.e. hands over the strategy.
   - Retained verbatim: frame/sign conventions, the ADR-0012 velocity
     closure note, the §3.2 NE fitting rule (factor retained, cubic
     ansatz, λ=1e-3, `max(mean(c_data²),1)` floor), §3.3 flux formulas,
     §4 regime rule, §5 tolerances, output schema. The de-scaffold must
     not reopen the ADR-0012 class of spec-vs-code ambiguity.

2. **L2 — Redesign case_4 into a mixed-regime transition case.** New truth
   `t⁺⁰(c)` crosses zero *inside* the graded c_grid band while the
   emergent `t⁺⁰_NE` stays well positive across it, so the 50 labels split
   into an `NE_deviates` block and an `NE_wrong_sign` block whose boundary
   must be recovered from the data. This makes the NE inversion
   load-bearing (labels are currently case-constant and effectively free
   once the case-level physics is right). Case_4 is the case to spend: its
   discriminator duplicates case_3's, and its original basin-trap intent
   was dropped in June. Keeping 4 cases avoids touching instruction,
   verifier count, container plumbing, and the time budget.

   **Label-robustness mask** (deliberately supersedes the 2026-06-26
   calibration rule "narrow c_grid to a single-regime band" for this
   case): the regime check grades only grid points where
   `|t⁺⁰_oracle| ≥ 0.03`. The mask is computed at case-generation time
   from truth (`regime_graded` bool[50] in `truth.npz`) and never shipped
   to the agent; the agent-facing spec states the exclusion rule
   abstractly (it reveals nothing about *where* the crossing is, since
   the agent does not know `t⁺⁰_oracle`). δ = 0.03 ≈ 2.7× the reference
   solution's worst-point t⁺⁰ error (~0.011), so the reference's labels
   at graded points are sign-safe with margin, while an agent at the
   check-#2 limit (0.05) can no longer collect the labels for free — near
   the crossing the label check is deliberately sharper than check #2.
   Truth `t⁺⁰(c)` is steepened near the crossing (target slope ≳1 /(mol/L))
   to keep the masked band a small minority of points (target ≤ ~10 of 50).
   The same rule applies to cases 1–3 vacuously (their min |t⁺⁰| ≥ 0.085).

**Stretch (separate go/no-go, not committed here):** L3 — sparse velocity
data (bundle `v_data` at a few x-locations to re-arm the genuine basin
trap). Requires a multi-start reference and re-scoped checks #4/#6; decide
only after L1+L2 are validated and only if ≥2 weeks remain before freeze.

**Rejected alternatives.**
- Tighter tolerances / more noise: empirically closed (Context above);
  the identifiability audit additionally showed the per-point D gate at
  sparse high-c knots is already ~2× tighter than local data power.
- Unknown thermodynamic factor: likely identifiability collapse (data
  constrains products like D·φ) and a scope change vs. approved proposal
  #335.
- Bigger grids / compute pressure: taxes wall-clock, not understanding;
  the near-miss rubric explicitly credits the task for not inflating
  difficulty artificially.

**Success target.** Opus-5-xhigh ≤ ~1/5 reviewer trials; reference passes
32/32 deterministically with ≥2× margin on every continuous check;
anti-cheat matrix (lab-frame, literature-lookup, v_pred:=v_data) still
catches on all non-calibration cases; proposal's 10–20 % band restored
against the strongest graded agent.

**Consequences.**
- **Precondition fix:** `case_gen/writers.py::FORMALISM_MD` is stale —
  it predates ADR-0012 (velocity closure), the §3.2 fitting-rule pin, and
  the §5 RMS wording, so regeneration today would silently revert three
  reviewer fixes. Sync the template to the shipped formalism.md first
  (verified byte-identical by regenerating), in its own commit, before
  any hardening edits.
- `case_gen/configs/case_4.yaml`: new `tp0_table` (zero crossing inside
  c_grid, steep near c*), possibly widened c_grid / retuned current;
  regenerate case_4 artifacts; re-run the ADR-0004 anti-cheat audit and
  the lab-frame cheat matrix for the new case.
- `case_gen/generate.py`: emit `regime_graded` into `truth.npz`.
- `tests/test_outputs.py::test_regime`: grade only masked-in points
  (backward-compatible: truth files without `regime_graded` grade all).
- `formalism.md` §4/§5 + `instruction.md` check list: state the exclusion
  rule; §3.3.1 removed; method-advice paragraph removed.
- Docs in the same commits: `case-design.md` case_4 rewrite +
  discriminator matrix, `verifier-spec.md` §3, margins re-recorded in
  `key-facts.md`.
- Full re-validation per CLAUDE.md before any commit touching `tests/`:
  regenerate cases → reference 32/32 → anti-cheat matrix →
  `harbor run -a oracle` reward = 1.
- The held identifiability maintainer note is folded into this redesign;
  relocation of `tests/oracle/{flux,invert_ne}.py` to `authoring/` rides
  the same commit series.

## ADR-0014 — L3 (sparse velocity) rejected: identifiability collapse, not a basin trap

**Status:** Accepted (2026-09-11) — L3 will NOT be implemented; the
ADR-0013 L1+L2 package (through `3f60e58`) is the final v0.2 hardening.

**Context.** ADR-0013 left L3 — bundling `v_data` at only a few
x-stations to re-arm case_4's original multi-modal basin trap — as a
stretch behind a feasibility gate. The hypothesis: dense v(x,t) is what
makes the joint inverse nearly convex, so starving it should create a
plausible positive-t⁺⁰ basin that captures single-start optimization
from a literature-prior init (+0.30), penalizing search strategy rather
than accuracy.

**Feasibility experiment (2026-09-11, scratchpad `exp_l3_basin.py`).**
Case_4; v restricted to x-station subsets {100 (control), 8, 5, 3, 0};
joint inversion (reference machinery, K=10, λ_tp=1e-5) from inits
t⁺⁰ ∈ {+0.30, +0.10, −0.10, −0.40}. Findings:

1. **No multi-modality at any sparsity.** At every station count, all
   four inits converge to the SAME solution (init-independent to ~1%).
   There is no wrong basin that captures bad inits — the landscape
   stays effectively unimodal.
2. **Instead, the unique optimum drifts off-truth as v thins.**
   Worst-point errors vs truth (any init): full → D 0.04 / t⁺⁰ 0.02
   (passes); 8 stations → D 0.14 / t⁺⁰ 0.03 (D FAILS, worst mid-band
   at c≈3.07, not an edge); 5 → D 0.12 / t⁺⁰ 0.19 (both fail;
   t⁺⁰ overshoots to −0.44); 3 → D 0.47 / t⁺⁰ 0.29 (collapse,
   t⁺⁰ ≈ 0 flat); c-only → D 0.14 / t⁺⁰ 0.046.
3. **The off-truth optima are data-PREFERRED:** the fitted solutions'
   pure data misfit is 0.55–0.92× the truth parameters' own misfit
   under the same sparse loss. The gates would reject solutions that
   fit the observable data strictly better than the ground truth does.

**Decision.** Reject L3. Sparse velocity does not create a search
problem (multi-start would rescue nothing — the global optimum itself
is outside the gates); it destroys identifiability, so any difficulty
gained would come entirely from grading against our specific curve in
a regime the data no longer constrains — the "verifier accepts only
the oracle's solution" anti-pattern in its strongest form, and worse
for the reference than for agents (the reference solution class itself
fails D at 8 stations).

**Consequences.**
- The v0.2 submission is the validated L1+L2 package (HEAD `3f60e58`):
  de-scaffolded spec, mixed-regime case_4 with graded-label mask,
  c_grid to 3.30, λ-ladder reference; 32/32, oracle-in-container 1.0,
  Opus-5-xhigh mock evidence 1/2.
- The June finding (basin trap absent under v-weighted joint fitting)
  now has a converse: the trap cannot be re-armed by weakening the
  v-channel either. Any future difficulty increase must come from new
  physics scope (e.g. a different regime/chemistry case), not from
  information starvation of the current cell.
- Next actions: refresh harbor-task-format pin, mirror subtree to the
  fork (PR #584), reviewer reply presenting the hardening + mock
  evidence, request `/run trials=3`.

## ADR-0015 — v0.3 hardening: graded adjoint sensitivities of the concentration polarization

**Status:** Accepted (2026-09-24; user approval "record your decisions and
do the implementation"; design rationale and experiments in
`docs/plan/gradient-hardening-proposal.md`, experiment scripts/results in
`docs/plan/_sens_exp/`).

**Context.** After ADR-0013 (L1+L2) the task sits at ~1/2 for
Opus-5-xhigh on n=2 mock trials and the reviewer still judges the pass rate
high. ADR-0014 closed information starvation. Trajectory review of both
mock trials and the reviewer pass shows every passing agent built a NumPy
forward solver driven by `scipy.optimize.least_squares` (finite-difference
Jacobians, zero autodiff): the "differentiable modeling" in the accepted
proposal's title is never exercised. Haotian Chen (DiffEC first author)
suggested grading gradient evaluation. DiffEC itself differentiates only
the fit loss w.r.t. two parameters — ≈ 0 at convergence, ungradeable — so
the graded gradient has to be of a physical quantity, chosen here.

**Decision.** Add one graded deliverable: the **adjoint gradient of the
end-of-plateau concentration polarization** with respect to every model
input, evaluated on a canonical knot model at the agent's own recovered
transport functions, verified truth-free.

1. **QoI.** `Q = c(x[Nx−1], t_qoi) − c(x[0], t_qoi)` (mol/L) on the
   bundled x grid, `t_qoi` = the last bundled time sample inside the
   constant-current plateau (`params.json: t_qoi_s, t_qoi_index`).
2. **Base point / knot model.** `c_knots[12]` uniform over
   `[min c_data, max c_data]` (shipped in `params.json`). The agent reports
   `D_knots[12]` (m²/s), `t_plus_0_knots[12]`; the knot model is linear
   interpolation on `c_knots`, constant outside — the verifier's existing
   evaluation rule, now stated for the extended grid.
3. **Gradients** (total derivatives through the coupled c–v₀ evolution
   of formalism §2): `dQ_dlnD[12]`, `dQ_dtp0[12]`, `dQ_di[Nt]` (current
   program perturbed by 1 A/m² hats on the bundled `t` grid), `dQ_dc0[Nx]`
   (initial concentration of one data cell; `v₀(x,0)` keeps the uniform
   `c_init` expression). Plus `Q_pol` itself.
4. **Verification (check #7, `test_sensitivity`).** Rebuild the knot model
   from the agent's knot values, differentiate the held-out oracle with
   `jax.value_and_grad`, and require per block
   `max|agent − verifier| ≤ 0.25 · max|verifier|`; `|Q_pol − Q_verifier|
   ≤ 0.10 |Q_verifier|`; and `|Q_verifier − Q_data| ≤ 0.10 |Q_data|` where
   `Q_data` is the same difference read from `c_data` (base point must
   reproduce the measured polarization). No oracle truth enters the
   check, so its tolerance covers only discretization/implementation
   differences (measured drift N=100→200 for the final definition,
   `docs/plan/_sens_exp/calib_final.json`: ln D 2.5–11 %, t⁺⁰ ≤ 2.5 %,
   i(t) ≤ 2.1 %, c₀ ≤ 0.6 % → 2.25× margin on the worst block, ≥ 8×
   elsewhere).
5. **Reference.** `solution/reference_solver.py` samples its fitted
   functions at `c_knots` and runs `jax.jacrev` through its clean-room
   `pde.py` on the knot model. Container images unchanged (JAX already
   pinned in both).

**Why this shape.** Reverse mode yields all `24 + Nt + Nx = 174`
sensitivities in one backward pass (0.5 s/case on the oracle). Finite
differences need 174 forward solves (≈ 3–5 min/case for the NumPy solvers
seen in trials, on top of 27–50 min already used). The route is not
forbidden — the accepted proposal promises method freedom — but the
budget makes the differentiable route the practical one, and every
gradient block has a textbook reading (which concentration window's D and
t⁺⁰ control polarization; the cell's current-memory kernel with its
causality cutoff; the adjoint state at t = 0).

**Rejected alternatives.**
- Grading vs. truth-evaluated sensitivities: gate-edge parameter error
  shifts the ln D block 27–41 %, so a truth-based gate would test nothing.
- Pointwise Jacobians on the 50-point `c_grid`: D-sensitivities to
  0.0075 mol/L-wide hats drift 26–70 % under refinement (D enters through
  a second derivative); not gradable.
- Velocity QoIs (electrode face, mid-cell velocity, displacement): face
  velocity is identically 0 by the flux BC; mid-cell gradients reduce to
  the analytic direct term in t⁺⁰ with an ill-conditioned D part.
- Fisher-information fraction (needs σ_c, σ_v published; 0.25 drift on
  narrow hats): deferred, not in the baseline.
- Posterior error bars: needs a canonical prior; ADR-0012-class ambiguity.

**Consequences.**
- `tests/oracle/sensitivity.py` (pure function) + `simulate(c0_override=)`
  hook in `tests/oracle/solver.py`; `test_outputs.py::test_sensitivity`;
  schema/declared-field checks extended (`c_knots`, `D_knots`,
  `t_plus_0_knots`, `sensitivity{…}`); 9 tests × 4 cases = 36.
- `case_gen`: `SENS_N_KNOTS = 12`; `params.json` gains `c_knots`,
  `t_qoi_s`, `t_qoi_index`; `truth.npz` gains the same; formalism §1/§3/
  §3.4/§5 and `instruction.md` updated; `data.h5` unchanged.
- Docs same commit series: verifier-spec §6b, key-facts margins,
  architecture, README (Difficulty/Reference/Verification), CLAUDE.md
  check counts, pre-PR audit file list.
- Full re-validation: regenerate → reference 36/36 → refinement
  calibration of the final definition → `harbor run -a oracle` = 1 →
  one cold Opus-5-xhigh mock trial → reviewer `/run`.
- Difficulty knob left in reserve: a finer canonical time grid for
  `dQ_di` raises the FD cost without touching the AD cost.
