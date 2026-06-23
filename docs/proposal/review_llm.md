## 📋 Task Proposal Rubric Review

**cc**
- **Author:** @ChangwenXu98
- **Domain reviewers (chemistry):** @AaronFeller
- **Secondary reviewers:** @AllenGrahamHart
- **Final reviewers:** @StevenDillmann

**Recommendation:** 🟢 **Accept**

<details>
<summary>Full Review</summary>

## Scientific Domain
**Domain:** Physical Sciences
**Field:** Chemistry
**Subfield:** Electrochemistry (concentrated-electrolyte mass transport)

## Problem Statement
The agent is given bundled operando-style data for 4 cases of a symmetric electrochemical cell under potentiostatic polarization. Each case provides salt concentration profiles c(x,t), solvent velocity profiles v(x,t), cell geometry, applied current, and a fixed concentration grid. The agent must solve an inverse problem: recover the concentration-dependent transport properties D(c) and t⁺⁰(c) under Newman's concentrated-solution theory (moving-frame PDEs), also report the Nernst-Einstein-equivalent transference number, classify each grid point into one of three regimes, predict the velocity field, and report a cation flux decomposition (diffusion/migration/convection) at 10 sampling points. The methodology is unconstrained. Verification is via numerical tolerances on D(c) and t⁺⁰(c), exact regime label matching, velocity RMSE, flux decomposition tolerances, and a self-consistency check that re-runs the oracle PDE solver on the agent's reported parameters.

## Verifiable
**Positives:** The verification is concrete and numerical: relative/absolute tolerances on D(c) and t⁺⁰(c), exact categorical matching of regime labels over 200 grid points, velocity RMSE thresholds, flux decomposition tolerances, and a self-consistency cross-check by re-running the oracle solver. These are deterministic, programmatically checkable, and the oracle is generated synthetically so ground truth is known. The multiple redundant checks reduce the chance of false positives (e.g., right-answer-wrong-physics being accepted). No LLM-as-judge needed.

**Negatives:** Some risk in tolerance calibration: D(c) within 10% relative and t⁺⁰ within 0.05 absolute at all 50 grid points is a stringent requirement that must be achievable given the noise injected into the oracle data. If the noise/data are insufficient to constrain the parameters to this precision, the task may be unverifiable in the sense that even a correct method cannot reliably pass. The self-consistency check re-using the oracle's PDE solver is reasonable. Overall the verification machinery is sound but tolerance feasibility must be demonstrated.

**Judgement:** Accept

## Well-Specified
**Positives:** The output schema is explicit (transport.json with named fields), the c_grid is canonical, interpolation rule is stated, regime classification rule is mechanically defined, and the formalism.md provides the PDE specification. This is more specified than many proposals.

**Negatives:** The task is large and multi-faceted: five distinct outputs, four cases, a moving-frame PDE that must be reconstructed from formalism.md, and a flux decomposition whose exact sign/frame conventions must match the oracle's. The flux decomposition (J_diff, J_mig, J_conv) verification requires the agent to match the oracle's exact decomposition convention — there are multiple ways to split fluxes, and unless formalism.md pins down the precise definitions, two reasonable implementers could produce different but "correct" decompositions that fail/pass differently. This is a moderate specification risk. The breadth of outputs also raises the corner-case burden somewhat, though it is not in the "hundreds of corner cases" territory.

**Judgement:** Uncertain

## Solvable
**Positives:** There is a published reference implementation (DiffEC, MIT license) and a recent paper describing exactly this inverse workflow. The authors run it routinely. They claim a 3-5 hour implementation for an expert who knows the approach, and they plan empirical solve-rate tests with frontier agents. The cases are synthetic with known ground truth, so solvability is essentially guaranteed by construction (the oracle generated the data).

**Negatives:** The 10% relative tolerance on D(c) across all 50 points and the multi-modal case (where single-start optimization lands in the wrong basin) raise concern about whether the *intended* solution reliably passes. The authors should demonstrate their own reference solution passes the verifier under the injected noise. The multi-modal case is deliberately adversarial in a way that could make consistent passing fragile. But since the oracle generated the data and the reference framework exists, solvability is plausible.

**Judgement:** Accept

## Difficult
**Positives:** This is a genuinely hard inverse problem requiring deep domain knowledge: recognizing the moving-frame requirement, implementing differentiable PDE-constrained optimization (adjoint/scan vs naive backprop), handling non-convex loss landscapes with multi-start, and correctly attributing solvent-motion physics. The failure modes (lab-frame vs moving-frame, positive-only t⁺⁰, convection under-prediction) are non-obvious and physically meaningful. Far beyond an undergraduate course project. The compounding scientific judgment is exactly the kind of difficulty TB-Science wants.

**Negatives:** A potential concern is that the public DiffEC repository contains a reference implementation of essentially this workflow. Although the benchmark cases use held-out parameters and noise seeds, an agent with internet access could adapt the public code to fit arbitrary held-out data — reducing the task to "run the existing tool on new data." However, the inverse-problem nature means the agent still must correctly configure and run the inversion, recognize the frame requirement, and handle multi-modality, so the difficulty likely survives. The authors plan empirical solve-rate validation, which is the right approach.

**Judgement:** Accept

## Scientifically Grounded & Interesting
**Positives:** This is a flagship application of a just-published method (ACS Energy Letters 2026), tied to real operando X-ray measurements and an active research program on battery electrolytes. The negative-transference-number phenomenon and solvent-motion physics are scientifically important and current. Clearly something scientists are paid to do. Strongly grounded.

**Negatives:** None substantive.

**Judgement:** Strong Accept

## Outcome-Verified
**Positives:** Grading is purely on output artifacts (transport.json contents), methodology is explicitly unconstrained ("any numerical approach"). The self-consistency check is an anti-cheat mechanism, which is the permitted exception. No process grading on the primary objective.

**Negatives:** The time budget (5-10 min/case on CPU) functions as a soft methodological constraint — it implicitly disfavors gradient-free methods. This is acceptable as a mechanistic constraint, but it does nudge toward particular approaches. The flux decomposition matching could be argued to verify "process" (correct physics) rather than outcome, but since it's a numerical output comparison, it stays on the outcome side.

**Judgement:** Accept

## Final Analysis
This is a strong, scientifically grounded proposal drawn directly from a recent, real research workflow with a published method and reference implementation. Difficulty and interest are clearly satisfied. The main risks are: (1) tolerance feasibility — whether the intended solution reliably passes the 10% D(c) and 0.05 t⁺⁰ tolerances at all grid points under injected noise, especially in the multi-modal case; (2) specification of the exact flux decomposition convention so that correct implementations are not penalized for frame/sign convention differences; and (3) the existence of public DiffEC code potentially lowering difficulty, though the inverse-problem framing likely preserves the challenge. None of these are fatal, and the authors plan empirical solve-rate validation, which is the correct path. There is a clear route to acceptance if the reference solution is shown to pass robustly and the decomposition convention is pinned down in formalism.md.

**Decision:** Accept

</details>

> 🤖 This is an automated recommendation for a human maintainer — not a final decision. Based on [rubrics/task-proposal.md](https://github.com/harbor-framework/terminal-bench-science/blob/main/rubrics/task-proposal.md). Model: `claude-opus-4-8`