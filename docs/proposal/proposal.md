# [Task Proposal #180] Differentiable modeling of concentrated-electrolyte mass transport from operando profiles

## Scientific Domain
Physical Sciences > Chemistry > Electrochemistry 

## Scientific Problem
Mass transport in concentrated electrolytes is central to the performance of modern electrochemical energy systems, governing extremely fast charging, low-temperature operation, and dendrite suppression in lithium-based batteries. Despite its importance, the concentrated regime is poorly described by classical Nernst-Planck theory, which is built on dilute-solution assumptions and ignores correlated ion-ion and ion-solvent motions. Newman's concentrated-solution theory captures these effects by explicitly coupling the cation transference number with respect to solvent motion (t⁺⁰), the salt diffusivity D, and the solvent velocity field — but parametrizing this theory from experiment has historically been a bottleneck.

Recent operando X-ray techniques have changed this picture. X-ray absorption microscopy (XAM) and X-ray photon correlation spectroscopy (XPCS) can simultaneously measure salt concentration profiles c(x, t) and solvent velocity profiles v(x, t) during electrochemical polarization, providing a direct experimental window into concentrated transport. The remaining inverse problem — recovering the concentration-dependent transport properties D(c) and t⁺⁰(c) from these fields under the constraint of Newman's PDEs — has traditionally relied on gradient-free optimization (PSO, Bayesian optimization, CMA-ES, Nelder-Mead), which is slow, data-inefficient, and vulnerable to multi-modal loss landscapes.

Chen et al. (ACS Energy Letters 2026) recently developed an end-to-end differentiable simulator that performs this inversion 1-2 orders of magnitude faster than gradient-free alternatives. A striking outcome is the recovery of a negative transference number at high salt concentration, which classical Nernst-Planck cannot produce. Historically attributed to negatively-charged ion clusters, this phenomenon is more accurately explained as a continuum-scale consequence of solvent motion (Mistry et al. 2022, 2023). This workflow is actively run by our group and collaborators on multiple electrolyte systems, and constitutes the flagship application of the just-published differentiable electrochemistry framework. It captures a real, unsolved research challenge with a published reference implementation and concrete failure modes that separate physics-aware solutions from physics-naive ones.


## Workflow Details
The agent is given bundled operando data from a symmetric cell under potentiostatic polarization, organized as \`cases/case\_{1..4}/\`, each containing:
- \`data.h5\`: salt concentration profiles c(x, t) at multiple time points (analog of XAM measurements) and solvent velocity profiles v(x, t) at the same times (analog of XPCS measurements)
- \`params.json\`: cell geometry, applied current density i(t), solvent concentration c₀, salt partial molar volume V̄, the canonical concentration grid c\_grid (50 uniformly-spaced points spanning the case's concentration range), and 10 flux-sampling coordinates (x\_k, t\_k)
- \`formalism.md\`: the moving-frame PDE specification (eqs 7-8 in Chen et al. 2026), the regime classification rule (see Evaluation), and the exact output schema

The agent must produce, per case, a \`results/case\_X/transport.json\` containing:
- D(c) and t⁺⁰(c) evaluated at the 50 c\_grid points (the canonical parameterization; linear interpolation between grid points is the evaluation rule)
- t⁺⁰\_NE(c) at the same c\_grid points — the Nernst-Einstein-equivalent transference number the system would exhibit if solvent motion were neglected
- regime classification per c\_grid point ∈ {NE\_valid, NE\_deviates, NE\_wrong\_sign}, mechanically determined from t⁺⁰ and t⁺⁰\_NE (rule defined below)
- predicted velocity field v\_pred(x, t) sampled on the same (x, t) grid as the bundled v(x, t)
- cation flux decomposition (J\_diff, J\_mig, J\_conv) at each of the 10 specified sampling points

The methodology is not prescribed. Agents may use any numerical approach — differentiable JAX/PyTorch simulators, gradient-free optimizers, surrogate models, neural networks, hand-tuned fits, anything. The per-case time budget (5-10 minutes on 4-8 CPU cores) and the case design naturally select against approaches that are too slow or too brittle.


## Dependencies & System Requirements
(i) Software/libraries: Python ≥ 3.10 and any standard Python scientific stack (NumPy, SciPy, JAX, PyTorch, etc. — agent's choice). Verifier uses pytest. No external scientific solvers required, no proprietary licenses, no scientific databases.

(ii) Hardware: 4-8 CPU cores. 8-16 GB RAM. < 100 MB storage for bundled input data. No GPU required. Per-case time budget: 5-10 min. Total wall-time across 4 cases: under 1 hour.


## Dataset
The published workflow in Chen et al. 2026 uses real experimental operando data: salt concentration profiles c(x, t) from XAM and solvent velocity profiles v(x, t) from XPCS, originally measured by Steinrück et al. (Energy Environ. Sci. 2020) on a Li | PEO-LiTFSI | Li symmetric cell at 90°C. Reference concentration-dependent transport properties for that system are available from independent experimental measurements (Pesko et al. 2017).

For the benchmark, we bundle 4 hidden oracle-generated cases produced by our adaptation of the DiffEC forward simulator (BattModels/DiffEC, MIT license). The ground-truth D(c) and t⁺⁰(c) functions, noise seeds, and concentration ranges for these cases are not published and not derivable from the public DiffEC results or Pesko 2017 literature values. The case shapes (concentration ranges, time scales, signal-to-noise) are calibrated against Steinrück 2020 to remain physically realistic. Each case contains profiles on a 1D spatial grid of ~100 points across ~50 time points; total bundled data ~50-100 MB.

Ground-truth parameters span four regimes of concentrated-electrolyte behavior:
- One case calibrated to reproduce a Steinrück-2020-like sign flip with concentration (the "matches reality" case, but with perturbed parameters distinct from any published values)
- One case with weak ion-solvent correlations (t⁺⁰ positive across all c, Nernst-Einstein-valid throughout)
- One case with intermediate correlations (Nernst-Einstein deviates but signs agree)
- One case engineered with a multi-modal loss landscape — well-separated positive-t⁺⁰ and negative-t⁺⁰ local minima — where single-start optimization lands in the wrong basin a substantial fraction of the time

The public DiffEC repository and Steinrück 2020 / Pesko 2017 papers serve as methodological references only; the benchmark cases use held-out parameter functions and noise seeds.


## Evaluation Strategy
Fully objective, programmatically verifiable via pytest. The verifier checks five quantities per case:

1\. Recovered transport parameters. At each of the 50 c\_grid points:
- |D\_agent(c) − D\_oracle(c)| / D\_oracle(c) ≤ 0.10 (relative)
- |t⁺⁰\_agent(c) − t⁺⁰\_oracle(c)| ≤ 0.05 (absolute)

2\. Regime classification (discrete invariant). Mechanically determined from agent's reported t⁺⁰(c) and t⁺⁰\_NE(c) at each c\_grid point:
- NE\_valid if |t⁺⁰(c) − t⁺⁰\_NE(c)| < 0.05
- NE\_deviates if |t⁺⁰(c) − t⁺⁰\_NE(c)| ≥ 0.05 and sign(t⁺⁰) = sign(t⁺⁰\_NE)
- NE\_wrong\_sign if sign(t⁺⁰) ≠ sign(t⁺⁰\_NE)
Must exactly match the oracle's regime labels at every c\_grid point. Across 4 cases × 50 grid points = 200 categorical labels with 3 levels — a discriminator that is statistically impossible to satisfy by luck.

3\. Velocity-field prediction (physics consistency). Relative RMSE:
- ||v\_pred − v\_data||₂ / max|v\_data| ≤ 0.15 per case
Catches agents whose concentration fit is acceptable but whose underlying physics is wrong (e.g., lab-frame implementations).

4\. Cation flux decomposition (interpretation handle). At each of the 10 sampling points (x\_k, t\_k):
- |J\_X\_agent(x\_k, t\_k) − J\_X\_oracle(x\_k, t\_k)| / |J\_total\_oracle(x\_k, t\_k)| ≤ 0.15 for each X ∈ {diff, mig, conv}
Agents who arrived at approximately-correct parameters via wrong physics — most commonly, lab-frame Nernst-Planck — will report ~zero convective contribution at high salt concentration and fail this check, even if D and t⁺⁰ happen to land near the oracle. This converts the "interpretation of results" requirement into a programmatic numerical comparison.

5\. Self-consistency (anti-cheat). The verifier independently loads the agent's reported D(c), t⁺⁰(c), runs the oracle's moving-frame PDE solver from those parameters, and checks that the simulated velocity field matches v\_data within tolerance #3. This catches agents who report fitted values directly (e.g., from literature or the public DiffEC results) without performing the actual inversion: their reported D, t⁺⁰ won't reproduce v\_data under the oracle's solver if their physics was wrong.

The combination of continuous tolerances, discrete pattern matching across 200 grid points, physics-consistency cross-validation, flux decomposition, and self-consistency verification provides redundant, deterministic checks with no judgment calls and no process verification.


## Complexity
(i) The underlying research framework — Chen et al. (ACS Energy Letters 2026), the first paper to systematically apply differentiable programming to electrochemistry — required a year-plus collaboration between domain experts. The bounded benchmark task is much narrower. With the formalism specified, operando data bundled, parameterization fixed, output schema explicit, and the open-source DiffEC repository as a methodological reference, an expert who knows the idea of the answer — that is, who has implemented or read DiffEC and understands moving-frame concentrated-solution theory — would complete the implementation in approximately 3-5 hours of focused work. The difficulty is recognizing the moving-frame requirement and implementing the inversion correctly, not writing a research codebase from scratch.

(ii) For a frontier AI agent: genuinely difficult — we expect a solve rate in the 10-20% target range. The agent must independently recognize several non-obvious requirements:
- Differentiable vs. gradient-free: agents who default to \`scipy.optimize.minimize\` with finite-difference gradients face 1-2 orders of magnitude disadvantage (benchmarked in Chen et al. 2026, Figure S25) and will exceed the time budget on the harder cases and will not finish within the 5-10 min CPU budget on the harder cases
- Moving electrolyte frame: agents who implement the PDE in the laboratory frame produce wrong velocity predictions even when concentration fits look acceptable — caught by the v(x, t) check
- Solvent-motion physics in transference: agents who default to lab-frame Nernst-Planck transference recover only positive t⁺⁰ across all concentrations and miss the sign-flip cases entirely
- Multi-modal loss landscape: the high-concentration case has well-separated positive-t⁺⁰ and negative-t⁺⁰ local minima. Single-start optimization lands in the wrong basin a substantial fraction of the time. Agents must recognize the non-convexity from physical intuition and design a multi-start protocol — there is no algorithmic hint that one is needed
- Right answer, wrong physics: agents who arrive at approximately-correct D and t⁺⁰ by lucky fitting but use lab-frame physics will report wrong flux decompositions — substantial under-prediction of the convective contribution at high salt concentration
- Memory/time blow-up on naive autograd: agents who unroll backprop through time-stepping (Python for-loop with jax.grad) rather than using jax.lax.scan or an adjoint solver will exceed memory or time on operando-length trajectories
Crucially, none of these failure modes are signaled by the inputs alone. The agent must derive the moving-frame requirement from the PDE specification, recognize the non-convexity from physical reasoning about the loss landscape, and choose appropriate algorithmic responses — exactly the kind of compounding scientific judgment that distinguishes physics-aware solutions from physics-naive ones.

We will run preliminary tests with frontier agents (Claude Opus 4.7, GPT-5, Gemini 2.5) on a prototype before final submission to provide empirical solve-rate data, analogous to the kagome proposal's "21% of numeric test points fail" benchmark.


## References & Resources
1. Chen, H.; Huang, C.; Rodríguez, A.; Mistry, A.; Viswanathan, V. Differentiable Electrochemistry: A Paradigm Characterizing Physical Laws in Electrochemical Systems. ACS Energy Letters 2026, 11, 2005-2018. https://doi.org/10.1021/acsenergylett.5c03761
2. Open-source reference implementation (mass transport benchmarks): https://github.com/BattModels/DiffEC/tree/main/Mass%20Transport%20in%20Concentrated%20Electrolytes%20and%20Benchmarks
3. Steinrück, H.-G. et al. Concentration and velocity profiles in a polymeric lithium-ion battery electrolyte. Energy Environ. Sci. 2020, 13, 4312-4321
4. Pesko, D. M. et al. Negative transference numbers in poly(ethylene oxide)-based electrolytes. J. Electrochem. Soc. 2017, 164, E3569
5. Mistry, A. et al. Effect of solvent motion on ion transport in electrolytes. J. Electrochem. Soc. 2022, 169, 040524
6. Mistry, A.; Yu, Z.; Cheng, L.; Srinivasan, V. On relative importance of vehicular and structural motions in defining electrolyte transport. J. Electrochem. Soc. 2023, 170, 110536
7. Mistry, A.; Steinrück, H.-G.; Toney, M. F.; Balsara, N. P.; Srinivasan, V. Multiple Operando Fields Can Identify a Predictive Mass Transport Theory in Electrolytes. J. Phys. Chem. C 2025, 129, 2874-2882
8. JAX: Bradbury, J. et al. JAX: composable transformations of Python+NumPy programs. arXiv:2203.17189


## Additional Information
None provided

## Author Information
Author: Changwen Xu
Email: changwex@umich.edu
Role: PhD Student in Mechanical Engineering
Profile: https://changwenxu98.github.io/; https://scholar.google.com/citations?user=GyVx78kAAAAJ&hl=en; https://www.linkedin.com/in/changwenxu/; https://x.com/changwen_xu98
GitHub: https://github.com/ChangwenXu98
Discord: changwenxu_43510
How did you hear about TB-Science: None provided
Recommended Reviewers: None provided

---
*Submitted via TB-Science Task Proposal Form*

## Supporting Files (optional)

*Attach any files that help illustrate the workflow task, e.g. scripts, notebooks, example inputs/outputs, or figures.*

@docs/proposal/formalism.md