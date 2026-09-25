"""Calibration of check #7 for the FINAL definition (oracle.sensitivity):
(1) module vs. inline re-implementation at N=100 (sanity), (2) N=100 vs
N=200/dt/2 refinement drift per block at the truth knot model, all cases."""
import os, sys, json
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, yaml, jax, jax.numpy as jnp
from jax import config; config.update("jax_enable_x64", True)
REPO = "/Users/changwenxu/Desktop/EEG/tb-science/DiffEC"
TASK = f"{REPO}/tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"
sys.path.insert(0, f"{TASK}/tests")
from oracle.solver import CaseInputs, interp_fn, simulate
from oracle.sensitivity import polarization_sensitivity, HAT_AMPLITUDE_A_PER_CM2

out = {}
for case_id in ("case_1", "case_2", "case_3", "case_4"):
    cfg = yaml.safe_load(open(f"{REPO}/case_gen/configs/{case_id}.yaml")); mat = cfg["material"]
    tr = np.load(f"{TASK}/tests/oracle_truth/{case_id}/truth.npz")
    D_fn0 = interp_fn(np.asarray(mat["D_table_cgs"]["c_mol_per_L"]), np.asarray(mat["D_table_cgs"]["D_cm2_per_s"]))
    tp0_fn0 = interp_fn(np.asarray(mat["tp0_table"]["c_mol_per_L"]), np.asarray(mat["tp0_table"]["tp0"]))
    c_knots = np.asarray(tr["c_knots"]); K = c_knots.size
    D_knots = np.asarray(D_fn0(jnp.asarray(c_knots))); tp_knots = np.asarray(tp0_fn0(jnp.asarray(c_knots)))
    factor_fn = interp_fn(np.asarray(tr["factor_c_mol_per_L"]), np.asarray(tr["factor_v"]))
    Vb = float(tr["V_bar_cm3_per_mol"]); V_bar_fn = lambda c: jnp.full_like(c, Vb)
    i_t = jnp.asarray(tr["i_t_knots_s"]); i_a = jnp.asarray(tr["i_amp_A_per_cm2"])
    t_out = np.asarray(tr["t_out_s"]); Nt = t_out.size
    Nx, dt, ns, L, c_init = int(tr["Nx"]), float(tr["dt_s"]), int(tr["num_steps"]), float(tr["L_cm"]), float(tr["c_init_mol_per_L"])
    tq_step = int(tr["t_indices"][int(tr["t_qoi_index"])])
    case = CaseInputs(N=Nx, L=L, dt=dt, num_steps=ns, c_init=c_init)
    mod = polarization_sensitivity(D_knots_cgs=D_knots, tp0_knots=tp_knots, c_knots_mol_per_L=c_knots,
        factor_fn=factor_fn, V_bar_fn=V_bar_fn, i_t_knots_s=np.asarray(tr["i_t_knots_s"]),
        i_amp_A_per_cm2=np.asarray(tr["i_amp_A_per_cm2"]), t_out_s=t_out, t_qoi_step=tq_step, case=case)

    def make(refine):
        cs = CaseInputs(N=Nx*refine, L=L, dt=dt/refine, num_steps=ns*refine, c_init=c_init)
        def Q_of(theta):
            lnD, tp, a, dc0 = theta[:K], theta[K:2*K], theta[2*K:2*K+Nt], theta[2*K+Nt:]
            D_fn = interp_fn(jnp.asarray(c_knots), jnp.exp(lnD)); tp_fn = interp_fn(jnp.asarray(c_knots), tp)
            i_fn = lambda t: jnp.interp(t, i_t, i_a) + HAT_AMPLITUDE_A_PER_CM2*jnp.interp(t, jnp.asarray(t_out), a)
            c0 = jnp.full(Nx*refine, c_init) + jnp.repeat(dc0, refine)
            _, _, ch, _ = simulate(D_fn, tp_fn, factor_fn, V_bar_fn, i_fn, cs, c0_override=c0)
            cq = ch[tq_step*refine]
            if refine > 1: cq = cq.reshape(Nx, refine).mean(-1)
            return cq[-1] - cq[0]
        th0 = jnp.concatenate([jnp.log(jnp.asarray(D_knots)), jnp.asarray(tp_knots), jnp.zeros(Nt), jnp.zeros(Nx)])
        Q, g = jax.jit(jax.value_and_grad(Q_of))(th0); return float(Q), np.asarray(g)
    Q1, g1 = make(1); Q2, g2 = make(2)
    blocks = {"dQ_dlnD": slice(0, K), "dQ_dtp0": slice(K, 2*K), "dQ_di": slice(2*K, 2*K+Nt), "dQ_dc0": slice(2*K+Nt, None)}
    modg = np.concatenate([mod.dQ_dlnD, mod.dQ_dtp0, mod.dQ_di, mod.dQ_dc0])
    r = {"Q_N100": Q1, "Q_N200": Q2, "Q_module": mod.Q_pol,
         "module_vs_inline": {b: float(np.max(np.abs(modg[s]-g1[s]))/np.max(np.abs(g1[s]))) for b, s in blocks.items()},
         "refine_drift": {b: float(np.max(np.abs(g2[s]-g1[s]))/np.max(np.abs(g1[s]))) for b, s in blocks.items()},
         "Q_data": float(tr["c_data"][int(tr["t_qoi_index"]), -1] - tr["c_data"][int(tr["t_qoi_index"]), 0])}
    out[case_id] = r
    print(case_id, json.dumps(r), flush=True)
json.dump(out, open(f"{REPO}/docs/plan/_sens_exp/calib_final.json", "w"), indent=1)
print("DONE")
