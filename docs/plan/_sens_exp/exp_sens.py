"""Feasibility experiment: AD sensitivities (hat directions on c_grid) of the
moving-frame oracle. Measures cost, kernel structure, discretization
robustness, parameter-error robustness, and FD-vs-AD agreement."""
import os, sys, time, json
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, yaml, jax, jax.numpy as jnp
from jax import config; config.update("jax_enable_x64", True)
REPO = "/Users/changwenxu/Desktop/EEG/tb-science/DiffEC"
TASK = f"{REPO}/tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"
sys.path.insert(0, f"{TASK}/tests"); sys.path.insert(0, REPO)
from oracle.solver import CaseInputs, interp_fn, simulate
from case_gen.generate import factor_cgs, V_bar_cgs_at, build_current_schedule

case_id = sys.argv[1]
OUT = sys.argv[2]
cfg = yaml.safe_load(open(f"{REPO}/case_gen/configs/{case_id}.yaml"))
tr = np.load(f"{TASK}/tests/oracle_truth/{case_id}/truth.npz")
mat = cfg["material"]; a, b, d = (mat["rho_polynomial"][k] for k in "abd")
M0, M = float(mat["M0_g_per_mol"]), float(mat["M_g_per_mol"])
c_grid = np.asarray(tr["c_grid"]); K = c_grid.size
D_fn0 = interp_fn(np.asarray(mat["D_table_cgs"]["c_mol_per_L"]), np.asarray(mat["D_table_cgs"]["D_cm2_per_s"]))
tp0_fn0 = interp_fn(np.asarray(mat["tp0_table"]["c_mol_per_L"]), np.asarray(mat["tp0_table"]["tp0"]))
factor_fn = interp_fn(np.asarray(tr["factor_c_mol_per_L"]), np.asarray(tr["factor_v"]))
Vb = float(tr["V_bar_cm3_per_mol"]); V_bar_fn = lambda c: jnp.full_like(c, Vb)
i_fn = interp_fn(np.asarray(tr["i_t_knots_s"]), np.asarray(tr["i_amp_A_per_cm2"]))
Nx, dt, nsteps, L, c_init = int(tr["Nx"]), float(tr["dt_s"]), int(tr["num_steps"]), float(tr["L_cm"]), float(tr["c_init_mol_per_L"])
t_idx = np.asarray(tr["t_indices"])
sig_c = float(cfg["noise"]["c_sigma_mol_per_L"]); sig_v = float(cfg["noise"]["v_sigma_rel"]) * float(np.max(np.abs(tr["v_data"])))
cg = jnp.asarray(c_grid)

def make_model(refine, D_base, tp_base):
    case = CaseInputs(N=Nx*refine, L=L, dt=dt/refine, num_steps=nsteps*refine, c_init=c_init)
    tix = jnp.asarray(t_idx*refine)
    def f(theta):
        D_fn = lambda c: D_base(c) * jnp.exp(jnp.interp(c, cg, theta[:K]))
        tp_fn = lambda c: tp_base(c) + jnp.interp(c, cg, theta[K:])
        _, _, ch, vh = simulate(D_fn, tp_fn, factor_fn, V_bar_fn, i_fn, case)
        c_sim = ch[tix]; v_sim = 0.5*(vh[:, :-1]+vh[:, 1:])[tix]*1e7
        if refine > 1:
            c_sim = c_sim.reshape(c_sim.shape[0], Nx, refine).mean(-1)
            v_sim = v_sim.reshape(v_sim.shape[0], Nx, refine).mean(-1)
        return c_sim, v_sim
    return f

def jacobian(f, label):
    f_jit = jax.jit(f); jvp_jit = jax.jit(lambda th, v: jax.jvp(f, (th,), (v,)))
    th0 = jnp.zeros(2*K)
    t0 = time.perf_counter(); c0, v0 = f_jit(th0); c0.block_until_ready(); t_fwd_compile = time.perf_counter()-t0
    t0 = time.perf_counter(); c0, v0 = f_jit(th0); c0.block_until_ready(); t_fwd = time.perf_counter()-t0
    Jc = np.zeros((c0.shape[0], c0.shape[1], 2*K)); Jv = np.zeros_like(Jc)
    t0 = time.perf_counter()
    for j in range(2*K):
        e = jnp.zeros(2*K).at[j].set(1.0)
        (_, _), (dc, dv) = jvp_jit(th0, e)
        Jc[:, :, j] = np.asarray(dc); Jv[:, :, j] = np.asarray(dv)
    t_jac = time.perf_counter()-t0
    print(f"[{label}] fwd compile {t_fwd_compile:.1f}s, fwd {t_fwd:.2f}s, {2*K} JVPs {t_jac:.1f}s", flush=True)
    return np.asarray(c0), np.asarray(v0), Jc, Jv, dict(t_fwd=t_fwd, t_jac=t_jac)

def derived(c0, v0, Jc, Jv):
    Ic = (Jc/sig_c)**2; Iv = (Jv/sig_v)**2
    I_c = Ic.sum((0, 1)); I_v = Iv.sum((0, 1))
    frac_v = I_v/(I_c+I_v)
    # QoI: end-of-polarization concentration difference (last output time)
    g_pol = Jc[-1, -1, :] - Jc[-1, 0, :]
    # QoI: c and v at 10 flux samples (nearest grid indices)
    x_cm = (np.arange(Nx)+0.5)*L/Nx; t_out = t_idx*dt
    fx = np.asarray(tr["flux_x"])*100; ft = np.asarray(tr["flux_t"])
    ix = [int(np.argmin(np.abs(x_cm-x))) for x in fx]; it = [int(np.argmin(np.abs(t_out-t))) for t in ft]
    Kc = np.stack([Jc[i, j, :] for i, j in zip(it, ix)]); Kv = np.stack([Jv[i, j, :] for i, j in zip(it, ix)])
    # RMS sensitivity of observables per node, normalised by max|obs|
    S_c = np.sqrt((Jc**2).mean((0, 1)))/np.max(np.abs(c0)); S_v = np.sqrt((Jv**2).mean((0, 1)))/np.max(np.abs(v0))
    return dict(I_c=I_c, I_v=I_v, frac_v=frac_v, g_pol=g_pol, Kc=Kc, Kv=Kv, S_c=S_c, S_v=S_v,
                pol=float(c0[-1, -1]-c0[-1, 0]))

def dev(a, b):  # deviation normalised by max |ref| over nodes, per block (lnD nodes, tp0 nodes)
    out = {}
    for name, sl in (("lnD", slice(0, K)), ("tp0", slice(K, 2*K))):
        ref = a[..., sl]; num = np.abs(b[..., sl]-ref)
        out[name] = float(np.max(num)/np.max(np.abs(ref)))
        out[name+"_interior"] = float(np.max(num[..., 1:-1])/np.max(np.abs(ref[..., 1:-1])))
    return out

res = {"case": case_id, "sig_c": sig_c, "sig_v": sig_v}
c0, v0, Jc, Jv, tm = jacobian(make_model(1, D_fn0, tp0_fn0), "truth N=100")
d0 = derived(c0, v0, Jc, Jv); res["timing"] = tm
res["truth"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d0.items()}

# FD check on 4 directions
f_jit = jax.jit(make_model(1, D_fn0, tp0_fn0)); fd = {}
for j in (0, K//2, K, K+K//2):
    e = np.zeros(2*K); e[j] = 1e-3
    cp, vp = f_jit(jnp.asarray(e)); cm, vm = f_jit(jnp.asarray(-e))
    dc = (np.asarray(cp)-np.asarray(cm))/2e-3; dv = (np.asarray(vp)-np.asarray(vm))/2e-3
    fd[j] = dict(c=float(np.max(np.abs(dc-Jc[:, :, j]))/np.max(np.abs(Jc[:, :, j]))), v=float(np.max(np.abs(dv-Jv[:, :, j]))/np.max(np.abs(Jv[:, :, j]))))
res["fd_vs_ad_maxreldev"] = fd; print("FD vs AD", fd, flush=True)

# Discretisation robustness
c1, v1, Jc1, Jv1, tm1 = jacobian(make_model(2, D_fn0, tp0_fn0), "truth N=200 dt/2")
d1 = derived(c1, v1, Jc1, Jv1)
res["refine"] = {"Jc": dev(Jc, Jc1), "Jv": dev(Jv, Jv1), "g_pol": dev(d0["g_pol"], d1["g_pol"]),
                 "Kc": dev(d0["Kc"], d1["Kc"]), "Kv": dev(d0["Kv"], d1["Kv"]),
                 "S_c": dev(d0["S_c"], d1["S_c"]), "S_v": dev(d0["S_v"], d1["S_v"]),
                 "frac_v_maxabs": float(np.max(np.abs(d0["frac_v"]-d1["frac_v"]))), "timing": tm1}
print("refine dev", json.dumps(res["refine"]), flush=True)

# Parameter-error robustness: base model at the gate edge (D x1.08, tp0 +0.04)
D_p = lambda c: D_fn0(c)*1.08; tp_p = lambda c: tp0_fn0(c)+0.04
c2, v2, Jc2, Jv2, _ = jacobian(make_model(1, D_p, tp_p), "perturbed N=100")
d2 = derived(c2, v2, Jc2, Jv2)
res["param_err"] = {"Jc": dev(Jc, Jc2), "Jv": dev(Jv, Jv2), "g_pol": dev(d0["g_pol"], d2["g_pol"]),
                    "Kc": dev(d0["Kc"], d2["Kc"]), "Kv": dev(d0["Kv"], d2["Kv"]),
                    "S_c": dev(d0["S_c"], d2["S_c"]), "S_v": dev(d0["S_v"], d2["S_v"]),
                    "frac_v_maxabs": float(np.max(np.abs(d0["frac_v"]-d2["frac_v"])))}
print("param-err dev", json.dumps(res["param_err"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", flush=True)
