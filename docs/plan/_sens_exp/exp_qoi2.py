"""QoI adjoint-sensitivity feasibility: gradients of 2 scalar QoIs w.r.t.
(i) ln D, t+0 on 12 wide knots over the visited c-range, (ii) current
program i(t) as hats on the 50-pt output time grid, (iii) initial state
c0(x) per data cell. Refinement + gate-edge robustness, timing."""
import os, sys, time, json
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, yaml, jax, jax.numpy as jnp
from jax import config; config.update("jax_enable_x64", True)
REPO = "/Users/changwenxu/Desktop/EEG/tb-science/DiffEC"
TASK = f"{REPO}/tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"
sys.path.insert(0, f"{TASK}/tests"); sys.path.insert(0, REPO)
from oracle.solver import interp_fn, _flux, _update_solvent_vel, _initial_v0, _mesh
from jax import lax

case_id, OUT = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(f"{REPO}/case_gen/configs/{case_id}.yaml"))
tr = np.load(f"{TASK}/tests/oracle_truth/{case_id}/truth.npz")
mat = cfg["material"]
D_fn0 = interp_fn(np.asarray(mat["D_table_cgs"]["c_mol_per_L"]), np.asarray(mat["D_table_cgs"]["D_cm2_per_s"]))
tp0_fn0 = interp_fn(np.asarray(mat["tp0_table"]["c_mol_per_L"]), np.asarray(mat["tp0_table"]["tp0"]))
factor_fn = interp_fn(np.asarray(tr["factor_c_mol_per_L"]), np.asarray(tr["factor_v"]))
Vb = float(tr["V_bar_cm3_per_mol"]); V_bar_fn = lambda c: jnp.full_like(c, Vb)
i_knots_t = jnp.asarray(tr["i_t_knots_s"]); i_knots_a = jnp.asarray(tr["i_amp_A_per_cm2"])
Nx, dt, nsteps, L, c_init = int(tr["Nx"]), float(tr["dt_s"]), int(tr["num_steps"]), float(tr["L_cm"]), float(tr["c_init_mol_per_L"])
t_idx = np.asarray(tr["t_indices"]); t_out = jnp.asarray(t_idx*dt)
cd = np.asarray(tr["c_data"]); KN = 12
c_knots = jnp.linspace(cd.min(), cd.max(), KN)
t_q = float(tr["i_t_knots_s"][2])      # end of the current plateau
iq = int(round(t_q/dt))
NT = t_idx.size
nd = 2*KN + NT + Nx                    # directions: lnD knots, tp0 knots, i(t) hats, c0 cells

def make_qoi(refine, D_base, tp_base):
    N = Nx*refine; h = dt/refine; ns = nsteps*refine; dx = L/N; iqr = iq*refine
    def f(theta):
        dlnD, dtp, di, dc0 = theta[:KN], theta[KN:2*KN], theta[2*KN:2*KN+NT], theta[2*KN+NT:]
        D_fn = lambda c: D_base(c) * jnp.exp(jnp.interp(c, c_knots, dlnD))
        tp_fn = lambda c: tp_base(c) + jnp.interp(c, c_knots, dtp)
        # i(t): base piecewise-linear program + hat perturbation on the output time grid, amplitude in A/cm^2 per 1 A/m^2
        i_fn = lambda t: jnp.interp(t, i_knots_t, i_knots_a) + 1e-4*jnp.interp(t, t_out, di)
        c0 = jnp.full(N, c_init) + jnp.repeat(dc0, refine)
        v0_0 = _initial_v0(c_init, N, tp_fn, V_bar_fn, i_fn)  # IC formula uses uniform c_init (spec)
        def step(carry, _):
            c, v0, t = carry
            it = i_fn(t)
            F = _flux(c, v0, it, dx, D_fn, tp_fn, factor_fn, False)
            cn = c - h/dx*(F[1:]-F[:-1])*1e3
            Fv = _flux(cn, jnp.zeros_like(v0), it, dx, D_fn, tp_fn, factor_fn, lab_frame=False)
            vn = _update_solvent_vel(Fv, cn, V_bar_fn)
            return (cn, vn, t+h), 0.5*(vn[N//2-1]+vn[N//2])*1e7
        (c_q, v_q, _), vmid_hist = lax.scan(step, (c0, v0_0, jnp.asarray(0.0)), None, length=iqr)
        vc = 0.5*(v_q[:-1]+v_q[1:])*1e7
        if refine > 1:
            c_q = c_q.reshape(Nx, refine).mean(-1); vc = vc.reshape(Nx, refine).mean(-1)
        Q1 = c_q[-1] - c_q[0]
        Q2 = vc[Nx//2-1]; Q3 = jnp.sum(vmid_hist)*h
        return jnp.stack([Q1, Q2, Q3])
    return f

def grad_all(f, label):
    g = jax.jit(jax.jacrev(f)); th0 = jnp.zeros(nd)
    t0 = time.perf_counter(); G = np.asarray(g(th0)); tc = time.perf_counter()-t0
    t0 = time.perf_counter(); G = np.asarray(g(th0)); tr_ = time.perf_counter()-t0
    fj = jax.jit(f); t0 = time.perf_counter(); Q = np.asarray(fj(th0)); fj(th0).block_until_ready(); t0 = time.perf_counter(); fj(th0).block_until_ready(); tf = time.perf_counter()-t0
    print(f"[{label}] Q={Q}, fwd {tf:.2f}s, jacrev(3 QoIs x {nd} inputs) compile+run {tc:.1f}s, rerun {tr_:.1f}s", flush=True)
    return Q, G

blocks = {"lnD": slice(0, KN), "tp0": slice(KN, 2*KN), "i_t": slice(2*KN, 2*KN+NT), "c0_x": slice(2*KN+NT, nd)}
def dev(A, B):
    return {q: {b: float(np.max(np.abs(B[qi, sl]-A[qi, sl]))/np.max(np.abs(A[qi, sl]))) for b, sl in blocks.items()} for qi, q in enumerate(("Q1_pol", "Q2_vmid", "Q3_dispmid"))}

Q0, G0 = grad_all(make_qoi(1, D_fn0, tp0_fn0), f"{case_id} truth N=100")
Q1, G1 = grad_all(make_qoi(2, D_fn0, tp0_fn0), f"{case_id} truth N=200 dt/2")
Q2, G2 = grad_all(make_qoi(1, lambda c: D_fn0(c)*1.08, lambda c: tp0_fn0(c)+0.04), f"{case_id} gate-edge N=100")
res = dict(case=case_id, c_knots=np.asarray(c_knots).tolist(), t_q=t_q, Q=Q0.tolist(),
           G_truth={b: G0[:, sl].tolist() for b, sl in blocks.items()},
           refine_dev=dev(G0, G1), gate_edge_dev=dev(G0, G2), Q_refine=Q1.tolist())
print("refine dev", json.dumps(res["refine_dev"])); print("gate-edge dev", json.dumps(res["gate_edge_dev"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("DONE", flush=True)
