"""Solve and evaluate the non-learning benchmarks: B1 (all tax configs, and inside the finite-N
simulator for all N), B0 (tau = 0 reference), B2 (best constant policy), random policy, and the
exploitability sanity check for B1. Writes results/benchmarks.json."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from econmarl.config import EconConfig
from econmarl.bench_cache import get_equilibrium
from econmarl.baselines import b1_policy_fn, constant_policy_fn, random_policy_fn, solve_b2
from econmarl.evaluate import rollout
from econmarl.metrics import exploitability, dp_best_response, welfare_loss_pct
from econmarl.b1_solver import euler_errors

EVAL = dict(M=8, T=500, burn_in=300, seed=12345)
KEYS = ["kappa", "K_pc", "hours", "save_rate", "mean_p", "r", "W", "nu", "C_pc", "Y_pc", "gini", "top10"]

torch.set_num_threads(4)
out = {"configs": [], "N_scan": {}, "tau0": {}}
t0 = time.time()
for tau in (0.0, 0.1, 0.2, 0.3):
    for psi in ((1.0,) if tau == 0 else (0.0, 1.0)):
        cfg = EconConfig(tau=tau, psi=psi)
        eq = get_equilibrium(cfg)
        o = rollout(cfg, 100, b1_policy_fn(eq.sol), **EVAL)
        ee = euler_errors(eq.sol)
        out["configs"].append(dict(tau=tau, psi=psi, kappa=eq.kappa, r=eq.r, W=eq.W, nu=eq.nu,
                                   theory={k: float(v) for k, v in eq.stats.items()},
                                   sim100={k: o["stats"][k] for k in KEYS}, welfare_sim100=o["welfare"],
                                   euler_mean_log10=float(ee.mean()), euler_max_log10=float(ee.max()),
                                   hh_iters=eq.sol.iters))
        print(f"tau={tau} psi={psi}: kappa*={eq.kappa:.4f} r*={eq.r:.4f} nu*={eq.nu:.4f} hours={eq.stats['hours']:.4f} "
              f"K={eq.stats['K']:.4f} gini={eq.stats['gini']:.3f} | sim kappa={o['stats']['kappa']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

cfg = EconConfig()
eq = get_equilibrium(cfg)
# B1 inside the finite-N simulator
for N in (10, 30, 100, 300, 1000):
    ev = dict(EVAL, M=8 if N <= 300 else 4)
    o = rollout(cfg, N, b1_policy_fn(eq.sol), **ev)
    out["N_scan"][str(N)] = dict(stats={k: o["stats"][k] for k in KEYS}, kappa_sd=o["stats"]["kappa_sd"],
                                 welfare=o["welfare"])
    print(f"B1-sim N={N}: kappa={o['stats']['kappa']:.4f} (sd across envs {o['stats']['kappa_sd']:.4f})", flush=True)

# B2, random
b2 = solve_b2(cfg)
o_b1 = rollout(cfg, 100, b1_policy_fn(eq.sol), **EVAL)
o_b2 = rollout(cfg, 100, constant_policy_fn(b2["p"], b2["h"], cfg), **EVAL)
o_r = rollout(cfg, 100, random_policy_fn(0), **EVAL)
for name, o in (("B1", o_b1), ("B2", o_b2), ("Random", o_r)):
    out["tau0"][name] = dict(stats={k: o["stats"][k] for k in KEYS}, welfare=o["welfare"],
                             welfare_loss_pct=welfare_loss_pct(o["welfare"], o_b1["welfare"], cfg.theta),
                             final_wealth=o["final_wealth"].tolist(), series={k: o["series"][k].tolist() for k in ("K_pc", "kappa")})
out["tau0"]["B2"]["params"] = b2
print("B2:", b2, "welfare loss vs B1 %.2f%%" % out["tau0"]["B2"]["welfare_loss_pct"],
      "Random loss %.2f%%" % out["tau0"]["Random"]["welfare_loss_pct"], flush=True)

# exploitability sanity check: B1 population, DP best response at the simulated prices (should be ~0)
snap = rollout(cfg, 100, b1_policy_fn(eq.sol), M=32, T=300, burn_in=200, seed=4242)
s = o_b1["stats"]
br = dp_best_response(cfg, s["W"], s["r"], s["nu"])
out["tau0"]["B1"]["exploit_dp"] = exploitability(cfg, 100, b1_policy_fn(eq.sol), b1_policy_fn(br), snap["final_a"], snap["final_e"])
# and B2's exploitability (a constant policy should be very exploitable)
snap2 = rollout(cfg, 100, constant_policy_fn(b2["p"], b2["h"], cfg), M=32, T=300, burn_in=200, seed=4242)
s2 = o_b2["stats"]
br2 = dp_best_response(cfg, s2["W"], s2["r"], s2["nu"])
out["tau0"]["B2"]["exploit_dp"] = exploitability(cfg, 100, constant_policy_fn(b2["p"], b2["h"], cfg), b1_policy_fn(br2),
                                                 snap2["final_a"], snap2["final_e"])
print("B1 DP-BR gain %.3f%%, B2 DP-BR gain %.3f%%" % (out["tau0"]["B1"]["exploit_dp"]["mean_ce_gain_pct"],
                                                    out["tau0"]["B2"]["exploit_dp"]["mean_ce_gain_pct"]))
# B1 stationary distribution (for plots)
out["b1_dist"] = dict(a=eq.a_fine.tolist(), mu_a=eq.mu.sum(1).tolist())
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/benchmarks.json", "w"))
print(f"done in {time.time()-t0:.0f}s")
