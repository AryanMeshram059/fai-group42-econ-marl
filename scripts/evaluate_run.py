"""Evaluate a trained run against the B1 benchmark: aggregates, policy gaps, welfare, inequality,
convergence and exploitability (DP best response, optionally PPO best response).
Writes <run>/eval.json."""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from econmarl.ppo import load_actor, make_policy_fn
from econmarl.evaluate import rollout
from econmarl.baselines import b1_policy_fn
from econmarl.bench_cache import get_equilibrium
from econmarl.metrics import (AggFeatureMeter, sigma_loge, policy_gap, convergence_iter, exploitability,
                              dp_best_response, welfare_loss_pct, actor_controls_on_points, b1_controls_on_points)
from econmarl.best_response import train_ppo_best_response

EVAL = dict(M=8, T=500, burn_in=300, seed=12345)
AGG_KEYS = ["kappa", "K_pc", "hours", "save_rate", "mean_p", "r", "W", "nu", "C_pc", "Y_pc", "gini", "top10"]


def evaluate(path, ppo_br=False, br_iters=100, log=print):
    cfg, pc, actor, _ = load_actor(path)
    torch.set_num_threads(2)
    N = pc.N
    M_eval = EVAL["M"] if N <= 300 else 2
    eq = get_equilibrium(cfg)
    hist = json.load(open(os.path.join(path, "history.json")))
    marl_fn = make_policy_fn(actor, pc.obs_level, cfg, deterministic=True)
    meter = AggFeatureMeter(pc.obs_level, sigma_loge(cfg))
    ev = dict(EVAL, M=M_eval)
    o_m = rollout(cfg, N, marl_fn, hook=meter, **ev)
    o_b = rollout(cfg, N, b1_policy_fn(eq.sol), **ev)
    sm, sb = o_m["stats"], o_b["stats"]
    res = dict(run=os.path.basename(path), algo=pc.algo, obs=pc.obs_level, N=N, seed=pc.seed, tau=cfg.tau, psi=cfg.psi,
               marl={k: sm[k] for k in AGG_KEYS}, b1_sim={k: sb[k] for k in AGG_KEYS},
               b1_theory=dict(kappa=eq.kappa, K_pc=eq.stats["K"], hours=eq.stats["hours"],
                              save_rate=eq.stats["saving_rate"], mean_p=eq.stats["mean_p"], r=eq.r, W=eq.W, nu=eq.nu,
                              gini=eq.stats["gini"], top10=eq.stats["top10"], C_pc=eq.stats["C"], Y_pc=eq.stats["Y"]),
               welfare_marl=o_m["welfare"], welfare_b1=o_b["welfare"])
    res["welfare_loss_pct"] = welfare_loss_pct(o_m["welfare"], o_b["welfare"], cfg.theta)
    res["gap"] = {
        "kappa_rel_theory": abs(sm["kappa"] - eq.kappa) / eq.kappa,
        "kappa_rel_sim": abs(sm["kappa"] - sb["kappa"]) / sb["kappa"],
        "kappa_signed_rel_theory": (sm["kappa"] - eq.kappa) / eq.kappa,
        "hours_abs": abs(sm["hours"] - eq.stats["hours"]),
        "save_rate_abs": abs(sm["save_rate"] - eq.stats["saving_rate"]),
        "gini_abs": abs(sm["gini"] - sb["gini"]),
        "top10_abs": abs(sm["top10"] - sb["top10"]),
    }
    agg = meter.mean()
    res["gap"].update(policy_gap(actor, pc.obs_level, cfg, eq, agg))
    res["convergence_iter"] = convergence_iter(hist, "eval_J")
    res["convergence_iter_kappa"] = convergence_iter(hist, "eval_kappa")
    res["train_time_s"] = hist[-1]["time"]
    res["samples"] = hist[-1]["samples"]
    # learned vs B1 policy functions for plotting
    a_plot = np.linspace(0, 20, 81)
    curves = {}
    for k in (1, 3, 5):
        pl, hl = actor_controls_on_points(actor, pc.obs_level, cfg, a_plot, np.full(81, eq.sol.e[k]), agg)
        pb, hb = b1_controls_on_points(eq.sol, a_plot, np.full(81, k))
        curves[str(k)] = dict(p_marl=pl.tolist(), h_marl=hl.tolist(), p_b1=pb.tolist(), h_b1=hb.tolist(),
                              anext_marl=None)
    res["policy_curves"] = dict(a=a_plot.tolist(), e=eq.sol.e.tolist(), curves=curves)
    res["series"] = {k: o_m["series"][k].tolist() for k in ("K_pc", "kappa", "hours", "gini")}
    res["series_b1"] = {k: o_b["series"][k].tolist() for k in ("K_pc", "kappa", "hours", "gini")}
    res["final_wealth"] = o_m["final_wealth"].tolist()
    res["final_wealth_b1"] = o_b["final_wealth"].tolist()
    # exploitability: snapshot of the learned economy's stationary state
    M_s = 32 if N <= 300 else 8
    snap = rollout(cfg, N, marl_fn, M=M_s, T=300, burn_in=200, seed=4242)
    br = dp_best_response(cfg, sm["W"], sm["r"], sm["nu"])
    res["exploit_dp"] = exploitability(cfg, N, marl_fn, b1_policy_fn(br), snap["final_a"], snap["final_e"])
    res["exploit_dp"]["finite_horizon"] = bool(br.finite_horizon)
    res["exploit_dp"]["beta_Ra"] = float(cfg.beta * br.Ra)
    if ppo_br:
        vsnap = rollout(cfg, N, marl_fn, M=16, T=300, burn_in=200, seed=4243)
        validate = lambda act: exploitability(cfg, N, marl_fn, make_policy_fn(act, pc.obs_level, cfg, True),
                                              vsnap["final_a"], vsnap["final_e"])["mean_ce_gain_pct"]
        br_actor, br_hist = train_ppo_best_response(cfg, N, actor, pc.obs_level, snap["final_a"], snap["final_e"],
                                                    iters=br_iters, seed=pc.seed, log=log, validate=validate)
        res["exploit_ppo"] = exploitability(cfg, N, marl_fn, make_policy_fn(br_actor, pc.obs_level, cfg, True),
                                            snap["final_a"], snap["final_e"])
        res["exploit_ppo_hist"] = br_hist
    json.dump(res, open(os.path.join(path, "eval.json"), "w"))
    log(f"{res['run']}: kappa {sm['kappa']:.3f} (B1 {eq.kappa:.3f}, B1-sim {sb['kappa']:.3f}) hours {sm['hours']:.3f} "
        f"gini {sm['gini']:.3f}/{sb['gini']:.3f} welfare loss {res['welfare_loss_pct']:.2f}% "
        f"p_gap {res['gap']['p_gap']:.3f} h_gap {res['gap']['h_gap']:.3f} "
        f"DP-BR gain {res['exploit_dp']['mean_ce_gain_pct']:.2f}%"
        + (f" PPO-BR gain {res['exploit_ppo']['mean_ce_gain_pct']:.2f}%" if ppo_br else ""))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ppo_br", action="store_true")
    ap.add_argument("--br_iters", type=int, default=100)
    a = ap.parse_args()
    for r in a.runs:
        evaluate(r, a.ppo_br, a.br_iters)
