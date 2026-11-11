"""Roll out any policy in the finite-N simulator and collect aggregate / welfare / inequality statistics."""
import math
import numpy as np
import torch

from .config import EconConfig
from .env import Economy


def gini_rows(x):
    """Gini coefficient of each row of a non-negative [M, N] tensor."""
    xs, _ = torch.sort(x, dim=1)
    n = x.shape[1]
    i = torch.arange(1, n + 1, dtype=x.dtype)
    tot = xs.sum(1).clamp(min=1e-12)
    return ((2 * i - n - 1) * xs).sum(1) / (n * tot)


def top_share_rows(x, q=0.10):
    xs, _ = torch.sort(x, dim=1, descending=True)
    k = max(1, math.ceil(q * x.shape[1]))
    return xs[:, :k].sum(1) / xs.sum(1).clamp(min=1e-12)


def utility_parts(c, h, cfg: EconConfig):
    c = torch.clamp(c, min=cfg.c_floor)
    uc = torch.log(c) if cfg.theta == 1.0 else c ** (1 - cfg.theta) / (1 - cfg.theta)
    uh = cfg.chi * h ** (1 + cfg.phi) / (1 + cfg.phi)
    return uc, uh


def consumption_equivalent(Uc, Uh, J_ref, theta):
    """lambda such that scaling this allocation's consumption by (1 + lambda) gives welfare J_ref."""
    if theta == 1.0:
        return math.exp((J_ref - (Uc - Uh)) * (1 - 0.96)) - 1  # not used (theta != 1 in experiments)
    ratio = (J_ref + Uh) / Uc
    return ratio ** (1.0 / (1.0 - theta)) - 1.0


@torch.no_grad()
def rollout(cfg: EconConfig, N: int, policy_fn, M: int = 8, T: int = 500, burn_in: int = 300,
            seed: int = 12345, env: Economy = None, a0=None, e_idx0=None, record_agents=None, hook=None):
    """policy_fn(env_state_dict) -> (p_hat, h_hat) tensors [M, N] in [0, 1].

    Uses a fixed seed so that different policies face common random numbers (same initial wealth
    and productivity shocks)."""
    env = env or Economy(cfg, N=N, M=M, T_ep=T, seed=seed)
    state = env.reset(a0=a0, e_idx0=e_idx0)
    disc = 1.0
    Uc = torch.zeros(env.M, N, dtype=torch.float64)
    Uh = torch.zeros(env.M, N, dtype=torch.float64)
    keys = ["kappa", "K_pc", "hours", "save_rate", "mean_p", "r", "W", "nu", "C_pc", "Y_pc", "gini", "top10",
            "safeguard"]
    ts = {k: [] for k in keys}
    agent_traj = []
    for t in range(T):
        p_hat, h_hat = policy_fn(state)
        if hook is not None and t >= burn_in:
            hook(state)
        rew, done, info = env.step(p_hat, h_hat)
        uc, uh = utility_parts(info["c"], info["h"], cfg)
        Uc += disc * uc
        Uh += disc * uh
        disc *= cfg.beta
        ts["kappa"].append(info["kappa"]); ts["K_pc"].append(info["K"] / N)
        ts["hours"].append(info["hours"])
        ts["save_rate"].append(info["a_next"].sum(1) / info["m"].sum(1))
        ts["mean_p"].append(info["p"].mean(1))
        ts["r"].append(info["r"]); ts["W"].append(info["W"]); ts["nu"].append(info["nu"])
        ts["C_pc"].append(info["C"] / N); ts["Y_pc"].append(info["Y"] / N)
        ts["gini"].append(gini_rows(info["a"])); ts["top10"].append(top_share_rows(info["a"]))
        ts["safeguard"].append(info["safeguard"].double())
        if record_agents is not None:
            agent_traj.append(dict(a=info["a"][:, record_agents].clone(), c=info["c"][:, record_agents].clone(),
                                   h=info["h"][:, record_agents].clone()))
        state = env.state()
    series = {k: torch.stack(v, 0).numpy() for k, v in ts.items()}   # [T, M]
    stat = {k: float(v[burn_in:].mean()) for k, v in series.items()}
    stat_sd_env = {k + "_sd": float(v[burn_in:].mean(0).std()) for k, v in series.items()}
    J = (Uc - Uh)
    out = dict(stats={**stat, **stat_sd_env},
               welfare=dict(J=float(J.mean()), Uc=float(Uc.mean()), Uh=float(Uh.mean())),
               J_agents=J.numpy(), Uc_agents=Uc.numpy(), Uh_agents=Uh.numpy(),
               series={k: v.mean(1) for k, v in series.items()},
               final_wealth=env.a.numpy().ravel(), final_e_idx=env.e_idx.numpy().ravel(),
               final_a=env.a.clone(), final_e=env.e_idx.clone())
    if record_agents is not None:
        out["agent_traj"] = agent_traj
    return out
