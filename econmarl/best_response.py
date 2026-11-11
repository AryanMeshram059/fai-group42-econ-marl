"""C6: single-agent PPO best response against a frozen learned population.

Household 0 in each of M economies is the learner; households 1..N-1 follow the frozen shared MARL
policy. The deviator is warm-started from the MARL actor, so any improvement it finds is a lower
bound on how exploitable the learned profile is (approximate exploitability). Checkpoints are
selected on a validation snapshot (different seed) and the selected deviator is then evaluated on
the test snapshot, so a failed best-response search reports ~0 rather than a negative gain."""
import copy, math
import numpy as np
import torch
import torch.nn as nn

from .config import EconConfig
from .env import Economy
from .obs import build_obs, OBS_DIMS
from .ppo import Critic, make_policy_fn


def train_ppo_best_response(cfg: EconConfig, N, base_actor, obs_level, a_pool, e_pool, iters=100, M=128,
                            T=200, lr=1e-4, critic_warmup=10, seed=0, epochs=4, minibatch=4096, clip=0.2,
                            gamma=0.96, lam=0.95, reward_scale=0.04, hidden=64, log=print, validate=None,
                            val_every=20):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    sig = cfg.sigma_e / math.sqrt(1 - cfg.rho_e ** 2)
    actor = copy.deepcopy(base_actor)
    critic = Critic(OBS_DIMS[obs_level], hidden)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3, eps=1e-5)
    base_fn = make_policy_fn(base_actor, obs_level, cfg, deterministic=True)
    env = Economy(cfg, N=N, M=M, T_ep=T, seed=50_000 + seed)
    hist = []
    best_val, best_sd = (validate(actor) if validate else None), copy.deepcopy(actor.state_dict())
    if validate:
        log(f"   BR validation at start (= MARL policy): {best_val:.4f}")
    for it in range(iters):
        idx = torch.randint(0, a_pool.shape[0], (M,), generator=g)
        # random household of the snapshot becomes the deviator (index 0)
        a0 = a_pool[idx].clone(); e0 = e_pool[idx].clone()
        j = torch.randint(0, N, (M,), generator=g)
        ar = torch.arange(M)
        a0[ar, 0], a0[ar, j] = a0[ar, j].clone(), a0[ar, 0].clone()
        e0[ar, 0], e0[ar, j] = e0[ar, j].clone(), e0[ar, 0].clone()
        state = env.reset(a0=a0, e_idx0=e0)
        O, A, LP, R, V = [], [], [], [], []
        with torch.no_grad():
            for t in range(T):
                p, h = base_fn(state)
                obs = build_obs(state, obs_level, sig)[:, 0]
                d = actor.dist(obs)
                act = d.sample().clamp(1e-6, 1 - 1e-6)
                p = p.clone(); h = h.clone()
                p[:, 0] = act[:, 0].double(); h[:, 0] = act[:, 1].double()
                rew, _, _ = env.step(p, h)
                O.append(obs); A.append(act); LP.append(d.log_prob(act).sum(-1)); V.append(critic(obs))
                R.append(rew[:, 0].float() * reward_scale)
                state = env.state()
            V.append(critic(build_obs(state, obs_level, sig)[:, 0]))
            R = torch.stack(R); V = torch.stack(V)
            adv = torch.zeros_like(R); gae = torch.zeros(M)
            for t in reversed(range(T)):
                gae = R[t] + gamma * V[t + 1] - V[t] + gamma * lam * gae
                adv[t] = gae
            ret = adv + V[:-1]
        obs_b = torch.stack(O).reshape(-1, O[0].shape[-1]); act_b = torch.stack(A).reshape(-1, 2)
        lp_b = torch.stack(LP).reshape(-1); adv_b = adv.reshape(-1); ret_b = ret.reshape(-1)
        adv_n = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        n = len(adv_b)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, minibatch):
                ii = perm[s:s + minibatch]
                vl = ((critic(obs_b[ii]) - ret_b[ii]) ** 2).mean()
                opt_c.zero_grad(); vl.backward(); opt_c.step()
                if it >= critic_warmup:
                    d = actor.dist(obs_b[ii])
                    ratio = torch.exp(d.log_prob(act_b[ii]).sum(-1) - lp_b[ii])
                    pl = -torch.min(ratio * adv_n[ii], ratio.clamp(1 - clip, 1 + clip) * adv_n[ii]).mean()
                    opt_a.zero_grad(); pl.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt_a.step()
        mu = (R.sum(0) / reward_scale / T).mean().item()
        hist.append(dict(iter=it + 1, mean_u=mu))
        if validate and (it + 1) % val_every == 0:
            v = validate(actor)
            hist[-1]["val"] = v
            if v > best_val:
                best_val, best_sd = v, copy.deepcopy(actor.state_dict())
            log(f"   BR it {it+1}: deviator mean per-period utility {mu:.4f}, validation gain {v:.4f}")
    if validate:
        actor.load_state_dict(best_sd)
    return actor, hist
