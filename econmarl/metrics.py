"""C5: evaluation metrics -- aggregate gaps, policy gaps vs. B1, consumption-equivalent welfare,
inequality, convergence speed -- and C6 (DP part): exact price-taking best response used to
measure approximate exploitability of a frozen learned policy."""
import math
import numpy as np
import torch

from .config import EconConfig
from .env import Economy
from .evaluate import rollout, utility_parts, consumption_equivalent
from .obs import build_obs
from .b1_solver import solve_household
from .baselines import b1_policy_fn


class AggFeatureMeter:
    """Accumulates the mean of the aggregate observation columns (2:) during the stationary period."""

    def __init__(self, obs_level, sigma_loge):
        self.level, self.sig, self.sum, self.n = obs_level, sigma_loge, None, 0

    def __call__(self, state):
        if self.level == 0:
            return
        o = build_obs(state, self.level, self.sig)[..., 2:].double().mean((0, 1))
        self.sum = o if self.sum is None else self.sum + o
        self.n += 1

    def mean(self):
        return None if self.sum is None else (self.sum / self.n).float()


def sigma_loge(cfg):
    return cfg.sigma_e / math.sqrt(1 - cfg.rho_e ** 2)


@torch.no_grad()
def actor_controls_on_points(actor, obs_level, cfg, a, e_vals, agg_feats):
    """Deterministic controls (p, h) of a learned actor at individual states (a, e) with aggregate
    observation features fixed at their long-run averages."""
    a_t = torch.as_tensor(a, dtype=torch.float32)
    e_t = torch.as_tensor(e_vals, dtype=torch.float32)
    cols = [torch.log1p(a_t), torch.log(e_t) / sigma_loge(cfg)]
    obs = torch.stack(cols, -1)
    if obs_level > 0:
        obs = torch.cat([obs, agg_feats[None, :].expand(len(a_t), -1)], -1)
    act = actor.mean_action(obs).double()
    p = cfg.p_min + (cfg.p_max - cfg.p_min) * act[:, 0]
    h = cfg.h_min + (cfg.h_max - cfg.h_min) * act[:, 1]
    return p.numpy(), h.numpy()


def b1_controls_on_points(sol, a, e_idx):
    cfg = sol.cfg
    p_hat, h_hat = sol.actions(a, e_idx)
    return cfg.p_min + (cfg.p_max - cfg.p_min) * p_hat, cfg.h_min + (cfg.h_max - cfg.h_min) * h_hat


def policy_gap(actor, obs_level, cfg, eq, agg_feats):
    """mu*-weighted mean absolute gap in saving ratio and hours between learned and B1 policies."""
    na, ne = eq.mu.shape
    A = np.repeat(eq.a_fine, ne); K = np.tile(np.arange(ne), na); w = eq.mu.ravel()
    keep = w > 1e-12
    A, K, w = A[keep], K[keep], w[keep] / w[keep].sum()
    p_l, h_l = actor_controls_on_points(actor, obs_level, cfg, A, eq.sol.e[K], agg_feats)
    p_b, h_b = b1_controls_on_points(eq.sol, A, K)
    return dict(p_gap=float(np.sum(w * np.abs(p_l - p_b))), h_gap=float(np.sum(w * np.abs(h_l - h_b))))


def convergence_iter(history, key="eval_J", frac=0.95):
    """First evaluation iteration at which `key` closes `frac` of the gap between its first and final value."""
    pts = [(h["iter"], h[key]) for h in history if key in h]
    if len(pts) < 2:
        return None
    first, last = pts[0][1], np.mean([v for _, v in pts[-2:]])
    for it, v in pts:
        if abs(v - first) >= frac * abs(last - first):
            return it
    return pts[-1][0]


# ------------------------------------------------------------------ exploitability (C6)
@torch.no_grad()
def deviation_value(cfg, N, base_policy_fn, dev_policy_fn, a0, e0, dev_wealth, dev_e_idx, T=200, seed=999):
    """Discounted utility of household 0 when it follows dev_policy_fn while all others follow
    base_policy_fn, vs. when it also follows base_policy_fn (common random numbers).
    Returns per-environment (J_base, Uc_base, Uh_base, J_dev)."""
    out = []
    for use_dev in (False, True):
        M = a0.shape[0]
        env = Economy(cfg, N=N, M=M, T_ep=T, seed=seed)
        a = a0.clone(); e = e0.clone()
        a[:, 0] = dev_wealth; e[:, 0] = dev_e_idx
        state = env.reset(a0=a, e_idx0=e)
        Uc = torch.zeros(M, dtype=torch.float64); Uh = torch.zeros(M, dtype=torch.float64)
        disc = 1.0
        for _ in range(T):
            p, h = base_policy_fn(state)
            if use_dev:
                pd, hd = dev_policy_fn(state)
                p = p.clone(); h = h.clone()
                p[:, 0] = pd[:, 0]; h[:, 0] = hd[:, 0]
            _, _, info = env.step(p, h)
            uc, uh = utility_parts(info["c"][:, 0], info["h"][:, 0], cfg)
            Uc += disc * uc; Uh += disc * uh
            disc *= cfg.beta
            state = env.state()
        out.append((Uc, Uh))
    (Uc0, Uh0), (Uc1, Uh1) = out
    return Uc0 - Uh0, Uc0, Uh0, Uc1 - Uh1


def exploitability(cfg, N, base_policy_fn, dev_policy_fn, a_snap, e_snap, quantiles=(0.1, 0.5, 0.9), T=200,
                   e_idx_dev=None):
    """Best-response gain of a single deviating household, in % consumption-equivalent of the
    baseline allocation, for households starting at several wealth quantiles."""
    e_idx_dev = cfg.n_e // 2 if e_idx_dev is None else e_idx_dev
    res = {}
    for q in quantiles:
        wq = torch.quantile(a_snap, q, dim=1)
        Jb, Ucb, Uhb, Jd = deviation_value(cfg, N, base_policy_fn, dev_policy_fn, a_snap, e_snap, wq, e_idx_dev, T)
        lam = [consumption_equivalent(float(Ucb[i]), float(Uhb[i]), float(Jd[i]), cfg.theta) for i in range(len(Jb))]
        res[f"q{int(q*100)}"] = dict(ce_gain_pct=100 * float(np.mean(lam)), ce_gain_pct_sd=100 * float(np.std(lam)),
                                     dJ=float((Jd - Jb).mean()))
    res["mean_ce_gain_pct"] = float(np.mean([v["ce_gain_pct"] for k, v in res.items() if k.startswith("q")]))
    return res


def dp_best_response(cfg, W, r, nu, horizon=200):
    """Best response of a price-taking household to constant prices (W, r) and rebate nu.
    Infinite-horizon (exact) when beta*Ra < 1; otherwise no stationary optimum exists (the household
    would accumulate without bound) and we use the period-0 policy of the ``horizon``-period problem,
    which still gives a valid deviation and hence a lower bound on exploitability."""
    Ra = 1.0 + (1.0 - cfg.tau) * r - cfg.tau_a
    if cfg.beta * Ra < 1.0:
        sol = solve_household(cfg, W, r, nu)
        sol.finite_horizon = False
    else:
        sol = solve_household(cfg, W, r, nu, horizon=horizon)
        sol.finite_horizon = True
    return sol


def welfare_loss_pct(test_welfare, ref_welfare, theta):
    """% of consumption the test allocation would need (every period) to reach the reference welfare."""
    return 100 * consumption_equivalent(test_welfare["Uc"], test_welfare["Uh"], ref_welfare["J"], theta)
