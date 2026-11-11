"""C3: baseline policies.

B0  : the tau = 0 (laissez-faire) economy, used as the reference point in the tax experiment E4.
B1  : the known-model DP policy (from b1_solver) executed inside the finite-N simulator.
B2  : the best constant policy (p_bar, h_bar) shared by all households (grid search in the simulator).
Rand: uniformly random actions (lower reference, as in TaxAI).
"""
import numpy as np
import torch

from .config import EconConfig
from .env import Economy
from .evaluate import utility_parts


def b1_policy_fn(sol):
    """Wrap a HouseholdSolution as a simulator policy (state -> normalised actions)."""
    def fn(state):
        a = state["a"].numpy(); k = state["e_idx"].numpy()
        p_hat, h_hat = sol.actions(a.ravel(), k.ravel())
        sh = a.shape
        return (torch.from_numpy(p_hat.reshape(sh)), torch.from_numpy(h_hat.reshape(sh)))
    return fn


def constant_policy_fn(p, h, cfg: EconConfig):
    p_hat = (p - cfg.p_min) / (cfg.p_max - cfg.p_min)
    h_hat = (h - cfg.h_min) / (cfg.h_max - cfg.h_min)

    def fn(state):
        sh = state["a"].shape
        return (torch.full(sh, p_hat, dtype=torch.float64), torch.full(sh, h_hat, dtype=torch.float64))
    return fn


def random_policy_fn(seed=0):
    g = torch.Generator().manual_seed(seed)

    def fn(state):
        sh = state["a"].shape
        return (torch.rand(*sh, generator=g, dtype=torch.float64), torch.rand(*sh, generator=g, dtype=torch.float64))
    return fn


@torch.no_grad()
def _eval_constant_grid(cfg, N, ps, hs, reps, T, seed):
    P, H = np.meshgrid(ps, hs, indexing="ij")
    P, H = P.ravel(), H.ravel()
    G = len(P)
    env = Economy(cfg, N=N, M=G * reps, T_ep=T, seed=seed)
    env.reset()
    p_hat = torch.tensor(np.repeat((P - cfg.p_min) / (cfg.p_max - cfg.p_min), reps))[:, None].expand(-1, N)
    h_hat = torch.tensor(np.repeat((H - cfg.h_min) / (cfg.h_max - cfg.h_min), reps))[:, None].expand(-1, N)
    J = torch.zeros(G * reps, dtype=torch.float64)
    disc = 1.0
    for _ in range(T):
        _, _, info = env.step(p_hat, h_hat)
        uc, uh = utility_parts(info["c"], info["h"], cfg)
        J += disc * (uc - uh).mean(1)
        disc *= cfg.beta
    J = J.view(G, reps).mean(1).numpy()
    return P, H, J


def solve_b2(cfg: EconConfig, N=100, reps=4, T=300, seed=7):
    """Best constant (p_bar, h_bar): coarse grid then a refined grid around the optimum."""
    P, H, J = _eval_constant_grid(cfg, N, np.linspace(0.05, 0.95, 19), np.linspace(0.05, 0.75, 15), reps, T, seed)
    i = int(np.argmax(J))
    p0, h0 = P[i], H[i]
    P2, H2, J2 = _eval_constant_grid(cfg, N, np.clip(np.linspace(p0 - 0.05, p0 + 0.05, 11), 0.01, 0.98),
                                     np.clip(np.linspace(h0 - 0.05, h0 + 0.05, 11), 0.01, 0.99), reps, T, seed)
    j = int(np.argmax(J2))
    return dict(p=float(P2[j]), h=float(H2[j]), J=float(J2[j]))
