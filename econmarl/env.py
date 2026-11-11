"""C1: vectorised finite-N heterogeneous-household economy (a general-sum Markov game).

M independent economies with N households each are simulated in parallel as torch tensors of
shape [M, N]. Timing within a period t (Section 3 of the report):

  1. households observe (a_i, e_i) (+ aggregate information depending on the observation level)
  2. they simultaneously choose a saving ratio p_i in [p_min, p_max] and hours h_i in [h_min, h_max]
  3. L = max(sum e h, L_min), kappa = clip(K/L), competitive prices W, R, r = R - delta
  4. taxes, lump-sum rebate nu = psi S / N, cash-on-hand m, next assets a' = p m, consumption c = (1-p) m
  5. productivity transitions with the Rouwenhorst chain.

When no safeguard binds the aggregate resource constraint Y = C + K' - (1-delta) K + G holds exactly
(see tests/test_accounting.py).
"""
import math
import numpy as np
import torch

from .config import EconConfig, utility
from .markov import productivity_process
from .tax import hsv_tax, asset_tax


class Economy:
    def __init__(self, cfg: EconConfig, N: int, M: int = 1, T_ep: int = 200, seed: int = 0,
                 device: str = "cpu", dtype=torch.float64):
        self.cfg, self.N, self.M, self.T_ep = cfg, N, M, T_ep
        self.device, self.dtype = device, dtype
        e_grid, P, pi = productivity_process(cfg)
        self.e_grid = torch.tensor(e_grid, dtype=dtype, device=device)
        self.P_cum = torch.tensor(np.cumsum(P, axis=1), dtype=dtype, device=device)
        self.P_cum[:, -1] = 1.0 + 1e-12
        self.pi_cum = torch.tensor(np.cumsum(pi), dtype=dtype, device=device)
        self.pi_cum[-1] = 1.0 + 1e-12
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.t = 0

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, dtype=self.dtype, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, dtype=self.dtype, device=self.device)

    def prices(self, K, L):
        """Competitive prices given aggregate K and (floored) L, with the kappa clip safeguard."""
        cfg = self.cfg
        kappa = (K / L).clamp(cfg.kappa_min, cfg.kappa_max)
        W = (1 - cfg.alpha) * kappa ** cfg.alpha
        R = cfg.alpha * kappa ** (cfg.alpha - 1)
        return kappa, W, R, R - cfg.delta

    # ------------------------------------------------------------------ API
    def reset(self, a0=None, e_idx0=None):
        cfg, M, N = self.cfg, self.M, self.N
        if a0 is None:
            mu = math.log(cfg.init_mean_wealth) - 0.5 * cfg.sigma_a0 ** 2
            a0 = torch.exp(mu + cfg.sigma_a0 * self._randn(M, N))
        if e_idx0 is None:
            e_idx0 = torch.searchsorted(self.pi_cum, self._rand(M, N))
        self.a = a0.to(self.dtype).clone()
        self.e_idx = e_idx0.clone()
        self.e = self.e_grid[self.e_idx]
        # "pre-period" aggregates so that lagged observations are defined at t = 0 (h = 1/3 assumed)
        K = self.a.sum(1)
        L = torch.clamp((self.e / 3.0).sum(1), min=cfg.L_min_per_cap * N)
        _, W, _, r = self.prices(K, L)
        self.W_prev, self.r_prev, self.L_prev = W, r, L
        self.income_prev = W[:, None] * self.e / 3.0 + r[:, None] * self.a
        self.t = 0
        return self.state()

    def state(self):
        return dict(a=self.a, e=self.e, e_idx=self.e_idx, W_prev=self.W_prev, r_prev=self.r_prev,
                    L_prev=self.L_prev, income_prev=self.income_prev, t=self.t)

    def actions_to_controls(self, p_hat, h_hat):
        cfg = self.cfg
        p = cfg.p_min + (cfg.p_max - cfg.p_min) * p_hat
        h = cfg.h_min + (cfg.h_max - cfg.h_min) * h_hat
        return p, h

    def step(self, p_hat, h_hat):
        """p_hat, h_hat in [0, 1], shape [M, N]. Returns (reward [M, N], done, info)."""
        cfg, N = self.cfg, self.N
        p, h = self.actions_to_controls(p_hat.to(self.dtype), h_hat.to(self.dtype))
        a, e = self.a, self.e
        K = a.sum(1)
        L_raw = (e * h).sum(1)
        L = torch.clamp(L_raw, min=cfg.L_min_per_cap * N)
        kappa, W, R, r = self.prices(K, L)
        Y = K ** cfg.alpha * L ** (1 - cfg.alpha)
        y = W[:, None] * e * h + r[:, None] * a
        Ty = hsv_tax(y, cfg.tau, cfg.xi, xp=torch)
        Ta = asset_tax(a, cfg.tau_a)
        S = (Ty + Ta).sum(1)
        nu = cfg.psi * S / N
        G = (1 - cfg.psi) * S
        m = a + y - Ty - Ta + nu[:, None]
        a_next = p * m
        c = (1 - p) * m
        reward = utility(torch.clamp(c, min=cfg.c_floor), h, cfg, xp=torch)
        safeguard = (L_raw < cfg.L_min_per_cap * N) | ((K / L) < cfg.kappa_min) | ((K / L) > cfg.kappa_max)
        info = dict(K=K, L=L, Y=Y, kappa=kappa, W=W, R=R, r=r, S=S, nu=nu, G=G, C=c.sum(1),
                    K_next=a_next.sum(1), hours=h.mean(1), c=c, h=h, p=p, m=m, y=y,
                    a=a, a_next=a_next, e=e, safeguard=safeguard)
        # transition
        u = self._rand(*self.e_idx.shape)
        self.e_idx = torch.searchsorted(self.P_cum[self.e_idx], u.unsqueeze(-1)).squeeze(-1)
        self.e = self.e_grid[self.e_idx]
        self.a = a_next
        self.W_prev, self.r_prev, self.L_prev, self.income_prev = W, r, L, y
        self.t += 1
        done = self.t >= self.T_ep
        return reward, done, info
