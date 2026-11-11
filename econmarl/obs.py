"""Observation levels O0-O3 and the permutation-invariant global summary used by the MAPPO critic.

O0: (a_i, e_i)
O1: O0 + lagged prices (W_{t-1}, r_{t-1})
O2: O1 + K_t, L_{t-1}
O3: O2 + wealth-group statistics (top-10% / bottom-50% mean wealth, lagged income and productivity),
    following the grouping used by TaxAI.
All features use fixed (not running) transformations so that policies are deterministic functions.
"""
import math
import torch

OBS_DIMS = {0: 2, 1: 4, 2: 6, 3: 12}
GLOBAL_DIM = 11


def _group_means(x, order, k_top, k_bot):
    xs = torch.gather(x, 1, order)
    return xs[:, -k_top:].mean(1), xs[:, :k_bot].mean(1)


def build_obs(state, level: int, sigma_loge: float):
    a, e = state["a"], state["e"]
    M, N = a.shape
    feats = [torch.log1p(a), torch.log(e) / sigma_loge]
    if level >= 1:
        feats += [torch.log(state["W_prev"])[:, None].expand(M, N),
                  (10.0 * state["r_prev"])[:, None].expand(M, N)]
    if level >= 2:
        feats += [torch.log1p(a.mean(1))[:, None].expand(M, N),
                  (3.0 * state["L_prev"] / N)[:, None].expand(M, N)]
    if level >= 3:
        k_top, k_bot = max(1, math.ceil(0.1 * N)), max(1, math.ceil(0.5 * N))
        order = torch.argsort(a, dim=1)
        a_t, a_b = _group_means(a, order, k_top, k_bot)
        i_t, i_b = _group_means(state["income_prev"], order, k_top, k_bot)
        e_t, e_b = _group_means(e, order, k_top, k_bot)
        for g in (torch.log1p(a_t), torch.log1p(a_b), torch.log1p(i_t.clamp(min=-0.99)),
                  torch.log1p(i_b.clamp(min=-0.99)), e_t, e_b):
            feats.append(g[:, None].expand(M, N))
    return torch.stack(feats, dim=-1).float()


def global_summary(state):
    """Permutation-invariant summary of the whole economy (for the centralised MAPPO critic)."""
    a, e = state["a"], state["e"]
    M, N = a.shape
    la = torch.log1p(a)
    q = torch.quantile(la, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=la.dtype), dim=1).T
    g = torch.cat([torch.log1p(a.mean(1))[:, None], q, la.std(1, unbiased=False)[:, None],
                   e.mean(1)[:, None], torch.log(state["W_prev"])[:, None],
                   (10.0 * state["r_prev"])[:, None], (3.0 * state["L_prev"] / N)[:, None]], dim=1)
    return g[:, None, :].expand(M, N, GLOBAL_DIM).float()
