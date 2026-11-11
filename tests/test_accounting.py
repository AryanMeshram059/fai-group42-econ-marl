"""C1 accounting test: Y = C + K' - (1-delta) K + G must hold to 1e-9 whenever no safeguard binds."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import pytest

from econmarl.config import EconConfig
from econmarl.env import Economy


@pytest.mark.parametrize("tau,tau_a,psi,xi", [(0.0, 0.0, 1.0, 0.0), (0.2, 0.0, 1.0, 0.0),
                                              (0.3, 0.01, 0.0, 0.0), (0.25, 0.02, 0.5, 0.1)])
@pytest.mark.parametrize("N", [10, 100])
def test_resource_constraint(tau, tau_a, psi, xi, N):
    cfg = EconConfig(tau=tau, tau_a=tau_a, psi=psi, xi=xi)
    env = Economy(cfg, N=N, M=16, T_ep=50, seed=1)
    env.reset()
    g = torch.Generator().manual_seed(2)
    max_err, checked = 0.0, 0
    for _ in range(50):
        p_hat = torch.rand(env.M, N, generator=g, dtype=torch.float64)
        h_hat = torch.rand(env.M, N, generator=g, dtype=torch.float64)
        _, _, info = env.step(p_hat, h_hat)
        ok = ~info["safeguard"]
        lhs = info["Y"]
        rhs = info["C"] + info["K_next"] - (1 - cfg.delta) * info["K"] + info["G"]
        rel = ((lhs - rhs).abs() / lhs.abs())[ok]
        if rel.numel():
            max_err = max(max_err, rel.max().item())
            checked += rel.numel()
    assert checked > 0
    assert max_err < 1e-9, max_err


def test_budget_and_nonnegativity():
    cfg = EconConfig(tau=0.2)
    env = Economy(cfg, N=50, M=4, T_ep=20, seed=3)
    env.reset()
    for _ in range(20):
        _, _, info = env.step(torch.full((4, 50), 0.5, dtype=torch.float64),
                              torch.full((4, 50), 0.3, dtype=torch.float64))
        assert torch.allclose(info["a_next"] + info["c"], info["m"], rtol=0, atol=1e-12)
        assert (info["a_next"] >= 0).all() and (info["c"] >= 0).all()
        # government budget: rebate + G = revenue
        assert torch.allclose(info["nu"] * 50 + info["G"], info["S"], atol=1e-10)


def test_competitive_prices():
    cfg = EconConfig()
    env = Economy(cfg, N=10, M=2, seed=0)
    env.reset()
    K = torch.tensor([30.0, 40.0], dtype=torch.float64)
    L = torch.tensor([3.0, 4.5], dtype=torch.float64)
    kappa, W, R, r = env.prices(K, L)
    Y = K ** cfg.alpha * L ** (1 - cfg.alpha)
    assert torch.allclose(W * L + R * K, Y, atol=1e-12)  # Euler's theorem (CRS)
