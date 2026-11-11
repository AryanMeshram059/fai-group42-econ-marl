"""C2 sanity checks for the B1 solver."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from econmarl.config import EconConfig
from econmarl.b1_solver import (solve_equilibrium, solve_household_vfi, euler_errors, aggregate,
                                stationary_distribution)


@pytest.fixture(scope="module")
def eq():
    return solve_equilibrium(EconConfig())


def test_market_clearing(eq):
    assert abs(eq.stats["kappa"] - eq.kappa) / eq.kappa < 1e-6


def test_interest_rate_below_time_preference(eq):
    # Aiyagari: precautionary saving pushes r below 1/beta - 1
    assert eq.r < 1 / eq.cfg.beta - 1


def test_euler_errors(eq):
    ee = euler_errors(eq.sol)
    assert np.mean(ee) < -5 and np.max(ee) < -3


def test_distribution_is_probability(eq):
    assert abs(eq.mu.sum() - 1) < 1e-10 and (eq.mu >= -1e-15).all()


def test_egm_matches_vfi(eq):
    v = solve_household_vfi(eq.cfg, eq.W, eq.r, eq.nu)
    a = v["a_grid"]
    sel = (a > 0.3) & (a < 30)
    for k in range(len(eq.sol.e)):
        _, h, an = eq.sol.policy(a[sel], np.full(sel.sum(), k))
        # VFI restricts a' to a grid, so allow grid-sized differences
        assert np.median(np.abs(an - v["a_next"][sel, k])) < 0.1
        assert np.median(np.abs(h - v["h"][sel, k])) < 0.02


def test_government_budget_with_tax():
    cfg = EconConfig(tau=0.2, psi=1.0)
    e = solve_equilibrium(cfg)
    assert abs(e.nu - e.stats["S"]) < 1e-8
    assert abs(e.stats["kappa"] - e.kappa) / e.kappa < 1e-6
