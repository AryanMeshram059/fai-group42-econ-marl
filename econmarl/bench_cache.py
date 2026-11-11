"""Disk cache for B1 equilibria (keyed by the tax configuration)."""
import os, pickle
from .b1_solver import solve_equilibrium

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "b1")


def key(cfg):
    return f"tau{cfg.tau:g}_taua{cfg.tau_a:g}_psi{cfg.psi:g}_xi{cfg.xi:g}_theta{cfg.theta:g}_chi{cfg.chi:g}"


def get_equilibrium(cfg):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, key(cfg) + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    eq = solve_equilibrium(cfg)
    with open(p, "wb") as f:
        pickle.dump(eq, f)
    return eq
