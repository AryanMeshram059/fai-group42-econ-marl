"""Calibrate chi so that mean hours in the tau = 0 B1 equilibrium equal h_max / 3, and set the mean of
the initial lognormal wealth distribution to the B1 per-capita capital. Writes results/calibration.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.optimize import brentq
from econmarl.config import EconConfig
from econmarl.b1_solver import solve_equilibrium

base = EconConfig()
target = base.h_max / 3.0

def gap(log_chi):
    eq = solve_equilibrium(base.with_(chi=float(np.exp(log_chi))))
    print(f"chi={np.exp(log_chi):.5f} hours={eq.stats['hours']:.5f}", flush=True)
    return eq.stats["hours"] - target

log_chi = brentq(gap, np.log(5.0), np.log(60.0), xtol=1e-6)
chi = float(np.exp(log_chi))
eq = solve_equilibrium(base.with_(chi=chi))
out = dict(chi=chi, init_mean_wealth=float(eq.stats["K"]), hours=float(eq.stats["hours"]),
           kappa=float(eq.kappa), r=float(eq.r), W=float(eq.W))
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/calibration.json", "w"), indent=2)
print(out)
