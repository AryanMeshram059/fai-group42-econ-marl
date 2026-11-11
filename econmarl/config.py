"""Economy configuration shared by the simulator (C1), the B1 solver (C2) and the learners (C4)."""
from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class EconConfig:
    # technology
    alpha: float = 0.36
    delta: float = 0.06
    # preferences
    beta: float = 0.96
    theta: float = 1.5          # CRRA
    phi: float = 2.0            # inverse Frisch elasticity
    chi: float = 29.7014       # labor disutility weight (calibrated by scripts/calibrate.py: mean B1 hours = h_max/3)
    # productivity: log e' = rho log e + sigma u, discretised with Rouwenhorst
    rho_e: float = 0.9
    sigma_e: float = 0.2
    n_e: int = 7
    # action bounds
    p_min: float = 0.001
    p_max: float = 0.99
    h_min: float = 0.0
    h_max: float = 1.0
    # government (HSV income tax T(y), linear asset tax T^a(a) = tau_a * a)
    tau: float = 0.0
    xi: float = 0.0
    tau_a: float = 0.0
    psi: float = 1.0            # share of revenue rebated lump-sum
    # safeguards
    L_min_per_cap: float = 1e-3
    kappa_min: float = 0.5
    kappa_max: float = 60.0
    c_floor: float = 1e-3       # only used inside the utility (never in accounting)
    # initial wealth: lognormal with this sigma and mean init_mean_wealth
    sigma_a0: float = 1.8
    init_mean_wealth: float = 2.9531   # = B1 per-capita capital at tau = 0 (scripts/calibrate.py)

    def to_dict(self):
        return asdict(self)

    def with_(self, **kw):
        return replace(self, **kw)


def utility(c, h, cfg: EconConfig, xp=None):
    """u(c,h) = c^(1-theta)/(1-theta) - chi h^(1+phi)/(1+phi); works for numpy and torch."""
    import numpy as np
    if xp is None:
        xp = np
    if cfg.theta == 1.0:
        uc = xp.log(c)
    else:
        uc = c ** (1.0 - cfg.theta) / (1.0 - cfg.theta)
    return uc - cfg.chi * h ** (1.0 + cfg.phi) / (1.0 + cfg.phi)
