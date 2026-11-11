"""HSV tax schedule (Heathcote, Storesletten & Violante, 2017) and a linear asset tax."""


def hsv_tax(x, tau: float, xi: float, ref: float = 1.0, xp=None):
    """T(x) = x - (1-tau) * x^(1-xi) / (1-xi), applied to reference-scaled income x/ref.

    xi = 0 gives the proportional tax tau * x (used in all reported experiments). For xi > 0 the
    schedule is applied to max(x, 0) to keep it well defined (a simulator safeguard).
    """
    if xi == 0.0:
        return tau * x
    import numpy as np
    if xp is None:
        xp = np
    z = xp.clip(x / ref, 0.0, None) if xp is np else x.clamp(min=0.0) / ref
    return ref * (z - (1.0 - tau) * z ** (1.0 - xi) / (1.0 - xi))


def asset_tax(a, tau_a: float):
    return tau_a * a
