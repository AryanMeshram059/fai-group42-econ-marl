"""Rouwenhorst discretisation of an AR(1) in logs (Kopecky & Suen, 2010)."""
import numpy as np


def rouwenhorst(n: int, rho: float, sigma: float):
    """Return (grid of log e, transition matrix P[i, j] = Pr(j | i)) for log e' = rho log e + sigma u."""
    p = (1.0 + rho) / 2.0
    P = np.array([[p, 1 - p], [1 - p, p]])
    for m in range(3, n + 1):
        Z = np.zeros((m, m))
        Z[:-1, :-1] += p * P
        Z[:-1, 1:] += (1 - p) * P
        Z[1:, :-1] += (1 - p) * P
        Z[1:, 1:] += p * P
        Z[1:-1, :] /= 2.0
        P = Z
    sd = sigma / np.sqrt(1.0 - rho ** 2)
    psi = np.sqrt(n - 1) * sd
    grid = np.linspace(-psi, psi, n)
    return grid, P


def stationary(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    i = np.argmin(np.abs(w - 1.0))
    pi = np.real(v[:, i])
    pi = pi / pi.sum()
    return pi


def productivity_process(cfg):
    """Productivity levels normalised so that E[e] = 1 under the stationary distribution."""
    logg, P = rouwenhorst(cfg.n_e, cfg.rho_e, cfg.sigma_e)
    e = np.exp(logg)
    pi = stationary(P)
    e = e / (pi @ e)
    return e, P, pi
