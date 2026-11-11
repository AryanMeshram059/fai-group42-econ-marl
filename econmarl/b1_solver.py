"""C2: known-model benchmark B1 -- stationary recursive competitive equilibrium (Aiyagari with
endogenous labour, proportional income tax, linear asset tax and lump-sum rebate).

Household problem at constant prices (W, r) and rebate nu:
    V(a, e) = max_{a' >= 0, h in [h_min, h_max]} u(c, h) + beta E[V(a', e') | e]
    c + a' = Ra * a + we * h + nu,    Ra = 1 + (1 - tau) r - tau_a,   we = (1 - tau) W e
solved with the endogenous grid method (EGM) with endogenous labour (intratemporal FOC
chi h^phi = c^-theta we). The stationary distribution is computed with Young's (2010) histogram
method, and (kappa*, nu*) with Brent root finding on K^s/L^s - kappa and a damped fixed point on nu.
A brute-force value function iteration (VFI) solver is included for cross-checking.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

from .config import EconConfig, utility
from .markov import productivity_process


def asset_grid(n=200, a_max=400.0, power=2.5):
    return a_max * np.linspace(0.0, 1.0, n) ** power


def _interp_extrap(x, xp, fp):
    """Linear interpolation with linear extrapolation on both sides (xp increasing, 1-D)."""
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    if lo.any():
        s = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[lo] = fp[0] + s * (x[lo] - xp[0])
    if hi.any():
        s = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[hi] = fp[-1] + s * (x[hi] - xp[-1])
    return y


@dataclass
class HouseholdSolution:
    a_grid: np.ndarray      # [na]
    e: np.ndarray           # [ne]
    P: np.ndarray           # [ne, ne]
    c: np.ndarray           # [na, ne]
    h: np.ndarray           # [na, ne]
    a_next: np.ndarray      # [na, ne]
    W: float
    r: float
    nu: float
    Ra: float
    cfg: EconConfig
    iters: int = 0

    def we(self):
        return (1 - self.cfg.tau) * self.W * self.e

    def policy(self, a, e_idx):
        """Evaluate (c, h, a') at arbitrary assets a (array) and productivity indices e_idx."""
        a = np.asarray(a, dtype=float)
        e_idx = np.asarray(e_idx)
        c = np.empty_like(a); h = np.empty_like(a); an = np.empty_like(a)
        for k in range(len(self.e)):
            msk = e_idx == k
            if not msk.any():
                continue
            ak = a[msk]
            h[msk] = np.clip(_interp_extrap(ak, self.a_grid, self.h[:, k]), self.cfg.h_min, self.cfg.h_max)
            an[msk] = np.maximum(_interp_extrap(ak, self.a_grid, self.a_next[:, k]), 0.0)
            m = self.Ra * ak + self.we()[k] * h[msk] + self.nu
            c[msk] = m - an[msk]
        return c, h, an

    def actions(self, a, e_idx):
        """Map the DP policy into the simulator's normalised actions (p_hat, h_hat)."""
        cfg = self.cfg
        c, h, an = self.policy(a, e_idx)
        m = an + c
        p = np.clip(an / np.maximum(m, 1e-12), cfg.p_min, cfg.p_max)
        p_hat = (p - cfg.p_min) / (cfg.p_max - cfg.p_min)
        h_hat = (h - cfg.h_min) / (cfg.h_max - cfg.h_min)
        return p_hat, h_hat


def _labor_from_c(c, we, cfg):
    return np.clip((c ** (-cfg.theta) * we / cfg.chi) ** (1.0 / cfg.phi), cfg.h_min, cfg.h_max)


def _constrained_labor(a, we, Ra, nu, cfg, iters=80):
    """Solve chi h^phi = (Ra a + we h + nu)^-theta we for h (a' = 0), vectorised bisection."""
    lo = np.full(np.broadcast(a, we).shape, cfg.h_min)
    hi = np.full_like(lo, cfg.h_max)
    base = Ra * a + nu

    def f(h):
        c = np.maximum(base + we * h, 1e-100)
        with np.errstate(over="ignore"):
            return cfg.chi * h ** cfg.phi - c ** (-cfg.theta) * we

    flo, fhi = f(lo), f(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        neg = fm < 0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    h = 0.5 * (lo + hi)
    h = np.where(fhi <= 0, cfg.h_max, h)
    h = np.where(flo >= 0, cfg.h_min, h)
    return h


def solve_household(cfg: EconConfig, W: float, r: float, nu: float, a_grid=None,
                    tol=1e-10, max_iter=5000, c_init=None, horizon=None) -> HouseholdSolution:
    """EGM for the stationary problem. If ``horizon`` is given, run exactly ``horizon`` backward
    steps from a terminal period that consumes everything (the period-0 policy of a finite-horizon
    problem); this is used for best responses when beta*Ra >= 1 and no stationary solution exists."""
    e, P, _ = productivity_process(cfg)
    a_grid = asset_grid() if a_grid is None else a_grid
    Ra = 1.0 + (1.0 - cfg.tau) * r - cfg.tau_a
    if not (0 < Ra and (horizon is not None or cfg.beta * Ra < 1.0)):
        raise ValueError(f"after-tax gross return {Ra} violates 0 < beta*Ra < 1")
    we = (1.0 - cfg.tau) * W * e                          # [ne]
    A = a_grid[:, None]
    # constrained solution (a' = 0) is independent of the iteration
    h_con = _constrained_labor(A, we[None, :], Ra, nu, cfg)
    c_con = Ra * A + we[None, :] * h_con + nu
    if horizon is not None:
        c, max_iter, tol = c_con.copy(), horizon, -1.0
    else:
        c = c_con.copy() + 0.05 * Ra * A if c_init is None else c_init.copy()
    it = 0
    for it in range(1, max_iter + 1):
        emu = (c ** (-cfg.theta)) @ P.T                   # [na', ne]: E[c(a',e')^-theta | e]
        c_endo = (cfg.beta * Ra * emu) ** (-1.0 / cfg.theta)
        h_endo = _labor_from_c(c_endo, we[None, :], cfg)
        a_endo = (c_endo + A - we[None, :] * h_endo - nu) / Ra
        c_new = np.empty_like(c)
        for k in range(len(e)):
            ck = _interp_extrap(a_grid, a_endo[:, k], c_endo[:, k])
            con = a_grid < a_endo[0, k]
            ck[con] = c_con[con, k]
            c_new[:, k] = ck
        c_new = np.maximum(c_new, 1e-12)
        diff = np.max(np.abs(c_new - c) / (1.0 + c))
        c = c_new
        if diff < tol:
            break
    h = _labor_from_c(c, we[None, :], cfg)
    # in the constrained region h comes from the constrained FOC
    a_next = Ra * A + we[None, :] * h + nu - c
    con = a_next <= 1e-12
    h = np.where(con, h_con, h)
    c = np.where(con, c_con, c)
    a_next = np.where(con, 0.0, a_next)
    return HouseholdSolution(a_grid, e, P, c, h, a_next, W, r, nu, Ra, cfg, it)


def euler_errors(sol: HouseholdSolution):
    """log10 relative Euler equation errors at unconstrained grid points."""
    cfg = sol.cfg
    errs = []
    for k in range(len(sol.e)):
        an = sol.a_next[:, k]
        ok = an > 1e-8
        cn = np.stack([sol.policy(an[ok], np.full(ok.sum(), j))[0] for j in range(len(sol.e))], 1)
        rhs = cfg.beta * sol.Ra * (cn ** (-cfg.theta)) @ sol.P[k]
        c_imp = rhs ** (-1.0 / cfg.theta)
        errs.append(np.abs(c_imp / sol.c[ok, k] - 1.0))
    e = np.concatenate(errs)
    return np.log10(np.maximum(e, 1e-16))


# ---------------------------------------------------------------- stationary distribution
def stationary_distribution(sol: HouseholdSolution, n_fine=1000, tol=1e-13, max_iter=20000):
    a_f = asset_grid(n_fine, sol.a_grid[-1])
    ne = len(sol.e)
    na = len(a_f)
    rows, cols, vals = [], [], []
    for k in range(ne):
        an = np.clip(_interp_extrap(a_f, sol.a_grid, sol.a_next[:, k]), 0.0, a_f[-1])
        j = np.clip(np.searchsorted(a_f, an, side="right") - 1, 0, na - 2)
        w_hi = (an - a_f[j]) / (a_f[j + 1] - a_f[j])
        src = np.arange(na) * ne + k
        for kp in range(ne):
            pr = sol.P[k, kp]
            if pr <= 0:
                continue
            rows += [src, src]
            cols += [j * ne + kp, (j + 1) * ne + kp]
            vals += [pr * (1 - w_hi), pr * w_hi]
    T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(na * ne, na * ne))
    TT = T.T.tocsr()
    _, _, pi = productivity_process(sol.cfg)
    mu = np.outer(np.full(na, 1.0 / na), pi).ravel()
    for _ in range(max_iter):
        mu_new = TT @ mu
        if np.abs(mu_new - mu).sum() < tol:
            mu = mu_new
            break
        mu = mu_new
    mu = mu.reshape(na, ne)
    return a_f, mu / mu.sum()


def gini_from_hist(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w); cxw = np.cumsum(x * w)
    tot = cxw[-1]
    # area under the Lorenz curve with trapezoids
    L = cxw / tot
    Lp = np.concatenate([[0.0], L[:-1]])
    B = np.sum(w * (L + Lp) / 2.0)
    return 1.0 - 2.0 * B


def top_share_from_hist(x, w, q=0.10):
    o = np.argsort(x)[::-1]
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    take = np.clip(q - (cw - w), 0.0, w)
    return np.sum(take * x) / np.sum(w * x)


@dataclass
class Equilibrium:
    cfg: EconConfig
    kappa: float
    W: float
    r: float
    nu: float
    sol: HouseholdSolution
    a_fine: np.ndarray
    mu: np.ndarray
    stats: dict


def aggregate(sol: HouseholdSolution, a_f, mu):
    cfg = sol.cfg
    ne = len(sol.e)
    c = np.empty_like(mu); h = np.empty_like(mu); an = np.empty_like(mu)
    for k in range(ne):
        c[:, k], h[:, k], an[:, k] = sol.policy(a_f, np.full(len(a_f), k))
    A = a_f[:, None]
    E = sol.e[None, :]
    y = sol.W * E * h + sol.r * A
    S = np.sum(mu * (cfg.tau * y + cfg.tau_a * A))
    K = np.sum(mu * A); L = np.sum(mu * E * h)
    m = an + c
    wealth = np.repeat(a_f, ne)
    return dict(K=K, L=L, kappa=K / L, hours=np.sum(mu * h), C=np.sum(mu * c), S=S,
                saving_rate=np.sum(mu * an) / np.sum(mu * m),
                mean_p=np.sum(mu * an / np.maximum(m, 1e-12)),
                gini=gini_from_hist(wealth, mu.ravel()),
                top10=top_share_from_hist(wealth, mu.ravel(), 0.10),
                frac_constrained=np.sum(mu[an <= 1e-10]))


def prices_from_kappa(cfg, kappa):
    W = (1 - cfg.alpha) * kappa ** cfg.alpha
    R = cfg.alpha * kappa ** (cfg.alpha - 1)
    return W, R - cfg.delta


def solve_equilibrium(cfg: EconConfig, verbose=False, xtol=1e-9) -> Equilibrium:
    cache = {}

    def inner(kappa):
        W, r = prices_from_kappa(cfg, kappa)
        nu = 0.0
        c_init = None
        for _ in range(200):
            sol = solve_household(cfg, W, r, nu, c_init=c_init)
            c_init = sol.c
            a_f, mu = stationary_distribution(sol)
            st = aggregate(sol, a_f, mu)
            nu_new = cfg.psi * st["S"]
            if abs(nu_new - nu) < 1e-11 or cfg.psi * cfg.tau == 0 and cfg.psi * cfg.tau_a == 0:
                nu = nu_new
                break
            nu = 0.5 * nu + 0.5 * nu_new
        cache[kappa] = (W, r, nu, sol, a_f, mu, st)
        if verbose:
            print(f"  kappa={kappa:.6f} r={r:.5f} nu={nu:.5f} K/L={st['kappa']:.6f}")
        return st["kappa"] - kappa

    # bracket: after-tax return must satisfy beta * Ra < 1
    r_hi = ((1.0 / cfg.beta - 1.0) + cfg.tau_a) / (1.0 - cfg.tau) - 1e-4
    k_lo = (cfg.alpha / (r_hi + cfg.delta)) ** (1.0 / (1.0 - cfg.alpha))
    k_hi = (cfg.alpha / (0.0 + cfg.delta)) ** (1.0 / (1.0 - cfg.alpha))
    while inner(k_hi) > 0:
        k_hi *= 1.3
    kappa = brentq(inner, k_lo, k_hi, xtol=xtol)
    if kappa not in cache:
        inner(kappa)
    W, r, nu, sol, a_f, mu, st = cache[kappa]
    st = dict(st, W=W, r=r, nu=nu, kappa_eq=kappa, Y=st["K"] ** cfg.alpha * st["L"] ** (1 - cfg.alpha))
    return Equilibrium(cfg, kappa, W, r, nu, sol, a_f, mu, st)


# ---------------------------------------------------------------- VFI cross-check
def solve_household_vfi(cfg: EconConfig, W, r, nu, a_grid=None, tol=1e-9, max_iter=3000):
    """Brute-force VFI with a' restricted to the grid (used only to validate EGM)."""
    e, P, _ = productivity_process(cfg)
    a_grid = asset_grid(300, 100.0, 2.0) if a_grid is None else a_grid
    Ra = 1.0 + (1.0 - cfg.tau) * r - cfg.tau_a
    we = (1.0 - cfg.tau) * W * e
    na, ne = len(a_grid), len(e)
    A = a_grid[:, None, None]; Ap = a_grid[None, :, None]; WE = we[None, None, :]
    # optimal h for each (a, a', e): chi h^phi = (Ra a + we h + nu - a')^-theta we  (bisection)
    lo = np.full((na, na, ne), cfg.h_min); hi = np.full((na, na, ne), cfg.h_max)
    base = Ra * A + nu - Ap
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cc = base + WE * mid
        f = np.where(cc > 0, cfg.chi * mid ** cfg.phi - np.maximum(cc, 1e-300) ** (-cfg.theta) * WE, -1.0)
        lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
    h = 0.5 * (lo + hi)
    c = base + WE * h
    u = np.where(c > 1e-10, utility(np.maximum(c, 1e-10), h, cfg), -1e10)
    V = np.zeros((na, ne))
    for it in range(max_iter):
        EV = V @ P.T                                  # [a', e]
        Q = u + cfg.beta * EV[None, :, :]
        V_new = Q.max(1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    j = Q.argmax(1)
    an = a_grid[j]
    hh = np.take_along_axis(h, j[:, None, :], 1)[:, 0, :]
    cc = np.take_along_axis(c, j[:, None, :], 1)[:, 0, :]
    return dict(a_grid=a_grid, a_next=an, h=hh, c=cc, V=V, iters=it)
