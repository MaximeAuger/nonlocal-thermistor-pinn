"""
Classical (non-neural) solvers for the non-local thermistor problem.

1. Finite differences + Newton
   -u'' = lambda exp(-u) / I^2 on a uniform grid (second-order centred
   differences, trapezoidal rule for I).  Because of the integral, the Jacobian
   is tridiagonal + rank one,  J = T + p q^T, and each Newton step costs O(N)
   with a banded solve and the Sherman-Morrison formula.

2. Amplitude continuation
   The unknowns are (u, lambda) and the amplitude m = u(1/2) is prescribed.
   Increasing m traces the solution branch towards blow-up and gives lambda(m),
   which tends to lambda* = 8 (exactly: lambda(m) = 8 (1 - e^{-m})).

3. Least-squares minimum of the VPINN weak residual (no neural network)
   u(x) = x (1-x) sum_j a_j P_j(2x-1), M coefficients; minimise
   sum_k R_k(a)^2 with the same test functions and quadrature as the VPINN.
   This is the mathematical floor that a VPINN with test space K can reach.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import time

import numpy as np
import torch
from numpy.polynomial import legendre as leg
from scipy.linalg import solve_banded
from scipy.optimize import least_squares

from exact_solution import DTYPE
from utils import gauss_legendre, LegendreTestFunctions


# ===================================================================
# 1. Finite differences + Newton
# ===================================================================

def _residual(u_int, lam, h):
    """F(u) for interior unknowns, with u_0 = u_N = 0."""
    u = np.concatenate(([0.0], u_int, [0.0]))
    w = np.full(u.size, h)
    w[0] = w[-1] = 0.5 * h                       # trapezoidal weights
    e = np.exp(-u)
    I = np.sum(w * e)
    lap = (-u[:-2] + 2.0 * u[1:-1] - u[2:]) / h ** 2
    return lap - lam * e[1:-1] / I ** 2, e, w, I


def _solve_T_plus_rank1(diag, h, p, q, rhs_list):
    """Solve (T + p q^T) x = rhs, T tridiagonal (-1, diag, -1)/h^2, via Sherman-Morrison."""
    n = diag.size
    ab = np.zeros((3, n))
    ab[0, 1:] = -1.0 / h ** 2
    ab[1, :] = diag
    ab[2, :-1] = -1.0 / h ** 2
    Y = solve_banded((1, 1), ab, np.column_stack([p] + rhs_list))
    z, xs = Y[:, 0], Y[:, 1:]
    denom = 1.0 + q @ z
    return [xs[:, i] - z * (q @ xs[:, i]) / denom for i in range(xs.shape[1])]


def _jacobian_parts(u_int, lam, h, e, w, I):
    """J = T(diag) + p q^T.

    dF_i/du_j through I:  -lam e_i * (-2 I^-3) * dI/du_j  with  dI/du_j = -w_j e_j
                       =  -2 lam e_i w_j e_j / I^3   ->  p_i = -2 lam e_i / I^3,  q_j = w_j e_j
    """
    ei = e[1:-1]
    diag = 2.0 / h ** 2 + lam * ei / I ** 2           # direct d/du_i of -lam e^{-u_i}/I^2
    p = -2.0 * lam * ei / I ** 3
    q = w[1:-1] * ei
    return diag, p, q


def newton_fd(lam, N=800, u0=None, tol=1e-12, max_iter=50):
    """Newton on the FD system. Returns (x, u, n_iter, converged)."""
    h = 1.0 / N
    u_int = np.zeros(N - 1) if u0 is None else u0.copy()
    for it in range(1, max_iter + 1):
        F, e, w, I = _residual(u_int, lam, h)
        if not np.all(np.isfinite(F)):
            return None, None, it, False
        diag, p, q = _jacobian_parts(u_int, lam, h, e, w, I)
        (du,) = _solve_T_plus_rank1(diag, h, p, q, [-F])
        u_int += du
        if not np.all(np.isfinite(u_int)):
            return None, None, it, False
        if np.max(np.abs(du)) < tol * max(1.0, np.max(np.abs(u_int))):
            x = np.linspace(0.0, 1.0, N + 1)
            return x, np.concatenate(([0.0], u_int, [0.0])), it, True
    return None, None, max_iter, False


def solve_with_continuation(lam, N=800, steps=None):
    """Natural continuation in lambda from 0 (warm starts)."""
    steps = steps or list(np.linspace(0.0, lam, 11)[1:])
    u, total = None, 0
    for l in steps:
        x, uf, it, ok = newton_fd(l, N, None if u is None else u[1:-1])
        total += it
        if not ok:
            return None, None, total, False
        u = uf
    return x, u, total, True


# ===================================================================
# 2. Amplitude continuation (u(1/2) = m prescribed, lambda unknown)
# ===================================================================

def amplitude_point(m, N=4000, u0=None, lam0=None, tol=1e-12, max_iter=60):
    """Solve F(u, lambda) = 0 with u(1/2) = m. N must be even."""
    h = 1.0 / N
    mid = N // 2 - 1                                  # index of x = 1/2 among interior nodes
    xi = np.linspace(h, 1.0 - h, N - 1)
    u_int = (4.0 * m * xi * (1 - xi)) if u0 is None else u0.copy()
    lam = 8.0 * (1.0 - np.exp(-m)) if lam0 is None else lam0
    for it in range(1, max_iter + 1):
        F, e, w, I = _residual(u_int, lam, h)
        G = u_int[mid] - m
        diag, p, q = _jacobian_parts(u_int, lam, h, e, w, I)
        J_lam = -e[1:-1] / I ** 2
        x1, x2 = _solve_T_plus_rank1(diag, h, p, q, [-F, J_lam])
        dlam = (x1[mid] + G) / x2[mid]
        du = x1 - dlam * x2
        u_int += du
        lam += dlam
        if max(np.max(np.abs(du)), abs(dlam)) < tol * max(1.0, m):
            return lam, u_int, it, True
    return lam, u_int, max_iter, False


def amplitude_branch(ms, N=4000):
    """lambda(m) along the branch, with Richardson extrapolation from N and 2N
    (the FD error is O(h^2)): lambda_R = (4 lambda_2N - lambda_N) / 3."""
    out, uN, lN, u2, l2 = [], None, None, None, None
    for m in ms:
        lN, uN, itN, okN = amplitude_point(m, N, uN, lN)
        l2, u2, it2, ok2 = amplitude_point(m, 2 * N, u2, l2)
        out.append({"m": m, "lam_N": lN, "lam_2N": l2, "lam_R": (4.0 * l2 - lN) / 3.0,
                    "iters": itN + it2, "ok": okN and ok2})
    return out


# ===================================================================
# 3. Least-squares minimum of the weak residual (spectral ansatz)
# ===================================================================

class SpectralNet(torch.nn.Module):
    """u(x) = x(1-x) sum_j a_j P_j(2x-1): a torch module so that utils.evaluate applies."""

    def __init__(self, a):
        super().__init__()
        self.a = np.asarray(a, dtype=float)
        self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def forward(self, x):
        xi = 2.0 * x - 1.0
        # Clenshaw-free evaluation with the three-term recurrence (differentiable in x)
        P0, P1 = torch.ones_like(xi), xi
        s = self.a[0] * P0 + (self.a[1] * P1 if self.a.size > 1 else 0.0)
        for n in range(1, self.a.size - 1):
            P0, P1 = P1, ((2 * n + 1) * xi * P1 - n * P0) / (n + 1)
            s = s + self.a[n + 1] * P1
        return x * (1.0 - x) * s + 0.0 * self.dummy


class WeakResidualLS:
    """Residual vector R(a) of the VPINN weak form for the spectral ansatz."""

    def __init__(self, lam, K, M=80, n_quad=128):
        x, w = gauss_legendre(n_quad)
        tf = LegendreTestFunctions(K)
        self.V, self.dV, self.w = tf.eval_v(x), tf.eval_dv(x), w
        xi = 2.0 * x - 1.0
        P = np.stack([leg.legval(xi, np.eye(M)[j]) for j in range(M)], axis=1)          # (Nq, M)
        dP = np.stack([2.0 * leg.legval(xi, leg.legder(np.eye(M)[j])) for j in range(M)], axis=1)
        b = (x * (1 - x))[:, None]
        self.B = b * P                                   # u   = B a
        self.dB = (1 - 2 * x)[:, None] * P + b * dP      # u'  = dB a
        self.lam = lam

    def R(self, a):
        u, du = self.B @ a, self.dB @ a
        e = np.exp(-u)
        I = self.w @ e
        load = self.lam / I ** 2 * ((self.w * e) @ self.V)
        return (self.w * du) @ self.dV - load, load

    def fun(self, a):
        return self.R(a)[0]

    def jac(self, a):
        u = self.B @ a
        e = np.exp(-u)
        I = self.w @ e
        we = self.w * e
        S = we @ self.V                                  # (K,)
        dI = -(we @ self.B)                              # (M,)
        dS = -(self.V.T * we) @ self.B                   # (K, M)
        stiff = (self.dV.T * self.w) @ self.dB           # (K, M)
        dload = self.lam * (dS / I ** 2 - 2.0 * np.outer(S, dI) / I ** 3)
        return stiff - dload


def ls_minimum(lam, K, M=80, n_quad=128, a0=None, max_nfev=3000):
    prob = WeakResidualLS(lam, K, M, n_quad)
    a0 = np.zeros(M) if a0 is None else a0
    t0 = time.perf_counter()
    sol = least_squares(prob.fun, a0, jac=prob.jac, method="trf", xtol=1e-15, ftol=1e-15,
                        gtol=1e-15, max_nfev=max_nfev)   # note: x_scale="jac" diverges here
    R, load = prob.R(sol.x)
    return {"a": sol.x, "loss": float(R @ R), "rel": float(np.linalg.norm(R) / np.linalg.norm(load)),
            "nfev": sol.nfev, "time": time.perf_counter() - t0}
