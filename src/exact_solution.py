"""
Exact solution of the 1-D steady non-local thermistor (Ohmic heating) problem.

PDE:  -u''(x) = lambda * f(u) / ( int_0^1 f(u) dx )^2,   x in (0, 1)
      u(0) = u(1) = 0,      f(u) = exp(-u)   (resistivity decreasing with T)

Closed-form solution (0 < lambda < 8)
-------------------------------------
Parametrise by b in (0, pi):

    lambda = 8 sin^2(b/2)              <=>  b = 2 arcsin( sqrt(lambda/8) )
    u(x)   = 2 ln( cos(b(x - 1/2)) / cos(b/2) )
    I[u]   = int_0^1 exp(-u) dx = sin(b) / b
    max u  = u(1/2) = -2 ln cos(b/2)

lambda(b) is strictly increasing on (0, pi) and maps it onto (0, 8), so the
steady problem has a unique solution for lambda < 8 and none for lambda >= 8
(thermal runaway: max u -> infinity as lambda -> 8^-).

The same function solves the *local* problem  -u'' = mu exp(-u)  with
mu = lambda / I^2 = 2 b^2 / cos^2(b/2)   (used in experiment E3).

Note on scaling: Lacey (1995) works on (-1, 1).  With x = 2y - 1 the factor
1/4 from u_xx and the factor 2^2 from (int_{-1}^{1} f dx)^2 cancel, so the
critical value lambda* = 8 is the same on (0, 1).

References
----------
    - Lacey, A.A. (1995). Thermal runaway in a non-local problem modelling
      Ohmic heating. Part I: Model derivation and some special cases.
      Eur. J. Appl. Math., 6(2), 127-144. doi:10.1017/S095679250000173X
    - Lacey, A.A. (1995). Part II: General proof of blow-up and asymptotics of
      runaway. Eur. J. Appl. Math., 6(3), 201-224. doi:10.1017/S0956792500001807

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import torch

DTYPE = torch.float64
LAMBDA_CRIT = 8.0


def f_resistivity(u):
    """f(u) = exp(-u)  (numpy or torch)."""
    if isinstance(u, torch.Tensor):
        return torch.exp(-u)
    return np.exp(-u)


def b_of_lambda(lam):
    """b = 2 arcsin(sqrt(lambda/8)), defined for 0 < lambda < 8."""
    lam = float(lam)
    if not (0.0 < lam < LAMBDA_CRIT):
        raise ValueError(f"No steady solution for lambda = {lam} (need 0 < lambda < 8)")
    return 2.0 * np.arcsin(np.sqrt(lam / 8.0))


def lambda_of_b(b):
    """lambda = 8 sin^2(b/2)."""
    return 8.0 * np.sin(0.5 * b) ** 2


def u_exact(x, lam):
    """Exact temperature u(x) for parameter lambda."""
    b = b_of_lambda(lam)
    return 2.0 * np.log(np.cos(b * (x - 0.5)) / np.cos(0.5 * b))


def u_exact_deriv(x, lam):
    """u'(x) = -2 b tan(b (x - 1/2))."""
    b = b_of_lambda(lam)
    return -2.0 * b * np.tan(b * (x - 0.5))


def u_exact_deriv2(x, lam):
    """u''(x) = -2 b^2 / cos^2(b (x - 1/2))."""
    b = b_of_lambda(lam)
    return -2.0 * b ** 2 / np.cos(b * (x - 0.5)) ** 2


def integral_exact(lam):
    """I[u] = int_0^1 exp(-u) dx = sin(b)/b."""
    b = b_of_lambda(lam)
    return np.sin(b) / b


def max_u_exact(lam):
    """max u = u(1/2) = -2 ln cos(b/2)."""
    b = b_of_lambda(lam)
    return -2.0 * np.log(np.cos(0.5 * b))


def mu_local(lam):
    """Equivalent local coefficient mu = lambda / I^2 (E3)."""
    return lam / integral_exact(lam) ** 2


if __name__ == "__main__":
    print(f"{'lambda':>7} {'b':>8} {'I[u]':>10} {'max u':>8} {'u_x(0)':>9} {'mu':>10}")
    for lam in [1, 4, 6, 7, 7.5, 7.9, 7.99]:
        print(f"{lam:7.2f} {b_of_lambda(lam):8.4f} {integral_exact(lam):10.4e} "
              f"{max_u_exact(lam):8.4f} {u_exact_deriv(0.0, lam):9.3f} {mu_local(lam):10.3e}")
