"""Checks of the closed-form solution (run: pytest tests/)."""

import os
import sys

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exact_solution import (  # noqa: E402
    b_of_lambda, lambda_of_b, u_exact, u_exact_deriv, u_exact_deriv2,
    integral_exact, max_u_exact, mu_local,
)
from utils import composite_gauss_legendre  # noqa: E402

LAMBDAS = [0.1, 1.0, 4.0, 6.0, 7.0, 7.5, 7.9, 7.99]


def test_symbolic_residual_and_integral():
    """sympy: -u'' - lambda e^{-u} / I^2 == 0 with I = sin(b)/b, lambda = 8 sin^2(b/2)."""
    x, b = sp.symbols("x b", positive=True)
    u = 2 * sp.log(sp.cos(b * (x - sp.Rational(1, 2))) / sp.cos(b / 2))
    I = sp.integrate(sp.exp(-u), (x, 0, 1))
    assert sp.simplify(sp.expand_trig(sp.simplify(I - sp.sin(b) / b))) == 0 or all(
        abs(float((I - sp.sin(b) / b).subs(b, bv))) < 1e-12 for bv in [0.2, 1.0, 2.5, 3.1])
    lam = 8 * sp.sin(b / 2) ** 2
    res = -sp.diff(u, x, 2) - lam * sp.exp(-u) / (sp.sin(b) / b) ** 2
    # change of variable t = b (x - 1/2) lets sympy close tan^2 + 1 - sec^2 = 0
    t = sp.Symbol("t")
    assert sp.simplify(res.subs(x, t / b + sp.Rational(1, 2))) == 0
    assert sp.simplify(u.subs(x, 0)) == 0 and sp.simplify(u.subs(x, 1)) == 0


@pytest.mark.parametrize("lam", LAMBDAS)
def test_lambda_b_roundtrip(lam):
    b = b_of_lambda(lam)
    assert 0 < b < np.pi
    assert abs(lambda_of_b(b) - lam) < 1e-13 * max(1, lam)


@pytest.mark.parametrize("lam", LAMBDAS)
def test_numerical_residual(lam):
    x, w = composite_gauss_legendre(2000, 8)
    I_num = np.sum(w * np.exp(-u_exact(x, lam)))
    assert abs(I_num - integral_exact(lam)) < 1e-12
    r = -u_exact_deriv2(x, lam) - lam * np.exp(-u_exact(x, lam)) / integral_exact(lam) ** 2
    assert np.max(np.abs(r)) < 1e-9 * max(1.0, mu_local(lam))
    # local reformulation (E3) has the same solution
    r_loc = -u_exact_deriv2(x, lam) - mu_local(lam) * np.exp(-u_exact(x, lam))
    assert np.max(np.abs(r_loc)) < 1e-9 * max(1.0, mu_local(lam))


@pytest.mark.parametrize("lam", [1.0, 7.0, 7.9])
def test_derivatives_finite_difference(lam):
    x = np.linspace(0.05, 0.95, 37)
    h = 1e-5
    fd1 = (u_exact(x + h, lam) - u_exact(x - h, lam)) / (2 * h)
    assert np.max(np.abs(fd1 - u_exact_deriv(x, lam))) < 1e-6 * max(1, abs(u_exact_deriv(0, lam)))
    h = 1e-4
    fd2 = (u_exact(x + h, lam) - 2 * u_exact(x, lam) + u_exact(x - h, lam)) / h ** 2
    assert np.max(np.abs(fd2 - u_exact_deriv2(x, lam))) < 1e-4 * max(1, abs(u_exact_deriv2(0, lam)))


def test_boundary_values_and_max():
    for lam in LAMBDAS:
        assert abs(u_exact(0.0, lam)) < 1e-14 and abs(u_exact(1.0, lam)) < 1e-14
        assert abs(u_exact(0.5, lam) - max_u_exact(lam)) < 1e-14


def test_runaway():
    """max u grows without bound as lambda -> 8^-, and lambda >= 8 has no solution."""
    m = [max_u_exact(8 - 10.0 ** (-k)) for k in range(1, 8)]
    assert all(np.diff(m) > 0) and m[-1] > 15
    for lam in [8.0, 9.0, 12.0, 0.0, -1.0]:
        with pytest.raises(ValueError):
            b_of_lambda(lam)
