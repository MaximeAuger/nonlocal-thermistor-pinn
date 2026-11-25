"""Checks of the classical solvers (run: pytest tests/)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exact_solution import u_exact  # noqa: E402
from classical_solver import (newton_fd, amplitude_point, WeakResidualLS, ls_minimum,  # noqa: E402
                              SpectralNet, _residual, _jacobian_parts)


def test_fd_jacobian_against_finite_differences():
    rng = np.random.default_rng(0)
    N, lam, h = 40, 6.0, 1.0 / 40
    u = rng.uniform(0, 1, N - 1)
    F, e, w, I = _residual(u, lam, h)
    diag, p, q = _jacobian_parts(u, lam, h, e, w, I)
    J = np.diag(diag) - (np.eye(N - 1, k=1) + np.eye(N - 1, k=-1)) / h ** 2 + np.outer(p, q)
    eps = 1e-7
    Jfd = np.stack([(_residual(u + eps * np.eye(N - 1)[j], lam, h)[0] - F) / eps for j in range(N - 1)], axis=1)
    assert np.max(np.abs(J - Jfd)) / np.max(np.abs(J)) < 1e-6


@pytest.mark.parametrize("lam", [1.0, 4.0, 7.0])
def test_fd_second_order_convergence(lam):
    errs = []
    for N in (100, 200, 400):
        x, u, it, ok = newton_fd(lam, N)
        assert ok and it <= 12
        errs.append(np.max(np.abs(u - u_exact(x, lam))))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.9), rates


def test_amplitude_continuation_recovers_lambda():
    from classical_solver import amplitude_branch
    for r in amplitude_branch([1.0, 3.0, 6.0], N=2000):
        assert r["ok"]
        exact = 8.0 * (1.0 - np.exp(-r["m"]))
        assert abs(r["lam_N"] - exact) < 5e-3            # O(h^2) with boundary layers
        assert abs(r["lam_R"] - exact) < 1e-6            # Richardson (N, 2N)


def test_weak_ls_jacobian_and_exact_zero():
    prob = WeakResidualLS(6.0, K=20, M=30)
    a = np.random.default_rng(1).normal(0, 0.3, 30)
    eps = 1e-7
    J = prob.jac(a)
    Jfd = np.stack([(prob.fun(a + eps * np.eye(30)[j]) - prob.fun(a)) / eps for j in range(30)], axis=1)
    assert np.max(np.abs(J - Jfd)) / np.max(np.abs(J)) < 1e-5
    from utils import evaluate
    # square Petrov-Galerkin system (M = K): converges to the exact solution
    res = ls_minimum(4.0, K=20, M=20)
    assert res["rel"] < 1e-10
    assert evaluate(SpectralNet(res["a"]), 4.0)["L2"] < 1e-6
    # under-determined (M > K): the residual still vanishes but the zero set is a
    # manifold, and the returned member is NOT the exact solution
    res = ls_minimum(4.0, K=20, M=30)
    assert res["rel"] < 1e-10
    assert evaluate(SpectralNet(res["a"]), 4.0)["L2"] > 1e-5
