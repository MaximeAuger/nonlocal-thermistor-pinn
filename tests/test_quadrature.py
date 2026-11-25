"""Quadrature, test functions and loss-consistency checks (run: pytest tests/)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exact_solution import b_of_lambda, integral_exact, DTYPE  # noqa: E402
from utils import gauss_legendre, composite_gauss_legendre, LegendreTestFunctions  # noqa: E402
import pinn_solver  # noqa: E402
import vpinn_solver  # noqa: E402


@pytest.mark.parametrize("n", [2, 5, 16, 64])
def test_gauss_exact_for_polynomials(n):
    x, w = gauss_legendre(n)
    assert abs(w.sum() - 1.0) < 1e-14
    for p in range(2 * n):                     # exact up to degree 2n-1
        assert abs(np.sum(w * x ** p) - 1.0 / (p + 1)) < 1e-13


def test_composite_rule():
    x, w = composite_gauss_legendre(50, 4)
    assert abs(np.sum(w * np.sin(np.pi * x)) - 2 / np.pi) < 1e-14


@pytest.mark.parametrize("n_quad,lam,tol", [(64, 4.0, 1e-12), (128, 7.0, 1e-10),
                                            (128, 7.9, 1e-6), (128, 7.99, 1e-3)])
def test_training_quadrature_of_I(n_quad, lam, tol):
    """Accuracy of the *training* rule for I[u_exact]; degrades near lambda = 8
    because exp(-u) develops boundary layers (documented in the README)."""
    from exact_solution import u_exact
    x, w = gauss_legendre(n_quad)
    I = np.sum(w * np.exp(-u_exact(x, lam)))
    assert abs(I - integral_exact(lam)) / integral_exact(lam) < tol


def test_test_functions_vanish_and_derivative():
    tf = LegendreTestFunctions(30)
    V = tf.eval_v(np.array([0.0, 1.0]))
    assert np.max(np.abs(V)) < 1e-12
    x = np.linspace(0.01, 0.99, 41)
    h = 1e-6
    fd = (tf.eval_v(x + h) - tf.eval_v(x - h)) / (2 * h)
    assert np.max(np.abs(fd - tf.eval_dv(x))) < 1e-5


class ExactNet(torch.nn.Module):
    """Wraps the exact solution as a torch module (to check the losses)."""

    def __init__(self, lam):
        super().__init__()
        self.b = b_of_lambda(lam)
        self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def forward(self, x):
        b = self.b
        return 2 * torch.log(torch.cos(b * (x - 0.5)) / np.cos(b / 2)) + 0 * self.dummy


@pytest.mark.parametrize("lam", [1.0, 7.0, 7.9])
def test_validation_residuals_vanish_on_exact_solution(lam):
    from utils import evaluate
    m = evaluate(ExactNet(lam), lam, 128)
    assert m["rho_val"] < 1e-8 and m["rho_weak"] < 1e-8
    assert m["L2"] < 1e-12 and m["I_rel"] < 1e-12


@pytest.mark.parametrize("problem", ["nonlocal", "local"])
@pytest.mark.parametrize("lam", [1.0, 6.0])
def test_losses_vanish_on_exact_solution(problem, lam):
    for mod, extra in [(pinn_solver, {}), (vpinn_solver, {"n_test": 20})]:
        cfg = {**mod.DEFAULT_CONFIG, "lam": lam, "problem": problem, **extra}
        loss, aux = mod.make_loss(cfg)(ExactNet(lam))
        assert aux["rel_res"] < 1e-9, (mod.__name__, aux)
        assert abs(aux["I"] - integral_exact(lam)) < 1e-12
