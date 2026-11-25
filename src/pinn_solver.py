"""
PINN solver (strong form) for the non-local thermistor problem.

    -u'' = lambda exp(-u) / I[u]^2,     I[u] = int_0^1 exp(-u) dx
    u(0) = u(1) = 0   (imposed exactly by the ansatz u = x(1-x) N(x))

Loss (single term, no penalty weights):

    L_PINN = (1/N) sum_i [ -u''(x_i) - lambda exp(-u(x_i)) / I[u]^2 ]^2

* u'' by automatic differentiation.
* I[u] by Gauss-Legendre quadrature with the *same* nodes as the VPINN,
  recomputed at every iteration and differentiated end-to-end (the gradient
  of the loss contains the non-local coupling dI/dtheta).
* Collocation points x_i = the Gauss-Legendre nodes (identical point set
  for both methods; plain mean, as in a standard PINN).

With problem="local" the non-local factor is replaced by the constant
mu = lambda / I_exact^2 (experiment E3).

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import torch

from utils import (COMMON_DEFAULTS, gauss_legendre, diff, to_t, build_net, train_loop, evaluate,
                   LocalCoefficient, NonlocalLambda)

DEFAULT_CONFIG = dict(COMMON_DEFAULTS)


def collocation_points(cfg):
    """Residual points: the Gauss nodes (default), a uniform grid or fixed random points."""
    n = cfg.get("n_col") or cfg["n_quad"]
    kind = cfg.get("collocation", "gauss")
    if kind == "gauss":
        return gauss_legendre(n)[0]
    if kind == "uniform":
        return np.linspace(0.0, 1.0, n + 2)[1:-1]
    if kind == "random":
        return np.sort(np.random.default_rng(1000 + cfg["seed"]).uniform(0.0, 1.0, n))
    raise ValueError(kind)


def make_loss(cfg):
    xq, wq = gauss_legendre(cfg["n_quad"])
    w = to_t(wq)
    same = cfg.get("collocation", "gauss") == "gauss" and not cfg.get("n_col")
    x = to_t(xq).requires_grad_(True)
    xc = x if same else to_t(collocation_points(cfg)).requires_grad_(True)
    local = cfg["problem"] == "local"
    coef_local = LocalCoefficient(cfg) if local else None
    lam_fn = None if local else NonlocalLambda(cfg)   # optional lambda ramp

    def loss_fn(net):
        u = net(x)
        e = torch.exp(-u)
        I = torch.sum(w * e)                      # always on the Gauss rule
        uc = u if same else net(xc)
        u_xx = diff(uc, xc, order=2)
        ec = e if same else torch.exp(-uc)
        coef = coef_local() if local else lam_fn() / I ** 2
        s = coef * ec
        r = -u_xx - s
        loss = torch.mean(r ** 2)
        with torch.no_grad():
            rel = float(torch.sqrt(loss / torch.mean(s ** 2)))
        return loss, {"I": float(I), "rel_res": rel}

    loss_fn.set_iter = (coef_local if local else lam_fn).set_iter
    return loss_fn


def train_pinn(config=None, verbose=True):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    net = build_net(cfg)
    hist, elapsed, status = train_loop(net, make_loss(cfg), cfg,
                                       tag=f"[PINN  lam={cfg['lam']}]", verbose=verbose)
    return {"method": "PINN", "net": net, "history": hist, "time": elapsed,
            "status": status, "metrics": evaluate(net, cfg["lam"], cfg["n_quad"]), "config": cfg}


if __name__ == "__main__":
    res = train_pinn({"lam": 4.0, "n_adam": 1000, "n_lbfgs": 500})
    print({k: v for k, v in res["metrics"].items() if not k.endswith("plot")})
