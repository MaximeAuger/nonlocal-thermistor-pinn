"""
VPINN solver (weak form) for the non-local thermistor problem.

Weak form: for every test function v_k with v_k(0) = v_k(1) = 0,

    int_0^1 u' v_k' dx  =  (lambda / I[u]^2) int_0^1 exp(-u) v_k dx,   k = 1..K

Test functions: v_k(x) = P_{k+1}(xi) - P_{k-1}(xi),  xi = 2x - 1  (Legendre).
No boundary term appears because the test functions vanish at both ends and
the trial function satisfies u(0) = u(1) = 0 exactly (hard ansatz).

Loss (single term):

    L_VPINN = sum_k R_k^2,
    R_k = sum_q w_q u'(x_q) v_k'(x_q) - (lambda / I^2) sum_q w_q exp(-u(x_q)) v_k(x_q)

Only first derivatives of the network are needed.  I[u] uses exactly the
same Gauss-Legendre nodes/weights as the PINN.

Reference
---------
    - Kharazmi, E., Zhang, Z. & Karniadakis, G.E. (2021). hp-VPINNs:
      Variational physics-informed neural networks with domain decomposition.
      Comput. Methods Appl. Mech. Engrg., 374, 113547.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import torch

from exact_solution import mu_local, DTYPE
from utils import (COMMON_DEFAULTS, gauss_legendre, LegendreTestFunctions, diff,
                   to_t, build_net, train_loop, evaluate, DEVICE)

DEFAULT_CONFIG = {**COMMON_DEFAULTS, "n_test": 20}


def make_loss(cfg):
    xq, wq = gauss_legendre(cfg["n_quad"])
    tf = LegendreTestFunctions(cfg["n_test"])
    V = torch.tensor(tf.eval_v(xq), dtype=DTYPE, device=DEVICE)     # (Nq, K)
    dV = torch.tensor(tf.eval_dv(xq), dtype=DTYPE, device=DEVICE)   # (Nq, K)
    x = to_t(xq).requires_grad_(True)
    w = to_t(wq)
    lam = cfg["lam"]
    local = cfg["problem"] == "local"
    mu = mu_local(lam) if local else None

    def loss_fn(net):
        u = net(x)
        u_x = diff(u, x, order=1)
        e = torch.exp(-u)
        I = torch.sum(w * e)
        coef = mu if local else lam / I ** 2
        stiff = (w * u_x).T @ dV            # (1, K)  int u' v_k'
        load = coef * ((w * e).T @ V)       # (1, K)  coef * int e^{-u} v_k
        R = stiff - load
        loss = torch.sum(R ** 2)
        with torch.no_grad():
            rel = float(torch.sqrt(loss / torch.sum(load ** 2)))
        return loss, {"I": float(I), "rel_res": rel}

    return loss_fn


def train_vpinn(config=None, verbose=True):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    net = build_net(cfg)
    hist, elapsed, status = train_loop(net, make_loss(cfg), cfg,
                                       tag=f"[VPINN lam={cfg['lam']}]", verbose=verbose)
    return {"method": "VPINN", "net": net, "history": hist, "time": elapsed,
            "status": status, "metrics": evaluate(net, cfg["lam"], cfg["n_quad"]), "config": cfg}


if __name__ == "__main__":
    res = train_vpinn({"lam": 4.0, "n_adam": 1000, "n_lbfgs": 500})
    print({k: v for k, v in res["metrics"].items() if not k.endswith("plot")})
