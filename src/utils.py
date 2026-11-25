"""
Utilities shared by the PINN and VPINN solvers.

Contents
--------
1. Gauss-Legendre quadrature (single interval and composite)
2. Legendre test functions  v_k = P_{k+1}(xi) - P_{k-1}(xi),  xi = 2x - 1
3. Automatic differentiation helper
4. Common training loop (Adam -> L-BFGS) with identical budget for both methods
5. Error metrics on a fine composite Gauss rule
6. Process-pool runner for parameter sweeps

Both solvers only differ by their loss function; everything else (network,
initialisation, optimiser, budget, quadrature for I[u], evaluation) lives
here so that the comparison is fair by construction.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import copy
import math
import os
import time

import numpy as np
import torch
from numpy.polynomial import legendre as leg

from exact_solution import (
    DTYPE, LAMBDA_CRIT, u_exact, u_exact_deriv, integral_exact, max_u_exact,
)
from network import HardDirichletNet

DEVICE = torch.device("cpu")   # 1-D problems: CPU float64 is faster than GPU here


# ===================================================================
# 1. Gauss-Legendre quadrature
# ===================================================================

def gauss_legendre(n, a=0.0, b=1.0):
    """Nodes and weights of an n-point Gauss-Legendre rule on [a, b]."""
    xi, wi = leg.leggauss(n)
    return 0.5 * (b - a) * xi + 0.5 * (a + b), 0.5 * (b - a) * wi


def composite_gauss_legendre(n_elem, n_pts, a=0.0, b=1.0):
    """Composite rule: n_elem equal sub-intervals, n_pts Gauss points each."""
    edges = np.linspace(a, b, n_elem + 1)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x, w = gauss_legendre(n_pts, lo, hi)
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


# ===================================================================
# 2. Legendre test functions (vanish at x = 0 and x = 1)
# ===================================================================

class LegendreTestFunctions:
    """v_k(x) = P_{k+1}(xi) - P_{k-1}(xi),  xi = 2x - 1,  k = 1..K.

    Since P_n(+-1) = (+-1)^n, v_k(0) = v_k(1) = 0.  Their derivatives are
    v_k'(x) = 2 (P'_{k+1} - P'_{k-1})(xi) = 2 (2k + 1) P_k(xi).
    """

    def __init__(self, n_test):
        self.n_test = n_test

    @staticmethod
    def _P(n, xi):
        c = np.zeros(n + 1)
        c[n] = 1.0
        return leg.legval(xi, c)

    def eval_v(self, x01):
        xi = 2.0 * np.asarray(x01) - 1.0
        return np.stack([self._P(k + 1, xi) - self._P(k - 1, xi)
                         for k in range(1, self.n_test + 1)], axis=1)

    def eval_dv(self, x01):
        xi = 2.0 * np.asarray(x01) - 1.0
        return np.stack([2.0 * (2 * k + 1) * self._P(k, xi)
                         for k in range(1, self.n_test + 1)], axis=1)


# ===================================================================
# 3. Automatic differentiation
# ===================================================================

def diff(u, x, order=1):
    """d^order u / dx^order via autograd (graph kept for training)."""
    du = u
    for _ in range(order):
        du = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du),
                                 create_graph=True, retain_graph=True)[0]
    return du


def to_t(a):
    return torch.tensor(np.asarray(a).reshape(-1, 1), dtype=DTYPE, device=DEVICE)


# ===================================================================
# 4. Common training loop
# ===================================================================

COMMON_DEFAULTS = {
    "lam": 4.0,             # non-local parameter lambda
    "problem": "nonlocal",  # "nonlocal" or "local" (E3: -u'' = mu exp(-u))
    "n_hidden": 32,
    "n_layers": 4,
    "n_quad": 128,          # Gauss-Legendre nodes on [0,1] (I[u], VPINN integrals, PINN collocation)
    "lr_adam": 1e-3,
    "n_adam": 5_000,        # Adam iterations
    "n_lbfgs": 2_000,       # max L-BFGS function evaluations
    "seed": 0,
    "log_every": 50,
}


def build_net(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    return HardDirichletNet(cfg["n_hidden"], cfg["n_layers"]).to(DTYPE).to(DEVICE)


_X_MON = None


def _monitor(net):
    """max u on a uniform grid (cheap, used for the max-u trajectories)."""
    global _X_MON
    if _X_MON is None:
        _X_MON = to_t(np.linspace(0.0, 1.0, 201))
    with torch.no_grad():
        return float(net(_X_MON).max())


def train_loop(net, loss_fn, cfg, tag="", verbose=False):
    """Adam then L-BFGS on loss_fn(net) -> (loss, aux dict).

    The same function is used for PINN and VPINN.  Budget: cfg["n_adam"]
    Adam steps followed by at most cfg["n_lbfgs"] L-BFGS loss evaluations.

    Non-finite losses are *not* hidden: training stops, the parameters are
    rolled back to the last finite state and `status` records what happened.
    """
    hist = {"loss": [], "rel_res": [], "log_it": [], "log_maxu": [], "log_I": []}
    status = "ok"
    params = list(net.parameters())
    last_good = copy.deepcopy(net.state_dict())
    it = [0]

    def record(loss, aux):
        hist["loss"].append(loss)
        hist["rel_res"].append(aux["rel_res"])
        if it[0] % cfg["log_every"] == 0:
            hist["log_it"].append(it[0])
            hist["log_maxu"].append(_monitor(net))
            hist["log_I"].append(aux["I"])
        it[0] += 1

    t0 = time.perf_counter()

    # ---- Phase 1: Adam ----
    opt = torch.optim.Adam(params, lr=cfg["lr_adam"])
    for ep in range(cfg["n_adam"]):
        opt.zero_grad()
        loss, aux = loss_fn(net)
        lv = loss.item()
        if not math.isfinite(lv):
            status = f"nonfinite_adam@{ep}"
            break
        last_good = copy.deepcopy(net.state_dict())
        record(lv, aux)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 1000 == 0:
            print(f"  {tag} Adam {ep+1:6d}  loss={lv:.3e}  rel_res={aux['rel_res']:.2e}  I={aux['I']:.4e}")

    # ---- Phase 2: L-BFGS ----
    if status == "ok" and cfg["n_lbfgs"] > 0:
        opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg["n_lbfgs"],
                                 max_eval=cfg["n_lbfgs"], history_size=50,
                                 tolerance_grad=1e-12, tolerance_change=1e-16,
                                 line_search_fn="strong_wolfe")
        bad = [False]

        def closure():
            opt2.zero_grad()
            loss, aux = loss_fn(net)
            lv = loss.item()
            if math.isfinite(lv):
                record(lv, aux)
                loss.backward()
            else:
                bad[0] = True
                # return +inf-like value so the line search rejects the step
                loss = torch.tensor(1e300, dtype=DTYPE, requires_grad=True)
                loss.backward()
            return loss

        opt2.step(closure)
        finite = all(torch.isfinite(p).all() for p in params)
        if bad[0] or not finite:
            status = "nonfinite_lbfgs"
        if not finite:
            net.load_state_dict(last_good)

    if status.startswith("nonfinite_adam"):
        net.load_state_dict(last_good)

    elapsed = time.perf_counter() - t0
    # training loss re-evaluated at the final parameters (L-BFGS' last closure
    # call may be a rejected line-search trial)
    loss, aux = loss_fn(net)
    hist["final_loss"] = loss.item()
    hist["final_rel"] = aux["rel_res"]
    hist["final_I"] = aux["I"]
    if verbose:
        fl = hist["loss"][-1] if hist["loss"] else float("nan")
        print(f"  {tag} done in {elapsed:.1f}s  final loss={fl:.3e}  status={status}")
    return hist, elapsed, status


# ===================================================================
# 5. Evaluation
# ===================================================================

_EVAL = {}


K_VAL = 150   # test functions of the a-posteriori H^{-1} residual


def _eval_rule():
    if "x" not in _EVAL:
        x, w = composite_gauss_legendre(400, 8)   # 3200 nodes, resolves the boundary layers
        tf = LegendreTestFunctions(K_VAL)
        # ||v_k'||_L2(0,1) = 2 sqrt(2k+1); the normalised v_k'/||v_k'|| are
        # L2-orthonormal, so sum_k (R_k/||v_k'||)^2 is the H^{-1} norm of the
        # residual projected on the first K_VAL modes.
        nrm = 2.0 * np.sqrt(2.0 * np.arange(1, K_VAL + 1) + 1.0)
        _EVAL.update(x=x, w=w, V=tf.eval_v(x) / nrm, dV=tf.eval_dv(x) / nrm)
    return _EVAL["x"], _EVAL["w"]


def evaluate(net, lam, n_quad_train=None):
    """Errors against the exact solution (when it exists) on a fine Gauss rule.

    Always returned (no exact solution needed, also valid for lambda >= 8):
      I_pred, maxu_pred  -- I[u_theta] and max u_theta on the fine rule
      rho_val            -- a-posteriori strong relative residual on the fine rule
                            ||-u'' - lam e^{-u}/I^2||_L2 / ||lam e^{-u}/I^2||_L2
                            (I computed on the fine rule, independent of training)
      rho_weak           -- a-posteriori relative residual in a (truncated) H^{-1}
                            norm, K_VAL normalised Legendre test functions on the
                            fine rule: ||R||_{H^-1} / ||lam e^{-u}/I^2||_{H^-1}
      quad_gap           -- |I_train - I_fine| / I_fine, I_train on the training rule
    When 0 < lambda < 8: L2, Linf, H1 (full norm), L2_rel, I_rel, maxu_rel.
    """
    x, w = _eval_rule()
    xt = to_t(x).requires_grad_(True)
    u = net(xt)
    du = diff(u, xt, 1)
    d2u = diff(du, xt, 1)
    u = u.detach().numpy().ravel()
    du = du.detach().numpy().ravel()
    d2u = d2u.detach().numpy().ravel()
    with torch.no_grad():
        u_mid = float(net(to_t([0.5])))
    I_pred = float(np.sum(w * np.exp(-u)))
    maxu_pred = float(max(u.max(), u_mid))
    src = lam * np.exp(-u) / I_pred ** 2
    rho_val = math.sqrt(np.sum(w * (-d2u - src) ** 2) / np.sum(w * src ** 2))
    load = (w * src) @ _EVAL["V"]
    Rw = (w * du) @ _EVAL["dV"] - load
    rho_weak = math.sqrt(np.sum(Rw ** 2) / np.sum(load ** 2))
    out = {"I_pred": I_pred, "maxu_pred": maxu_pred, "rho_val": rho_val, "rho_weak": rho_weak}
    if n_quad_train is not None:
        xq, wq = gauss_legendre(n_quad_train)
        with torch.no_grad():
            I_tr = float(torch.sum(to_t(wq) * torch.exp(-net(to_t(xq)))))
        out["I_train"] = I_tr
        out["quad_gap"] = abs(I_tr - I_pred) / I_pred
    if 0.0 < lam < LAMBDA_CRIT:
        ue, due = u_exact(x, lam), u_exact_deriv(x, lam)
        e, de = u - ue, du - due
        L2 = math.sqrt(np.sum(w * e ** 2))
        semi = math.sqrt(np.sum(w * de ** 2))
        out.update({
            "L2": L2,
            "Linf": float(max(np.abs(e).max(), abs(u_mid - max_u_exact(lam)))),
            "H1": math.sqrt(L2 ** 2 + semi ** 2),
            "L2_rel": L2 / math.sqrt(np.sum(w * ue ** 2)),
            "I_rel": abs(I_pred - integral_exact(lam)) / integral_exact(lam),
            "maxu_rel": abs(maxu_pred - max_u_exact(lam)) / max_u_exact(lam),
        })
    xp = np.linspace(0.0, 1.0, 401)
    with torch.no_grad():
        out["u_plot"] = net(to_t(xp)).numpy().ravel()
    out["x_plot"] = xp
    return out


# ===================================================================
# 6. Parallel runner
# ===================================================================

def _worker_init():
    torch.set_num_threads(1)


def run_one(job):
    """job = (method, config) -> light-weight, picklable result dict."""
    method, cfg = job
    if method == "PINN":
        from pinn_solver import train_pinn as train
    else:
        from vpinn_solver import train_vpinn as train
    res = train(cfg, verbose=False)
    res.pop("net")
    return res


def run_jobs(jobs, n_proc=None):
    """Run a list of (method, config) jobs in a process pool (spawn-safe)."""
    import multiprocessing as mp
    if n_proc is None:
        n_proc = max(1, (os.cpu_count() or 2) - 2)
    if n_proc == 1 or len(jobs) == 1:
        _worker_init()
        return [run_one(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_proc, initializer=_worker_init) as pool:
        return pool.map(run_one, jobs, chunksize=1)
