"""
Classical baselines and the mechanism behind the VPINN observations.

    python study_classical.py          # ~5 min, single process (needs run_all.py and study_robustness.py results)

C1  Finite differences + Newton: accuracy and wall time vs the PINN / VPINN.
C2  Amplitude continuation u(1/2) = m: lambda(m) -> lambda* (classical way to find the threshold).
C3  Square Galerkin least squares (spectral ansatz with M = K coefficients, same test
    functions and quadrature as the VPINN, no neural network):
      - residual floor vs lambda, compared with the VPINN floor;
      - existence threshold lambda*_K of the K-mode discrete problem;
      - nature of the discrete solutions beyond 8 (a-posteriori residual, max u).
C4  Over-parameterised polynomial ansatz (M > K): the weak residual can be zeroed for any lambda.

Outputs: ../results/classical_summary.md, ../results/fig8_classical.png,
CSV tables in ../results/classical/.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import csv
import os
import pickle
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exact_solution import u_exact, b_of_lambda  # noqa: E402
from classical_solver import (newton_fd, solve_with_continuation, amplitude_branch,  # noqa: E402
                              ls_minimum, SpectralNet)
from utils import evaluate  # noqa: E402
from run_all import METHODS, write_csv, HERE  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "results"))
CDIR = os.path.join(OUT, "classical")
ZERO = 1e-8   # relative weak residual below which the discrete problem is considered solved


def load_raw(name):
    with open(os.path.join(OUT, "raw", f"{name}.pkl"), "rb") as fh:
        return pickle.load(fh)["results"]


def fd_l2_error(x, u, lam):
    """L2 error of the piecewise-linear FD solution, on a fine Gauss rule per cell."""
    g, wg = np.polynomial.legendre.leggauss(4)
    h = x[1] - x[0]
    err2 = 0.0
    for gi, wi in zip(0.5 * (g + 1), 0.5 * wg):
        xm = x[:-1] + gi * h
        um = u[:-1] + gi * (u[1:] - u[:-1])
        err2 += np.sum(wi * h * (um - u_exact(xm, lam)) ** 2)
    return np.sqrt(err2)


# ===================================================================
# C1 -- finite differences
# ===================================================================

def run_C1():
    rows = []
    for lam in (1.0, 4.0, 7.0, 7.9, 7.99):
        for N in (200, 800, 3200, 12800):
            reps = 5
            t0 = time.perf_counter()
            for _ in range(reps):
                x, u, it, ok = newton_fd(lam, N)
            t_cold = (time.perf_counter() - t0) / reps
            mode = "cold start"
            if not ok:
                t0 = time.perf_counter()
                x, u, it, ok = solve_with_continuation(lam, N)
                t_cold = time.perf_counter() - t0
                mode = "10-step λ continuation"
            rows.append({"lam": lam, "N": N, "ok": ok, "iters": it, "time": t_cold, "mode": mode,
                         "L2": fd_l2_error(x, u, lam) if ok else np.nan})
    beyond = []
    for lam in (8.0, 8.5, 9.0):
        _, _, it, ok = newton_fd(lam, 800)
        _, _, itc, okc = solve_with_continuation(lam, 800)
        beyond.append((lam, ok, it, okc))
    return rows, beyond


def nn_best():
    """Best (mean over seeds) L2 and mean time of the networks, from the cached runs."""
    out = {}
    for name in ("E1",):
        for r in load_raw(name):
            lab = "PINN" if r["method"] == "PINN" else f"VPINN K={r['config']['n_test']}"
            out.setdefault((r["config"]["lam"], lab, "default"), []).append(r)
    try:
        for r in load_raw("R6b_budget_only"):
            c = r["config"]
            if c["n_quad"] == 256:
                lab = "PINN" if r["method"] == "PINN" else f"VPINN K={c['n_test']}"
                out.setdefault((c["lam"], lab, "4× budget, n_quad 256"), []).append(r)
    except FileNotFoundError:
        pass
    return {k: (np.mean([r["metrics"]["L2"] for r in v]), np.mean([r["time"] for r in v])) for k, v in out.items()}


# ===================================================================
# C2 -- amplitude continuation
# ===================================================================

def run_C2():
    t0 = time.perf_counter()
    br = amplitude_branch([1, 2, 4, 6, 8, 10, 12], N=4000)
    return br, time.perf_counter() - t0


# ===================================================================
# C3 -- square Galerkin least squares
# ===================================================================

def galerkin_sweep(K, lams, M=None):
    """Residual floor along lambda, best of cold start and continuation."""
    M = M or K
    a, out = None, []
    for lam in lams:
        rc = ls_minimum(lam, K, M=M, a0=a)
        r0 = ls_minimum(lam, K, M=M)
        best = rc if rc["rel"] <= r0["rel"] else r0
        a = best["a"]
        out.append((lam, best["rel"], best["a"]))
    return out


def discrete_threshold(K):
    """Largest lambda for which the K-mode Galerkin system has a solution (resolution 0.01)."""
    coarse = list(np.round(np.arange(7.5, 9.0, 0.05), 2)) + list(np.arange(9.0, 20.01, 0.5))
    sw = galerkin_sweep(K, coarse)
    last_zero = max((l for l, f, _ in sw if f < ZERO), default=None)
    if last_zero is None or last_zero >= coarse[-1]:
        return last_zero, sw, ">"
    nxt = coarse[coarse.index(last_zero) + 1]
    a0 = [a for l, f, a in sw if l == last_zero][0]
    fine = np.round(np.arange(last_zero, nxt + 1e-9, 0.01), 2)
    a, best_zero = a0, last_zero
    for lam in fine[1:]:
        r = ls_minimum(lam, K, M=K, a0=a)
        if r["rel"] < ZERO:
            best_zero, a = lam, r["a"]
        else:
            break
    return float(best_zero), sw, "="


def run_C3():
    lams = list(np.round(np.arange(7.5, 9.0001, 0.05), 2))
    floors = {K: galerkin_sweep(K, lams) for K in (30, 60)}
    fine = galerkin_sweep(60, list(np.round(np.arange(7.95, 8.2001, 0.01), 2)))
    thresholds = {}
    for K in (10, 15, 20, 25, 30, 40, 60):
        lt, sw, sign = discrete_threshold(K)
        thresholds[K] = (lt, sign)
    # nature of the discrete K = 20 solution at lambda = 8.3 vs the K = 60 floor minimiser
    nature = []
    for K, lam in ((20, 8.3), (20, 8.6), (60, 8.3)):
        r = galerkin_sweep(K, list(np.round(np.arange(7.5, lam + 1e-9, 0.05), 2)))[-1]
        m = evaluate(SpectralNet(r[2]), lam)
        nature.append((K, lam, r[1], m["rho_weak"], m["maxu_pred"], m["res_mode1_frac"], m["res_modes_gt20_frac"]))
    return lams, floors, fine, thresholds, nature


def run_C5():
    """Does the FD discretisation 'see' non-existence?  Fixed-N solutions beyond 8 and
    the amplitude branch lambda_N(m) for several N (no Richardson)."""
    from classical_solver import amplitude_point
    fixed = []
    for N in (200, 800, 3200):
        for lam in (8.0, 9.0):
            x, u, it, ok = newton_fd(lam, N)
            fixed.append((N, lam, ok, it, float(u.max()) if ok else np.nan))
    ms = [6, 8, 10, 12, 14, 16, 20]
    branch = {}
    for N in (800, 3200, 12800):
        u = l = None
        vals = []
        for m in ms:
            l, u, it, ok = amplitude_point(m, N, u, l)
            vals.append(l if ok else np.nan)
        branch[N] = vals
    return fixed, ms, branch


def run_C4():
    out = []
    for K, M in ((20, 80), (60, 80)):
        for lam in (8.3, 9.0):
            r = ls_minimum(lam, K, M=M)
            out.append((K, M, lam, r["rel"]))
    return out


# ===================================================================
# Report + figure
# ===================================================================

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(CDIR, exist_ok=True)
    t_all = time.time()
    c1, beyond = run_C1()
    best = nn_best()
    c2, t_c2 = run_C2()
    lams, floors, fine, thr, nature = run_C3()
    c4 = run_C4()
    c5_fixed, c5_ms, c5_branch = run_C5()
    sweep = {}
    with open(os.path.join(OUT, "E2_sweep.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sweep[(r["method"], round(float(r["lambda"]), 2))] = float(r["rel_train_residual"])

    rep = ["# Classical baselines and mechanism\n"]
    # C1
    t = ["| λ | N | Newton iterations | wall time | L2 error |", "|---|---|---|---|---|"]
    for r in c1:
        t.append(f"| {r['lam']:g} | {r['N']} | {r['iters']} ({r['mode']}) | {1e3 * r['time']:.1f} ms | "
                 f"{r['L2']:.2e} |")
    rep += ["## C1 — finite differences + Newton (O(N) per step, Sherman-Morrison)\n", "\n".join(t), ""]
    t = ["| λ | network | setting | mean L2 | mean time |", "|---|---|---|---|---|"]
    for (lam, lab, setting), (l2, tm) in sorted(best.items()):
        if lam in (7.0, 7.9, 7.99) and lab in ("PINN", "VPINN K=60"):
            t.append(f"| {lam:g} | {lab} | {setting} | {l2:.2e} | {tm:.0f} s |")
    rep += ["Neural networks for comparison (cached runs):\n", "\n".join(t), ""]
    rep += ["Beyond the threshold (N = 800): " + "; ".join(
        f"λ = {l:g}: cold Newton {'converged' if ok else 'fails'}, continuation {'converged' if okc else 'fails'}"
        for l, ok, it, okc in beyond) + "\n"]
    write_csv(os.path.join(CDIR, "C1_fd.csv"), ["lambda", "N", "iters", "mode", "time_s", "L2"],
              [[r["lam"], r["N"], r["iters"], r["mode"], r["time"], r["L2"]] for r in c1])
    # C2
    t = ["| m = u(1/2) | λ (N = 4000) | λ (Richardson) | exact 8(1 − e^{−m}) | error (Richardson) |",
         "|---|---|---|---|---|"]
    for r in c2:
        ex = 8 * (1 - np.exp(-r["m"]))
        t.append(f"| {r['m']} | {r['lam_N']:.6f} | {r['lam_R']:.8f} | {ex:.8f} | {abs(r['lam_R'] - ex):.1e} |")
    rep += [f"## C2 — amplitude continuation (total {1e3 * t_c2:.0f} ms)\n", "\n".join(t), ""]
    # C3
    t = ["| λ | Galerkin K = M = 30 | Galerkin K = M = 60 | VPINN K = 60 (training floor, median 2 seeds) | "
         "VPINN K = 20 |", "|---|---|---|---|---|"]
    rows = []
    for i, lam in enumerate(lams):
        g30, g60 = floors[30][i][1], floors[60][i][1]
        v60, v20 = sweep.get(("VPINN K=60", lam), np.nan), sweep.get(("VPINN K=20", lam), np.nan)
        rows.append([lam, g30, g60, v60, v20])
        if lam >= 7.9 and (round(lam * 20) % 2 == 0 or lam <= 8.1):
            t.append(f"| {lam:g} | {g30:.3e} | {g60:.3e} | {v60:.3e} | {v20:.3e} |")
    write_csv(os.path.join(CDIR, "C3_floors.csv"), ["lambda", "galerkin_K30", "galerkin_K60", "vpinn_K60", "vpinn_K20"], rows)
    above = [r for r in rows if r[0] >= 8.1]
    rel_diff = max(abs(r[3] - r[2]) / r[2] for r in above)
    rep += ["## C3 — square Galerkin least squares (no network): residual floor\n", "\n".join(t), "",
            f"Max relative difference between the VPINN K = 60 floor and the Galerkin K = 60 floor for "
            f"λ ∈ [8.1, 9.0]: {100 * rel_diff:.2f} %.\n"]
    t = ["| λ | " + " | ".join(f"{l:g}" for l, _, _ in fine) + " |", "|---|" + "---|" * len(fine),
         "| Galerkin K = 60 floor | " + " | ".join(f"{f:.1e}" for _, f, _ in fine) + " |"]
    rep += ["Fine scan near the threshold (K = M = 60):\n", "\n".join(t), ""]
    L = np.array([r[0] for r in above if r[0] <= 8.5])
    y = np.array([r[2] for r in above if r[0] <= 8.5])
    a, b = np.polyfit(L, y, 1)
    rep += [f"Linear fit of the Galerkin K = 60 floor on [8.1, 8.5]: zero at λ = {-b / a:.4f} "
            f"(max deviation {100 * np.max(np.abs(np.polyval([a, b], L) - y)) / y.max():.2f} %).\n"]
    t = ["| K | discrete existence threshold λ*_K |", "|---|---|"]
    for K, (lt, sign) in thr.items():
        t.append(f"| {K} | {'> ' if sign == '>' else ''}{lt:g} |" if lt is not None else f"| {K} | — |")
    rep += ["## C3 — existence threshold of the K-mode Galerkin problem (resolution 0.01)\n", "\n".join(t), ""]
    t = ["| K | λ | Galerkin residual | ρ_weak (150 modes) | max u | energy in mode 1 | energy k > 20 |",
         "|---|---|---|---|---|---|---|"]
    for K, lam, rel, rw, mu, f1, f20 in nature:
        t.append(f"| {K} | {lam:g} | {rel:.1e} | {rw:.2e} | {mu:.2f} | {100 * f1:.1f}% | {100 * f20:.2g}% |")
    rep += ["Nature of the discrete minimisers:\n", "\n".join(t), ""]
    t = ["| K | M | λ | relative weak residual |", "|---|---|---|---|"]
    for K, M, lam, rel in c4:
        t.append(f"| {K} | {M} | {lam:g} | {rel:.1e} |")
    rep += ["## C4 — over-parameterised polynomial ansatz (M > K)\n", "\n".join(t), ""]
    t = ["| N | λ | Newton (cold start) | max u of the discrete solution |", "|---|---|---|---|"]
    for N, lam, ok, it, mu in c5_fixed:
        t.append(f"| {N} | {lam:g} | {'converged, ' + str(it) + ' it.' if ok else 'fails'} | "
                 f"{mu:.2f} |" if ok else f"| {N} | {lam:g} | fails | — |")
    t2 = ["| N | " + " | ".join(f"m = {m}" for m in c5_ms) + " |", "|---|" + "---|" * len(c5_ms)]
    for N, vals in c5_branch.items():
        t2.append(f"| {N} | " + " | ".join(f"{v:.4g}" for v in vals) + " |")
    t2.append("| exact | " + " | ".join(f"{8 * (1 - np.exp(-m)):.4g}" for m in c5_ms) + " |")
    rep += ["## C5 — does the finite-difference discretisation see non-existence?\n",
            "Fixed-resolution solutions beyond the threshold:\n", "\n".join(t), "",
            "Amplitude branch λ_N(m) without Richardson extrapolation:\n", "\n".join(t2), ""]

    with open(os.path.join(OUT, "classical_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep))
    print("\n".join(rep))

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for lam, mk in ((7.9, "o"), (7.99, "s")):
        rs = [r for r in c1 if r["lam"] == lam and r["ok"]]
        ax.loglog([r["time"] for r in rs], [r["L2"] for r in rs], color="0.2", marker=mk, ls="-",
                  label=f"finite differences, λ = {lam:g} (N = 200 … 12800)")
        for (l, lab, setting), (l2, tm) in best.items():
            if l == lam and lab in ("PINN", "VPINN K=60"):
                st = METHODS[lab]
                ax.scatter([tm], [l2], color=st["color"], marker=mk, s=45, zorder=3,
                           facecolors=st["color"] if setting == "default" else "white",
                           label=f"{lab}, λ = {lam:g}, {setting}")
    ax.set_xlabel("wall time [s]"); ax.set_ylabel("L2 error")
    ax.set_title("C1 — accuracy vs cost")
    ax.legend(fontsize=6.5)
    ax = axes[1]
    LL = np.array(lams)
    ax.plot(LL, [f for _, f, _ in floors[60]], color="black", lw=2.2, label="Galerkin K = M = 60 (no network)")
    ax.plot(LL, [f for _, f, _ in floors[30]], color="0.5", lw=1.2, ls="--", label="Galerkin K = M = 30")
    for name in ("VPINN K=20", "VPINN K=60"):   # PINN floor is a strong-form residual: not comparable
        st = METHODS[name]
        ax.plot(LL, [sweep.get((name, l), np.nan) for l in lams], color=st["color"], marker=st["marker"],
                ms=4, lw=0, label=f"{name} training floor")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.axvline(8.0, color="0.5", lw=1, ls=":")
    ax.set_xlabel("λ"); ax.set_ylabel("relative weak residual (minimum reached)")
    ax.set_title("C3 — the VPINN floor is the Galerkin floor")
    ax.legend(fontsize=7)
    ax = axes[2]
    Ks = list(thr)
    vals = [thr[K][0] for K in Ks]
    ax.plot(Ks, vals, color="#eb6834", marker="o")
    for K, v in zip(Ks, vals):
        if thr[K][1] == ">":
            ax.annotate(f"> {v:g}", (K, v), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(8.0, color="black", lw=1, ls="--", label="continuous λ* = 8")
    ax.set_ylim(7.5, 21.5)
    ax.set_xlabel("K (test functions = Galerkin modes)"); ax.set_ylabel("discrete threshold λ*_K")
    ax.set_title("C3 — existence threshold of the K-mode problem")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_classical.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nTotal wall time: {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
