"""
Run every experiment of the non-local thermistor PINN / VPINN benchmark and
regenerate all figures and tables in ../results/.

    python run_all.py                 # full benchmark (~30 min on 12 CPU cores)
    python run_all.py --quick         # short version (Colab, ~5 min) -> ../results_quick/
    python run_all.py --only E1,E2    # subset; raw results are cached in results/raw/
    python run_all.py --force         # ignore the cache

Experiments
-----------
E1  accuracy vs lambda (lambda < 8, exact solution available)
E2  non-existence (lambda >= 8): loss floors, max-u trajectories, and a fine
    lambda sweep around 8 to estimate the critical value "seen" by each method
E3  cost of the non-local term: same exact solution, local problem
    -u'' = mu exp(-u) with mu = lambda / I_exact^2
E4  ablations: number of test functions K, width/depth, quadrature points

Methods compared: PINN, VPINN with K = 20 and K = 60 Legendre test functions.
All share the same network, initialisation (seed), quadrature and budget.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import argparse
import csv
import os
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exact_solution import (  # noqa: E402
    u_exact, integral_exact, max_u_exact, mu_local, LAMBDA_CRIT,
)
from utils import run_jobs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
})

# Fixed categorical order (validated palette) + marker as secondary encoding
METHODS = {
    "PINN":        {"job": ("PINN", {}),               "color": "#2a78d6", "marker": "o"},
    "VPINN K=20":  {"job": ("VPINN", {"n_test": 20}),  "color": "#eb6834", "marker": "s"},
    "VPINN K=60":  {"job": ("VPINN", {"n_test": 60}),  "color": "#1baf7a", "marker": "^"},
}
EXACT_KW = {"color": "black", "ls": "--", "lw": 1.2}


# ===================================================================
# Settings (full / quick)
# ===================================================================

def settings(quick):
    if not quick:
        return {
            "budget": {"n_adam": 5000, "n_lbfgs": 2000},
            "E1_lams": [1.0, 4.0, 6.0, 7.0, 7.5, 7.9, 7.99], "E1_seeds": [0, 1, 2],
            "E2_lams": [8.0, 9.0, 12.0], "E2_seeds": [0, 1, 2],
            "sweep_lams": list(np.round(np.arange(7.5, 9.0001, 0.05), 2)), "sweep_seeds": [0, 1],
            "modes_lams": [8.3, 9.0, 12.0],
            "E3_seeds": [0, 1, 2],
            "E4_lam": 7.9, "E4_seeds": [0, 1],
            "E4_K": [5, 10, 20, 40, 60, 80, 120], "E4_K_lams": [4.0, 7.9],
            "E4_arch": [(16, 3), (32, 3), (32, 4), (64, 4), (64, 6)],
            "E4_quad": [32, 64, 128, 256],
        }
    return {
        "budget": {"n_adam": 1500, "n_lbfgs": 500},
        "E1_lams": [1.0, 7.0, 7.9], "E1_seeds": [0],
        "E2_lams": [8.0, 9.0], "E2_seeds": [0],
        "sweep_lams": [7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0], "sweep_seeds": [0],
        "modes_lams": [8.5, 9.0],
        "E3_seeds": [0],
        "E4_lam": 7.9, "E4_seeds": [0],
        "E4_K": [10, 20, 60], "E4_K_lams": [7.9],
        "E4_arch": [(16, 3), (32, 4)],
        "E4_quad": [64, 128],
    }


def job(name, lam, seed, budget, **extra):
    method, base = METHODS[name]["job"]
    return (method, {**base, **budget, "lam": float(lam), "seed": seed, **extra})


def label_of(res):
    if res["method"] == "PINN":
        return "PINN"
    return f"VPINN K={res['config']['n_test']}"


# ===================================================================
# Cache
# ===================================================================

def cached(name, jobs, args):
    path = os.path.join(args.out, "raw", f"{name}.pkl")
    if os.path.exists(path) and not args.force:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if [j for j in data["jobs"]] == jobs:
            print(f"[{name}] loaded {len(jobs)} runs from cache")
            return data["results"]
    print(f"[{name}] running {len(jobs)} trainings on {args.nproc} processes ...")
    t0 = time.time()
    results = run_jobs(jobs, n_proc=args.nproc)
    print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"jobs": jobs, "results": results}, fh)
    return results


def group(results, key):
    out = {}
    for r in results:
        out.setdefault(key(r), []).append(r)
    return out


def runmin(loss):
    """Best-so-far loss normalised by the initial loss (hides L-BFGS line-search trials)."""
    L = np.asarray(loss, dtype=float)
    return np.minimum.accumulate(L) / L[0]


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return np.nanmean(v), (np.nanstd(v) if len(v) > 1 else 0.0)


def fmt(m, s):
    return f"{m:.2e} ± {s:.1e}" if s > 0 else f"{m:.2e}"


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ===================================================================
# E1 -- accuracy vs lambda
# ===================================================================

METRICS = ["L2", "Linf", "H1", "I_rel", "maxu_rel"]


def run_E1(S, args, problem="nonlocal", tag="E1"):
    jobs = [job(n, lam, s, S["budget"], problem=problem)
            for lam in S["E1_lams"] for n in METHODS for s in S["E1_seeds"]]
    return cached(tag, jobs, args)


def table_E1(results, args, tag="E1"):
    g = group(results, lambda r: (r["config"]["lam"], label_of(r)))
    rows, md = [], []
    md.append("| λ | Method | L2 | L∞ | H1 | rel. err. I[u] | rel. err. max u | time (s) | status |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for (lam, lab), rs in sorted(g.items(), key=lambda kv: (kv[0][0], list(METHODS).index(kv[0][1]))):
        stats = [ms([r["metrics"][k] for r in rs]) for k in METRICS]
        tm = ms([r["time"] for r in rs])
        st = ",".join(sorted({r["status"] for r in rs}))
        rows.append([lam, lab, len(rs)] + [x for m in stats for x in m] + list(tm) + [st])
        md.append(f"| {lam:g} | {lab} | " + " | ".join(fmt(*m) for m in stats)
                  + f" | {tm[0]:.0f} | {st} |")
    hdr = ["lambda", "method", "n_seeds"] + [f"{k}_{s}" for k in METRICS for s in ("mean", "std")] \
        + ["time_mean", "time_std", "status"]
    write_csv(os.path.join(args.out, f"{tag}_accuracy.csv"), hdr, rows)
    return "\n".join(md)


def plot_E1(results, args):
    g = group(results, lambda r: (r["config"]["lam"], label_of(r)))
    lams = sorted({k[0] for k in g})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, key, title in [(axes[0, 0], "L2", "L2 error"), (axes[0, 1], "I_rel", "Relative error on I[u]")]:
        for name, st in METHODS.items():
            m = [ms([r["metrics"][key] for r in g[(l, name)]]) for l in lams]
            ax.errorbar(lams, [a for a, _ in m], yerr=[b for _, b in m], color=st["color"],
                        marker=st["marker"], ms=6, capsize=3, label=name)
        ax.set_yscale("log"); ax.set_xlabel("λ"); ax.set_title(title)
        ax.axvline(LAMBDA_CRIT, color="0.5", lw=1, ls=":")
        ax.legend()
    lam_show = max(l for l in lams if l <= 7.9)
    ax = axes[1, 0]
    x = g[(lam_show, "PINN")][0]["metrics"]["x_plot"]
    ax.plot(x, u_exact(x, lam_show), label="exact", **EXACT_KW)
    for name, st in METHODS.items():
        ax.plot(x, g[(lam_show, name)][0]["metrics"]["u_plot"], color=st["color"], label=name, alpha=0.85)
    ax.set_title(f"Solution, λ = {lam_show:g} (seed 0)"); ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend()
    ax = axes[1, 1]
    for name, st in METHODS.items():
        ax.semilogy(x, np.abs(g[(lam_show, name)][0]["metrics"]["u_plot"] - u_exact(x, lam_show)) + 1e-16,
                    color=st["color"], label=name)
    ax.set_ylim(1e-7, 1); ax.set_title(f"Pointwise error |u_θ − u|, λ = {lam_show:g}"); ax.set_xlabel("x"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig1_accuracy_vs_lambda.png"), bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# E2 -- non-existence and critical lambda seen by each method
# ===================================================================

TAUS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def run_E2(S, args):
    jobs = [job(n, lam, s, S["budget"]) for lam in S["E2_lams"] for n in METHODS for s in S["E2_seeds"]]
    fixed = cached("E2_fixed", jobs, args)
    jobs = [job(n, lam, s, S["budget"]) for lam in S["sweep_lams"] for n in METHODS for s in S["sweep_seeds"]]
    sweep = cached("E2_sweep", jobs, args)
    jobs = [job(n, lam, 0, S["budget"]) for lam in S["modes_lams"] for n in METHODS]
    modes = cached("E2_modes", jobs, args)
    return fixed, sweep, modes


def lambda_star(lams, det, tau):
    """Smallest grid lambda from which the detector stays above tau."""
    above = np.asarray(det) > tau
    for i in range(len(lams)):
        if above[i:].all():
            return lams[i]
    return np.nan


FIT_WINDOWS = [(8.1, 8.5), (8.2, 8.6), (8.3, 8.7), (8.5, 9.0)]


def floor_extrapolation(lams, det, lo, hi):
    """Fit det ~ a (lambda - lambda*) on [lo, hi] (lambda >= 8 side) and return
    (lambda*, max relative deviation of the data from the line)."""
    L, y = np.asarray(lams), np.asarray(det)
    m = (L >= lo - 1e-9) & (L <= hi + 1e-9)
    if m.sum() < 3:
        return np.nan, np.nan
    a, b = np.polyfit(L[m], y[m], 1)
    dev = np.max(np.abs(np.polyval([a, b], L[m]) - y[m])) / np.max(y[m])
    return -b / a, dev


def analyse_E2(fixed, sweep, modes, args):
    lines = []
    # --- fixed lambdas: loss floors ---
    g = group(fixed, lambda r: (r["config"]["lam"], label_of(r)))
    md = ["| λ | Method | final training loss | rel. training residual | ρ_weak (a-post.) | max u | I[u] |",
          "|---|---|---|---|---|---|---|"]
    for (lam, lab), rs in sorted(g.items(), key=lambda kv: (kv[0][0], list(METHODS).index(kv[0][1]))):
        md.append(f"| {lam:g} | {lab} | {fmt(*ms([r['history']['final_loss'] for r in rs]))} | "
                  f"{fmt(*ms([r['history']['final_rel'] for r in rs]))} | "
                  f"{fmt(*ms([r['metrics']['rho_weak'] for r in rs]))} | "
                  f"{ms([r['metrics']['maxu_pred'] for r in rs])[0]:.2f} | "
                  f"{ms([r['metrics']['I_pred'] for r in rs])[0]:.3e} |")
    lines.append("\n".join(md))

    # --- sweep ---
    gs = group(sweep, lambda r: (label_of(r), r["config"]["lam"]))
    lams = sorted({r["config"]["lam"] for r in sweep})
    det = {}
    rows = []
    for name in METHODS:
        tr = [np.median([r["history"]["final_rel"] for r in gs[(name, l)]]) for l in lams]
        va = [np.median([r["metrics"]["rho_weak"] for r in gs[(name, l)]]) for l in lams]
        st = [np.median([r["metrics"]["rho_val"] for r in gs[(name, l)]]) for l in lams]
        mu = [np.median([r["metrics"]["maxu_pred"] for r in gs[(name, l)]]) for l in lams]
        det[name] = {"train": tr, "weak": va, "strong": st, "maxu": mu}
        for l, a, b, c, d in zip(lams, tr, va, st, mu):
            rows.append([name, l, a, b, c, d])
    write_csv(os.path.join(args.out, "E2_sweep.csv"),
              ["method", "lambda", "rel_train_residual", "rho_weak", "rho_strong", "max_u"], rows)

    md = ["| τ | " + " | ".join(f"{n} (train)" for n in METHODS) + " | "
          + " | ".join(f"{n} (ρ_weak)" for n in METHODS) + " |",
          "|---|" + "---|" * (2 * len(METHODS))]
    star = {}
    for tau in TAUS:
        cells = []
        for kind in ("train", "weak"):
            for n in METHODS:
                ls = lambda_star(lams, det[n][kind], tau)
                star[(n, kind, tau)] = ls
                cells.append("—" if np.isnan(ls) else (f"≤ {ls:g}" if ls == lams[0] else f"{ls:g}"))
        md.append(f"| {tau:g} | " + " | ".join(cells) + " |")
    lines.append("\n".join(md))

    # --- criterion B: linear extrapolation of the post-critical floor ---
    md = ["| Method | detector | " + " | ".join(f"fit on [{lo:g}, {hi:g}]" for lo, hi in FIT_WINDOWS) + " |",
          "|---|---|" + "---|" * len(FIT_WINDOWS)]
    for n in METHODS:
        for kind, kname in (("train", "rel. training residual"), ("weak", "ρ_weak")):
            cells = []
            for lo, hi in FIT_WINDOWS:
                ls, dev = floor_extrapolation(lams, det[n][kind], lo, hi)
                cells.append("—" if np.isnan(ls) else f"{ls:.3f} (dev {100 * dev:.0f}%)")
            md.append(f"| {n} | {kname} | " + " | ".join(cells) + " |")
    lines.append("\n".join(md))

    # --- modal content of the a-posteriori residual (lambda > 8) ---
    md = ["| λ | Method | ρ_weak | energy in mode k=1 (uniform sink) | energy in modes k > 20 |",
          "|---|---|---|---|---|"]
    for r in sorted(modes, key=lambda r: (r["config"]["lam"], list(METHODS).index(label_of(r)))):
        m = r["metrics"]
        md.append(f"| {r['config']['lam']:g} | {label_of(r)} | {m['rho_weak']:.3e} | "
                  f"{100 * m['res_mode1_frac']:.1f}% | {100 * m['res_modes_gt20_frac']:.2g}% |")
    lines.append("\n".join(md))
    return lines, det, lams, star


def plot_E2(fixed, det, lams, args):
    g = group(fixed, lambda r: (r["config"]["lam"], label_of(r)))
    lam2 = sorted({k[0] for k in g})
    fig, axes = plt.subplots(2, len(lam2), figsize=(4.2 * len(lam2), 7), squeeze=False)
    for j, lam in enumerate(lam2):
        for name, st in METHODS.items():
            r = g[(lam, name)][0]
            h = r["history"]
            axes[0, j].semilogy(runmin(h["loss"]), color=st["color"], label=name, lw=1.4)
            axes[1, j].plot(h["log_it"], h["log_maxu"], color=st["color"], label=name)
        n_adam = g[(lam, "PINN")][0]["config"]["n_adam"]
        for i in range(2):
            axes[i, j].axvline(n_adam, color="0.5", lw=1, ls=":")
        axes[0, j].set_title(f"λ = {lam:g}: best loss so far / initial (seed 0)")
        axes[1, j].set_title(f"λ = {lam:g}: max u during training")
        axes[1, j].set_xlabel("iteration (Adam | L-BFGS evaluations)")
        axes[0, j].legend(); axes[1, j].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2_nonexistence_training.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes2 = plt.subplots(2, 2, figsize=(11.5, 8))
    axes = axes2.ravel()
    for name, st in METHODS.items():
        kw = dict(color=st["color"], marker=st["marker"], ms=4, label=name)
        axes[0].semilogy(lams, det[name]["train"], **kw)
        axes[1].semilogy(lams, det[name]["weak"], **kw)
        axes[2].plot(lams, det[name]["maxu"], **kw)
    ll = np.linspace(min(lams), min(7.999, max(lams)), 200)
    axes[2].plot(ll, [max_u_exact(l) for l in ll], label="exact max u", **EXACT_KW)
    # (d) criterion B: linear extrapolation of rho_weak from the lambda > 8 side
    lo, hi = FIT_WINDOWS[0]
    L = np.asarray(lams)
    for name, st in METHODS.items():
        m = L >= 8.0 - 1e-9
        axes[3].plot(L[m], np.asarray(det[name]["weak"])[m], color=st["color"], marker=st["marker"],
                     ms=4, lw=0, label=name)
        ls, dev = floor_extrapolation(lams, det[name]["weak"], lo, hi)
        if not np.isnan(ls) and dev < 0.1:   # only draw fits that are actually linear
            mm = (L >= lo - 1e-9) & (L <= hi + 1e-9)
            a, b = np.polyfit(L[mm], np.asarray(det[name]["weak"])[mm], 1)
            xx = np.linspace(ls, hi, 50)
            axes[3].plot(xx, a * xx + b, color=st["color"], lw=1.2, ls="--",
                         label=f"fit → λ* ≈ {ls:.3f}")
    axes[3].axhline(0, color="0.3", lw=0.8)
    titles = ["Final relative training residual", "A-posteriori residual ρ_weak (H⁻¹, K=150)", "max u",
              f"ρ_weak for λ ≥ 8, linear fit on [{lo:g}, {hi:g}]"]
    for ax, t in zip(axes, titles):
        ax.axvline(LAMBDA_CRIT, color="0.5", lw=1, ls=":")
        ax.set_xlabel("λ"); ax.set_title(t); ax.legend(fontsize=7.5)
    axes[2].set_ylim(0, 9)
    axes[3].set_ylim(-0.01, 0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3_lambda_sweep.png"), bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# E3 -- local vs non-local
# ===================================================================

def table_E3(nonlocal_res, local_res, args):
    gn = group(nonlocal_res, lambda r: (r["config"]["lam"], label_of(r)))
    gl = group(local_res, lambda r: (r["config"]["lam"], label_of(r)))
    md = ["| λ | μ = λ/I² | Method | L2 non-local | L2 local | ratio NL/L | it. to loss < 1e-3·L0 (NL / L) | status local |",
          "|---|---|---|---|---|---|---|---|"]
    rows = []
    for (lam, lab) in sorted(gn, key=lambda k: (k[0], list(METHODS).index(k[1]))):
        a = ms([r["metrics"]["L2"] for r in gn[(lam, lab)]])
        b = ms([r["metrics"]["L2"] for r in gl[(lam, lab)]])

        def it_to(rs):
            out = []
            for r in rs:
                L = np.asarray(r["history"]["loss"])
                idx = np.nonzero(L < 1e-3 * L[0])[0]
                out.append(idx[0] if len(idx) else np.nan)
            return np.nanmedian(out) if not np.all(np.isnan(out)) else np.nan

        ia, ib = it_to(gn[(lam, lab)]), it_to(gl[(lam, lab)])
        md.append(f"| {lam:g} | {mu_local(lam):.3g} | {lab} | {fmt(*a)} | {fmt(*b)} | {a[0] / b[0]:.2f} | "
                  f"{'—' if np.isnan(ia) else int(ia)} / {'—' if np.isnan(ib) else int(ib)} | "
                  f"{', '.join(r['status'] for r in gl[(lam, lab)])} |")
        rows.append([lam, mu_local(lam), lab, *a, *b, a[0] / b[0], ia, ib])
    write_csv(os.path.join(args.out, "E3_local_vs_nonlocal.csv"),
              ["lambda", "mu", "method", "L2_nl_mean", "L2_nl_std", "L2_loc_mean", "L2_loc_std",
               "ratio", "iters_nl", "iters_loc"], rows)
    return "\n".join(md)


def plot_E3(nonlocal_res, local_res, args):
    gn = group(nonlocal_res, lambda r: (r["config"]["lam"], label_of(r)))
    gl = group(local_res, lambda r: (r["config"]["lam"], label_of(r)))
    lams = sorted({k[0] for k in gn})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, st in METHODS.items():
        a = [ms([r["metrics"]["L2"] for r in gn[(l, name)]])[0] for l in lams]
        b = [ms([r["metrics"]["L2"] for r in gl[(l, name)]])[0] for l in lams]
        axes[0].semilogy(lams, a, color=st["color"], marker=st["marker"], label=f"{name} non-local")
        axes[0].semilogy(lams, b, color=st["color"], marker=st["marker"], ls="--", mfc="white",
                         label=f"{name} local")
    axes[0].set_xlabel("λ"); axes[0].set_title("L2 error: non-local (solid) vs local (dashed)")
    axes[0].legend(ncol=2, fontsize=7.5)
    lam_show = 7.5 if 7.5 in lams else lams[len(lams) // 2]
    for name, st in METHODS.items():
        for res, ls in ((gn, "-"), (gl, "--")):
            h = res[(lam_show, name)][0]["history"]
            axes[1].semilogy(runmin(h["loss"]), color=st["color"], ls=ls, lw=1.3)
    axes[1].set_title(f"Best loss so far / initial, λ = {lam_show:g} (solid NL, dashed local)")
    axes[1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig4_local_vs_nonlocal.png"), bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# E4 -- ablations
# ===================================================================

def run_E4(S, args):
    B, lam, seeds = S["budget"], S["E4_lam"], S["E4_seeds"]
    jK = [("VPINN", {**B, "lam": l, "seed": s, "n_test": K})
          for l in S["E4_K_lams"] for K in S["E4_K"] for s in seeds]
    jA = [job(n, lam, s, B, n_hidden=w, n_layers=d)
          for (w, d) in S["E4_arch"] for n in ("PINN", "VPINN K=60") for s in seeds]
    jQ = [job(n, lam, s, B, n_quad=q)
          for q in S["E4_quad"] for n in ("PINN", "VPINN K=60") for s in seeds]
    return cached("E4_K", jK, args), cached("E4_arch", jA, args), cached("E4_quad", jQ, args)


def analyse_E4(rK, rA, rQ, args):
    md = []
    # K
    g = group(rK, lambda r: (r["config"]["lam"], r["config"]["n_test"]))
    t = ["| λ | K | L2 | ρ_weak (a-post.) | final training loss |", "|---|---|---|---|---|"]
    for (lam, K), rs in sorted(g.items()):
        t.append(f"| {lam:g} | {K} | {fmt(*ms([r['metrics']['L2'] for r in rs]))} | "
                 f"{fmt(*ms([r['metrics']['rho_weak'] for r in rs]))} | "
                 f"{fmt(*ms([r['history']['final_loss'] for r in rs]))} |")
    md.append("\n".join(t))
    # arch
    g = group(rA, lambda r: ((r["config"]["n_hidden"], r["config"]["n_layers"]), label_of(r)))
    t = ["| width × depth | Method | L2 | time (s) |", "|---|---|---|---|"]
    for (wd, lab), rs in sorted(g.items()):
        t.append(f"| {wd[0]} × {wd[1]} | {lab} | {fmt(*ms([r['metrics']['L2'] for r in rs]))} | "
                 f"{ms([r['time'] for r in rs])[0]:.0f} |")
    md.append("\n".join(t))
    # quad
    g = group(rQ, lambda r: (r["config"]["n_quad"], label_of(r)))
    t = ["| n_quad | Method | L2 | rel. err. I[u] | ρ_weak |", "|---|---|---|---|---|"]
    for (q, lab), rs in sorted(g.items()):
        t.append(f"| {q} | {lab} | {fmt(*ms([r['metrics']['L2'] for r in rs]))} | "
                 f"{fmt(*ms([r['metrics']['I_rel'] for r in rs]))} | "
                 f"{fmt(*ms([r['metrics']['rho_weak'] for r in rs]))} |")
    md.append("\n".join(t))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    gK = group(rK, lambda r: (r["config"]["lam"], r["config"]["n_test"]))
    for i, lam in enumerate(sorted({k[0] for k in gK})):
        Ks = sorted({k[1] for k in gK if k[0] == lam})
        c = ["#1baf7a", "#eda100"][i % 2]
        axes[0].loglog(Ks, [ms([r["metrics"]["L2"] for r in gK[(lam, K)]])[0] for K in Ks],
                       color=c, marker="o", label=f"L2 error, λ = {lam:g}")
        axes[0].loglog(Ks, [ms([r["metrics"]["rho_weak"] for r in gK[(lam, K)]])[0] for K in Ks],
                       color=c, marker="o", ls="--", mfc="white", label=f"ρ_weak, λ = {lam:g}")
    axes[0].set_xlabel("K (test functions)"); axes[0].set_title("VPINN: effect of K"); axes[0].legend()
    gA = group(rA, lambda r: (label_of(r), (r["config"]["n_hidden"], r["config"]["n_layers"])))
    archs = sorted({k[1] for k in gA})
    xs = np.arange(len(archs))
    for k, name in enumerate(("PINN", "VPINN K=60")):
        st = METHODS[name]
        axes[1].semilogy(xs, [ms([r["metrics"]["L2"] for r in gA[(name, a)]])[0] for a in archs],
                         color=st["color"], marker=st["marker"], label=name)
    axes[1].set_xticks(xs, [f"{w}×{d}" for w, d in archs])
    axes[1].set_xlabel("width × depth"); axes[1].set_title(f"Architecture, λ = {args.E4_lam:g}")
    axes[1].legend()
    gQ = group(rQ, lambda r: (label_of(r), r["config"]["n_quad"]))
    qs = sorted({k[1] for k in gQ})
    for name in ("PINN", "VPINN K=60"):
        st = METHODS[name]
        axes[2].loglog(qs, [ms([r["metrics"]["L2"] for r in gQ[(name, q)]])[0] for q in qs],
                       color=st["color"], marker=st["marker"], label=name)
    axes[2].set_xlabel("Gauss-Legendre points n_quad"); axes[2].set_title(f"Quadrature, λ = {args.E4_lam:g}")
    axes[2].legend()
    from matplotlib.ticker import NullLocator
    allK = sorted({k[1] for k in gK})
    for ax, ticks in ((axes[0], allK), (axes[2], qs)):
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(ticks, [str(t) for t in ticks])
    for ax in axes[1:]:
        ax.set_ylabel("L2 error")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig5_ablations.png"), bbox_inches="tight")
    plt.close(fig)
    return md


# ===================================================================
# Main
# ===================================================================

def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--only", default="E1,E2,E3,E4")
    p.add_argument("--force", action="store_true")
    p.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()
    args.out = os.path.normpath(os.path.join(HERE, "..", "results_quick" if args.quick else "results"))
    os.makedirs(args.out, exist_ok=True)
    S = settings(args.quick)
    args.E4_lam = S["E4_lam"]
    only = args.only.split(",")
    t0 = time.time()
    report = [f"# Results ({'quick' if args.quick else 'full'} run)\n",
              f"Budget per training: {S['budget']['n_adam']} Adam iterations + "
              f"≤ {S['budget']['n_lbfgs']} L-BFGS evaluations. Mean ± std over seeds.\n"]

    if "E1" in only or "E3" in only:
        e1 = run_E1(S, args)
        report += ["## E1 — accuracy vs λ\n", table_E1(e1, args), ""]
        plot_E1(e1, args)
    if "E2" in only:
        fixed, sweep, modes = run_E2(S, args)
        lines, det, lams, star = analyse_E2(fixed, sweep, modes, args)
        report += ["## E2 — λ ≥ 8: loss floors\n", lines[0], "",
                   "## E2 — criterion A: λ* seen by each method (smallest λ from which the detector stays > τ)\n",
                   lines[1], "",
                   "## E2 — criterion B: λ* from linear extrapolation of the post-critical residual floor\n",
                   lines[2], "",
                   "## E2 — modal content of the a-posteriori residual (seed 0)\n", lines[3], ""]
        plot_E2(fixed, det, lams, args)
    if "E3" in only:
        e3 = run_E1(S, args, problem="local", tag="E3_local")
        report += ["## E3 — non-local vs local (same exact solution)\n", table_E3(e1, e3, args), ""]
        plot_E3(e1, e3, args)
    if "E4" in only:
        rK, rA, rQ = run_E4(S, args)
        md = analyse_E4(rK, rA, rQ, args)
        report += [f"## E4 — ablations (λ = {S['E4_lam']:g})\n", "### Number of test functions K (VPINN)\n",
                   md[0], "", "### Width × depth\n", md[1], "", "### Quadrature points\n", md[2], ""]

    with open(os.path.join(args.out, "results_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print("\n".join(report))
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
