"""
Robustness checks of the main claims of run_all.py.

    python study_robustness.py            # ~25 min on 10 CPU processes
    python study_robustness.py --force    # ignore cached runs

R1  Local problem (E3): is the failure for lambda >= 7.5 only a budget issue?
    Variants: 4x budget, Adam learning-rate decay, explicit continuation in mu
    (geometric ramp lambda -> mu, mimicking what the non-local term does).
R2  Spurious VPINN solutions: frequency vs K and lambda over 5 seeds,
    longer training, and test functions normalised by ||v_k'|| (H^-1 loss).
R3  PINN collocation: Gauss nodes (default) vs uniform / random points.
R4  Two more seeds (5 in total) at lambda = 7.5 and 7.9 for E1.
R5  Criterion B (floor extrapolation): per-seed estimates from only 5 points
    lambda in {8.1, ..., 8.5}, other architecture, other K, normalised tests.
R6  lambda = 7.99: continuation in lambda, finer quadrature, 4x budget.
R7  Criterion B from 3-point windows at various distances (no new training).
R8  Growth of max u at the end of training as a marker of lambda* (no new training).
R9  Spurious-solution threshold in K vs network width.

Outputs: ../results/robustness_summary.md, ../results/fig6_robustness.png,
../results/fig7_followups.png,
CSV tables in ../results/robustness/.  Raw runs are cached in ../results/raw/.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_all import cached, METHODS, group, ms, fmt, label_of, write_csv, HERE  # noqa: E402

B = {"n_adam": 5000, "n_lbfgs": 2000}
B4 = {"n_adam": 20000, "n_lbfgs": 8000}
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical order

SPURIOUS_TRAIN = 1e-6    # relative training residual below this ...
SPURIOUS_VAL = 1e-2      # ... while the a-posteriori rho_weak is above this


def is_spurious(r):
    return r["history"]["final_rel"] < SPURIOUS_TRAIN and r["metrics"]["rho_weak"] > SPURIOUS_VAL


def load_raw(args, name):
    with open(os.path.join(args.out, "raw", f"{name}.pkl"), "rb") as fh:
        return pickle.load(fh)["results"]


def vp(K, **kw):
    return ("VPINN", {**B, "n_test": K, **kw})


def pn(**kw):
    return ("PINN", {**B, **kw})


def with_(j, **kw):
    return (j[0], {**j[1], **kw})


# ===================================================================
# R1 -- local problem
# ===================================================================

R1_LAMS = [7.5, 7.9]
R1_VARIANTS = {
    "4× budget": {**B4},
    "lr decay 1e-3→1e-5": {"lr_end": 1e-5},
    "μ-continuation (3000 it.)": {"mu_ramp": 3000},
}


def run_R1(args):
    jobs = []
    for vname, extra in R1_VARIANTS.items():
        for lam in R1_LAMS:
            for base in (pn(), vp(60)):
                for s in range(3):
                    jobs.append(with_(base, lam=lam, seed=s, problem="local", **extra))
    return cached("R1_local", jobs, args)


def report_R1(res, args):
    base = [r for r in load_raw(args, "E3_local") if r["config"]["lam"] in R1_LAMS
            and label_of(r) in ("PINN", "VPINN K=60")]
    nonloc = [r for r in load_raw(args, "E1") if r["config"]["lam"] in R1_LAMS
              and label_of(r) in ("PINN", "VPINN K=60")]

    def variant(r):
        c = r["config"]
        if c["n_adam"] == B4["n_adam"]:
            return "4× budget"
        if c.get("lr_end"):
            return "lr decay 1e-3→1e-5"
        if c.get("mu_ramp"):
            return "μ-continuation (3000 it.)"
        return "baseline"

    rows = {}
    for r in base + res:
        rows.setdefault((r["config"]["lam"], label_of(r), variant(r)), []).append(r)
    for r in nonloc:
        rows.setdefault((r["config"]["lam"], label_of(r), "non-local (reference)"), []).append(r)
    order = ["non-local (reference)", "baseline"] + list(R1_VARIANTS)
    md = ["| λ | Method | Variant | L2 per seed | converged (L2 < 1e-3) | status |", "|---|---|---|---|---|---|"]
    csv_rows = []
    for lam in R1_LAMS:
        for meth in ("PINN", "VPINN K=60"):
            for v in order:
                rs = sorted(rows[(lam, meth, v)], key=lambda r: r["config"]["seed"])
                l2 = [r["metrics"]["L2"] for r in rs]
                st = ", ".join(sorted({r["status"] for r in rs}))
                md.append(f"| {lam:g} | {meth} | {v} | " + ", ".join(f"{x:.1e}" for x in l2)
                          + f" | {sum(x < 1e-3 for x in l2)}/{len(l2)} | {st} |")
                for r in rs:
                    csv_rows.append([lam, meth, v, r["config"]["seed"], r["metrics"]["L2"],
                                     r["history"]["final_loss"], r["status"], r["time"]])
    write_csv(os.path.join(args.rob, "R1_local.csv"),
              ["lambda", "method", "variant", "seed", "L2", "final_loss", "status", "time_s"], csv_rows)
    return "\n".join(md), rows, order


# ===================================================================
# R2 -- spurious VPINN solutions
# ===================================================================

R2_LAMS = [7.5, 7.9, 8.0, 8.3, 8.6]
R2_K = [10, 20, 30, 40]


def run_R2(args):
    jobs = [with_(vp(K), lam=lam, seed=s) for K in R2_K for lam in R2_LAMS for s in range(5)]
    jobs += [with_(vp(K, normalize_tests=True), lam=lam, seed=s)
             for K in (20, 60) for lam in R2_LAMS for s in range(3)]
    jobs += [with_(vp(20, **B4), lam=lam, seed=s) for lam in (8.0, 8.3, 8.6) for s in range(3)]
    return cached("R2_spurious", jobs, args)


def r2_key(r):
    c = r["config"]
    tag = f"K={c['n_test']}"
    if c.get("normalize_tests"):
        tag += " normalised"
    if c["n_adam"] == B4["n_adam"]:
        tag += " 4× budget"
    return tag


def report_R2(res, args):
    g = group(res, lambda r: (r2_key(r), r["config"]["lam"]))
    keys = sorted({k[0] for k in g}, key=lambda t: (("normalised" in t) + 2 * ("budget" in t),
                                                    int(t.split()[0][2:])))
    md = ["| Variant | " + " | ".join(f"λ = {l:g}" for l in R2_LAMS) + " |",
          "|---|" + "---|" * len(R2_LAMS)]
    frac, counts = {}, {}
    csv_rows = []
    for k in keys:
        cells = []
        for lam in R2_LAMS:
            rs = g.get((k, lam))
            if not rs:
                cells.append("—")
                continue
            n_sp = sum(is_spurious(r) for r in rs)
            frac[(k, lam)] = n_sp / len(rs)
            counts[(k, lam)] = f"{n_sp}/{len(rs)}"
            l2 = [r["metrics"].get("L2", np.nan) for r in rs]
            extra = f", L2 {np.median(l2):.0e}" if lam < 8 else ""
            cells.append(f"**{n_sp}/{len(rs)}**{extra}" if n_sp else f"0/{len(rs)}{extra}")
            for r in rs:
                csv_rows.append([k, lam, r["config"]["seed"], r["history"]["final_loss"],
                                 r["history"]["final_rel"], r["metrics"]["rho_weak"],
                                 r["metrics"].get("L2", np.nan), int(is_spurious(r))])
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    write_csv(os.path.join(args.rob, "R2_spurious.csv"),
              ["variant", "lambda", "seed", "final_loss", "rel_train", "rho_weak", "L2", "spurious"], csv_rows)
    return "\n".join(md), frac, keys, counts


# ===================================================================
# R3 -- PINN collocation
# ===================================================================

R3_LAMS = [4.0, 7.0, 7.5, 7.9]
R3_VARIANTS = {"uniform, 128": {"collocation": "uniform"},
               "random, 128": {"collocation": "random"},
               "uniform, 512": {"collocation": "uniform", "n_col": 512}}


def run_R3(args):
    jobs = [with_(pn(**v), lam=lam, seed=s) for v in R3_VARIANTS.values() for lam in R3_LAMS for s in range(3)]
    return cached("R3_collocation", jobs, args)


def report_R3(res, args):
    e1 = [r for r in load_raw(args, "E1") if r["config"]["lam"] in R3_LAMS]

    def vname(r):
        c = r["config"]
        if r["method"] != "PINN":
            return label_of(r)
        if c.get("collocation", "gauss") == "gauss":
            return "PINN, Gauss 128 (default)"
        return "PINN, " + c["collocation"] + ", " + str(c.get("n_col") or c["n_quad"])

    g = group(e1 + res, lambda r: (vname(r), r["config"]["lam"]))
    order = ["PINN, Gauss 128 (default)", "PINN, uniform, 128", "PINN, random, 128", "PINN, uniform, 512",
             "VPINN K=60"]
    md = ["| Variant | " + " | ".join(f"λ = {l:g}" for l in R3_LAMS) + " |",
          "|---|" + "---|" * len(R3_LAMS)]
    for v in order:
        md.append(f"| {v} | " + " | ".join(fmt(*ms([r["metrics"]["L2"] for r in g[(v, l)]]))
                                          for l in R3_LAMS) + " |")
    return "\n".join(md)


# ===================================================================
# R4 -- more seeds
# ===================================================================

def run_R4(args):
    jobs = []
    for lam in (7.5, 7.9):
        for name in METHODS:
            m, base = METHODS[name]["job"]
            for s in (3, 4):
                jobs.append((m, {**base, **B, "lam": lam, "seed": s}))
    return cached("R4_seeds", jobs, args)


def report_R4(res, args):
    e1 = [r for r in load_raw(args, "E1") if r["config"]["lam"] in (7.5, 7.9)]
    g = group(e1 + res, lambda r: (r["config"]["lam"], label_of(r)))
    md = ["| λ | Method | L2 (5 seeds) mean ± std | median | min – max | H1 mean |", "|---|---|---|---|---|---|"]
    for lam in (7.5, 7.9):
        for name in METHODS:
            l2 = np.array([r["metrics"]["L2"] for r in g[(lam, name)]])
            h1 = np.array([r["metrics"]["H1"] for r in g[(lam, name)]])
            md.append(f"| {lam:g} | {name} | {fmt(l2.mean(), l2.std())} | {np.median(l2):.1e} | "
                      f"{l2.min():.1e} – {l2.max():.1e} | {h1.mean():.1e} |")
    return "\n".join(md)


# ===================================================================
# R5 -- robustness of criterion B
# ===================================================================

R5_LAMS = [8.1, 8.2, 8.3, 8.4, 8.5]
LINEAR_TOL = 0.05   # acceptance test of criterion B: max deviation of the 5 points from the fitted line


def run_R5(args):
    jobs = [with_(j, lam=lam, seed=s) for j in (pn(), vp(60)) for s in (2, 3, 4) for lam in R5_LAMS]
    jobs += [with_(j, lam=lam, seed=s, n_hidden=64, n_layers=4) for j in (pn(), vp(60))
             for s in (0, 1) for lam in R5_LAMS]
    jobs += [with_(vp(K), lam=lam, seed=s) for K in (40, 120) for s in (0, 1) for lam in R5_LAMS]
    jobs += [with_(vp(60, normalize_tests=True), lam=lam, seed=s) for s in (0, 1) for lam in R5_LAMS]
    return cached("R5_criterionB", jobs, args)


def r5_config(r):
    c = r["config"]
    name = "PINN" if r["method"] == "PINN" else f"VPINN K={c['n_test']}"
    if c["n_hidden"] != 32:
        name += f" ({c['n_hidden']}×{c['n_layers']})"
    if c.get("normalize_tests"):
        name += " normalised"
    return name


def report_R5(res, args):
    sweep = [r for r in load_raw(args, "E2_sweep")
             if any(abs(r["config"]["lam"] - l) < 1e-9 for l in R5_LAMS) and label_of(r) != "VPINN K=20"]
    allr = sweep + res
    g = group(allr, lambda r: (r5_config(r), r["config"]["seed"]))
    est = {}
    for (cname, seed), rs in g.items():
        rs = sorted(rs, key=lambda r: r["config"]["lam"])
        if len(rs) != len(R5_LAMS):
            continue
        L = np.array([r["config"]["lam"] for r in rs])
        for kind, get in (("ρ_weak", lambda r: r["metrics"]["rho_weak"]),
                          ("training residual", lambda r: r["history"]["final_rel"])):
            y = np.array([get(r) for r in rs])
            a, b = np.polyfit(L, y, 1)
            dev = np.max(np.abs(np.polyval([a, b], L) - y)) / y.max()
            est.setdefault((cname, kind), []).append((seed, -b / a, dev))
    order = ["PINN", "PINN (64×4)", "VPINN K=40", "VPINN K=60", "VPINN K=60 (64×4)",
             "VPINN K=60 normalised", "VPINN K=120"]
    md = ["| Configuration | detector | seeds | λ* per seed | mean ± std | max deviation from linear | "
          f"accepted fits (dev. < {100 * LINEAR_TOL:g}%) mean |",
          "|---|---|---|---|---|---|---|"]
    csv_rows = []
    for cname in order:
        for kind in ("ρ_weak", "training residual"):
            e = sorted(est.get((cname, kind), []))
            if not e:
                continue
            vals = np.array([x[1] for x in e])
            acc = np.array([x[1] for x in e if x[2] < LINEAR_TOL])
            acc_s = f"{acc.mean():.3f} ({len(acc)}/{len(e)})" if len(acc) else f"— (0/{len(e)})"
            md.append(f"| {cname} | {kind} | {len(e)} | " + ", ".join(f"{v:.3f}" for v in vals)
                      + f" | {vals.mean():.3f} ± {vals.std():.3f} | {100 * max(x[2] for x in e):.1f}% | {acc_s} |")
            for s, v, d in e:
                csv_rows.append([cname, kind, s, v, d])
    write_csv(os.path.join(args.rob, "R5_criterionB.csv"),
              ["configuration", "detector", "seed", "lambda_star", "max_rel_deviation"], csv_rows)
    return "\n".join(md), est, order


# ===================================================================
# Figure
# ===================================================================

def plot(r1, r2, r5, args):
    rows1, order1 = r1[1], r1[2]
    frac, keys2 = r2[1], r2[2]
    est, order5 = r5[1], r5[2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    xs = np.arange(len(order1))
    for i, (meth, st) in enumerate((("PINN", METHODS["PINN"]), ("VPINN K=60", METHODS["VPINN K=60"]))):
        for lam, filled in ((7.5, True), (7.9, False)):
            for j, v in enumerate(order1):
                l2 = [r["metrics"]["L2"] for r in rows1[(lam, meth, v)]]
                off = (i - 0.5) * 0.3 + (0.07 if lam == 7.9 else -0.07)
                ax.scatter(np.full(len(l2), j + off), l2, s=28, color=st["color"], marker=st["marker"],
                           facecolors=st["color"] if filled else "white", linewidths=1.2,
                           label=f"{meth}, λ = {lam:g}" if j == 0 else None, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(xs, [o.replace(" (", "\n(").replace("lr decay ", "lr decay\n") for o in order1], fontsize=7.5)
    ax.axhline(1e-3, color="0.5", lw=0.8, ls=":")
    ax.set_ylabel("L2 error (one point per seed)")
    ax.set_title("R1 — local problem −u'' = μ exp(−u): training variants")
    ax.legend(fontsize=7.5)

    # R2 as an annotated grid (one sequential hue; the count is printed in every cell)
    ax = axes[1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f6f5f1", "#eb6834"])
    M = np.full((len(keys2), len(R2_LAMS)), np.nan)
    for i, k in enumerate(keys2):
        for j, lam in enumerate(R2_LAMS):
            if (k, lam) in frac:
                M[i, j] = frac[(k, lam)]
    ax.imshow(np.nan_to_num(M, nan=0.0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    counts = r2[3]
    for i, k in enumerate(keys2):
        for j, lam in enumerate(R2_LAMS):
            txt = counts.get((k, lam), "—")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="black" if np.isnan(M[i, j]) or M[i, j] < 0.6 else "white")
    ax.set_xticks(range(len(R2_LAMS)), [f"λ = {l:g}" for l in R2_LAMS])
    ax.set_yticks(range(len(keys2)), keys2, fontsize=8)
    ax.grid(False)
    ax.set_title("R2 — spurious VPINN solutions (seeds)\ntraining residual < 1e-6 and ρ_weak > 1e-2")

    ax = axes[2]
    pos = 0
    ticks, labels, off = [], [], []
    lo, hi = 7.88, 8.12
    for cname in order5:
        for kind, mk in (("ρ_weak", "o"), ("training residual", "s")):
            e = est.get((cname, kind))
            if not e:
                continue
            v = np.array([x[1] for x in e])
            c = METHODS["PINN"]["color"] if cname.startswith("PINN") else METHODS["VPINN K=60"]["color"]
            inside = (v >= lo) & (v <= hi)
            ax.scatter(np.full(inside.sum(), pos), v[inside], color=c, marker=mk, s=26, zorder=3,
                       facecolors=c if kind == "ρ_weak" else "white")
            for s_, val, dev in e:
                if not lo <= val <= hi:
                    off.append((pos, val, dev))
            ticks.append(pos)
            labels.append(f"{cname} · {'ρ_w' if kind == 'ρ_weak' else 'train'}")
            pos += 1
    for p_, val, dev in off:
        ax.annotate(f"off-scale: {val:.2f}\n(linearity dev. {100 * dev:.0f}%)", xy=(p_, lo + 0.005),
                    xytext=(p_ + 0.6, lo + 0.05), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.scatter([p_], [lo + 0.003], marker="v", color="0.3", s=30, zorder=3)
    ax.axhline(8.0, color="black", lw=1, ls="--", label="exact λ* = 8")
    ax.set_ylim(lo, hi)
    ax.set_xticks(ticks, labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("extrapolated λ*")
    ax.set_title("R5 — criterion B from 5 points λ ∈ [8.1, 8.5]\n(one point per seed; filled ρ_weak, open training)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig6_robustness.png"), bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# R6 -- continuation in lambda close to the threshold
# ===================================================================

R6_VARIANTS = {
    "default": {},
    "n_quad 256": {"n_quad": 256},
    "4× budget": {**B4},
    "n_quad 256 + 4× budget": {"n_quad": 256, **B4},
    "λ-ramp": {"lam_ramp": 3000},
    "λ-ramp + n_quad 256": {"lam_ramp": 3000, "n_quad": 256},
    "λ-ramp + n_quad 256 + 4× budget": {"lam_ramp": 12000, "n_quad": 256, **B4},
}
R6_METHODS = {"PINN": pn(), "VPINN K=60": vp(60), "VPINN K=120": vp(120)}


R6_BUDGET_ONLY = ("4× budget", "n_quad 256 + 4× budget")   # added afterwards to separate the effects


def run_R6(args):
    jobs = []
    for vname, extra in R6_VARIANTS.items():
        if vname in R6_BUDGET_ONLY:
            continue
        for mname, base in R6_METHODS.items():
            for s in range(3):
                if vname == "default" and mname != "VPINN K=120":
                    continue            # already in E1
                jobs.append(with_(base, lam=7.99, seed=s, **extra))
    jobs += [with_(base, lam=7.9, seed=s, lam_ramp=3000) for base in R6_METHODS.values() for s in range(3)]
    jobs += [with_(vp(120), lam=7.9, seed=s) for s in range(3)]
    res = cached("R6_lambda_continuation", jobs, args)
    jobs = [with_(base, lam=7.99, seed=s, **R6_VARIANTS[v]) for v in R6_BUDGET_ONLY
            for base in (pn(), vp(60)) for s in range(3)]
    return res + cached("R6b_budget_only", jobs, args)


def r6_variant(r):
    c = r["config"]
    for vname in reversed(list(R6_VARIANTS)):
        ex = R6_VARIANTS[vname]
        if ex and all(c.get(k) == v for k, v in ex.items()):
            return vname
    return "default"


def report_R6(res, args):
    e1 = [r for r in load_raw(args, "E1") if r["config"]["lam"] in (7.9, 7.99) and label_of(r) != "VPINN K=20"]
    g = group(e1 + res, lambda r: (r["config"]["lam"], label_of(r), r6_variant(r)))
    md = ["| λ | Method | Variant | L2 (3 seeds) | rel. err. max u | time (s) |", "|---|---|---|---|---|---|"]
    order = list(R6_VARIANTS)
    for lam in (7.99, 7.9):
        for mname in R6_METHODS:
            for v in order:
                rs = g.get((lam, mname, v))
                if not rs:
                    continue
                md.append(f"| {lam:g} | {mname} | {v} | {fmt(*ms([r['metrics']['L2'] for r in rs]))} | "
                          f"{fmt(*ms([r['metrics']['maxu_rel'] for r in rs]))} | "
                          f"{ms([r['time'] for r in rs])[0]:.0f} |")
    return "\n".join(md), g, order


# ===================================================================
# R7 -- criterion B with fewer / other points (no new training)
# ===================================================================

def r7_windows():
    wins = []
    for n, step in ((3, 0.05), (3, 0.1), (5, 0.1)):
        for start in np.round(np.arange(8.0, 9.0 - (n - 1) * step + 1e-9, 0.05), 2):
            wins.append((n, step, float(start)))
    return wins


def report_R7(args):
    sweep = [r for r in load_raw(args, "E2_sweep") if label_of(r) in ("PINN", "VPINN K=60")]
    g = group(sweep, lambda r: (label_of(r), r["config"]["seed"]))
    out = {}
    for (name, seed), rs in g.items():
        lam2r = {round(r["config"]["lam"], 2): r for r in rs}
        for kind, get in (("ρ_weak", lambda r: r["metrics"]["rho_weak"]),
                          ("training residual", lambda r: r["history"]["final_rel"])):
            for (n, step, start) in r7_windows():
                L = np.round(start + step * np.arange(n), 2)
                if not all(l in lam2r for l in L):
                    continue
                y = np.array([get(lam2r[l]) for l in L])
                a, b = np.polyfit(L, y, 1)
                dev = np.max(np.abs(np.polyval([a, b], L) - y)) / y.max()
                out.setdefault((name, kind, n, step), []).append((start, seed, -b / a, dev))
    picks = [(3, 0.05, 8.05), (3, 0.1, 8.0), (3, 0.1, 8.1), (3, 0.1, 8.3), (3, 0.1, 8.6),
             (5, 0.1, 8.0), (5, 0.1, 8.1), (5, 0.1, 8.5)]
    hdr = " | ".join(f"{n} pts [{s:g}, {s + (n - 1) * st:g}]" for n, st, s in picks)
    md = [f"| Method | detector | {hdr} |", "|---|---|" + "---|" * len(picks)]
    for name in ("PINN", "VPINN K=60"):
        for kind in ("ρ_weak", "training residual"):
            cells = []
            for n, st, s in picks:
                v = [x for x in out.get((name, kind, n, st), []) if abs(x[0] - s) < 1e-9]
                if not v:
                    cells.append("—")
                    continue
                vals = [x[2] for x in v]
                flag = "" if max(x[3] for x in v) < LINEAR_TOL else " ✗"
                cells.append(f"{np.mean(vals):.3f}{flag}")
            md.append(f"| {name} | {kind} | " + " | ".join(cells) + " |")
    return "\n".join(md), out


# ===================================================================
# R8 -- dynamic detector: growth of max u at the end of training
# ===================================================================

def growth(r, last=1000):
    """Relative growth of max u over the last `last` loss evaluations."""
    it = np.asarray(r["history"]["log_it"])
    mu = np.asarray(r["history"]["log_maxu"])
    if len(it) < 3:
        return np.nan
    j = np.searchsorted(it, it[-1] - last)
    return (mu[-1] - mu[j]) / abs(mu[-1])


def report_R8(args):
    sweep = load_raw(args, "E2_sweep")
    g = group(sweep, lambda r: (label_of(r), r["config"]["lam"]))
    lams = sorted({r["config"]["lam"] for r in sweep})
    curves = {}
    md = ["| Method | max growth for λ < 7.95 | growth at 7.95 / 7.99 region | min growth for λ ≥ 8 | "
          "growth at λ = 8.0 / 8.5 / 9.0 |", "|---|---|---|---|---|"]
    for name in METHODS:
        gr = [np.median([growth(r) for r in g[(name, l)]]) for l in lams]
        curves[name] = gr
        G = dict(zip(lams, gr))
        below = [G[l] for l in lams if l < 7.95 - 1e-9]
        above = [G[l] for l in lams if l >= 8.0 - 1e-9]
        md.append(f"| {name} | {max(below):.2e} | {G.get(7.95, np.nan):.2e} | {min(above):.2e} | "
                  f"{G.get(8.0, np.nan):.2e} / {G.get(8.5, np.nan):.2e} / {G.get(9.0, np.nan):.2e} |")
    return "\n".join(md), curves, lams


# ===================================================================
# R9 -- spurious threshold in K vs network width
# ===================================================================

R9_K = [15, 20, 25, 30]
R9_W = [16, 32, 64]


def n_params(w, d=4):
    return (1 * w + w) + (d - 1) * (w * w + w) + (w + 1)


def run_R9(args):
    jobs = [with_(vp(K), lam=lam, seed=s, n_hidden=w, n_layers=4)
            for lam in (7.9, 8.3) for w in R9_W for K in R9_K for s in range(3)]
    return cached("R9_K_width", jobs, args)


def report_R9(res, args):
    g = group(res, lambda r: (r["config"]["lam"], r["config"]["n_hidden"], r["config"]["n_test"]))
    md = []
    grids = {}
    for lam in (7.9, 8.3):
        t = [f"| width (params) | " + " | ".join(f"K = {K}" for K in R9_K) + " |", "|---|" + "---|" * len(R9_K)]
        for w in R9_W:
            cells = []
            for K in R9_K:
                rs = g[(lam, w, K)]
                n = sum(is_spurious(r) for r in rs)
                grids[(lam, w, K)] = f"{n}/{len(rs)}"
                extra = f", L2 {np.median([r['metrics']['L2'] for r in rs]):.0e}" if lam < 8 else ""
                cells.append((f"**{n}/{len(rs)}**" if n else f"0/{len(rs)}") + extra)
            t.append(f"| {w} ({n_params(w)}) | " + " | ".join(cells) + " |")
        md.append(f"λ = {lam:g}\n\n" + "\n".join(t))
    return "\n\n".join(md), grids


def plot_followups(r6, r7, r8, r9, args):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # (a) R6
    ax = axes[0, 0]
    g6, order6 = r6[1], r6[2]
    cols = {"PINN": PAL[0], "VPINN K=60": PAL[2], "VPINN K=120": PAL[3]}
    mks = {"PINN": "o", "VPINN K=60": "^", "VPINN K=120": "D"}
    for i, m in enumerate(R6_METHODS):
        for j, v in enumerate(order6):
            rs = g6.get((7.99, m, v), [])
            l2 = [r["metrics"]["L2"] for r in rs]
            ax.scatter(np.full(len(l2), j + (i - 1) * 0.2), l2, color=cols[m], marker=mks[m], s=28,
                       label=m if j == 0 else None, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order6)), [v.replace(" + ", "\n+ ") for v in order6], fontsize=7.5)
    ax.set_ylabel("L2 error (one point per seed)")
    ax.set_title("R6 — λ = 7.99 (max u = 6.7, u'(0) = 174)")
    ax.legend(fontsize=8)
    # (b) R7: lambda* vs window start, 3-point windows step 0.1
    ax = axes[0, 1]
    out7 = r7[1]
    for name, c in (("PINN", METHODS["PINN"]["color"]), ("VPINN K=60", METHODS["VPINN K=60"]["color"])):
        for kind, ls, mk in (("ρ_weak", "-", "o"), ("training residual", "--", "s")):
            v = out7.get((name, kind, 3, 0.1), [])
            starts = sorted({x[0] for x in v})
            ys = [np.mean([x[2] for x in v if x[0] == s]) for s in starts]
            ax.plot(starts, ys, color=c, ls=ls, marker=mk, ms=4, mfc="white" if ls == "--" else c,
                    label=f"{name} · {'ρ_w' if kind == 'ρ_weak' else 'train'}")
    ax.axhline(8.0, color="black", lw=1, ls=":")
    ax.set_ylim(7.7, 8.4)
    ax.set_xlabel("first λ of the 3-point window (step 0.1)")
    ax.set_ylabel("extrapolated λ*")
    ax.set_title("R7 — criterion B from only 3 points")
    ax.legend(fontsize=7.5)
    # (c) R8
    ax = axes[1, 0]
    curves, lams = r8[1], r8[2]
    for name, st in METHODS.items():
        ax.plot(lams, curves[name], color=st["color"], marker=st["marker"], ms=4, label=name)
    ax.axvline(8.0, color="0.5", lw=1, ls=":")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("λ"); ax.set_ylabel("relative growth of max u, last 1000 evaluations")
    ax.set_title("R8 — is max u still growing at the end of training?")
    ax.legend(fontsize=8)
    # (d) R9 grid at lambda = 8.3
    ax = axes[1, 1]
    grids = r9[1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f6f5f1", "#eb6834"])
    rows = [(lam, w) for lam in (7.9, 8.3) for w in R9_W]
    M = np.array([[int(grids[(lam, w, K)].split("/")[0]) / 3 for K in R9_K] for lam, w in rows])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, (lam, w) in enumerate(rows):
        for j, K in enumerate(R9_K):
            ax.text(j, i, grids[(lam, w, K)], ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] >= 0.6 else "black")
    ax.set_xticks(range(len(R9_K)), [f"K = {K}" for K in R9_K])
    ax.set_yticks(range(len(rows)), [f"λ = {lam:g}, width {w}" for lam, w in rows], fontsize=8)
    ax.axhline(2.5, color="0.3", lw=1)
    ax.grid(False)
    ax.set_title("R9 — spurious VPINN solutions vs K and width (3 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig7_followups.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()
    args.out = os.path.normpath(os.path.join(HERE, "..", "results"))
    args.rob = os.path.join(args.out, "robustness")
    os.makedirs(args.rob, exist_ok=True)
    for need in ("E1", "E3_local", "E2_sweep"):
        if not os.path.exists(os.path.join(args.out, "raw", f"{need}.pkl")):
            sys.exit(f"Run run_all.py first (missing results/raw/{need}.pkl)")
    t0 = time.time()
    r1 = report_R1(run_R1(args), args)
    r2 = report_R2(run_R2(args), args)
    r3 = report_R3(run_R3(args), args)
    r4 = report_R4(run_R4(args), args)
    r5 = report_R5(run_R5(args), args)
    plot(r1, r2, r5, args)
    r6 = report_R6(run_R6(args), args)
    r7 = report_R7(args)
    r8 = report_R8(args)
    r9 = report_R9(run_R9(args), args)
    plot_followups(r6, r7, r8, r9, args)
    rep = ["# Robustness checks\n",
           f"Default budget {B['n_adam']} Adam + ≤ {B['n_lbfgs']} L-BFGS evaluations; "
           f"'4× budget' = {B4['n_adam']} + {B4['n_lbfgs']}.\n",
           "## R1 — local problem: is the failure a budget issue?\n", r1[0], "",
           f"## R2 — spurious VPINN solutions (training residual < {SPURIOUS_TRAIN:g} and "
           f"ρ_weak > {SPURIOUS_VAL:g}); count over seeds, median L2 for λ < 8\n", r2[0], "",
           "## R3 — PINN collocation points (L2, 3 seeds)\n", r3, "",
           "## R4 — 5 seeds at λ = 7.5 and 7.9\n", r4, "",
           "## R5 — criterion B from 5 points λ ∈ {8.1, …, 8.5}, one estimate per seed\n", r5[0], "",
           "## R6 — continuation in λ near the threshold\n", r6[0], "",
           f"## R7 — criterion B from other windows (mean of 2 seeds; ✗ = a fit fails the "
           f"{100 * LINEAR_TOL:g}% linearity test)\n", r7[0], "",
           "## R8 — relative growth of max u over the last 1000 loss evaluations (median of 2 seeds)\n",
           r8[0], "",
           "## R9 — spurious VPINN solutions vs K and network width (3 seeds)\n", r9[0], ""]
    with open(os.path.join(args.out, "robustness_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep))
    print("\n".join(rep))
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
