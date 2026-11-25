# Robustness checks

Default budget 5000 Adam + ≤ 2000 L-BFGS evaluations; '4× budget' = 20000 + 8000.

## R1 — local problem: is the failure a budget issue?

| λ | Method | Variant | L2 per seed | converged (L2 < 1e-3) | status |
|---|---|---|---|---|---|
| 7.5 | PINN | non-local (reference) | 5.5e-06, 1.6e-05, 9.6e-06 | 3/3 | ok |
| 7.5 | PINN | baseline | 2.1e-04, 1.6e+00, 2.8e+00 | 1/3 | nonfinite_lbfgs, ok |
| 7.5 | PINN | 4× budget | 9.4e-07, 3.0e-06, 6.0e-06 | 3/3 | ok |
| 7.5 | PINN | lr decay 1e-3→1e-5 | 2.8e+00, 5.6e-01, 3.0e+00 | 0/3 | nonfinite_lbfgs, ok |
| 7.5 | PINN | μ-continuation (3000 it.) | 2.3e-06, 2.2e-06, 1.7e-05 | 3/3 | ok |
| 7.5 | VPINN K=60 | non-local (reference) | 5.4e-06, 1.1e-05, 2.4e-06 | 3/3 | ok |
| 7.5 | VPINN K=60 | baseline | 7.1e-05, 2.4e-01, 2.4e-01 | 1/3 | ok |
| 7.5 | VPINN K=60 | 4× budget | 3.7e-06, 2.4e-01, 4.7e-06 | 2/3 | ok |
| 7.5 | VPINN K=60 | lr decay 1e-3→1e-5 | 3.1e-05, 2.4e-01, 2.4e-01 | 1/3 | ok |
| 7.5 | VPINN K=60 | μ-continuation (3000 it.) | 3.6e-06, 1.3e-06, 3.9e-06 | 3/3 | ok |
| 7.9 | PINN | non-local (reference) | 3.2e-03, 1.8e-02, 2.8e-03 | 0/3 | ok |
| 7.9 | PINN | baseline | 1.7e+01, 1.7e+01, 1.5e+01 | 0/3 | ok |
| 7.9 | PINN | 4× budget | 5.0e-06, 1.7e+01, 8.5e-05 | 2/3 | ok |
| 7.9 | PINN | lr decay 1e-3→1e-5 | 8.4e-05, 1.7e+01, 1.7e+01 | 1/3 | ok |
| 7.9 | PINN | μ-continuation (3000 it.) | 5.7e-05, 6.5e-05, 3.0e-04 | 3/3 | ok |
| 7.9 | VPINN K=60 | non-local (reference) | 6.9e-05, 4.9e-05, 4.2e-05 | 3/3 | ok |
| 7.9 | VPINN K=60 | baseline | 1.3e+00, 1.3e+00, 1.3e+00 | 0/3 | ok |
| 7.9 | VPINN K=60 | 4× budget | 1.3e+00, 1.3e+00, 1.3e+00 | 0/3 | ok |
| 7.9 | VPINN K=60 | lr decay 1e-3→1e-5 | 1.3e+00, 1.3e+00, 1.3e+00 | 0/3 | nonfinite_lbfgs, ok |
| 7.9 | VPINN K=60 | μ-continuation (3000 it.) | 3.8e-05, 7.4e-06, 4.8e-06 | 3/3 | ok |

## R2 — spurious VPINN solutions (training residual < 1e-06 and ρ_weak > 0.01); count over seeds, median L2 for λ < 8

| Variant | λ = 7.5 | λ = 7.9 | λ = 8 | λ = 8.3 | λ = 8.6 |
|---|---|---|---|---|---|
| K=10 | **5/5**, L2 5e-03 | **5/5**, L2 4e-01 | **5/5** | **5/5** | **5/5** |
| K=20 | 0/5, L2 4e-05 | **5/5**, L2 1e-02 | **5/5** | **4/5** | **5/5** |
| K=30 | 0/5, L2 6e-06 | 0/5, L2 8e-04 | 0/5 | 0/5 | 0/5 |
| K=40 | 0/5, L2 2e-06 | 0/5, L2 1e-04 | 0/5 | 0/5 | 0/5 |
| K=20 normalised | 0/3, L2 2e-04 | 0/3, L2 1e-02 | **2/3** | **2/3** | **3/3** |
| K=60 normalised | 0/3, L2 9e-06 | 0/3, L2 4e-04 | 0/3 | 0/3 | 0/3 |
| K=20 4× budget | — | — | **3/3** | **3/3** | **2/3** |

## R3 — PINN collocation points (L2, 3 seeds)

| Variant | λ = 4 | λ = 7 | λ = 7.5 | λ = 7.9 |
|---|---|---|---|---|
| PINN, Gauss 128 (default) | 4.77e-06 ± 2.9e-06 | 5.85e-06 ± 2.1e-06 | 1.03e-05 ± 4.2e-06 | 8.02e-03 ± 7.1e-03 |
| PINN, uniform, 128 | 1.51e-06 ± 1.5e-06 | 4.37e-06 ± 4.4e-06 | 1.55e-04 ± 5.4e-05 | 1.96e-01 ± 4.5e-02 |
| PINN, random, 128 | 1.11e-05 ± 1.3e-05 | 4.41e-05 ± 2.6e-05 | 4.83e-04 ± 2.9e-04 | 5.25e-01 ± 4.3e-01 |
| PINN, uniform, 512 | 1.26e-06 ± 1.2e-06 | 3.69e-06 ± 3.8e-06 | 1.17e-05 ± 4.9e-06 | 1.32e-02 ± 1.6e-02 |
| VPINN K=60 | 2.29e-06 ± 1.7e-06 | 2.58e-06 ± 1.1e-06 | 6.22e-06 ± 3.5e-06 | 5.34e-05 ± 1.1e-05 |

## R4 — 5 seeds at λ = 7.5 and 7.9

| λ | Method | L2 (5 seeds) mean ± std | median | min – max | H1 mean |
|---|---|---|---|---|---|
| 7.5 | PINN | 1.05e-05 ± 5.8e-06 | 9.6e-06 | 3.2e-06 – 1.8e-05 | 7.6e-05 |
| 7.5 | VPINN K=20 | 6.14e-05 ± 4.7e-05 | 4.1e-05 | 1.1e-05 – 1.4e-04 | 8.5e-03 |
| 7.5 | VPINN K=60 | 5.98e-06 ± 3.9e-06 | 5.4e-06 | 1.2e-06 – 1.1e-05 | 1.0e-04 |
| 7.9 | PINN | 1.42e-02 ± 1.6e-02 | 3.2e-03 | 2.5e-03 – 4.4e-02 | 7.2e-02 |
| 7.9 | VPINN K=20 | 1.07e-02 ± 3.8e-03 | 1.0e-02 | 4.9e-03 – 1.7e-02 | 9.2e-01 |
| 7.9 | VPINN K=60 | 8.59e-05 ± 7.5e-05 | 4.9e-05 | 3.4e-05 – 2.3e-04 | 5.3e-04 |

## R5 — criterion B from 5 points λ ∈ {8.1, …, 8.5}, one estimate per seed

| Configuration | detector | seeds | λ* per seed | mean ± std | max deviation from linear | accepted fits (dev. < 5%) mean |
|---|---|---|---|---|---|---|
| PINN | ρ_weak | 5 | 7.949, 7.945, 7.949, 7.950, 7.949 | 7.948 ± 0.002 | 1.1% | 7.948 (5/5) |
| PINN | training residual | 5 | 8.073, 8.070, 8.073, 8.073, 8.073 | 8.072 ± 0.001 | 4.2% | 8.072 (5/5) |
| PINN (64×4) | ρ_weak | 2 | 7.949, 7.950 | 7.949 ± 0.000 | 1.1% | 7.949 (2/2) |
| PINN (64×4) | training residual | 2 | 8.073, 8.073 | 8.073 ± 0.000 | 3.8% | 8.073 (2/2) |
| VPINN K=40 | ρ_weak | 2 | 7.974, 2.381 | 5.177 ± 2.797 | 32.4% | 7.974 (1/2) |
| VPINN K=40 | training residual | 2 | 7.998, 8.015 | 8.007 ± 0.008 | 1.9% | 8.007 (2/2) |
| VPINN K=60 | ρ_weak | 5 | 7.992, 7.992, 7.992, 7.992, 7.992 | 7.992 ± 0.000 | 0.3% | 7.992 (5/5) |
| VPINN K=60 | training residual | 5 | 7.998, 7.998, 7.998, 7.998, 7.998 | 7.998 ± 0.000 | 0.2% | 7.998 (5/5) |
| VPINN K=60 (64×4) | ρ_weak | 2 | 7.992, 7.992 | 7.992 ± 0.000 | 0.3% | 7.992 (2/2) |
| VPINN K=60 (64×4) | training residual | 2 | 7.998, 7.998 | 7.998 ± 0.000 | 0.2% | 7.998 (2/2) |
| VPINN K=60 normalised | ρ_weak | 2 | 8.006, 8.006 | 8.006 ± 0.000 | 0.2% | 8.006 (2/2) |
| VPINN K=60 normalised | training residual | 2 | 8.006, 8.006 | 8.006 ± 0.000 | 0.2% | 8.006 (2/2) |
| VPINN K=120 | ρ_weak | 2 | 7.992, 7.992 | 7.992 ± 0.000 | 0.3% | 7.992 (2/2) |
| VPINN K=120 | training residual | 2 | 7.998, 7.998 | 7.998 ± 0.000 | 0.2% | 7.998 (2/2) |
