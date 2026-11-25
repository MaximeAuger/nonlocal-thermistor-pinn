# Classical baselines and mechanism

## C1 — finite differences + Newton (O(N) per step, Sherman-Morrison)

| λ | N | Newton iterations | wall time | L2 error |
|---|---|---|---|---|
| 1 | 200 | 4 (cold start) | 0.3 ms | 3.16e-06 |
| 1 | 800 | 4 (cold start) | 0.3 ms | 1.97e-07 |
| 1 | 3200 | 4 (cold start) | 0.7 ms | 1.23e-08 |
| 1 | 12800 | 4 (cold start) | 2.8 ms | 7.71e-10 |
| 4 | 200 | 5 (cold start) | 0.3 ms | 5.50e-05 |
| 4 | 800 | 5 (cold start) | 0.4 ms | 3.44e-06 |
| 4 | 3200 | 5 (cold start) | 0.8 ms | 2.15e-07 |
| 4 | 12800 | 5 (cold start) | 3.4 ms | 1.34e-08 |
| 7 | 200 | 6 (cold start) | 0.3 ms | 2.86e-03 |
| 7 | 800 | 6 (cold start) | 0.4 ms | 1.80e-04 |
| 7 | 3200 | 6 (cold start) | 1.0 ms | 1.13e-05 |
| 7 | 12800 | 8 (cold start) | 5.3 ms | 7.03e-07 |
| 7.9 | 200 | 8 (cold start) | 0.4 ms | 2.92e-01 |
| 7.9 | 800 | 9 (cold start) | 0.6 ms | 2.96e-02 |
| 7.9 | 3200 | 9 (cold start) | 1.4 ms | 1.95e-03 |
| 7.9 | 12800 | 19 (cold start) | 12.0 ms | 1.22e-04 |
| 7.99 | 200 | 9 (cold start) | 0.5 ms | 1.93e+00 |
| 7.99 | 800 | 11 (cold start) | 0.8 ms | 8.64e-01 |
| 7.99 | 3200 | 14 (cold start) | 2.3 ms | 1.74e-01 |
| 7.99 | 12800 | 32 (cold start) | 20.7 ms | 1.42e-02 |

Neural networks for comparison (cached runs):

| λ | network | setting | mean L2 | mean time |
|---|---|---|---|---|
| 7 | PINN | default | 5.85e-06 | 35 s |
| 7 | VPINN K=60 | default | 2.58e-06 | 26 s |
| 7.9 | PINN | default | 8.02e-03 | 37 s |
| 7.9 | VPINN K=60 | default | 5.34e-05 | 25 s |
| 7.99 | PINN | 4× budget, n_quad 256 | 1.01e+00 | 171 s |
| 7.99 | PINN | default | 1.59e+00 | 33 s |
| 7.99 | VPINN K=60 | 4× budget, n_quad 256 | 1.02e-02 | 137 s |
| 7.99 | VPINN K=60 | default | 4.88e-01 | 21 s |

Beyond the threshold (N = 800): λ = 8: cold Newton converged, continuation converged; λ = 8.5: cold Newton converged, continuation converged; λ = 9: cold Newton converged, continuation converged

## C2 — amplitude continuation (total 26 ms)

| m = u(1/2) | λ (N = 4000) | λ (Richardson) | exact 8(1 − e^{−m}) | error (Richardson) |
|---|---|---|---|---|
| 1 | 5.056966 | 5.05696447 | 5.05696447 | 1.6e-13 |
| 2 | 6.917326 | 6.91731773 | 6.91731773 | 1.1e-12 |
| 4 | 7.853567 | 7.85347489 | 7.85347489 | 1.4e-10 |
| 6 | 7.980946 | 7.98016999 | 7.98016998 | 1.0e-08 |
| 8 | 8.003299 | 7.99731693 | 7.99731630 | 6.4e-07 |
| 10 | 8.044396 | 7.99967212 | 7.99963680 | 3.5e-05 |
| 12 | 8.326456 | 8.00166884 | 7.99995085 | 1.7e-03 |

## C3 — square Galerkin least squares (no network): residual floor

| λ | Galerkin K = M = 30 | Galerkin K = M = 60 | VPINN K = 60 (training floor, median 2 seeds) | VPINN K = 20 |
|---|---|---|---|---|
| 7.9 | 4.889e-16 | 7.153e-16 | 4.907e-05 | 4.315e-10 |
| 7.95 | 3.523e-16 | 8.917e-16 | 3.889e-05 | 1.415e-09 |
| 8 | 4.586e-16 | 8.091e-16 | 2.285e-03 | 8.758e-11 |
| 8.05 | 9.722e-03 | 9.726e-03 | 9.809e-03 | 9.338e-11 |
| 8.1 | 1.951e-02 | 1.951e-02 | 1.951e-02 | 3.854e-10 |
| 8.2 | 3.905e-02 | 3.905e-02 | 3.905e-02 | 4.088e-04 |
| 8.3 | 5.849e-02 | 5.849e-02 | 5.849e-02 | 2.999e-10 |
| 8.4 | 7.777e-02 | 7.777e-02 | 7.777e-02 | 3.543e-10 |
| 8.5 | 9.688e-02 | 9.688e-02 | 9.688e-02 | 2.169e-10 |
| 8.6 | 1.158e-01 | 1.158e-01 | 1.158e-01 | 3.487e-10 |
| 8.7 | 1.346e-01 | 1.346e-01 | 1.346e-01 | 6.099e-04 |
| 8.8 | 1.531e-01 | 1.531e-01 | 1.531e-01 | 1.531e-01 |
| 8.9 | 1.715e-01 | 1.715e-01 | 1.715e-01 | 8.575e-02 |
| 9 | 1.897e-01 | 1.897e-01 | 1.897e-01 | 1.897e-01 |

Max relative difference between the VPINN K = 60 floor and the Galerkin K = 60 floor for λ ∈ [8.1, 9.0]: 0.01 %.

Fine scan near the threshold (K = M = 60):

| λ | 7.95 | 7.96 | 7.97 | 7.98 | 7.99 | 8 | 8.01 | 8.02 | 8.03 | 8.04 | 8.05 | 8.06 | 8.07 | 8.08 | 8.09 | 8.1 | 8.11 | 8.12 | 8.13 | 8.14 | 8.15 | 8.16 | 8.17 | 8.18 | 8.19 | 8.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Galerkin K = 60 floor | 8.9e-16 | 8.7e-16 | 8.5e-16 | 9.9e-16 | 7.8e-16 | 1.0e-15 | 1.9e-03 | 3.9e-03 | 5.8e-03 | 7.8e-03 | 9.7e-03 | 1.2e-02 | 1.4e-02 | 1.6e-02 | 1.8e-02 | 2.0e-02 | 2.1e-02 | 2.3e-02 | 2.5e-02 | 2.7e-02 | 2.9e-02 | 3.1e-02 | 3.3e-02 | 3.5e-02 | 3.7e-02 | 3.9e-02 |

Linear fit of the Galerkin K = 60 floor on [8.1, 8.5]: zero at λ = 7.9983 (max deviation 0.18 %).

## C3 — existence threshold of the K-mode Galerkin problem (resolution 0.01)

| K | discrete existence threshold λ*_K |
|---|---|
| 10 | > 20 |
| 15 | > 20 |
| 20 | > 20 |
| 25 | 8.05 |
| 30 | 8.05 |
| 40 | 8.05 |
| 60 | 8 |

Nature of the discrete minimisers:

| K | λ | Galerkin residual | ρ_weak (150 modes) | max u | energy in mode 1 | energy k > 20 |
|---|---|---|---|---|---|---|
| 20 | 8.3 | 3.2e-16 | 1.09e-01 | 8.86 | 0.0% | 1e+02% |
| 20 | 8.6 | 2.9e-16 | 1.63e-01 | 9.53 | 0.0% | 1e+02% |
| 60 | 8.3 | 5.8e-02 | 6.01e-02 | 3.75 | 88.8% | 5.7e-07% |

## C4 — over-parameterised polynomial ansatz (M > K)

| K | M | λ | relative weak residual |
|---|---|---|---|
| 20 | 80 | 8.3 | 1.1e-15 |
| 20 | 80 | 9 | 1.8e-15 |
| 60 | 80 | 8.3 | 6.8e-15 |
| 60 | 80 | 9 | 9.4e-15 |

## C5 — does the finite-difference discretisation see non-existence?

Fixed-resolution solutions beyond the threshold:

| N | λ | Newton (cold start) | max u of the discrete solution |
|---|---|---|---|
| 200 | 8 | converged, 9 it. | 4.66 |
| 200 | 9 | converged, 8 it. | 7.20 |
| 800 | 8 | converged, 12 it. | 6.01 |
| 800 | 9 | converged, 9 it. | 9.94 |
| 3200 | 8 | converged, 15 it. | 7.38 |
| 3200 | 9 | converged, 18 it. | 12.71 |

Amplitude branch λ_N(m) without Richardson extrapolation:

| N | m = 6 | m = 8 | m = 10 | m = 12 | m = 14 | m = 16 | m = 20 |
|---|---|---|---|---|---|---|---|
| 800 | 8 | 8.145 | 9.057 | 14.8 | 50.28 | 292.9 | 1.504e+04 |
| 3200 | 7.981 | 8.007 | 8.069 | 8.505 | 11.36 | 28.75 | 957.8 |
| 12800 | 7.98 | 7.998 | 8.004 | 8.032 | 8.237 | 9.64 | 72.69 |
| exact | 7.98 | 7.997 | 8 | 8 | 8 | 8 | 8 |
