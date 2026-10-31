# Session B notes: stops 7 memory, 8 noise, 9 adaptive

## Shared changes needed
- None required. Two small wishes (worked around, not changed):
  - `src/lib/charts.js` niceTicks can print `-6e-17` for the 0 tick when `equal: true` widens a range (worked around in
    `course/memory/viz.js` lookahead with a `yfmt`).
  - Auditing 3D visuals in a hidden browser pane: screenshots of WebGL canvases are stale. I captured frames with
    `stage.snapshot()` POSTed to `/__thumb?id=zzsnapb` (written to `public/thumbs/zzsnapb.png`, deleted afterwards, never
    committed). A tiny `window.__course.snap(name)` hook in `learn.js` would make this standard.

## Stop 7 — memory (`course/memory`, check `scripts/check-math/course_memory.py`)
Sections: 1 Back to the zig-zag · 2 The heavy ball · 3 Momentum is an average of past gradients · 4 Nesterov's look-ahead ·
5 Memory in a narrow valley · 6 Why the square root: a 2×2 matrix · 7 Momentum overshoots · 8 Acceleration without
convexity · 9 Lookahead: averaging the trajectory · atlas · problem (full gradients too expensive → stop 8).

Visuals:
- `echo` (2D): g_t = 1 + a(−1)^t and the momentum sum Σβᵏg_{t−k}; side chart βᵏ bars. Shows gains 1/(1−β), 1/(1+β).
- `valley` (3D): f = ½(μx² + 10y²), κ ∈ {10, 100}; GD with 2/(L+μ) (violet), heavy ball (coral), Nesterov (white), same η, β;
  buttons for Polyak's and Nesterov's tunings; counts are exact runs to f < 1e-8 (GD 48/469, HB 20/71, NAG 31/100).
- `roots` (2D): roots of the per-curvature characteristic polynomial for 121 λ ∈ [μ, L] in the complex plane, unit circle,
  √β circle; method seg GD/HB/NAG; side chart |u_t| along λ = μ and L.
- `overshoot` (2D): heavy ball on stop 1's curve (window widened to show the 4th minimum at 3.5032), η = 0.05; from 2.4:
  β = 0 → 1.50, 0.9 → 1.50 (f up on 92/200 steps), 0.95 → −0.50 (global), 0.99 → −2.50.
- `lookahead` (2D contour): κ = 100 valley, η = 0.1, heavy ball β = 0.99 alone (1425 steps) vs Lookahead(k = 20, α = 0.5,
  momentum kept across syncs) 340 steps; per-step rate 0.9950 vs 0.9759 (exact cycle-map spectral radius on the two
  curvatures); with Polyak's tuning Lookahead is slower (0.980 vs 0.818).

Paper claims and sources:
- Polyak 1964 heavy ball and √κ rate on quadratics — findings polyak1964some (standard formulas; original not openly
  available); the rate is derived in the text from the 2×2 matrix and checked.
- Nesterov's method: convex O(1/t²), strongly convex (1 − 1/√κ)^t, optimal order among first-order methods — standard
  results cited to Nesterov 2004, "Chapter 2" without theorem numbers (findings: book not available line by line). The
  on-quadratic rate 1 − 1/√κ at η = 1/L is checked numerically. "Nesterov 1983" date is the standard attribution.
- Jin, Netrapalli, Jordan 2018: Theorem 3 Õ(ε^{-7/4}) ε-SOSP w.h.p. vs Õ(ε^{-2}) PGD; first Hessian-free single-loop method
  faster than GD; Lemma 4 Hamiltonian f + ‖v‖²/(2η) decreases — findings jin2018accelerated.
- Carmon, Duchi, Hinder, Sidford 2018: O(ε^{-7/4} log(1/ε)) gradient evaluations to an ε-stationary point, Lipschitz gradient
  and Hessian, gradient-only, plus ∇²f ⪰ −O(ε^{1/2})I — read from the arXiv abstract (1611.00756); not in findings.
- Zhang et al. 2019 Lookahead: Eq. (1), k = 20 / α = 0.5 deterministic quadratic (Sec. 3.2, Fig. 4: improves under-damped,
  slightly worse over-damped), Prop. 2 variance, Table 1 CIFAR-10 95.27 / 95.23 / 95.26 (the 95.26 column is averaging
  wrapped around SGD) — findings zhang2019lookahead + the paper PDF (Sec. 3.2 text, Table 1 caption, "we maintain the
  state of our inner optimizer" p. 3 and App. D.1).
- Sutskever et al. 2013: momentum with a slowly increasing schedule + well-designed initialization trains deep and recurrent
  nets; worse without or with poorly tuned momentum — PMLR abstract (no findings entry, no Atlas page content yet).
- Kingma & Ba 2015: Adam's first moment is an EMA — findings kingma2015adam.

Not verifiable / caveats: the heavy ball's lack of a global accelerated guarantee is phrased as "only guaranteed locally"
(no counterexample paper cited). Sutskever 2013 and Carmon 2018 have no findings entries; claims limited to their abstracts.

## Stop 8 — noise (`course/noise`, check `scripts/check-math/course_noise.py`)
Sections: 1 A full gradient costs a pass over the data · 2 One random example is an unbiased guess · 3 Stochastic gradient
descent · 4 The noise floor · 5 Shrinking steps: the Robbins–Monro conditions · 6 How many gradients? The ε⁻⁴ rate ·
7 Variance reduction · 8 Noise near saddles: help, and its limits · atlas · problem (one step size for all coordinates →
stop 9, using the line fit's curvatures 0.45 / 12.33, κ = 27.4).

The check script ports mulberry32 + Box–Muller from `src/lib/math.js` exactly, so every random readout on the page is
reproduced to the printed digits (cloud mean/spread, floor averages, schedule Monte Carlo, SVRG/SGD/GD, escape step).

Dataset (all visuals but `escape`): n = 50, x_i = 4i/49, y_i = 0.8x_i − 0.5 + 0.6 sin(2.7i + 1); w* = 0.800, b* = −0.488,
f* = 0.180, curvatures 0.450 / 12.33, σ*² = 4.81.

Visuals:
- `cloud` (2D contour, draggable): 400 minibatch steps (η = 0.08) around the full step at a point; B slider. Default
  (1.8, −1.2), B = 5: measured spread 15.02 vs exact σ²/B = 15.05.
- `floor` (3D): constant-step SGD (coral), decaying η/(1 + t/200) (violet), GD (white), 4000 steps from (1.8, −1.2);
  averages over the last 1000 steps vs the small-step formula ησ*²/(4B). Default η = 0.02, B = 1: 0.0314 vs 0.0241,
  decaying 7.9e-4, GD < 1e-12. Text: η = 0.005 → 0.0062 vs 0.0060.
- `schedules` (2D log-log): f = x²/2 with N(0, 1) gradient noise, x₀ = 2, 100 runs, constant / 1/(t + 2) / 1/(t + 2)²
  with the exact E x² recursion dashed. Closed forms checked: η/(2 − η), (x₀² + t)/(t + 1)², stall at 1.04.
- `vr` (2D): f − f* per epoch for SGD, SVRG (m = n), GD (1/L) + estimator-error scatter. 60 epochs at η = 0.03: SGD
  4.1e-2, GD 2.7e-3, SVRG < 1e-15 (strongly convex finite sum: linear convergence; stated in the text).
- `escape` (3D): f = x²/2 + y⁴/4 − y²/2, two-example finite sum f ± s·y (or ± s·x), start (1.5, 0) on the stable line,
  η = 0.1. Default s = 0.3, seed 31: escapes in 23 steps; along-x noise and GD never leave y = 0.

Paper claims and sources:
- Robbins & Monro 1951: Σa_n² < ∞ plus a condition implying Σa_n = ∞; Theorems 1–2; convergence in probability, a.s.
  later — findings robbins1951stochastic.
- Ghadimi & Lan 2013 Corollary 2.2 formula, random output x_R, first-order only — findings ghadimi2013stochastic. The
  N ~ 8LΔσ²/ε⁴ reading with D̃ = D_f is my algebra (checked in the script).
- Bottou, Curtis, Nocedal 2018 Theorem 4.8 (fixed step: floor ∝ ᾱ + 1/(Kᾱ) term) — findings bottou2018optimization.
- Arjevani et al. 2023 Theorem 1 Ω(ΔLσ²ε⁻⁴), Theorem 2 ε⁻³ under mean-squared smoothness — findings arjevani2023lower.
- Reddi et al. 2016 SVRG: Corollary 4, O(n + n^{2/3}/ε) for E‖∇f‖² ≤ ε; "faster than GD by n^{1/3}" — read in the arXiv
  PDF (1603.06160), not in findings.
- Fang et al. 2018 SPIDER: Theorems 1, 2, 3 (lower bound when n ≤ O(ε⁻⁴)), Lemma 1's mean-squared smoothness — findings.
- Wang et al. 2019 SpiderBoost: Theorem 1 O(n + √n ε⁻²) with step 1/(2L), online ε⁻³, first-order only — findings.
- Daneshmand et al. 2018 CNC: Assumption 1, Theorems 1–2 (ε, ε^{2/5})-SOSP, isotropic r²/d — findings.
- Fang, Lin, Zhang 2019: plain SGD with restart/average, dispersive noise, Õ(ε^{-3.5}) — findings fang2019sharp.
- Gürbüzbalaban et al. 2021: heavy tails; Theorems 2, 4; Proposition 5 η_crit = 2b/(σ²(d + b + 1)), α < 2 above it;
  2/3 for d = b = σ² = 1 — findings.
- Ge et al. 2019, Li 2019 named only as escape variants of VR methods — findings.

Not verifiable / caveats: "networks are mostly trained with SGD with momentum or adaptive methods; VR mainly a theoretical
benchmark" is a general observation, not a cited result. "Deep-learning schedules (step, cosine) do not satisfy RM
literally" is a general statement about common practice. New glossary term added: `noise-floor` (stop 8 block).

## Stop 9 — adaptive (`course/adaptive`, check `scripts/check-math/course_adaptive.py`)
Sections: 1 One step size, many curvatures · 2 Preconditioning: reshaping the landscape · 3 AdaGrad · 4 RMSProp · 5 Adam:
momentum plus RMSProp, with bias correction · 6 What is known about Adam's convergence · 7 Gradient clipping and
(L0, L1)-smoothness · 8 Beyond the diagonal · atlas · problem (which minimum, and does it matter → stop 10, opened with
SASSHA's reported sharpness table).

Visuals:
- `precond` (2D contour): rotated quadratic, κ ∈ {10, 100}, angle slider; GD, diagonal (Jacobi) preconditioner, Newton,
  each with its best fixed step. κ = 100: diagonal κ_eff = 1 / 75.49 / 100 at 0 / 30 / 45°; steps GD 506 / 525 / 528,
  diagonal 1 / 396 / 528, Newton 1.
- `bias` (2D): uncorrected Adam step / α = (1 − 0.9ᵗ)/√(1 − β₂ᵗ) vs corrected (= 1). Peaks 2.131 (t = 13), 6.569 (12),
  20.72 (12) for β₂ = 0.99 / 0.999 / 0.9999; first step 3.16α at 0.999 (matches findings kingma2015adam).
- `race` (3D): κ = 100 valley along an axis or rotated 45°, GD best step / AdaGrad η = 0.5 / Adam α = 0.05, 400 steps:
  aligned 8.2e-7 / 5.6e-123 / 1.2e-18, rotated 1.8e-6 / 2.1e-5 / 5.6e-5. Height drawn as √f.
- `adamfail` (2D): Taniguchi et al.'s Eq. 17 toy, one mulberry32 stream (seed 1), 400 000 steps, Adam / AMSGrad / ADOPT,
  k ∈ {10, 50}, β₂ ∈ {0.1 … 0.999}. k = 10: Adam +1 for β₂ ≤ 0.9, −0.992 / −0.994 for 0.99 / 0.999; AMSGrad −0.998;
  ADOPT −0.98 … −0.994. k = 50: Adam ≥ 0.19 for all β₂, AMSGrad −0.052, ADOPT ≤ −0.966.
- `clip` (3D): f = x²/2 + x⁴/4 + y²/2 from (x₀, 1): best fixed step 6 / 22 / 81 / 322 iterations vs clipped GD (η = 1,
  γ = 0.5) 4 / 6 / 10 / 18 for x₀ = 1 / 2 / 4 / 8 (same as findings zhang2020gradient).

Paper claims and sources:
- Duchi et al. 2011: diagonal AdaGrad, Corollary 1 (best diagonal preconditioner in hindsight up to a factor), Sec. 1.3
  sparse example O(log d √T) vs O(√(dT)); online convex — findings duchi2011adaptive.
- Antonakopoulos et al. 2022 Theorem 2 (deterministic AdaGrad norm/diag/full avoid strict saddles) — findings.
- Kingma & Ba 2015: Algorithm 1 defaults, bias correction, Sec. 2.1 step bound and scale invariance, Theorem 4.1 proof
  flawed — findings kingma2015adam.
- Reddi, Kale, Kumar 2018: Theorem 1 (online convex, non-zero average regret; the C-once-every-3-steps example on [−1, 1]),
  Theorem 2 (any β1 < √β2), Theorem 3 (stochastic), AMSGrad v̂ = max(v̂, v) — read in the arXiv PDF (1904.09237), not in
  findings.
- Li, Rakhlin, Jadbabaie 2023 Theorem 4.1 (ρ < 2 relaxed smoothness, bounded noise, w.h.p., O(ε⁻⁴), no β2 condition,
  momentum weight β = O(ε²)), VRAdam O(ε⁻³) — findings li2023convergence. "their β plays the role of 1 − β1" is my
  reading of the paper's parameterization (findings only says "β, λ = O(ε²)"); worth a glance by the coordinator.
- Taniguchi et al. 2024: ADOPT normalization by v_{t−1}, momentum after normalization, Theorem 4.1 for any β2 — findings;
  our run vs the paper's Figure 1 (β2 = 0.99 succeeds here) noted on the page.
- Staib et al. 2019 Theorem 4.4 / Corollary 4.2 (RMSProp, (τ, √(ρτ)), Õ(τ⁻⁵)) — findings.
- Zhang et al. 2020: Definition 1, Lemma 2, Theorems 3 and 4, Figure 1 LSTM motivation — findings.
- Gupta et al. 2018: Algorithm 1, Lemma 4, Theorem 7 (online convex O(√T) regret), experiments — findings.
- Shin et al. 2025 SASSHA: method summary, Theorem 4.4 convex, Table 1 reported numbers — findings shin2025sassha.
- Crawshaw et al. 2026: Props. 3.4–3.6 (Adam and Muon as steepest descent in norms), spectral LMO = −polar — findings.
- Defazio et al. 2024: Schedule-Free SGD/AdamW exist (atlas line only) — findings.

Not verifiable / caveats: RMSProp's attribution (Tieleman & Hinton 2012 lecture) and "AdaHessian and Sophia estimate the
Hessian diagonal" are standard facts not in findings (the AdaHessian / Sophia papers are not in the Atlas). "Clipping is
widely used for recurrent and transformer language models" and "Adam is the default for much of deep learning" are
general observations.
