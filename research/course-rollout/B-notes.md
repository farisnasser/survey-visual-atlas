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
