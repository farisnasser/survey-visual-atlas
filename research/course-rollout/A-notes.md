# Session A notes: stops 4 curvature, 5 critical, 6 escape

## Shared changes needed
(none so far)

## Stop 4 `curvature`
Sections: 1 second derivative · 2 Hessian (symmetry, $u^\top H u$, quadratic model) · 3 eigenvalues as principal curvatures
(spectral theorem, $u^\top H u$ = weighted average, link to $L$-smoothness) · 4 GD per eigen-direction
($u_i \leftarrow (1-\eta\lambda_i)u_i$, stability $\eta < 2/\lambda_{\max}$, zig-zag) · 5 condition number ($\kappa\ln(1/\varepsilon)$,
best fixed step $2/(L+\mu)$) · 6 PSD / convex / strongly convex (local = global proof; PL from strong convexity + descent lemma ⇒
$(1-1/\kappa)^t$) · 7 Hessian-vector products (memory of a full Hessian, finite differences, exact HVP by autodiff, power/Lanczos).

Visuals (check: `scripts/check-math/course_curvature.py`):
- `bend` (2D): stop 1's loss $0.1x^2 + 0.5\sin 3x$ coloured by sign of $f''$, tangent vs 1-D quadratic model at a point; readout errors at $x+h$.
- `quad` (3D): $\tfrac12\Delta^\top H\Delta$, $H = R(\theta)\,\mathrm{diag}(\lambda_1,\lambda_2)R(\theta)^\top$, eigen-lines + direction $u$ with $u^\top H u$; heights ×0.1, fixed.
- `zigzag` (2D contour + chart): GD on a 30°-tilted quadratic ($\mu = 1$, $L = \kappa$); chart of $|u_i|$ vs $(1-\eta\lambda_i)^t$.
- `kappa` (2D chart): measured steps until both eigen-coordinates < $10^{-6}$ vs $\kappa$, for $\eta = 1/L$ and $2/(L+\mu)$.
- `hvp` (2D chart): forward / central finite-difference HVP error vs $h$ on stop 2's surface at (1.1, 0.9).

Paper claims (Atlas box):
- Dauphin 2014: GD step along eigenvector $i$ is $-\lambda_i\Delta v_i$; SFN rescales by $|H|^{-1}$ — findings `dauphin2014identifying` (Sec. 4, Eq. 4–5).
- Carmon & Duchi 2018: Krylov/Lanczos solution of the trust-region/cubic subproblem with rates in $\lambda_{\min}$, $\lambda_{\max}$ — findings `carmon2018analysis` (Theorem 1).
- NEON / NEON2 replace HVPs by gradient differences — findings `xu2018first` (Alg. 1), `allenzhu2018neon2` (reduction, Alg. 2).
- Tripuraneni 2018 (gradients + HVPs only), Arjevani 2020 (stochastic HVPs) — findings `tripuraneni2018stochastic`, `arjevani2020second`.
Standard facts stated without a paper: Schwarz symmetry, spectral theorem, convex ⇔ PSD Hessian, strong convexity ⇒ PL ⇒ linear rate
(derived on the page), exact HVP by reverse-mode autodiff at a constant multiple of a gradient's cost (stated, not cited: Pearlmutter 1994 is not in the Atlas).
Unverifiable: none.
