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

## Stop 5 `critical`
Sections: 1 flat is not finished (critical point, second-derivative test table) · 2 the zoo (six origin-critical functions, index) ·
3 strict saddles and degenerate points (Lee Def. 2; NP-hardness warn box) · 4 saddles made by symmetry ($(1-uv)^2$, $f(-p)=f(p)$ ⇒
$\nabla f(0)=0$, chord between swapped minima, unit-swap symmetry, deep chain $(1-abc)^2$ with zero Hessian, plateaus) · 5 why saddles
outnumber minima (Dauphin's argument as stated, GOE Monte Carlo, warn box: what is proved vs observed) · 6 FOSP vs SOSP (Hessian-Lipschitz,
Jin's $\varepsilon$-SOSP, units argument for $\sqrt{\rho\varepsilon}$, worked example with $\rho = 9$).

Visuals (check: `scripts/check-math/course_critical.py`):
- `zoo` (3D): six functions with a critical point at 0 (two button rows: all λ ≠ 0 / some λ = 0), eigen-tubes, 12 GD balls from the unit
  circle (η = 0.1, 400 steps); caught counts 12/0/2/3/12/2 reproduced by the script.
- `sym` (3D): $(1-uv)^2$ vs the slice $c = b$ of $(1-abc)^2$; minima curves, chord between symmetric minima, GD from $(s, 0.6s)$;
  steps to $f < 0.01$: 37/61/85 vs 85/833/8301 for $s = 0.1/0.01/0.001$. Note: on the slice the axis $v = 0$ is a line of critical
  points (for $u < 0$ they look like minima on the slice but are strict saddles of the full 3-weight chain); the page does not dwell on it.
- `random` (2D + chart): GOE 100×100 eigenvalue histogram shifted by $m$ with the semicircle; chart P(all N eigenvalues > 0) for N = 1..10,
  2,000 seeded samples each (mulberry32 ported exactly to Python; browser readouts equal the script: 0.496 / 0.0015 at m = 0, 0.755 / 0.235 / 0.0015 at m = 1).
- `sosp` (2D): $\tfrac12x^2 + \tfrac14y^4 - \tfrac12y^2$, regions $\lVert\nabla f\rVert \le \varepsilon$ (amber) and ε-SOSP (green), $\rho = 9$ on $|y| \le 1.5$.

Paper claims:
- Lee 2016 Def. 2 (strict saddle, zero eigenvalues allowed, includes maxima) — findings `lee2016gradient`.
- Murty & Kabadi 1987 via Ahmadi–Zhang: local minimality of a quartic (zero Hessian) NP-hard — findings `murty1987some`.
- Anandkumar & Ge 2016: third-order condition polynomial (Claim 1, Thms 8–9), fourth-order local min NP-hard (Thm 10) — findings `anandkumar2016efficient`.
- Zhang, Saxe & Latham 2026 Thm 1 (embedded fixed points), Thm 3 (identical units stay identical) — findings `zhang2026saddle`.
- Baldi & Hornik 1989; Kawaguchi 2016 Thm 2.3 and Cor. 2.4 (H = 1: every saddle has a negative eigenvalue; H ≥ 2: saddles without) — findings.
- Dauphin 2014 Sec. 2 (Bray & Dean monotone α–ε curve, shifted semicircle, most critical points saddles for large N; Fig. 1 MLPs on
  downsampled MNIST/CIFAR-10) — findings `dauphin2014identifying`.
- Jin 2017 Def. 6 (ε-SOSP), Cor. 4 (strict saddle ⇒ near local minima) — findings `jin2017escape`.
- Nesterov & Polyak 2006 Lemma 1 (cubic error bound of the quadratic model) — findings `nesterov2006cubic`.
- Ge 2017 / Bhojanapalli 2016 (strict saddle + no spurious minima for matrix sensing, completion, robust PCA) — findings.
Unverifiable: the $\sqrt{\rho\varepsilon}$ "units" explanation is ours (pedagogical), not a claim attributed to Jin et al.
