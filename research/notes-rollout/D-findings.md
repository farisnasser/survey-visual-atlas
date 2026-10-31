# Session D — in-page notes findings

## Shared changes needed
None so far.

Conventions used on this branch (for the coordinator's consistency pass):
- Notes without a one-minute summary: the minute box is built from the notes' own sentences, with labels taken from
  the notes' own section/box names (Problem / Principal result / Scope, …).
- The notes' closing "Original source: … Verification: …" sentence moves verbatim into the `>` source line; the
  rest of the "Our notes" paragraph stays in its section.
- Notes headers say "Ref. [NN]" in the old numbering; the page header shows the survey number, so it is dropped with the title block.

## [1] antonakopoulos2022adagrad — AdaGrad avoids saddle points
- Checked against: ICML 2022 PMLR version (proceedings.mlr.press/v162/antonakopoulos22a.pdf), abstract and pp. 1–9
  (Sec. 2.1 assumptions, Sec. 2.2 ADANORM/ADADIAG/ADAFULL, Prop. 1, Theorem 1, Prop. 3, Theorem 2).
- No discrepancies found. The notes' diagonal rule ($s_k = s_{k-1} + g_k\odot g_k$, $D_k = \mathrm{diag}((s_k+\delta\mathbf 1)^{-1/2})$,
  step $\eta$) is labelled a teaching representation; the paper's ADADIAG is $\Gamma_t = (\delta_0^2 I + \mathrm{diag}\sum_{s\le t}\nabla f\nabla f^\top)^{-1/2}$
  with no separate $\eta$. "Step-size stabilization arguments and a recursive representation" is the abstract's wording.
  The paper treats deterministic gradients only (p. 4), consistent with the notes.
- Minute box: the notes have none; built from §1 (Problem), the "Principal result" box and §5 (Scope).
- Primer: (1) strict saddle and "almost every start" = zero-volume stable set; (2) a preconditioner is a step per direction
  (the paper's three $\Gamma_t$), still a descent direction; (3) Sylvester's law of inertia, worked $DH$ example
  (eigenvalues 3, −1 → 1.693, −0.443); (4) a history-dependent step is a non-autonomous map, rescued because $\Gamma_t$
  converges (Prop. 1); ADANORM on $x^2/2$ settles at 0.691. All numbers in `scripts/check-math/antonakopoulos2022adagrad.py`.

## [2] cheridito2024gradient — Gradient descent escapes saddles in shallow ReLU training
- Checked against: arXiv 2208.02083v2 (= JOTA 2024), abstract, pp. 1–4 (Theorem 1.1, Theorem 1.2, Assumption 2.1,
  Theorem 2.2, Corollary 2.3, Lemma 2.4), Sec. 4–5 headings (Theorem 4.7 ReLU, Theorem 5.4 leaky ReLU).
- No discrepancies found. The notes' schematic model $c + \sum a_j\sigma(w_jx+b_j)$ is the paper's $\mathcal N_\theta$
  (Theorem 1.2); the paper's loss is $\int_\alpha^\beta(\cdot)^2\,dx$ (Lebesgue on an interval), the notes write a generic
  $d\mu$ and label the equations as illustrative. "Explicit threshold involving the limiting loss" matches the abstract and
  Theorem 1.2 (threshold $f'(\alpha)^2(\beta-\alpha)^3/(12(N-1)^4)$ on $\lim_k\mathcal L(\Theta_k)$, N even).
- Minute box: the notes have none; built from §1 (Problem), the "Principal result" box and the "Scope" box.
- Primer: (1) GD step linearized at a saddle, center-stable space; (2) orbits staying near lie on a Lipschitz graph
  (Theorem 2.2) of dimension ≤ d−1; (3) pulling back needs a map that does not crush volume: ReLU $\max(x,0)$
  counterexample, Lemma 2.4 (nonzero Jacobian a.e.); (4) worked one-neuron loss $\mathcal L(b)$: $C^2$ at $b=0$, only $C^1$
  at $b=-1$ where the neuron dies. Numbers checked in `scripts/check-math/cheridito2024gradient.py`.

## [3] du2017gradient — Gradient descent can take exponential time to escape saddles
- Checked against: arXiv 1705.10412 (= NeurIPS 2017), abstract, Sec. 3 ("un-natural" initialization), Theorem 4.1,
  Corollary 4.3 (Gaussian init), key observation Eq. (3) (p. 7), remark on PGD (Jin et al.).
- No discrepancies found. "Non-pathological", "fairly natural random initialization" and the PGD poly(d, 1/ε) clause are
  the paper's own wording (Theorem 4.1 part 2). The toy $\tfrac12x^2-\tfrac12y^2+\tfrac14y^4$ is labelled illustrative;
  its saddle, minima $(0,\pm1)$, GD map and $T_{\rm esc}$ formula are checked (δ = 1e-3: 67 steps vs 65.2; δ = 1e-9: 212 vs 210.2).
  "Jin et al. [8]" is correct in the survey numbering; survey Fig. 10 is the perturbation mechanism figure (checked in the survey PDF).
- Minute box: the notes have none; built from §1 (Main problem), the "Principal result" box and the "Critical distinction" box.
- Primer: (1) avoidance vs complexity, polynomial vs exponential ($2^{10}$, $2^{30}$); (2) multipliers $1+2\eta\gamma$, $1-2\eta L$
  on the paper's quadratic block; (3) the paper's Eq. (3) two-saddle chain with numbers (γ = L = 1, η = 0.1:
  $t_1 = 26$, $x_2 \approx 0.0015$, $t_2 = 62 \ge 52$); (4) a uniform-ball kick: $P(|u_1| < wr) \approx w\sqrt{2d/\pi}$
  (d = 100, w = 1e-6: 8e-6). All in `scripts/check-math/du2017gradient.py`.

## [4] daneshmand2018escaping — Escaping saddles with stochastic gradients
- Checked against: ICML 2018 PMLR version, pp. 1–8: Assumption 1 / Eq. (6), Assumption 2, Algorithms 1–2, Tables 1–3,
  Theorems 1–2 (rates $(\ell L)^4(\delta\gamma\epsilon)^{-2}\log$, $(\delta\gamma\epsilon/(L\ell^{5/2}))^{-4}\log^2$, target
  $(\epsilon,\sqrt\rho\,\epsilon^{2/5})$), proof sketch Sec. 3.2 (three regimes, uniformly random returned iterate, Eqs. 15–20),
  Assumption 3 / Eq. (23), Lemma 4 / Eq. (24), Figures 1–3 and their captions.
- No discrepancies found. Notes' "bounded input support $\|z\|\le1$": paper assumes the support of P lies in the unit sphere
  (footnote 5 says this is equivalent to z inside it). The "notice the square" remark is right: the paper's discussion says
  "proportional to the magnitude", the bound (24) is $(\lambda/c)^2$. Page pointers in §14 match the PMLR PDF.
- Minute box: verbatim from the notes. Box mapping: "Translate the equation into English", "Numerical scale comparison",
  "Why is a bigger step inserted periodically?" → key; "A nuance that matters for the survey", "What the guarantee is NOT" → warn;
  "Papers Master — 1–3-line note" → plain. The notes' "Full paper (PMLR)" link → the PMLR PDF.
- Primer: (1) second-order stationarity and what the exponent means (ε = 0.01: 0.1, 0.158, 0.316); (2) directional second
  moment as a quadratic form, isotropic $\mathbb E[uu^\top] = (r^2/d)I$; (3) negative curvature amplifies a kick
  (0.2 → 1.35 in 20 steps, loss −0.88 on the quadratic); (4) returning a random iterate turns the expected bad fraction into
  probability $1-\delta$ (the paper's Eqs. 15–20). Checked in `scripts/check-math/daneshmand2018escaping.py`.

## [5] fang2019sharp — SGD can escape faster when we analyze whole local runs
- Checked against: the COLT 2019 PMLR version (proceedings.mlr.press/v99/fang19a/fang19a.pdf, 43 pp.), whose numbering the
  notes follow: Assumptions 1–4, Definitions 1–3, Proposition 4 (dispersive examples), Algorithm 2 + parameters (8),
  Theorem 5, Appendix A (Propositions 7, 9, 10), Appendices C–F. The arXiv copy in `data/papers.json` (1902.00247, June 2019)
  numbers the same results Theorem 1 and Propositions 1, 2, 3, 4; the notes are right for the COLT copy, so the source line
  links the PMLR PDF.
- No discrepancies found. Verified: target $(\epsilon,\sqrt{\rho\epsilon})$; Definition 3 width $\sigma/(4\sqrt d)$, probability ≤ 1/4;
  Algorithm 2 returns the mean of the K₀ in-ball iterates; Prop. 9 decrease $B^2/(7\eta K_0)$; Theorem 5 bounds $18\rho B^2$,
  $-17\delta$, success $1-(T_1+1)p$; the two printed-formula slips the notes flag are real ($\log(0.7)$ in (8) vs $\log(0.7^{-1})$
  in Prop. 7 / Appendix C; Appendix E writes $\lambda_{\min}(\nabla f(\cdot))$ for the Hessian).
- Layout: the notes' plain-text math is set in KaTeX symbol for symbol; the italic SGD-step sentence stays an italic sentence.
  The notes' "Source:" line is moved after the Papers Master box and reading recommendation (end of page), with the paper
  citation and PMLR link appended.
- Primer: (1) oracle complexity and why per-step descent gives $\varepsilon^{-4}$ (one-step bound with noise); (2) noise sums
  grow like $\sqrt K$, averages shrink like $1/\sqrt K$ — scale check that $\eta\sigma\sqrt{K_0} \approx B$ (labelled a scale
  check, not the proof); (3) a ρ-Lipschitz Hessian bounds the quadratic model's gradient error by $\rho r^2/2$ (cos x example).
  Checked in `scripts/check-math/fang2019sharp.py`.
