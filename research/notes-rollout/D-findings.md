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

## [6] gurbuzbalaban2021heavy — How Gaussian data can produce heavy-tailed SGD iterates
- Checked against: ICML 2021 PMLR version (v139, 12 pp.) and its 16-page supplement (the "publisher" copies the notes'
  source line cites): Eq. (3.5)–(3.10), Theorems 2, 3, 4, 6, 8, Proposition 5, Corollaries 7, 9, 11, Section 4 experiments
  (Figure 1 synthetic, Figure 2 FCN, Figure 3 VGG11, 10K iterations, average of last 1K, α > 2 truncated to 2), Section 5
  (saddle-escape time listed as future work), supplement C.1–C.6, D.
- **Fix (1):** heading of §3. Original: "3. Worked example: stable distribution, infinite variance". Fixed: "3. Worked
  example: **stationary** distribution, infinite variance". The example (and Theorem 2) give a *stationary* law with a power-law
  tail; the paper never claims the stationary law is α-stable (only centred sums are, Corollary 11), and the notes' own §2 warns
  "A power-law stationary law need not itself be an exactly α-stable distribution."
- Everything else verified: $h(2) = 1 - 2\eta\sigma^2 + \eta^2\sigma^4(d+b+1)/b$ equals Corollary 9's rate $1-2\eta\sigma^2(1-\eta/\eta_{\rm crit})$;
  $\eta_{\rm crit} = 2b/(\sigma^2(d+b+1))$ (Prop. 5); Theorem 8 for $1 \le p < \alpha$ (with α > 1); Theorem 3's fixed-direction
  identities vs the operator-norm bound (3.10). The notes' "several printed auxiliary formulas contain slips" is not checked
  line by line; one visible example: below Theorem 4 the paper says the non-Gaussian $\hat\alpha$ is "strictly increasing in
  stepsize η", which contradicts Theorem 4 (decreasing).
- Scalar example: $4\varphi(1) = 0.967883$, $E M^2 = 2$, $E x_k^2 = 2^k - 1$ (the notes' "2k - 1" in the text layer is $2^k-1$
  in the rendered page), $\eta_{\rm crit} = 2/3$; the script also finds $\alpha \in (1, 2)$.
- PDF-generation artefacts fixed silently: missing spaces in the source line ("PDF1-9", "Theorems2-4/8", …).
- Primer: (1) least-squares SGD is a random affine recursion; (2) typical vs average product growth, two-point M ∈ {0.2, 1.6}
  (E log ≈ −0.57, h(1) = 0.9, h(2) = 1.3); (3) tail index ↔ finite moments via the tail integral; (4) Kesten heuristic
  $\mathbb E|M|^\alpha = 1$ (labelled heuristic), α ≈ 1.35 with a simulated tail ratio ≈ 22 ≈ 10^1.35.

## [7] azizian2025global — Why a rare detour can dominate global optimization time
- Checked against: the ICML 2025 PMLR version (63 pp., from the PMLR v267 repository; the arXiv copy in data/papers.json has
  the same results): Assumptions 1–3, Eq. (12)–(23), Theorem 1, Theorem 2, Fig. 2 caption (500 runs, σ = 50, δ = 1e-2,
  modified three-hump camel), Definition 2 (B.14), Lemma D.7 / (D.42)–(D.46), Proposition D.1 (Appendix D.6), Appendix F
  (Styblinski–Tang, Himmelblau).
- No discrepancies found. The notes' tolerance "a" is the paper's ε in Theorem 1 (renamed by the notes on purpose, and the
  notes say so). The proof flag is right for the ICML copy too: (D.44) $M_n = \sum_{k\le n}(1-\mu\eta)^{n-k}\langle u_k, p_k-y_k\rangle$
  gives $E[M_n \mid \mathcal F_{n-1}] = (1-\mu\eta)M_{n-1}$, yet the text calls it a martingale before Doob (D.46). We keep the
  notes' wording ("does not certify a repaired proof").
- Three-state example re-derived in the script: forests 7, 10, 11; prunings 3, 5, 8; E = 4; closure (B→G becomes 7) leaves 4;
  $T_A$ closed form = linear solve; detour and residence scales $e^{-1/\eta}$, $e^{5/\eta}$. Gaussian action identities checked.
- Source line: artefact spaces fixed ("PDF1-9" → "PDF 1-9"); citation + arXiv link appended (no stable PMLR URL found for
  the v267 PDF at the time of writing; the repository copy was used for checking).
- Primer: (1) geometric waiting time, mean $1/p = e^{c/\eta}$; (2) large deviations with a coin (rate 0.131; −log P/n = 0.151,
  0.138, 0.133); (3) deep traps cost $\exp(2\Delta f/(\sigma^2\eta))$ (uses the notes' §7 Gaussian action); (4) a mean ruled by
  rare long runs (10 vs 10,010).

## [8] tripuraneni2018stochastic — Cubic regularization when both derivatives are noisy
- Checked against: the NeurIPS 2018 proceedings PDF (10 pp.) and supplement (15 pp., printed 11–25), which the notes follow:
  Assumption 2, Definition 1, Algorithm 1, Condition 1, **Algorithm 2 = Cubic-Subsolver, Algorithm 3 = Cubic-Finalsolver**
  (the arXiv copy in data/papers.json swaps them: Alg. 2 Finalsolver, Alg. 3 Subsolver — the notes are right for NeurIPS),
  Theorem 1, Remark 1, Lemma 1, Corollary 1, supplement Lemma 2 (errors $c_1\varepsilon$, $c_2\sqrt{\rho\varepsilon}$), B.5,
  D.1–D.2 (10 inner iterations, batch 100 / 10, 55,000/5,000 split, 2,000,000-call budget), Figures 1–3, Kohler & Lucchi remark.
- No discrepancies found. Cosine example fully re-derived (r = 1.811043, m = −0.504055, q = −0.649939, cos r − 1 = −1.237943,
  bound 0.182104 vs 0.145883; the perturbation family is unbiased and can produce g = 0.01, B = −0.9).
- Source line: artefact spaces fixed; citation + NeurIPS PDF link appended.
- Primer: (1) with exact derivatives the ρ/6 cubic model is an upper bound (cos example, s = ±2); (2) Hessian-vector
  products from gradients (x²y + y³ example, Hv = (4, 2)); (3) averaging and the two accuracy scales (n₁ ≈ 10⁴, n₂ ≈ 100).

## [9] carmon2018analysis — When a small Krylov space can solve a nonconvex model
- Checked against: the NeurIPS 2018 proceedings PDF (main pp. 1–9) and its 12-page supplement (printed 12–23), which the notes
  cite: optimality conditions (3), Theorem 1 (4)–(5), Corollaries 2–4 (13)–(18), Theorem 5 (19)–(21) and the resisting-oracle
  remark (deterministic methods, d > 2t), Section 5 (5,000 instances, d = 10⁶, three condition numbers; hard-case comparison),
  supplement E (diagonal instances justified by rotational invariance, unit initial gap, hard case b₁ = 0, eigen-gap γ figure).
- No discrepancies found. Note on wording only: Theorem 1's sublinear bound is $(\lambda_{\max}-\lambda_{\min})\|s^*\|^2/(t-\tfrac12)^2$
  times $[4 + \tfrac18\mathbb 1\{\lambda_{\min}<0\}\log^2(4\|b\|^2/(u_{\min}^\top b)^2)]$; the notes' "proportional to …/t², multiplied
  in the nonconvex case by a logarithmic factor" is a correct scale description, left as is.
- Hard-case example re-derived (A = diag(−2, 1), b = (0, 1), R = 1: Krylov value −1/2, global −7/6 at (±2√2/3, −1/3), gap 2/3;
  brute-force disk minimum agrees).
- Source line: artefact spaces fixed; citation + NeurIPS PDF link appended.
- Primer: (1) trust-region optimality conditions with a 1-D example (λ* = 1.5); (2) Krylov space / Lanczos, Vandermonde example
  (det 2); (3) the √κ rate with the paper's constants (36e⁻⁴ ≈ 0.66, 36e⁻¹² ≈ 2.2e-4 at κ = 100); (4) a random vector's
  component, $|u^\top v|^2 \ge \delta^2/d$ w.p. ≥ 1 − δ (Corollary 2's proof), log factor ≈ 98 for d = 10⁶, δ = 0.1.

## [10] carmon2019gradient — Gradient descent finds the cubic-regularized nonconvex Newton step
- Checked against: arXiv 1612.00547 (= SIAM J. Optim. 29(3), 2019): model (1), bounds (7a)–(8), Eq. (9), Assumptions A–B,
  Lemma 2.3, Proposition 2.5, Theorems 3.1–3.2 (rates O(1/ε) → O(1/√ε) gap-dependent → O(log 1/ε); τ_grow = 0 for convex
  problems; perturbed b̃ = b + σq), Section 5 (line search), Section 6 / Algorithm 1 (outer method).
- **Fix (1), numbering only:** "Wang et al. (2020), Ref 72" → "Ref 77". The notes use the old reference numbering; in the
  current survey [72] is Azizian et al. 2025 and Wang et al. 2020 (CR with momentum) is [77], so the old number would send a
  reader to the wrong paper on the Atlas. (The notes' "Ref 71" header for this paper is title-block metadata and is dropped.)
- Otherwise no discrepancies. The notes' 1-D example (η = 0.25) is labelled pedagogical; it is outside Assumption A
  (bound 1/(4(β + ρR)) = 0.125 for β = ρ = 1, b = 0), which the notes do not claim; GD still converges to +1 there (checked).
  The primer says this explicitly.
- Layout: the notes' blue sub-headings → `###`; "Important precision" → warn; "What to read" and "Papers Master note" → plain.
  The "Survey tag" line is title-block metadata (dropped); arXiv link appended to the source line.
- Primer: (1) the global-minimizer certificate $(A+\rho\|s^*\|I)s^*=-b$, $A+\rho\|s^*\|I\succeq0$, 2-D example
  (λ* ≈ 1.092, s* ≈ (−1.090, −0.065), q ≈ −0.278; the other two critical points fail the shift test); (2) Assumption A step
  (R ≈ 2.106, η ≤ 0.061, GD reaches s*); (3) the hard case b = (0, 0.2): GD stops at (0, −0.095), q ≈ −0.0097 vs −0.173.
