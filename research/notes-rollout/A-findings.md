# Session A — in-page notes findings

## Shared changes needed
None so far.

Audit method used by this session: the in-app browser pane was hidden (its viewport reported 0 px), so each page was
loaded in a fixed 375 × 812 px iframe on `localhost:8571` and measured there (KaTeX errors, `__atlasErrors`, section
count, `scrollWidth − innerWidth`, elements past the right edge outside `.nt-eq`/`.nt-table`). Verbatim check: a
word-level diff (`difflib`) of the PDF text against `notes.md` with the primer, math and SVG removed; every remaining
difference is listed per paper (title block, box/fence keywords and math tokens only, unless a fix is logged).

## [1] murty1987some — Hardness in Quadratic and Nonlinear Programming
- Notes checked against: author scan public.websites.umich.edu/~murty/np.pdf, pp. 117–119 (abstract, Sec. 1, Sec. 2 Examples 1–2), read as images.
- Fixes to the notes: none. No discrepancies found. The notes make no theorem-numbered claim; every claim they do make
  (hardness of local-optimality / boundedness questions for indefinite QPs, worst-case scope) matches the abstract and
  Sec. 1 (p. 118). The "Verification limit" box describes how the notes were written and was kept verbatim; the paper
  itself was read for this page.
- Primary-source line: the notes' own "Source:" sentence (end of Sec. 4) moved into the `>` source line, verbatim.
- Primer (4 ideas): worst case is about a growing family (2^20 vs 2^60); the paper's Example 2 reduction of Subset Sum
  to a box QP, with d = (3, 5): d0 = 8 → min 0 at (1, 1), d0 = 4 → min 5/32 at (1, 3/16); copositivity vs eigenvalues
  ([[1,2],[2,1]] indefinite but copositive); the abstract's two NP-complete questions and the copositivity corollary.
  All numbers in `scripts/check-math/murty1987some.py`.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 4 sections, no overflow at 375 px.

## [2] blum1992training — Training a Tiny Network Can Be NP-Complete
- Notes checked against: author reprint people.csail.mit.edu/rivest/pubs/BR93.pdf (journal pp. 117–127): Sec. 4.1.2,
  Def. 1 and Fig. 1 (pp. 3–4: two hidden threshold nodes N1, N2 + output node N3), Theorem 2 (p. 5), Sec. 4.3.1–4.3.3
  (pp. 6–8), conclusions (p. 15–16: logistic units left open).
- Fixes to the notes: none. No discrepancies found. Remark (not changed): the notes write a threshold neuron as
  $\mathbf{1}\{w^\top x \ge b\}$ with outputs {0, 1}; the paper uses ±1 outputs and a strict inequality
  ($a_1x_1 + \dots + a_mx_m > a_0$). On a finite example set both conventions realise the same labelings, and the notes
  present it as their own definition, so it is not an error.
- Source line: the notes' "Sources:" paragraph, verbatim (DOI link parentheses URL-encoded so the link works).
- Primer (4 ideas): one threshold unit = one half-space (AND yes, XOR impossible — the w1 + w2 ≥ 2b > b contradiction);
  two hidden units compute XOR (h1 = 1{x1+x2 ≥ 1}, h2 = 1{x1+x2 ≥ 2}, ŷ = 1{h1 − h2 ≥ 1}); NP-completeness = short
  checkable certificate (O(n log n) bits per weight, the paper citing Raghavan) + reduction; the paper's Set-Splitting
  example with S1 = {s1, s3}, S2 = {s2}: a = (−1, 3, −1), b = (3, −1, 3). All checked in `blum1992training.py`.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 5 sections, no overflow at 375 px.

## [3] baldi1989neural — Neural Networks and Principal Component Analysis
- Notes checked against: Neural Networks 2:53–58 (open copy vision.jhu.edu/teaching/learning/deeplearning19/assets/Baldi_Hornik-89.pdf):
  abstract, Sec. 1 (AB = (AC)(C⁻¹B), p. 54), Sec. 2 main result and Facts 1–4 (pp. 54–55: unique local = global
  minimum at the top-p eigenvector projection, all other critical points saddles, rank-deficient ones included).
- Fixes to the notes: none. No discrepancies found. The notes' "nondegenerate critical configurations are saddles" is
  narrower than the paper (which also classifies rank-deficient critical points as saddles), but true.
- Silent artefact fixes: the PDF prints every section number twice ("1 1. Start with…", "3.1 3.1 The Hessian"); the page
  keeps one.
- Title: the notes' own title is the paper title ("Neural Networks and Principal Component Analysis"), used as-is.
  Box mapping follows the notes' colour legend: red "results" → theorem (key landscape conclusion), blue "why in ML" →
  key, amber "teaching clarification" → warn. The x → B → h → A → x̂ diagram is redrawn as SVG.
- Primer (4 ideas): spectral theorem on Σ = [[5,4],[4,5]] = 9u1u1ᵀ + 1u2u2ᵀ; uᵀΣu as a weighted average of eigenvalues
  (u = (1,0): 5); trace = total variance (10 = 9 + 1), the paper's eq. (10) error = total variance − kept eigenvalues;
  convex in each factor (Facts 1–2; (1 − 2b)² minimised at b = 1/2) but not jointly. All checked in `baldi1989neural.py`
  together with every number in the notes (gradient, H(0,0) eigenvalues ±2, the (3,2) projection, the trace identity,
  E(θ) = 1 + 8 sin²θ, AB = (AC)(C⁻¹B)).
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 8 sections, no overflow at 375 px.

## [4] dauphin2014identifying — Escaping Saddle Points
- Notes checked against: NeurIPS 2014 proceedings PDF (papers.nips.cc …/04192426585542c54b96ba14445be996-Paper.pdf):
  Sec. 2 and Fig. 1 (random Gaussian fields, index–error curve, downsampled MNIST/CIFAR-10), Sec. 4 and Fig. 2 (GD vs
  Newton near saddles), Sec. 5 Eq. (2)–(3), Sec. 6 Lemma 1, Eq. (4)–(5), Algorithm 1 (Lanczos/Krylov, damping),
  Sec. 7 (10 × 10 downsampling; seven-hidden-layer MNIST autoencoder: SGD stuck at MSE 1.0, SFN reaches 0.57 vs
  Hessian-free 0.69; Penn Treebank RNN), Figs. 1–4.
- Fixes to the notes: none. No discrepancies found. The experiment numbers (1.0, 0.57, 0.69, 10 × 10, seven layers)
  were checked against the paper's Sec. 7 text, not recomputed.
- Title: "Paper 3 — Escaping Saddle Points" → "Escaping Saddle Points" (the "Paper 3" numbering is title-block material).
- Primer (4 ideas): Newton step −H⁻¹g from the quadratic model (2(x−3)² from 0: g = −12, H = 4, step 3); functions of a
  symmetric matrix act on eigenvalues (|H| = 2I for [[0,2],[2,0]] vs elementwise |H| = H); the coin-flip cartoon
  2^−N (1/1024 at N = 10), labelled a cartoon, next to the paper's random-field statement (Sec. 2: high-error critical
  points exponentially likely to be saddles); linear model in an ellipse trust region → s ∝ −M⁻¹g (M = diag(1,4),
  g = (1,1): s ≈ (−0.894, −0.224)). All in `dauphin2014identifying.py`, with every number of the notes' worked example
  (GD (0, 1.01), Newton (0, 0), SFN (0, 2), f values −0.005 / −0.00510 / −0.020) and Lemma 1 on a random 3×3 Hessian.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 9 sections, no overflow at 375 px.

## [5] jin2017escape — Efficient Saddle Escape
- Notes checked against: ICML 2017 PMLR 70 PDF (proceedings.mlr.press/v70/jin17a/jin17a.pdf): Def. 6 (p. 4), Algorithm 2,
  A1, Theorem 3 (p. 5; ε ≤ ℓ²/ρ, probability 1 − δ, O(ℓ(f(x0) − f*)/ε² · log⁴(dℓΔf/(ε²δ)))), Sec. 3.1 A2 and
  Corollary 4 (pp. 5–6), Sec. 3.2 A3.a/b and Theorem 5 (p. 6: + (β/α) log(ζ/ε)), Sec. 4 Eq. (2), Lemmas 6–7 and
  Theorem 8 (p. 7), Sec. 5 Lemmas 9–11 and Figs. 1–2 (pp. 7–8, "thin pancake" / "narrow band").
- Fixes to the notes: none. No discrepancies found (every page, lemma, theorem and assumption number matches).
- Title: "Paper — Efficient Saddle Escape" → "Efficient Saddle Escape". The perturbation-ball figure is redrawn as SVG
  (stuck band and e₁ arrow in the negative-curvature red, kicks in optimizer coral). Primary-source line built from
  `data/papers.json` (the notes have only a title-block link).
- Primer (4 ideas): descent lemma and the 2ℓΔ/ε² count (20 000 for ℓ = Δ = 1, ε = 0.01); the √(ρε) curvature scale
  from balancing γ³/ρ² against εγ/ρ (labelled a scale argument; ρ = 1, ε = 1e−4 → 0.01); a thin slab holds a small
  share of a ball (w = 0.05: 0.064 of the disk, 0.075 of the 3-ball, Monte Carlo-checked); repetition gives log(1/δ)
  (p = 1/2, δ = 1e−6 → 20 kicks). All in `jin2017escape.py`, with the notes' example (Hessian diag(1, −1) and
  diag(1, −cos y), y1 ≈ 0.0240, ℓ = ρ = 1, bounded below).
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 11 sections, no overflow at 375 px.

## [6] jin2018accelerated — Why Accelerated Descent Escapes Saddles Faster
- Notes checked against: arXiv 1711.10456 (COLT 2018, PMLR 75): abstract (Õ(1/ε^{7/4}) vs Õ(1/ε²)), Sec. 1.1
  (Hamiltonian Eq. (1) E_t = f(x_t) + ‖v_t‖²/(2η), "improve or localize"), Algorithms 1–3 (AGD core
  y_t = x_t + (1 − θ)v_t, perturbation, Negative Curvature Exploitation), Eq. (3) parameters, Theorem 3.
- Fixes to the notes: none. No discrepancies found. The notes write the momentum parameter as α where the paper writes
  θ; the notes present the equations as a simplified core, so this was kept (the primer uses α and names θ once).
- Minute box: the notes' summary box has no "paper in one minute" heading (labels Problem / Method / Main result);
  the standard box title was used and the labels kept verbatim. The notes' "Source:" sentence (end of Sec. 5) moved
  into the `>` source line.
- Primer (4 ideas): κ vs √κ on a bowl (κ = 100, 10⁶ reduction: ≈ 1375 GD vs ≈ 131 AGD steps; the paper's Sec. 2);
  momentum speeds escape (ηγ = 1e−4: GD ≈ 69 000 steps for 1000× growth, AGD core with the paper's θ = ½√(ηγ) =
  0.005 grows ≈ 1.0079 per step, ≈ 940 steps; flagged as a one-direction simplification); an energy that falls while
  f rises (x²/2, η = 0.25, α = 0.05: f = 0.5, 0.281, 0.074, 0.0004, 0.027; energy decreasing for 60 steps); reading
  Õ(ε^{−7/4}) vs Õ(ε^{−2}) at ε = 1e−4 (10⁷ vs 10⁸). All in `jin2018accelerated.py`, plus the notes' u² − v² saddle.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 5 sections, no overflow at 375 px.

## [7] fang2018spider — SPIDER: Tracking Gradients with Less Noise
- Notes checked against: arXiv 1807.01695 v4 (NeurIPS 2018): abstract (O(min(n^{1/2}ε⁻², ε⁻³)) for an ε-FOSP;
  Õ(min(n^{1/2}ε⁻² + ε^{−2.5}, ε⁻³)) for an (ε, O(ε^{0.5}))-SOSP with SPIDER-SFO+), Sec. 2 estimator and Lemma 1,
  Algorithm 1 (refresh every q steps, same S₂ at x_k and x_{k−1}, normalized step), Algorithm 2 (SFO+), Theorems 1–3.
- Fixes to the notes: none. No discrepancies found.
- Minute box: no "paper in one minute" heading in the notes (labels Problem / Contribution / Results); standard box
  title used, labels verbatim. The notes' "Source:" sentence moved into the `>` line.
- Primer (4 ideas): minibatch variance σ²/b (4/16 = 0.25, Monte Carlo); Var(X − Y) and same-sample differences
  (f_i = c_i x²/2: variance Var(c)·Δx² = 1e−4 for Δx = 0.01); Lemma 1's linear error growth kL²ε₁²/S₂ (= 1e−3,
  std ≈ 0.032, with a Monte Carlo of the running estimate); counting gradient evaluations (the paper's quoted GD/SGD
  O(min(nε⁻², ε⁻⁴)); at ε = 0.01, n = 100: 1e5 vs 1e6; n = 1e6: 1e6 vs 1e10; crossover n = ε⁻²). All in
  `fang2018spider.py`, plus the notes' exact cancellation in the g_i = h + ξ_i toy model.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 5 sections, no overflow at 375 px.

## [8] arjevani2020second — What Can Second-Order Information Buy?
- Notes checked against: arXiv 2006.13476 (COLT 2020, PMLR 125): abstract (O(ε⁻³) stochastic gradients + HVPs, matching
  lower bound, no improvement for any p ≥ 2 even with Lipschitz first p derivatives), Fig. 1 (elbow: ε⁻⁴ → ε⁻³,
  noiseless ε⁻² → ε⁻¹), Sec. 1 (single-point vs two-point same-seed queries; integral identity for ∇F(x) − ∇F(x′);
  footnote 2: HVPs roughly as expensive as stochastic gradients, Pearlmutter 1994), (ε, γ)-SOSP upper/lower bounds.
- Fixes to the notes: none. No discrepancies found.
- Minute box: no "paper in one minute" heading in the notes (labels Problem / Method / Result); standard title used.
  "Meaning of a lower bound." (green) → key, "Survey caution." (amber) → warn; "Our takeaway." stays a paragraph as in
  the PDF; the "Source:" sentence moved into the `>` line.
- Primer (4 ideas): noise costs a power of ε (σ²/ε² samples per step → ε⁻⁴, heuristic labelled; the rate itself from
  the paper); the elbow of Fig. 1 (ε = 0.01: 1e8 vs 1e6); HVP as the directional derivative of the gradient
  (F = x²y at (1, 2), v = (1, 0): Hv = (4, 2), finite difference (4, 2.001)); gradient differences as integrals of
  HVPs (x³/3 from 1 to 1.2: 0.44 exact, 0.40 with K = 1, 0.43 with K = 4). All in `arjevani2020second.py`, plus an
  O(δ²) check of the notes' Taylor expansion of the gradient.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 5 sections, no overflow at 375 px.

## [9] du2019gradient — Why Wide Deep Networks Can Reach Zero Loss
- Notes checked against: ICML 2019 PMLR 97 PDF (proceedings.mlr.press/v97/du19c/du19c.pdf): Sec. 3.3 Eq. (1) loss
  ½Σ(f − y)², Eq. (3) ResNet layer x^(h) = x^(h−1) + (c_res/(H√m)) σ(W^(h)x^(h−1)), Sec. 4 (technique), Sec. 5
  Def. 5.1 and Theorem 5.1, Sec. 6 Theorem 6.1, Sec. 7 Theorem 7.1 (each with (1 − ηλ_min/2)^k), Sec. 8.
- Fixes to the notes: none. No discrepancies found (theorem and section numbers, the bound and the layer formula match).
- Title: "Paper 6 — Why Wide Deep Networks Can Reach Zero Loss" → without "Paper 6". Primary-source line built from
  `data/papers.json` (the notes have none).
- Primer (4 ideas): chain rule r ↦ (I − ηJJᵀ)r (linear model x1 = (1,0), x2 = (1,1): G = [[1,1],[1,2]], eigenvalues
  ≈ 2.618, 0.382); λ_min sets the slowest rate (η = 1/λ_max ≈ 0.382: factors 0 and ≈ 0.854); averages concentrate
  (1/√m; 100 coins → std 0.1, m = 1e4 → 0.01); wide nets move each weight O(1/√m) (labelled intuition; the paper's
  per-layer 1/√m scaling). All in `du2019gradient.py`, plus the notes' fixed-Gram example (r1..r3, losses 1.00000 /
  0.12500 / 0.03125 / 0.00781, λ_min = 0.5) and 2^10 = 1024 vs 1.1^10 ≈ 2.59.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 11 sections, no overflow at 375 px (tables stack).

## [10] allenzhu2019convergence — Why Very Wide ReLU Networks Can Be Trained
- Notes checked against: ICML 2019 PMLR 97 PDF (proceedings.mlr.press/v97/allen-zhu19a/allen-zhu19a.pdf): Sec. 2
  (model h_{i,0} = φ(Ax_i), h_{i,ℓ} = φ(W_ℓ h_{i,ℓ−1}), ŷ = Bh_{i,L}; A, B fixed), Sec. 3 Theorems 1–2,
  Theorem 3 (no critical point, Ω(δm/(dn²))), Theorem 4 (semi-smoothness: + O(nL²m/d)‖W′‖² + poly(L)√(nm log m)/√d ·
  ‖W′‖ √F), Fig. 1 (CIFAR-10 slices along ∇F and the most negatively curved direction "after smoothing"), the remark
  "in practice, we acknowledge that one often goes beyond this theory-predicted spectral-norm boundary" (p. 4),
  Sec. 4 proof steps, Sec. 5 Theorems 5–7 (arbitrary loss, CNN, ResNet), Sec. 6.
- Fixes to the notes: none. No discrepancies found ("Theorems 1–7 and Sections 3–6" matches the ICML numbering).
- Title: "Paper 11 — …" → without "Paper 11". Boxes: "Do not overread 'global minimum.'" → warn, "What transfers to
  the paper:" and "Why this is more than…" → key, "Our notes (copy into Papers Master):" → plain.
- Primer (4 ideas): PL inequality without convexity (x² + 3 sin²x: F″ reaches −4, F′²/(2F) ≥ ≈ 0.17); ReLU kinks break
  the descent lemma (slope ratio 5 over ±0.1, 500 over ±0.001); few ReLUs flip under small moves (P(|z| < 0.01) ≈ 0.8%,
  labelled an illustration of proof step 2); geometric decrease gives log(1/ε) steps (1375 for γ = 0.01, ε = 1e−6) and a
  bounded path (1/(1 − √0.99) ≈ 200). All in `allenzhu2019convergence.py`, with the notes' Jacobian example (K
  eigenvalues 1, 3; ‖∇F‖² ≥ 2F; F1 = 0.5625) and the ReLU values. (Script is slow, ~2–3 min: it trains small nets.)
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 13 sections, no overflow at 375 px.

## [11] cohen2021gradient — Gradient Descent at the Edge of Stability
- Notes checked against: arXiv 2103.00065 (ICLR 2021): Sec. 2 quadratic instability (Prop. 1, Fig. 2: eigenvalues 20
  and 1, η = 0.11), sharpness = λ_max of the training-loss Hessian and "we do not claim that this quantity has any
  connection to generalization" (p. 3), Sec. 3 progressive sharpening / edge of stability, Sec. 6 and p. 3 ("the
  sharpness does not flatline at any value during SGD").
- Fixes to the notes: none. No discrepancies found.
- Minute box: no "paper in one minute" heading in the notes (labels Problem / Method / Main finding); standard title.
  "What this calculation proves:" (amber) → warn; "Connection to our survey." (green) → key. "Source:" → `>` line.
- Primer (4 ideas): power iteration for sharpness (diag(20,1): 19.95, 19.9999); thresholds 1/ℓ (descent lemma) vs 2/λ
  (quadratic stability), with the paper's Fig. 2 numbers (2/0.11 ≈ 18.2, multipliers −1.2, 0.89); sharpness depends on
  position (½(uv − 1)²: sharpness u² + v² on the valley, 4.25 at (2, 0.5) vs 2 at (1, 1); η = 0.6 leaves the first,
  stays at the second — simulated); period-2 oscillation just past 2 (multiplier −1.02). All in `cohen2021gradient.py`
  together with the notes' e_t = (1 − ηa)^t e_0 behaviour.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 4 sections, no overflow at 375 px.

## [12] lewkowycz2020large — The Catapult Mechanism at Large Learning Rates
- Notes checked against: arXiv 2003.02218: Sec. 1.3 (three phases; λ_final < 2/η; c_act ≈ 12 for ReLU), Sec. 2.1
  warm-up model f = n^{−1/2}vᵀu, Eq. (5) λ = n⁻¹(‖v‖² + ‖u‖²), exact Eqs. (6)–(7), η_crit = 2/λ₀, η_max = 4/λ₀,
  Sec. 3 experiments (flatter minima, test accuracy vs learning rate), Secs. 4–5.
- Fixes to the notes: none. No discrepancies found ("Sections 2–5" matches; the toy L(w) = ½(1 − w²)² is labelled
  by the notes as not the paper's model and L″ = −2 + 6w² is correct).
- Minute box: no "paper in one minute" heading in the notes (Problem / Method / Result); standard title.
  "Characteristic trajectory:" (green) → key.
- Primer (4 ideas), built on the paper's own warm-up model and its exact Eqs. (6)–(7): the kernel as output curvature
  (n = 1, u = 2, v = 0.35: f0 = 0.7, λ0 = 4.1225, 2/λ0 ≈ 0.485); the kernel shrinks while ηλ < 4; the catapult in
  numbers (ηλ0 = 2.6: loss 0.245 → 0.484 → 0, final λ ≈ 1.21 < 2/η ≈ 3.17; n = 1e4: 3.98 converges, 4.02 diverges);
  width delays the spike by ~log n (peaks at steps 2, 6, 9 for n = 1e2, 1e4, 1e6). All in `lewkowycz2020large.py`,
  including a check that Eqs. (6)–(7) reproduce GD on (u, v) exactly.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 5 sections, no overflow at 375 px.

## [13] zhang2026saddle — Why Networks Learn One Unit at a Time
- Notes checked against: ICLR 2026 proceedings PDF: Eq. (1), Theorem 1 and Corollary 2 (Sec. 3, proofs App. E),
  Theorem 3 (Sec. 4, App. F), Eqs. (8)–(10) and Theorem 4 (Sec. 5.1: multiplicity r < D, N(0, ε²) init,
  ‖(I − P)θ_i‖ = O(ε^{1 − s_{r+1}/s_1}) a.s.), Proposition 5 (quadratic case), Figs. 1–2 and captions, tanh discussion
  (Secs. 5–7), Apps. G–H.
- Fixes to the notes: none. No discrepancies found (Fig. 2 rows match the caption: width barely matters for linear
  nets, shortens plateaus in linear self-attention; κ = 0 removes plateaus in linear nets only; larger init shortens).
- Title: "Paper 8 — …" → without "Paper 8". The two-box stage schematic is redrawn as SVG. Boxes: "Why this example
  matters", "Difference to memorize" → key; "What Theorem 4 proves…" → theorem; "Do not mix the two mechanisms",
  "One subtlety for a good survey" → warn; "Papers Master — 2-line note" → plain.
- Primer (4 ideas): gradient flow as the η → 0 limit (0.99^100 ≈ 0.366 vs e⁻¹ ≈ 0.368); W = Σ v_i u_iᵀ and merged
  identical units (rank 1); growth time log(1/ε)/s (6.9 vs 13.8 for ε = 1e−3, 1e−6); a plateau is time near a saddle
  (½(1 − a²)² from a = 0.01: L = 0.462 at t = 1.5, half-way at t ≈ 2.3, 0.0017 at t = 3). All in `zhang2026saddle.py`,
  with the notes' numbers (t = log(100)/3 ≈ 1.535, a2 ≈ 0.0464, v1(9) = 1, v2(9) ≈ 0.091, unit duplication).
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 11 sections, no overflow at 375 px.

## [14] bottou2018optimization — Optimization Methods for Large-Scale ML
- Notes checked against: arXiv 1606.04838 (SIAM Review 60(2)): Sec. 4 (Assumption 4.3 first/second moment limits,
  Sec. 4.3 nonconvex SG theorems), Secs. 5–6 (noise reduction, second-order methods).
- Fixes to the notes: none. No discrepancies found. The notes' expected-descent inequality and the telescoped bound
  (1/T)Σ E‖∇F‖² ≤ 2Δ/(αT) + Lασ² are correct under their stated simplification (unbiased, variance ≤ σ², α ≤ 1/L),
  and the notes themselves flag that the paper's Assumption 4.3 is more general.
- Minute box: no "paper in one minute" heading in the notes; standard title. "Source distinction." (amber) → warn,
  "Survey relevance." (green) → key.
- Primer (4 ideas): one full gradient = n sample gradients (n = 1e6); E‖g‖² = ‖∇F‖² + noise (g ∈ {1, 5}: 13 = 9 + 4);
  telescoping; optimising the bound over α (Δ = L = σ² = 1, T = 1e4: α* ≈ 0.0141, bound ≈ 0.0283, 1/√T). All in
  `bottou2018optimization.py`, plus Monte Carlo checks of the notes' two inequalities on a quadratic (L = 3, σ² = 0.5)
  and of the averaged bound along one SG run.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 5 sections, no overflow at 375 px.

## [15] jain2017non — Non-Convex Optimization for Machine Learning
- Notes checked against: arXiv 1712.07897 (FnT ML 10(3–4)) and the author copy named in the notes (cse.iitk.ac.in
  …/ncom-book.pdf, HTTP 200): Parts I–II contents, Ch. 3 (non-convex projections onto sparse vectors and low-rank
  matrices, RSC/RSS, Algorithm 2 gPGD, Theorem 3.3), Ch. 6 (escaping saddles).
- Fixes to the notes: none. No discrepancies found ("Parts I–III, especially Chapters 3 and 6" matches the monograph).
- Minute box: no heading in the notes (Problem / Approach / Main message); standard title. "Why this matters for RQ4."
  (green) → key.
- Primer (4 ideas): convex sets and why sparse sets fail ((1,0,0), (0,1,0) midpoint); easy projections ((0.3, −2, 1.1,
  0.05), s = 2 → (0, −2, 1.1, 0); rank-1 of diag(3,1)); restricted curvature (A = [[1,0,1],[0,1,1]]: AᵀA eigenvalues
  0, 1, 3, but over 2-sparse directions [0.382, 2.618]); restricted isometry Monte Carlo (50 × 200 Gaussian A, 3-sparse
  w: ratios in [0.38, 1.97], 98% in [0.60, 1.51]). The key box quotes Theorem 3.3 (β/α < 2, η = 1/β,
  O(α/(2α − β) log 1/ε)). All in `jain2017non.py`.
- Audit: 0 KaTeX errors, `__atlasErrors` = [], 5 sections, no overflow at 375 px.
