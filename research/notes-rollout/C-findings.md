# Session C — in-page notes findings (branch notes-C)

## Shared changes needed
- None so far.

Dev server note: port 8573 was already taken by a stale `scripts/serve.py 8573` from the round-3 C worktree (PID 35136,
started 30 Oct); requests to :8573 reached that server. This session audited on **:8583** instead and left the old process alone.

## [1] ghadimi2013stochastic — Stochastic Gradients & Stationarity
- **Checked against:** arXiv 1309.5549v1 (= SIAM J. Optim. 23(4)), pp. 1–8: abstract, Sec. 1, Assumption A1 (unbiased,
  variance ≤ σ²), f bounded below, RSG (2.3) random output with P(R = k) ∝ 2γ_k − Lγ_k² (uniform for a constant step),
  Corollary 2.2 (γ_k ∝ N^{-1/2}), Sec. 3 zeroth-order RSGF.
- **Notes verified:** the §3 "teaching derivation" is correct line by line (E‖G‖² ≤ ‖∇f‖² + σ², 1 − Lη/2 ≥ 1/2 for
  η ≤ 1/L, telescoping); checked numerically on a noisy quadratic in the script. DOI correct.
- **Fixes:** none. No discrepancies found.
- **Layout:** the notes' minute box has "Question … Answer …" in one paragraph; split into two labelled paragraphs.
  "Our takeaway" → `:::ours`. The notes have no source line; built from the header (authors, title, venue, DOI).
- **Primer (5 ideas):** L-smoothness gives the quadratic upper bound (sin w example) · noise adds σ² to E‖G‖² (G ∈ {1,3}
  example) · telescoping turns progress into a budget Δ · the step size balances 2Δ/(ηN) and Lησ² (η*, N = 100 vs
  10 000) · a random output iterate inherits the average (why RSG returns w_R).

## [2] arjevani2023lower — Lower Bounds for Non-Convex Stochastic Optimization
- **Checked against:** arXiv 1912.02365 (Math. Program. 199, 2023): Sec. 2–4, Theorem 1 (Ω(ΔLσ²ε⁻⁴), bounded variance,
  tight for SGD), Theorem 2 (Ω(ΔL̄σε⁻³ + σ²ε⁻²), mean-squared smoothness, tight for SPIDER/SNVRG), Theorem 3 (all
  randomized algorithms), proof of Theorem 1 (scaling λ = 2εℓ₁/L, T ∝ LΔ/ε², p ∝ ε²/σ²).
- **Notes verified:** §2 SGD derivation correct (checked numerically); both lower bounds, the oracle classes and the
  SPIDER comparison match the paper.
- **Fixes:** none. No discrepancies found.
- **Wording kept, flagged:** §5 calls the guarantee "asymptotic in the stationarity tolerance". The paper's bounds are
  non-asymptotic (they hold for every ε ≤ c′√(LΔ)); the phrase is read as "about the order in ε" and left as written.
  The notes write the mean-squared smoothness constant as L̃ (paper: L̄); symbol choice, kept.
- **Layout:** the notes have no one-minute summary; built from the notes' own sentences (§1 opening paragraph → Problem
  and Approach, the "Main result" box → Main result). "Main result." box → `:::key Main result`, "Important distinction."
  → `:::warn Important distinction` (bold box labels become box titles, as in notes-F's convention). The notes' header "Ref. [40]" (old numbering; survey [48]) is dropped with the title block.
- **Primer (5 ideas):** complexity counts oracle queries (minimax) · ε⁻⁴ vs ε⁻³ in numbers · a zero-chain hides one
  coordinate per query (T ∝ ΔL/ε²) · a probability-p oracle makes each discovery cost 1/p (T = 10, p = 0.1 → 100;
  z/p variance 9; p ∝ ε²/σ² as in the proof) · same-sample differences cancel noise (why SPIDER reaches ε⁻³).

## [3] polyak1964some — Some Methods of Speeding up the Convergence of Iteration Methods
- **Checked against:** the Russian original, Zh. Vychisl. Mat. Mat. Fiz. 4(5) (1964) 791–803 (Math-Net.ru full text;
  the English translation is USSR Comput. Math. Math. Phys. 4(5) 1–17, paywalled). Method (9) two-step iteration,
  Theorem 9 (p. 800): for 0 ≤ β < 1, 0 < α < 2(1 + β)/M the method converges locally with q₂, minimal value
  q̄₂ = (√M − √m)/(√M + √m) at α = 4/(√M + √m)², β = ((√M − √m)/(√M + √m))²; p. 801–802: the continuous version (10)
  is the motion of a body in a potential field, so (9) is called the "heavy ball method"; GD's q̄₁ = (M − m)/(M + m).
- **Notes verified:** heavy-ball update, per-eigen-direction recurrence, characteristic equation, the optimal tuning and
  the (√κ − 1)/(√κ + 1) factor (checked: worst root modulus over [μ, L] for κ = 4, 25, 100). §4 saddle remark
  p(1) = −ηa is correct (the PDF has the η). "Our Ref. [10]" (perturbed accelerated GD) matches survey ref. [10]
  (Jin, Netrapalli, Jordan 2018).
- **Fixes:** none. No discrepancies found. The DOI contains parentheses; the link URL is percent-encoded
  (`%2864%29`) so the renderer's link pattern does not cut it.
- **Layout:** no one-minute summary in the notes; built from §1 (research goal), §2 (method sentence and update),
  the "Key insight" box, and §5 (limitation). "Key insight." → `:::key Key insight`; "Historical attribution." → `:::warn Historical attribution`.
  Header "Ref. [41]" (old numbering; survey [49]) dropped with the title block.
- **Primer (5 ideas):** GD's worst multiplier (κ − 1)/(κ + 1) (κ = 100: 0.980, ~690 steps for 10⁶) · two-step
  recurrences are solved by characteristic roots (0.9, 0.2 example) · complex roots share modulus √β, so tuned heavy
  ball contracts at (√κ − 1)/(√κ + 1) (0.818, ~69 steps) · the heavy-ball name and inertia (from the paper) ·
  p(1) < 0 ⇒ a root above 1 (η = 0.1, β = 0.9: growth 1.316 vs 1.1 for GD).

## [4] duchi2011adaptive — Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
- **Checked against:** JMLR 12 (2011) 2121–2159, journal PDF: Sec. 1 (notation, subgradients, online/stochastic
  equivalence via Cesa-Bianchi et al. 2004), Sec. 1.2 Corollary 1 (η = D∞/√2, R(T) ≤ √2 D∞ Σᵢ‖g_{1:T,i}‖₂ form),
  Sec. 1.3 sparse example (pᵢ = min{1, c i^{−α}}, O(log d √T) for α ≥ 2 vs √(dT)), diagonal update with δI + diag(s_t).
- **Notes verified:** diagonal AdaGrad update (δ added outside the root, as in the paper), the proximal-surrogate
  derivation (checked numerically), regret definition, the schematic diagonal bound (labelled schematic by the notes).
- **Fixes (cross-reference numbers, old draft numbering → survey numbering, as in notes-F's convention):**
  §5 "Adam (Ref. [43])" → "Adam (Ref. [51])"; §5 "(our Ref. [59])" → "(our Ref. [66])" — old [59] is Antonakopoulos
  et al., *AdaGrad Avoids Saddle Points* (its notes header reads "Ref. [59]"), survey [66].
- **Layout:** no one-minute summary in the notes; built from §1 (Problem, Method), the "Main result" box and the
  "Do not transfer guarantees" box (Limitation). Bold box labels → box titles. Header "Ref. [42]" (survey [50]) dropped.
- **Primer (5 ideas):** online regret and sublinear growth (√T → 0.01 average at T = 10⁴) · subgradients (|x| at 0) ·
  one rate starves rare features (G = 10⁴ vs 10² → steps η/100 vs η/10) · the paper's §1.3 sparse example in numbers
  (harmonic H_d 2.93 / 9.79 vs √d 3.16 / 100) · online-to-batch conversion needs convexity.

## [5] kingma2015adam — Adam: A Method for Stochastic Optimization
- **Checked against:** arXiv 1412.6980v9 (ICLR 2015): Algorithm 1 (ε added after the square root), Sec. 2.1 (|Δ_t|
  roughly bounded by α, "signal-to-noise ratio" m̂/√v̂ and automatic annealing, invariance to gradient scale), Sec. 3
  (initialization bias correction), Sec. 4 (online convex regret analysis), abstract ("little memory requirements").
- **Notes verified:** both moment recurrences, bias corrections, the update, the constant-gradient derivation
  m_t = (1 − β₁ᵗ)g (checked), the claims of §4–§6 (regret analysis is online-convex; later work found flaws).
- **Fixes (cross-reference numbers → survey numbering):** §6 "such as our Ref. [34]" → "our Ref. [42]" — old [34] is
  Li, Rakhlin & Jadbabaie, *Convergence of Adam Under Relaxed Assumptions* (its notes header reads "Ref. [34]"), survey [42].
- **Layout:** no one-minute summary; built from §1 (Goal), the "Main contribution" box and the "Accuracy note" box.
  Bold box labels → box titles. Header "Ref. [43]" (survey [51]) dropped.
- **Primer (5 ideas):** EMA memory ≈ 1/(1 − β) (10 and 1000 steps) · averaging shrinks noise variance by (1 − β)/(1 + β)
  (= 1/19) · dividing by the RMS removes gradient scale (paper Sec. 2.1) · m̂/√v̂ as signal-to-noise and automatic
  annealing (paper's terms) · uncorrected early steps are too large (3.16α at t = 1, peak 6.57α at t = 12).

## [6] laurent2018deep — All Local Minima Are Global (Deep Linear Networks with Arbitrary Loss)
- **Checked against:** PMLR 80 (ICML 2018) PDF, pp. 1–6: hypotheses (i) convex + differentiable loss, (ii) thinnest
  layer is the input or output layer (structural condition (3)); Theorem 1 (local = global), Theorem 2 (convex,
  Lipschitz, non-differentiable counterexample; Eq. (22)–(23) f(x, y) = |x| + (1 − y)₊ − 1), Theorem 3 (local minimizer
  of (P2) ⇒ ∇f(Â) = 0), Sec. 3 remark (critical points ≠ local minima).
- **Notes verified:** the scalar example (zero-loss settings, ∂F/∂w at (0, 0), f′(0) = −8), Theorem 3 and the width
  condition, the role of differentiability. All numbers checked in the script.
- **Fixes:** none. No discrepancies found.
- **Layout:** the notes' "Paper in one minute." box has no labels; kept as one paragraph with its bold sentence.
  "The actual bridge … (Theorem 3)" box → `:::theorem`; "Limits / survey connection." → `:::warn`; "Takeaway:" →
  `:::ours Takeaway`; the notes' own "Source:" line kept verbatim, with the PMLR link added.
- **Figure (§2):** redrawn from the real function f(x) = (x² − 1)²(1 + 0.1x) on [−2, 2] (two minima of equal value 0 at
  x = ±1, asymmetric walls like the PDF sketch; checked in the script), dashed amber line at the lowest loss, coral ball
  in the left valley, green "global min" dots.
- **Primer (5 ideas):** convexity + zero gradient ⇒ global minimum · the product breaks convexity (midpoint of (2, 2)
  and (−2, −2) has loss 16) · (0, 0) is a saddle (Hessian eigenvalues ±8) · a width-1 layer limits the product to rank 1 ·
  at a kink a linear cost beats a linear gain (the paper's counterexample; ε = 0.1, δ = 0.5 → 0.05 > 0).

## [7] nguyen2017loss — Why Width Can Make Bad Minima Exceptional
- **Checked against:** PMLR 70 (ICML 2017) PDF: Assumptions 3.2 (distinct samples; analytic, strictly increasing σ with
  growth conditions; l ∈ C² with l′(a) = 0 ⇒ global min), Lemma 3.5, Theorem 3.8 (non-degenerate on a subset
  I ⊆ {k+1, …, L} containing k+1; full column rank of W_{k+2..L}), Corollary 3.9, Lemmas 4.3–4.4 (measure zero),
  Theorem 5.3 (separability, class-dependent loss), Sec. 6 discussion (sparse connectivity as future work).
- **Notes verified:** every theorem/lemma number and condition; the 3-example matrix (entries tanh(−1), tanh(1),
  tanh(3)), det ≈ −0.982 and β ≈ (3.339, −2.326, 1.772) reproduced in the script; the gradient identity in the box.
- **Fixes:** none. No discrepancies found.
- **Layout:** minute box verbatim (Question / Model/Method / Finding / Why we care). "Why this example matters" and
  "Think of a locked door" → `:::key`; "Be precise about 'almost all'" and "Important limits" → `:::warn`;
  "Papers Master — Our notes (copy)" → `:::plain`; its "Source:" line (with the PDF's own link) → source line.
  Four tables as pipe tables. Title block (Paper 16, key, tags, tier) dropped.
- **Primer (5 ideas):** full row rank ⇔ every target fittable (n_k + 1 ≥ N) · the loss enters only through l′(aᵢ) and
  Assumption 3.2.3 · analytic functions vanish on measure-zero sets (Lemmas 4.3 → 4.4) · implicit function theorem
  and what degeneracy breaks (u³/3 − vu example) · full column rank keeps backpropagated error (pyramidal tail).

## [8] haeffele2017global — Global Optimality in Neural Network Training
- **Checked against:** CVPR 2017 open-access PDF (pp. 7331–7339): abstract, contributions paragraph, Eq. (17), Theorem 1
  (local minimizer with a zero subnetwork ⇒ global), Corollary 1, Theorem 2 (r > card(X) ⇒ non-increasing path),
  limitations paragraph. The official supplement URL in the notes now returns 404; read from the Wayback Machine copy
  (snapshot 2026-03-20, 8 pages): **Proposition 2** (convex ℓ with ∂ℓ(0) ≠ ∅, Φ of degree p, Θ of degree p′ < p ⇒
  local minimum at the origin), proof Eqs. (20)–(22) = the notes' §4 inequality.
- **Notes verified:** the model, matching-degree assumption, ReLU homogeneity, the supplement's counterexample and its
  "Proposition 2" number, "eight-page" supplement.
- **Fixes:** none. No discrepancies found.
- **Observation (not changed):** Theorem 1 itself has no size condition; the notes' §3 and "Main result" say "sufficiently
  large"/"sufficient model size". This follows the paper's own framing (contributions: "if the network size is large
  enough …"; limitations: "large enough … is essential to prove that local minima are global"), and size is needed for
  Theorem 2 — so it is not a discrepancy with the paper. The survey [54] sentence has the same framing.
- **Dead link:** the notes' Supplement URL is 404 on openaccess.thecvf.com (kept verbatim; an archived copy exists at
  web.archive.org/web/20260320233928/…/Haeffele_Global_Optimality_in_2017_CVPR_supplemental.pdf). Coordinator may want
  to point the link there.
- **Layout:** no one-minute summary; built from §1 (Problem), the "Main result" box and the "Scope" box. Bold box
  labels → box titles. Header "Ref. [47]" (survey [54]) dropped.
- **Primer (5 ideas):** bias-free ReLU units are homogeneous of degree 2 (6 → 54 under ×3) · balancing makes weight
  decay the convex |x| (x = 4 at a = b = 2) · with matched degrees the origin is a saddle (eigenvalues λ ± 1; f* = 0.18 at
  λ = 0.2) · with mismatched degrees the origin traps (λ = 0.15, s = 0.1: −0.00995 vs +0.03; 0.5 vs 0.3) · a zero unit
  lets the local-min condition test every new unit (intuition for Theorem 1, flagged as such).

## [9] safran2018spurious — When Bad Local Minima Really Exist
- **Checked against:** PMLR 80 (ICML 2018) PDF: Eq. (2) population loss, Theorem 1 (n = k ∈ {6..20}; (8, 9) and (k, k+1)
  for 10 ≤ k ≤ 19), Corollary 1 (d > k; probability ≥ 1 − exp(−Ω(d)) over Gaussian teacher vectors), Lemmas 1–2
  (certificate), Sec. 4.2 closed-form kernel, experiments (1000 runs, step 0.1, Xavier init), Table 1 (6: 0.3 %,
  10: 34.6 %, 15: 80.3 %, 20: 94 %), Table 2 ((15, 16): 10.1 %, (19, 20): 49.1 %), Example 1 certificate for n = k = 6
  (λ_min ≥ 0.004699, r = 1.12·10⁻⁷, objective ≥ 0.02508), Figure 1.
- **Notes verified:** every table value and certificate number against the paper; the one-neuron example
  (E[x²1{x>0}] = 1/2, F(0.6) = 0.04) checked in the script.
- **Fixes:** none. No discrepancies found.
- **Layout:** minute box verbatim. "The contrast with Lee (2016)", "A proof certificate from the paper (n = k = 6)",
  "Survey-worthy contrast", "The narrative for the survey" → `:::key`; §13 Papers Master note → `:::plain` under its
  heading; "Source:" line verbatim with the PDF's link. Five tables as pipe tables. W_* written with a subscript star as
  in the rendered PDF.
- **Check script runtime:** the existing GD replication makes `safran2018spurious.py` take ≈ 17 min; it passes.
- **Primer (5 ideas):** closed-form Gaussian ReLU kernel (1/2 and 1/(2π)) · neuron permutations give k! global minima
  and force non-convexity (k = 2 midpoint loss ≈ 0.091) · a sphere higher than its centre traps a local minimum
  (ε = 0.001, λ = 0.1, B = 1: +5.4·10⁻⁵ at t = 0.05) · error bars on 1000-run frequencies (0.17, 1.5, 0.75 points) ·
  splitting a neuron by homogeneity (why extra width adds room; observed, not proved).

## [10] wu2025loss — Loss Landscape of Shallow ReLU-like Neural Networks
- **Checked against:** ICLR 2025 proceedings PDF (= openreview ogKE7LcvW6): Eq. (1)–(2) model and squared loss,
  Def. 3.5 (directional stationarity: all one-sided directional derivatives ≥ 0), Sec. 3 discussion of Clarke
  stationarity (|x| vs −|x| example), Def. 4.1 escape neuron, Theorem 4.2 (no escape neuron ⇒ local min; necessary when
  |J| = 1), Corollary 4.4 (second-order decrease Cδ², C < 0), Fact 4.5 (‖wᵢ‖² − h² conserved), Cor. 4.6 / Fact 4.7
  (vanishing initialization), Prop. 4.8 (unit replication / embedding).
- **Notes verified:** model, the smooth-region gradient formula (checked by finite differences), both directions of
  the escape-neuron result, the splitting identity, the scope statements.
- **Fixes:** none. No discrepancies found.
- **Layout:** no one-minute summary; built from §1 (question), the "Paper result" box and the "Do not conflate" box.
  Bold box labels → box titles. Header "Ref. [49]" (survey [56]) dropped.
- **Primer (5 ideas):** one-sided directional derivatives and the paper's stationarity · Clarke is weaker (|x| vs −|x|,
  the paper's own example) · a stationary point can still drop at second order (one neuron: L(0.1, 0.1) = 0.49005 < 0.5;
  dead half-plane) · the conservation law ‖w‖² − a² (Fact 4.5; checked under small-step GD) · plateaus from small
  initialization (5·10⁻⁶ → 0.1 at ηλ = 0.001: ≈ 9900 steps; labelled a linearized estimate).

## [11] islamov2024loss — Loss Landscape Characterization Without Over-Parametrization
- **Checked against:** arXiv 2410.12455 (NeurIPS 2024): Definition 1 / Eq. (2) (for any x ∈ X some x_p ∈ Proj(x, S),
  α > β > 0; reduces to QCvx when β = 0 and S is a singleton), Assumption 1 (each fᵢ L-smooth), Assumption 2
  (σ²_int := Eᵢ[f* − fᵢ*]), Theorem 1 (SGD, γ ≤ (α − β)/2L, Eq. (9): O(K^{-1/2}) up to 2β/(α − β)·σ²_int), Theorems 2–3
  (SPSmax, NGN), Sec. 5 (checks along training runs: MLP, CNN, ResNet, LSTM, GNN, Transformer).
- **Notes verified:** the condition, the distance expansion (checked), the error-floor statement and the empirical caveat.
- **Fixes:** none. No discrepancies found.
- **Observation (not changed):** the "Paper result" box says the condition supports "meaningful" bounds. On the paper's
  own saddle example (Example 2) the condition needs α ≈ 8.8·10⁶ and the floor is vacuous (see research/findings.md);
  the paper says it does not aim for tight constants. Wording kept (it is the paper's claim, and §4/§5 state the floor).
- **Layout:** no one-minute summary; built from §1 (question), the "Paper result" box and the "Critical distinction"
  box. Bold box labels → box titles. Header "Ref. [50]" (survey [57]) dropped. Source line uses the paper's exact
  title (*Loss Landscape Characterization of Neural Networks without Over-Parametrization*).
- **Primer (4 ideas):** PL forces every stationary point to be global (Lee's quartic saddle: gap 1/4 at zero gradient)
  · convex alignment ⟨∇f, x − x*⟩ ≥ f − f* and the distance expansion (4 → 1) · the β term rewritten as slack
  (α − β)(fᵢ(x) − fᵢ(x_p)) − β(fᵢ(x_p) − fᵢ*); averaging shows saddles need σ²_int > 0 (derived here from the definition,
  stated as such) · two disagreeing examples: σ²_int = 0.5 and a constant-step SGD floor ≈ 0.026 at γ = 0.1.

## [12] bhojanapalli2016global — When Local Search Becomes Global
- **Checked against:** NeurIPS 2016 proceedings PDF (and arXiv 1605.07221 for Sec. 5): Definition 2.1 (RIP), Theorem 3.1
  (δ₄ᵣ < 1/10, ‖UUᵀ − X*‖_F ≤ 20√(log n/m)σ_w), Theorem 3.2 (strict saddle), Theorem 3.3 (projected perturbed update (4)),
  Theorem 3.4 (approximately low rank), Theorem 4.1 (noiseless, δ₄ᵣ < 1/5), Lemma 4.2, Corollary 4.1, Lemma 4.4, m = O(nr),
  Figure 1 (n = 100, transition near m = 2nr), Sec. 5 (NP-hardness, Examples 1–2).
- **Notes verified:** the 2 × 2 example (expansion, gradient, f(0.1, 0) = 0.9801, Hessian diag(−4, 0)), all theorem
  numbers and constants, Figure 1 description.
- **Fix — error in the paper itself, repeated by the notes (§8):** the paper's Example 1 (Sec. 5), f(X) = (X₁₁ + X₂₂ − 1)²
  + (X₁₁ − 1)² + X₁₂² + X₂₁² with a rank-1 constraint, says "X = diag(0, 1) is a rank 1 local minimum with f(X) = 1". It is
  not: the rank-one PSD matrices [[s, √(s(1−s))], [√(s(1−s)), 1−s]] → diag(0, 1) as s → 0 and have f = 1 − s² < 1
  (checked in the script). Example 2 fails the same way (rank-2 PSD neighbours of diag(0, ½, ½) with f = 1 − s). The
  arXiv version's Example 1 (without the off-diagonal terms) fails even faster (f = 1 − 2ε + … along X₁₂ = X₂₁ = √ε).
  Notes §8 original: "Section 5 constructs measurement operators … for which a rank-constrained recovery problem does
  have a nonglobal local minimum. One example has … but X = diag(0, 1) is a suboptimal rank-one local minimum with loss
  1." → now attributes both sentences to the paper ("the paper states") and adds a one-paragraph "Correction (our
  check)" with the counter-path, keeping the NP-hardness argument of Sec. 5, which stands. The "Interpretation"
  paragraph is kept. The existing visual page does not use this example. **Coordinator:** the survey text for [58]
  should not cite Sec. 5's explicit examples as proof that spurious minima exist without RIP.
- **Layout:** minute box verbatim; "What the paper is really asking", "Why 4r?", "Key implication", "Don't memorize the
  constants in the lemmas" → `:::key`; "Papers Master note (copy)" → `:::plain`; "Source:" line verbatim (its link).
- **Primer (4 ideas):** factorization saves parameters (500 vs 5050) but is non-convex (midpoint of ±(1, 0) has loss 1)
  · one Gaussian measurement set preserves ‖Z‖_F² on average (sd √(2/m) = 0.1 at m = 200); RIP demands it for all
  low-rank Z · first- and second-order necessary conditions and how Theorems 3.1/4.1 + 3.2 combine · worst-case
  NP-hardness vs the structured RIP class (points to the §8 correction).

## [13] bhojanapalli2016dropping — Dropping Convexity for Faster Semi-definite Optimization
- **Checked against:** PMLR 49 (COLT 2016) PDF (53 pp.): abstract, Sec. 1 problem (1)–(2), Algorithm 1 (FGD) with step
  (8) η = 1/(16(M‖X₀‖₂ + ‖∇f(X₀)‖₂)), DIST(U, U*) up to rotation, Theorem 4.1 (smooth: O(1/k)), Theorem 4.2 /
  Corollary 4.3 (restricted strongly convex: linear), Sec. 5 initialization (first-order oracle), "for several problem
  instances … global convergence guarantees" (abstract).
- **Notes verified:** the factor gradient (checked by finite differences), rotation symmetry, the local/global
  distinction and the two rates.
- **Fixes:** none. No discrepancies found.
- **Layout:** no one-minute summary; built from §1 (question), the "Paper result" box and the "Do not overstate" box.
  The notes write X_* (subscript) in the teaching example; kept. Header "Ref. [52]" (survey [59]) dropped.
- **Primer (4 ideas):** chain rule through the product (scalar g(u) = f(u²); g′(1) = −6; a new stationary point at 0)
  · what O(1/k) and linear rates mean (0.9⁵⁰ ≈ 0.0052, 0.9²⁰⁰ ≈ 7·10⁻¹⁰) · local = within a radius measured up to
  rotation (g″ = 16 at u = 2, −8 at u = 0) · curvature of g grows like 4a², so a fixed step 0.2 diverges at a = 2 —
  why the paper's step uses ‖X₀‖.

## [14] ge2017no — When Non-Convexity Has No Bad Minima
- **Checked against:** PMLR 70 (ICML 2017) PDF + arXiv 1704.00708: Definition 6 (aligned Δ), Lemma 6 (‖ΔΔᵀ‖²_F ≤ 2‖M − M*‖²_F),
  Lemma 7 / Eq. (9), Theorem 3 ((2r, 1/20)-RIP, balancing penalty ⅛‖UᵀU − VᵀV‖²_F), Theorem 4 (completion, μ-incoherence,
  w.h.p.), Theorem 5 (robust PCA, pseudo strict saddle).
- **Notes verified:** the 2 × 2 example (both expansions, gradient, Hessian diag(−2, 0), F(0.5, 0) = 0.28125); Lemma 7
  with H = I and no regularizer checked numerically at random U (the identity holds everywhere, including the
  4⟨∇f, Δ⟩ term) together with Lemma 6.
- **Fixes:** none. No discrepancies found. Note: the paper's Eq. (9) has "+ 4⟨∇f(U), Δ⟩ + [Δ:∇²Q:Δ − 4⟨∇Q, Δ⟩]" with f
  the full objective; the notes write "+ 4⟨∇F_data(U), Δ⟩ + regularizer correction", which is the same identity regrouped.
- **Layout:** minute box verbatim. §1's three-row table has no header in the PDF; rendered with an empty header row.
  "Keep the scope straight", "What this teaches", "Idealized case H = I …", "Survey takeaway" → `:::key`; §10 Papers
  Master note → `:::plain`. The notes have no source line; built from the PMLR link. Title block dropped.
- **Primer (4 ideas):** curvature along a direction = second derivative on a line (U = 0: −2 = 1 − 3 ≤ −1, Lemma 7 in
  numbers) · rescaling symmetry and the balancing penalty ((1, 2) pays 9/8, (√2, √2) pays 0) · incoherence (e₁e₁ᵀ is
  missed by 10 % sampling with probability 0.9) · a touching smooth upper bound (|x| ≤ x²/2 + 1/2) — the "pseudo" in
  Theorem 5.

## [15] anandkumar2016efficient — Escaping Higher-Order Saddle Points
- **Checked against:** PMLR 49 (COLT 2016) PDF: abstract ("first efficient algorithm guaranteed to converge to a third
  order local optimum … NP-hard to extend this further to finding fourth order local optima"), Definition 1 (p-th order
  local minimum), Definition 4 (third-order necessary condition), Theorems 8–9 / Corollary 1 (algorithm), Theorem 10
  (fourth order NP-hard) via Theorem 11 (nonnegativity of quartic forms).
- **Notes verified:** x² − y⁴ (Hessian diag(2, 0)), x² + y³, the deep-linear toy (gradient and Hessian vanish at 0,
  L(t, t, t) = ½ − t³ + ½t⁶) — all checked in the script; the algorithmic and hardness statements match the paper.
- **Fixes:** none. No discrepancies found.
- **Layout:** no one-minute summary; built from §1 (question), the "Paper result" box and the "Scope" box. The
  underbraced "⇏" display kept as `\underbrace` with `\not\Rightarrow`. Header "Ref. [54]" (survey [61]) dropped.
- **Primer (4 ideas):** p-th order local minimum (Definition 1; x² + y³ passes order 2, fails order 3) · third
  derivatives along a line, odd so the sign flips (±6) · fourth order is even and quartic-form nonnegativity is
  NP-hard ((u₁² − u₂²)² vs u₁⁴ + u₂⁴ − 3u₁²u₂² = −1) · deep products push escape to higher order (3 layers: −6;
  4 layers: −t⁴).

## [16] li2018visualizing — Visualizing Loss Landscapes
- **Checked against:** arXiv 1712.09913 (NeurIPS 2018): Secs. 3–4 (slices, filter normalization; BN parameters held
  fixed, footnote 2), Sec. 5 + Figs. 2–3 (batch 128 vs 8192; with weight decay 6.00 % vs 10.19 %, without 7.37 % vs
  11.07 %), Sec. 6 + Fig. 5 (ResNet-110 5.79 % vs no-short 16.44 %), Fig. 6 (Wide-ResNet-56 k = 1…8), Fig. 7
  (|λ_min/λ_max| via Lanczos), "A note of caution: Are we really seeing convexity?" (Sec. 6).
- **Notes verified:** every reported number and section/figure reference; the scalar rescaling example (1.05 / 1.005,
  5 % relative → 1.05 both) and the hidden-saddle example checked in the script.
- **Fixes:** none. No discrepancies found.
- **Layout:** minute box (one paragraph with four inline labels in the PDF) split into the four labelled paragraphs.
  "The key warning", "Connections to papers we have already studied" → `:::key`; "Read the result precisely" →
  `:::warn`; "One–three lines for Papers Master" → `:::plain`; "Source:" line verbatim with the PDF's link. Title block
  (Papers Master key, tags) dropped.
- **Primer (4 ideas):** random Gaussian directions are nearly orthogonal (sd of cosine ≈ 1/√D = 0.01 at D = 10⁴) · slice
  curvature uᵀHu is a weighted average of eigenvalues (−0.5 vs +1 on diag(1, 1, −2)) · random slices see the mean
  eigenvalue and hide rare negative ones (0.997 with one −2 among 999 +1) · Hessian–vector products and Lanczos
  (10¹² entries = 4 TB; Hv from two gradients).

## [17] goodfellow2015qualitatively — Qualitatively Characterizing Neural Network Optimization Problems
- **Checked against:** arXiv 1412.6544 (ICLR 2015): Sec. 2 method ℓ(α) = J((1 − α)θ₀ + αθ_f), Secs. 3–4 (seven models,
  mostly monotone; the paper's own exceptions: a shallow early barrier, conv maxout Fig. 6, and a secondary minimum on
  the MP-DBM line, Fig. 7), Fig. 5 (two SGD solutions are separate minima on their joining line), Fig. 14 (factored model
  (1 − w₁w₂)² with a saddle at the origin).
- **Notes verified:** the method, the chain-rule slope (checked), the saddle example and the hedged statements ("often",
  "several", "particular line") all match the paper.
- **Fixes:** none. No discrepancies found.
- **Layout:** no one-minute summary; built from §1 (question, method), the "Paper result" box and the "Careful wording"
  box. Header "Ref. [56]" (survey [63]) dropped. Source line built (authors, ICLR 2015, arXiv link); the notes' own link
  (saxelab.org PDF) stays in §6.
- **Primer (3 ideas):** convex ⇒ every line to the minimum goes down (9(1 − α)²), but non-convex functions can also give
  monotone lines · symmetric solutions are separated by barriers (paper's Fig. 14 toy: peak 1 at the midpoint; Fig. 5) ·
  the line is not the SGD path (GD from (0.5, 0.1): monotone line 0.9025 → 0, path deviates up to 0.062).

## [18] panageas2019first — First-Order Methods Almost Always Avoid Saddle Points: Vanishing Step-Sizes
- **Checked against:** arXiv 1906.07772**v3** (27 Sep 2025; NeurIPS 2019 paper with the authors' correction): Theorem 1.1
  (informal; "isolated and non-isolated"), Sec. 2 (now: "We also assume that the Hessian matrix of the objective function
  f is invertible"), Theorem 4.1 (stable manifold for time non-homogeneous systems, H invertible), Corollaries 4.2–4.3
  (α_k → 0, α_k ∈ Ω(1/k)), Sec. 5 (GD, mirror descent, proximal point, manifold GD; "We also assume that the Hessian
  ∇²f(x*) is invertible"), Remark 3.1 + Fig. 1 (summable steps, 1/k⁴ converges to a non-critical point), Acknowledgement
  ("the proof of the main theorem to work requires the extra assumption that the Hessian of f is invertible").
- **Fix (authors' own correction):** §3 original "including isolated and non-isolated strict-saddle sets, for suitably
  controlled vanishing step sizes" → kept, with the inserted qualifier "(as stated in the NeurIPS version; the authors'
  arXiv v3 correction adds the assumption that the Hessian at the saddle is invertible, which only isolated saddles
  satisfy)". A critical point with an invertible Hessian is isolated, so non-isolated strict-saddle sets are not covered
  by the corrected proof. The source line cites the v3 correction. **Coordinator/survey:** the survey's [64] text should
  say "with an invertible Hessian at the saddle" if it mentions non-isolated saddles (round-3 finding says it does not).
- **Notes verified:** the linearization, the ∑α_k argument (checked), Ω(1/k), the four method families, the limits.
- **Layout:** no one-minute summary; built from §1 (question), the "Principal result" box and the "Do not conflate" box.
  Header "Ref. [57]" (survey [64]) dropped.
- **Primer (4 ideas):** products of growth factors are controlled by ∑α_k (∏(1 + 1/k) = t + 1 vs ∏(1 + 1/k²) =
  sinh π/π ≈ 3.68) · summable steps freeze GD at non-critical points (x_∞ ≈ 0.174 with α_k = 0.8/(k+1)²; paper Fig. 1) ·
  why a time-varying map needs Theorem 4.1 (diagonal example) · mirror descent with the entropy ((0.5, 0.5) → (0.269, 0.731)).

## [19] mertikopoulos2020almost — On the Almost Sure Convergence of SGD in Non-Convex Problems
- **Checked against:** NeurIPS 2020 PDF + arXiv 2006.11144: Assumptions 1–5 (Assumption 5: uniformly exciting noise,
  E[⟨Z, u⟩₊] ≥ c for every unit u), Theorem 1 (bounded APTs), Theorem 2 (γ_n = Θ(1/n^p), p ∈ (2/(q+2), 1] ⇒ a.s. convergence
  to a component of the critical set), Theorem 3 (γ_n ∝ 1/n^p, p ∈ (0, 1], q = ∞ ⇒ P(X_n → S) = 0 for every strict saddle
  manifold S, any initial condition), Theorem 4 (local rate near regular minimizers).
- **Notes verified:** the update, descent lemma and the conditional-expectation inequality (checked by Monte Carlo on a
  non-convex quadratic), the saddle toy (checked), the result statements and limits.
- **Fixes:** none. No discrepancies found.
- **Layout:** no one-minute summary; built from §1 (research problem), the "Principal result" box and the "Scope" box.
  Header "Ref. [58]" (survey [65]) dropped.
- **Primer (4 ideas):** expectation vs almost-sure (P = 1/n: E → 0 but ones recur, ≈ 12.1 expected in 10⁵ draws; 1/n²:
  finitely many) · step sizes 1/n^p and the paper's p > 2/(q+2) (½, ⅓, → 0) · uniformly exciting noise (no y-noise ⇒ stuck
  on y = 0) · saddle manifolds with a singular Hessian (radial φ, ridge r = 1 with eigenvalues −1, 0) — ties to the
  Panageas v3 invertibility assumption.
