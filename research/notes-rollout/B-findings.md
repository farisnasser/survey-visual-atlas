# Session B findings — in-page notes rollout (branch `notes-B`)

## Shared changes needed
- None so far.

Conventions used on every page of this session (the notes PDFs of this batch share one layout, unlike Lee's):
- The opening box (Problem / Method / Reported results, or similar) is the notes' one-minute summary: it goes into
  `:::minute` verbatim, one bold label per paragraph.
- Boxes with an inline bold lead ("**Concrete example.** …") become `:::key|warn|plain <lead>` with the lead as title.
- The title-block subtitle line ("RQ1 / RQ4 — …") is treated as part of the PDF title block and left out (the page header
  shows the tags; the RQs reappear in each page's "Survey connection" paragraph).
- "Source: <url>" at the bottom → `> **Source:** <url>` verbatim.
- Audit server: port 8582 (8572 is still held by the round-3 B server from 2026-10-30, a different worktree).

## [1] hochreiter1997flat — Flat Minima
- Checked against: the authors' copy (bioinf.jku.at 3304.pdf, Neural Computation 9(1):1–42), abstract p. 1 and Sec. 1–3
  (definition of a flat minimum as a large connected region of approximately constant error; MDL / Gibbs-algorithm argument;
  under/overfitting split; prior over input/output functions; second-order derivatives at backprop's order of complexity;
  pruning of units, weights, input lines; feedforward and recurrent experiments; stock-market comparison with backprop,
  weight decay, OBS/OBD).
- Fixes to the notes: none. No discrepancies found.
- Numbers checked in `scripts/check-math/hochreiter1997flat.py`: the Sec. 2 example (|a| ≤ √δ, |b| ≤ 10√δ).
- Unverified: none.
- Primer ideas: (1) curvature sets the room |u| ≤ √(2δ/λ) a weight can move (δ = 0.01: 0.014 vs 0.14);
  (2) precision costs log₂(R/w) bits (7.6 vs 4.3 bits); (3) MDL: model bits + error bits, overfitting vs underfitting;
  (4) the same function at different weights: (ab − 1)², λ_max = 2(a² + b²) = 4 at (1, 1), 32.1 at (4, 0.25).

## [2] keskar2017large — On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima
- Checked against: arXiv 1609.04836 v2 (ICLR 2017): Sec. 2.1 (Figure 1 conceptual sketch, MDL link), Sec. 2.2 (SB = 256,
  LB = 10% of the training data, ADAM; sharpness metric Eq. (4): maximum of f over a box C_ε scaled coordinatewise by
  ε(|x_i| + 1), normalized by 1 + f(x)), Sec. 4 (noise "pushes the iterates out of the basin of attraction of sharp
  minimizers"; warm-starting; data augmentation, conservative training, robust training as remedies).
- Fixes to the notes: none. No discrepancies found. (The notes' "maximum observed objective increase over a local
  neighborhood (with coordinatewise scaling)" omits the 1 + f(x) normalization of Eq. (4); not wrong, left as is.)
- Numbers checked in `scripts/check-math/keskar2017large.py`: the Sec. 2 bound max ½uᵀHu = ½r²λ_max over the r-ball,
  the 100× teaching example.
- Unverified: none.
- Primer ideas: (1) steps per epoch n/b (50,000 images: 196 vs 10 steps; the paper's SB/LB regimes); (2) minibatch noise
  falls like 1/√b (sd 1, 0.25, 0.0625); (3) generalization gap via a shifted test loss (the paper's Fig. 1: shift 0.1
  costs 0.01 vs 1); (4) basin half-width √(2δ/λ) vs noise kick ησ/√b (0.014/0.14 vs 0.1/0.006), labelled as the
  intuition the paper supports empirically.

## [3] dinh2017sharp — Sharp Minima Can Generalize for Deep Nets
- Checked against: arXiv 1703.04933 (ICML 2017): Theorem 1 (non-negative homogeneity of the rectifier), Sec. 4 α-scale
  transformation, Theorems 2–5 (ε-flat volume infinite, Hessian under reparameterization, spectral norm and many
  eigenvalues made arbitrarily large).
- Fixes to the notes: none. No discrepancies found. (The paper says "non-negative homogeneity"; the notes' "positive
  homogeneity" with c > 0 is the same identity.)
- The notes have no one-minute box: built from the notes' own sentences (Sec. 1 question, Sec. 2 method line, the
  "Main result" box, the Sec. 4 limits sentence), labels Question / Method / Main result / Limits.
- Numbers checked in `scripts/check-math/dinh2017sharp.py`: the rescaling identity f(x; cW₁, W₂/c) = f(x; W₁, W₂),
  the Hessian 2[[c⁻², 1], [1, c²]] at (c, 1/c) and its eigenvalues 0, 2(c² + c⁻²).
- Unverified: none.
- Primer ideas: (1) weights → function is many-to-one ((1, 1), (2, 0.5), (4, 0.25)); (2) curvature depends on the
  ruler (θ² → curvature 2, (φ/10)² → 0.02; H → DᵀHD); (3) ε-ball sharpness of (ab − 1)² is 0.021 at (1, 1) and 0.161
  at (4, 0.25).

## [4] saxe2014exact — How Deep Linear Networks Learn
- Checked against: arXiv 1312.6120 v3 (ICLR 2014): abstract (plateaus then rapid transitions; finite, depth-independent
  delay for special initial conditions; pretraining; random orthogonal initializations; edge of chaos in nonlinear nets),
  Sec. 2 Eqs. (3)–(6) mode decomposition, Eqs. (10)–(12) (τ du/dt = 2u(s − u), learning time τ/(2s) ln(…)).
- Fixes to the notes: none to content. Silent artefact fix: the PDF's quick-reference table has a stray third column
  holding only the word "Source:" (it belongs to the source line below the table); dropped.
- Kept verbatim: the notes number two sections "5." ("From the example to Saxe et al.'s paper" and "Initialization and
  the depth problem"); the duplicate is in the PDF and is not a factual error, so it stays.
- Run-in heads "Gradient recap" and "Orthogonal initialization from first principles" → `###` sub-heads.
- Numbers checked in `scripts/check-math/saxe2014exact.py`: gradients, the w = 0.01 example (prediction 0.0001, loss
  ≈ 0.5, |∇| ≈ 0.01), the depth-d product rule, rates 0.01 / 0.375 / 0, the logistic solution (simulated gradient flow
  from a = b), the factor-2 warning, orthogonal length preservation.
- Unverified: none.
- Primer ideas: (1) small weights start next to the saddle at the origin (Hessian eigenvalues ±1); (2) gradient flow
  conserves a² − b² (0.08 from (0.3, 0.1)), so balanced layers stay balanced; (3) modes of strength s are half-learned
  at ln(s/u₀ − 1)/(2s) (Eq. 11; 1.33 for s = 3 vs 3.45 for s = 1 from u₀ = 0.001); (4) depth multiplies scales
  (0.9⁵⁰ ≈ 0.005, 1.1⁵⁰ ≈ 117).

## [5] chizat2018global — Global Convergence Using Optimal Transport
- Checked against: arXiv 1805.09545 v2 (NeurIPS 2018): Eq. (1) J(µ) = R(∫φ dµ) + G(µ) over signed measures, Eq. (2)
  particle discretization (1/m)Σ wᵢδ_θᵢ (positions and weights), Sec. 1.2 contribution summary, Theorem 2.6
  (many-particle limit = Wasserstein gradient flow), Theorems 3.3 / 3.5, Sec. 4 (ReLU / sigmoid), numerical section.
- **Fix to the notes (one):** the "Actual contribution" box said "…the appropriate infinite-particle gradient flow
  converges to global minimizers of the convex-in-measure objective". The paper proves a conditional statement:
  Theorem 3.3 (and 3.5) "If (µ_t)_t converges to µ_∞ in W2, then µ_∞ is a global minimizer of F"; Sec. 1.2 (p. 3):
  "we prove that if this Wasserstein gradient flow converges, then the limit is a global minimizer of J". Fixed to
  "…if the appropriate infinite-particle gradient flow converges, its limit is a global minimizer of the
  convex-in-measure objective." (The convergence of the flow is a hypothesis, not a result.)
- The notes have no one-minute box: built from the notes' own sentences (Sec. 1 problem and idea, the corrected
  "Actual contribution" box, the Sec. 5 qualification), labels Problem / Idea / Actual contribution / Limits.
- Notes' µ_m = Σ a_j δ_θj vs the paper's (1/m)Σ w_i δ_θi: the same object (the 1/m is absorbed into a_j); not changed.
- Numbers checked in `scripts/check-math/chizat2018global.py`: ∫φ dµ_m = Σ a_j φ(θ_j) for a particle measure.
- Unverified: none.
- Primer ideas: (1) (ab − 1)² is non-convex in (a, b) (Hessian eigenvalues ±2 at 0) but convex in p = ab; (2) the
  predictor is linear in µ, so J is convex along mixtures (predictions 0 and 2 → mixture loss 0 ≤ 1); (3) gradient
  descent moves a cloud of particles, and duplicating neurons leaves the m-scaled motion unchanged (checked).

## [6] allenzhu2018neon2 — NEON2: Finding Local Minima via First-Order Oracles
- Checked against: arXiv 1711.06673 (NeurIPS 2018): Sec. 1.1 Theorems 1–7 (informal), Figure 1 (Neon vs Neon2),
  Sec. 3 Algorithms 1–2 and Theorem 1 (Õ(log²(d/p) L²/δ²) stochastic gradients; ⊥ ⇒ ∇²f ⪰ −δI, else ‖v‖ = 1,
  vᵀ∇²f v ≤ −δ/2), Sec. 4 Algorithm 3 / Theorem 3 (Neon2det, Chebyshev, Õ(√L/√δ) full gradients), Neon2finite
  (Theorem 5), Theorems 7a–7d (SGD, SCSG, GD, SVRG), Theorem 2 (Natasha2).
- Fixes to the notes: none. No discrepancies found. The "Ref 74 / 76 / 77" labels are the notes' old numbering; kept.
- Numbers checked in `scripts/check-math/allenzhu2018neon2.py`: the worked example (Hu = (0.2, −0.1) for both samples,
  u⁺ = (0.06, 0.12)).
- Unverified: none.
- Primer ideas: (1) oracles and the (ε, δ) second-order target (x² − y² has eigenvalue −2 at a zero-gradient point);
  (2) ∇f(x + u) − ∇f(x) = Hu + O(‖u‖²) (error 0.01 at u = (0.1, 0.1)); (3) the power method on I − ηH (multipliers
  0.6, 1.2: 1024× in ten steps); (4) repetition with verification: (1/3)¹⁰ ≈ 1.7·10⁻⁵.

## [7] foret2021sharpness — Sharpness-Aware Minimization for Efficiently Improving Generalization
- Checked against: arXiv 2010.01412 (ICLR 2021): Sec. 2 (L_S, L_D; Theorem 1 informal: L_D(w) ≤ max_{‖ε‖₂≤ρ} L_S(w + ε)
  + h(‖w‖²/ρ²); Eq. (1) SAM objective with p-norm; Eqs. (2)–(3) first-order ε̂ and dropped second-order terms;
  Theorem 2 in App. A), Figure 1 (error reduction; sharp vs wide minimum), Figure 2 ("Schematic of the SAM parameter
  update"), Sec. 3 (CIFAR-10/100, ImageNet, fine-tuning, label noise), Sec. 4.
- Fixes to the notes: none. No discrepancies found.
- Kept verbatim: two sections numbered "5." in the PDF.
- Numbers checked in `scripts/check-math/foret2021sharpness.py`: the Sec. 2 example (0.01 vs 0.10) and ε̂ = ρg/‖g‖.
- Unverified: none.
- Primer ideas: (1) training loss L_S vs population loss L_D; (2) Cauchy–Schwarz worst nudge (g = (3, 4), ρ = 0.1:
  ε = (0.06, 0.08), max 0.5); (3) Danskin: slope of the worst case is the slope at the worst point ((|w| + 0.1)²,
  slope 2.2 = L′(1.1)); (4) the Theorem 1 bound and the radius ρ as a dial.

## [8] damian2023self — Self-Stabilization at the Edge of Stability
- Checked against: arXiv 2209.15594 v2 (ICLR 2023): abstract (cubic Taylor term; "GD at the edge of stability implicitly
  follows projected gradient descent (PGD) under the constraint S(θ) ≤ 2/η"; predictions for loss, sharpness,
  deviation), Sec. 4.1 four stages and Eq. (3), Lemma 2 (∇S = ∇³L(u, u)), Theorem 1, App. B toy.
- Fixes to the notes: none. No discrepancies found. (The notes' "constrained by S(θ) ≤ 2/η" matches the abstract; the
  full constraint set of Eq. (2) also fixes ∇L·u = 0, but the notes say "resembling", so left as is.)
- The notes have no one-minute box: built from the notes' own sentences (Sec. 1 problem, Sec. 3 mechanism sentence,
  "Main result" box, Sec. 4 caveat), labels Problem / Mechanism / Main result / Caveat.
- Numbers checked in `scripts/check-math/damian2023self.py`: |1 − ηλ| < 1 ⇔ 0 < ηλ < 2; the toy gradient x²/2·∇S.
- Unverified: none.
- Primer ideas: (1) zig-zag multipliers −0.8 vs −1.2 (η = 1, λ = 1.8 / 2.2); (2) curvature moves: L = ½(2 + y)x², a
  swing x = 1 moves y by −0.5; (3) the paper's Eq. (3) with η = 1, α = 0.01, β = 1 (+0.01 per step, −0.115 at
  |x| = 0.5, threshold √(2α/β) ≈ 0.14); one full loop checked; the primer says the two-number map's cycles grow (it
  leaves |x| > 10 by step 191), consistent with Theorem 1 being finite-horizon.
- Note for the coordinator: a first draft of the primer used α = 0.1; the check script showed the discrete map (3)
  diverges within one cycle there, so the example was changed to α = 0.01 before commit.

## [9] kalra2024warmup — Why Warmup the Learning Rate? Underlying Mechanisms and Improvements
- Checked against: arXiv 2406.09405 (NeurIPS 2024): Sec. 1 (warmup's main benefit is tolerating a larger η_trgt; the
  "estimate the moments" explanation), Sec. 3–4 (progressive sharpening vs natural sharpness reduction, catapults),
  Sec. 4.3 / App. B.1 (preconditioned sharpness λ_max(P⁻¹H) in the paper's convention), Sec. 5 (catapult-based
  η_init via line search), Sec. 6 GI-Adam (v₀ = g₀²; equivalent to an automatic warmup η_t = η_trgt √(1 − β₂ᵗ)),
  Figures 1–4 (CIFAR-10/100, TinyImageNet, WikiText-2/103).
- Fixes to the notes: none. No discrepancies found. The notes write the update as Δθ = −ηP_t g_t (their P is the
  paper's P⁻¹); λ_max(P^{1/2}HP^{1/2}) is then the same quantity. Kept verbatim: two sections numbered "5.".
- Numbers checked in `scripts/check-math/kalra2024warmup.py`: Sec. 2 thresholds (2/20 = 0.1, 2/5 = 0.4) and Sec. 5
  P^{1/2}HP^{1/2} = diag(0.2, 1).
- Unverified: none.
- Primer ideas: (1) linear warmup η₅₀ = 0.155 (0.01 → 0.3 over 100 steps), 2/η from 200 to 6.7; (2) a catapult in
  ½(uv − 1)² from (2.2, 0.45) with η = 0.6 (sharpness 5.04 > 3.33, loss spike 1.55, ends at sharpness 2.01);
  (3) Adam's first step is η·sign(g₀); GI-Adam's is √(1 − β₂) = 0.0316 times smaller.

## [10] kalra2025universal — Why Neural-Network Sharpness Changes During Training
- Checked against: arXiv 2311.02076 (ICLR 2025 camera-ready; the proceedings PDF does not render in MuPDF past p. 4):
  Sec. 2 UV model (f = vᵀUx/√n^{1−p}, p ∈ [0, 1], n_eff), Sec. 3 regimes T1–T4, Sec. 4 Eqs. (1)–(2) and fixed points,
  Result 1 (η_c = √n_eff/(‖x‖y)), Corollary 5.1 (η_max < η_c ⇒ no EoS), Eq. (3) EoS-manifold map, Figure 3
  (bifurcation, ‖x‖ = 1, y = 2, n_eff = 1, η_c = 0.5), Figure 4 (EoS phase diagram and test accuracy), Figure 5
  (synthetic vs CIFAR-10 sharpness bands), Sec. 6.2 (weight norm near the saddle), Sec. 7 limitations (no
  non-monotonic loss decrease at EoS, no loss catapult at n_eff = 1, no long-range correlations).
- Fixes to the notes: none. No discrepancies found.
- Numbers checked in `scripts/check-math/kalra2025universal.py`: the Sec. 4 worked example (0.1198, 0.02870; 1.4,
  3.92), Eqs. (1)–(2) against a parameter-space GD step, η_c = 4/2 = 2, the cubic analogue's multiplier 1 − 2η.
- Unverified: none.
- Primer ideas: (1) (r, λ) summarize the weights (first-order change 0.00396 + Eq. 1's second-order term 0.000392 =
  0.004352); (2) fixed-point stability and period doubling in the notes' cubic map (η = 0.9 fixed, 1.1 two-cycle
  0.815/1.116, 1.25 four-cycle, 1.35 no short cycle); (3) η_c for n = 512: 11.3 (NTP, p = 0) vs 0.5 (µP, p = 1).

## [11] galli2026flatland — Flatland: Large-Step GD
- Checked against: PMLR 306 copy (galli26a.pdf): Def. 2.3 (large steps), Def. 2.4 and Eq. (2) (reference R_k,
  Zhang–Hager), equality line search (halve/double), Prop. 2.5 (η_k ≥ 2δ(1 − c)/ℓ(w_k, η_{w_k})), Def. 2.6 /
  Prop. 2.8 (Hölder), Theorem 2.9, Corollary 2.10, Sec. 3 (5000-image subsets; CIFAR10/100, SVHN, EMNIST; MLP, CNN,
  resnet34, wide resnet50_2, densenet121, vgg11; MSE main, CE in App. H; c = 10⁻⁴, δ = 0.5 do not ensure (i)/(ii)),
  Lemma 3.1 (bias block (2/K)I), "unsuccessfully-flat" vs "successfully-flat", saddles, Sec. 3.2 NLS-ub (η_k < K;
  "larger experiments would be needed"), Sec. 5 (SGD convergence open).
- Fixes to the notes: none. No discrepancies found. (research/findings.md records Prop. 2.5 as "2(1 − c)/ℓ(w_k, 2η_k)";
  the paper's statement is the notes' 2δ(1 − c)/ℓ(w_k, η_{w_k}) — worth correcting in findings.md, which I did not
  touch.)
- Numbers checked in `scripts/check-math/galli2026flatland.py`: multipliers −0.6 / −1 / −1.2, the accepted increase
  (0.605 ≤ 0.679), the (2/K)I bias block (K = 2, 10, 26, 100), the Rayleigh/block bound.
- Unverified: none.
- Primer ideas: (1) Armijo on ½x² accepts η ≤ 2(1 − c) = 1.98; halving/doubling lands in [(1 − c)/λ, 2(1 − c)/λ]
  (Prop. 2.5's shape); (2) Zhang–Hager reference (ξ = 0.85, losses 1.0, 0.5 → 0.73); (3) small sharpness with a
  negative eigenvalue: [[0.2, 0.3], [0.3, −0.5]] → 0.31, −0.61.

## [12] nesterov2006cubic — Cubic Regularization of Newton Method
- Checked against: the open CORE DP 2003/41 preprint (UCLouvain DIAL, "Cubic regularization of a Newton scheme and its
  global performance", April 2003): Introduction (Newton "may … converge to a saddle point or even to a point of local
  maximum"), Lemma 1 / Eq. (2.3) (cubic Taylor bound L/6‖y − x‖³), Eq. (2.4) model, Eq. (2.5) first-order condition
  g + Hs + (M/2)‖s‖s = 0, Proposition 1 (H + (M/2)r I ⪰ 0), Theorems 1–3 (global rates, limit points second-order,
  local quadratic convergence), Theorems 4–5 (star-convex classes).
- Fixes to the notes: none. No discrepancies found. The notes open with a "Source blocker" box saying the full article
  was not available when they were written; it is part of the notes and stays verbatim. Everything the notes state
  matches the paper (they use M for the Hessian Lipschitz constant, the paper uses L for it and M ≥ L in the model).
- The notes have no one-minute box: built from the notes' own sentences (Sec. 1, Sec. 2, "Documented result" box,
  Sec. 5 trade-off), labels Problem / Mechanism / Documented result / Trade-off.
- Numbers checked in `scripts/check-math/nesterov2006cubic.py`: the global minimizer of the cubic model satisfies
  g + Hs + (M/2)‖s‖s = 0 and (H + M‖s‖/2·I)s = −g (5 random indefinite instances).
- Unverified: none (journal version not read; the preprint is the same paper).
- Primer ideas: (1) Newton from (1, 0.5) on x² − y² lands on the saddle in one step; (2) the cubic Taylor cap is tight
  (x³/6: f(2) = 4/3 = cap); (3) at g = 0, H = −1, M = 1 the cubic step has length 2 = 2|H|/M and model value −2/3.

## [13] zhang2020gradient — Why Gradient Clipping Accelerates Training
- Checked against: arXiv 1905.11881 (ICLR 2020): abstract ("converge arbitrarily faster than gradient descent with
  fixed stepsize"), Definition 1 / Assumption 3 ((L0, L1)-smoothness), Lemma 2 (polynomials), Theorem 3 (clipped GD),
  Theorems 4 and 6 (fixed-step lower/upper bounds), empirical smoothness measurements (Fig. 1).
- Fixes to the notes: none. No discrepancies found.
- The notes have no one-minute box: built from the notes' own sentences (Sec. 1, Sec. 2, "Main result" box), labels
  Problem / Model / Main result / Limits.
- Numbers checked in `scripts/check-math/zhang2020gradient.py`: the clipped direction's norm min(‖g‖, c), e^x
  (f′ = f″) and the normalized step length.
- Unverified: none.
- Primer ideas: (1) descent lemma and η < 2/L; (2) ½x² + ¼x⁴ is (1, 3)-smooth (76 ≤ 1 + 3·130 at x = 5);
  (3) fixed step forced below 2/76 ≈ 0.026 (contraction 0.974 near 0) vs clipped h = 0.5/130 ≈ 0.0038 at x = 5
  (h·f″ ≈ 0.29) and h = 1 near the minimum.

## [14] li2023convergence — Convergence of Adam Under Relaxed Assumptions
- Checked against: arXiv 2304.13972 (NeurIPS 2023): Definition 3.2 / Assumption 2 ((ρ, L0, Lρ) smoothness, ρ < 2, a.e.),
  Lemma 3.4 (local Lipschitz gradient), Adam with h_t = η/(√v̂_t + λ), Theorem 4.1 (O(ε⁻⁴), bounded gradients along the
  trajectory w.h.p.), Theorem 4.2 (sub-Gaussian noise), Sec. 6 VRAdam (O(ε⁻³)), Proposition B.1 (rational functions).
- Fixes to the notes: none. No discrepancies found. (Notation only: the notes write δ in Adam's denominator and q for
  the exponent; the paper writes λ and ρ.)
- The notes have no one-minute box: built from the notes' own sentences (Sec. 1, Sec. 4 first sentence, the "Reported
  result" box, the "Do not claim" box), labels Problem / Proof idea / Reported result / Do not claim.
- Numbers checked in `scripts/check-math/li2023convergence.py`: e^x (f′ = f″).
- Unverified: none.
- Primer ideas: (1) EMA bias and its correction (g = 2, β₁ = 0.9: m₁₀ ≈ 1.30, m̂ = 2); (2) ε-stationarity and counting
  (ε = 0.01: 10⁸ vs 10⁶); (3) exponents q = 1 (eˣ), 2/3 (x⁴/4), 3/2 (1/x); (4) bounded gradients ⇒ local constant
  L0 + L1 G^q (31), the paper's Lemma 3.4.
- Self-correction before commit: the primer first said "for x⁴, f″ = 3|f′|^{2/3}" (true for x⁴/4 only); fixed.

## [15] apidopoulos2026taming — Taming Stochastic Gradient Descent
- Checked against: PMLR 306 copy (apidopoulos26a.pdf): abstract ((L0, L1) with superlinear, even exponential growth;
  TSGD under an ABC-type noise condition; dissipative taming under weaker moment bounds; "for both taming schemes …
  avoid strict saddle points (and/or manifolds thereof) with probability 1 so, generically, both methods only converge
  to local minimizers"), (TSGD) with deterministic scale r(x) and Remark 2 (random ‖ĝ‖ normalization introduces bias),
  Theorems 1–3.
- Fixes to the notes: none. No discrepancies found. The notes' "Verification scope" box (formulas not reconstructed) is
  part of the notes and stays verbatim.
- The notes have no one-minute box: built from the notes' own sentences (Sec. 1, Sec. 4, the "Reported result" box),
  labels Problem / Method / Reported result / Limits.
- Numbers checked in `scripts/check-math/apidopoulos2026taming.py`: the teaching step's length < 1, the multiplier
  1 + η_tγ.
- Unverified: none.
- Primer ideas: (1) almost surely vs finite time (ten tails: 1/1024); (2) Robbins–Monro step conditions (H₁₀₀₀ ≈ 7.49,
  π²/6); (3) x⁴/4 with η = 0.1 blows up from 5 (→ −7.5 → 34.7) while the teaching tamed step goes to 4.07, and the
  paper tames with a deterministic r(x) (Remark 2); (4) noise leaves the stable manifold, |y| = 0.01 → 1 in 49 steps.
