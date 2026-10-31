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
