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
