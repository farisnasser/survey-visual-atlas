# Session F findings — in-page notes rollout

## Shared changes needed

- None to `src/` so far.
- Convention (coordinator to confirm across sessions): the notes' cross-references "(Ref NN)" use the old draft
  numbering (the notes headers say "Ref 94" for survey [97]). On the Atlas a reader takes them as survey numbers, so
  session F corrects the number to the current survey number and logs each one as a fix below; the names in the notes
  are unchanged. If the coordinator prefers the old numbers verbatim, revert those lines (each is listed).

## [1] zhang2019lookahead — Lookahead Optimizer

- Checked against: arXiv 1907.08610v2 (19 pp.) and the NeurIPS 2019 proceedings PDF linked from the notes' source line
  (same section numbering): Algorithm 1 and Eqs. (1)–(3) p. 2–3, Prop. 1 p. 3, Sec. 3.1 / Eqs. (4)–(7) / Prop. 2 p. 4,
  Sec. 3.2 p. 5, Sec. 5 Experiments, Table 1 p. 6.
- Fixes to the notes:
  - "Empirical evidence (Section 4)" → "(Section 5)". Paper: Section 4 is "Related work"; the experiments
    (CIFAR, ImageNet, Penn Treebank, WMT 2014 En–De) are "5 Experiments" in both the arXiv and NeurIPS versions.
  - Cross-references "Wang et al. (Ref 72)" → "(Ref 77)" (survey [77] = Wang, Zhou, Liang, Lan, *Cubic regularization
    with momentum*; survey [72] is Azizian et al.) and "Staib et al. (Ref 85)" → "(Ref 89)" (survey [89] = Staib et al.;
    [85] is Natasha 2). Old draft numbering.
- Verified, no change: Eq. (1) outer update; the notes' unrolled EMA formula (consistent indexing; the paper's Eq. (2)
  writes φ_{t+1} with (1 − α)^t φ_0, an off-by-one in the paper, not in the notes); A and Σ diagonal (Sec. 3.1, "joint
  eigenbasis"); Prop. 2 same γ for both, Lookahead variance fixed point strictly smaller for α ∈ (0, 1); Table 1
  Lookahead 95.27 ± .06 / 78.34 ± .05 (reported, checked against the table; SGD 95.23 ± .19 / 78.24 ± .18, so "small"
  is fair); worked φ updates 10 → 9 → 8 and V = α σ²/(2 − α) = σ²/3 (check script).
- "0 < α ≤ 1": the paper never writes this range explicitly but uses α ∈ (0, 1] (Fig. 3) and clips the adaptive α to
  [α_low, 1]; kept.
- Box mapping: "Core intuition" → key, "Not a saddle-escape theorem" → warn, Papers Master note → untitled plain box.
- Primer (four ideas): (1) SGD on a noisy quadratic has a stationary variance γ²a²σ²/(1 − (1 − γa)²) = 1/3 at
  a = 1, γ = 0.5, σ² = 1; (2) expected loss = ½a(mean² + variance + σ²) (paper Eq. 5), so lower variance = lower
  steady-state risk; (3) interpolation cancels oscillation: a = 1.9, γ = 1, k = 5, α = 0.5 gives round multiplier
  0.205 vs (−0.9)⁵ ≈ −0.590 (labelled as a toy; the paper's Sec. 3.2 wraps momentum); (4) EMA weights 0.5, 0.25, 0.125.
  All numbers in `scripts/check-math/zhang2019lookahead.py`.

## [2] wang2022analyzing — Analyzing Sharpness along GD Trajectory (Li, Wang, Li)

- Checked against: arXiv 2207.12678v2 (NeurIPS 2022; 53 pp.): Sec. 2 (sharpness defined as λ_max of the Gram/NTK
  matrix M, shown ≈ the Hessian's), Sec. 3.1 four phases p. 4, Sec. 3.2 M = M_A + M_W and Eq. (2) p. 4–5, Fig. 1–2,
  Sec. 3.4, Sec. 4.1 update rule D(t+1) = (I − ηM*(t))D(t) p. 9, Sec. 4.2 Assumptions 4.1–4.5, Theorems 1–2 p. 9–10,
  App. C (symmetric initialization), Sec. 5 (high-frequency oscillations unexplained, inner layers not in the theory).
- No discrepancies found. The notes define sharpness with the Hessian; the paper defines it with the Gram matrix after
  arguing the two top eigenvalues nearly coincide — compatible, left as is. Theorem 2's "nearby threshold" is the
  paper's (2/η)(1 + ρ), ρ = O(1).
- Kept verbatim: the notes number two sections "5." ("Proof idea…" and "Assumptions and limits"), so the page shows
  sections 1–5, 5–8 like the PDF. Not a claim about the paper; left unchanged.
- Box mapping: "Why does the output layer matter?" → key; "Copy into Our notes" → plain with that title.
- Primer (four ideas): (1) sharpness depends on the point: L = (aw − 1)² has Hessian 2[[w², 1], [1, a²]] on aw = 1,
  top eigenvalue 4 at (1, 1) and 8.5 at (2, 0.5); (2) for squared loss the top curvature is the Gram/NTK matrix's
  (paper Sec. 2), M = 2(a² + w²) in the toy; (3) first-order residual dynamics D ← (I − ηM)D (Assumption 3.2), top
  component multiplied by 1 − ηΛ; (4) d‖A‖²/dt = −(4/n)DᵀF (Eq. 2): DᵀF = −0.25 at aw = 0.5 (output layer grows),
  0.75 at aw = 1.5 (shrinks). All in `scripts/check-math/wang2022analyzing.py`.

## [3] chen2023beyond — Beyond the Edge of Stability via Two-step Gradient Updates

- Checked against: ICML 2023 PMLR PDF linked from the notes (and arXiv 2206.04172): Sec. 4.1 Theorem 1 (conditions
  (i) f‴ ≠ 0, (ii) 3[f‴]² − f″f⁗ > 0, η slightly above 2/f″), Lemma 1, Props. 1–2; Sec. 4.2 Theorem 2 (¼(x² − μ)²,
  η = K/μ, 1 < K < √4.5 − 1 ≈ 1.121, 0 < x₀ < √μ, period-2 orbit except a measure-zero set), Theorem 3 (½(xy − μ)²,
  1 < K < 1.5, x, y > 0), Prop. 3; Sec. 5 single neuron (teacher–student, population loss), Sec. 6 matrix
  factorization (Theorem 6 in the appendix).
- No discrepancies found. Worked numbers x₁ = 1.1024, x₂ ≈ 0.8532 and 3·6² − 2·6 = 96 checked. The notes call
  ½(xy − μ)² a "bilinear teacher–student loss"; the paper calls it "a special 2-D function" and uses the teacher–student
  model in Sec. 5, which it links back to Theorem 3 — a loose label, not a wrong statement; kept.
- Kept verbatim: two sections numbered "5." (as in the PDF).
- Box mapping: "Check the worked quartic" → key; "Copy into Our notes" → plain.
- Primer (four ideas): (1) a minimum is a fixed point of F_η with multiplier 1 − ηf″ = −1.1 at η = 1.05 (repelling);
  (2) curvature changes across the minimum (Taylor, f‴): f″(0.8) = 0.92, f″(1.1024) ≈ 2.646; (3) convergence to a cycle:
  from 0.8 at η = 1.05 the iterates settle on {0.8729, 1.0911}, loss 0.0142 / 0.0091, two-step multiplier 0.595;
  (4) period doubling: in our simulation the 2-cycle lasts until η ≈ 1.237 (labelled as ours), so Theorem 2's 1.121 is
  sufficient, not sharp. All in `scripts/check-math/chen2023beyond.py`.

## [4] mohtashami2023special — Special Properties of GD with Large Learning Rates

- Checked against: ICML 2023 PMLR PDF linked from the notes (23 pp.; arXiv 2205.15142 for the appendix): Sec. 3 Main
  Results (Theorems 1–2 informal), Sec. 4.1 Def. 1–2 (L-smoothness, μ-OPSC), Lemma 1 (2/μ† < γ ≤ μ*/L², μ† > 2L²/μ*),
  Theorem 1 formal (small γ < μ†/L²_global from inside M → x†; large step from any positive-measure set → x* a.s.),
  Sec. 4.2 Theorem 2, Sec. 5 experiments (5.1 full-batch vs stochastic), Figs. 1, 3, 4 (Fig. 4 = regions of Lemma 1).
- No discrepancies found. The paper writes the step as γ; the notes' η is kept. Worked multipliers 1 − 0.1·10 = 0 and
  1 − 0.25·10 = −1.5 checked (and η = 0.25 > 2/κ = 0.2).
- Kept verbatim: two sections numbered "5."; "an one-point" (notes' wording).
- Box mapping: "Why the theorem is a genuine contribution" → key; "Copy into Our notes" → plain. The two-line
  "small step / large step" display → `aligned` with `\text`.
- Primer (four ideas): (1) descent lemma for η < 2/L (drop 0.05‖∇f‖² at L = 10, η = 0.1), so small steps never leave a
  basin; (2) η = 0.25 is stable at κ = 0.3 (0.925) and unstable at κ = 10 (−1.5); (3) one-point strong convexity
  (Def. 2) and the contraction 1 − 2ημ + η²L² = 0.75 at μ = 1, L = 2, η = μ/L² (also checked on the nonconvex
  f = x² + 0.5 sin²x); (4) the step window 2/μ† < η ≤ μ*/L² (Lemma 1's form), (0.2, 0.25] for μ† = 10, needs μ† > 8.
  All in `scripts/check-math/mohtashami2023special.py`.

## [5] agarwala2023sam — SAM Operates Far from Home

- Checked against: arXiv 2302.08692 (ICML 2023, 15 pp.): Sec. 1.2 contributions list (Theorem 2.1 = limited
  suppression near initialization; **Theorem 2.2 = SAM-modified EOS**; **Theorem 2.3 = SGD / batch fraction β**),
  Eq. (6) unnormalized SAM, Eq. (7), Theorem 2.2 p. 4 (C^∞ model, MSE, zero-residual θ*, JJᵀ positive definite,
  convergent if ηλᵢ(1 + ρλᵢ) < 2 − ε for all i, divergent if > 2 + ε for one) and Eq. (13), Sec. 2.3 Theorem 2.3,
  Fig. 3 (FC net, MSE, 2-class CIFAR; GD η = 4·10⁻³, SAM ρ = 10⁻²), Fig. 4 (WideResNet 28-10, CIFAR-10).
- Fixes to the notes (the theorem numbers are swapped in the notes):
  - Sec. 4: "derives the SAM stability condition near a zero-residual solution in Theorem 2.3" → "Theorem 2.2".
    Paper: "SAM induces a modified edge of stability (EOS) (Theorem 2.2)" and Theorem 2.2 is the zero-residual
    stability statement giving Eq. (13).
  - Sec. 4 "Why do batch size…": "Theorem 2.2 shows minibatch effects can strengthen regularization" → "Theorem 2.3".
    Paper: "As batch size decreases, the effect of SAM is stronger and the dynamics is no longer controlled by the
    Hessian alone (Theorem 2.3)."
  - Sec. 6 first bullet: "Theorem 2.3 concerns a smooth model near a zero-residual solution…" → "Theorem 2.2".
- Verified, no change: λ_SAM = (−1 + √4.2)/0.4 ≈ 2.62 and multipliers −0.5 / −1.4 (check script); Figure 3 and the
  WideResNet / CIFAR-10 experiments as described.
- The notes have no source line (only the title-block "Original publication" link, to the PMLR page); the page ends
  with a "Primary source" line built from the paper's authors/title/venue and that link.
- Quick reference table has no header row in the PDF; written with an empty header row (renders as label/value cards
  on phones). Box mapping: "Read the equation…" → key; "Connection to the previous EoS papers" → plain.
- Primer (four ideas): (1) SAM takes the gradient at an uphill point: L = 1.5x² at x = 1, ρ = 0.2: normalized → x = 1.2
  (loss 2.16, gradient 3.6), unnormalized → x = 1.6 (gradient 4.8 = (λ + ρλ²)x); (2) edge of stability as a ceiling
  on sharpness (Cohen et al. 2021, stated as background); (3) MSE residual dynamics z ← (I − ηJJᵀ)z, SAM →
  I − ηJJᵀ(I + ρJJᵀ) (checked on a random linear model); (4) normalized SAM on the quadratic bounces at ±0.6 from
  0.05 (our toy, labelled). All in `scripts/check-math/agarwala2023sam.py`.

## [6] shin2025sassha — SASSHA

- Checked against: arXiv 2502.18153 (ICML 2025, 27 pp.): Sec. 3 / Table 1 (ResNet-32, CIFAR-100: SGD λ_max 265 /
  69.32 %, SASSHA 107 / 72.14 %; AdaHessian, Sophia-H, Shampoo compared), Sec. 4.1 Eqs. (1)–(4) (min–max objective,
  ε* = ρ∇f/‖∇f‖₂, Hutchinson diagonal Hessian at the perturbed point), Sec. 4.2 (absolute value + square root),
  Sec. 4.3 (lazy Hessian, every k steps), Algorithm 1, Assumptions 4.1–4.3 (bounded below; twice differentiable,
  convex, β-smooth; ∇f(x_t) ≠ 0 for t ∈ {1, …, n}), Theorem 4.4 (Σηₜ = ∞, Σηₜ² < ∞, Σρₜ²ηₜ < ∞ ⇒ liminf ‖∇f‖ = 0),
  Def. 4.5, Corollary 4.6.
- No discrepancies found. Worked example (100, 0.25) → (10, 0.5) checked.
- The notes have no source line (title-block "Original publication" link → PMLR page); added a "Primary source" line.
- Quick reference table: no header row in the PDF (empty header row); norms in cells written with \lVert…\rVert.
  Box mapping: "Why absolute values and square roots?" → key; "Don't overstate the theorem" → warn; "Why would lazy
  Hessian updates help?" is a sub-heading in the PDF (no box) → `###`.
- Primer (four ideas): (1) Newton fixes ill-conditioning: on ½(0.01x² + 4y²) GD at η = 0.5 needs ~139 steps to halve
  the flat coordinate, Newton one step; (2) negative curvature sends a Newton step to the maximum (−½x² from 0.5 → 0),
  |H| reverses it (→ 1); (3) Hutchinson diagonal estimate E[z ⊙ Hz] = diag H, probes (3, 4) and (1, 2) average to
  (2, 3) — the paper uses Hutchinson (Sec. 4.1); (4) |h|^{-1/2} between GD and Newton: multipliers 0 and 0.95 at
  η = 0.5, ~14 steps to halve. All in `scripts/check-math/shin2025sassha.py`. (I left out a linear-stability idea:
  the paper's Corollary 4.6 has its own form, a(1 + ρa) ≤ 2ε/η, and a simplified version would not match it.)

## [7] taniguchi2024adopt — ADOPT

- Checked against: arXiv 2411.02853 (NeurIPS 2024, 39 pp.): Sec. 2.1 Assumptions 2.1–2.6 (the analysis uses 2.1,
  2.2, 2.3, 2.5; 2.6 = almost-sure bound, removed), the measure min_t (E‖∇f‖^{4/3})^{3/2}, Sec. 3 (Theorem 3.1, lagged
  normalization, order swap with momentum), Algorithm 1 (max{√v, ε}), Theorem 4.1 (α = Θ(1/√T), O(1/√T) for any
  β2), toy problem Eq. (17), experiments (image classification, generative, language, RL).
- No discrepancies found. Worked example v = 2.5, 4/√2.5 ≈ 2.53 vs 4 checked.
- The notes have no source line (title-block "Original publication" link → NeurIPS proceedings page); added a
  "Primary source" line.
- Box mapping: "Key first-principles connection" → key; "Why this belongs in the survey" → plain (neutral title,
  cautionary content). Quick reference: no header row in the PDF; norm written with \lVert…\rVert.
- Primer (three ideas): (1) Adam's per-coordinate scale (constant gradient c → step α·sign c); (2) same-step
  normalization biases the step: the paper's Eq. 17 at k = 10 (g = 100 w.p. 0.1, −10 otherwise, mean +1) with
  v_{t−1} = 1, β2 = 0.9 gives normalized values 3.16 / −3.03, average −2.41 (wrong sign) vs +1 with the past scale —
  labelled as a single-step illustration; (3) the 4/3-moment measure is ≤ E‖∇f‖² by Jensen (1.41 vs 2 for ‖∇f‖ ∈
  {0, 2}), and O(T^{−1/2}) is the minimax SGD rate the paper cites. All in `scripts/check-math/taniguchi2024adopt.py`.

## [8] gupta2018shampoo — Shampoo

- Checked against: arXiv 1802.09568 (ICML 2018, 21 pp.): Algorithm 1 (L₀ = εI_m, R₀ = εI_n, L_t = L_{t−1} + GGᵀ,
  R_t = R_{t−1} + GᵀG, W ← W − ηL^{−1/4}GR^{−1/4}), Lemma 1 (online mirror descent regret), Lemma 4
  ((L ⊗ Rᵀ)vec G = vec(LGR)), Lemmas 5–6 (operator monotonicity), Theorem 7 (rank ≤ r matrices, convex regret),
  Lemmas 8–9 (Kronecker upper bound on the full-matrix statistic), Algorithm 2 / Theorem 10 (tensors, mode ranks rᵢ).
- No discrepancies found. Worked example diag(2, 1) → I₂ and the 10¹² vs 2 × 10⁶ memory figures checked.
- The notes have no source line (title-block "Original publication" link → PMLR page); added a "Primary source" line.
- Box mapping: "The important memory saving" → key; "What makes the proof work?" is a sub-heading in the PDF → `###`;
  "Different use of the word 'second order'" → warn. Quick reference: no header row in the PDF.
- Primer (four ideas): (1) regret and O(√T) (100 after 10⁴ rounds = 0.01 per round); (2) diagonal vs full-matrix
  AdaGrad and its 10¹² / 10¹⁸ cost for a 1000 × 1000 layer; (3) spectral vs entrywise matrix powers:
  [[2, 1], [1, 2]]^{−1/2} = [[0.789, −0.211], [−0.211, 0.789]] vs entrywise [[0.707, 1], [1, 0.707]]; (4) Lemma 4 and
  (L ⊗ R)^p = L^p ⊗ R^p. All in `scripts/check-math/gupta2018shampoo.py`.

## [9] defazio2024road — The Road Less Scheduled

- Checked against: the NeurIPS 2024 camera-ready PDF linked from the notes (34 pp.) and arXiv 2405.15682v4: Eqs.
  (3)–(5), Theorem 1 (convex, G-Lipschitz, i.i.d., γ = D/(G√T), any β ∈ [0, 1]: E[F(x_T) − F(x*)] ≤ DG/√T), Theorem 2
  (online-to-batch), Theorem 3, AlgoPerf 2024 self-tuning win.
- No discrepancies found. Note on figure numbers: the notes' "Figures 2–4 … language modeling and the AlgoPerf suite"
  matches the NeurIPS version (Fig. 2 averaging comparison + momentum on quadratics, Fig. 3 deep-learning experiments
  incl. GPT-2/OpenWebText, Fig. 4 AlgoPerf). In the arXiv v4 numbering these are Figs. 2, 3, 5 and 7 — the notes cite
  the version they link, so no change.
- Worked example (y₂ = 0.6, z₃ = 1.7, x₃ = 0.9) checked.
- The notes have no source line (title-block "Original publication" link → NeurIPS page); added a "Primary source"
  line. Box mapping: "How to think about the algorithm" → key; "What happens in real optimizers?" sub-heading;
  "Practical caveat" → warn. The three-sequence display with right-hand labels → `aligned` with `&&\text{}`.
- Primer (three ideas): (1) constant-step noise ball γ/(2 − γ): 1/3 at 0.5, 0.026 at 0.05, and why schedules need T;
  (2) Polyak–Ruppert averaging divides variance by ~t (0.01 for 100 draws) but lags (10, 0, 0, … → 1 after 10
  iterates); (3) the convex Lipschitz benchmark DG/√T (0.01 at D = G = 1, T = 10⁴), optimal in the worst case. All in
  `scripts/check-math/defazio2024road.py`.

## [10] ahn2025general — Schedule-Free SGD Is Also Effective for Nonconvex Optimization

- Checked against: arXiv 2411.07061 (32 pp.) and the ICML 2025 PMLR PDF the notes link (24 pp.): Assumption 1
  (ΔF, well-behavedness line integral, ‖∇F‖ ≤ G), Assumption 2 (independent unbiased oracle, variance ≤ σ²),
  Definition 1 ((λ, ε)-stationarity, O(λ^{1/2}ε^{−7/2}) optimal per Zhang & Cutkosky 2024), Proposition 1 (arXiv) /
  2.1 (ICML), Theorem 2, Algorithm 5 and the base-step identity z_{t+1} − z_t = −η/(1 − ζ)·g_t.
- No discrepancies found. Version note: the notes' "Proposition 2" (base-step identity) is the arXiv numbering; in the
  ICML copy linked at the bottom of the notes it is "Proposition 5.1". The Atlas header links the arXiv PDF, where
  "Proposition 2" is right, so the text is unchanged.
- Worked example (x₂ = 0.5, y₂ = 0.625, g₂ = −1.375, z₃ = 1.6875) checked. The notes write κ_t for the paper's β_t.
- Box mapping: "Conceptual proof path" → key; "Why the interpolation matters" and "Why this is genuinely a schedule-free
  SGD result" are sub-headings in the PDF. Quick reference: no header row in the PDF; norms with \lVert…\rVert.
- Primer (three ideas): (1) on F = |x| the gradient never gets small; the cloud {−0.01, 0.03} has mean 0.01, mean
  gradient 0, spread 0.0004 (Definition 1's two terms); (2) regret and online gradient descent's DG√T bound for any
  gradient sequence (simulated: 75.5 ≤ 100); (3) reading √λ ε^{−7/2}: 2^{3.5} ≈ 11.3, λ = L²/ε gives Lε^{−4}, λ = H/2
  gives ε^{−3.5} (Proposition 1). All in `scripts/check-math/ahn2025general.py`.

## [11] crawshaw2026exploration — Muon and Its Many Variants

- Checked against: arXiv 2510.09827 (29 pp.): Sec. 3.1 CSD vs RSD, Sec. 3.2 product norms, Prop. 3.6 (MuonAdam is
  exactly CSD in a spectral/adaptive product norm), MuonMax (stale dual norms, Eq. 16), Sec. 4 Momo (lower bound F*),
  Sec. 5: 36 variants on GPT2-Small / FineWeb 1B tokens (5.1), four methods on GPT2-Large / SlimPajama with 1B and
  6B tokens (5.2, Fig. 1–2), Table 1 (stale norms: MuonMax-Momo overhead vs MuonAdam 11 % → 5 %).
- No discrepancies found. Worked example (−√10 vs −4 for diag(3, 1)) checked. The ICML 2026 PMLR page the notes link
  resolves (HTTP 200); the statements were checked on the arXiv version.
- Box mapping: "One norm, a different algorithm" → key; "Do not confuse two questions" → warn; "Why 'stale' nuclear
  norms?" is a sub-heading in the PDF. Quick reference: no header row in the PDF.
- Primer (four ideas): (1) unit balls of different norms (I₂: spectral 1, Frobenius √2); (2) steepest descent and
  the dual norm: g = (3, −1) gives √10 (Euclidean) vs 4 = ‖g‖₁ with sign descent (max-norm); (3) polar factor and
  nuclear norm: [[1, 1], [0, 1]] has σ = 1.618, 0.618, polar [[0.894, 0.447], [−0.447, 0.894]], ⟨G, polar G⟩ = √5;
  Muon computes it by Newton–Schulz (cubic version checked); (4) CSD vs RSD lengths 0.1 vs 0.5 for g = (3, 4),
  η = 0.1. All in `scripts/check-math/crawshaw2026exploration.py`.

## [12] he2016deep — Deep Residual Learning for Image Recognition

- Checked against: arXiv 1512.03385 (CVPR 2016, 12 pp.; the notes link the CVF open-access copy): abstract ("100
  and 1000 layers" on CIFAR-10), Sec. 3.1–3.3 (Eqs. 1–2, zero-padding / projection shortcuts W_s), Fig. 1 (20- vs
  56-layer plain nets on CIFAR-10, training error), Sec. 4.1 (plain nets with BN: "neither forward nor backward
  signals vanish"; bottleneck 1×1–3×3–1×1 for 50/101/152 layers), Tables 3–5 (ResNet-152 single model 4.49 % top-5
  val; ensemble 3.57 % top-5 test, ILSVRC 2015 1st), Sec. 4.2 (110 and 1202 layers), COCO detection.
- No discrepancies found. The notes' "100 and even 1000 layers" is the paper's own abstract wording (the runs are
  110 and 1202 layers).
- Box mapping: "A subtle but crucial distinction" → key; "Connection to the survey" → plain; "Worked scalar
  example", "Plain versus residual comparison", "Bottleneck blocks…" are sub-headings. Quick reference: no header row.
- Primer (three ideas): (1) a deeper net can copy a shallower one, so higher training error is an optimization
  failure, not overfitting; (2) chain-rule products through depth (0.9⁵⁰ ≈ 0.005, 1.1⁵⁰ ≈ 117; 20 layers at weight
  0.1: plain 10⁻¹⁹ vs residual 1.1¹⁹ ≈ 6.1), stated as the textbook mechanism, with the paper's own caveat that its BN
  plain nets do not have vanishing signals; (3) batch normalization ((2, 4, 6) → (−1.22, 0, 1.22)) and the paper's
  "not vanishing gradients" argument. All in `scripts/check-math/he2016deep.py`.
