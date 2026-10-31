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
