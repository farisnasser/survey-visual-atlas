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
