# Session C notes: stops 10 (sharpness), 11 (landscape), 12 (limits)

## Shared changes needed
- `src/lib/atlas3d.js` `Stage.stepBalls` clamps every ball step to the surface's domain (`d.clampP`). For a ball whose GD
  iterates bounce outside the drawn box this silently changes the dynamics (found in stop 10: the valley ball at η = 0.4
  ended at sharpness 2.85 instead of the true 2.54). Workaround used here: (a) slider ranges chosen so the check script
  proves every path stays inside the drawn box, or (b) the ball runs on a second, unclamped `def` with a huge domain and
  the surface's display scales copied over (`wells` in stop 10). Suggested shared fix: an opt-out flag (`clamp: false`)
  on `stage.ball`, or a check-script helper that asserts paths stay inside the box.

## Stop 10 · sharpness (`course/sharpness`, check `scripts/check-math/course_sharpness.py`)
Sections: 1 many minima / sharpness · 2 stability η < 2/S · 3 valley of minima, step size picks the flat end (catapult) ·
4 edge of stability · 5 self-stabilization · 6 flat minima and generalization (Hochreiter, Keskar, Dinh) · 7 SAM
(Foret, Agarwala) · 8 warmup (Kalra & Barkeshli 2024).

Visuals:
- `wells` (3D): f = 0.06(x² + y²) − two Gaussian wells (σ 0.22 and 0.75, amplitudes 1 and 1.045); depths −1.008 / −1.007,
  sharpness 20.8 / 1.97. 357 grid starts × 4000 GD steps, coloured by fate; η slider. Fates checked: η < 0.096 → 87 sharp /
  270 flat; 0.1–0.2 → 87 never settle (bounce inside the sharp well); 0.25–1.0 → all flat; 1.05 → none settle.
- `family` (3D + chart): ½(uv − 1)², GD from (2.5, 0.05), η ∈ [0.02, 0.46]; end sharpness 6.54 / 5.42 / 2.54 / 2.00 at
  η = 0.05 / 0.3 / 0.4 / 0.44; chart of λ_max(x_t) vs 2/η.
- `eos` (2D + chart): replays `papers/cohen2021gradient/data.json` (tanh MLP, our replication), sharpness and loss for
  η ∈ {0.1, 0.3, 0.36, 0.42}; numbers re-read from the data file by the check script.
- `stabilize` (3D + chart): Damian App. B toy (x, y part) with η = 1, α slider [0.001, 0.003]; GD = Eq. (3) exactly;
  stage readout.
- `sam` (2D contour, draggable start): GD vs unnormalized / normalized SAM on ½(uv − 1)², η = 0.3, ρ slider; SAM edge
  λ = (−1 + √(1 + 8ρ/η))/(2ρ).

Paper claims (all from `research/findings.md`):
- Cohen 2021: empirical; progressive sharpening, EoS "at or just above 2/η", momentum MSS (2 + 2β)/η (Eq. 1), SGD does not
  flatline, CE sharpness drops late (App. C); architectures FC/VGG/ResNet/Transformer. → findings `cohen2021gradient`.
- Lewkowycz 2020: lazy / catapult (2/λ₀ < η < 4/λ₀) / divergent; ReLU nets c ≈ 12; best accuracy often in catapult phase.
- Damian 2023: Eq. (3), α, β, δ = √(2α/β), four stages, Theorem 1 (Assumptions 1–5, O(ε⁻¹) steps, constrained trajectory
  with S ≤ 2/η and ∇L·u = 0); discrete toy spirals out slowly.
- Hochreiter & Schmidhuber 1997 (flat region definition, MDL, Flat Minimum Search); Keskar 2017 (SB 256, LB 10%, Adam,
  6 nets, sharper LB minima, noise explanation); Dinh 2017 (α-scale transformation, Theorem 4).
- Foret 2021 (Eq. 1–3, PAC-Bayes motivation, 2× cost, reported gains, lower Hessian spectrum); Agarwala 2023 (unnormalized
  SAM, quadratic factor 1 − ηλ(1 + ρλ), Theorem 2.2, example η 0.5 ρ 0.2 → 2.62, Atlas replication within 0.1%).
- Kalra & Barkeshli 2024 (main benefit of warmup, Eq. 1, C1/C2 depends on init/parameterization, Adam threshold
  (2 + 2β₁)/((1 − β₁)η), GI-Adam factor √(1 − β₂), "two orders" is the paper's rounding).
- Kalra 2025 and Galli 2026 only in the Atlas box (findings lines).
Unverified: none.

## Stop 11 · landscape (`course/landscape`, check `scripts/check-math/course_landscape.py`)
Sections: 1 worst case vs trained cases · 2 symmetry · 3 benign landscapes (linear nets, matrix problems) · 4 bad minima
exist (one neuron, Safran–Shamir) · 5 PL condition (derivation from the descent lemma) · 6 wide networks (Gram matrix,
Du, Allen-Zhu, Zou–Gu) · 7 lazy training / NTK (Arora, Chizat–Oyallon–Bach, Li–Ma–Zhang) · 8 mean field (Chizat–Bach,
Mei) · 9 architecture and landscape pictures (He, Li 2018, Goodfellow 2015).

Visuals (random starts and initial weights use `rng` from `src/lib/math.js`; the check script ports it to Python, so
the readouts match exactly):
- `mirror` (3D): two tanh units, L = mean (tanh(w₁x) + tanh(w₂x) − y)², targets from units 1.6 and 0.4. Exactly three
  critical points in [−0.5, 2.5]²: minima (1.6, 0.4), (0.4, 1.6), saddle (0.736, 0.736) with eigenvalues −0.106 / 0.775.
  24 random starts split 12 / 12; a start on the diagonal ends at the saddle; offset slider d.
- `benign` (3D): ½‖uuᵀ − diag(1, s)‖², s slider; for every slider value all 24 random starts reach ±(1, 0); the start
  (0, 1.2) on the saddle's stable set ends at the saddle.
- `neuron` (3D): F = (1 − vφ(w))² (Sun et al. 2020 Fig. 2), φ = −(z − 1)²/8 vs tanh; 30 starts → 21 / 9 vs 30 / 0;
  41 × 41 grid → 30.4% stuck (same as the sun2020global page).
- `pl` (2D + 2 charts): x² + 3 sin²x, μ = 0.1755, L = 8; the PL bound is checked for every slider η and starts −6…6.
- `wide` (2D + chart): live training of a one-hidden-layer tanh net (8 points, widths 4…1024, η = 1/λ_max(G₀)); weight
  movement 4.09 → 0.174, Δλ_min 96.5% → 0.9%, every run below (1 − ηλ₀/2)ᵏ L(0).
- All balls drawn on an unclamped copy of the surface def (`freeDef`), so the shown paths are the true GD paths.

Paper claims (findings entries): baldi1989neural, kawaguchi2016deep (Thm 2.3, Cor 2.4), laurent2018deep (Thm 1),
bhojanapalli2016global (noiseless all local minima global, strict saddle Thm 3.2), ge2017no, lee2016gradient (Thm 4),
sun2020global (Prop. IV.1, Fig. 2), safran2018spurious (Thm 1 ranges, Table 1, n ≥ k + 2 observed), zhang2026saddle
(Thms 1, 3), du2019gradient (Thm 5.1, 6.1, footnote), allenzhu2019convergence (Thms 1–3), zou2019improved (n⁸ vs n²⁴),
arora2019fine (Thms 4.1, 5.1), chizat2019lazy (Thms 2.2, 2.5, experiments), li2020learning (Thm 1.2), chizat2018global
(Thms 3.3/3.5, qualitative), mei2019mean (Thm 1, dimension-free), he2016deep (Table 2, BN remark), li2018visualizing
(Fig. 5 errors, slice caveat), goodfellow2015qualitatively, nguyen2017loss (N − 1, Atlas box only), danilova2022recent
(μ = 0.1755 for the same function).
Not from findings: the PL condition's history (Polyak 1963, Łojasiewicz 1963) is stated as standard background, no
Atlas link.

## Stop 12 · limits (`course/limits`, check `scripts/check-math/course_limits.py`)
Sections: 1 oracles, upper and lower bounds, Ω, Õ, worst case · 2 the zero chain (Carmon et al. Ψ, Φ, Lemmas 1–3) ·
3 scaling to Ω(ε⁻²) (derivation with a symbolic ℓ₁) and the ε^−(p+1)/p table · 4 stochastic lower bounds (Arjevani 2023
probability-p chain, heuristic balance p ≈ ε²/σ², Theorems 1–3; Arjevani 2020 elbow) · 5 NP-hardness (Murty–Kabadi,
quartic / copositivity, Blum–Rivest, Anandkumar–Ge, Arjevani 2020 footnote) · 6 the complexity ladder · 7 how to read a
paper (six questions, worked on Jin et al. 2017 Theorem 3) · 8 where to go next (links to the hub's branches
`index.html#b-III` … `#b-VII` and the glossary). Ends with the Atlas box, no next problem (last stop).

Visuals:
- `chain2` (3D): f̄₂ with GD from 0, η slider [0.01, 0.1]; amber line x₁ = ½. η = 0.05: x₂ first moves at step 8 (x₁ =
  0.554), ‖∇f̄₂‖ ≤ 1 at step 29. The JS uses a ~1e-14 erf (series + continued fraction) so paths match Python's math.erf.
- `chain` (2D bars + chart): T slider 2–16, oracle seg exact / p = 0.2 / p = 0.05 (Arjevani et al. Eq. 17, seeded rng
  1000T + 100p). Queries until ‖∇f̄_T‖ ≤ 1: T = 8: 71 / 90 / 191; T = 16: 121 / 156 / 529; exact GD ≈ 6.5 per coordinate.
- `quartic` (3D): u₁⁴ + 2q u₁²u₂² + u₂⁴, q slider; Hessian 0 for every q, local min iff q ≥ −1 (= Q copositive); ball
  leaves the square after 30 steps at q = −1.3.
- `ladder` (2D custom): 17 rows (upper bounds coral, lower bounds violet outline) grouped by oracle; row and ε sliders.
  It covers the main rates (stops 3, 4, 6, 7, 8, 11, 12) from findings; stops 4–9 written by A and B may quote further
  rates (e.g. SVRG's n^{2/3}) that the coordinator may want to add.

Paper claims (findings entries): carmon2020lower (Eq. 10, Lemmas 1–3, Theorems 1–2, sharpness for p = 1, 2),
arjevani2023lower (Eq. 17, Lemma 1, Lemma 3 variance, Theorems 1–3), arjevani2020second (O(ε⁻³) with HVPs, Ω(ε⁻³) for
p ≥ 2, elbow, Murty–Kabadi footnote), murty1987some (+ Ahmadi–Zhang restatement), blum1992training (Theorem 2, two hidden
+ one output), anandkumar2016efficient (Theorem 10), fang2018spider (Thm 1 online ε⁻³, Thm 3 finite-sum lower bound),
jin2017escape (Def. 6, Thm 3 with log⁴), jin2018accelerated (Thm 3), nesterov2006cubic + wang2020cubic (O(ε^−3/2) to an
ε-SOSP), ghadimi2013stochastic (Cor. 2.2), fang2019sharp, tripuraneni2018stochastic, allenzhu2018natasha (Cor. 4.6).
Stated without a findings line (standard): the definition of zero-respecting methods and that GD, momentum and Adam are
zero-respecting; ℓ₁ (the smoothness constant of f̄_T) is kept symbolic because findings do not record its value.
Glossary: new term `zero-chain` added to the limits block (and to course.json terms).
