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
