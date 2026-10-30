# Round 3 findings — session C

Same conventions as `research/findings.md`: what each paper actually proves (checked against the official PDF), the notes' examples reproduced, survey/notes discrepancies, and the visuals built. Numbers quoted from check scripts are in `scripts/check-math/<key>.py`.

## cohen2021gradient — Cohen, Kaur, Li, Kolter, Talwalkar, ICLR 2021 (arXiv 2103.00065) [survey 16]
- Empirical paper (no convergence theorem). Sec. 2: on a quadratic GD diverges along any eigenvector with eigenvalue a > 2/η (Prop. 1, App. B); Fig. 2 example eigenvalues 20 and 1 with η = 0.11.
- Eq. (1): maximum stable sharpness MSS_Polyak = (2 + 2β)/η (previously Goh 2017), MSS_Nesterov = (2 + 2β)/(η(1 + 2β)) (new).
- Sec. 3.1 progressive sharpening: while S < 2/η full-batch GD has an "overwhelming tendency" to increase S. Sec. 3.2 edge of stability: S hovers at or just above 2/η, loss non-monotone short-term, decreasing long-term. Sec. 4 / App.: FC, VGG, ResNet on CIFAR-10 subsets, plus Transformer (WikiText-2), deep linear net, 1-d regression; MSE and cross-entropy (with CE the sharpness drops at the end of training, App. C). Fig. 7: with momentum S plateaus at the MSS. Sec. 6: SGD does not flatline at one value.
- Notes: correct; quadratic derivation and "empirical, not a theorem" caveat match the paper.
- Survey (Sec. VII-B): consistent ("hovers at or slightly above", momentum-modified threshold).
- Visuals: (quad, 3D) Fig. 2 quadratic f = ½(20x² + y²), GD with η slider, |x_t| chart; (mlp) offline replication, tanh MLP 10–48–48–1 on 128 synthetic points, full-batch GD, sharpness by power iteration: S₀ = 4.05; η = 0.1 saturates at 7.75 < 20; η = 0.3/0.36/0.42 end at 6.66/5.55/4.75 vs 2/η = 6.67/5.56/4.76 with non-monotone loss; (mom) Eq. (1) curves vs live spectral-radius bisection.

## lewkowycz2020large — Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari, arXiv 2003.02218 (2020) [survey 17]
- Sec. 2.1 warmup model: x = 1, y = 0, f = n^{-1/2} vᵀu, L = f²/2, kernel λ = n⁻¹(‖u‖² + ‖v‖²). Eqs. (6)–(7) are exact ("no higher-order terms were neglected"): f_{t+1} = (1 − ηλ_t + η²f_t²/n) f_t, λ_{t+1} = λ_t + η f_t²/n (ηλ_t − 4).
- Phases: lazy η < η_crit = 2/λ₀ (kernel constant as n → ∞); catapult 2/λ₀ < η < η_max = 4/λ₀ (loss grows for ~log n steps, kernel drops below 2/η, converges to a flatter minimum); divergent η > 4/λ₀. Sec. 2.2 extends to the full model (Eq. 9–11) along the top kernel eigenvector.
- Experiments (MSE, SGD, wide FC ReLU nets on MNIST/CIFAR-10, WRN 28-10 on CIFAR-10): η_crit = 2/λ₀ matches; η_max ≈ c/λ₀ with c ≈ 12 for ReLU nets; best test accuracy often in the catapult phase.
- Notes: correct. Their teaching toy L(w) = ½(1 − w²)² has L'' = −2 + 6w² (checked); the notes correctly flag it as not the paper's model and that the upper edge is model-dependent.
- Survey: consistent ("4/λ₀ in their solvable model and larger for ReLU networks"). Venue: arXiv preprint (as listed).
- Visuals: (spike, 3D) n = 1 model ½(uv)² from (2, 0.35), λ₀ = 4.12, η·λ₀ slider: 2.6 → peak loss 0.484 (start 0.245), ends at λ = 1.21 < 2/η; 4.2 diverges; (width) Eqs. (6)–(7) with n slider, loss peak at steps 2/6/9 for n = 10²/10⁴/10⁶ (log n); (phases) final λ/λ₀ vs ηλ₀ for chosen n, divergence at exactly 4 (3.98 converges, 4.02 diverges).

## foret2021sharpness — Foret, Kleiner, Mobahi, Neyshabur, ICLR 2021 (arXiv 2010.01412) [survey 34]
- Eq. (1): min_w L^SAM_S(w) + λ‖w‖², L^SAM_S(w) = max_{‖ε‖_p ≤ ρ} L_S(w + ε), motivated by a PAC-Bayes bound (Theorem 1 informal, Theorem 2 in App. A). Eq. (2): first-order inner solution ε̂ = ρ sign(∇L)|∇L|^{q−1}/(‖∇L‖_q^q)^{1/p}, = ρ∇L/‖∇L‖ for p = 2. Eq. (3): ∇L^SAM ≈ ∇L|_{w+ε̂} (second-order terms dropped). Algorithm 1; ≈ 2× the gradient cost.
- Reported: error reductions on CIFAR-10/100, ImageNet, fine-tuning; label-noise robustness; m-sharpness (Sec. 4.1: per-subset inner max, smaller m generalizes better and correlates better with the gap); lower Hessian spectrum for WRN-40-10 trained with SAM (Sec. 4.2, Fig. 3).
- Notes: correct (the "max over ‖ε‖ ≤ 0.1" example gives 0.01 vs 0.10 — checked). Header carries old numbering ("Ref 100", "Paper 70"); the Atlas maps it to survey [34].
- Survey (VII-C, Eq. 20): consistent.
- Toy finding worth knowing: on a 2-well toy (sharp λ_max 39.2, flat 3.5, equal depth), descent on the *exact* SAM objective shrinks the sharp basin (25% → 17% of starts at ρ = 0.4), and the practical one-step update shrinks it more (→ 11%) but leaves 23% of starts jittering between the wells instead of reaching the flat one (e.g. from (−1.36, −0.32) at ρ = 0.3 it hovers near (−0.18, 0.14), L ≈ 0.89). The one-step approximation is not the exact minimax.
- Visuals: (objective, 3D) L and the exact SAM objective side by side with GD / one-step SAM / exact-objective descent; (basins) 41×41 start-grid fate maps for the three methods and ρ ∈ {0.1, 0.2, 0.4}.

## damian2023self — Damian, Nichani, Lee, ICLR 2023 (arXiv 2209.15594) [survey 35]
- Def. 1 sharpness; Def. 2 progressive sharpening coefficient α = −∇L·∇S, Assumption 1 α > 0; β = ‖∇S‖². Sec. 4.1 four stages; simplified dynamics Eq. (3): x_{t+1} = −(1 + ηy_t)x_t, y_{t+1} = y_t + η(α − βx_t²/2). ODE Eq. (4) X' = XY, Y' = α − βX²/2 with fixed point (δ, 0), δ = √(2α/β); Lemma 3: g = h(βX²/(2α)) + Y²/α conserved, h(z) = z − log z − 1; Corollary 1 bounds the swings.
- Main result: constrained trajectory θ†_{t+1} = proj_M(θ†_t − η∇L(θ†_t)), M = {S ≤ 2/η, ∇L·u = 0} (Eq. 2); Theorem 1 (Assumptions 1–5) predicts loss, sharpness and deviation for T = O(ε⁻¹) steps.
- App. B toy L(x, y, z) = (2/η + √β y)x²/2 − (α/√β)y − z satisfies all Sec. 4 assumptions; GD on its (x, y) part is exactly Eq. (3) (checked to 1e-13).
- Check-script finding: the discrete map (3) is not conservative — its period-2 orbit (±δ, 0) is a weakly unstable focus (two-step eigenvalues (1 − δ²) ± 2iδ, |·| ≈ 1 + δ²), so iterates follow the ODE loops for a few cycles and then spiral outward (g: 0.38 → 2.0 in 450 steps at α = 0.002). Consistent with Theorem 1 being finite-horizon. Over the shown window the mean sharpness is 1.9992 vs 2/η = 2.
- Notes: correct; they list the source as "Ref. [30]" (old numbering). The notes write the expansion as "L(θ + xu) ≈ L + x∇L·u + x²/2 S" — fine.
- Survey (VII-B, Eq. 19, Fig. 17): consistent with the paper (schematic figure).
- Visuals: (stages, 3D) the App. B toy with GD zig-zagging across the trough, |x_t| and S_t − 2/η charts, stage readout; (orbit) (|x|, S − 2/η) phase plane, ODE orbits vs GD iterates, click to restart.
