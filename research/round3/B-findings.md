# Round 3 · session B findings

Format follows `research/findings.md`. Theorem numbers refer to the version named in each title line.

## nesterov2006cubic [40] — Nesterov & Polyak, Math. Program. 108 (2006); read from the open CORE DP 2003/41 preprint (UCLouvain DIAL)
- Assumption 1: ∇²f is L-Lipschitz on a convex set containing the initial level set. Step T_M(x) = global argmin of ⟨g,s⟩ + ½⟨Hs,s⟩ + (M/6)‖s‖³ (Eq. 2.4); it satisfies g + Hs + (M/2)‖s‖s = 0 (2.5) and H + (M/2)r_M I ⪰ 0 (Prop. 1).
- Lemma 1 (Eq. 2.3): |f(y) − quadratic model| ≤ (L/6)‖y − x‖³, so for M ≥ L the cubic model is an upper bound and f(T_M(x)) ≤ f̄_M(x) (Lemma 4). Scheme (3.3): M_k ∈ [L₀, 2L], monotone.
- Theorem 1: Σ r³ ≤ 12(f(x₀) − f*)/L₀ and min_{i≤k} μ_L(x_i) ≤ (8/3)(3(f(x₀) − f*)/(2kL₀))^{1/3}, i.e. min ‖∇f‖ = O(k^{−2/3}) vs O(k^{−1/2}) for the gradient method. Theorem 2: limit points satisfy ∇f = 0, ∇²f ⪰ 0. Lemma 6: the method cannot be attracted to a non-degenerate saddle or maximum, even started exactly on it. Theorem 3: local quadratic convergence near a non-degenerate minimum. Sec. 5: the subproblem is a 1-D convex problem.
- Notes: say the full article was not available ("Source blocker"); they state only the canonical model, the first-order condition and generic claims. Everything they state matches the paper. They label the paper "Ref. [32]" (old numbering) — survey number is [40].
- Survey reference text: correct (title, authors, Math. Program. 2006).
- Visuals: (1) f = x²/2 − y²/2 + y⁴/4 + 0.3xy² from (1.2, 0.03): pure Newton → saddle, CR (M = L = 10, exact subproblem) → minimum in 12 steps, GD (1/L₁, L₁ = 8) in 167; (2) the cubic model sheet at the current point for M ∈ {0, 3, 10, 25} with "take the step", plus the λ_min slice (model above f on the box iff M ≥ L there); (3) min-so-far ‖∇f‖ of both runs vs Theorem 1 and the GD bound (1.2.13) evaluated with this f's constants (both runs far below the worst case).

## zhang2021escape [82] — Zhang & Li, NeurIPS 2021 (arXiv 2111.14069)
- Setting: f ℓ-smooth, ρ-Hessian Lipschitz; ε-approximate SOSP: ‖∇f‖ ≤ ε, λ_min(∇²f) ≥ −√(ρε) (Eq. 1).
- Algorithm 1 (gradient-only Hessian power method): y₀ ~ Unif(B(r)), y_t = y_{t−1} − ‖y_{t−1}‖/(ℓr)·(∇f(x̃ + r y_{t−1}/‖y_{t−1}‖) − ∇f(x̃)). Proposition 3: if λ_min ≤ −√(ρε), w.p. ≥ 1 − δ₀ the output ê has êᵀHê ≤ −√(ρε)/4 after T = Õ(log n/√ε)-type many iterations. Lemma 6: a step of length ¼√(ε/ρ) along ±ê lowers f by ≥ (1/384)√(ε³/ρ).
- Theorem 7 (Algorithm 2, accelerated): Õ(log n/ε^1.75) iterations; Theorem 9 (Algorithm 3, stochastic): Õ(log² n/ε⁴). Table 1: the gain is in powers of log n (PGD log⁴n/ε², PAGD log⁶n/ε^1.75, PSGD log¹⁵n/ε⁴); in the stochastic setting ε⁻⁴ is worse in ε than nested-loop Neon2 (ε^−3.5) and SPIDER-type (ε⁻³).
- Fig. 1 setup reproduced: f = x₁⁴/16 − x₁²/2 + 9x₂²/8, η = 0.05, r = 0.1, M = 300. Check: 94.7% of Algorithm-1 directions within 5° of ±e₁ at t = 30; PGD at t = 90: 44.0% of paths decrease f by ≤ 0.9 (paper: "over 40%"). Our equal-budget completion of Algorithm 1 (kick of length 1/ρ = 0.267 from Lemma 6, then 60 GD steps) stalls in 0.7% (paper: "less than 5%" for its own descent procedure, not specified in detail).
- Notes: consistent with the paper (Prop. 3, Thm 9 numbering correct; worked example x₁ grows ×1.2/step with step 0.2 on −x²/2 + 2y² is right). The notes call the stochastic rate an improvement "in dimension dependence" — correct, but they do not mention that ε⁻⁴ is worse in ε than Neon2/SPIDER; the page says so.
- Survey reference text: correct.
- Visuals: (1) live replication of Fig. 1 (both clouds, alignment share over time, histogram of decrease at t = 90); (2) Table 1 orders vs dimension n for both settings.

## wang2019spiderboost [84] — Wang, Ji, Zhou, Liang, Tarokh, NeurIPS 2019 (arXiv 1810.10690)
- Assumption 1: Ψ bounded below and **each** ∇f_i L-Lipschitz. Eq. (1) is the SARAH/SPIDER recursive estimator. Algorithm 1: x_{k+1} = x_k − v_k/(2L), q = |S| = √n, output a uniformly random iterate.
- Theorem 1: E‖∇f(x_ξ)‖ ≤ ε with K = O(L(f(x₀) − f*)/ε²) iterations, SFO O(√n ε⁻² + n) (matches the SPIDER lower bound for n ≤ O(ε⁻⁴)). Theorem 2: Prox-SpiderBoost, E‖G_η(x_ξ)‖ ≤ ε with the same SFO and O(ε⁻²) prox calls. Theorem 3: Prox-SpiderBoost-M (momentum) same order; the speed-up is empirical (Sec. 5). Theorem 6/Corollary 1: online version O(ε⁻³).
- Notes: correct (example f₁,₂ = ½(x ∓ 1)² gives v = 2 exactly at x = 2; prox of 0.3|x| at 0.8 = 0.5 verified). They rightly say the paper has no saddle-escape result.
- Survey reference text: correct. Note the survey cites it in Sec. V (escape algorithms) although the paper only proves first-order stationarity — worth a word in the survey if it claims escape.
- Visuals: (1) SPIDER (normalized ε/L steps) vs SpiderBoost (1/(2L)) on a 196-component non-convex finite sum: check run reaches ‖∇f‖ < 0.05 after 2 744 vs 11 984 SFO calls; (2) Ψ = f + 0.3(|x| + |y|): prox step lands on y = 0 exactly, subgradient steps chatter at |y| ~ 10⁻².

## allenzhu2018natasha [85] — Allen-Zhu, NeurIPS 2018 (arXiv 1708.08694 v4)
- Online problem f = E_i f_i; assumptions (A1) bounded variance, (A2) L-smooth, (A4) L₂-Hessian Lipschitz. (ε, δ)-approximate local minimum: ‖∇f‖ ≤ ε, ∇²f ⪰ −δI (Def. 4.2).
- Theorem 1 (informal): Natasha1.5 finds ‖∇f‖ ≤ ε in O(ε⁻³ + σ^{1/3}ε^{−10/3}) for σ-bounded nonconvexity. Theorem 2 (informal): for δ ∈ (0, ε^{1/4}), Natasha2 finds an (ε, δ)-local minimum w.p. ≥ 2/3 in Õ(δ⁻⁵ + δ⁻¹ε⁻³) stochastic gradients + HVPs; Remark 4.4: with δ = ε^{1/4}, Õ(ε^{−3.25}) (Cor. 4.6); δ = ε^{1/2}: Õ(ε^{−3.5}) (Cor. 4.7). Sec. 4.2: the reduction (Carmon et al.) — Oja for curvature below −δ, else solve F_k = f + L·max(0, ‖x − y_k‖ − δ/L₂)² in the "safe zone". Remark 4.5: HVPs replaceable by gradient differences (Neon2).
- Notes: consistent. Their "δ⁻⁵ + δ⁻¹ε⁻³ + ε^{−13/4}" matches Remark 4.4; worked example f = x⁴/4 − x²/2 + y²/2 (Hessian at x = 0.6: diag(0.08, 1)) correct.
- Survey reference text: correct.
- Visuals: (1) the Sec. 4.2 loop on a 200-component finite sum from (0, 1.3) on the saddle's stable manifold: Oja (30 stochastic HVPs) → step δ/L₂ along negative curvature, else minibatch SGD in the safe zone (simplified: the paper uses Natasha1.5 there). Check: 20/20 seeds reach a (0.05, 0.5)-local min (median 30 rounds, 10 NC steps), never closer than 0.87 to the saddle; GD from the same start converges to the saddle. (2) Theorem 2 orders vs ε with a δ slider.

## ge2019stabilized [86] — Ge, Li, Wang, Wang, COLT 2019 (arXiv 1905.00529 v1)
- Algorithm 2 (Perturbed SVRG): SVRG + uniform perturbation in B(δ) when ‖∇f(x_sm)‖ ≤ G, then a super epoch. Theorem 2 (PMLR; Thm 1 in arXiv v1): Õ((n^{2/3}LΔ/ε² + n√ρΔ/ε^{1.5})(1 + (ρ′/(n^{1/3}ρ))²)) where ρ′ is the component Hessian-Lipschitz constant.
- Algorithm 3 (Stabilized SVRG): also subtract v_shift = ∇f(x̃), i.e. run SVRG on f̂ = f − ⟨∇f(x̃), x − x̃⟩, with a much smaller perturbation. Theorem 3 (PMLR; Thm 2 in arXiv v1): Õ(n^{2/3}LΔ/ε² + n√ρΔ/ε^{1.5}), only polylog in ρ′, b = Õ(n^{2/3}), m = n/b, η = Õ(1/L). Sec. 4.3: Phase 1 stays within Õ(δ) of x̃ (PMLR Lemma 12; arXiv Lemma 9) while the projection on the negative eigenspace reaches δ/10 (Lemma 11; arXiv 8); Lemma 10 (arXiv 7): decrease Ω̃(γ³/ρ²) w.p. ≥ 1/8.
- Theorem numbering: COLT/PMLR version has Perturbed SVRG = Theorem 2 and Stabilized SVRG = Theorem 3 (arXiv v1: Theorems 1 and 2). The notes' "Theorem 3" follows PMLR — correct; the page uses PMLR numbers. Worked example (shifting f = x + 2x² − y²/2 at 0) correct.
- Survey reference text: correct.
- Visuals: (1) 125 components with zero-mean cubic terms (ρ′ ≈ 25 vs ρ = 9); same perturbation, Perturbed vs Stabilized SVRG: median ‖x_t − x̃‖ at t = 1/(ηγ) = 10 is 0.041 vs 0.020 (= δ) over 200 seeds; both escape on this easy instance (the benefit is in the variance analysis, stated on the page); (2) Table 1 orders vs n.

## li2019ssrgd [87] — Z. Li, NeurIPS 2019 (arXiv 1904.09265)
- Algorithm 1: SARAH/SPIDER estimator with full gradient every m steps, plus uniform perturbation (radius r) when ‖∇f(x_sm)‖ ≤ g_thres outside a super epoch. Theorem 1: η ≤ (√5 − 1)/(2L), m = b = √n → O(n + LΔ√n/ε²) (first order). Theorem 2: Õ(LΔ√n/ε² + Lρ²Δ√n/δ⁴ + ρ²Δn/δ³) for (ε, δ)-SOSP. Theorems 3–4: online versions (O(σ²/ε² + LΔσ/ε³) first order). Lower bound Ω(√n/ε²) (Fang et al.) → near-optimal.
- Notes: correct (Theorems 1–2 match; both worked examples verified: x₁ = 1.8 recursion; x₁₀ = 0.00349, y₁₀ = 0.02594).
- Survey reference text: correct.
- Visuals: (1) start (0, 1.3) on the stable manifold: plain SARAH keeps x = 0 exactly and converges to the saddle; SSRGD reaches (±1, 0) in 50/50 seeds (super-epoch stop simplified to distance/time); (2) √n vs n^{2/3} orders with Fang's lower bound.

## chen2022faster [88] — Chen, Zhou, Gu, ALT 2022 (PMLR 167; arXiv 2110.13144)
- Notation: (ε, ε_H)-local minimum ‖∇F‖₂ ≤ ε, λ_min ≥ −ε_H. Algorithm 1 (LENA): GD phase with normalized steps η/‖d_t‖ while ‖d_t‖ > ε; escape phase: perturb, η_H steps, and if Σ_{i=m_s}^t η_i²‖d_i‖² > (t − m_s + 1)D shrink the last step to equality and restart; if ℓ_thres steps pass, output x_{m_s}. Eq. (4.1): ‖d_t − ∇F(x_t)‖² = Õ(Σ‖x_{i+1} − x_i‖²) for SPIDER/STORM — why movement is capped. Theorem 5.1 (SPIDER) and 5.3 (STORM): Õ(σLΔε⁻³ + σρ³LΔε_H⁻⁶); Remark 5.2: Õ(ε⁻³) for ε_H = √ε vs Õ(ε^{−3.5}) for SSRGD; lower bound Ω(ε⁻³ + ε_H⁻⁵) (Arjevani et al. 2020) — optimal in ε, not in ε_H.
- Notes: correct; the shrinkage example √(0.003 − 0.03² − 0.04²) = 0.0224 is an analogy (they say so).
- Survey reference text: venue "ALT" correct (PMLR 167, 2022).
- Visuals: (1) LENA-SPIDER, online noise that vanishes in x on the stable manifold (so only the kick starts the escape); budget chart and ‖d_t‖ vs ‖∇F‖. Check: 30/30 seeds output ‖∇F‖ ≤ 0.1 next to a minimum after a median of one shrink; ℓ_thres = 120 (60 failed in 3/30 runs because the kick was too small — the theorem's log factors matter). (2) ε⁻³ + ε_H⁻⁶ vs lower bound and SSRGD with an ε_H = ε^p slider.

## staib2019escaping [89] — Staib, Reddi, Kale, Kumar, Sra, ICML 2019 (arXiv 1901.09149)
- Eq. (1–3): idealized RMSProp x_{t+1} = x_t − ηA_tg_t, A = E[ggᵀ]^{−1/2}; noise ξ = A(g − ∇f) has Cov(ξ) = I − G^{−1/2}∇∇ᵀG^{−1/2} ≈ I near stationary points. Definition 4.2 ((Λ₁, Λ₂, Γ, ν, λ₋)-preconditioner). Theorem 4.1: EMA estimation error with β = 1 − Cη^{2/3} and burn-in W = O(η^{−2/3}). Theorem 4.3: first-order rate with an inexact preconditioner. Theorem 4.4 (Algorithm 4, occasional large step r): (τ, √(ρτ))-stationary point in Õ(τ⁻⁵); Corollary 4.2: same for RMSProp (Appendix B).
- Sec. 5 experiment replicated exactly: f_i = ½xᵀdiag(1, −0.1)x + b_iᵀx + ‖x‖₁₀¹⁰, Cov(b) = diag(1, 0.01), b_i = (±1, ±0.1). Check (20 seeds, 10⁴ steps from the saddle): median steps to |x₂| > 0.4: RMSProp(0.001) 2 994, SGD(0.001) never, SGD(0.01) 2 846; tail mean f: −0.0124 (RMSProp) vs −0.0101 (SGD 0.01) — the paper's Fig. 1 conclusion.
- Notes: correct (Definition 4.2, Theorems 4.3–4.4, Corollary 4.2, Õ(ε⁻⁵) all match; whitening example diag(9,1) verified).
- Survey reference text: correct.
- Visuals: (1) live replication of the experiment with the check-script medians as dashed curves; (2) covariance ellipses of raw vs preconditioned noise at (0, s) from Eq. (3).

## chizat2018global [32] — Chizat & Bach, NeurIPS 2018 (arXiv 1805.09545)
- Problem (1): min over signed measures J(µ) = R(∫φ dµ) + G(µ) (convex in µ); particle discretization J_m is non-convex. Gradient descent on particles ↔ Wasserstein gradient flow as m → ∞ (Theorem 2.6). Theorem 3.3 (2-homogeneous φ, e.g. ReLU) / 3.5 (partially 1-homogeneous, e.g. sigmoid): if the initial support separates the inner and outer spheres (resp. {−r₀}×Θ from {r₀}×Θ) and the flow converges, the limit is a global minimizer; lim_{t,m→∞} J(µ_{m,t}) = min J. Qualitative only (no rates). Prop. 4.3: for ReLU the classical parameterization has discontinuous dΦ; the theorem uses a "signed square" reparameterization.
- Fig. 2: ReLU net, d = 2, Gaussian features, teacher with m₀ = 4 neurons, particles start on a small sphere: failure with 5, success with 10 and 100. Fig. 3: particle complexity only slightly above m₀ vs exponential for convex optimization on a fixed grid.
- Our replication (fixed teacher, 600 samples, step 0.4m per particle, 5 000 steps): m = 5 and m = 10 stall at loss ≈ 3×10⁻⁵ (a plateau: unchanged after 20 000 steps), m = 20 and m = 100 reach ≤ 10⁻⁶ and keep decreasing. Same phenomenon, different threshold (the paper's 10-particle success is instance-dependent) — stated on the page.
- Notes: correct and careful (they say the result is qualitative, many-particle). They label the paper "Ref. [29]" (old numbering) — survey number [32].
- Survey reference text: correct.
- Visuals: (1) target surface vs the student's output sheet replayed during training, m ∈ {5, 10, 20, 100}; (2) the Fig. 2 view, trajectories of |w_j|θ_j with target directions.

## zou2019improved [90] — Zou & Gu, NeurIPS 2019 (arXiv 1906.04688)
- Assumptions 3.1 (‖x_i‖ = 1, (x_i)_d = µ > 0) and 3.2 (‖x_i − x_j‖ ≥ φ). Algorithm 1: hidden rows N(0, 2/m I), output rows N(0, I/d). Theorem 3.3 (GD, L hidden layers): m = Ω(kn⁸L¹² log³m/φ⁴), η = O(k/(L²m)) → ε training loss in O(n²L² log(1/ε)/φ) iterations. Remark 3.4: vs Allen-Zhu et al. Ω̃(kn²⁴L¹²/φ⁸), O(n⁶L² log(1/ε)/φ²). Corollary 3.5: two-layer, m = Ω(kn⁸ log³m/φ⁴). Theorem 3.8 (SGD): m = Ω(kn¹⁷L¹² log³m/(B⁴φ⁸)), T = O(n⁵ log m log²(1/ε)/(Bφ²)). Theorem 3.10: two-layer SGD.
- Proof ingredients: Lemma 4.1 (gradient lower bound ‖∇L‖²_F ≥ Ω(mφL(W)/(kn²)) in a τ-ball), Eq. (4.1)/Lemma 4.2 (telescoping trajectory length, ‖W_t − W_0‖₂ ≤ O(√(kn² log n/(mφ)))), Lemma 4.4 (semi-smoothness, from Allen-Zhu et al.).
- Notes: correct (n⁸ vs n²⁴ width, Theorems 3.3/3.8/3.10; telescoping example). Minor: notes say "Theorem 3.10 gives tighter special-case bounds for two-layer networks" — for GD the two-layer case is Corollary 3.5.
- Survey reference text: correct.
- Visuals (one-hidden-layer net, n = 10, d = 10, η = 1/λ_max(Gram) ∝ 1/m as in the theorem): (1) real loss slice along the GD displacement and a random direction — convex along the training direction at m = 8 192, not at m = 32; (2) linear convergence for widths 32…8 192 and the relative spectral movement 1.83 → 0.10 and activation flips 13.4% → 2.0% (≈ 1/√m); (3) width-condition orders vs n.

## arora2019fine [91] — Arora, Du, Hu, Li, Wang, ICML 2019 (arXiv 1901.08584)
- Model: f(x) = m^{−1/2} Σ a_r relu(w_rᵀx), w_r ~ N(0, κ²I), a_r = ±1 fixed; H∞_ij = E[x_iᵀx_j 1{wᵀx_i ≥ 0, wᵀx_j ≥ 0}] (closed form x_iᵀx_j(π − arccos x_iᵀx_j)/(2π) for unit x). Theorem 4.1: ‖y − u(k)‖₂ = √(Σ(1 − ηλ_i)^{2k}(v_iᵀy)²) ± ε (small κ, large m, η = O(λ₀/n²)). Theorem 5.1 (informal Thm 3.3): population loss ≤ √(2yᵀ(H∞)⁻¹y/n) + lower-order terms, width-independent. Sec. 6: provably learnable function classes.
- Check: closed form vs Monte Carlo 3×10⁻³; n = 60, d = 10, m = 8 000, κ = 1 (we keep u(0) in the prediction because κ is not small): GD residual within 0.021√n of the spectral prediction; structured labels sign(x₁) vs random: yᵀ(H∞)⁻¹y 99.1 vs 351.5, top-10 energy 76% vs 12%, loss at k = 100 0.005 vs 0.023.
- Notes: correct (Theorems 4.1, 5.1; example H = diag(2, 0.2), η = 0.25 → factors 0.5, 0.95, yᵀH⁻¹y = 5.5 verified).
- Survey reference text: correct.
- Visuals: (1) real loss on a weight-space slice spanned by the top/bottom kernel directions (elongated valley, GD on the slice); (2) actual vs predicted loss curves for structured/random labels and the label energy per eigenvalue.

## nguyen2021proof [92] — Q. Nguyen, ICML 2021 (arXiv 2101.09612)
- Theorem 2.2: deep ReLU net (standard parameterization, all layers trained by GD, square loss) with last hidden width n_{L−1} ≥ N. With α₀ = σ_min(F⁰_{L−1}), λ̄_l = ‖W⁰_l‖₂ + C_l: if conditions (7)–(9) hold and η satisfies (10), then Φ(θ_k) ≤ (1 − ηα₀²/8)^k Φ(θ₀) (11). Proof tracks only F_{L−1} via the Lipschitz property of ReLU (no NTK / activation-pattern tracking).
- Sec. 3.1: W_L = 0 and the other layers scaled by large β satisfy (7)–(9) → width N suffices. Sec. 3.2: LeCun init, two layers, width Ω(N²). Corollary 3.2 (Sec. 3.3): LeCun-type init, one layer of width Ω̃(N³ max(m, n₀)³), others Θ(m). Lemma 3.3: σ_min(F_{L−1})² = Θ(n₀ Π n_l β_l²).
- Check (L = 3, widths 4→8→3→1, N = 3): smallest β meeting (7)–(9) is ≈ 212, α₀ ≈ 6 685, and (10) then forces η ≈ 1.7·10⁻¹⁶ — guaranteed rate ≈ 10⁻⁹ per step. Eq. (11) and σ_min ≥ α₀/2 hold at all 4 000 steps. An ordinary run (β = 1, step 0.23) reaches 3·10⁻²⁹ with σ_min growing 0.149 → 0.694 (empirical, no guarantee). Worth stating in the survey if it describes the result as practical: it is an existence/width result.
- Notes: correct (worked example verified). They call the result "Theorem 2.2" correctly.
- Survey reference text: correct.
- Visuals: (1) last-layer least-squares slice with features frozen at a chosen training step (strictly convex bowl, GD on it); (2) theorem run vs bound (11) and ordinary run; σ_min/α₀ for both.

## mei2019mean [93] — Mei, Misiakiewicz, Montanari, COLT 2019 (PMLR 99; arXiv 1902.06015)
- Theorem 1 (A) (noiseless SGD, fixed coefficients): sup_k |R_N(θ^k) − R(ρ_{kε})| ≤ Ke^{KT}(√log N + z)/√N + Ke^{KT}(√(D + log N) + z)√ε (Eq. 9) — the N requirement is dimension-free; D enters only via ε ≪ 1/D. (B): general coefficients, e^{KT³}. Noisy SGD: arXiv Theorem 2 = COLT Theorem 4; part (B) is not dimension-free (arXiv Remark 3.3 = COLT Remark 5). Anisotropic-Gaussian example: arXiv Theorem 3 = COLT Theorem 6 (N = O(1) instead of O(d)). Kernel limit: COLT Theorem 8.
- **Notes error:** they attribute the noiseless approximation bound to "Theorem 3"; it is Theorem 1 in both versions (their "Theorem 4 … Remark 5" for the noisy case follows the COLT numbering).
- Our replication of the Sec. 3.2 example (Δ = 0.6, truncated ReLU 0.5/1.5/−2.5/7.5 — constants are ours, ε = 0.05/d): sup-gap to an N = 3 200 proxy 0.240 → 0.106 → 0.115 (N = 50, 200, 800) at d = 20 and 0.202 → 0.065 → 0.038 at d = 80 — no growth with d; the plateau at d = 20 is the √ε term plus proxy noise. Closed-form R∞(r₁, r₂) verified against a 1 500-neuron network at d = 400 and its d → ∞ limit.
- Survey reference text: correct.
- Visuals: (1) neuron cloud (‖P_V w‖, ‖P_⊥ w‖) replayed on the R∞ surface (ends near its minimum (1.55, 0.70)); (2) risk curves for N = 50/200/800 vs the proxy and the gap vs N for both d.

## lu2020mean [94] — Lu, Ma, Lu, Lu, Ying, ICML 2020 (PMLR 119; arXiv 2003.05508)
- Model (2.1)–(2.2): Ẋ_ρ(x, t) = ∫ f(X_ρ(x, t), θ) ρ(θ, t) dθ, X(0) = w₂x, E(ρ) = E½(⟨w₁, X(1)⟩ − y)². ICML numbering: Theorem 1 (well-posedness, Lipschitz in W₂), Theorem 2 (first variation δE/δρ = E f(X, θ) p(x, t), adjoint p), Lemma 1 (‖p(·, t)‖ ≥ e^{−(C₁+C₂r)}E(ρ)), Theorem 3 (E > 0 ⇒ descent direction ⇒ every local minimizer global), Theorem 4 (informal; positively p-homogeneous blocks, if the Wasserstein flow converges and the support separates two spheres, the limit has E = 0). arXiv numbers: Thm 2.3, 3.1, 3.2, 3.9.
- Notes: consistent with ICML numbering (Theorems 2–4); Euler example (1 + 1/L)^L verified.
- Our model: L = 16 layers × M = 32 tanh blocks, 1/(LM) scaling, 1-D input lifted to ℝ², target sin(πx), GD on all blocks: loss 0.107 → 3.1·10⁻⁵; adjoint ratio ‖p(t)‖/‖p(1)‖ ≥ 0.34 over depth and training; L = 4/16/64 give nearly the same loss curves (depth limit).
- Survey reference text: correct.
- Visuals: (1) forward flow of the data through depth (3-D curves) at a chosen training step; (2) loss vs depth L and the adjoint profile.

## li2020learning [95] — Li, Ma, Zhang, COLT 2020 (PMLR 125; arXiv 2007.04596)
- Setup: x ~ N(0, I), f*(x) = Σ aᵢ|w*ᵢᵀx|, orthonormal W*, aᵢ ∈ [1/(κd), κ/d]; student f_W = (1/m)Σ‖wᵢ‖ReLU(wᵢᵀx) (Eq. 1.2); Algorithm 1 = truncated GD in two stages (λ₀, λ₁). Theorem 1.1: L(Ŵ) ≤ O(1/d^{1+Q}) with poly samples/width/time. Theorem 1.2: any poly(d)-size feature map or kernel (NTK included) has L = Ω(1/d). Eq. (2.1): Hermite expansion into tensor-decomposition terms.
- Notes: correct in substance; they call the lower bound "Theorem 2" (paper: Theorem 1.2). Survey title matches the COLT title ("…Two-Layer Neural Networks beyond NTK"; arXiv adds "ReLU").
- Our d = 3 replication (exact population loss via the arc-cosine kernel, m = 60, truncated GD): loss 0.234 → 3.4·10⁻⁵, neuron mass within 15° of ±eᵢ 4.4% → 97.0%; best fit on the 60 frozen initial features 1.2·10⁻³ (35× worse; crossed after ≈1 240 steps). At d = 8 and 16 (m = 20d, 3 000 steps) the trained loss sat at the random-feature level — the theorem's second stage needs far longer; the page says the d = 3 run is an illustration, not the asymptotic theorem.
- Survey reference text: correct.
- Visuals: (1) neurons in ℝ³ gathering on the six teacher rays; (2) trained loss vs the same-width fixed-feature optimum and the alignment share.

## xu2023over [96] — Xu & Du, COLT 2023 (arXiv 2302.10034)
- Theorem 1 (informal): n ≥ 2 students, d = Ω(log(n/δ)), small init, suitable η ⇒ GD converges globally with L ≤ poly(n, ‖v‖, η⁻¹)·T⁻³. Theorem 2 (informal): gradient flow L(w(t)) ≥ (Γ₁t + Γ₂)⁻³. n = 1: exp(−Ω(T)) (Yehudai & Shamir 2020). Eq. (3): closed-form gradient (Safran & Shamir) — note the w̄ᵢ (normalized) in the sin term.
- Fig. 1 setting reproduced (σ = 0.1, η = 0.05, ‖v‖ = 1, d = 10 our choice): n = 1 reaches L < 10⁻²⁸ in 750 steps; fitted slopes −2.99/−2.99/−2.98 for n = 2/3/5; log L/log(ηT) → −2.96/−3.01/−3.10 at T = 6·10⁴. Mechanism checked on a 2-D slice: two students summing to v have loss ∝ |b|^{3.000} across the minimum segment, and GD there decays as t^{−3.00}.
- Notes: correct (Theorems 1–2, worked example (0.4, 0.6) split verified).
- Survey reference text: correct.
- Visuals: (1) side-by-side n = 1 bowl vs n = 2 cubic trough with GD; (2) Fig. 1 replication and the −3 ratio.

## chizat2019lazy [110] — Chizat, Oyallon, Bach, NeurIPS 2019 (arXiv 1812.07956)
- κ_h(w₀) (Eq. 1) controls laziness; scaling α: κ_{αh} ∝ 1/α; q-homogeneous models: init scale λ acts like α = λ^q. Theorem 2.2 (h(w₀) = 0, horizon T): ‖w_α − w₀‖ = O(1/α), ‖w_α − w̄_α‖ = O(1/α²), outputs O(1/α). Theorem 2.3: square-loss version without exponential T dependence. Theorem 2.4: over-parameterized, linear convergence for all t. Theorem 2.5: under-parameterized, convergence to a local (not necessarily global) minimum. Experiments: lazy regime generalizes worse (two-layer nets, CIFAR-10 CNNs).
- Check: toy h(w) = w₁ + ½w₂ + w₁w₂: slopes −0.96 and −1.97 for the two Theorem 2.2 quantities over α = 2…64. Fig. 1 setting (n = 15, m = 20, d = 2, symmetrized init; τ values ours): τ = 0.1 fits with relative weight change 3.12, τ = 5 with 0.008; change ∝ τ^{−1.94} for τ ≥ 1. (First attempt used a step too large for large τ and blew up early — fixed with η = 1/λ_max of the tangent kernel.)
- Notes: correct (Theorem 2.2 statement; example h = w + w², α = 100 verified).
- Survey reference text: correct.
- Visuals: (1) model vs linearization surfaces on a 1/α window with GD on each and the gap ×α²; (2) Fig. 1(a)–(b) replication and the τ⁻² scan.
