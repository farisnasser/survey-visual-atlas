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
