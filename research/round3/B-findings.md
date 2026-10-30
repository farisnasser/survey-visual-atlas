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
