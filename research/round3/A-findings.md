# Round 3 — session A findings

Same style as `research/findings.md`: what each paper proves (checked against the PDF), the toy models on its page, and what the survey or the notes get wrong. Page/theorem numbers refer to the PDF named in each heading.

## haeffele2017global [54] — Haeffele & Vidal, CVPR 2017 (openaccess.thecvf.com) + long version arXiv 1506.07540
- Setting (Eq. 17): $f_r(W)=\ell(Y,\sum_{i\le r}\phi(W_i))+\lambda\sum_i\theta(W_i)$, $\ell$ convex and once differentiable, $\lambda>0$, $(\phi,\theta)$ a nondegenerate pair: both positively homogeneous of the same degree $p$.
- Proposition 1: $\Omega_{\phi,\theta}$ (Eq. 15) is convex, positively homogeneous of degree 1; the infimum is attained with $r\le\mathrm{card}(X)$.
- Theorem 1: any local minimizer with some subnetwork $W_{i_0}=0$ is a global minimizer of $f_r$ and its output solves the convex problem (16). **No size condition.**
- Corollary 1: a local minimizer is global if appending a zero subnetwork gives a local minimizer of $f_{r+1}$.
- Theorem 2: if $r>\mathrm{card}(X)$, from any point with finite objective there is a non-increasing path to a global minimizer (constructive meta-algorithm).
- Degree mismatch: Prop. 19 of arXiv 1506.07540 (the notes cite "Proposition 2 of the CVPR supplement"; that supplement URL now returns 404, so we checked the long version): degree $p'<p$ regularizer ⇒ the origin is always a local minimum.
- Survey [54] text: "if the network is large enough, a local minimum in which some subnetwork is identically zero is a global minimum". Imprecise: Theorem 1 needs no size condition; size ($r>\mathrm{card}(X)$) enters only Theorem 2 (the non-increasing path) and, in practice, the chance that some subnetwork is zero. The survey also omits that the regularizer ($\lambda>0$, same degree) is essential.
- Notes: header says "Ref. [47]" (older numbering; survey number is [54]). Content correct.
- Visuals: (1) matched toy $\tfrac12(1-uv)^2+\tfrac\lambda2(u^2+v^2)$, GD; origin strict saddle with $\lambda_{\min}=\lambda-1$, all 200 runs reach $f^\star=\lambda-\lambda^2/2$. (2) mismatch toy $\tfrac12(1-uv)^2+\lambda(|u|+|v|)$, proximal gradient; origin is a spurious local min (39/400 runs trapped at λ = 0.15). (3) offline: one-hidden-layer ReLU nets with weight decay (bias inside $w$), r ∈ {1,…,12}, 16 seeds; every run with a zero neuron that passes the local-min test $\max_{\|w\|=1}|\langle\rho,\mathrm{ReLU}(Xw)\rangle|/n\le\lambda$ ends at the same value 0.0143253 (none lower); 81 runs with dead zero neurons fail the test and sit higher — Theorem 1 needs a true local minimum, not just a zero neuron.

## islamov2024loss [57] — Islamov, Ajroldi, Orvieto, Lucchi, NeurIPS 2024 (arXiv 2410.12455)
- Definition 1 (Eq. 2), α-β-condition: for every x some $x_p\in\mathrm{Proj}(x,S)$ with $\langle\nabla f_i(x),x-x_p\rangle\ge\alpha(f_i(x)-f_i(x_p))-\beta(f_i(x)-f_i^\ast)$ for all i, $\alpha>\beta>0$. Not unique: $(\alpha+\delta,\beta+\delta)$ also works.
- Examples 1–3 (Eqs. 3–6): unbounded minimizer set, a saddle (Example 2, β = α − 8), local minima. Examples 4–5: matrix factorization, two-layer ReLU net with logistic loss + weight decay on bounded sets.
- Theorem 1 (SGD, $\gamma\le(\alpha-\beta)/2L$, Assumptions 1–2): Eq. (9), $O(K^{-1/2})$ up to a ball $O(\beta\sigma^2_{\rm int})$. Theorems 2–3: SPSmax, NGN. Sec. 5 checks Eq. (2) along training runs with the final iterate standing in for $x_p$.
- Our check of Example 2: minimizers ≈ (0.00067, 0.00067) and mirror, f* = 0.49983, strict saddle at (1,1) (eigs −0.812, 0.271). The condition does hold, but needs α ≥ 8.8×10⁶ on [−1.5, 3.5]² because $f_1(x_p)\approx9\times10^{-7}$; Theorem 1's floor is then ≈ 1.1×10⁶ ≫ max(f − f*) = 0.5, for any split α − β. So the paper's own saddle example only gets a vacuous rate (the paper says it does not aim for tight constants). Worth one sentence in the survey: "convergence guarantees" are up to a neighbourhood whose size can be large.
- Survey text: accurate in substance; "convergence guarantees" should read "convergence up to a neighbourhood of size O(βσ²_int)".
- Notes: header "Ref. [50]" (older numbering; survey [57]). Content correct.
- Visuals: (1) Example 2 surface with SGD (pick one $f_i$ per step, step 0.5), saddle eigen-lines; (2) heat map of the α needed per point + PL ratio along the diagonal (zero at the saddle).

## bhojanapalli2016dropping [59] — Bhojanapalli, Kyrillidis, Sanghavi, COLT 2016 (arXiv 1509.03917)
- Algorithm 1 (FGD): $U^+=U-\eta\nabla f(UU^\top)U$ (factor 2 absorbed), step (8) $\eta=1/(16(M\|X_0\|_2+\|\nabla f(X_0)\|_2))$.
- Assumptions (A1)/(A2): start within $\rho\,\sigma_r(U^\star_r)$ with $\rho=\frac{1}{100}\frac{\sigma_r}{\sigma_1}$ (smooth) or $\frac{1}{100\kappa}\frac{\sigma_r}{\sigma_1}$ (strongly convex); (A3) approximate low rank.
- Theorem 4.1 (smooth): $f(X_k)-f(X^\star_r)=O(1/k)$ (Eq. 9). Theorem 4.2 (restricted strongly convex): Eq. (10), factor $\alpha=1-m\sigma_r(X^\star)/(64(M\|X^\star\|_2+\|\nabla f(X^\star)\|_2))$. Corollary 4.3: linear convergence when rank $X^\star=r$. Theorem 5.2: initialization $X_0\propto P_+(-\nabla f(0))$.
- The paper's Sec. 3 toy $\|uu^\top-Y\|_F^2$, $Y=\alpha^2v_1v_1^\top-\beta^2v_2v_2^\top$, is on the page. Our checks: Hessian at 0 = diag(−4a², 4b²) (saddle created by the factorization), at $U^\star$ = diag(8a², 4a²+4b²) ⇒ constant steps need $\eta<1/(2a^2)$; from the paper's example point $(a/2,b/10)$, $\eta=1/M$ diverges for a ≥ 1.5 while step (8) converges (209 steps at a = 2). From an (A2) start at a = 2: bound factor 0.9875, measured 0.880.
- Survey text: accurate.
- Notes: header "Ref. [52]" (older numbering). Content correct.
- Visuals: 3D surface of the toy with FGD balls (step (8) vs 1/M) and a slider for a; 2D Dist² vs k against the Cor. 4.3 bound.

## anandkumar2016efficient [61] — Anandkumar & Ge, COLT 2016 (arXiv 1602.05908)
- Definition 1: p-th order local minimum ($f(y)\ge f(x)-C\|y-x\|^{p+1}$ near x). Definition 4: third-order necessary condition (∇f = 0, ∇²f ⪰ 0, $\nabla^3f[u,u,u]=0$ on the Hessian null space), checkable in polynomial time (Claim 1).
- Algorithm: cubic regularization (Nesterov–Polyak) alternated with a third-order step in a "competitive subspace" (Def. 6). Theorem 9: converges to a point satisfying Def. 4; Theorem 8 / Corollary 1: approximate version in polynomial time. Theorem 10: finding a fourth-order local minimum is NP-hard (via Theorem 11, non-negativity of quartic forms).
- Survey text: accurate. "Certifying fourth-order local optimality is NP-hard [61]" (Sec. IX) is a fair paraphrase (the paper: "we cannot even verify whether a point is a fourth order local minimum").
- Notes: header "Ref. [54]" (older numbering). Their toys ($x^2-y^4$, $x^2+y^3$, deep-linear $\frac12(1-abc)^2$) are correct and labelled as teaching examples.
- Visuals (our toys, bounded below): $f_3=x^2+y^3+y^4$ (second- but not third-order min at 0; GD stalls there with $y_k\approx1/(3\eta k)$, GD + a simplified third-order step escapes to (0, −3/4)); $f_4=x^2-y^4+y^6$ (passes Def. 4 at 0 but not fourth order; GD escape time ≈ 1/(8ηy₀²): 1013/256/63 steps for y₀ = 0.05/0.1/0.2).

## goodfellow2015qualitatively [63] — Goodfellow, Vinyals, Saxe, ICLR 2015 (arXiv 1412.6544)
- Method (Sec. 2): $\ell(\alpha)=J((1-\alpha)\theta_0+\alpha\theta_f)$. Findings: smooth decrease for seven models (Secs. 3–4: FC sigmoid/ReLU/maxout, adversarial maxout, conv maxout on CIFAR-10, MP-DBM, LSTM). The paper's own exceptions: a shallow barrier right after $\theta_0$ (Fig. 6, conv maxout) and a secondary high-error minimum on the MP-DBM line (Fig. 7). Fig. 5: a barrier between two SGD solutions. Sec. 5 / Fig. 10: same picture analytically for deep linear regression. Fig. 14: factored model $(1-w_1w_2)^2$.
- Survey text ("often decreases monotonically, with no visible barrier"): accurate, "often" is the right word.
- Notes: header "Ref. [56]" (older numbering). Content correct; the notes' warning ("not that SGD encounters no saddles") is right.
- Our own MLP check (2-16-16-1 tanh, two spirals, SGD): run 2 monotone; run 1 has a bump of 0.111 at α = 0.25 (17% of the drop) despite 100% train accuracy; in a 24-run sweep 19 were monotone within 1% of the drop. Between two solutions: barrier 1.83. So monotone interpolation is common but not universal (consistent with later work, e.g. Lucas et al. 2021).
- Visuals: (1) factored model surface, SGD from near the saddle, start→solution line (monotone, computed live) vs the (1,1)→(−1,−1) line over the saddle; (2) offline MLP interpolation curves + projection/residual of the SGD path.

## apidopoulos2026taming [43] — Apidopoulos, Lytras, Mertikopoulos, ICML 2026 (PMLR 306:3194–3227)
- Assumption 1: C², gradient-coercive. Assumption 2: (L0, L1)-smoothness $\|\nabla^2f\|\le L_0+L_1\|\nabla f\|$. Assumption 3: sub-Gaussian noise with state-dependent proxy σ(x). Assumption 4 (ABC_σ): $\sigma^2\le A(f-f^\ast)+B\|\nabla f\|^2+C$.
- (TSGD) $x_{t+1}=x_t-\gamma_t\hat g_t/(1+\lambda_tr(x_t))$, deterministic scale $r$ (Eq. 12). Theorem 1: $\sum\gamma=\infty$, $\sum\gamma^2/\lambda<\infty$, $\gamma/\lambda\to0$ ⇒ ∇f(x_t) → 0 a.s. (DTSGD) adds dissipation; Theorem 2 without Assumption 4. Theorem 3 (with Assumption 5: uniform excitation, bounded noise near crit f): strict saddle manifolds are avoided a.s.; parameter regions (25a)/(25b). Experiments: phase retrieval, deep diagonal net (Figs. 1–2).
- Survey text: accurate. Small omissions: two schemes (TSGD needs the ABC_σ noise bound, DTSGD does not), plus gradient coercivity.
- Notes: header "Ref. [35]" (older numbering). Their "teaching" update $\eta_t/(1+\eta_t\|g_t\|)$ normalizes by the random $\|g_t\|$, which the paper deliberately avoids (Remark 2: biased); the notes do say it is not the paper's update. Fine as a caveat.
- Our toy: $f=x^4/4-x^2/2+y^2/2+y^4/4$ ((L0,L1) with L0 = 4, L1 = 3; not globally Lipschitz; coercive). With p = 0.8, q = 0.5, r = 1 + ‖x‖³, σ = 0.3: plain SGD diverges in 25%/64%/100% of runs at γ₀ = 0.3/1/3; TSGD and clipped SGD stay ≥ 98% at a minimum up to γ₀ = 10; TSGD from the stable manifold x = 0 always leaves the saddle.
- Visuals: 3D tamed vs plain SGD from far starts + a start on the stable manifold; 2D success rate vs γ₀ and the (p, q) region (25a).

## panageas2019first [64] — Panageas, Piliouras, Wang, NeurIPS 2019 (arXiv 1906.07772v3, Sep 2025)
- Theorem 4.1: stable-manifold theorem for time non-homogeneous systems. Corollary 4.2/4.3: $\alpha_k\to0$, $\alpha_k\in\Omega(1/k)$ ⇒ the stable set of a strict saddle has measure zero. Theorems 5.1/5.3/5.4/5.5–5.6: GD, mirror descent, proximal point, manifold GD. Remark 3.1 + Fig. 1: summable steps can converge to a non-critical point (Fig. 1: $x^2-y^2$ with $1/\sqrt k$, $1/k$, $1/k^4$).
- **Correction in arXiv v3**: the acknowledgement says the proof of the main theorem requires the extra assumption that the Hessian at the saddle is invertible (Sec. 2 and Theorem 5.1 now assume it). An invertible Hessian rules out non-isolated saddle manifolds (they have a zero eigenvalue along the manifold), so Theorem 1.1's "isolated and non-isolated" is not covered by the corrected proof.
- Survey text: accurate (it does not claim non-isolated saddles; "decaying no faster than Ω(1/k)" and the summable-step remark are right). Suggest adding "with an invertible Hessian at the saddle" or citing v3.
- Notes: header "Ref. [57]" (older numbering). "including isolated and non-isolated strict-saddle sets" should be qualified by the v3 invertibility assumption.
- Our toy $f=x^2/2-y^2/2+y^4/4$, $\alpha_k=0.8/(k+1)^s$: s = ½, 1 reach (0, 1) from (1.5, 0.05); s = 2 stops at (0.174, 0.146), ‖∇f‖ = 0.225 ($x_\infty=1.5\prod(1-0.8/(k+1)^2)=0.1738$). Grid: for s = ½, 1 only the rows y = 0 and y = ±1.5 (mapped exactly onto y = 0 by the first step) end at the saddle; for s = 2, 92.4% stop at non-critical points.

## mertikopoulos2020almost [65] — Mertikopoulos, Hallak, Kavis, Cevher, NeurIPS 2020 (arXiv 2006.11144)
- Assumptions 1–4: f G-Lipschitz and L-smooth, bounded sublevel and gradient-sublevel sets, noise with finite q-th moment. Theorem 1: bounded APTs a.s.; Theorem 2: $\gamma_n=\Theta(1/n^p)$, $p\in(2/(q+2),1]$ ⇒ a.s. convergence to a connected component of crit f. Theorem 3: $p\in(0,1]$, bounded ($q=\infty$) uniformly exciting noise (Assumption 5) ⇒ $P(X_n\to S)=0$ for every strict saddle manifold S (Fig. 1: a ridge). Theorem 4: local $O(1/n^p)$ rate near a regular minimizer, conditioned on staying in a neighbourhood.
- Survey text: "SGD with Robbins–Monro step sizes ... avoids strict saddles, including nonisolated saddle manifolds ... under suitable noise conditions". Mostly right, but the paper's point is that it goes **beyond** Robbins–Monro: Theorem 2 allows non-square-summable steps ($p>2/(q+2)$) and Theorem 3 allows any $p\in(0,1]$ (with bounded noise). Suggest "with steps $\gamma_n\propto1/n^p$".
- Notes: header "Ref. [58]" (older numbering). Content correct.
- Our toy: radial $\varphi(r)=r^4/4-r^3+r^2$, ridge of strict saddles at r = 1 (eigenvalues −1, 0). GD started exactly on the ridge stays; SGD from 200 ridge points always leaves (≈ 38–44% to the centre, 56–63% to the outer ring for p = 0.5, 0.75, 1). Theorem 4 slopes: −0.509, −0.754, −1.004 for p = 0.5, 0.75, 1. The toy's gradient is unbounded at infinity (Assumption 1 wants bounded); runs stay in r < 2.6.

## antonakopoulos2022adagrad [66] — Antonakopoulos, Mertikopoulos, Piliouras, Wang, ICML 2022 (PMLR 162)
- Deterministic AdaGrad: ADANORM, ADADIAG, ADAFULL with $G_t=\delta_0^2I+\sum_{s\le t}g_sg_s^\top$ (diag or full), $\Gamma_t=G_t^{-1/2}$. Proposition 1: $\Gamma_t\to\Gamma\succ0$, $\sum\|\nabla f(x_t)\|^2<\infty$. Theorem 1: stable-manifold theorem for the non-autonomous system. Proposition 3: local diffeomorphism for $\delta_0\ge L$. Theorem 2: all three avoid strict saddles from almost any initialization.
- Survey text: accurate. Note the analysis is for deterministic gradients ("all the step-size policies and gradients are deterministic").
- Notes: header "Ref. [59]" (older numbering). Content correct.
- Our toy (saddle with curvature axes at 30°, δ₀ = 10): all methods escape every start off their stable set, in time ∝ log(1/d) (no uniform bound). Observation worth a sentence: the preconditioner changes **where** the stable set is: for ADADIAG it crosses our test segment at d* = 0.180, not at the Hessian's stable manifold (d* = 0 for GD, ADANORM, ADAFULL).
- Visuals: 3D GD / AdaGrad-full / AdaGrad-diag from 10⁻³ off the Hessian's stable manifold + AdaGrad-diag from 10⁻⁹ off its own stable set; 2D escape steps vs d for the four methods.

## cheridito2024gradient [67] — Cheridito, Jentzen, Rossmannek, JOTA 2024 (arXiv 2208.02083)
- Theorem 1.1 / 2.2: center-stable manifold theorem for maps that are only $C^1$ with locally Lipschitz derivative on a full-measure open set (Lipschitz center-stable manifold under Assumption 2.1). Prop. 4.5: for an affine target every non-global critical point has $L=f'(\alpha)^2(\beta-\alpha)^3/(12(n+1)^4)$, n even ≤ N, and is a sawtooth; non-global local minima exist (dead networks with $f'(\alpha)v_jw_j<0$). Theorem 4.7: for a.e. step, starts converging to the strict saddles in S are null. Theorem 1.2 (N even): GD limit loss below $f'(\alpha)^2(\beta-\alpha)^3/(12(N-1)^4)$ ⇒ limit loss 0. Sec. 5: leaky ReLU (small leak).
- Survey text: accurate. One nuance: "under a favorable initialization, given by an explicit threshold on the limiting loss" — the condition is on the limit of the run, not on the start (the abstract phrases it the same way).
- Notes: header "Ref. [60]" (older numbering). Content correct.
- Our checks: sawtooth levels 1/12, 1.029e−3, 1.333e−4 (n = 0, 2, 4); the n = 2 sawtooth built as a 4-neuron net with the sign pattern of Prop. 4.5(iii)(b) is critical (|∇L| = 7e−17) and a strict saddle (λ_min = −0.024). 40 GD runs (N = 4) all end below the threshold at loss ≤ 3.5e−8. A slice through a dead neuron shows the flat critical region (L = 1/12) where GD freezes.

## du2017gradient [68] — Du, Jin, Lee, Jordan, Póczos, Singh, NeurIPS 2017 (arXiv 1705.10412)
- Theorem 4.1: f on R^d with B, ℓ, ρ = poly(d), init U[−1,1]^d: GD stays Ω(1) from local minima for $e^{\Omega(d)}$ steps; PGD reaches one in poly(d, 1/ε). Cor. 4.2 (no ε-SOSP visited), Cor. 4.3 (any init concentrated in an ℓ∞ ball, e.g. Gaussian). Construction: "octopus" of d saddles, $2^d$ minima, spline gluing + Whitney extension; waits grow by $(L+\gamma)/\gamma$; experiments d = 5, 10 (Figs. 3–4).
- Survey text: accurate ("under fairly natural random initialization schemes" matches Cor. 4.3). Note the minimizers lie outside the initialization cube (the paper says so).
- Notes: header "Ref. [61]" (older numbering). The $\frac12x^2-\frac12y^2+\frac14y^4$ example is labelled as illustrative. Correct.
- Our chain (smooth, not the paper's construction, same mechanism): successive GD waits grow by 1.95–2.09 (L = 1) and 3.07–3.17 (L = 2), matching (L+γ)/γ; PGD's stay bounded. In 2/5 runs with L = 2, d = 6 the last coordinate underflows to ~3e−323 and double-precision GD never leaves the last saddle.

## wu2025loss [56] — new notes check (49_Wu_Simsek_Ged_ReLU_Stationarity.pdf)
- The replaced notes agree with the existing page: directional stationarity, escape neurons (no escape neuron ⇒ local minimum; converse for scalar output), embedding by splitting a neuron, saddle-to-saddle from vanishing initialization. No contradiction with `papers/wu2025loss/`.
- One stale reference only: the page's "Kink & escape" tab note says "Teaching example from our notes (x = (1, 0), y = 1)". The new notes no longer contain that one-neuron example (they use the generic $\sum_j a_j\sigma(w_j^\top x)$ model and the neuron-splitting identity). Suggest rewording the note to "Teaching example (one neuron, one data point)". Not edited (page belongs to an earlier round). Notes header says "Ref. [49]" (older numbering; survey [56]).
