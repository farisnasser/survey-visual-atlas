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
