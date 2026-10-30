## step | Factored gradient descent
$$\min_{X\succeq0}f(X)\quad\longrightarrow\quad\min_{U\in\mathbb R^{n\times r}}f(UU^\top),\qquad U^+=U-\eta\,\nabla f(UU^\top)\,U$$
$f$ convex in $X$; the factored problem is non-convex ($U$ and $UR$ give the same $X$ for orthogonal $R$, and new saddles appear). The true gradient is $2\nabla f(UU^\top)U$; the paper absorbs the 2 in $\eta$.
### Step size (Eq. 8)
$$\eta=\frac{1}{16\,(M\lVert X_0\rVert_2+\lVert\nabla f(X_0)\rVert_2)}$$
For separable $f$ the Hessian of $g(u)=f(uu^\top)$ scales like $M\lVert X\rVert_2+\lVert\nabla f(X)\rVert_2$, so the textbook $1/M$ step can overshoot.
### Toy on screen (the paper's Sec. 3 example)
$$g(u)=\lVert uu^\top-Y\rVert_F^2,\quad Y=a^2e_1e_1^\top-b^2e_2e_2^\top,\quad \nabla f(X)=2(X-Y),\ M=2$$
$$\nabla^2g(0)=\mathrm{diag}(-4a^2,4b^2),\qquad\nabla^2g(\pm ae_1)=\mathrm{diag}(8a^2,4a^2+4b^2)$$
A constant step is locally stable at $U^\star$ only if $\eta<1/(2a^2)=1/(2\lVert X^\star\rVert_2)$.
> Check script ($b=1$, start $(a/2,b/10)$): step (8) reaches Dist $<10^{-8}$ in 267, 180, 209, 235 steps for $a=0.8,1.5,2,3$; $\eta=1/M=0.5$ converges for $a=0.8$ and diverges for $a=1.5,2,3$.

## rate | Local rates (Sec. 4)
Assumptions: (A1)/(A2) a start with $\mathrm{Dist}(U_0,U^\star_r)\le\rho\,\sigma_r(U^\star_r)$, $\rho=\frac{1}{100}\frac{\sigma_r(X^\star)}{\sigma_1(X^\star)}$ (smooth) or $\frac{1}{100\kappa}\frac{\sigma_r(X^\star)}{\sigma_1(X^\star)}$ (restricted strongly convex); (A3) $X^\star$ close to its rank-$r$ part.
- Theorem 4.1 ($f$ $M$-smooth): $f(X_k)-f(X^\star_r)=O(1/k)$, as gradient descent on a smooth convex $f$
- Theorem 4.2 ($f$ $(m,r)$-restricted strongly convex): $\mathrm{Dist}(U^+,U^\star_r)^2\le\alpha\,\mathrm{Dist}(U,U^\star_r)^2+\beta\lVert X^\star-X^\star_r\rVert_F^2$, $\alpha=1-\frac{m\sigma_r(X^\star)}{64(M\lVert X^\star\rVert_2+\lVert\nabla f(X^\star)\rVert_2)}$
- Corollary 4.3 (rank $X^\star=r$): $\mathrm{Dist}(U^+,U^\star)^2\le\alpha\,\mathrm{Dist}(U,U^\star)^2$, linear convergence
- Theorem 5.2: an initialization from $P_+(-\nabla f(0))$ that meets (A2) when the condition number is small
> Check script ($a=2$): $\rho=0.9875$, measured per-step ratio of Dist² 0.880 with step (8); the bound holds for all 400 steps from the (A2) start.
