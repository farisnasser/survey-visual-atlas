## sensing | Factorized matrix sensing
$$y_i=\langle A_i,X^\star\rangle+w_i,\qquad \min_{U\in\mathbb R^{n\times r}}f(U)=\sum_{i=1}^m\big(\langle A_i,UU^\top\rangle-y_i\big)^2$$
non-convex in $U$ although the measurements are linear in $X=UU^\top$.
### RIP (Definition 2.1)
$(1-\delta_r)\lVert X\rVert_F^2\le\frac1m\sum_i\langle A_i,X\rangle^2\le(1+\delta_r)\lVert X\rVert_F^2$ for all $X$ of rank $\le r$.
### Results (published NeurIPS version)
- Theorem 3.1: $(4r,\delta_{4r})$-RIP with $\delta_{4r}<\frac1{10}$ ⇒ every local minimum has $\lVert UU^\top-X^\star\rVert_F\le20\sqrt{\frac{\log n}{m}}\,\sigma_w$; noiseless ⇒ every local minimum is global (Theorem 4.1 relaxes to $\delta_{4r}<\frac15$)
- Theorem 3.2 (strict saddle): at a critical point with $UU^\top\ne X^\star$, $\lambda_{\min}\big(\tfrac1m\nabla^2f(U)\big)\le-\tfrac45\sigma_r(X^\star)$
- Theorem 3.3: projected perturbed SGD (Eq. 4) reaches a global optimum in polynomial time from random initialization
- Gaussian $A_i$ satisfy the RIP with $m=O(nr)$
### Toy on screen
$n=2$, $r=1$, $u^\star=(1,0.5)$. Notes' fully observed example: $X^\star=\mathrm{diag}(1,0)$, $f=(a^2-1)^2+2a^2b^2+b^4$, $\nabla^2f(0)=\mathrm{diag}(-4,0)$.
> Check script (40 draws × 20 GD runs per $m$): draws with a spurious local minimum: 12/40 ($m=3$), 6/40 ($m=4$), 0/40 for $m\in\{6,10,20,50\}$. In the browser's own draws $m=6$ still shows them in 2 of 40 (draws 4 and 27), none at $m=20$. Spurious points have positive loss, $\lVert\nabla f\rVert\approx10^{-7}$ and a positive-definite Hessian.

## rip | How large m must be
$\delta=\max|\mathrm{eig}(\Phi^\top\Phi/m)-1|$ with $\Phi_{ij}=\langle A_i,E_j\rangle$ in an orthonormal basis $\{E_j\}$ of $2\times2$ symmetric matrices. For $n=2$ every matrix has rank $\le4r=4$.
> Check script: median $\delta$ = 1.06 ($m=3$), 0.38 (50), 0.20 (200), 0.12 (500), 0.059 (2000).
