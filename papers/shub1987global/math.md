## manifold | Stable manifolds of a map
For a map $G$ with fixed point $p$, linearize $G(p+h)\approx p+DG(p)h$. Eigenvalues of $DG(p)$ inside the unit circle contract, outside expand; $p$ is hyperbolic if none lies on the circle.
### Theorems (Shub 1987)
- Stable manifold theorem (Ch. 5): near a hyperbolic fixed point the points whose orbits converge to $p$ form a smooth embedded manifold tangent to the contracting eigenspace; globally an immersed manifold
- Center-stable manifold theorem (Appendix III, Theorem III.7): without hyperbolicity, a local center-stable manifold tangent to the eigenspace of eigenvalues of modulus $\le1$ contains every orbit that stays near $p$; it has dimension $<d$ whenever some eigenvalue has modulus $>1$
### Why it matters for gradient descent
$G(x)=x-\eta\nabla f(x)$ has $DG(x^\ast)=I-\eta\nabla^2f(x^\ast)$. At a strict saddle some eigenvalue is $1+\eta|\lambda_{\min}|>1$, so the starts converging to it lie on a lower-dimensional manifold; with $G$ a diffeomorphism ($\eta<1/L$) their preimages stay measure zero (Lee et al. 2016, citing Theorem III.7).
### Toy on screen
$$f=\tfrac{x^2}2-\tfrac{u^2}2+\tfrac{u^4}4,\quad u=y-0.8x^2,\qquad\nabla^2f(0)=\mathrm{diag}(1,-1),\quad DG(0)=\mathrm{diag}(0.9,1.1)$$
> Check script ($\eta=0.1$): a start on the computed manifold at $x=0.8$ comes within $10^{-8}$ of the saddle ($2.7\times10^{-9}$ in the check script; the exact value depends on rounding); starts $\delta$ above it leave after 40, 89, 137, 185, 233 steps for $\delta=10^{-2},\dots,10^{-10}$.

## curve | Tangent to the eigenspace, but curved
> Check script: $y^\ast(0)=0$, $|y^\ast(x)|\le0.282x^2$ for $|x|\le0.25$ (tangent to the $x$-axis, the contracting eigenvector), quadratic fit $0.283x^2$, $y^\ast(\pm1.2)=0.533$.
The notes cite Theorem 5.20 for the hyperbolic case; the book is not online, so we checked the statement through Lee et al. (2016), who quote Shub's Theorem III.7.
