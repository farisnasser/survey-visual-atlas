## schedules | Time-varying steps
$$x_{k+1}=g(k,x_k)=x_k-\alpha_k\nabla f(x_k),\qquad \alpha_k\to0$$
A time non-homogeneous dynamical system, so the classical (autonomous) stable-manifold theorem behind Lee et al. does not apply directly.
### Results (NeurIPS 2019; arXiv v3)
- Theorem 4.1: a stable-manifold theorem for time non-homogeneous systems near a hyperbolic fixed point
- Corollary 4.2/4.3: with $\alpha_k\to0$, $\alpha_k\in\Omega(1/k)$, the stable set of a strict saddle has Lebesgue measure zero
- Theorems 5.1, 5.3, 5.4, 5.5–5.6: gradient descent, mirror descent, proximal point, manifold gradient descent avoid strict saddles for almost every start
- v3 acknowledgement: the proof needs the Hessian at the saddle to be **invertible**
- Remark 3.1: if $\sum_k\alpha_k<\infty$ (e.g. $1/k^2$, $1/2^k$), $x_k$ converges but not necessarily to a critical point (Fig. 1 uses $1/k^4$ on $x^2-y^2$)
### Why $\Omega(1/k)$
Along the unstable eigenvector ($\lambda=-1$): $y_{k+1}\approx(1+\alpha_k)y_k$, so $y_k\approx y_0\prod_{j<k}(1+\alpha_j)$, which diverges iff $\sum\alpha_j=\infty$.
### Toy on screen
$$f(x,y)=\tfrac{x^2}2-\tfrac{y^2}2+\tfrac{y^4}4,\qquad\nabla^2f(0)=\mathrm{diag}(1,-1),\qquad\alpha_k=0.8/(k+1)^s$$
> Check script, start $(1.5,0.05)$, 4000 steps: $s=\tfrac12$ and $s=1$ end at $(0,1)$; $s=2$ stops at $(0.174,0.146)$ with $\lVert\nabla f\rVert=0.225$, and $x_\infty=1.5\prod_k(1-0.8/(k+1)^2)=0.1738$.

## stable | The stable set
> Check script, 121 × 97 grid of starts, 3000 steps: $s=\tfrac12$ and $s=1$: 96.91% at a minimum, 363 starts (3.09%) at the saddle, all on the rows $y=0$ and $y=\pm1.5$ (the first step maps $y=\pm1.5$ to $y=1.5(1+0.8-0.8\cdot2.25)=0$); $s=2$: 6.92% at a minimum, 0.64% at the saddle, 92.44% at non-critical points.
On a finite grid the measure-zero lines show up as whole rows; in the plane they are curves.
