## neon2 | Neon2online (Algorithm 2)
$$x_1=x_0+\xi,\quad \xi\sim\mathcal N(0,\sigma^2I)\ \text{tiny};\qquad x_{t+1}=x_t-\eta\big(\nabla f_i(x_t)-\nabla f_i(x_0)\big),\ i\sim\mathrm{Unif}[n]$$
(the **same** $i$ at both points). Stop and return $v=\frac{x_{t+1}-x_0}{\lVert x_{t+1}-x_0\rVert}$ once $\lVert x_{t+1}-x_0\rVert\ge r$; return $\perp$ after $T$ steps.
Near $x_0$: $\nabla f_i(x_t)-\nabla f_i(x_0)\approx\nabla^2f_i(x_0)(x_t-x_0)$, so this is Oja's algorithm on $I-\eta\nabla^2f(x_0)$, with error $O(r^2)$ from Hessian-Lipschitzness.
### Toy on screen
$$f(x,y)=\tfrac12x^2-\tfrac{0.4}2y^2+\tfrac14y^4,\qquad f_i=f+a_ix+b_iy+c_i\tfrac{x^2-y^2}2+e_ixy$$
Linear terms cancel in the difference; only the curvature noise $c_i,e_i$ remains.

## outcomes | Theorem 1
With probability $\ge1-p$:
- $v=\perp$ $\Rightarrow$ $\nabla^2f(x_0)\succeq-\delta I$
- $v\ne\perp$ $\Rightarrow$ $\lVert v\rVert=1$ and $v^\top\nabla^2f(x_0)v\le-\tfrac\delta2$

using $\tilde O\big(\tfrac{L^2}{\delta^2}\log^2\tfrac dp\big)$ stochastic gradients (matches Oja's algorithm). Neon2det uses Chebyshev polynomials with $\tilde O(1/\sqrt\delta)$ full gradients. Plugging Neon2 into SGD, SCSG, Natasha2 or GD turns stationary-point finders into local-minimum finders using gradients only.
> Check script ($d=20$): $P(v)\ge0.97$ for $\lambda_{\min}\le-\delta$, $P(v)\le0.01$ for $\lambda_{\min}\ge-\delta/2$; 1 of 1448 returned directions had $v^\top Hv>-\delta/2$.
