## anchor | Algorithm 2 and Theorem 1
Input $\tilde x$, $K_0\sim\epsilon^{-2}$, $\eta\sim\epsilon^{1.5}$, $B\sim\epsilon^{0.5}$ (up to logs). Starting from $x_0=\tilde x$, $k=0$:
$$x_{k+1}=x_k-\eta\nabla F(x_k;\zeta_{k+1});\qquad \text{if }\lVert x_k-x_0\rVert>B:\ x_0\leftarrow x_k,\ k\leftarrow0$$
After $K_0$ consecutive in-ball steps return $\bar x=\frac1{K_0}\sum_{k<K_0}x_k$.
### Assumptions
- (1) component gradients $L$-Lipschitz, Hessian $\rho$-Lipschitz; (2) finite gap $\Delta$; (3) $\lVert\nabla F-\nabla f\rVert\le\sigma$
- (4) **dispersive noise** (Definition 3): every set that is narrow along some direction gets probability $\le\frac14$; uniform-ball noise qualifies. Algorithm 3 drops (4) by injecting Gaussian noise every $K_o$ steps.
### Theorem 1 (arXiv 1902.00247; the notes call it Theorem 5)
W.p. $\ge1-(T_1+1)p$, Algorithm 2 stops within $T_0=\tilde O(\epsilon^{-3.5})$ stochastic gradients and $\bar x$ satisfies $\lVert\nabla f(\bar x)\rVert\le18B^2\rho$-type and $\lambda_{\min}(\nabla^2f(\bar x))\ge-17\sqrt{\rho\epsilon}$-type bounds, i.e. an $(\epsilon,O(\sqrt\epsilon))$-SOSP. Classical analyses of SGD needed $\epsilon^{-4}$.
### Why averaging
Summing $x_{k+1}-x_k=-\eta(\nabla f(x_k)+\xi_k)$ telescopes: the mean gradient is bounded by displacement $/(\eta K_0)$ plus mean noise; for a quadratic $\nabla f(\bar x)$ equals the mean gradient exactly.
> Check script (20 runs from the stable set): every run stops; outputs lie near $(\pm1,0)$ with $\lVert\nabla f\rVert<0.1$ (median 628 steps, 7 anchor moves).

## coupling | The coupling argument (Appendix C)
On $q(u,v)=\tfrac12(-u^2+v^2)$ with $\eta=0.1$: $u_{k+1}=1.1u_k-0.1\xi_k$, $v_{k+1}=0.9v_k-0.1\zeta_k$. Two runs with the same noise and starts $0.002$ apart in $u$ satisfy
$$\Delta u_k=0.002\cdot1.1^k,\qquad \Delta v_k=0$$
so after $k=\lceil\log10/\log1.1\rceil=25$ steps they are $0.0216694>2B=0.02$ apart. The stuck region is thin along the unstable direction; Hessian-Lipschitzness and component smoothness keep this true beyond quadratics.
