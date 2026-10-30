## escape | SSRGD (Algorithm 1)
At epoch $s$: if not in a super epoch and $\lVert\nabla f(x_{sm})\rVert\le g_{\rm thres}$, set $x_{sm}\leftarrow x_{sm}+\xi$, $\xi\sim\mathrm{Unif}(B_0(r))$, and start a super epoch. Then $v_{sm}=\nabla f(x_{sm})$ and for $k=1..m$:
$$x_t=x_{t-1}-\eta v_{t-1},\qquad v_t=\frac1b\sum_{i\in I_b}\big(\nabla f_i(x_t)-\nabla f_i(x_{t-1})\big)+v_{t-1}$$
The super epoch stops once the iterate has moved far enough / the function has dropped enough, or after $t_{\rm thres}$ steps.
### Why the kick is needed here
$$f_i(x,y)=\tfrac14(x^2-1)^2+\tfrac12y^2+a_ix+b_iy+\tfrac{c_i}2(x^2-y^2)$$
On $x=0$ the $x$-part of $\nabla f_i(x_t)-\nabla f_i(x_{t-1})$ is exactly 0 and $\partial_xf=0$, so the recursion never leaves the line and converges to the saddle.
> Check script: without perturbation $x_t=0$ exactly for 2 500 steps and $y\to0$; SSRGD reaches $(\pm1,0)$ in all 50 seeded runs (median 6 160 stochastic gradients until $|x|>0.9$). $\eta=(\sqrt5-1)/(2L)=0.0745$, $m=b=\sqrt n=14$.
- Notes' example ($f=\tfrac12x^2-\tfrac12y^2$, $\eta=0.1$, start $(0.01,0.01)$): $x_{10}=0.00349$, $y_{10}=0.02594$ — checked.

## rates | Theorems 1–4
Assumption 1: each $f_i$ is $L$-smooth; Assumption 2: $\rho$-Hessian Lipschitz. $(\epsilon,\delta)$-SOSP: $\lVert\nabla f\rVert\le\epsilon$, $\lambda_{\min}\ge-\delta$.
- Theorem 1 (finite sum, first order): $\eta\le\frac{\sqrt5-1}{2L}$, $m=b=\sqrt n$: $O\big(n+\frac{L\Delta_f\sqrt n}{\epsilon^2}\big)$.
- Theorem 2 (finite sum, second order): $\tilde O\Big(\frac{L\Delta_f\sqrt n}{\epsilon^2}+\frac{L\rho^2\Delta_f\sqrt n}{\delta^4}+\frac{\rho^2\Delta_fn}{\delta^3}\Big)$ with $r=\tilde O(\min(\frac{\delta^3}{\rho^2\epsilon},\frac{\delta^{3/2}}{\rho\sqrt L}))$, $g_{\rm thres}=\epsilon$.
- Theorem 3 (online, first order): $O\big(\frac{\sigma^2}{\epsilon^2}+\frac{L\Delta_f\sigma}{\epsilon^3}\big)$; Theorem 4 gives the online second-order rate ($\tilde O(\epsilon^{-3.5})$ for $\delta=\sqrt\epsilon$).
- Lower bound (Fang et al. 2018): $\Omega(\sqrt n/\epsilon^2)$ even for first-order points, so the $\sqrt n$ is optimal.
