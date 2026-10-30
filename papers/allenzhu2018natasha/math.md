## swing | Swinging by saddle points (Sec. 4.2)
Online problem $f(x)=\mathbb E_i[f_i(x)]$ with bounded variance (A1), $L$-smooth (A2) and $L_2$-Hessian-Lipschitz (A4). Target: an $(\epsilon,\delta)$-approximate local minimum, $\lVert\nabla f(x)\rVert\le\epsilon$ and $\nabla^2f(x)\succeq-\delta I$ (Def. 4.2).
### One round at $y_k$
- Oja's algorithm (an online power method using stochastic Hessian-vector products $\nabla^2 f_i(y_k)w$) looks for $v$ with $v^\top\nabla^2f(y_k)v\le-\delta/2$.
- If found: move along $\pm v$ (negative-curvature step).
- If not: $f$ is $O(\delta)$-nonconvex near $y_k$; find an approximate stationary point of
$$F_k(x)=f(x)+L\max\Big(0,\lVert x-y_k\rVert-\tfrac{\delta}{L_2}\Big)^2$$
which stays in the safe zone $\lVert x-y_k\rVert\le\delta/L_2$ (Natasha 1.5 in the paper).
### Toy model
$$f_i(x,y)=\tfrac14(x^2-1)^2+\tfrac12y^2+a_ix+b_iy+\tfrac{c_i}2(x^2-y^2),\quad n=200$$
$\delta=0.5$, $L_2=9.6$ on $|x|\le1.6$, so the safe radius is $0.052$; 30 HVPs per Oja call, 12 SGD steps (batch 4) per safe-zone round.
> Check script (20 seeds, start $(0,1.3)$): every run ends at an $(0.05,\,0.5)$-approximate local minimum after a median of 30 rounds, 10 of them negative-curvature steps, never closer than 0.87 to the saddle; gradient descent from the same start converges to the saddle.

## rates | Theorem 1 and Theorem 2 (informal)
- Natasha 1.5 (stationary points, $\sigma$-bounded nonconvexity): $T=O\big(\epsilon^{-3}+\sigma^{1/3}\epsilon^{-10/3}\big)$.
- Natasha 2: for $\epsilon\in(0,1)$, $\delta\in(0,\epsilon^{1/4})$, w.p. $\ge2/3$ an $(\epsilon,\delta)$-approximate local minimum with
$$T=\tilde O\Big(\frac1{\delta^5}+\frac1{\delta\epsilon^3}\Big)$$
stochastic gradients plus Hessian-vector products. With $\delta=\epsilon^{1/4}$ (Remark 4.4): $\tilde O(\epsilon^{-3.25})$ (Corollary 4.6); with $\delta=\epsilon^{1/2}$: $\tilde O(\epsilon^{-3.5})$ (Corollary 4.7).
- SGD needs $O(\epsilon^{-4})$ even for stationary points; perturbed SGD $O(\mathrm{poly}(d)\,\epsilon^{-4})$.
- Hessian-vector products can be replaced by gradient differences (Neon2, Remark 4.5).
