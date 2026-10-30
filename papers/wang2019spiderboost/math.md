## boost | SpiderBoost (Algorithm 1, Theorem 1)
Finite sum $f=\frac1n\sum_i f_i$. Every $q$ steps $v_k=\nabla f(x_k)$; otherwise, with one minibatch $S$ used at both points (Eq. 1),
$$v_k=\frac1{|S|}\sum_{i\in S}\big(\nabla f_i(x_k)-\nabla f_i(x_{k-1})\big)+v_{k-1},\qquad x_{k+1}=x_k-\eta v_k,\ \ \eta=\frac1{2L}$$
SPIDER uses the same $v_k$ but a normalized step of length $O(\epsilon/L)$.
### Theorem 1
If $\Psi^*>-\infty$ and **each** $\nabla f_i$ is $L$-Lipschitz (Assumption 1), then with $q=|S|=\sqrt n$, $\eta=1/(2L)$ and $K\ge O\big(L(f(x_0)-f^*)/\epsilon^2\big)$, a uniformly random iterate satisfies $\mathbb E\lVert\nabla f(x_\xi)\rVert\le\epsilon$. SFO complexity $O(\sqrt n\,\epsilon^{-2}+n)$, matching the lower bound for $n\le O(\epsilon^{-4})$.
### Toy model
$$f_i(x,y)=\tfrac14(x^2-1)^2+\tfrac12(y-\tfrac14)^2+a_ix+b_iy+\tfrac{c_i}2(x^2-y^2)$$
with $\sum a_i=\sum b_i=\sum c_i=0$, $n=196$, $L=11.1$ on $|x|\le1.8$.
> Check script (one seeded run each): $\lVert\nabla f\rVert<0.05$ after 2 744 SFO calls for SpiderBoost vs 11 984 for SPIDER. The live page uses fresh random minibatches, so its counts vary.
- Theorem 6 / Corollary 1: an online version gives $O(\epsilon^{-3})$ SFO calls.

## prox | Prox-SpiderBoost (Algorithm 2, Theorem 2)
For $\Psi=f+h$ with convex, possibly nonsmooth $h$:
$$x_{k+1}=\mathrm{prox}_{\eta h}(x_k-\eta v_k),\qquad \mathrm{prox}_{\eta h}(z)=\arg\min_x\ h(x)+\tfrac1{2\eta}\lVert x-z\rVert^2$$
Stationarity is measured by the gradient mapping $G_\eta(x)=\tfrac1\eta\big(x-\mathrm{prox}_{\eta h}(x-\eta\nabla f(x))\big)$; Theorem 2 gives $\mathbb E\lVert G_\eta(x_\xi)\rVert\le\epsilon$ with the same $O(\sqrt n\epsilon^{-2}+n)$ SFO and $O(\epsilon^{-2})$ prox calls.
### Here
$h=0.3(|x|+|y|)$, so the prox is soft-thresholding $z\mapsto\mathrm{sign}(z)\max(|z|-0.3\eta,0)$ per coordinate. Minimizers of $\Psi$: $y=0$ and $x^3-x+0.3=0$, $x=\pm0.787$.
> Check script: Prox-SpiderBoost ends at $(-0.7865,\,0)$ with $y$ exactly zero; with subgradient steps $|y|$ stays around $10^{-2}$. Notes example: prox of $0.3|x|$ at $0.8$ with $\eta=1$ is $0.5$.
- Algorithm 3 (Prox-SpiderBoost-M) adds momentum with the same complexity (Theorem 3); its speed-up is shown only in experiments.
