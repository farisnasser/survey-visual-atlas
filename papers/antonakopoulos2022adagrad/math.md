## adagrad | AdaGrad as a time-varying system
$$x_{t+1}=x_t-\Gamma_t\nabla f(x_t)$$
- (ADANORM) $\Gamma_t=\big(\delta_0^2+\sum_{s\le t}\lVert\nabla f(x_s)\rVert^2\big)^{-1/2}$
- (ADADIAG) $\Gamma_t=G_t^{-1/2}$, $G_t=\delta_0^2I+\mathrm{diag}\sum_{s\le t}\nabla f(x_s)\nabla f(x_s)^\top$
- (ADAFULL) $\Gamma_t=G_t^{-1/2}$, $G_t=\delta_0^2I+\sum_{s\le t}\nabla f(x_s)\nabla f(x_s)^\top$

The step depends on the whole history, so neither the constant-step nor the vanishing-step stable-manifold theorem applies directly. Deterministic gradients throughout.
### Results (ICML 2022)
- Proposition 1: $\Gamma_t\to\Gamma\succ0$ and $\sum_t\lVert\nabla f(x_t)\rVert^2<\infty$ (so $\lVert\nabla f\rVert^2$ decays like $O(1/T)$ on average, Remark 3)
- Theorem 1: a stable-manifold theorem for the resulting non-autonomous system (Lyapunov–Perron / Banach fixed point)
- Proposition 3: the updates are local diffeomorphisms for $\delta_0\ge L$
- Theorem 2: ADANORM, ADADIAG and ADAFULL avoid strict saddle points from almost every initialization (the proof extends from countably many saddles to uncountably many)
### Toy on screen
$$f=5u^2-\tfrac{v^2}2+\tfrac{v^4}4,\quad\begin{pmatrix}u\\v\end{pmatrix}=\begin{pmatrix}\cos30^\circ&\sin30^\circ\\-\sin30^\circ&\cos30^\circ\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}$$
Hessian at 0: eigenvalues $10$ (along $u$) and $-1$ (along $v$). With a fixed $\Gamma\succ0$ the linearization $I-\Gamma H$ is similar to $I-\Gamma^{1/2}H\Gamma^{1/2}$, which has the same inertia as $H$: one unstable direction, but its stable manifold depends on $\Gamma$.
> Check script ($\delta_0=10$): limits of $\Gamma_t$ eigenvalues from the near-saddle start: norm 0.0508; diag 0.0585, 0.0774; full 0.0511, 0.0988. All four methods end at a minimum.

## escape | Escape time and stable sets
> Check script, starts $1.5(\cos30^\circ,\sin30^\circ)+(d^\ast+d)(-\sin30^\circ,\cos30^\circ)$: $d^\ast=0$ for GD, norm and full, $d^\ast=0.180$ for diag. Steps to $|v|>0.5$ from $d=10^{-12}$ and $d=10^{-1}$: GD 315 / 21, norm 544 / 35, diag 388 / 26, full 285 / 20; growth 10–20 steps per unit of $\log(1/d)$.
Avoidance (Theorem 2) is about the measure of the bad set, not about speed; how slow escape can be is the subject of Du et al. [68].
