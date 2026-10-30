## three | Schedule-Free SGD (Eqs. 3–5)
$$y_t=(1-\beta)z_t+\beta x_t,\qquad z_{t+1}=z_t-\gamma\nabla f(y_t,\zeta_t),\qquad x_{t+1}=(1-c_{t+1})x_t+c_{t+1}z_{t+1}$$
with $c_{t+1}=1/(t+1)$ and $z_1=x_1$, so $x$ is the equal-weight average of the $z$'s. $z$: base sequence (the SGD step); $y$: gradient location; $x$: evaluation sequence (what you report). $\beta=0$ is Polyak–Ruppert averaging, $\beta=1$ primal averaging (Eqs. 6–7); $\beta\approx0.9$ acts like momentum: a gradient enters $y$ with weight $1-\beta$ at once and the rest slowly through $x$.
### Notes' example
$f=\tfrac12(x-2)^2$, $\gamma=0.5$, $\beta=0.8$, $z_1=x_1=0$: $y_1=0$, $z_2=1$, $x_2=0.5$; $y_2=0.6$, $z_3=1.7$, $x_3=0.9$.
> Check script: both steps reproduced exactly; $x_T$ equals the mean of $z_1,\dots,z_T$.

## anytime | Theorems 1–3
**Theorem 1.** $F$ convex, $f(\cdot,\zeta)$ $G$-Lipschitz, $D=\lVert x_1-x^\star\rVert$, $\gamma=D/(G\sqrt T)$: for **any** $\beta\in[0,1]$
$$\mathbb E[F(x_T)-F(x^\star)]\le\frac{DG}{\sqrt T}$$
the optimal non-smooth convex rate (EMA momentum, by contrast, can hurt it). **Theorem 2** turns any online learner's regret into this kind of bound (an online-to-batch conversion that unifies schedules and averaging). **Theorem 3:** large steps $\gamma=D/G$ remain optimal when $\sum\langle g_t,z_t-z_1\rangle\le D\sum\lVert g_t\rVert^2$.
- Reported: competitive with or better than tuned cosine/linear schedules from logistic regression to large deep-learning tasks; Schedule-Free AdamW won the MLCommons AlgoPerf 2024 self-tuning track.
> Check script (20-D noisy least squares, best step per method): at $T=250,1000,4000$ the per-$T$ tuned linear decay reaches $9.8\times10^{-3},2.5\times10^{-3},7.4\times10^{-4}$; one Schedule-Free run reaches $1.5\times10^{-2},1.4\times10^{-3},4.2\times10^{-4}$; constant SGD $2.9\times10^{-3}$ at 4000.
