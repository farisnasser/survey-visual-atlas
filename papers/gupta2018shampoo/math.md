## rotate | Algorithm 1 (matrix case)
$$L_t=L_{t-1}+G_tG_t^\top,\qquad R_t=R_{t-1}+G_t^\top G_t,\qquad W_{t+1}=W_t-\eta\,L_t^{-1/4}\,G_t\,R_t^{-1/4}$$
with $L_0=\epsilon I_m$, $R_0=\epsilon I_n$; powers are spectral (eigen-decomposition), not elementwise. Memory $m^2+n^2$ instead of $m^2n^2$; preconditioner cost $O(m^3+n^3)$ instead of $O(m^3n^3)$. The general order-$k$ tensor version keeps one statistic per mode with power $-1/(2k)$.
**Lemma 4:** $(L\otimes R^\top)\,\mathrm{vec}(G)=\mathrm{vec}(LGR)$, so Shampoo is preconditioning $\mathrm{vec}(G)$ by $L_t^{-1/4}\otimes R_t^{-1/4}$.
### On screen
$W\in\mathbb R^{1\times2}$: $L_t$ is a scalar and $R_t$ a full $2\times2$ matrix, so the update is $L_t^{-1/4}\,g\,R_t^{-1/4}$; quadratic with curvatures 8 and 0.2, rotated as a whole.
> Check script: Shampoo's final loss is identical at 0° and 45° ($4.0\times10^{-11}$), full-matrix AdaGrad reaches $10^{-31}$ at both; diagonal AdaGrad goes from $5\times10^{-32}$ (aligned) to $4\times10^{-5}$ (45°).

## matrix | Theorem 7 and Lemma 8
**Lemma 8.** If every $G_t$ has rank $\le r$ and $g_t=\mathrm{vec}(G_t)$,
$$\epsilon I_{mn}+\frac1r\sum_tg_tg_t^\top\ \preceq\ \Big(\epsilon I_m+\sum_tG_tG_t^\top\Big)^{1/2}\otimes\Big(\epsilon I_n+\sum_tG_t^\top G_t\Big)^{1/2}$$
so the Kronecker preconditioner does not lose the small eigenvalues of full-matrix AdaGrad's.
**Theorem 7.** For convex losses, the regret against any $W^\star$ is at most $\sqrt{2r}\,D\,\mathrm{Tr}(L_T^{1/4})\,\mathrm{Tr}(R_T^{1/4})$, $D=\max_t\lVert W_t-W^\star\rVert_F$, which is $O(\sqrt T)$.
- Experiments (reported): ResNet-32 and Inception on CIFAR-10, ResNet-55 on CIFAR-100, an attention model on LM1B; the loss falls faster per epoch than with momentum SGD, Adagrad or Adam, at similar steps per second (Table 1; 3.5 vs about 4.9 steps/s on LM1B).
### Notes' example
$G=\mathrm{diag}(2,1)$ after one step, no damping: $L=R=\mathrm{diag}(4,1)$ and $L^{-1/4}GR^{-1/4}=I$. A $1000\times1000$ layer: $10^{12}$ entries for the flattened preconditioner vs $2\times10^6$ for Shampoo.
> Check script: Lemma 4 exact; Lemma 8 holds in 30 random tests (min eigenvalue of the difference 4.4). Matrix least squares ($8\times8$, best step each): GD $4.6\times10^{-4}$, diagonal AdaGrad $3.1\times10^{-4}$, Shampoo $1.2\times10^{-18}$, full AdaGrad $1.7\times10^{-31}$ after 1500 steps.
