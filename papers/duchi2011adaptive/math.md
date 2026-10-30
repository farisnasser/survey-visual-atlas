## sparse | Diagonal AdaGrad (Eq. 1)
$$x_{t+1}=\Pi_{\mathcal X}^{\mathrm{diag}(G_t)^{1/2}}\Big(x_t-\eta\,\mathrm{diag}(G_t)^{-1/2}g_t\Big),\qquad G_t=\sum_{s\le t}g_sg_s^\top$$
Per coordinate (projection omitted): $x_{t+1,i}=x_{t,i}-\eta\,g_{t,i}/\sqrt{\sum_{s\le t}g_{s,i}^2}$. It is the proximal step $\arg\min_x\ \langle g_t,x\rangle+\tfrac1{2\eta}\lVert x-x_t\rVert_{H_t}^2$ with $H_t=\mathrm{diag}(G_t)^{1/2}$; the full-matrix version uses $G_t^{1/2}$.
### On screen
Logistic loss $\frac1n\sum\log(1+e^{-y_i\langle x_i,w\rangle})$, $n=4000$; $x_{i,2}\ne0$ for 3% of examples.
> Check script (1500 steps, best step each): excess loss SGD 0.0050, AdaGrad 0.0021; $|w_2-w_2^\star|$: SGD 1.37, AdaGrad 0.53.

## regret | Regret bounds (Theorem 5, Corollary 1)
With $D_\infty=\max_t\lVert x_t-x^\star\rVert_\infty$ and $\eta=D_\infty/\sqrt2$:
$$R(T)\le\sqrt2\,D_\infty\sum_{i=1}^d\lVert g_{1:T,i}\rVert_2=\sqrt{2d}\,D_\infty\,\inf_{s\succeq0,\ \langle\mathbf 1,s\rangle\le d}\sqrt{\sum_t\langle g_t,\mathrm{diag}(s)^{-1}g_t\rangle}$$
so AdaGrad competes with the best fixed diagonal preconditioner in hindsight. Full-matrix version (Theorem 7): $R(T)=O(\lVert x^\star\rVert_2\,\mathrm{tr}(G_T^{1/2}))$.
**Sparse example (Sec. 1.3):** hinge loss, $z_t\in\{-1,0,1\}^d$, feature $i$ present w.p. $p_i=\min\{1,ci^{-\alpha}\}$: $\mathbb E\sum_i\lVert g_{1:T,i}\rVert_2\le\sum_i\sqrt{p_iT}=O(\log d\,\sqrt T)$ for $\alpha\ge2$, vs $O(\sqrt{dT})$ for online gradient descent on the box. Convex online setting; no non-convex guarantee.
> Check script ($\alpha=2$): $\sum_i\sqrt{p_i}=2.93,5.19,7.49,9.79$ for $d=10^1\dots10^4$ (log growth) vs $\sqrt d=3.2\dots100$.
