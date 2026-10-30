## valley | Setting (Sec. 3)
$$f_{W,a}(x)=\frac1{\sqrt m}\sum_{r=1}^ma_r\,\mathrm{relu}(w_r^\top x),\qquad w_r\sim N(0,\kappa^2I),\ a_r\sim\mathrm{Unif}\{\pm1\}\ \text{fixed},\qquad\Phi(W)=\tfrac12\lVert y-u\rVert_2^2$$
GD on $W$. The Gram matrix of the infinitely wide network (Eq. 6), for unit inputs:
$$H^\infty_{ij}=\mathbb E_{w}\big[x_i^\top x_j\,\mathbb 1\{w^\top x_i\ge0,w^\top x_j\ge0\}\big]=\frac{x_i^\top x_j\,(\pi-\arccos x_i^\top x_j)}{2\pi}$$
Near initialization $u(k+1)-y\approx(I-\eta H^\infty)(u(k)-y)$: the component of the residual along eigenvector $v_i$ shrinks by $1-\eta\lambda_i$ per step.
> Check script: $\lambda_{\max}=2.63$, $\lambda_{\min}=0.065$ (closed form agrees with Monte Carlo to $3\times10^{-3}$). The slice directions are the weight changes $Z q$ for the top and bottom eigenvectors $q$ ($Z$ = Jacobian of $u$ at $W_0$), normalized.

## labels | Theorem 4.1 and Theorem 5.1
**Theorem 4.1.** With $\lambda_0=\lambda_{\min}(H^\infty)>0$, small $\kappa$, $m$ large enough and $\eta=O(\lambda_0/n^2)$, w.h.p. for all $k$
$$\lVert y-u(k)\rVert_2=\sqrt{\sum_{i=1}^n(1-\eta\lambda_i)^{2k}(v_i^\top y)^2}\pm\epsilon .$$
**Theorem 5.1** (any 1-Lipschitz loss): the trained network's population loss is at most $\sqrt{2y^\top(H^\infty)^{-1}y/n}$ plus lower-order terms — computable from data, independent of width.
### Here
Structured labels $y=\mathrm{sign}(x_1)$ vs random $\pm1$ labels on the same 60 inputs.
> Check script: $y^\top(H^\infty)^{-1}y$ = 99.1 (structured) vs 351.5 (random), leading bound terms 1.82 vs 3.42; 76% vs 12% of the label energy on the top-10 eigenvectors; the real GD residual stays within $0.021\sqrt n$ of the prediction; loss at $k=100$: $0.005$ vs $0.023$.
- Notes' example: $H=\mathrm{diag}(2,0.2)$, $\eta=0.25$, $y=(1,1)$: residual factors $0.5$ and $0.95$; $y^\top H^{-1}y=5.5$ — checked.
