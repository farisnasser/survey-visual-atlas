## sg | SG on nonconvex objectives (Sec. 4.3)
$$w_{k+1}=w_k-\alpha_kg(w_k,\xi_k)$$
### Assumptions
- 4.1: $\nabla F$ is $L$-Lipschitz
- 4.3: $F\ge F_{\inf}$ on the iterates; $\nabla F^\top\mathbb E[g]\ge\mu\lVert\nabla F\rVert^2$, $\lVert\mathbb E[g]\rVert\le\mu_G\lVert\nabla F\rVert$; $\mathbb V[g]\le M+M_V\lVert\nabla F\rVert^2$ ($M_G=M_V+\mu_G^2$)
### Theorem 4.8 (fixed step $\bar\alpha\le\mu/(LM_G)$)
$$\mathbb E\Big[\frac1K\sum_{k=1}^K\lVert\nabla F(w_k)\rVert^2\Big]\le\frac{\bar\alpha LM}{\mu}+\frac{2(F(w_1)-F_{\inf})}{K\mu\bar\alpha}\ \xrightarrow{K\to\infty}\ \frac{\bar\alpha LM}{\mu}$$
### Theorem 4.10 / 4.9 / Corollary 4.12 (diminishing steps, $\sum\alpha_k=\infty$, $\sum\alpha_k^2<\infty$)
$\lim_K\mathbb E\big[\sum\alpha_k\lVert\nabla F(w_k)\rVert^2\big]<\infty$, so the $\alpha$-weighted average tends to 0; $\liminf\mathbb E\lVert\nabla F(w_k)\rVert^2=0$, and $\lim=0$ with extra smoothness (Cor. 4.12).
### Toy on screen
$$F(x,y)=\tfrac{x^2+y^2}4+\sin x\sin y,\quad\nabla^2F\ \text{eigenvalues}\ \tfrac12-\sin x\sin y\pm|\cos x\cos y|\in[-\tfrac12,\tfrac32]$$
$g=\nabla F+\mathcal N(0,I)$: $\mu=\mu_G=1$, $M=2$, $M_V=0$, $L=1.5$, $F_{\inf}=-1$; $\bar\alpha\le2/3$.
> Check script (2000 runs, $K=3000$): running average 0.073, 0.237, 0.641 for $\bar\alpha=0.05,0.2,0.5$, below the bound (4.28b) 0.190, 0.610, 1.504 at every $K$; diminishing step $0.5/(1+k/50)$: weighted average still falling (0.177 at $K=3000$), last-iterate $\mathbb E\lVert\nabla F\rVert^2=0.0094$.
The review also covers noise reduction (dynamic sampling, gradient aggregation such as SVRG/SAGA) and second-order methods (Hessian-free Newton, quasi-Newton, Gauss–Newton) in Secs. 5–6.
