## hardcase | Trust region, Krylov solutions, hard case
$$s_\star^{tr}\in\arg\min_{\lVert x\rVert\le R}\ f_{A,b}(x)=\tfrac12x^\top Ax+b^\top x$$
Characterization: $(A+\lambda_\star I)s_\star=-b$, $\lambda_\star\ge(-\lambda_{\min})_+$, $\lambda_\star(R-\lVert s_\star\rVert)=0$. If $u_{\min}^\top b\ne0$ then $\lambda_\star>-\lambda_{\min}$ and $s_\star=-(A+\lambda_\star I)^{-1}b$ is unique.
Krylov solution: $s_t^{tr}=\arg\min\{f_{A,b}(x):x\in\mathcal K_t(A,b),\ \lVert x\rVert\le R\}$, $\mathcal K_t=\mathrm{span}\{b,Ab,\dots,A^{t-1}b\}$, computed with $t$ matrix-vector products (Lanczos).
**Hard case:** $u_{\min}^\top b=0$ and $\lambda_\star=-\lambda_{\min}$. Every $\mathcal K_t$ is orthogonal to $u_{\min}$, so $s_t$ may never converge. Fix (Sec. 2.5): perturb $b$ or add a random vector.
### The notes' example
$$A=\mathrm{diag}(-2,1),\ b=(0,1),\ R=1:\quad \mathcal K_t=\mathrm{span}\{(0,1)\}\Rightarrow f=-\tfrac12;\qquad s_\star=\big(\pm\tfrac{2\sqrt2}3,-\tfrac13\big),\ f_\star=-\tfrac76$$

## rates | Theorem 1
For every $t>0$:
$$f(s_t)-f(s_\star)\le36\,[f(0)-f(s_\star)]\exp\!\Big(-4t\sqrt{\tfrac{\lambda_{\min}+\lambda_\star}{\lambda_{\max}+\lambda_\star}}\Big)$$
$$f(s_t)-f(s_\star)\le\frac{(\lambda_{\max}-\lambda_{\min})\lVert s_\star\rVert^2}{(t-\frac12)^2}\Big[4+\frac{\mathbb 1\{\lambda_{\min}<0\}}8\log^2\frac{4\lVert b\rVert^2}{(u_{\min}^\top b)^2}\Big]$$
Linear regime once $t\gtrsim\sqrt\kappa$, $\kappa=\frac{\lambda_{\max}+\lambda_\star}{\lambda_{\min}+\lambda_\star}$; sublinear ($\approx1/\sqrt\epsilon$ iterations) before. Both are tight up to constants (Sec. 4). Same results for the cubic-regularized problem with $(\rho/3)\lVert x\rVert^3$.
> Check script ($d=400$, $\lambda_{\min}=-0.6$): generic and perturbed gaps stay below both bounds for all $t\le120$ and reach $10^{-16}$; the hard case stalls at $2.3\times10^{-2}$.
