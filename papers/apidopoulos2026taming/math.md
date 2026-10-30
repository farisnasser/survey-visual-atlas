## taming | Tamed SGD
### Assumptions
- 1: $f\in C^2$ and gradient-coercive, $\liminf_{\lVert x\rVert\to\infty}\langle\nabla f(x),x\rangle/\lVert x\rVert^2>0$ (so $f^\ast>-\infty$ and crit $f$ is compact)
- 2: $(L_0,L_1)$-smoothness, $\lVert\nabla^2f(x)\rVert\le L_0+L_1\lVert\nabla f(x)\rVert$; gradients may grow superlinearly, even exponentially
- 3: sub-Gaussian noise with a state-dependent proxy $\sigma(x)$; 4 (ABC$_\sigma$): $\sigma(x)^2\le A[f(x)-f^\ast]+B\lVert\nabla f(x)\rVert^2+C$
### The schemes
$$x_{t+1}=x_t-\frac{\gamma_t}{1+\lambda_t r(x_t)}\,\hat g_t\qquad\text{(TSGD)},\qquad r(x)=\Omega\big(\max\{1,\sqrt{f(x)-f^\ast},\lVert\nabla f(x)\rVert\}\big)$$
$r$ is deterministic (Remark 2): normalizing by the random $\lVert\hat g_t\rVert$ would add a bias. (L0,L1)-GD, normalized GD and clipped GD are special cases. (DTSGD) adds a dissipative term and drops Assumption 4.
### Results (ICML 2026)
- Theorem 1 (TSGD): $\sum\gamma_t=\infty$, $\sum\gamma_t^2/\lambda_t<\infty$, $\gamma_t/\lambda_t\to0$ ⇒ $\nabla f(x_t)\to0$ a.s.; with $\gamma_t\propto t^{-p}$, $\lambda_t\propto t^{-q}$: $(q+1)/2<p\le1$
- Theorem 2 (DTSGD): same conclusion under Assumptions 1–3, $0\le1-p<q<(p-1/2)/\varepsilon$
- Theorem 3: with uniform noise excitation (Assumption 5), $P(x_t\to Q)=0$ for every strict saddle manifold $Q$; TSGD needs $2/3<p\le1$, $p/2<q<2p-1$ (25a)
### Toy on screen
$$f(x,y)=\tfrac{x^4}4-\tfrac{x^2}2+\tfrac{y^2}2+\tfrac{y^4}4,\qquad\nabla^2f(0)=\mathrm{diag}(-1,1),\qquad\nabla^2f(\pm1,0)=\mathrm{diag}(2,1)$$
> Check script: $\lVert\nabla^2f\rVert\le4+3\lVert\nabla f\rVert$ on $[-30,30]^2$ while $\lVert\nabla^2f\rVert$ exceeds 2000 there (no global Lipschitz constant); gradient-coercive; $p=0.8$, $q=0.5$ satisfy Theorem 1 and (25a). TSGD from 200 starts on the stable manifold $x=0$: all 200 end at a minimum.

## stability | Robustness to the step size
> Check script, 400 runs, 5000 steps, share ending within 0.15 of a minimum (diverged): $\gamma_0=0.1$: SGD 100%, clipped 100%, TSGD 100%; $0.3$: SGD 74.8% (25.3%), clipped and TSGD 100%; $1$: SGD 36.5% (63.5%), others 100%; $3$: SGD 0% (100%), others 100%; $10$: SGD 0% (100%), clipped 98.3%, TSGD 98.5%. No run of any method ends near the saddle.
### Reported in the paper (not reproduced)
Phase retrieval ($d=10$, $N=200$): TSGD with the "tight" $r(x)=\lVert x\rVert^3$ reaches $\lVert\nabla f\rVert\le10^{-3}$ more often than SGD, clipped SGD and TSGD with $\lVert x\rVert^5$ or $e^{\lVert x\rVert}$ (Fig. 1). Regularized deep diagonal network ($d=5$, $N=128$, $L=4$): TSGD keeps converging for large initial steps where SGD and clipped SGD fail (Fig. 2).
