## find | Negative curvature finding with gradients (Algorithm 1, Proposition 3)
Near $\tilde x$, $\nabla f(\tilde x+y)-\nabla f(\tilde x)\approx H(\tilde x)\,y$ with error $\le\rho\lVert y\rVert^2$ (Eq. 6), so a gradient difference is a Hessian-vector product. Algorithm 1 runs the power method on $I-H/\ell$, re-evaluating the difference at radius $r$:
$$y_t=y_{t-1}-\frac{\lVert y_{t-1}\rVert}{\ell r}\Big(\nabla f\big(\tilde x+r\,y_{t-1}/\lVert y_{t-1}\rVert\big)-\nabla f(\tilde x)\Big),\qquad y_0\sim\mathrm{Unif}(B_{\tilde x}(r))$$
### Proposition 3
If $f$ is $\ell$-smooth, $\rho$-Hessian Lipschitz and $\lambda_{\min}(H(\tilde x))\le-\sqrt{\rho\epsilon}$, then with probability $\ge1-\delta_0$ the output $\hat e$ satisfies $\hat e^\top H(\tilde x)\hat e\le-\sqrt{\rho\epsilon}/4$ after $T=\tilde O(\log n/\sqrt{\epsilon})$ iterations.
### Lemma 6 (using the direction)
A step of length $\tfrac14\sqrt{\epsilon/\rho}$ along $\pm\hat e$ decreases $f$ by at least $\tfrac1{384}\sqrt{\epsilon^3/\rho}$.
### The paper's Fig. 1 setup
$$f(x_1,x_2)=\tfrac1{16}x_1^4-\tfrac12x_1^2+\tfrac98x_2^2,\quad H(0)=\mathrm{diag}(-1,\tfrac94)$$
step $\eta=0.05$, $r=0.1$, $M=300$ samples, $t_{\rm NCGD}=30$ vs $t_{\rm PGD}=90$.
> Check script: 94.7% of Algorithm-1 directions are within 5° of $\pm e_1$ after 30 steps. Equal budget of 90 gradient steps (30 + Lemma-6 kick of length 0.267 + 60 GD): 0.7% of Algorithm-1 paths decrease $f$ by at most 0.9, against 44.0% of PGD paths (paper: under 5% vs over 40%).

## rates | Iteration complexity (Theorem 7, Theorem 9)
- Deterministic (Algorithm 2 = accelerated version): an $\epsilon$-approximate SOSP ($\lVert\nabla f\rVert\le\epsilon$, $\lambda_{\min}\ge-\sqrt{\rho\epsilon}$) in $\tilde O(\log n/\epsilon^{1.75})$ iterations, vs $\tilde O(\log^4 n/\epsilon^2)$ for PGD and $\tilde O(\log^6n/\epsilon^{1.75})$ for PAGD.
- Stochastic (Algorithm 3): $\tilde O(\log^2 n/\epsilon^4)$, vs $\tilde O(\log^{15}n/\epsilon^4)$ for PSGD. Nested-loop methods are better in $\epsilon$: Neon2 $\tilde O(\log^2n/\epsilon^{3.5})$, SPIDER-type $\tilde O(\log^2 n/\epsilon^3)$.
- Every method is single-loop; the improvement is in powers of $\log n$, not in $1/\epsilon$.
> Orders from Table 1 of the paper; $\tilde O$ hides further logarithmic factors.
