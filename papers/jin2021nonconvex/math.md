## saddle | Perturbed GD (JACM version)
$$x_{t+1}=x_t-\eta\big(\nabla f(x_t)+\xi_t\big),\qquad\xi_t\sim\mathcal N\big(0,(r^2/d)I\big)\quad\text{(Algorithm 1)}$$
The noise is added at **every** step, so $\mathbb E\lVert\xi_t\rVert^2=r^2$; the paper notes that the 2017 version added a perturbation only when certain conditions held (small gradient).
### Results
- Theorem 13 (Assumption A: $\ell$-gradient- and $\rho$-Hessian-Lipschitz): with $\eta=\tilde\Theta(1/\ell)$, $r=\tilde\Theta(\epsilon)$, PGD visits an $\epsilon$-second-order stationary point ($\lVert\nabla f\rVert\le\epsilon$, $\lambda_{\min}(\nabla^2f)\ge-\sqrt{\rho\epsilon}$) within $\tilde O\big(\ell(f(x_0)-f^\ast)/\epsilon^2\big)$ iterations, the same as GD for first-order points up to polylog factors in $d$
- Theorem 16 (PSGD, Algorithm 2) and mini-batch PSGD: $\tilde O\big(\ell\Delta_f/\epsilon^2\cdot\mathcal N\big)$ with $\mathcal N$ depending on the noise (polylog in $d$ when the stochastic gradients are Lipschitz, Assumption C)
- Sec. 3: for a wide class of ML and signal-processing problems (e.g. low-rank matrix problems) every second-order stationary point is global, so these rates give global guarantees there
### Toy on screen
$$f(x,y)=-\tfrac{x^2}2+\tfrac{x^4}4+\tfrac{y^2}2,\qquad\nabla^2f(0)=\mathrm{diag}(-1,1)$$
> Check script: GD from exactly the saddle does not move (the gradient is zero).

## dim | Dimension dependence
$$f(x)=-\tfrac\gamma2x_1^2+\tfrac{x_1^4}4+\sum_{i\ge2}\tfrac\lambda2x_i^2,\qquad x_0=0$$
Along $x_1$: $x_1\leftarrow(1+\eta\gamma)x_1+\eta\,\mathcal N(0,r^2/d)$, so escape needs about $\log\big(c\sqrt d/(\eta r)\big)/\log(1+\eta\gamma)$ steps.
> Check script ($\eta=0.1$, $\gamma=\lambda=1$, $r=0.01$, 400 runs): median escape step 63 ($d=1$), 78 (16), 91 (256), 107 (4096), 113 (16384); fitted slope 5.17 per unit $\log d$ vs $1/(2\log1.1)=5.25$.
