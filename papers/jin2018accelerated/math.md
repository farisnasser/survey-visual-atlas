## escape | AGD and PAGD (Algorithms 1–2)
### AGD (Algorithm 1)
$$y_t=x_t+(1-\theta)v_t,\qquad x_{t+1}=y_t-\eta\nabla f(y_t),\qquad v_{t+1}=x_{t+1}-x_t$$
### PAGD (Algorithm 2) adds
- **Perturbation:** if $\lVert\nabla f(x_t)\rVert\le\epsilon$ and no kick in the last $\mathscr T$ steps, $x_t\leftarrow x_t+\xi$, $\xi\sim\mathrm{Unif}(B_0(r))$
- **NCE (Alg. 3)** when $f(x_t)\le f(y_t)+\langle\nabla f(y_t),x_t-y_t\rangle-\tfrac\gamma2\lVert x_t-y_t\rVert^2$: if $\lVert v_t\rVert\ge s$ keep $x_t$, else move $\pm s\,v_t/\lVert v_t\rVert$ to the better side; reset $v=0$
### Parameters (Eq. 3)
$$\eta=\frac1{4\ell},\quad \theta=\frac1{4\sqrt\kappa},\quad \gamma=\frac{\theta^2}{\eta},\quad s=\frac{\gamma}{4\rho},\quad \kappa=\frac{\ell}{\sqrt{\rho\epsilon}}$$
### Toy on screen
$$f(x,y)=\tfrac12x^2-\tfrac\lambda2y^2+\tfrac\lambda4y^4,\qquad \nabla^2f(0)=\mathrm{diag}(1,-\lambda)$$
with the target curvature $\sqrt{\rho\epsilon}=\lambda$, so $\theta=\sqrt\lambda/4$.

## scaling | Theorem 3
With probability $\ge1-\delta$, PAGD finds an $\epsilon$-second-order stationary point within
$$\tilde O\!\Big(\frac{\ell^{1/2}\rho^{1/4}\,(f(x_0)-f^\star)}{\epsilon^{7/4}}\Big)\ \text{iterations}$$
versus $\tilde O(\epsilon^{-2})$ for perturbed GD (Jin et al. 2017). First Hessian-free, single-loop method faster than GD for second-order stationarity.
> Check script: escape steps GD 178 → 3469 and AGD 47 → 226 for λ = 0.2 → 0.01; fitted slopes −0.992 and −0.524.

## energy | Hamiltonian (Lemmas 4–5)
$$E_t=f(x_t)+\frac1{2\eta}\lVert v_t\rVert^2$$
- Lemma 4: when (2) fails, $E_{t+1}\le E_t$ (needs $\eta\le\frac1{4\ell}$, $\theta\in[2\eta\gamma,\tfrac12]$)
- Lemma 5: when NCE runs, $E_{t+1}\le E_t-\min\{\frac{s^2}{2\eta},\frac12(\gamma-2\rho s)s^2\}$
- Corollary 6 (improve or localize): $\sum_\tau\lVert x_\tau-x_{\tau-1}\rVert^2\le\frac{2\eta}{\theta}(E_t-E_{t+T})$
> Check script: over 4000 PAGD steps $f$ rose on 1556 steps, $E$ on none (kicks excluded); NCE fired 31 times.
