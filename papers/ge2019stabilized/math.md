## stabilize | Perturbed and Stabilized SVRG (Algorithms 2–3)
SVRG estimator with snapshot $x_s$ (one full gradient per epoch of $m$ steps, minibatch $I_b$):
$$v_t=\frac1b\sum_{i\in I_b}\big(\nabla f_i(x_t)-\nabla f_i(x_s)\big)+\nabla f(x_s)\ \big[-\,v_{\rm shift}\big],\qquad x_{t+1}=x_t-\eta v_t$$
When $\lVert\nabla f(x_s)\rVert\le\mathcal G$ outside a super epoch: perturb $x_s\leftarrow x_s+\xi$, $\xi\sim B_0(\delta)$, and (Stabilized only) set $v_{\rm shift}=\nabla f(\tilde x)$. The super epoch ends after $T_{\max}$ steps or a travel distance $\mathscr L$, and then $v_{\rm shift}\leftarrow0$.
### Why shift (Sec. 4.3)
SVRG then runs on $\hat f(x)=f(x)-\langle\nabla f(\tilde x),x-\tilde x\rangle$: same Hessian, $\nabla\hat f(\tilde x)=0$. Phase 1 stays in a ball of radius $\tilde O(\delta)$ (Lemma 12) while the projection on the negative eigenspace grows to $\ge\delta/10$ (Lemma 11) — a negative-curvature search without a subroutine. Lemma 10: a super epoch decreases $f$ by $\tilde\Omega(\gamma^3/\rho^2)$ with constant probability.
### Toy model
$$f_i(x,y)=g_0x+x^2-\tfrac12y^2+\tfrac14y^4+a_ix+b_iy+\tfrac{c_i}2(x^2-y^2)+\tfrac{d_i}3x^3$$
$g_0=0.08$, zero-mean $a,b,c,d$, $n=125$, $b=25$, $m=5$, $\eta=0.1$, $\delta=0.02$, $\mathscr L=0.3$.
> Check script (200 seeds): at $t=1/(\eta\gamma)=10$ the median $\lVert x_t-\tilde x\rVert$ is 0.020 for Stabilized vs 0.041 for Perturbed SVRG (which drifts to $x\approx-0.036$); both escape along $e_1$ in every run on this easy instance, and the stabilized super epoch lowers $f$ by 0.25 on average.

## rates | Theorems 2–3
Assume $f$ is $\rho$-Hessian Lipschitz and each $f_i$ is $L$-smooth and $\rho'$-Hessian Lipschitz; $\epsilon$-SOSP: $\lVert\nabla f\rVert\le\epsilon$, $\lambda_{\min}(\nabla^2f)\ge-\sqrt{\rho\epsilon}$.
- Perturbed SVRG (Theorem 2): $\tilde O\Big(\big(\frac{n^{2/3}L\Delta_f}{\epsilon^2}+\frac{n\sqrt\rho\Delta_f}{\epsilon^{1.5}}\big)\big(1+(\tfrac{\rho'}{n^{1/3}\rho})^2\big)\Big)$ stochastic gradients.
- Stabilized SVRG (Theorem 3): $\tilde O\big(\frac{n^{2/3}L\Delta_f}{\epsilon^2}+\frac{n\sqrt\rho\Delta_f}{\epsilon^{1.5}}\big)$, only polylog in $\rho'$, with $b=\tilde O(n^{2/3})$, $m=n/b$, $\eta=\tilde O(1/L)$.
> Theorem numbers from the COLT 2019 (PMLR 99) version; arXiv v1 numbers them 1 and 2.
