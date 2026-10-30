## lena | LENA (Algorithm 1)
Goal: $\lVert\nabla F(x)\rVert_2\le\epsilon$, $\lambda_{\min}(\nabla^2F(x))\ge-\epsilon_H$, using only stochastic gradients $\nabla f(x;\xi)$.
- **GD phase**: while $\lVert d_t\rVert>\epsilon$: $x_{t+1}=x_t-\frac{\eta}{\lVert d_t\rVert}d_t$.
- **Escape phase**: $m_s\leftarrow t$, $x\leftarrow x+\xi$, $\xi\sim\mathrm{Unif}(B_0(r))$; then for up to $\ell_{\rm thres}$ steps $\eta_t=\eta_H$, unless
$$\sum_{i=m_s}^t\eta_i^2\lVert d_i\rVert^2>(t-m_s+1)\,\mathcal D$$
in which case the **last step is shrunk** so equality holds, it is taken, and a new GD phase starts. If no shrink happens, output $x_{m_s}$.
### Why cap the movement (Eq. 4.1)
SPIDER/STORM errors grow with the path: $\lVert d_t-\nabla F(x_t)\rVert_2^2=\tilde O\big(\sum_{i=s_t}^{t-1}\lVert x_{i+1}-x_i\rVert_2^2\big)$. Bounding the average movement keeps $d_t$ accurate enough to use $\lVert d_t\rVert$ as the stopping test, without the big batch SSRGD needs at every escape (the source of its $\epsilon^{-3.5}$).
### Here
$F=\tfrac14(x^2-1)^2+\tfrac12y^2$, $\nabla f(x;\xi)=\nabla F(x)+(c\,x,\;b-c\,y)$, $b\sim N(0,0.5^2)$, $c\sim N(0,0.3^2)$; GradEst-SPIDER with $B=400$, $b=q=20$.
> Check script (30 seeds from $(0,1.3)$): every run returns a point with $\lVert\nabla F\rVert\le0.1$ next to a minimum, after one last-step shrinkage (median); the 95th percentile of $\lVert d_t-\nabla F(x_t)\rVert$ is 0.056. Notes' example: $\sqrt{0.003-0.03^2-0.04^2}=0.0224$.

## rates | Theorems 5.1 and 5.3
Assumptions 3.1–3.3: each $f(\cdot;\xi)$ is $L$-smooth and $\rho$-Hessian Lipschitz; bounded noise $\lVert\nabla f(x;\xi)-\nabla F(x)\rVert_2\le\sigma$.
- LENA-SPIDER (Theorem 5.1, $B=\tilde O(\sigma^2\epsilon^{-2}+\sigma^2\rho^2\epsilon_H^{-4})$, $b=q=\sqrt B$): $\tilde O(\sigma L\Delta\epsilon^{-3}+\sigma\rho^3L\Delta\epsilon_H^{-6})$ stochastic gradients.
- LENA-STORM (Theorem 5.3): the same order with a single loop.
- $\epsilon_H=\sqrt\epsilon$: $\tilde O(\epsilon^{-3})$ vs $\tilde O(\epsilon^{-3.5})$ for SSRGD (Remark 5.2). Lower bound $\Omega(\epsilon^{-3}+\epsilon_H^{-5})$ (Arjevani et al. 2020).
