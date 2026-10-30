## escape | The paper's saddle experiment (Sec. 5)
$$f(x)=\frac1n\sum_i f_i(x),\qquad f_i(x)=\tfrac12x^\top Hx+b_i^\top x+\lVert x\rVert_{10}^{10},\quad H=\mathrm{diag}(1,-0.1)$$
$\mathbb E b=0$, $\mathrm{Cov}(b)=\mathrm{diag}(1,0.01)$: the escape direction $x_2$ has 10× less noise. Minima at $(0,\pm0.01^{1/8})=(0,\pm0.562)$, $f^*=-0.0127$.
### Diagonal RMSProp (Algorithm 2, diagonal version)
$$\hat G_t=\beta\hat G_{t-1}+(1-\beta)g_t\odot g_t,\qquad x_{t+1}=x_t-\eta\,(\hat G_t+\varepsilon)^{-1/2}\odot g_t,\qquad\beta=1-\eta^{2/3}$$
with a burn-in of $W=\Theta(\eta^{-2/3})$ samples (Algorithm 3).
> Check script (20 seeds, $10^4$ steps, from the saddle): median steps until $|x_2|>0.4$: RMSProp $\eta=0.001$: 2 994; SGD $\eta=0.001$: none within $10^4$; SGD $\eta=0.01$: 2 846. Mean $f$ over the last 2 000 steps: RMSProp $-0.0124$ vs SGD($0.01$) $-0.0101$ — the paper's conclusion (Fig. 1).

## whiten | Why the preconditioner helps (Eq. 1–3)
Idealized RMSProp $x_{t+1}=x_t-\eta A_tg_t$ with $A_t=G_t^{-1/2}$, $G_t=\mathbb E[g_tg_t^\top]$:
$$x_{t+1}=x_t-\eta A_t\nabla_t-\eta\,\underbrace{A_t(g_t-\nabla_t)}_{\xi_t},\qquad\mathrm{Cov}(\xi_t)=I-G_t^{-1/2}\nabla_t\nabla_t^\top G_t^{-1/2}$$
Near a stationary point $\nabla_t\approx0$, so the noise is close to isotropic: every direction, including the escape direction, gets unit-variance noise.
### Guarantees
- Definition 4.2: $(\Lambda_1,\Lambda_2,\Gamma,\nu,\lambda_-)$-preconditioner; $\nu\le\lambda_{\min}(A\,\mathbb E[gg^\top]A^\top)$ is the noise floor in every direction.
- Theorem 4.4 (Algorithm 4, occasional large steps $r$): a $(\tau,\sqrt{\rho\tau})$-stationary point in $\tilde O\big(\frac{\Lambda_1^4\Lambda_2^4\Gamma^4}{\lambda_-^{10}\nu^4}\frac{L^3}{\rho\delta^3}\tau^{-5}\big)$ iterations; Corollary 4.2 transfers this to RMSProp with an online estimate of $G$ (Theorem 4.1).
> Check script: Eq. (3) verified exactly at $(0.05,0.03)$; notes' example $\Sigma=\mathrm{diag}(9,1)\Rightarrow A=\mathrm{diag}(1/3,1)$ whitens the noise.
