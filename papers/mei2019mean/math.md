## neurons | Mean-field description (Sec. 1, Eq. 8)
$$\hat f_N(x;\theta)=\frac1N\sum_{i=1}^N\sigma_\star(x;\theta_i),\qquad R(\rho)=\mathbb E(y^2)+2\!\int\!V\,d\rho+\!\iint\!U\,d\rho\,d\rho$$
with $V(\theta)=-\mathbb E\{y\,\sigma_\star(x;\theta)\}$, $U(\theta_1,\theta_2)=\mathbb E\{\sigma_\star(x;\theta_1)\sigma_\star(x;\theta_2)\}$. One-pass SGD $\theta_i^{k+1}=\theta_i^k+2\varepsilon\,(y_k-\hat f_N(x_k))\nabla_\theta\sigma_\star(x_k;\theta_i^k)$ is approximated, with $t=k\varepsilon$, by the distributional dynamics $\partial_t\rho_t=2\xi(t)\nabla_\theta\cdot(\rho_t\nabla_\theta\Psi(\theta;\rho_t))$, $\Psi=V+\int U\,d\rho_t$ (a Wasserstein gradient flow).
### Sec. 3.2 example
$y=\pm1$ w.p. $\tfrac12$, $x\sim N(0,\Sigma_y)$, $\Sigma_\pm=\mathrm{diag}((1\pm\Delta)^2I_{s_0},I_{d-s_0})$; $\sigma_\star=\sigma(\langle w,x\rangle)$ with a truncated ReLU. For neurons with $\lVert P_Vw\rVert=r_1$, $\lVert P_\perp w\rVert=r_2$ and uniform directions, as $d\to\infty$
$$R_\infty(r_1,r_2)=\tfrac12\sum_{y=\pm1}\big(y-\mathbb E\,\sigma(Z\sqrt{(1+\Delta y)^2r_1^2+r_2^2})\big)^2$$
(the paper's $R_\infty(\bar\rho)$ for a point mass), computed exactly via $\mathbb E\,\mathrm{relu}(sZ-a)=s\varphi(a/s)-a(1-\Phi(a/s))$.
> Check script: a 1 500-neuron network at $d=400$ matches the infinite-$N$ formula to $3\times10^{-4}$ at $(1,0.5)$ and to 0.014 at $(0.4,1.2)$; the finite-$d$ formula converges to $R_\infty$ ($d=20\,000$: within $10^{-3}$). Grid minimum $R_\infty\approx0.0007$ at $(1.55,0.70)$.
- Theorem 3 (COLT: Theorem 6): with $s_0=\gamma d$, $N\ge C_0$ (independent of $d$) and $\varepsilon\le1/(C_0d)$, SGD reaches risk $\le\inf R_N+\eta$ after $O(d)$ samples — earlier work needed $N=O(d)$.

## dimfree | Theorem 1 (A): dimension-free approximation
Under A1–A4 (bounded Lipschitz step schedule, bounded activation and labels, smooth $V,U$, bounded coefficients), for noiseless SGD with fixed coefficients, w.p. $\ge1-e^{-z^2}$,
$$\sup_{k\in[0,T/\varepsilon]}\big|R_N(\theta^k)-R(\rho_{k\varepsilon})\big|\le Ke^{KT}\frac{\sqrt{\log N}+z}{\sqrt N}+Ke^{KT}\big(\sqrt{D+\log N}+z\big)\sqrt\varepsilon\qquad(9)$$
The number of neurons needed does not depend on the dimension $D$; the dimension enters only through $\varepsilon\ll1/D$. Part (B): general coefficients, $e^{KT^3}$ and a step condition. Theorem 2 (COLT: Theorem 4): noisy SGD and the diffusion PDE; its part (B) is **not** dimension-free (Remark 3.3 / COLT Remark 5). Sec. 4: with an output scale $\alpha\to\infty$ the dynamics linearize and kernel ridge regression is recovered (COLT: Theorem 8).
> Check script (3 seeds): gap $0.240\to0.106\to0.115$ for $N=50,200,800$ at $d=20$ and $0.202\to0.065\to0.038$ at $d=80$; the $N=3200$ risk falls from 0.99 to 0.14 ($d=20$) and from 0.83 to 0.04 ($d=80$).
