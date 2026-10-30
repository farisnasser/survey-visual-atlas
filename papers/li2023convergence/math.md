## adam | Relaxed smoothness and Theorem 4.1
**Definition 3.2.** $f$ is $(\rho,L_0,L_\rho)$ smooth if, almost everywhere,
$$\lVert\nabla^2f(x)\rVert\le L_0+L_\rho\lVert\nabla f(x)\rVert^\rho$$
**Assumption 2:** $0\le\rho<2$. $\rho=1$ is the $(L_0,L_1)$ condition of Zhang et al. [41]; $\rho=0$ or $L_\rho=0$ is ordinary $L$-smoothness. Assumption 3: centered, almost surely bounded gradient noise (Theorem 4.2: sub-Gaussian).
**Theorem 4.1.** With a small momentum weight $\beta$ and small $\lambda$ (both $O(\epsilon^2)$), a small enough $\eta$, and any $0\le\beta_{sq}\le1$, with probability $1-\delta$: $\lVert\nabla f(x_t)\rVert\le G$ for all $t\le T$ and
$$\frac1T\sum_{t=1}^T\lVert\nabla f(x_t)\rVert^2\le\epsilon^2,\qquad T=O(\epsilon^{-4})$$
$G$ depends on $\rho, L_0, L_\rho$, the noise and $f(x_1)-f^\star$ but not on $\epsilon$. The key step: bounded gradients are proved along the trajectory, not assumed. Sec. 6: a variance-reduced Adam (VRAdam) reaches $O(\epsilon^{-3})$. The bound scales as $\lambda^{-2}$, which the authors flag as a limitation for tiny $\lambda$.
### On screen
$f=\cosh(1.5x)+(y^2-1)^2$, start $(3.6,1.9)$, $\eta=0.05$.
> Check script: $\lVert\nabla^2f\rVert\le8+3\lVert\nabla f\rVert$ on $[-3,3]^2$ (max ratio 0.985), but $\lVert\nabla^2f\rVert=1.8\times10^5$ at $x=8$. GD with 0.05 leaves after 2 steps; GD with 0.01 converges; Adam with 0.05 converges with $\max_t\lVert\nabla f(x_t)\rVert=167.2$, its starting value.

## classes | Which functions qualify (Prop. B.1)
- Polynomials and $e^x$: $(L_0,L_1)$ smooth ($\rho=1$); $x^4$: $|f''|=12(|f'|/4)^{2/3}$.
- Rational functions, e.g. $1/x$: $|f''|=2|f'|^{3/2}$, so $1<\rho<2$, not $(L_0,L_1)$.
- Double exponentials $a^{(b^x)}$: e.g. $e^{e^x}$ has $f''/f'=1+e^x$, unbounded, so not $(L_0,L_1)$; $(\rho,L_0,L_\rho)$ smooth for $1<\rho<2$.
- Lemma 3.4: a bounded gradient at $x$ gives a local Lipschitz bound on $\nabla f$ in a neighbourhood, which is what the proof uses.
> Check script: the four identities above hold to machine precision on their ranges.
