## chain | Why a chain of saddles slows GD down
Near a strict saddle with escape curvature $-\gamma$, GD multiplies the escape coordinate by $1+\eta\gamma$ per step, so leaving from a distance $\delta$ takes about $\log(1/\delta)/(\eta\gamma)$ steps. If, meanwhile, the **next** escape coordinate sits in a direction of curvature $+L$, it shrinks by $1-\eta L$ per step. Each wait therefore makes the next start exponentially smaller:
$$T_{i+1}\approx\frac{L}{\gamma}\sum_{j\le i}T_j\quad\Longrightarrow\quad T_{i+1}\approx\frac{L+\gamma}{\gamma}\,T_i$$
(the multiplicative factor stated in Sec. 4.1 of the paper).
### Our two-saddle chain
$$f=-\tfrac{\gamma}2x^2+\tfrac12q(x)y^2+\tfrac{x^4}4+\tfrac{y^4}4,\qquad q(x)=L-(L+\gamma)\,S\!\big(x^2/c^2\big)$$
$S$ the quintic smoothstep, $\gamma=1$, $c=0.5$. Saddles at $0$ (Hessian $\mathrm{diag}(-\gamma,L)$) and $(\pm1,0)$ (Hessian $\mathrm{diag}(2\gamma,-\gamma)$); minima $(\pm1,\pm1)$, $f=-\tfrac12$.
> Check script, GD from $(0.001,0.05)$, step $1/(4(L+3))$: leaves saddle 1 / saddle 2 at steps 105/242 ($L=1$), 131/432 ($L=2$), 181/975 ($L=4$).
### Perturbed GD (Jin et al., Algorithm 1 in the paper)
When $\lVert\nabla f\rVert$ is small, add a uniform kick from a ball of radius $r$; the escape coordinate restarts from about $r$ instead of from an exponentially small value.

## exp | Exponential time in high dimension
### Results (NeurIPS 2017)
- Theorem 4.1: there is $f$ on $\mathbb R^d$, $B$-bounded, $\ell$-gradient- and $\rho$-Hessian-Lipschitz with $B,\ell,\rho=\mathrm{poly}(d)$, such that from $U[-1,1]^d$ GD with $\eta\le1/\ell$ stays $\Omega(1)$ away from every local minimum for $T\le e^{\Omega(d)}$ steps (with probability one), while perturbed GD reaches an $\epsilon$-neighbourhood of a local minimum in $\mathrm{poly}(d,1/\epsilon)$ steps w.p. $1-e^{-d}$
- Corollary 4.2: GD visits no $\epsilon$-second-order stationary point for $e^{\Omega(d)}$ steps; Corollary 4.3: the same for any initialization concentrated in an $\ell_\infty$ ball (e.g. Gaussian)
- Construction ("octopus", Sec. 4): $d$ saddles visited in turn, $2^d$ minima at $(\pm c,\dots,\pm c)$, glued with splines and extended by Whitney's theorem; experiments with $d=5,10$ (Figs. 3–4) show the GD plateaus lengthening by $(L+\gamma)/\gamma$ while PGD's stay $\approx1/(\eta\gamma)$
### Our six-saddle chain
$$f(x)=-\tfrac{\gamma}2x_1^2+\sum_{i\ge2}\tfrac12q(x_{i-1})x_i^2+\sum_i\tfrac{x_i^4}4$$
> Check script ($x_0\sim U[-0.1,0.1]^6$): ratios of successive GD waits 1.95–2.09 for $L=1$ and 3.07–3.17 for $L=2$ (theory 2 and 3). PGD clears all six saddles in 569–889 steps; GD needs 2095–4020 ($L=1$) and 15 467–20 904 ($L=2$), and in two of five $L=2$ runs the sixth coordinate underflows ($3\times10^{-323}$), so GD never leaves the last saddle in double precision.
