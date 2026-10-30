## matched | Matched degrees: the certificate
$$\min_{W}\ f_r(W)=\ell\Big(Y,\sum_{i=1}^r\phi(W_i)\Big)+\lambda\sum_{i=1}^r\theta(W_i)$$
$\phi$ (one subnetwork) and $\theta$ (its regularizer) positively homogeneous of the **same** degree $p$, $\ell$ convex and once differentiable, $\lambda>0$.
### Results (CVPR 2017)
- Proposition 1: $\Omega_{\phi,\theta}(X)=\inf_r\inf\{\sum_i\theta(W_i):\Phi_r(W)=X\}$ is convex, so $\min_X \ell(Y,X)+\lambda\Omega(X)$ (Eq. 16) is a convex lower bound of $f_r$
- Theorem 1: any **local minimizer** of $f_r$ with some subnetwork $W_{i_0}=0$ is a global minimizer of $f_r$, and its output solves the convex problem (16)
- Corollary 1: a local minimizer of $f_r$ is global if appending a zero subnetwork gives a local minimizer of $f_{r+1}$
- Theorem 2: if $r>\operatorname{card}(X)$, from any point there is a non-increasing path to a global minimizer
### Toy on screen
$\phi(u,v)=uv$, $\theta=\tfrac12(u^2+v^2)$ (both degree 2), $f=\tfrac12(1-uv)^2+\tfrac\lambda2(u^2+v^2)$. Here $\Omega(X)=|X|$ and (16) is $\min_X\tfrac12(1-X)^2+\lambda|X|$, solved by $X^\star=\max(1-\lambda,0)$.
$$\nabla^2f(0)=\begin{pmatrix}\lambda&-1\-1&\lambda\end{pmatrix},\qquad \lambda_{\min}=\lambda-1$$
> Check script: $\lambda=0.3$: origin strict saddle ($\lambda_{\min}=-0.7$), global minima $u=v=\pm\sqrt{0.7}$ with $f^\star=\lambda-\lambda^2/2=0.255$; 200/200 GD runs reach $f^\star$. $\lambda=1.2$: the origin is a local minimum and the global one.

## mismatch | Mismatched degrees
### Proposition 19 (arXiv 1506.07540, long version)
$\Phi$ positively homogeneous of degree $p$, regularizer $G$ of degree $p'<p$ with $G>0$ wherever $\Phi\ne0$, $\ell$ convex with a subgradient at 0. Then the origin is always a local minimum:
$$f(\epsilon W)-f(0)\ \ge\ \epsilon^{p}\langle\partial\ell(0),\Phi(W)\rangle+\epsilon^{p'}G(W)>0\quad(\epsilon\text{ small})$$
so leaving it requires an increasing path, unless 0 happens to be the global minimizer.
### Toy on screen
$f=\tfrac12(1-uv)^2+\lambda(|u|+|v|)$, $p=2$, $p'=1$. Global minima at $u=v=\pm a$ with $a(1-a^2)=\lambda$ on the larger root. Optimizer: proximal gradient $u\leftarrow\operatorname{soft}(u+\eta(1-uv)v,\ \eta\lambda)$.
> Check script ($\lambda=0.15$): global value $0.2877$ at $a=0.914$ vs $f(0)=0.5$; $f>0.5$ on the punctured disc of radius 0.05; 39 of 400 proximal-gradient runs from $U[-1.4,1.4]^2$ end at the origin, the other 361 at the global minimum.

## relu | One-hidden-layer ReLU network
$$g(x)=\sum_{i=1}^r a_i\,\mathrm{ReLU}(w_i^\top[x;1]),\qquad F=\frac1{2n}\sum_j(g(x_j)-y_j)^2+\frac\lambda2\sum_i(a_i^2+\lVert w_i\rVert^2)$$
Network and penalty are both degree 2 in $(a_i,w_i)$ (the bias sits inside $w_i$, so it is regularized too), the paper's one-hidden-layer case.
### Local-minimum test at a zero neuron
Adding $\epsilon(a,w)$ to a zero neuron changes $F$ by $\epsilon^2\big(a\langle\rho,\mathrm{ReLU}(Xw)\rangle/n+\tfrac\lambda2(a^2+\lVert w\rVert^2)\big)$ with $\rho$ the residual, so no new neuron helps iff
$$\max_{\lVert w\rVert=1}\ \tfrac1n\big|\langle\rho,\mathrm{ReLU}(Xw)\rangle\big|\le\lambda .$$
That is the polar condition behind Corollary 1.
> Check script: 7 widths × 16 runs, 200 000 GD steps. The 8 runs with a zero neuron that pass the test all end at $F=0.0143253$ and no run goes lower; 81 runs end with dead zero neurons that fail the test, all higher (lowest 0.01631); 23 runs have no zero neuron. 23 runs still drift by less than $10^{-4}$ over the last 20 000 steps.
