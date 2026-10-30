## scale | Lazy training (Sec. 1.1, Sec. 2)
Model $h:\mathbb R^p\to\mathcal F$, loss $R$, objective $F(w)=R(h(w))$. Linearization at $w_0$: $\bar h(w)=h(w_0)+Dh(w_0)(w-w_0)$. Training is **lazy** when it follows the gradient flow of $\bar F=R\circ\bar h$. The inverse relative scale (Eq. 1)
$$\kappa_h(w_0)=\lVert h(w_0)-y^\star\rVert\,\frac{\lVert D^2h(w_0)\rVert}{\lVert Dh(w_0)\rVert^2}$$
controls it; for $\alpha h$ it is $\kappa_{\alpha h}=\frac1\alpha\lVert\alpha h(w_0)-y^\star\rVert\frac{\lVert D^2h\rVert}{\lVert Dh\rVert^2}\to0$. Rescaled objective $F_\alpha(w)=\alpha^{-2}R(\alpha h(w))$.
### Theorem 2.2 (general lazy training)
If $h(w_0)=0$, on a fixed horizon $[0,T]$: $\sup_t\lVert w_\alpha(t)-w_0\rVert=O(1/\alpha)$, $\sup_t\lVert w_\alpha(t)-\bar w_\alpha(t)\rVert=O(1/\alpha^2)$, $\sup_t\lVert\alpha h(w_\alpha(t))-\alpha\bar h(\bar w_\alpha(t))\rVert=O(1/\alpha)$ (constants can grow exponentially in $T$; Theorem 2.3 removes that for the square loss).
### Toy model
$h(w)=w_1+\tfrac12w_2+w_1w_2$, $y^\star=1$, $R(y)=\tfrac12(y-1)^2$.
> Check script (flows to $T=6$): $\sup\lVert w-w_0\rVert$ = 0.389, 0.207, 0.108, 0.055, 0.028, 0.014 for $\alpha=2,\dots,64$ (slope $-0.96$); $\sup\lVert w-\bar w\rVert$ slope $-1.97$. Notes' example ($h=w+w^2$, $\alpha=100$): fit at $w=0.0099020$ vs linearized $0.01$ — checked.

## neurons | Over-/under-parameterized lazy training and the experiments
- Theorem 2.4 (over-parameterized, $Dh(w_0)^\top$ with $\sigma_{\min}>0$, strongly convex $R$): $\lVert\alpha h(w_\alpha(t))-y^\star\rVert\le\sqrt\kappa\,\lVert\alpha h(w_0)-y^\star\rVert e^{-m\sigma_{\min}^2t/4}$ — linear convergence, parameters move $O(1/\alpha)$ for all $t$.
- Theorem 2.5 (under-parameterized): for large $\alpha$ the flow converges geometrically to a **local** minimum of $F_\alpha$, not necessarily global.
- Two-layer nets: $f(w,x)=\sum_ja_j\max(b_j\cdot x,0)$ is 2-homogeneous, so a larger initialization scale $\tau$ acts like a larger $\alpha$. Symmetrized init ($b_{j+m/2}=b_j$, $a_{j+m/2}=-a_j$) gives $h(w_0)=0$.
- Experiments: lazy training (large $\tau$, or $1/\sqrt m$ scaling) generalizes worse (Fig. 1c, Fig. 2a); on CIFAR-10 CNNs the test accuracy degrades in the lazy regime.
> Check script (Fig. 1 setup, $n=15$, $m=20$, $d=2$): $\tau=0.1$ fits (loss $8.6\times10^{-7}$) with relative weight change 3.12; $\tau=5$ fits to $6.5\times10^{-4}$ with change 0.008; for $\tau\ge1$ the change scales like $\tau^{-1.94}$.
