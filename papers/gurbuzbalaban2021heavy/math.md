## cloud | SGD as a random affine recursion
Streaming least squares $F(x)=\tfrac12\mathbb E(a^\top x-y)^2$ with a fresh batch every step:
$$x_k=M_kx_{k-1}+q_k,\qquad M_k=I-\frac\eta b\sum_{i\in\Omega_k}a_ia_i^\top,\qquad q_k=\frac\eta b\sum_{i\in\Omega_k}a_iy_i$$
$(M_k,q_k)$ are i.i.d. across $k$ (a Kesten recursion). Mostly contracting, occasionally expanding products $M_k\cdots M_1$ create large excursions: a heavy tail from multiplicative noise, not from the data.
### Theorem 2
With $\rho=\lim\frac1k\mathbb E\log\lVert M_k\cdots M_1\rVert<0$ and $h(s)=\lim(\mathbb E\lVert M_k\cdots M_1\rVert^s)^{1/k}$, if $h(\alpha)=1$ has a unique positive root, the stationary law has a power-law tail with index $\alpha$: $\mathbb P(\langle u,x_\infty\rangle>t)\sim c\,t^{-\alpha}$.
### Toy on screen
$a\sim\mathcal N(0,I_2)$, $y\sim\mathcal N(0,1)$ independent, $x^\star=0$.

## tail | Theorem 3 (Gaussian inputs)
For $a\sim\mathcal N(0,\sigma^2I_d)$: $\rho=\mathbb E\log\lVert Me_1\rVert$ and $h(s)=\mathbb E\lVert Me_1\rVert^s$ (a fixed direction, by rotational symmetry).
$$\lVert Me_1\rVert^2=1-\tfrac{2\eta}bw_1+\tfrac{\eta^2}{b^2}\lVert w\rVert^2,\qquad w=\textstyle\sum_ia_i(a_i^\top e_1)$$
$\alpha>2$: finite variance; $1<\alpha<2$: finite mean, infinite variance. Corollary 11: a generalized CLT with an $\alpha$-stable limit for averages when $\alpha<2$.
### Notes' scalar example
$d=b=1$, $\eta=\sigma=1$: $\mathbb E|1-a^2|=4\varphi(1)\approx0.96788<1$ (contraction) but $\mathbb E(1-a^2)^2=2$, so $1<\alpha<2$.
> Check script: $\alpha=1.089$ for this example; a Hill estimate from 20 000 simulated chains gives 1.10.

## alpha | Theorem 4 and Proposition 5
$$h(2)=1-2\eta\sigma^2+\frac{\eta^2\sigma^4}{b}(d+b+1)\ \Rightarrow\ \eta_{\text{crit}}=\frac{2b}{\sigma^2(d+b+1)}$$
- $\eta<\eta_{\text{crit}}$: $\alpha>2$; $\eta=\eta_{\text{crit}}$: $\alpha=2$; slightly above: $\alpha<2$ (Prop. 5)
- Theorem 4: $\alpha$ strictly increases with $b$ and decreases with $\eta$, $\sigma^2$ (for $\alpha\ge1$) and with $d$
- Theorem 8: $W_p$ convergence to the stationary law at rate $h(p)^{k/p}$ for $p<\alpha$
- Experiments (Figs. 2–3): estimated tail indices of fully connected and VGG networks on MNIST/CIFAR-10 decrease with $\eta/b$ (reported, not reproduced here)
> Check script: $h(2)$ formula matches Monte Carlo within 1%; $\alpha(\eta_{\text{crit}})=1.99$ for $d=5$, $b=2$; monotonicity in $b$, $\eta$ and $d$ confirmed.
