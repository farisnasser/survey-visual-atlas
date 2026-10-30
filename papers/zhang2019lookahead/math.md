## noisy | Algorithm 1 and the noisy quadratic (Sec. 3.1)
Fast weights start at the slow weights, $\theta_{t,0}=\phi_t$, and take $k$ steps of any inner optimizer $A$; then
$$\phi_{t+1}=\phi_t+\alpha\,(\theta_{t,k}-\phi_t),\qquad 0<\alpha\le1\qquad(1)$$
Unrolled, $\phi_{t+1}=\alpha\sum_{i=0}^{t}(1-\alpha)^{t-i}\theta_{i,k}+(1-\alpha)^{t+1}\phi_0$: an exponential average of the inner loops' endpoints.
### Model (Eq. 4)
$$\hat L(x)=\tfrac12(x-c)^\top A(x-c),\qquad c\sim\mathcal N(x^\star,\Sigma),\ A,\Sigma\ \text{diagonal},\ x^\star=0$$
$\mathbb E\hat L=\tfrac12\sum_ia_i\big(\mathbb E[x_i]^2+\mathbb V[x_i]+\sigma_i^2\big)$ (Eq. 5).
### On screen
$a=(1,0.3)$, $\sigma^2=(1,1)$, $\gamma=0.6$, $k=5$, $\alpha=0.5$.
> Check script: steady-state excess loss $\tfrac12\sum a_i\mathbb V[x_i]$: SGD 0.229, Lookahead slow weights 0.078 (ratio 0.34).

## ratio | Proposition 2 (steady-state risk)
For $0<\gamma<2/\max_ia_i$, both converge to $0$ in expectation, with variance fixed points
(per coordinate, $r_i=1-\gamma a_i$)
$$V_{\text{SGD},i}=\frac{\gamma^2a_i^2\sigma_i^2}{1-r_i^2}$$
$$V_{\text{LA},i}=\frac{\alpha^2\big(1-r_i^{2k}\big)}{\alpha^2\big(1-r_i^{2k}\big)+2\alpha(1-\alpha)\big(1-r_i^k\big)}\,V_{\text{SGD},i}$$
(The paper writes $\gamma^2A^2\Sigma^2$ with $\Sigma$ the noise covariance; the variance that matches simulation uses $\sigma_i^2$, the diagonal of $\Sigma$.)
The prefactor is $<1$ for $\alpha\in(0,1)$. Sec. 3.2: on deterministic quadratics Lookahead around classical momentum improves the rate in the under-damped regime (Fig. 4). Prop. 1 gives the loss-minimizing $\alpha$ on a quadratic.
- Reported: CIFAR-10 / CIFAR-100 accuracy 95.27 ± .06 / 78.34 ± .05 (Lookahead) vs SGD 95.23 ± .19 / 78.24 ± .18 and Adam 94.84 ± .16 / 76.88 ± .39 (Table 1); also ImageNet, Penn Treebank, WMT.
> Check script: $V_{\text{SGD}}$ and $V_{\text{LA}}$ match Monte Carlo within 2% for three settings; Prop. 1's $\alpha^\star$ matches a line search.
