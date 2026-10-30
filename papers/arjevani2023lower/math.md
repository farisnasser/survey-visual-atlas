## oracle | Probabilistic zero-chains (Sec. 3)
Progress $\mathrm{prog}_\alpha(x)=\max\{i\ge0:|x_i|>\alpha\}$ ($x_0\equiv1$). A stochastic gradient $g(x,z)$ is a **probability-$p$ zero-chain** (Def. 2) if a query reveals the next coordinate with probability at most $p$ and never more than one. On the Carmon et al. chain $F_T$ (Eq. 15) the paper uses (Eq. 17)
$$[g_T(x,z)]_i=\nabla_iF_T(x)\Big(1+\mathbb 1\{i>\mathrm{prog}_{1/4}(x)\}\big(\tfrac zp-1\big)\Big),\qquad z\sim\mathrm{Bernoulli}(p)$$
- Lemma 3: unbiased, probability-$p$ zero-chain, $\mathbb E\lVert g_T-\nabla F_T\rVert^2\le23^2\frac{1-p}p$.
- Lemma 1: any zero-respecting algorithm has $\mathrm{prog}_0<T$ for all $t\le\frac{T-\log(1/\delta)}{2p}$ w.p. $\ge1-\delta$.
> Check script: unbiasedness (200 000 samples) and the variance bound hold; with SGD (step 0.05) the mean number of queries to activate all coordinates is 22/44/90 ($p=1$), 27/51/98 ($p=0.2$), 84/170/336 ($p=0.05$) for $T=4/8/16$.

## bounds | Theorems 1–3
- Theorem 1 (bounded variance $\sigma^2$, $L$-smooth, gap $\Delta$): $m_\epsilon\ge c\,\Delta L\sigma^2\epsilon^{-4}$ for zero-respecting methods, tight (SGD); dimension $O(\Delta L/\epsilon^2)$.
- Theorem 2 (mean-squared smoothness $\bar L$): $\bar m_\epsilon\ge c\big(\Delta\bar L\sigma\epsilon^{-3}+\sigma^2\epsilon^{-2}\big)$, tight (SPIDER, SNVRG).
- Theorem 3: both bounds for all randomized algorithms (any batch size $K$), via random rotations.
- The $\epsilon^{-4}$ is the product of the deterministic $\Delta L\epsilon^{-2}$ and the sample cost $\sigma^2\epsilon^{-2}$ of estimating one gradient.
