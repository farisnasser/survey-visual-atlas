## chain | The hard instance (Eq. 10) and its properties
$$\bar f_T(x)=-\Psi(1)\Phi(x_1)+\sum_{i=2}^T\big[\Psi(-x_{i-1})\Phi(-x_i)-\Psi(x_{i-1})\Phi(x_i)\big]$$
$$\Psi(t)=\begin{cases}0&t\le\tfrac12\\ \exp\!\big(1-\tfrac1{(2t-1)^2}\big)&t>\tfrac12\end{cases},\qquad\Phi(t)=\sqrt e\int_{-\infty}^te^{-s^2/2}ds$$
- **Zero-chain** (Def. 3): if $x$ is supported on the first $i-1$ coordinates, every derivative of every order is supported on the first $i$. A zero-respecting algorithm therefore uncovers at most one coordinate per query (Observation 1).
- Lemma 2: if $|x_i|<1$ for some $i\le T$, then $\lVert\nabla\bar f_T(x)\rVert>1$.
- Lemma 3: $\bar f_T(0)-\inf\bar f_T\le12T$, $\lVert\nabla\bar f_T\rVert\le23\sqrt T$, $p$-th derivatives $\ell_p$-Lipschitz.
> Check script: gradient matches finite differences; zero-chain and Lemma 2 hold on 300/2 000 random points; for $T=2$, GD from 0 first moves $x_2$ at step 8, when $x_1=0.624>\tfrac12$.

## bounds | Theorems 1 and 2
For any $p\ge1$, $\Delta,L_p,\epsilon>0$ and any (even randomized) algorithm with full derivative information, there is $f$ with $L_p$-Lipschitz $p$-th derivative and $f(x^{(0)})-\inf f\le\Delta$ that needs at least
$$c_p\,\Delta\,L_p^{1/p}\,\epsilon^{-(p+1)/p}$$
queries to find $\lVert\nabla f(x)\rVert\le\epsilon$ (Theorem 1 deterministic, Theorem 2 randomized; dimension polynomial in $1/\epsilon$).
- Sharp: $p=1$ matches gradient descent ($\epsilon^{-2}$), $p=2$ cubic-regularized Newton ($\epsilon^{-3/2}$), general $p$ the $p$-th order regularization methods.
- About first-order stationarity only — not second-order points or global minima.
> Check script: GD (step 0.05) needs 15, 29, 44, 71, 96, 121, 146 steps for $T=1,2,4,8,12,16,20$ — about 6.4 steps per coordinate.
