## estimator | SPIDER estimator and Lemma 1
Every $q$ steps $v_k=\nabla f(x_k)$ (full gradient in a finite sum, a large batch $S_1$ online); otherwise
$$v_k=\nabla f_{S_2}(x_k)-\nabla f_{S_2}(x_{k-1})+v_{k-1}$$
with the **same** minibatch $S_2$ at both points.
### Lemma 1
If $\lVert x_k-x_{k-1}\rVert\le\epsilon_1$ and $\mathbb E\lVert B_i(x)-B_i(y)\rVert^2\le L^2\lVert x-y\rVert^2$, then
$$\mathbb E\lVert v_k-\nabla f(x_k)\rVert^2\le\frac{kL^2\epsilon_1^2}{|S_2|}+\mathbb E\lVert v_0-\nabla f(x_0)\rVert^2$$
### SPIDER-SFO (Option I)
$$x_{k+1}=x_k-\eta\,\frac{v_k}{\lVert v_k\rVert},\qquad \eta=\frac{\epsilon}{Ln_0}$$
so $\epsilon_1=\eta$ exactly.
> Check script (400 runs, $q=25$, $|S_2|=4$, $\eta=0.02$): error $1.6\times10^{-3}$ at $k=24$, bound $0.21$; plain minibatch error 1.32 vs SPIDER $8\times10^{-4}$ on average.

## rates | Complexity (Theorems 1–3)
- Online: $O(\epsilon^{-3})$ stochastic gradients for $\lVert\nabla f\rVert\le\epsilon$ (Theorem 1), vs $O(\epsilon^{-4})$ for SGD
- Finite sum: $O(n+n^{1/2}\epsilon^{-2})$ (Theorem 2)
- Lower bound: $\Omega(L\Delta\,n^{1/2}\epsilon^{-2})$ for finite sums with $n\le O(\epsilon^{-4})$ (Theorem 3), so SPIDER is near-optimal there
- SPIDER-SFO⁺ reaches an $(\epsilon,O(\sqrt\epsilon))$-SOSP in $\tilde O(\min(n^{1/2}\epsilon^{-2}+\epsilon^{-2.5},\epsilon^{-3}))$; zeroth-order version costs an extra factor $d$
