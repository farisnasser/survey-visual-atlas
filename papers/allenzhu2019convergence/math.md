## slice | Theorem 3: no bad critical point near initialization
### Model
$h_{i,0}=\phi(Ax_i)$, $h_{i,\ell}=\phi(W_\ell h_{i,\ell-1})$, $\hat y_i=Bh_{i,L}$, $\phi=\mathrm{ReLU}$; $A,B$ fixed random, $W_1..W_L$ trained,
$$F(\vec W)=\tfrac12\sum_i\lVert y_i-Bh_{i,L}\rVert^2$$
### Gradient lower bound
With high probability, for every $\vec W$ within $\omega\le 1/\mathrm{poly}(n,L,\delta^{-1})$ of initialization:
$$\lVert\nabla F(\vec W)\rVert_F^2\ \ge\ \Omega\!\Big(\frac{\delta\,m}{d\,n^2}\Big)\,F(\vec W)$$
Positive loss forces a non-zero gradient: there is no bad critical point in that neighbourhood, even though the Hessian may have negative eigenvalues (Fig. 1).
> On screen: $L=3$, $m=256$, $n=10$, $d=10$; slice $F(W_t+a\,u_1+b\,u_2)$ with $u_1=-\nabla F/\lVert\nabla F\rVert$, $u_2$ the most-negative-curvature direction (power iteration on Hessian–vector products).

## width | Theorems 1, 2 and 4
### Semi-smoothness (Theorem 4)
$$F(\vec W+\vec W')\le F(\vec W)+\langle\nabla F(\vec W),\vec W'\rangle+O\big(\cdots\big)\sqrt{F(\vec W)}\,\lVert\vec W'\rVert+O\big(nL^2m/d\big)\lVert\vec W'\rVert^2$$
Combined with Theorem 3 this gives $F(\vec W^{(t+1)})\le(1-\Omega(\eta\delta m/(dn^2)))F(\vec W^{(t)})$.
### Theorem 1 (GD)
If $m\ge\mathrm{poly}(n,L,\delta^{-1})\cdot d$ and $\eta=\Theta\big(\tfrac{d\delta}{\mathrm{poly}(n,L)\,m}\big)$, then $F(\vec W^{(T)})\le\varepsilon$ after
$$T=\frac{\mathrm{poly}(n,L)}{\delta^2}\log\frac1\varepsilon$$
Theorem 2: the same for minibatch SGD. Weights stay close to initialization, which is why the local bounds keep applying.

## separation | The role of δ
Assumption: $\lVert x_i-x_j\rVert\ge\delta$ for all $i\ne j$ (unit-norm inputs). The width and iteration bounds are polynomial in $1/\delta$; with $\delta=0$ and conflicting labels zero loss is impossible.
> Offline: $\delta=1$: 28 iterations, $0.3$: 261, $0.1$: 1733, $0.03$: not reached in 3000.
