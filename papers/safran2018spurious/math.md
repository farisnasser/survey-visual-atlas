## slice | The objective and a certified-style spurious minimum
### Teacher–student population loss (Eq. 2)
Teacher with $k$ orthonormal neurons $v_j$, student with $n\ge k$ neurons, unit output weights, $x\sim\mathcal N(0,I)$:
$$F(w_1,\dots,w_n)=\tfrac12\,\mathbb E_x\Big(\sum_{i=1}^n[w_i^\top x]_+-\sum_{j=1}^k[v_j^\top x]_+\Big)^2$$
Global minimum $F=0$ at $w_i=v_i$ (up to permutation).
### Closed form (Sec. 4.2)
$$\mathbb E\big[[w^\top x]_+[v^\top x]_+\big]=\frac{\lVert w\rVert\lVert v\rVert}{2\pi}\big(\sin\theta+(\pi-\theta)\cos\theta\big)$$
so $F$ and its gradient are exact, no sampling.
### Theorem 1
Spurious local minima exist for $k=n\in\{6,\dots,20\}$ and for $(k,n)=(8,9)$ and $(k,k+1)$, $10\le k\le19$ (computer-assisted proof).
> On screen: a point found by GD for $k=n=8$ with $F=0.0215$, $\lVert\nabla F\rVert=4\times10^{-10}$, $\lambda_{\min}(\nabla^2F)=0.021>0$; slice spanned by the direction to the nearest global minimum and a random orthogonal direction.

## odds | The experiment (Sec. 3)
1000 GD runs per $(k,n)$, Xavier init $w_i\sim\mathcal N(0,I/k)$, step 0.1, stop when every $\lVert\nabla_{w_i}F\rVert\le10^{-9}$, ignore $F<10^{-3}$; then verify each candidate with Lemmas 1–2.
### Certificate (Lemma 1)
If $\lVert\nabla F\rVert\le\epsilon$, $\nabla^2F\succeq\lambda_{\min}I$ and third derivatives are bounded by $B$, then
$$F(w+tu)\ge F(w)-t\epsilon+\tfrac{\lambda_{\min}}2t^2-\tfrac B6t^3$$
so a true local minimum lies within a small radius.
### Takeaway
Strict-saddle avoidance (Lee 2016) does not help: these are genuine minima with positive curvature. Over-parameterization by 2+ neurons removed them in the tested sizes (observed, not proved).

## cdf | Fig. 1
The empirical probability of converging to a minimum with objective value below a threshold. A curve that reaches 1 only at positive $F$ means many runs stop at spurious minima.
