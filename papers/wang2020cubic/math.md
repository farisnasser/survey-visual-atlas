## race | CRm (Algorithm 1)
**Cubic step**
$$s_{k+1}=\arg\min_s\ \nabla f(x_k)^\top s+\tfrac12s^\top\nabla^2f(x_k)s+\tfrac M6\lVert s\rVert^3,\qquad y_{k+1}=x_k+s_{k+1}$$
**Momentum step**
$$\beta_{k+1}=\min\{\beta,\ \lVert\nabla f(y_{k+1})\rVert,\ \lVert y_{k+1}-x_k\rVert\},\qquad v_{k+1}=y_{k+1}+\beta_{k+1}(y_{k+1}-y_k)$$
**Monotone step:** $x_{k+1}=\arg\min_{x\in\{y_{k+1},v_{k+1}\}}f(x)$. Inputs: $x_0=y_0$, $\beta<1$, $M>L_2$.
### Guarantees
- Theorem 1: an $\epsilon$-SOSP within $O(\epsilon^{-3/2})$ iterations (same order as CR, optimal for this class)
- Theorem 2: local quadratic convergence under a local error bound $\mathrm{dist}(x,\mathcal X)\le\kappa\lVert\nabla f(x)\rVert$
- Theorems 3–4: an inexact (subsampled) cubic step keeps the order; Hessian sample complexity
The acceleration is shown experimentally (Sec. 5), not as a better worst-case exponent.
### Toy on screen
$$f(x,y)=\tfrac12(1-x)^2+5(y-x^2)^2,\qquad \nabla^2f=\begin{pmatrix}1-20y+60x^2&-20x\\-20x&10\end{pmatrix}$$
> Check script: from $(-0.4,1.4)$, $M=180$: CR 65 iterations, CRm 61 (momentum chosen in all 61); $M=60$: 42 vs 39.

## saddle | Zero gradient, negative curvature
At $\nabla f=0$ with $\lambda_{\min}<0$ the cubic model $\tfrac12s^\top Hs+\tfrac M6\lVert s\rVert^3$ is minimized at $s=\pm\frac{2|\lambda_{\min}|}{M}u_{\min}$, a strict decrease.
### Notes' 1D example
$f=-x^2/2+x^4/4$, $M=2$: from $x_0=0$ the step lands at $\pm1$, $f(1)=-\tfrac14$. Momentum check: $y=0.8$, $y_{\text{prev}}=0.5$, $\beta=0.2$ give $v=0.86$ with $f(0.86)\approx-0.2330<f(0.8)=-0.2176$, so $v$ is kept.
