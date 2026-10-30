## race | SGD with HVP-RVR (Algorithms 1–2)
### Estimator (Alg. 1)
With probability $b$: reset $g=\frac1n\sum_{i\le n}\nabla F(x,z_i)$. Otherwise, with $x^{(k)}=x_{\text{prev}}+\tfrac kK(x-x_{\text{prev}})$,
$$g=g_{\text{prev}}+\sum_{k=1}^{K}\nabla^2F\big(x^{(k-1)},z^{(k)}\big)\big(x^{(k)}-x^{(k-1)}\big),\qquad K\propto\frac{\lVert x-x_{\text{prev}}\rVert^2}{b\,\epsilon^2}$$
Every query uses a **single point** (no two-point same-seed oracle). The estimate is biased but its error is controlled.
### Alg. 2
SGD with $\eta=\frac1{2L_1}$ and this estimator; output a uniformly random iterate.
### Toy on screen
$$f(x,y)=\tfrac14(x^2-1)^2+\tfrac12y^2+0.3\sin(2x)\,y,\qquad \nabla F=\nabla f+z,\quad \nabla^2F=\nabla^2f+\tfrac{Z+Z^\top}2$$

## estimator | Error of the two gradient-difference estimates
$$\mathbb E\lVert\nabla F(x,z_1)-\nabla F(x',z_2)-(\nabla f(x)-\nabla f(x'))\rVert^2=2d\sigma_g^2$$
$$\mathrm{Var}\Big[\sum_k\tfrac{Z_k+Z_k^\top}2\,\tfrac{\Delta}{K}\Big]=\frac{d+1}2\,\sigma_H^2\,\frac{\lVert\Delta\rVert^2}{K}\quad(d=2:\ 1.5\lVert\Delta\rVert^2/K)$$
> Check script: two-gradient MSE ≈ 4.0 for every $\lVert\Delta x\rVert$; HVP integral $4.05\times10^{-1}$ ($\lVert\Delta x\rVert=0.5$, $K=1$) down to $3.7\times10^{-5}$ ($0.02$, $K=16$).

## elbow | Upper and lower bounds
- Upper: $O(\epsilon^{-3})$ stochastic gradient + HVP queries for $\mathbb E\lVert\nabla f\rVert\le\epsilon$
- Lower: $\Omega(\epsilon^{-3})$ for **any** stochastic $p$-th order method, $p\ge2$, so the exponent drops once ($4\to3$) and then stays
- Noiseless $p$-th order methods: $\epsilon^{-(p+1)/p}$
- $(\epsilon,\gamma)$-SOSP: upper $O(\epsilon^{-3}+\epsilon^{-2}\gamma^{-2}+\gamma^{-5})$, lower $\Omega(\epsilon^{-3}+\gamma^{-5})$
> Footnote 1 cites Murty & Kabadi [2]: deciding whether a second-order stationary point is a local minimum is NP-hard.
