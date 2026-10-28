## valleys | No poor local minima
### Theorem 2.3 (deep linear networks)
For $\hat Y = W_{H+1}\cdots W_1 X$ and squared loss, assume $XX^\top$ and $XY^\top$ full rank, $d_y\le d_x$, and $\Sigma$ with distinct eigenvalues. For any depth and widths:
- the loss is non-convex and non-concave;
- every local minimum is a global minimum;
- every other critical point is a saddle.
### Toy model on the right
$$L(w_1,w_2) = (1 - w_1 w_2)^2$$
Global minima: the hyperbola $w_1 w_2 = 1$ (green).
### Contrast on the left
$$L = 0.22\,(x^2+y^2) + 0.3\,(1-\cos 4x\cos 4y)$$
A landscape the theorem rules out for deep linear nets: shallow pits with $L>0$.
> Kawaguchi, NeurIPS 2016, Sec. 2.3. Checked in scripts/check-math/kawaguchi2016deep.py.

## good | Strict saddle
### Strict saddle
A critical point with $\lambda_{\min}(\nabla^2 L) < 0$.
### Corollary 2.4, one hidden layer
For $H = 1$ every saddle has a strictly negative Hessian eigenvalue.
$$L(w_1,w_2)=(1-w_1w_2)^2,\qquad \nabla^2 L(0)=\begin{pmatrix}0&-2\\-2&0\end{pmatrix},\quad \lambda=\pm 2$$
Down-curving direction: $w_1 = w_2$.

## bad | Degenerate saddle
### Corollary 2.4, two or more hidden layers
For $H\ge 2$ there exist saddles whose Hessian has **no** negative eigenvalue.
$$L(w_1,w_2)=(1-w_1w_2^2)^2,\qquad \nabla^2 L(0)=\begin{pmatrix}0&0\\0&0\end{pmatrix}$$
It is still a saddle: $L(t,t) = (1-t^3)^2 < 1$ for small $t>0$. The decrease is third order, so gradients near the origin are tiny.
### Theorem 2.3 (iv)
If $\operatorname{rank}(W_H\cdots W_2)=p$, the Hessian at any saddle has a negative eigenvalue. The bad saddles live where that product loses rank (here $w_2 = 0$).

## race | Same start, same step
Gradient descent $w \leftarrow w - \eta\nabla L(w)$ with $\eta = 0.02$ from $(0.05, 0.04)$ on both surfaces.
- strict saddle: loss below $10^{-3}$ after 121 steps;
- bad saddle: after 363 steps (3×). From $(0.01, 0.008)$ the gap grows to 11×.
Near a strict saddle the escape is exponential, $\propto (1+2\eta)^t$; near the flat one the gradient is $O(\|w\|^2)$, so escape is polynomially slow.
