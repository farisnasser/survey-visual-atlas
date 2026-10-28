## convex | All local minima are global
### Theorem 1
Deep linear network $L(W_1,\dots,W_L)=\frac1N\sum_i \ell(W_L\cdots W_1x^{(i)},y^{(i)})$. If
- $\ell$ is convex and differentiable, and
- the thinnest layer is the input or the output layer ($\min_k d_k\ge\min(d_0,d_L)$),
then every local minimum of $L$ is a global minimum.
### The bridge (Theorem 3)
For any differentiable $f$ of the product $A=W_L\cdots W_1$, under the width condition:
$$\hat W\ \text{local min of}\ f(W_L\cdots W_1)\ \Longrightarrow\ \nabla f(\hat A)=0$$
Convexity of $f$ then makes $\hat A$ a global minimizer.
### Toy model on screen
$$F(w_1,w_2)=\ell(w_2w_1-2),\quad \ell\in\{(\cdot)^2,\ \log\cosh(\cdot)\}$$
The right chart shows $\ell$ as a function of the product: balls settle where $\ell'(A)=0$.
> Critical points are not the issue: for $L\ge3$ the zero point is always critical (Sec. 3). The claim is about local minima.

## kink | Differentiability is necessary
### Theorem 2 / Eq. 22–23
$$f(x,y)=|x|+(1-y)_+-1,\qquad F(W_1,W_2)=f(W_2W_1),\ W_1\in\mathbb R,\ W_2\in\mathbb R^2$$
$f$ is convex and Lipschitz but not differentiable. $\hat W_1=0,\ \hat W_2=(1,0)^\top$ is a local minimizer with $F=0$, while $\min F=-1$.
### Slice on screen
$W_1=\alpha,\ W_2=(1-\beta,\beta)$:
$$F(\alpha,\beta)=|\alpha(1-\beta)|+(1-\alpha\beta)_+-1$$
- $F\ge0$ near $(0,0)$ (checked on a radius-0.25 disc), so it is a bad local minimum
- global minima: $\beta=1,\ \alpha\ge1$ with $F=-1$
The proof of Theorem 3 needs a single-valued $\nabla f(\hat A)$; a subdifferential breaks it.
