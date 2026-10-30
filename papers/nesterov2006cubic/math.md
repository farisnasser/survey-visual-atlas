## saddle | Cubic regularization of Newton's method
Assumption 1: the Hessian is $L$-Lipschitz, $\lVert\nabla^2 f(x)-\nabla^2 f(y)\rVert\le L\lVert x-y\rVert$. The step is the **global** minimizer of the cubic model (Eq. 2.4):
$$T_M(x)\in\arg\min_y\ \langle\nabla f(x),y-x\rangle+\tfrac12\langle\nabla^2 f(x)(y-x),y-x\rangle+\tfrac M6\lVert y-x\rVert^3$$
and the method is $x_{k+1}=T_{M_k}(x_k)$ with $M_k\in[L_0,2L]$ (scheme 3.3); it is monotone, $f(x_{k+1})\le f(x_k)$.
### Saddles repel (Lemma 6, Theorem 2)
If $\nabla f(x^*)=0$ and $\lambda_{\min}(\nabla^2f(x^*))<0$, there are $\epsilon,\delta>0$ such that from any $x_i$ with $\lVert x_i-x^*\rVert\le\epsilon$, $f(x_i)\ge f(x^*)$ the next point has $f(x_{i+1})\le f(x^*)-\delta$, even when $x_i=x^*$. With bounded level sets every limit point satisfies $\nabla f=0$, $\nabla^2 f\succeq0$ (Theorem 2).
### Toy model
$$f(x,y)=\tfrac12x^2-\tfrac12y^2+\tfrac14y^4+0.3\,xy^2$$
strict saddle at $0$ ($\lambda=\pm1$), minima at $(-0.3y_*^2,\pm y_*)$ with $y_*^2=1/0.82$, $f=-0.305$.
> Check script: from $(1.2,0.03)$ pure Newton converges to the saddle; cubic steps with $M=L=10$ reach the minimum with $\lVert\nabla f\rVert<10^{-6}$ after 12 steps, gradient descent (step $1/8$) after 167.

## model | The cubic model is an upper bound (Lemma 1, Lemma 4)
From the Lipschitz Hessian (Eq. 2.3):
$$\Big|f(y)-f(x)-\langle\nabla f(x),y-x\rangle-\tfrac12\langle\nabla^2f(x)(y-x),y-x\rangle\Big|\le\tfrac L6\lVert y-x\rVert^3$$
so for $M\ge L$ the cubic model lies above $f$ and $f(T_M(x))\le\bar f_M(x)$ (Lemma 4). The quadratic Newton model ($M=0$) is unbounded below whenever $\nabla^2f(x)$ has a negative eigenvalue.
### Computing the step (Sec. 5)
$T_M(x)$ solves $\big(\nabla^2f(x)+\tfrac M2 r I\big)s=-\nabla f(x)$ with $r=\lVert s\rVert$ and $\nabla^2f(x)+\tfrac M2rI\succeq0$ (Prop. 1); finding $r$ is a one-dimensional convex problem.
> Check script: with $M=L$ the model is above $f$ on the whole box; with $M=0$ it falls below $f$ by up to $1.9$. At the exact saddle the step has length $2|\lambda_{\min}|/M=0.2$ along the negative eigenvector.

## rate | Global rate (Theorem 1) and local rate (Theorem 3)
With $\mu_M(x)=\max\Big\{\sqrt{\tfrac{2}{L+M}\lVert\nabla f(x)\rVert},\ -\tfrac{2}{2L+M}\lambda_{\min}(\nabla^2f(x))\Big\}$,
$$\min_{1\le i\le k}\mu_L(x_i)\le\frac83\Big(\frac{3(f(x_0)-f^*)}{2kL_0}\Big)^{1/3}$$
so $\min_i\lVert\nabla f(x_i)\rVert=O(k^{-2/3})$, against $O(k^{-1/2})$ for the gradient method. Near a non-degenerate local minimum the convergence is quadratic (Theorem 3). Section 4 gives global rates for star-convex, gradient-dominated and other classes.
- In $\epsilon$ terms: $O(\epsilon^{-3/2})$ iterations to $\lVert\nabla f\rVert\le\epsilon$, $\lambda_{\min}\ge-\sqrt{L\epsilon}$ (the form later papers quote).
> Theorem numbers follow the 2003 CORE discussion paper version (the journal version is Math. Program. 108, 2006).
