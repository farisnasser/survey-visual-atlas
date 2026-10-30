## descent | Smoothness, the descent lemma and the gradient method (Ch. 1–2)
$f$ has an $L$-Lipschitz gradient: $\lVert\nabla f(x)-\nabla f(y)\rVert\le L\lVert x-y\rVert$. Then (descent lemma)
$$\big|f(y)-f(x)-\langle\nabla f(x),y-x\rangle\big|\le\tfrac L2\lVert y-x\rVert^2$$
and a gradient step $x^+=x-h\nabla f(x)$ gives
$$f(x^+)\le f(x)-h\Big(1-\frac{Lh}2\Big)\lVert\nabla f(x)\rVert^2,$$
a decrease for $0<h<2/L$, largest ($\lVert\nabla f\rVert^2/2L$) at $h=1/L$. Summing over $k+1$ steps with $f\ge f^*$:
$$\min_{0\le i\le k}\lVert\nabla f(x_i)\rVert^2\le\frac{2L\,(f(x_0)-f^*)}{k+1},$$
so $O(L(f(x_0)-f^*)\epsilon^{-2})$ steps reach $\lVert\nabla f\rVert\le\epsilon$ — no convexity needed (the $O(k^{-1/2})$ gradient rate the cubic-regularization paper cites from this book). With convexity, $\nabla f(x)=0$ implies a global minimum and faster (accelerated) rates follow; without it a stationary point can be a saddle.
### Here
$f(x,y)=\tfrac14x^2-\tfrac14\cos2y$, Hessian $\mathrm{diag}(\tfrac12,\cos2y)$, $L=1$, $f^*=-\tfrac14$.
> Check script: sampled gradient-Lipschitz ratio 0.95 ≤ 1; descent lemma and the one-step bound hold for $h=0.5,1,1.9$; along the $h=1/L$ run from $(1.8,1.3)$ the bound on $\min\lVert\nabla f\rVert^2$ holds at every step and $f$ is monotone; with $h=2.2/L$ $f$ increases at some step.
> The book was not available for a line-by-line check; these are the standard results the survey cites it for, without theorem numbers.
