## model | Sampled cubic model and Cubic-Subsolver (Algorithms 1, 3)
At $x_t$ sample $S_1$ gradients and $S_2$ Hessians:
$$g=\frac1{|S_1|}\sum_{i\in S_1}\nabla f(x_t;\xi_i),\qquad B=\frac1{|S_2|}\sum_{i\in S_2}\nabla^2f(x_t;\xi_i)$$
$$m_t(\Delta)=g^\top\Delta+\tfrac12\Delta^\top B\Delta+\tfrac\rho6\lVert\Delta\rVert^3$$
### Cubic-Subsolver (Alg. 3), Hessian-vector products only
- if $\lVert g\rVert\ge\ell^2/\rho$: Cauchy step $\Delta=-R_c\,g/\lVert g\rVert$, $R_c=-\frac{g^\top Bg}{\rho\lVert g\rVert^2}+\sqrt{\big(\frac{g^\top Bg}{\rho\lVert g\rVert^2}\big)^2+\frac{2\lVert g\rVert}\rho}$
- else gradient descent on $m$ with $\tilde g=g+\sigma\zeta$ and step $\frac1{20\ell}$
> Check script: $R_c$ equals the line-search minimizer of $m$ along $-g$ in six random instances; perturbed GD reaches the exact global model minimum to within $4\times10^{-6}$.

## outer | Outer loop and rate
$x_{t+1}=x_t+\Delta_t$; if the predicted decrease $\Delta m\ge-\frac1{100}\sqrt{\epsilon^3/\rho}$, call Cubic-Finalsolver (Alg. 2) and stop.
**Result:** an $\epsilon$-second-order stationary point ($\lVert\nabla f\rVert\le\epsilon$, $\lambda_{\min}\ge-\sqrt{\rho\epsilon}$) in
$$\tilde O(\epsilon^{-3.5})\ \text{stochastic gradient and Hessian-vector evaluations, vs } \tilde O(\epsilon^{-4})\ \text{for perturbed SGD}$$
### Toy on screen
$$f(x,y)=\tfrac12x^2+\cos y,\qquad \nabla^2f=\mathrm{diag}(1,-\cos y)$$
> Check script: from $(1.2,0)$ with $|S_1|=|S_2|=20$, all 20 runs escape the saddle line and stop near $y=\pm\pi$.

## oned | The notes' 1D example
$f(s)=\cos s$ at $s=0$, sampled $g=0.01$, $B=-0.9$, $\rho=1$:
$$g+B\Delta+\tfrac\rho2\Delta|\Delta|=0\ \Rightarrow\ \Delta^*\approx-1.811040$$
$m(\Delta^*)\approx-0.504055$, noiseless model $q(\Delta^*)\approx-0.649939$, true change $\cos\Delta^*-1\approx-1.237943$.
