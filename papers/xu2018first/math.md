## fhat | NEON (Algorithm 1)
Centered function at the current point $x$ (Eq. 13):
$$\hat f_x(u)=f(x+u)-f(x)-\nabla f(x)^\top u,\qquad \hat f_x(0)=0,\ \nabla\hat f_x(0)=0,\ \nabla^2\hat f_x(0)=\nabla^2f(x)$$
$u_0$ uniform on the sphere of radius $r$, then gradient descent on $\hat f_x$:
$$u_{\tau+1}=u_\tau-\eta\big(\nabla f(x+u_\tau)-\nabla f(x)\big)\approx(I-\eta\nabla^2f(x))\,u_\tau$$
Return the $u_\tau$ (with $\lVert u_\tau\rVert\le U$) minimizing $\hat f_x$ if $\min\hat f_x(u_\tau)\le-2.5F$, else 0.
### NEON⁺ (Algorithm 2)
$$y_{\tau+1}=u_\tau-\eta\nabla\hat f_x(u_\tau),\qquad u_{\tau+1}=y_{\tau+1}+\zeta(y_{\tau+1}-y_\tau),\qquad \zeta=1-\sqrt{\eta\gamma}$$
(NCFind searches the history when the iterate becomes too non-convex.)
### Toy on screen
$$f(x,y)=\tfrac12x^2-0.3y^2+\tfrac14y^4+0.4xy^2,\qquad x=(0.15,0.05),\ \lambda_{\min}(\nabla^2f(x))=-0.4736$$

## speed | Theorems 2–3
- NEON: if $\lambda_{\min}(\nabla^2f(x))\le-\gamma$, returns $u$ with $\frac{u^\top\nabla^2f(x)u}{\lVert u\rVert^2}\le-\tilde\Omega(\gamma)$ after $\tilde O(1/(\eta\gamma))$ steps; returning 0 certifies $\lambda_{\min}\ge-\gamma$ w.h.p.
- NEON⁺: $t=\sqrt{\hat c\log(\cdot)/(\eta\gamma)}$ steps, the $\sqrt{\cdot}$ speed-up of Lanczos over the power method
- Stochastic version: minibatch $F_S$; combined with SGD/SCSG (NEON-A), the best total time is $\tilde O(d/\epsilon^{3.5})$, almost linear in $d$
> Check script ($d=50$): $\gamma=0.1\to0.001$: NEON 151 → 19 331 iterations (slope −1.05), NEON⁺ 50 → 713 (slope −0.58).
