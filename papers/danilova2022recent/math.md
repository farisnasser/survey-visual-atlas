## pl | Beyond stationarity: structure that gives global rates
The survey covers first-order complexity for finding $\epsilon$-stationary points (deterministic and stochastic, variance reduction, lower bounds), classes with global guarantees (PŁ, star-convexity, $\alpha$-weak quasi-convexity), higher-order (cubic regularization, tensor) and zeroth-order methods.
### Polyak–Łojasiewicz condition (Sec. 5.1, Eq. 36)
$$f(x)-f(x^\ast)\le\frac1{2\mu}\lVert\nabla f(x)\rVert^2\quad\forall x$$
Every stationary point is a global minimum (not necessarily unique). With $L$-smooth $f$, GD with step $1/L$: $f(x_k)-f^\ast\le e^{-\mu k/L}\big(f(x_0)-f^\ast\big)$. PŁ holds e.g. for $\tfrac12\lVert g(x)\rVert^2$ when $\lambda_{\min}(J_gJ_g^\top)\ge\mu>0$; it is also invoked for over-parameterized networks.
### Toy on screen
$$f(x,y)=x^2+3\sin^2x+y^2,\qquad\partial_x^2f=2+6\cos2x\in[-4,8]$$
non-convex, $L=8$, PŁ (the $x$-part is the example of Karimi, Nutini & Schmidt 2016).
> Check script: $\mu=0.1755$ (minimum of $\lVert\nabla f\rVert^2/(2(f-f^\ast))$ on $[-6,6]$, at $x=-2.20$); GD from 200 starts in $[-5,5]^2$ obeys the bound at every step for 400 steps; worst gap $8\times10^{-12}$ after 50 steps.
PŁ rules out saddles and spurious minima altogether, which is why it cannot describe general neural-network losses (compare Islamov et al. [57]).
