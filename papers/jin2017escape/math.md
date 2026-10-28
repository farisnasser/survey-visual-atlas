## kick | Perturbed gradient descent (Algorithm 2)
### Target: ε-second-order stationary point (Def. 6)
$$\lVert\nabla f(x)\rVert\le\epsilon,\qquad \lambda_{\min}\big(\nabla^2 f(x)\big)\ge-\sqrt{\rho\,\epsilon}$$
Assumption A1: $\nabla f$ is $\ell$-Lipschitz, $\nabla^2 f$ is $\rho$-Lipschitz.
### Algorithm 2
- GD step $x_{t+1}=x_t-\eta\nabla f(x_t)$ with $\eta=c/\ell$
- if $\lVert\nabla f(x_t)\rVert\le g_{\text{thres}}$ and no kick in the last $t_{\text{thres}}$ steps: save $\tilde x=x_t$, set $x_t=\tilde x+\xi$, $\xi\sim\mathrm{Unif}(B_0(r))$
- if $t_{\text{thres}}$ steps after a kick $f$ has not dropped by $f_{\text{thres}}$: return $\tilde x$
### Toy model on screen
$$f(x,y)=\tfrac12x^2+\cos y-1,\qquad \nabla^2f(0)=\mathrm{diag}(1,-1),\quad \ell=\rho=1$$
Starting on the $x$-axis, plain GD converges to the saddle; PGD escapes to $y=\pm\pi$, $f=-2$.

## pancake | Why kicks rarely fail (Sec. 5)
Let $\mathcal X_{\text{stuck}}$ be the kicks after which GD does not decrease $f$ by $f_{\text{thres}}$ within $t_{\text{thres}}$ steps. For a quadratic it is a slab $\{|\langle \xi,e_1\rangle|\ \text{small}\}$ around the stable directions; for general $f$ it bends but stays thin: two points separated by enough along $e_1$ cannot both be stuck (Lemma 11). Its volume fraction is small, so the kick succeeds with high probability.
### Toy model
$$f(x,y,z)=\tfrac12(x^2+z^2)+\cos\!\big(y-\tfrac12x^2\big)-1$$
> Offline: stuck fraction 2.0% at $t_{\text{thres}}=20$; the stuck points follow $y\approx x^2/2$ (correlation 0.96).

## band | The 2D band (Fig. 2)
$$f(x,y)=\tfrac12x^2+\gamma(\cos y-1)$$
Weaker curvature $\gamma$ ⇒ slower escape ⇒ wider band; longer window $t_{\text{thres}}$ ⇒ thinner band.

## dimension | Theorem 3
With probability $\ge1-\delta$, PGD outputs an $\epsilon$-SOSP within
$$O\!\Big(\frac{\ell\,(f(x_0)-f^\star)}{\epsilon^2}\,\log^4\!\Big(\frac{d\,\ell\,\Delta_f}{\epsilon^2\delta}\Big)\Big)\ \text{iterations}$$
the GD rate for first-order stationarity, up to polylog$(d)$. A random kick of radius $r$ has a component $\approx r/\sqrt d$ along $e_1$; growing it to $O(1)$ at rate $(1+\eta\gamma)$ costs $\approx\log(\sqrt d/r)/\log(1+\eta\gamma)$ steps.
> Second-order stationarity is not global optimality; Corollary 4 adds a strict-saddle assumption to land near local minima.
