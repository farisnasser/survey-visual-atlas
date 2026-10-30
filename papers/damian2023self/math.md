## stages | The four stages (Sec. 4.1)
Sharpness $S(\theta)=\lambda_{\max}(\nabla^2L(\theta))$, top eigenvector $u$. Near a reference point $\theta^\star$ with $S(\theta^\star)=2/\eta$, track
$$x_t=u\cdot(\theta_t-\theta^\star),\qquad y_t=\nabla S\cdot(\theta_t-\theta^\star)\approx S(\theta_t)-2/\eta$$
with progressive-sharpening coefficient $\alpha=-\nabla L\cdot\nabla S>0$ and $\beta=\lVert\nabla S\rVert^2$.
1. **Progressive sharpening:** $y_{t+1}-y_t\approx\eta\alpha$.
2. **Blow-up:** $x_{t+1}\approx-(1+\eta y_t)x_t$ grows once $y_t>0$.
3. **Self-stabilization:** the cubic Taylor term adds $\tfrac{x^2}{2}\nabla S$ to the gradient, so $y_{t+1}-y_t\approx\eta(\alpha-\tfrac\beta2x_t^2)$ turns negative once $|x|>\sqrt{2\alpha/\beta}$.
4. **Return to stability:** $y<0$, $|x|$ shrinks.
$$x_{t+1}=-(1+\eta y_t)\,x_t,\qquad y_{t+1}=y_t+\eta\Big(\alpha-\frac\beta2x_t^2\Big)\qquad(3)$$
### On screen: App. B toy (x, y part)
$L(x,y)=(2/\eta+\sqrt\beta\,y)\tfrac{x^2}{2}-\tfrac{\alpha}{\sqrt\beta}y$; GD on it is exactly Eq. (3).
> Check script ($\eta=1$, $\alpha=0.002$, $\beta=1$): mean $S$ over 450 steps $=1.9992$ vs $2/\eta=2$; $S\in[1.939,2.053]$; cycles of 105, 109, 113 steps.

## orbit | Simplified ODE and its invariant (Sec. 4.2)
$$X'=XY,\qquad Y'=\alpha-\tfrac\beta2X^2\qquad(4)$$
A Lotka–Volterra system with fixed point $(\delta,0)$, $\delta=\sqrt{2\alpha/\beta}$. **Lemma 3:** with $h(z)=z-\log z-1$,
$$g(X,Y)=h\Big(\frac{\beta X^2}{2\alpha}\Big)+\frac{Y^2}{\alpha}$$
is conserved. **Corollary 1:** the swings in $X$ and $Y$ are bounded by $\delta$ and $\sqrt\alpha$ times $\sqrt{\log(\delta/X(0))}$-type factors: a smaller initial oscillation gives a larger loop. Over a cycle $\overline{Y}=0$ (mean sharpness $2/\eta$) and $\overline{X^2}=\delta^2$.
### Main result (Sec. 5)
GD at the EoS tracks the **constrained trajectory** $\theta^\dagger_{t+1}=\mathrm{proj}_{\mathcal M}(\theta^\dagger_t-\eta\nabla L(\theta^\dagger_t))$, $\mathcal M=\{S\le2/\eta,\ \nabla L\cdot u=0\}$: Theorem 1 predicts loss, sharpness and deviation from it up to higher-order terms for $T=O(\epsilon^{-1})$ steps, under Assumptions 1–5.
> Check script: $g$ conserved to $3\times10^{-14}$ along the ODE. The discrete map's period-2 orbit is a weakly unstable focus (two-step eigenvalues $(1-\delta^2)\pm2i\delta$, modulus $1.004$), so GD spirals out slowly: $g$ rises from 0.38 to 2.0 in 450 steps.
