## quad | Stability of GD on a quadratic (Sec. 2)
For $f(x)=\tfrac12x^\top Ax+b^\top x+c$ the error along an eigenvector $q$ with eigenvalue $a$ evolves as
$$q^\top(x_{t+1}-x^\star)=(1-\eta a)\,q^\top(x_t-x^\star)$$
so GD diverges along $q$ whenever $a>2/\eta$ (Proposition 1, App. B).
### Sharpness
$$S(\theta)=\lambda_{\max}\big(\nabla^2L(\theta)\big)$$
A neural loss is not quadratic, but its second-order Taylor model at any point is; if $S>2/\eta$ there, GD on that model would diverge along the top eigenvector.
### On screen (Fig. 2 of the paper)
$f=\tfrac12(20x^2+y^2)$: $|1-20\eta|<1\iff\eta<0.1$.
> Check script: $\eta=0.11$ gives $|1-20\eta|=1.2$, $|x_{60}|=2.8\times10^3$; $\eta=0.09$ converges.

## mlp | Progressive sharpening and the edge of stability (Sec. 3)
- **Progressive sharpening (3.1):** while $S<2/\eta$, full-batch GD (equivalently gradient flow) has an "overwhelming tendency" to increase $S$.
- **Edge of stability (3.2):** once $S$ reaches $2/\eta$ it hovers right at or just above $2/\eta$; the loss is non-monotone over short timescales but decreases over long ones.
- Holds across architectures (FC, CNN, VGG, ResNet, Transformer) and losses (MSE, cross-entropy); for cross-entropy the sharpness eventually drops late in training.
- Empirical paper: no convergence theorem. SGD does not flatline at a single value (Sec. 6).
### Our replication
tanh MLP $10\!-\!48\!-\!48\!-\!1$, 128 Gaussian inputs, MSE, $S$ by power iteration on Hessian-vector products.
> Check script: initial $S=4.05$. $\eta=0.1$: saturates at 7.75 $<20$. $\eta=0.3,0.36,0.42$: late $S=6.66,\,5.55,\,4.75$ vs $2/\eta=6.67,\,5.56,\,4.76$; loss non-monotone in 73, 42, 96 of 300 ten-step windows, still falls to $\le5\times10^{-6}$.

## mom | Maximum stable sharpness with momentum (Eq. 1)
Heavy ball $v_{t+1}=\beta v_t-\eta\nabla f(x_t),\ x_{t+1}=x_t+v_{t+1}$ and Nesterov $v_{t+1}=\beta v_t-\eta\nabla f(x_t+\beta v_t)$ diverge on a quadratic once
$$S>\text{MSS}_{\text{Polyak}}=\frac{2+2\beta}{\eta},\qquad S>\text{MSS}_{\text{Nesterov}}=\frac{2+2\beta}{\eta(1+2\beta)}$$
(Polyak case previously in Goh 2017; Nesterov case new, App. B). Fig. 7: with momentum the sharpness rises to the MSS and plateaus there.
> Check script: both formulas match the spectral-radius boundary of the $2\times2$ iteration matrix for $\beta\in\{0.5,0.9\}$.
