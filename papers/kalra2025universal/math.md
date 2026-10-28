## bounce | The UV model and the EoS threshold
### Model (Sec. 2)
$$f(x)=\frac{1}{\sqrt{n_{\text{eff}}}}\,v^\top Ux,\qquad n_{\text{eff}}=n^{1-p},\qquad \ell=\tfrac12\,(f(x)-y)^2$$
trained by GD on a single example. Sharpness proxy $\lambda=\operatorname{Tr}H=\big(\lVert Ux\rVert^2+\lVert x\rVert^2\lVert v\rVert^2\big)/n_{\text{eff}}$ (the top eigenvalue follows it, App. C.5).
### Result 1
A necessary condition for edge-of-stability behaviour is
$$\eta>\eta_c=\frac{\sqrt{n_{\text{eff}}}}{\lVert x\rVert\,y}$$
### On screen: $n=1$, $x=1$, $y=2$
$L(u,v)=\tfrac12(uv-2)^2$. On the valley $uv=2$ the Hessian has eigenvalues $0$ and $u^2+v^2\ge 2y=4$. GD can only settle where $u^2+v^2<2/\eta$, impossible when $\eta>1/y=0.5=\eta_c$.
> Offline: η = 0.45 settles at λ = 4.000; η = 0.55 oscillates between 2.65 and 4.98.

## flow | Fixed-point analysis (Eqs. 1–2)
With residual $\tilde f_t=f_t-y$:
$$\tilde f_{t+1}=\tilde f_t\Big(1-\eta\lambda_t+\frac{\eta^2\lVert x\rVert^2}{n_{\text{eff}}}\tilde f_t(\tilde f_t+y)\Big)$$
$$\lambda_{t+1}=\lambda_t+\frac{\eta\lVert x\rVert^2\tilde f_t}{n_{\text{eff}}}\big(\eta\lambda_t\tilde f_t-4(\tilde f_t+y)\big)$$
- I: zero-loss line $\tilde f=0$, stable iff $\lambda<2/\eta$
- II: $(-y,0)$, the parameter origin, a saddle
- III, IV: unstable, function-space only
Sharpening vs reduction is decided by the sign of $\eta\lambda\tilde f-4(\tilde f+y)$.
> These two equations reproduce the full $n$-parameter UV model to $10^{-15}$ in the check script.

## chaos | Route to chaos (Sec. 5.2, Fig. 3)
For $\eta>\eta_c$ every point of line I is unstable and the trajectory is trapped on the EoS manifold between II and IV, where the dynamics reduce to a 1D cubic map. As $\eta$ grows: period 2, 4, … chaos (period doubling), up to $\eta_{\max}$ where training diverges (Corollary 5.1).
### Limits
Two-layer linear net, one example, MSE, full-batch GD. Real-data sharpness fluctuations show broad correlated bands rather than a clean cascade (Fig. 5).
