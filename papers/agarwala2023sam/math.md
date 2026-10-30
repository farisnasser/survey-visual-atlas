## uv | Unnormalized SAM (Eq. 6)
$$\theta_{t+1}=\theta_t-\eta\,\nabla L\big(\theta_t+\rho\nabla L(\theta_t)\big)$$
(The perturbation is not normalized; Foret et al.'s SAM divides by $\lVert\nabla L\rVert$.) In the quadratic regression model $f=\theta\cdot J+\tfrac12Q(\theta,\theta)$ the residual dynamics are governed by the modified kernel $(1+\rho JJ^\top)JJ^\top$ (Eq. 7).
### On screen
$L(u,v)=\tfrac12(uv-1)^2$; on $uv=1$ the Hessian has eigenvalues $0$ and $u^2+v^2$.
> Check script ($\eta=0.3$, start $(2.2,0.1)$): end sharpness GD 4.915; SAM $\rho=0.05$: 4.196; $\rho=0.1$: 2.956; $\rho=0.2$: 2.000 (the flattest point, on the negative branch). Each is below its own edge (6.667, 5.275, 4.574, 3.792).

## mlp | SAM edge of stability (Sec. 2.2.2)
On $\tfrac12x^\top Hx$:
$$x_{t+1}-x_t=-\eta\,(H+\rho H^2)\,x_t\qquad(12)$$
so $x_t\to0$ iff $\eta(\lambda_{\max}+\rho\lambda_{\max}^2)<2$. **Theorem 2.2** formalizes this for any $C^2$ model with MSE loss near a zero-residual point: convergent if $\eta\lambda_i(1+\rho\lambda_i)<2-\epsilon$ for all NTK eigenvalues, divergent if $>2+\epsilon$ for one, so the SAM edge is $\eta\lambda_{\max}(1+\rho\lambda_{\max})\approx2$ (Eq. 13).
- Theorem 2.1: at initialization the full-batch SAM change of the singular values is negative but proportional to $(z\cdot w_\alpha)^2$, so the regularization of large eigenvalues can be weak early (small $\eta$).
- Theorem 2.3: with SGD (batch fraction $\beta$), the first-step averages gain extra terms, i.e. minibatching strengthens the regularization.
- Experiments (reported): a fully connected net on 2-class CIFAR (MSE, full batch) and WideResNet 28-10 on CIFAR-10; the SAM-normalized eigenvalue $\eta(\lambda_{\max}+\rho\lambda_{\max}^2)$ hovers at 2 during training.
> Check script (tanh MLP, $\eta=0.3$): late sharpness GD 6.657 vs $2/\eta=6.667$; SAM $\rho=0.05$: 5.271 vs 5.275; SAM $\rho=0.1$: 4.571 vs 4.574.

## edge | The SAM threshold
$$\lambda_{\text{SAM}}=\frac{-1+\sqrt{1+8\rho/\eta}}{2\rho}<\frac2\eta$$
### Notes' example
$\eta=0.5$, $\rho=0.2$: $0.2\lambda^2+\lambda-4=0$, $\lambda_{\text{SAM}}\approx2.62<4$. At $\lambda=3$ GD multiplies by $-0.5$ (stable), SAM by $-1.4$ (unstable).
