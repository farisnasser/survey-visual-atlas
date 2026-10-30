## ball | The heavy-ball method
$$x_{t+1}=x_t-\alpha\nabla f(x_t)+\beta(x_t-x_{t-1})$$
(the survey's Eq. 5). With $v_t=x_t-x_{t-1}$: $v_{t+1}=\beta v_t-\alpha\nabla f(x_t)$, a ball with friction rolling on $f$.
### Quadratic analysis
$f=\tfrac12x^\top Hx$, eigenvalues $\lambda_i\in[\mu,L]$. Each eigen-coordinate obeys $z_{t+1}=(1+\beta-\alpha\lambda_i)z_t-\beta z_{t-1}$, with characteristic polynomial $r^2-(1+\beta-\alpha\lambda_i)r+\beta$. The choice
$$\alpha=\frac{4}{(\sqrt L+\sqrt\mu)^2},\qquad\beta=\Big(\frac{\sqrt L-\sqrt\mu}{\sqrt L+\sqrt\mu}\Big)^2$$
makes every root have modulus at most $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$, $\kappa=L/\mu$; GD with its best step $2/(L+\mu)$ only achieves $\frac{\kappa-1}{\kappa+1}$.
> Check script: worst spectral radius over $[\mu,L]$ equals $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ for $\kappa=10,25,100$; at $\kappa=25$ from $(1.4,1.8)$: GD 126 steps to $f<10^{-8}$, heavy ball 35 ($\alpha=0.556$, $\beta=0.444$).

## kappa | What the 1964 paper is and is not
- Polyak studied accelerating iterative solvers for $P(x)=0$ (optimization as $P=\nabla f$) by two-step (multi-step) methods; the heavy ball is the origin of momentum.
- The rate improvement is a statement about quadratics / strongly convex problems near a minimum. Near a strict saddle with curvature $-a$, $p(1)=-\alpha a<0$, so a root exceeds 1 and the unstable component grows (the notes' teaching example; the escape theory is Jin et al. 2018, [10]).
- The original 1964 article is not openly available; the formulas here are the standard ones (as in the notes and the survey's Eq. 5) and were checked numerically.
