## wells | Algorithm 1
$$g_t=\nabla f_B(x_t),\quad \epsilon_t=\rho\frac{g_t}{\lVert g_t\rVert_2},\quad \tilde g_t=\nabla f_B(x_t+\epsilon_t),\quad m_t=\beta_1m_{t-1}+(1-\beta_1)\tilde g_t$$
Every $k$ steps (lazy update, Sec. 4.3; $k=10$ in experiments): $D_t=\beta_2D_{t-1}+(1-\beta_2)\,\lvert\tilde H_t\rvert$ with $\tilde H_t$ a diagonal Hessian estimate at $x_t+\epsilon_t$ and $\lvert H\rvert=\sum_i\lvert H_{ii}\rvert e_ie_i^\top$ (Eq. 5, avoids saddles and maxima). Then, with bias corrections,
$$x_{t+1}=x_t-\eta_t\,\hat D_t^{-1/2}\,\hat m_t-\eta_t\lambda x_t$$
The square root (Sec. 4.2) tames tiny curvature estimates and preserves their relative scale.
### On screen (simplified)
$\beta_1=\beta_2=0$, $k=1$, exact diagonal Hessian. Near a minimum with curvature $a$, ignoring the SAM term: diagonal Newton gives $x'=(1-\eta)x$, the square-root step $x'=(1-\eta\sqrt a)x$, unstable for $a>4/\eta^2$.
> Check script (start $(-0.3,0.2)$; sharp well $a=39.1$, flat $3.5$): $\eta=0.5$, $\rho=0.05$: diagonal Newton ends sharp, SASSHA flat ($\eta\sqrt a=3.13$ vs $0.94$). $\eta=0.1$: SASSHA stays sharp. $\eta=0.2$: $\rho=0$ stays sharp, $\rho=0.05$ reaches the flat well.

## table | Theory and reported results
- **Theorem 4.4:** for a convex, lower-bounded, smooth objective (Assumptions 4.1–4.3) with suitable step and radius sequences, $\liminf_t\lVert\nabla f(x_t)\rVert=0$. A convex result, not a non-convex guarantee.
- **Corollary 4.6:** necessary conditions for a minimum to be linearly stable for SASSHA (in the sense of Def. 4.5) involve the sharpness $a=\lambda_{\max}(\mathbb E[H])$ and the non-uniformity $s_k$ of the stochastic Hessian, and tighten with larger $\eta$ and $\rho$; approximate second-order methods with $P\approx H^{-1}$ stay stable at minima of any sharpness.
- Table 1 (ResNet-32, CIFAR-100), $\lambda_{\max}(H)$ / val. acc.: SGD 265 / 69.32%, AdaHessian 11 992 / 68.06%, Sophia-H 22 797 / 67.76%, Shampoo 436 374 / 64.08%, SASSHA 107 / 72.14% (reported).
### Notes' example
$g=(1,1)$, $\mathrm{diag}\,H=(0.01,4)$: $H^{-1}g=(100,0.25)$ but $\lvert H\rvert^{-1/2}g=(10,0.5)$.
