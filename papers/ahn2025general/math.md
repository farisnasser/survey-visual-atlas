## kink | Schedule-free SGD and (λ, ε)-stationarity
$$x_t=(1-c_t)x_{t-1}+c_tz_t,\qquad y_t=(1-\beta_t)z_t+\beta_tx_t,\qquad z_{t+1}=z_t-\gamma g_t,\quad g_t=\text{stoch. grad at }y_t\qquad\text{(SF-SGD)}$$
**Definition 1.** $x$ is $(\lambda,\epsilon)$-stationary if
$$\inf_{p:\ \mathbb E_{w\sim p}[w]=x}\ \big\lVert\mathbb E[\nabla F(w)]\big\rVert+\lambda\,\mathbb E\lVert w-x\rVert^2\le\epsilon$$
(an approximate Goldstein stationary point: gradients at nearby points must average to something small).
**Sec. 5 (Theorem 1, informal; Prop. 3).** With the online learner chosen as discounted online mirror descent, the conversion is exactly SF-SGD with $\beta_t\sim U[\beta,1]$, $\beta$ close to 1, constant $c_t=1-\beta$ and base step $\gamma/(1-\beta)$, i.e. larger than the OMD step. This explains the empirically good choices of Defazio et al. ($\beta$ near 1, large steps), which their convex theory does not.
### Notes' example
$f=\tfrac12(x-2)^2$, $\gamma=\tfrac12$, $\beta=\tfrac34$, $c=\tfrac12$, $x_0=z_1=0$: $x_1=y_1=0$, $g_1=-2$, $z_2=1$; $x_2=0.5$, $y_2=0.625$, $g_2=-1.375$, $z_3=1.6875$.
> Check script (toy on screen, 4000 steps): over the last 1000, the true gradients at $y$ have mean length 1.41 but their mean has length 0.019; $\mathrm{std}\,\lVert x\rVert=0.015$ vs $0.071$ for $z$.

## rates | Complexity (Def. 1, Prop. 1)
- $(\lambda,\epsilon)$-stationary points: $O(\lambda^{1/2}\epsilon^{-7/2})$ stochastic gradients, optimal (Zhang & Cutkosky 2024). Assumption 1: lower-bounded, well-behaved, $\lVert\nabla F\rVert\le G$; Assumption 2: unbiased gradients with variance $\le\sigma^2$.
- **Prop. 1:** $F$ $L$-smooth: an $(L^2\epsilon^{-1},\epsilon)$-stationary point has $\lVert\nabla F\rVert\le2\epsilon$, so $O(\epsilon^{-4})$; $F$ $H$-second-order-smooth: an $(H/2,\epsilon)$-stationary point has $\lVert\nabla F\rVert\le2\epsilon$, so $O(\epsilon^{-3.5})$. Both optimal (Arjevani et al. 2020, 2023).
- First-order guarantees only: no second-order stationarity or saddle escape is claimed; the theory covers SF-SGD, not Schedule-Free AdamW, and needs different $\beta_t,c_t$ than the convex analysis.
