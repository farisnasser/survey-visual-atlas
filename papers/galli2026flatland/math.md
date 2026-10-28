## linesearch | Large steps without a global Lipschitz constant
### Local segment smoothness (Sec. 2)
$$\ell(w,\eta)=\sup_{0<t\le\eta}\frac{\lVert\nabla f(w)-\nabla f(w-t\nabla f(w))\rVert}{t\,\lVert\nabla f(w)\rVert}$$
Def. 2.3: steps are **large** if $\eta_k\ge1/\ell(w_k,\eta_k)$ and, within every window of $P$ steps, $\eta>2/\ell$ at least once (an edge-of-stability regime).
### Nonmonotone acceptance (Eq. 2)
$$f(w_k-\eta_k\nabla f(w_k))\le R_k-c\,\eta_k\lVert\nabla f(w_k)\rVert^2,\qquad R_k\ge f(w_k)$$
Zhang–Hager reference: $R_k=\dfrac{\xi Q_{k-1}R_{k-1}+f(w_k)}{Q_k}$, $Q_k=\xi Q_{k-1}+1$. Monotone search: $R_k=f(w_k)$.
### Equality line search (Alg. 2)
Start from the previous step; double while accepted, halve until accepted. Prop. 2.5: $\eta_k\ge2(1-c)/\ell(w_k,2\eta_k)$.
### Theorem 2.9
If $f\in C^1$ is bounded below with a compact initial level set, GD with this nonmonotone equality search has $\lim_k\lVert\nabla f(w_k)\rVert=0$ (first-order only).
> Offline: nonmonotone median $\eta\lambda_{\max}=2.2$, 198 accepted increases, $\lVert\nabla f\rVert\to10^{-7}$.

## flatland | The 2/K floor (Sec. 3)
MSE over $K$ outputs with a final-layer bias $b$: for one example $f=\tfrac1K\sum_j(z_j-y_j)^2$, so
$$\frac{\partial^2 f}{\partial b_j\partial b_l}=\frac2K\,\mathbb 1_{j=l}\quad\Rightarrow\quad H_{bb}=\tfrac2K I\quad\Rightarrow\quad\lambda_{\max}(H)\ge\tfrac2K$$
(Rayleigh quotient restricted to bias directions). In the paper's large-step runs, sharpness reached exactly $2/K$ early, with the top $K$ eigenvalues equal, near-random loss and vanishing hidden-layer gradients: flat, almost stationary saddle regions. Capping the step below $K$ (NLS-ub) trained better in their experiments.
> Offline: at the flat point the gradient is 0, the top-$K$ eigenvalues are $2/K$ and $\lambda_{\min}<0$ for $K=2,4,8$.
