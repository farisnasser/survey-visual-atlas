## 1. BFGS: curvature from differences of gradients

This chapter follows Sections IV-E, IV-F and IV-G of the survey. Newton uses $H_t$ directly, which needs Hessian access and a linear solve at every step. Quasi-Newton methods build an approximation from what gradient descent already sees: the change in parameters and the change in gradients (survey Eq. 48),

$$s_t = \theta_{t+1} - \theta_t, \qquad y_t = g_{t+1} - g_t.$$

By the mean-value theorem, $y_t \approx Hs_t$ for small steps. So each pair $(s_t, y_t)$ carries information about the Hessian *without explicitly computing it*. If $B_t$ approximates $H_t$, ask the new approximation to reproduce the observed change exactly: the **[[secant]] condition** $B_{t+1}s_t = y_t$. Broyden, Fletcher, Goldfarb and Shanno (1970) found the classical solution, **[[bfgs|BFGS]]**. For $y_t^\top s_t > 0$ (survey Eq. 49):

$$B_{t+1} = B_t - \frac{B_ts_ts_t^\top B_t}{s_t^\top B_ts_t} + \frac{y_ty_t^\top}{y_t^\top s_t}.$$

Why this rank-two correction? Multiply the right side by $s_t$. The middle term gives $-B_ts_t$, cancelling the old estimate's prediction; the last term gives $y_t$. So $B_{t+1}s_t = y_t$, exactly as required. If $B_t \succ 0$ and the curvature condition $y_t^\top s_t > 0$ holds, $B_{t+1}$ stays positive definite, so the quasi-Newton step $B_t\Delta_t = -g_t$ is always a descent direction. With noisy minibatch gradients, $y_t$ can violate the secant geometry, which is why stochastic variants add damping and regularization.

## 2. Inverse BFGS and limited memory

Solving with $B_t$ each step is avoidable: many implementations update an inverse approximation $C_t \approx H_t^{-1}$ instead. With $\rho_t = (y_t^\top s_t)^{-1}$ (survey Eqs. 50 and 51):

$$C_{t+1} = (I - \rho_ts_ty_t^\top)\,C_t\,(I - \rho_ty_ts_t^\top) + \rho_ts_ts_t^\top,$$

and then $\Delta_t = -C_tg_t$: a matrix-vector product, no solve.

:::viz bfgs
:::

:::read How to read this visual
**What is drawn.** The curved valley $L = (1 - x)^2 + 10(y - x^2)^2$ of the survey's Fig. 10 in 3D (heights log-scaled), with the minimum at $(1, 1)$ in green. Three balls start at $(-1.2, 1)$. Violet: gradient descent with $\eta = 0.01$, about as large as the steep walls allow. Coral: BFGS with the inverse update of Eqs. (50) and (51), starting from $C = I$, with a backtracking line search. White: exact Newton. The chart plots each loss on a log scale.

**What to notice.** Gradient descent creeps along the curved floor: after 400 steps it is still far from the minimum. BFGS starts out like a gradient method, because $C = I$ is just GD. After a few steps its estimate $C$ has learned the valley's curvature from the pairs $(s_t, y_t)$, and it converges as fast as Newton does near the end: locally superlinear convergence. It never computes a second derivative. The readout also counts the extra loss evaluations its line search used, a real cost the survey warns about ("even the meaning of one step differs"). Newton, with the true Hessian, needs the fewest steps but the most information per step.
:::

Dense $B_t$ or $C_t$ needs $O(d^2)$ memory, which is impossible for a network. **Limited-memory BFGS** (L-BFGS) never forms the matrix. It keeps only the last $k$ pairs $(s_i, y_i)$ and applies the implicit inverse-Hessian approximation through the *two-loop recursion*, using only inner products and scalings. Memory drops from $O(d^2)$ to $O(kd)$. L-BFGS is curvature-aware through secant observations, not through second derivatives.

The survey adds a precise caution. A method can save Hessian storage and still need extra gradient evaluations for line searches, full-batch consistency for its curvature pairs, or safeguards. Its strong behaviour on smooth deterministic problems should not be transferred uncritically to noisy minibatch training.

## 3. Hessian-vector products

Second derivatives do not require storing the Hessian. For any vector $v$, a **[[hvp|Hessian–vector product]]** is a directional derivative of the gradient (survey Eq. 52):

$$Hv = \frac{d}{d\alpha}\nabla L(\theta + \alpha v)\Big|_{\alpha=0}.$$

Automatic differentiation computes it exactly, without forming $H$ (Pearlmutter, 1994), at a cost comparable to a few gradient evaluations. The simplest approximation is a finite difference of two gradients.

:::viz hvp
:::

:::read How to read this visual
**What is drawn.** The error of the finite-difference estimate of $Hv$, against the difference step $h$, both on log scales. The function is the curved surface of chapter 2 at $(1.1, 0.9)$, with $v = (1, 1)/\sqrt2$. Coral: the forward difference $(\nabla f(x + hv) - \nabla f(x))/h$. Violet: the central difference $(\nabla f(x + hv) - \nabla f(x - hv))/(2h)$.

**What moves.** The step $h$.

**What to notice.** Each curve is a V. Too large an $h$, and the curvature of $f$ spoils the estimate (the right side). Too small, and rounding errors in subtracting two almost equal gradients take over (the left side). The central difference is far more accurate at its best $h$. Automatic differentiation avoids the trade-off entirely, which is why real Hessian-free methods use it.
:::

## 4. Hessian-free optimization

A **Hessian-free** method solves a damped curvature system approximately, with an iterative solver that only ever needs products. Its standard solver is **[[conjugate-gradient|conjugate gradients]]** (CG). Given a positive-definite operator $A \approx H + \lambda I$, CG minimizes the quadratic model over growing Krylov subspaces $\{g, Ag, A^2g, \dots\}$, using one product $Av$ per iteration. The survey's Algorithm 1 is one outer step:

1. Evaluate the gradient $g_t$ at $\theta_t$.
2. Define the operator $A_tv = G_tv + \lambda_tv$ through matrix–vector products.
3. Approximately solve $A_t\Delta = -g_t$ with CG.
4. Accept, damp or rescale $\Delta$ using model-quality safeguards.
5. Update $\theta_{t+1} = \theta_t + \Delta$.

Martens applied this to deep networks (2010) and, with Sutskever, to recurrent networks (2011). One outer step can contain many inner CG iterations, so its cost varies. $G_t$ is often the Gauss–Newton matrix instead of the raw Hessian, because it is positive semidefinite and gives better-behaved linear systems (chapter 12).

:::viz cg
:::

:::read How to read this visual
**What is drawn.** Solving $Hx = b$ for a 60-dimensional positive-definite $H$ whose eigenvalues are spread from $1$ to $\kappa$. The vertical axis is the error $\|x - x^*\|_H$ relative to the start, on a log scale; the horizontal axis counts iterations, each costing one product $Hv$. Coral: conjugate gradients. Violet: gradient descent with its best fixed step, $2/(1 + \kappa)$. Dashed lines: the classical worst-case bounds, $2\left(\tfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^k$ for CG and $\left(\tfrac{\kappa - 1}{\kappa + 1}\right)^k$ for GD.

**What moves.** The condition number $\kappa$.

**What to notice.** For the same number of Hessian–vector products, CG is dramatically faster, and the gap widens with $\kappa$: its rate depends on $\sqrt\kappa$ instead of $\kappa$. CG also stops: in exact arithmetic it solves a 60-dimensional system in at most 60 iterations, and the coral curve drops off a cliff before that. The readout gives the products each method needs to reach $10^{-6}$. This is how Hessian-free methods buy Newton-quality steps with only gradient-like operations; the price is a variable number of inner iterations per outer step.
:::

:::problem The next step
Both quasi-Newton and Hessian-free methods can use matrices other than the Hessian. Chapter 12 separates the three matrices that papers call "second order".
:::
