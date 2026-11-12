## 1. Newton's direction from the quadratic model

This chapter follows Sections IV-A to IV-D of the survey and Appendix D-A. The linear model of chapter 2 cannot tell how the gradient changes after a step. The quadratic model can. If the Hessian is positive definite, the model $g^\top\Delta + \tfrac12\Delta^\top H\Delta$ is a bowl with a unique minimizer. Differentiate it (survey Section IV-A) and set the result to zero:

$$\nabla_\Delta\left(g^\top\Delta + \tfrac12\Delta^\top H\Delta\right) = g + H\Delta = 0.$$

This gives the **[[newton|Newton]] equation** (survey Eq. 32):

$$H_t\Delta_t = -g_t, \qquad \theta_{t+1} = \theta_t + \Delta_t.$$

Formally $\Delta_t = -H_t^{-1}g_t$, but numerical algorithms *solve* the linear system instead of forming the inverse. An explicit inverse wastes work and magnifies rounding error; factorizations and iterative solvers address $H\Delta = -g$ directly.

**A worked example.** For $L(\theta) = 5\theta^2$, the gradient is $g = 10\theta$ and the Hessian is $H = 10$. From $\theta_0 = 4$, Newton's step is $\Delta = -40/10 = -4$, so $\theta_1 = 0$: the minimum in one step. Gradient descent with $\eta = 0.01$ gives $\theta_1 = 4 - 0.01 \times 40 = 3.6$. On an exact quadratic the Taylor model *is* the loss, so Newton solves it at once. The same holds in any dimension. For $L(\theta) = \tfrac12\theta^\top H\theta - b^\top\theta + c$ (survey Section IV-A) the gradient is $H\theta - b$, and a full step gives $\theta^+ = \theta - H^{-1}(H\theta - b) = H^{-1}b$, the exact minimizer.

Compare with gradient descent eigen-direction by eigen-direction. GD multiplies the error along eigenvector $j$ by $1 - \eta\lambda_j$. Newton divides each gradient component by its own curvature: the step size along direction $j$ is $|g_j|/\lambda_j$ instead of $\eta|g_j|$. That is the essential difference between second-order and first-order descent. It is also why Newton is immune to the conditioning problem of chapter 4.

:::survey Fig. 6 of the survey: Newton minimizes a quadratic local model. For an exactly quadratic positive-definite loss the full step reaches the optimum; for a changing non-convex loss the quadratic model may be inaccurate or unbounded below.
public/survey3/fig6.webp
:::

:::read How to read this figure
**(a)** The worked example $L = 5\theta^2$ (black), which is its own quadratic model. The dashed blue line is the tangent at $\theta_0 = 4$ (the linear model). GD with $\eta = 0.01$ moves down the tangent to $\theta_1 = 3.6$; Newton jumps straight to $0$.

**(b)** The non-convex $L = x^4 - x^2$ and its quadratic models at two points. At $x_0 = 1.2$ (green) the curvature is positive, the model is a bowl, and the Newton step heads toward the minimum. At $x_0 = 0.3$ (vermillion) the curvature is negative: the model is an upside-down parabola, unbounded below, and the step goes *uphill*, toward its maximum.

**(c)** An ill-conditioned bowl $\tfrac12(10x^2 + y^2)$ drawn in 3D, and one white ball. One GD step $-\eta g$ with $\eta = 0.09$ (blue) runs mostly across the valley. The Newton step $-H^{-1}g$ (vermillion) goes straight to the bottom.
:::

## 2. Local quadratic convergence

Newton's appeal is also its local *rate*. Suppose $\theta^*$ satisfies $g(\theta^*) = 0$, $H(\theta^*)$ is positive definite, and $H$ is Lipschitz-continuous nearby. Close to $\theta^*$, Taylor's theorem for the gradient gives (survey Section IV-B)

$$g(\theta_t) = H(\theta_t)(\theta_t - \theta^*) + r_t, \qquad \|r_t\| \le C\|\theta_t - \theta^*\|^2.$$

The Newton step $\Delta_t = -H(\theta_t)^{-1}g(\theta_t)$ cancels the first term exactly, leaving only the remainder (survey Section IV-B):

$$\|\theta_{t+1} - \theta^*\| \le C'\|\theta_t - \theta^*\|^2.$$

This is **local quadratic convergence**: the error is squared at every step, so once close, the number of correct digits roughly doubles each step. Every assumption matters: nonsingularity, smoothness, a neighbourhood. It says nothing about a Newton step from a random network initialization, and each step costs far more than a gradient step.

## 3. Why undamped Newton can fail

Take the survey's example $L(x) = x^4 - x^2$. At $x = 0$, $g(0) = 0$ and $H(0) = -2$: a local *maximum*. More generally, if $H$ has a negative eigenvalue, the quadratic model is unbounded below along that eigenvector. Even when $H$ is invertible, $g^\top(-H^{-1}g)$ can be positive if $H$ is indefinite: the Newton direction is then not a descent direction at all.

The simplest remedy is **[[damping]]**: replace $H$ by $H + \lambda I$ (survey Eq. 33),

$$(H_t + \lambda_t I)\Delta_t = -g_t.$$

If $H_t + \lambda_t I \succ 0$, the step descends whenever $g_t \ne 0$, because $g_t^\top\Delta_t = -g_t^\top(H_t + \lambda_tI)^{-1}g_t < 0$. Choosing $\lambda_t$ changes the model and may need spectral information or safeguards; a line search can also make sure the true loss, not just the model, decreases.

:::viz newton1d
:::

:::read How to read this visual
**What is drawn.** The survey's loss $L(x) = x^4 - x^2$ (white), with minima at $\pm1/\sqrt2 \approx \pm0.707$ (green) and a local maximum at $0$. At the starting point $x_0$ (white dot), the dashed violet parabola is the damped quadratic model, with curvature $h + \lambda$, where $h = L''(x_0)$. The coral segment is the first damped Newton step; the coral dots are the next eight steps.

**What moves.** The starting point $x_0$ and the damping $\lambda$.

**What to try.** Start at $x_0 = 0.3$ with $\lambda = 0$. Then raise $\lambda$ past $-h$. Then start at $x_0 = 1.2$.

**What to notice.**
- **$x_0 = 0.3$, no damping.** $h$ is negative ($h < 0$ whenever $|x| < 0.408$), so the model opens downward and its stationary point is its maximum. The step goes uphill: the readout reports "not a descent direction" and an increase in $L$. The later steps converge to the local maximum at $0$: Newton finds a stationary point, not a minimum.
- **Damping above $-h$.** The model becomes a bowl, the step turns downhill, and the iterates reach a minimum.
- **$x_0 = 1.2$.** Here $h > 0$: plain Newton heads straight to the minimum and converges in a few steps, the quadratic convergence of section 2.

Too much damping makes the step short, like gradient descent with a small step. The damping trades Newton's speed for safety.
:::

## 4. Trust regions and cubic regularization

Instead of trusting the model everywhere, a **[[trust-region|trust-region]]** method limits the step (survey Section IV-D):

$$\min_\Delta\; g_t^\top\Delta + \tfrac12\Delta^\top H_t\Delta \qquad \text{subject to } \|\Delta\| \le r_t.$$

The radius $r_t$ says how far the model is believed. After each step, the method compares the actual decrease of the loss with the decrease the model predicted, and grows or shrinks $r_t$ accordingly. This handles negative curvature naturally: the constrained problem always has a solution.

**[[cubic-reg|Cubic regularization]]** adds a penalty instead (survey Section IV-D):

$$\min_\Delta\; g_t^\top\Delta + \tfrac12\Delta^\top H_t\Delta + \frac{\rho_t}{3}\|\Delta\|^3.$$

For large $\|\Delta\|$ the positive cubic term beats any negative quadratic term, so the subproblem is bounded below even when $H_t$ has negative eigenvalues. Nesterov and Polyak established its theory; adaptive cubic regularization (Cartis, Gould and Toint) made it robust. Both are global guarantees under precisely stated assumptions, and solving the subproblem can be expensive for a network. The key idea: these methods change *the model being minimized*. They are not Adam with a different learning rate.

:::viz model
:::

:::read How to read this visual
**What is drawn.** A two-dimensional loss with a saddle, as a 3D surface (heights log-scaled for display). The violet sheet is the cubic model at the current point (coral): $f(x) + \nabla f(x)^\top s + \tfrac12 s^\top\nabla^2 f(x)s + \tfrac{M}{6}\|s\|^3$. This is Nesterov and Polyak's way of writing the cubic model above, with $\rho = M/2$. The white ring is the model's global minimizer, where the step goes. The side chart is a slice through the current point along the most negative Hessian direction.

**What moves.** The cubic weight $M$: $0$ (plain Newton's quadratic), $3$, $10$ (the Hessian's Lipschitz constant on this box), or $25$. "Take the step" moves to the ring and rebuilds the model.

**What to notice.** With $M = 0$ the model is Newton's quadratic: along the negative-curvature direction it falls to $-\infty$ (the slice shows it). With $M \ge 10$ the cubic term bends the model back up, and the model lies above $f$ everywhere. So stepping to its minimum can only lower $f$, whatever the curvature.
:::

## 5. The saddle where Newton reaches the wrong destination (Appendix D-A)

Appendix D-A makes the failure exact. Take $L(x, y) = \tfrac12x^2 - y^2$, with $\nabla L = (x, -2y)$ and $H = \mathrm{diag}(1, -2)$ (survey Eq. 53). The origin is a strict saddle. From $(x_0, y_0) = (1, 0.1)$, a gradient step gives (survey Eq. 54)

$$(x_1, y_1) = ((1 - \eta)x_0, (1 + 2\eta)y_0).$$

The $x$ part contracts for $0 < \eta < 2$; the $y$ part grows, because $1 + 2\eta > 1$. Negative curvature pushes GD *away* from the saddle. The full Newton step uses $H^{-1} = \mathrm{diag}(1, -\tfrac12)$ (survey Eq. 55):

$$\begin{bmatrix}x_1\\y_1\end{bmatrix} = \begin{bmatrix}x_0\\y_0\end{bmatrix} - H^{-1}\nabla L(x_0, y_0) = \begin{bmatrix}0\\0\end{bmatrix}.$$

Newton reaches the saddle in one step. This is no contradiction: Newton's equation finds a stationary point of the quadratic model, and a non-convex quadratic's stationary point need not be a minimum. Saying "second order is better because it knows the curvature" is incomplete. What matters is whether a method uses the right curvature model, with the right safeguards, for the region it is in.

:::survey Fig. 14 of the survey: the saddle of Eq. (53) from (1, 0.1). GD moves away from the saddle; the unrestricted Newton step of Eq. (55) lands exactly on it; damping as in Eq. (33) makes the step escape again.
public/survey3/fig14.webp
:::

:::read How to read this figure
The strict saddle $L = \tfrac12x^2 - y^2$ as a 3D surface. From the white ball $(1, 0.1)$: GD with $\eta = 0.1$ (blue, 10 steps) contracts $x$ by $0.9$ and grows $y$ by $1.2$ per step, so it rolls off the saddle along $y$. The Newton step of Eq. (55) (vermillion) lands exactly on the saddle, marked with a cross. Damping with $\lambda = 3$ (purple, 2 steps) makes $H + 3I = \mathrm{diag}(4, 1)$ positive definite; its step triples $y$ and escapes faster than GD.
:::

:::viz newton
:::

:::read How to read this visual
**What is drawn.** A 3D version with a strict saddle at the centre and two minima (green): $f = \tfrac12x^2 - \tfrac12y^2 + \tfrac14y^4 + 0.3xy^2$. Three balls start at $(1.2, y_0)$. White: Newton, or saddle-free Newton, which divides by $|\lambda|$ instead of $\lambda$ (chosen with the control). Violet: cubic regularization with $M = 10$. Coral: gradient descent with $\eta = 1/8$.

**What moves.** The white ball's method and the start $y_0$.

**What to notice.** Newton jumps to the critical point of its quadratic model, which here is the saddle, and stays there. Cubic regularization and saddle-free Newton leave toward a minimum, and so does GD, slowly. Curvature helps only with safeguards that respect its sign.
:::

:::problem The next step
Newton needs the Hessian and a linear solve. Chapter 11 shows how to get curvature from gradients alone.
:::
