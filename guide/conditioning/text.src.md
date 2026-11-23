## 1. Start with the survey's narrow valley

This chapter follows Sections II-D and II-E of the survey. Take the quadratic of the survey's Eq. (8), where $x$ and $y$ are two trainable weights, not data points:

$$L(x, y) = \tfrac12(100x^2 + y^2), \qquad g = \begin{bmatrix}100x\\ y\end{bmatrix}, \qquad H = \begin{bmatrix}100 & 0\\ 0 & 1\end{bmatrix}.$$

The eigenvalues are $100$ and $1$: the loss curves a hundred times more strongly along $x$ than along $y$. Picture a long, narrow valley. Moving across it is very sensitive; moving along it is gentle.

**What does gradient descent do here?** Write the update $\theta_{t+1} = \theta_t - \eta g_t$ coordinate by coordinate:

$$x_{t+1} = (1 - 100\eta)\,x_t, \qquad y_{t+1} = (1 - \eta)\,y_t.$$

The survey sets $\eta = 0.01$. Then $x_{t+1} = 0$ and $y_{t+1} = 0.99\,y_t$. Starting from $(1, 1)$, the first step gives $(0, 0.99)$; after ten steps $y \approx 0.904$, and after a hundred $y \approx 0.366$. The steep direction is solved at once in this specially chosen example, while the flat direction crawls. One learning rate has to serve two very different curvatures.

@@viz narrow@@

## 2. The condition number

For a positive-definite Hessian, the **[[condition-number|condition number]]** is

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}},$$

here $100/1 = 100$. If $\kappa = 1$, every direction has the same curvature and the level sets are circles. A large $\kappa$ means curvature varies dramatically across directions, and the survey calls such problems *ill-conditioned*. For strongly convex quadratics it limits how fast any fixed step can go: stability in the steepest direction caps $\eta$ at $2/\lambda_{\max}$, and that step is tiny for the flattest one.

@@viz kappa@@

## 3. What does it mean to rotate the valley?

Imagine turning the long ellipse by $45^\circ$ without changing its shape. Before the turn, the steep and the shallow directions lie along the axes. Afterwards, both are diagonal combinations of $x$ and $y$. Mathematically, take the rotation

$$Q = \frac{1}{\sqrt2}\begin{bmatrix}1 & -1\\ 1 & 1\end{bmatrix}, \qquad Q^\top Q = I.$$

If the valley's own coordinates are $z = Q\theta$, the loss becomes $\tfrac12(Q\theta)^\top H(Q\theta) = \tfrac12\theta^\top(Q^\top HQ)\theta$, which is the survey's rotated Hessian $Q^\top HQ$. Multiplying out,

$$H_{\text{rot}} = Q^\top\begin{bmatrix}100 & 0\\ 0 & 1\end{bmatrix}Q = \begin{bmatrix}50.5 & -49.5\\ -49.5 & 50.5\end{bmatrix}.$$

Its eigenvalues are still $100$ and $1$: rotation changes the eigen-directions, not the curvature strengths, so $\kappa$ is unchanged.

**Why the off-diagonal entries matter.** Before the turn, $g_x = 100x$ depends only on $x$. Afterwards,

$$g_x = 50.5x - 49.5y, \qquad g_y = -49.5x + 50.5y.$$

Even with $x$ fixed, changing $y$ changes the $x$-slope: the two weights are coupled. A diagonal method scales each coordinate on its own, $\Delta = -\eta D^{-1}g$ with $D = \mathrm{diag}(d_1, d_2)$, so it cannot mix $g_x$ with $g_y$. A full-matrix step can. Here

$$H_{\text{rot}}^{-1} = \begin{bmatrix}0.505 & 0.495\\ 0.495 & 0.505\end{bmatrix},$$

and its off-diagonal entries let the first update coordinate depend on both gradient components.

@@viz precond@@

The survey's Fig. 3 shows the same comparison with the exact $100:1$ matrices.

@@survey 3@@

**One gradient, three steps.** Start the rotated valley at $\theta_0 = (1, 0)$, where $L = 25.25$. Every method sees the same gradient, $g_0 = H_{\text{rot}}\theta_0 = (50.5, -49.5)^\top$:

| method | step | new point $\theta_1$ | new loss |
|---|---|---|---|
| GD, $\eta = 0.01$ | $-0.01\,g_0$ | $(0.495, 0.495)$ | $0.245025$ |
| exact diagonal, $D = 50.5I$ | $-D^{-1}g_0$ | $(0,\ 0.9802)$ | about $24.26$ |
| full matrix (Newton) | $-H_{\text{rot}}^{-1}g_0$ | $(0, 0)$ | $0$ |

The exact diagonal of $H_{\text{rot}}$ is $50.5$ in both coordinates, so the diagonal step is just gradient descent with a step of $1/50.5$, which jumps across the valley to almost the same height. The full matrix uses the coupling and lands on the minimum.

:::idea In plain English
A straight valley has an easy-to-name narrow direction: just $x$. A rotated valley is narrow along a combination such as $x - y$. A method that only rescales $x$ and $y$ separately cannot generally separate that combination from the shallow one, $x + y$. A full-matrix method can.
:::

:::key What you should remember
Rotation does not make the quadratic intrinsically worse conditioned. It exposes a limitation of update rules tied to the stored coordinate axes, which is what motivates structured and full-matrix preconditioning in Sections IV and V.
:::

## 4. Local stability: where the multiplier $1 - \eta\lambda$ comes from

Now ask a different question. Suppose $\theta^*$ is a stationary point, $\nabla L(\theta^*) = 0$. If gradient descent starts close to it, will it move toward it or away? Let $e_t = \theta_t - \theta^*$, the displacement from that point (an "error" vector, not a prediction error).

**Deriving the survey's Eq. (9), one line at a time.** Taylor's approximation of the gradient near $\theta^*$ gives $g(\theta^* + e_t) \approx g(\theta^*) + H^*e_t = H^*e_t$, because $g(\theta^*) = 0$. Substitute into gradient descent and subtract $\theta^*$:

$$e_{t+1} = \theta_{t+1} - \theta^* \approx e_t - \eta H^*e_t = (I - \eta H^*)\,e_t.$$

Now let the error lie along an eigenvector, $e_t = a_t v_j$ with $H^*v_j = \lambda_j v_j$:

$$e_{t+1} \approx a_t v_j - \eta a_t\lambda_j v_j = (1 - \eta\lambda_j)\,a_t v_j.$$

So each eigen-component is multiplied by $m = 1 - \eta\lambda$ at every step: $e_1 = m\,e_0$, $e_2 = m^2e_0$, and so on, as long as the local approximation holds.

**Three cases.**

- **$0 < \eta\lambda < 2$: settle.** Then $-1 < m < 1$. With $m = 0.8$: $1 \to 0.8 \to 0.64 \to 0.512$. A ball moving toward the bottom of a bowl; if $m$ is negative it crosses the bottom back and forth, but the swings shrink.
- **$\lambda < 0$: escape.** For any $\eta > 0$, $m > 1$. With $m = 1.2$: $1 \to 1.2 \to 1.44 \to 1.728$. A ball just off the top of a hill rolls away. Here growth is useful: it is how gradient descent leaves a saddle along negative curvature. (At a point with exactly zero gradient, unperturbed GD does not move at all.)
- **$\eta\lambda > 2$: overshoot.** Then $m < -1$. With $m = -2$: $1 \to -2 \to 4 \to -8 \to 16$. The ball jumps across the bowl farther each time: the steps are too large.

We do not choose the case; the curvature and the learning rate do:

| example | $\lambda$ | $\eta$ | $m = 1 - \eta\lambda$ | behaviour |
|---|---|---|---|---|
| A | $5$ | $0.1$ | $0.5$ | contracts |
| B | $-5$ | $0.1$ | $1.5$ | escapes |
| C | $5$ | $0.5$ | $-1.5$ | oscillates and grows |

A real Hessian can have positive and negative eigenvalues at once, so one stationary point can attract gradient descent along some directions and repel it along others. That is exactly a saddle.

:::viz cases
:::

:::read How to read this visual
**What is drawn.** The local error $e_t = m^t e_0$ along one eigen-direction, for $t = 0, \dots, 12$, with $m = 1 - \eta\lambda$ and $e_0 = 1$. The shaded band is $|e| \le 1$; the dashed lines mark $\pm1$.

**What moves.** The curvature $\lambda$ (negative for a hill or saddle direction) and the learning rate $\eta$. The buttons load the three examples of the table.

**What to notice.** Inside $0 < \eta\lambda < 2$ the dots shrink toward zero, smoothly if $m > 0$ and alternating in sign if $m < 0$. With $\lambda < 0$ they grow without changing sign: escape. With $\eta\lambda > 2$ they grow and flip sign: the overshoot. The readout names the case from $m$ alone.
:::

**Deriving the $2/\lambda_{\max}$ rule.** For a positive eigenvalue, the error shrinks when $|1 - \eta\lambda| < 1$. Expand the absolute value, $-1 < 1 - \eta\lambda < 1$; subtract 1, $-2 < -\eta\lambda < 0$; multiply by $-1$ and flip the signs, $0 < \eta\lambda < 2$. So $\eta < 2/\lambda$, and to make every positive-curvature direction stable at once, the most restrictive one wins:

$$\eta < \frac{2}{\lambda_{\max}}.$$

This is a local, quadratic threshold, not a universal rule for networks; Section VI-E of the survey returns to the "edge of stability", where training keeps working beyond it.

@@read pgdstab@@

:::read How to read this visual
**What is drawn.** The same stability rule for a preconditioned step $\theta \leftarrow \theta - \eta M^{-1}H\theta$ on the rotated valley with curvatures $\kappa$ and $1$. Such a step behaves like gradient descent on $K = M^{-1/2}HM^{-1/2}$, so each eigen-direction of $K$ is multiplied by $1 - \eta\lambda(K)$. The curves show the worst factor $\max|1 - \eta\lambda(K)|$ against $\eta$ for three choices: $M = I$ (plain GD, violet), $M = \mathrm{diag}(H)$ (coral) and $M = H$ (Newton, white). Below the dashed line at $1$ the step contracts; the dotted verticals mark each $2/\lambda_{\max}(K)$.

**What moves.** The valley's angle, $\kappa$, and the step size $\eta$.

**What to notice.** With $M = H$, $K = I$: every direction has the same curvature and any $\eta < 2$ is stable. The diagonal choice equals Newton at $0^\circ$ but, at $45^\circ$ and $\kappa = 100$, its $K$ has eigenvalues $0.0198$ and $1.98$, just as badly conditioned as plain GD. The readout gives each method's stable range and its step count at the best $\eta$.
:::

## 5. What gradient-based guarantees actually promise

These results clarify what convergence means for non-convex losses. The survey distinguishes two targets. A point is an approximate **[[eps-stationary|first-order stationary point]]** if

$$\|\nabla L(\theta)\|_2 \le \epsilon,$$

with $\epsilon$ a small tolerance. It is approximately **second-order stationary** if, in addition,

$$\lambda_{\min}\big(H(\theta)\big) \ge -\gamma,$$

where $\gamma$ limits how strongly any direction may curve downward. A small gradient alone cannot tell a saddle from a minimum, and even both conditions together do not promise a global minimum. Gradient-based analyses usually guarantee only the first; escaping saddles needs extra assumptions or mechanisms such as random perturbations. The survey's Fig. 2(e)–(h) shows why a zero gradient can belong to very different shapes. Test them here.

@@viz zoo@@

:::key What you should remember
The multiplier $1 - \eta\lambda$ is the centre of Section II-E. A growing distance is good when it means leaving negative curvature, and bad when it comes from steps that are too aggressive for positive curvature. Convergence statements for networks are about small gradients, not global optimality.
:::

:::problem The next step
Every bound so far assumed the exact gradient. Chapter 5 asks what changes when it is estimated from a minibatch.
:::
