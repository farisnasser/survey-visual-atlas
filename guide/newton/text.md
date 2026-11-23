## 1. Why the gradient is not the whole story

This chapter follows Sections IV-A to IV-D of the survey. Picture a hillside in fog. The gradient says which way the ground slopes under your feet, but two hillsides can have the same slope here and bend very differently a few metres away. A first-order method sees the slope; a second-order method also uses how the slope changes. The basic comparison is

$$\Delta_{\text{GD}} = -\eta g \;\;\text{(follow the current slope)}, \qquad H\Delta_{\text{Newton}} = -g \;\;\text{(adjust for local curvature)}.$$

Take $L(w) = 5w^2$, with gradient $10w$ and constant second derivative $H = 10$. From $w_0 = 4$ the loss is $80$, the gradient $40$, the curvature $10$. Gradient descent with $\eta = 0.01$ moves to $w_1 = 4 - 0.01(40) = 3.6$: lower, but still far from the minimum at $0$.

## 2. Minimize the curved model, not the straight one

Section II's second-order model bends where the first-order one cannot:

$$L(\theta + \Delta) \approx L(\theta) + g^\top\Delta + \tfrac12\Delta^\top H\Delta.$$

For our loss it reads $L(4 + \Delta) = 80 + 40\Delta + 5\Delta^2$, and here it is exact, because $5w^2$ is itself a quadratic. Choose the step at the bottom of this model: differentiate, $40 + 10\Delta$, set it to zero, $\Delta = -4$, so $w_1 = 4 - 4 = 0$. The minimum in one step, because the loss was exactly quadratic with positive curvature.

With many parameters the gradient of the model with respect to $\Delta$ is $g + H\Delta$; setting it to zero gives the **[[newton|Newton]] equation** (survey Eq. 32):

$$H_t\Delta_t = -g_t, \qquad \theta_{t+1} = \theta_t + \Delta_t.$$

When $H$ is invertible this is $\Delta = -H^{-1}g$, but implementations solve the linear system rather than forming an inverse, which is cheaper and numerically safer.

**Curvature changes the right step in each direction.** On chapter 4's valley $\tfrac12(100x^2 + y^2)$ at $(1, 1)$, the gradient is $(100, 1)$ and $H = \mathrm{diag}(100, 1)$. GD with $\eta = 0.01$ takes $\Delta = (-1, -0.01)$: one coordinate is solved, the other barely moves. Newton solves $100\Delta_x = -100$ and $\Delta_y = -1$, so $\Delta = (-1, -1)$ and lands on $(0, 0)$. Newton uses the gradient *relative to* the curvature: a large slope along a sharply curved direction needs only a small correction, and a small slope along a flat direction may need a large one.

:::survey Fig. 6 of the survey: Newton minimizes a quadratic local model. For an exactly quadratic positive-definite loss the full step reaches the optimum; for a changing non-convex loss the quadratic model may be inaccurate or unbounded below.
public/survey3/fig6.webp
:::

:::read How to read this figure
**(a)** The worked example $L = 5\theta^2$ (black), which is its own quadratic model. The dashed blue line is the tangent at $\theta_0 = 4$ (the linear model). GD with $\eta = 0.01$ moves down the tangent to $\theta_1 = 3.6$; Newton jumps straight to $0$.

**(b)** The non-convex $L = x^4 - x^2$ and its quadratic models at two points. At $x_0 = 1.2$ (green) the curvature is positive, the model is a bowl, and the Newton step heads toward the minimum. At $x_0 = 0.3$ (vermillion) the curvature is negative: the model is an upside-down parabola, unbounded below, and the step goes *uphill*, toward its maximum.

**(c)** An ill-conditioned bowl $\tfrac12(10x^2 + y^2)$ drawn in 3D, and one white ball. One GD step $-\eta g$ with $\eta = 0.09$ (blue) runs mostly across the valley. The Newton step $-H^{-1}g$ (vermillion) goes straight to the bottom.
:::

:::key Remember IV-A
GD uses the slope; Newton uses the slope and the local curvature to decide how far to move. This relies on a useful local quadratic model, typically a positive-definite Hessian near a minimum; it does not promise a one-step solution for a network.
:::

## 3. Local quadratic convergence, and its fine print

Let $e_t = \|\theta_t - \theta^*\|$ be the distance to a suitable minimum. With *linear* convergence the error shrinks by a roughly fixed fraction per step, perhaps $0.1 \to 0.01 \to 0.001$. With **quadratic** convergence, once close enough, the new error scales like the square of the old one. As a teaching example only, if $e_{t+1} \le 2e_t^2$, then $0.1$ becomes at most $0.02$, and then at most $0.0008$: the number of correct digits roughly doubles each step.

Why so fast? Near a smooth, well-behaved minimum the quadratic model becomes very accurate; Newton's step removes the dominant error, and what is left is of second order. But the guarantee is **local**, and it needs a stationary point with a positive-definite Hessian, a Hessian that varies smoothly nearby, and a start that is already close.

A network usually starts far from any minimum, its Hessian changes as it moves and may curve downward, and one Newton step can cost far more than one gradient step. If GD needs $100$ cheap iterations and Newton $6$ expensive ones, the iteration counts alone do not say which finishes first.

:::key Remember IV-B
"Quadratic convergence" is a local mathematical rate under assumptions, not a claim that Newton trains every network in fewer seconds.
:::

## 4. Why undamped Newton can fail on non-convex losses

Take the survey's example $L(x) = x^4 - x^2$, with $g(x) = 4x^3 - 2x$ and $H(x) = 12x^2 - 2$. At $x = 0$ the gradient is zero but $H(0) = -2$: the top of a hill. Zero gradient does not mean a minimum.

**How negative curvature sends Newton uphill.** Try $x = 0.2$: $g(0.2) = 4(0.008) - 0.4 = -0.368$ and $H(0.2) = 12(0.04) - 2 = -1.52$. Newton solves $-1.52\,\Delta = +0.368$, so $\Delta \approx -0.242$. But the gradient is negative, so downhill is to the *right*, and this step goes left. Its first-order effect is $g\Delta = (-0.368)(-0.242) \approx +0.089 > 0$: the loss goes up. An invertible Hessian is not enough: with negative curvature the quadratic model is an upside-down bowl, and Newton steps to its top.

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

**Damping.** Add a positive shift before solving (survey Eq. 33):

$$(H_t + \lambda_t I)\Delta_t = -g_t.$$

Adding $\lambda_t I$ raises every eigenvalue by $\lambda_t$. With $\lambda = 3$, $H + \lambda = -1.52 + 3 = 1.48 > 0$ and $\Delta = 0.368/1.48 \approx +0.249$: the right direction, which agrees with the negative gradient. Whenever $H + \lambda I$ is positive definite, the step is a descent direction (though not every finite step must lower the true loss).

**Why not damp enormously?** If $\lambda$ dominates, $H + \lambda I \approx \lambda I$ and $\Delta \approx -g/\lambda$: plain gradient descent with step $1/\lambda$. Too little damping leaves the negative curvature in; too much throws away the curvature we wanted. **[[damping|Damping]]** trades some of Newton's benefit for stability.

:::key Remember IV-C
A Hessian can point toward a hilltop or a saddle. Damping changes the local model so that its step is a safer descent direction.
:::

## 5. Trust regions and cubic regularization

Even a safe direction can be paired with a bad step length: the quadratic model is like a detailed map of only the next ten metres. A **[[trust-region|trust-region]]** method uses the model only inside a radius $r_t$ where it is believed: choose a promising $\Delta$ with $\|\Delta\| \le r_t$. The radius is not a learning rate; it caps how far the step may go.

**How the radius adapts.** Suppose the model predicts the loss will fall from $10$ to $6$, a reduction of $4$, and the actual loss falls to $7$, a reduction of $3$. The ratio of actual to predicted improvement is $3/4 = 0.75$: a useful model, so the region may grow. Had the loss only reached $9.9$, the ratio would be $0.1/4 = 0.025$, and the region should shrink. Predict, test, compare, adjust.

**[[cubic-reg|Cubic regularization]]** has no hard radius; it adds a penalty that grows with the cube of the step length. Steps of length $1$ and $3$ pay in proportion to $1^3 = 1$ and $3^3 = 27$, so very large steps are strongly discouraged even when the quadratic part promises a lot. Where negative curvature would make the quadratic model fall forever, a strong enough cubic term eventually wins, so the local problem has a sensible answer.

The two differ in how they control the step: a trust region says "do not leave this neighbourhood"; a cubic penalty says "moving far costs rapidly more". Both change the small problem solved to pick $\Delta$, and both can cost an inner optimization per step.

:::viz model
:::

:::read How to read this visual
**What is drawn.** A two-dimensional loss with a saddle, as a 3D surface (heights log-scaled for display). The violet sheet is the cubic model at the current point (coral): $f(x) + \nabla f(x)^\top s + \tfrac12 s^\top\nabla^2 f(x)s + \tfrac{M}{6}\|s\|^3$. This is Nesterov and Polyak's way of writing the cubic model above, with $\rho = M/2$. The white ring is the model's global minimizer, where the step goes. The side chart is a slice through the current point along the most negative Hessian direction.

**What moves.** The cubic weight $M$: $0$ (plain Newton's quadratic), $3$, $10$ (the Hessian's Lipschitz constant on this box), or $25$. "Take the step" moves to the ring and rebuilds the model.

**What to notice.** With $M = 0$ the model is Newton's quadratic: along the negative-curvature direction it falls to $-\infty$ (the slice shows it). With $M \ge 10$ the cubic term bends the model back up, and the model lies above $f$ everywhere. So stepping to its minimum can only lower $f$, whatever the curvature.
:::

:::viz newton
:::

:::read How to read this visual
**What is drawn.** A 3D version with a strict saddle at the centre and two minima (green): $f = \tfrac12x^2 - \tfrac12y^2 + \tfrac14y^4 + 0.3xy^2$. Three balls start at $(1.2, y_0)$. White: Newton, or saddle-free Newton, which divides by $|\lambda|$ instead of $\lambda$ (chosen with the control). Violet: cubic regularization with $M = 10$. Coral: gradient descent with $\eta = 1/8$.

**What moves.** The white ball's method and the start $y_0$.

**What to notice.** Newton jumps to the critical point of its quadratic model, which here is the saddle, and stays there. Cubic regularization and saddle-free Newton leave toward a minimum, and so does GD, slowly. Curvature helps only with safeguards that respect its sign.
:::

:::key Remember IV-D
Curvature can propose an intelligent step, but an optimizer must still decide how much to trust its local prediction.
:::

:::problem The next step
All of these need the Hessian, or something like it. Chapter 11 asks whether curvature can be learned from gradients we already have, or used without ever storing the matrix.
:::
