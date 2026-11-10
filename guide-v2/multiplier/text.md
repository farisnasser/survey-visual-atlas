## 1. Gradient descent near a critical point

This chapter follows Section II-F of the survey, the single algebraic fact that connects its two halves. It is the most important chapter of the preliminaries.

Let $\theta^\star$ be a critical point, so $\nabla L(\theta^\star) = 0$, and let $H = \nabla^2 L(\theta^\star)$ be the Hessian there. Near $\theta^\star$, the Taylor model of chapter 2 has no first-order term, and the gradient is approximately linear:

$$\nabla L(\theta) \approx H(\theta - \theta^\star).$$

Put this into one step of gradient descent, $\theta_{t+1} = \theta_t - \eta\nabla L(\theta_t)$, and subtract $\theta^\star$ from both sides:

$$\theta_{t+1} - \theta^\star = (I - \eta H)(\theta_t - \theta^\star).$$

This is the **linearized GD map**, survey Eq. (9). It says that near a critical point, gradient descent is just repeated multiplication by the matrix $I - \eta H$.

Now use the eigenvectors $v_1, \dots, v_d$ of $H$ (chapter 2). Write the error $\theta_t - \theta^\star$ in those directions, with coordinate $u_j$ along $v_j$. Along each eigenvector, $I - \eta H$ acts as multiplication by one number:

$$u_j \;\leftarrow\; (1 - \eta\lambda_j)\, u_j.$$

That number, $1 - \eta\lambda_j$, is the **[[multiplier]]** of direction $j$. After $t$ steps, $u_j$ has been multiplied by $(1 - \eta\lambda_j)^t$. Everything in this chapter, and much of the survey, follows from asking whether $|1 - \eta\lambda_j|$ is smaller or larger than one.

## 2. Three regimes

The multiplier depends only on the product $\eta\lambda$: step size times curvature. Its size decides everything.

| $\eta\lambda$ | multiplier $1 - \eta\lambda$ | what happens to $u$ | name |
|---|---|---|---|
| $\eta\lambda < 0$ | larger than $1$ | grows, same sign | **negative curvature**: GD leaves |
| $0 < \eta\lambda < 1$ | between $0$ and $1$ | shrinks, same sign | stable, monotone |
| $\eta\lambda = 1$ | $0$ | gone in one step | stable, exact |
| $1 < \eta\lambda < 2$ | between $-1$ and $0$ | shrinks, flips sign | stable, bouncing |
| $\eta\lambda > 2$ | smaller than $-1$ | grows, flips sign | **excess curvature**: GD bounces out |

The direction is stable exactly in the middle band, $0 < \eta\lambda < 2$. It is unstable in two ways: on the left, because the loss curves down; on the right, because it curves up too sharply for the step size. These two unstable regimes are the two halves of the survey.

Some worked numbers, all over 20 steps. With $\eta\lambda = -0.1$ (a saddle direction), the multiplier is $1.1$, and $1.1^{20} = 6.73$: the distance grows almost sevenfold. With $\eta\lambda = 0.5$ the multiplier is $0.5$, and $0.5^{20} \approx 9.5 \times 10^{-7}$. With $\eta\lambda = 1.5$ it is $-0.5$, with the same size of shrinkage but alternating sign. With $\eta\lambda = 2.2$ it is $-1.2$, and $1.2^{20} = 38.3$: the iterate bounces across the valley, each bounce bigger than the last.

:::viz multiplier
:::

:::read How to read this visual
**What is drawn.** Three views of one number. The top plot shows $u_t$, the distance from the critical point along one eigenvector, for $t = 0, \dots, 30$, starting from $u_0 = 1$. Each coral dot is one step, and the faint stems drop to zero so you can see the sign. The dashed lines are $\pm 1$, the starting size. The first chart below is the line $m = 1 - \eta\lambda$ against $\eta\lambda$: the red band is negative curvature, the green band is stable, and the blue band is too sharp. The colored dot is where your choice sits. The second chart shows $|u_t|$ on a log scale, so that growth and decay of any size are visible.

**What to try.** Use the presets in order: *saddle −0.1*, *stable 0.5*, *bouncing 1.5*, *edge 2*, *too sharp 2.2*. Then drag the slider slowly across $0$ and across $2$. *Replay* redraws the steps one at a time.

**What to notice.** In the red band the dots climb steadily away: this is a strict saddle letting go. In the green band they shrink, either from one side or alternating. At exactly $\eta\lambda = 2$ they bounce between $+1$ and $-1$ forever. In the blue band they bounce with growing size. On the log chart, a straight line means a constant multiplier: rising lines are the two instabilities, falling lines are stability. The readout names the regime and gives $|u_{30}|$.
:::

## 3. Stability of minima: the $2/\eta$ threshold

At a minimum, every eigenvalue of $H$ is positive. Gradient descent converges to it if and only if every direction is in the stable band, $|1 - \eta\lambda_j| < 1$ for all $j$. The binding direction is the sharpest one, so the condition is

$$\eta\,\lambda_{\max}(H) < 2.$$

This is survey Eq. (10). The largest eigenvalue $\lambda_{\max}(H)$ of the Hessian is called the **[[sharpness]]** of the minimum. A minimum whose sharpness exceeds $2/\eta$ is **linearly unstable** under GD, however low its loss: gradient descent with step $\eta$ cannot settle there. This [[stability-threshold|stability threshold]] $2/\eta$ is the central number of Section IV of the survey. Chapter 11 shows that practical training does not stay safely below it. It operates exactly at it.

The same multiplier also explains why gradient descent is slow in narrow valleys: the flat direction has a multiplier close to one, while the step size is capped by the sharp direction.

:::viz zigzag
:::

:::read How to read this visual
**What is drawn.** A map, seen from above, of the quadratic valley $f = \tfrac12(\mu u_1^2 + L u_2^2)$, tilted so that its eigen-directions (blue lines) are not the axes. Here $\mu = 1$ is the flat curvature and $L = \kappa$ the sharp one. The coral path is gradient descent; the green dot is the minimum. The chart plots the size of each eigen-coordinate, $|u_1|$ and $|u_2|$, on a log scale against the step $t$. The dots are the actual run, and the lines are the prediction $(1 - \eta\lambda)^t$ times the start.

**What to try.** Raise the step size $\eta$ towards $2/L$, then just past it. Change the condition number $\kappa$.

**What to notice.** The dots sit exactly on the lines: on a quadratic, GD *is* multiplication by $1 - \eta\lambda$ in each eigen-direction, with no approximation. The sharp coordinate bounces (negative multiplier) when $\eta > 1/L$, and the readout says stability needs $\eta < 2/L$. Past that value, the sharp coordinate grows, the run explodes, and nothing about the flat direction can save it.
:::

## 4. Escape from strict saddles: the escape-time formula

At a strict saddle, some eigenvalue is negative, $\lambda_j = -|\lambda_j|$. Its multiplier is $1 + \eta|\lambda_j| > 1$, so the component $u_j$ grows by that factor at every step. Suppose GD starts at distance $\delta$ from the **[[stable-set|stable manifold]]**, the set of points that are attracted to the saddle; in the linear picture, that is the set with $u_j = 0$. To travel a distance $r$ away, it needs about

$$T_{\rm esc} \approx \frac{\log(r/\delta)}{\log(1 + \eta|\lambda_j|)}$$

iterations. This is survey Eq. (11). Read it carefully, because chapters 7 to 9 are a commentary on it.

- The escape time is **finite whenever $\delta > 0$**. A strict saddle cannot hold GD forever, unless GD starts exactly on the stable manifold.
- It **grows without bound as $\delta \to 0$**, but only logarithmically. With $\eta|\lambda_j| = 0.1$ and $r = 1$, every factor of 100 in $\delta$ costs $\log 100/\log 1.1 = 48.3$ extra steps. $\delta = 10^{-2}$ needs about $48$ steps, $\delta = 10^{-6}$ about $145$, and $\delta = 10^{-12}$ about $290$.
- It is **long when the negative curvature is weak**: for small $\eta|\lambda_j|$, $\log(1 + \eta|\lambda_j|) \approx \eta|\lambda_j|$, so $T_{\rm esc}$ grows like $1/|\lambda_j|$.

The whole saddle-escape literature can be read as an effort to control the two quantities in this formula: to make sure $\delta$ is not too small, through randomness (chapter 8), and that $|\lambda_j|$ is not too small, through the strict-saddle property (chapter 3).

:::viz escape
:::

:::read How to read this visual
**What is drawn.** The surface $f = \tfrac12x^2 + \tfrac14y^4 - \tfrac12y^2$: a strict saddle at the center with curvatures $+1$ (blue tube, along $x$) and $-1$ (red tube, along $y$), and two minima at $(0, \pm 1)$ (green). The saddle's stable manifold is the line $y = 0$. The chart plots the distance from that line, $|y_t|$, on a log scale against the step $t$, together with the prediction $\delta \cdot 1.1^t$ (here $\eta = 0.1$, so $1 + \eta|\lambda| = 1.1$).

**What moves.** The coral ball starts at $x = 1$, a distance $\delta$ off the line, and runs gradient descent.

**What to try.** Move the offset $\delta$ from $10^{-1}$ down to $10^{-12}$ and drop the ball each time.

**What to notice.** The ball first rolls towards the saddle along $x$, then waits, then leaves along $y$ into one of the minima. On the chart, $|y_t|$ is a straight line on the log scale, the run sitting exactly on the prediction, until the quartic term takes over near $|y| \approx 0.5$. The readout gives the steps until $|y| > 0.5$ and the formula's prediction $\ln(0.5/\delta)/\ln 1.1$. Each factor of 100 in $\delta$ adds about 48 steps: the wait is never infinite, but it is longer for every digit of $\delta$.
:::

## 5. One instability, local in both cases

Both kinds of instability are **local**. The multiplier says that GD cannot remain near the critical point; it does not say where GD goes next. The terms that the linear map discards, the third-order terms of the Taylor expansion and beyond, decide what happens once the iterate has moved away.

- **At a saddle**, they decide where the escaping trajectory goes: into a minimum, or towards the next saddle (chapter 9).
- **At the edge of stability**, they produce a self-stabilizing feedback that keeps training alive instead of letting it diverge (chapter 12).

The survey's Fig. 4 draws exactly the multiplier chart of the first visual in this chapter: the line $1 - \eta\lambda$, the stable band in the middle, saddle escape on the left and the edge of stability on the right.

:::key What to keep from this chapter
- Near a critical point, GD multiplies each eigen-component of the error by $1 - \eta\lambda$.
- Stable band: $0 < \eta\lambda < 2$. Left of it ($\lambda < 0$): saddle escape. Right of it ($\eta\lambda > 2$): excess curvature, the edge of stability.
- Stability of a minimum: $\eta\lambda_{\max} < 2$. Escape time from a saddle: $\log(r/\delta)/\log(1 + \eta|\lambda|)$, which is finite but unbounded.
:::

:::problem The next step
The multiplier describes one step near one point. To say what happens over a whole run, from a random start, with minibatch noise, we need three more tools: GD seen as a map with stable sets, the shape of the noise, and a way to measure curvature in a network with millions of weights. The next chapter builds them, and then Section III begins.
:::
