## 1. Gradient descent on a single weight

This chapter follows Sections III-A, III-B and III-C of the survey. Every method in Section III shares the action $\theta_{t+1} = \theta_t + \Delta_t$; what changes is only how the step $\Delta_t$ is built. Start with one weight and $L(w) = 5w^2$, whose derivative is $g(w) = 10w$. At $w_0 = 3$ the gradient is $30$: moving right raises the loss, so move left. **[[gd|Gradient descent]]** does exactly that, $w_{t+1} = w_t - \eta\,g(w_t)$. With $\eta = 0.05$,

$$w_1 = 3 - 0.05(30) = 1.5,$$

and the loss falls from $5(3)^2 = 45$ to $5(1.5)^2 = 11.25$. The gradient picks the direction; the learning rate picks the distance. Too small a rate crawls; too large a rate jumps across the minimum, as chapter 3 showed. With many weights the derivative becomes the gradient vector (survey Eq. 13):

$$\theta_{t+1} = \theta_t - \eta_t\nabla L(\theta_t).$$

## 2. Why avoid the exact gradient?

With a million training examples, the full gradient needs a pass over all of them before every single step. SGD replaces it with the minibatch estimate of chapter 5 (survey Eq. 14):

$$\theta_{t+1} = \theta_t - \eta_t\hat g_t.$$

If four examples have scalar gradients $2, 4, 6, 8$, the full gradient is $5$. A minibatch of the first two gives $3$, one of the last two gives $7$: either step is noisy, although on average over random minibatches the estimate is $5$. SGD buys cheaper, more frequent updates with noisier directions, and the batch size trades one for the other.

**What does GD guarantee on a non-convex loss?** Suppose $L$ is $\beta$-smooth and bounded below by $L_*$, and take $\eta \le 1/\beta$. Chapter 3 showed that each step lowers the loss by at least $\tfrac{\eta}{2}\|g_t\|^2$. Add these inequalities for $t = 0, \dots, T-1$; the losses telescope:

$$\frac{\eta}{2}\sum_{t=0}^{T-1}\|g_t\|^2 \le L(\theta_0) - L(\theta_T) \le L(\theta_0) - L_*.$$

The smallest of $T$ numbers is at most their average, so (survey Eq. 15)

$$\min_{0 \le t < T}\|g_t\|^2 \le \frac{2[L(\theta_0) - L_*]}{\eta T}.$$

To push the gradient norm below $\epsilon$ it is enough that the right side is at most $\epsilon^2$: $T = O(\epsilon^{-2})$ iterations. Notice what is promised: **[[eps-stationary|first-order stationarity]]**, a point with a small gradient, not the best network. SGD needs extra assumptions on the gradient noise, which is where chapter 5's $O(\epsilon^{-4})$ comes from.

:::viz rate
:::

:::read How to read this visual
**What is drawn.** For gradient descent with $\eta = 1/L$ on a one-dimensional non-convex function (this visual calls the smoothness constant $L$), two curves against the number of steps $T$, both axes logarithmic. Coral: the smallest gradient size actually seen in the first $T$ steps. Violet: the guarantee $\sqrt{2L(f(x_0) - f^*)/T}$, the square root of Eq. (15).

**What moves.** The starting point $x_0$.

**What to try.** Start near $x_0 = 0$, a local maximum where the gradient is tiny.

**What to notice.** The real run is always below the guarantee, as Eq. (15) promises, and often far below. The bound is a worst case over all smooth functions, so on any particular function it is loose. The readout compares the step at which the bound promises a gradient below $10^{-3}$ with the step at which GD actually got there.
:::

:::key Remember III-A
GD follows the exact slope of the full loss; SGD follows a cheap, noisy estimate of it. The first major trade-off is gradient accuracy against cost per update, and neither method remembers anything about earlier steps.
:::

## 3. Momentum: why store past directions?

In a narrow valley, the gradient keeps pointing from one wall to the other. Plain GD zig-zags across the valley and makes little progress along its floor. No single gradient is wrong; the waste comes from treating every step independently. **[[heavy-ball|Momentum]]** keeps a running direction. Polyak's heavy-ball method writes it as (survey Eqs. 16 and 17)

$$v_t = \mu v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta v_t, \qquad 0 \le \mu < 1,$$

where $v_t$ is the accumulated direction and $\mu$ says how much of the past is kept. With $\mu = 0$ it is gradient descent again. (Some texts write $(1 - \mu)g_t$ instead of $g_t$; only the scaling convention differs.)

**A two-step example.** Let $v_0 = 0$, $\mu = 0.8$, and gradients $g_1 = 5$, $g_2 = 4$:

$$v_1 = 0.8(0) + 5 = 5, \qquad v_2 = 0.8(5) + 4 = 8.$$

Both gradients pushed the same way, so the accumulated direction is stronger than the latest gradient alone. If instead $g_2 = -4$, then $v_2 = 0.8(5) - 4 = 0$: opposing gradients cancel. That is exactly why the zig-zag across a valley shrinks while motion along the floor builds up.

:::viz echo
:::

:::read How to read this visual
**What is drawn.** A made-up gradient sequence $g_t$ (white dots) with a steady part equal to $1$ plus a part that flips sign every step, of size $a$. The coral line is the momentum sum $s_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots$ (this visual calls the coefficient $\beta$; the survey calls it $\mu$). The amber line marks the steady level $1/(1-\beta)$. The side chart shows the weight $\beta^k$ given to the gradient from $k$ steps ago.

**What moves.** The momentum $\beta$ and the zig-zag size $a$.

**What to notice.** The steady part is multiplied by $1/(1-\beta)$, which is $10$ for $\beta = 0.9$. The flipping part is multiplied by only $1/(1+\beta)$. So momentum amplifies the persistent direction and damps the oscillation, which is exactly what a narrow valley needs. The readout also says how far back the memory reaches: about $1/(1-\beta)$ steps.
:::

:::viz valley
:::

:::read How to read this visual
**What is drawn.** A narrow valley $f = \tfrac12(\mu x^2 + 10y^2)$ in 3D, with condition number $\kappa = 10/\mu$; the y-axis is stretched so the valley is visible. Three balls start at $(-1.45, 0.5)$. Violet: GD with its best step $2/(L + \mu)$. Coral: heavy ball. White: Nesterov (next section). The chart shows $f$ on a log scale.

**What moves.** $\kappa$, the step $\eta$ and the momentum $\beta$. The buttons apply Polyak's and Nesterov's classical tunings.

**What to notice.** With the classical tunings, both momentum methods reach $f < 10^{-8}$ in far fewer steps than GD; the readout counts them. Push $\beta$ toward $1$ and the heavy ball swings across the valley for a long time before settling: the long-lived oscillation of the survey's Fig. 4.
:::

## 4. Why bigger momentum is not always better

A moving object can overshoot its target, and so can an accumulated direction: it may keep pushing through a minimum after the local gradient has reversed. On the one-dimensional quadratic $L(\theta) = \lambda\theta^2/2$, eliminating the velocity (with $\theta_t - \theta_{t-1} = -\eta v_{t-1}$) gives a two-step recurrence,

$$\theta_{t+1} = (1 + \mu - \eta\lambda)\,\theta_t - \mu\,\theta_{t-1},$$

and trying $\theta_t = r^t$ gives its characteristic polynomial, $r^2 - (1 + \mu - \eta\lambda)\,r + \mu = 0$. Its two roots decide everything: both inside the unit circle means the iterates contract, complex roots mean they oscillate, a root outside means they grow. For $\lambda > 0$ and $0 \le \mu < 1$ the iterates settle exactly when $0 < \eta\lambda < 2(1 + \mu)$. So $\mu$ and $\eta$ must be tuned together, and this is a statement about a fixed quadratic, not about a network's changing Hessian.

:::viz roots
:::

:::read How to read this visual
**What is drawn.** The complex plane. For a quadratic whose curvatures range from $\mu_{\text{curv}}$ (amber) to $L = 10$ (blue), each dot is a root $r$ of the characteristic polynomial for one curvature. Along that eigen-direction the error is multiplied by $|r|$ at every step. The white circle is the unit circle; the dashed circle has radius $\sqrt\beta$, where complex roots lie. The side chart shows the error along the flattest and the sharpest direction over 150 steps.

**What moves.** The method (GD, heavy ball, Nesterov), $\kappa$, the step size and the momentum $\beta$.

**What to notice.** Every dot must sit inside the unit circle for the method to converge, and the farthest dot sets the speed. Gradient descent's roots lie on the real axis, and the flattest curvature's root sits near $1$: that is the slow direction. With momentum, the roots move onto the circle of radius $\sqrt\beta$, which can be much smaller than GD's worst root. Raise $\eta$ until a blue dot leaves the unit circle: that is the condition $\eta\lambda < 2(1+\beta)$ failing, and the side chart's blue curve starts to grow.
:::

The survey's Fig. 4 shows both sides of momentum.

:::survey Fig. 4 of the survey: momentum integrates directions over time. Across a narrow valley, alternating components may cancel, but excessive momentum can create long-lived oscillations or instability.
public/survey3/fig4.webp
:::

:::read How to read this figure
**(a)** The valley $L = \tfrac12(25x^2 + y^2)$ as a 3D surface (height $\sqrt{L}$), with 40 steps of SGD (blue) and heavy ball with $\mu = 0.8$ (orange), both with $\eta = 0.07$ and the same Gaussian gradient noise, from the white ball. SGD zig-zags across the steep walls and creeps along the floor. The heavy ball travels along the valley much faster, but swings past the bottom.

**(b)** The one-dimensional recurrence above with $\eta\lambda = 0.3$, for $\mu = 0, 0.6, 0.9, 0.97$. Without momentum, $\theta_t$ decays smoothly. With more momentum, it oscillates, and at $\mu = 0.97$ the oscillation lasts for the whole window.

**(c)** Where it settles: over the plane of $\eta\lambda$ and $\mu$, green where every root of the characteristic polynomial above has size below one, so $\theta_t$ settles, and red where one is larger, so $\theta_t$ blows up. The black line $\eta\lambda = 2(1 + \mu)$ is the boundary, the survey's stability condition.
:::

:::key Remember III-B
Momentum smooths repeated directions and cancels alternating ones. Its price is one extra state vector, the velocity, and the risk of overshooting.
:::

## 5. Nesterov: the gradient at the look-ahead point

Heavy-ball momentum pushes with its past movement but measures the new gradient at the current position. If we are racing toward the far wall of a valley, the current gradient may not yet show what lies ahead. **[[nesterov|Nesterov acceleration]]** measures the slope where momentum is about to take us. A commonly used form is (survey Eqs. 18 and 19)

$$y_t = \theta_t + \mu(\theta_t - \theta_{t-1}), \qquad \theta_{t+1} = y_t - \eta\nabla L(y_t).$$

Here $y_t$ is the look-ahead point: the last movement $\theta_t - \theta_{t-1}$, scaled by $\mu$, extrapolated once more.

**A tiny example.** Take $L(w) = w^2$, $w_{t-1} = 3$, $w_t = 2$ and $\mu = 0.5$: we were moving left by one unit. The look-ahead point is $y_t = 2 + 0.5(2 - 3) = 1.5$. The derivative at the current point is $L'(2) = 4$, but at the look-ahead point it is $L'(1.5) = 3$. With $\eta = 0.1$ the update is $w_{t+1} = 1.5 - 0.1(3) = 1.2$. The example shows *where* the gradient is measured, not a universal speed advantage.

For smooth convex objectives, suitable versions of Nesterov's method provably converge faster than gradient descent, but those guarantees depend on the objective and the parameters, and they do not automatically carry over to non-convex network training.

:::key Remember III-C
Momentum remembers where it has been. Nesterov evaluates the slope where momentum expects to go. Neither gives different parameters different step sizes, which is the next idea.
:::

:::problem The next step
Both momentum methods still use one learning rate for every weight. Chapter 8 gives each coordinate its own.
:::
