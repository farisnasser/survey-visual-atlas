## 1. Gradient descent and SGD

This chapter follows Sections III-A, III-B and III-C of the survey. **[[gd|Gradient descent]]** uses the full gradient (survey Eq. 13):

$$\theta_{t+1} = \theta_t - \eta_t\nabla L(\theta_t).$$

The step has two parts. The direction $-\nabla L$ comes from the linear model of chapter 2; the positive scalar $\eta_t$ says how far the update trusts that model. The sign is ordinary subtraction. If a derivative is $g = 8$ and $\eta = 0.1$, the parameter decreases by $0.8$. If $g = -3$, the same rule *increases* it by $0.3$. SGD replaces the full gradient with the minibatch estimate of chapter 5 (survey Eq. 14), $\theta_{t+1} = \theta_t - \eta_t\hat g_t$. It comes from the stochastic-approximation method of Robbins and Monro (1951).

## 2. What GD guarantees on a non-convex loss

A basic guarantee needs nothing beyond chapter 3. Suppose $L$ is $\beta$-smooth and bounded below by $L_*$, and take $\eta \le 1/\beta$. Each step lowers the loss by at least $\tfrac{\eta}{2}\|g_t\|^2$ (Eq. 7 of chapter 3). Add these inequalities for $t = 0, \dots, T-1$; the losses telescope (survey Section III-A):

$$\frac{\eta}{2}\sum_{t=0}^{T-1}\|g_t\|^2 \le L(\theta_0) - L(\theta_T) \le L(\theta_0) - L_*.$$

The smallest of $T$ numbers is at most their average, so (survey Eq. 15)

$$\boxed{\min_{0 \le t < T}\|g_t\|^2 \le \frac{2[L(\theta_0) - L_*]}{\eta T}.}$$

To make the gradient norm at most $\epsilon$, it is enough that the right side is at most $\epsilon^2$: $T = O(\epsilon^{-2})$ iterations. Notice what is promised: **[[eps-stationary|first-order stationarity]]**, a point with a small gradient, not the best network. Stochastic versions must also control the noise term of chapter 5, which is where $O(\epsilon^{-4})$ comes from.

:::viz rate
:::

:::read How to read this visual
**What is drawn.** For gradient descent with $\eta = 1/L$ on a one-dimensional non-convex function (this visual calls the smoothness constant $L$), two curves against the number of steps $T$, both axes logarithmic. Coral: the smallest gradient size actually seen in the first $T$ steps. Violet: the guarantee $\sqrt{2L(f(x_0) - f^*)/T}$, the square root of Eq. (15).

**What moves.** The starting point $x_0$.

**What to try.** Start near $x_0 = 0$, a local maximum where the gradient is tiny.

**What to notice.** The real run is always below the guarantee, as Eq. (15) promises, and often far below. The bound is a worst case over all smooth functions, so on any particular function it is loose. The readout compares the step at which the bound promises a gradient below $10^{-3}$ with the step at which GD actually got there.
:::

## 3. Momentum: why store past directions?

On a narrow valley, the trouble with plain GD is not only that steps are small. Gradients across the steep direction alternate in sign, so the iterates zig-zag from wall to wall, while the shallow direction creeps forward. Polyak's **[[heavy-ball|heavy-ball method]]** keeps a velocity (survey Eqs. 16 and 17):

$$v_t = \mu v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta v_t, \qquad 0 \le \mu < 1.$$

Unrolling the recurrence (with $v_{-1} = 0$) gives $v_t = g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \cdots$. The velocity is a discounted sum of past gradients. A direction that persists accumulates weight, up to a factor $1/(1-\mu)$; a direction that alternates largely cancels. Unlike averaging past *parameters*, momentum averages *directions*.

:::viz echo
:::

:::read How to read this visual
**What is drawn.** A made-up gradient sequence $g_t$ (white dots) with a steady part equal to $1$ plus a part that flips sign every step, of size $a$. The coral line is the momentum sum $s_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots$ (this visual calls the coefficient $\beta$; the survey calls it $\mu$). The amber line marks the steady level $1/(1-\beta)$. The side chart shows the weight $\beta^k$ given to the gradient from $k$ steps ago.

**What moves.** The momentum $\beta$ and the zig-zag size $a$.

**What to notice.** The steady part is multiplied by $1/(1-\beta)$, which is $10$ for $\beta = 0.9$. The flipping part is multiplied by only $1/(1+\beta)$. So momentum amplifies the persistent direction and damps the oscillation, which is exactly what a narrow valley needs. The readout also says how far back the memory reaches: about $1/(1-\beta)$ steps.
:::

## 4. Momentum can overshoot

Momentum also adds a dynamical state, and states can oscillate. Take the one-dimensional quadratic $L(\theta) = \lambda\theta^2/2$ with full gradients $g_t = \lambda\theta_t$. Eliminating the velocity with $\theta_t - \theta_{t-1} = -\eta v_{t-1}$ gives a two-step recurrence (survey Section III-B):

$$\theta_{t+1} = (1 + \mu - \eta\lambda)\,\theta_t - \mu\,\theta_{t-1}.$$

Try solutions of the form $\theta_t = r^t$. Substituting gives the characteristic polynomial (survey Section III-B)

$$r^2 - (1 + \mu - \eta\lambda)\,r + \mu = 0.$$

Its two roots decide everything: if both have $|r| < 1$ the iterates contract; complex roots make them oscillate; a root with $|r| > 1$ makes them grow. For $\lambda > 0$ and $0 \le \mu < 1$, the roots stay inside the unit circle exactly when $0 < \eta\lambda < 2(1 + \mu)$. So $\mu$ and $\eta$ must be tuned *together*. And this is a statement about a fixed quadratic, not about the changing Hessian of a network.

:::viz roots
:::

:::read How to read this visual
**What is drawn.** The complex plane. For a quadratic whose curvatures range from $\mu_{\text{curv}}$ (amber) to $L = 10$ (blue), each dot is a root $r$ of the characteristic polynomial for one curvature. Along that eigen-direction the error is multiplied by $|r|$ at every step. The white circle is the unit circle; the dashed circle has radius $\sqrt\beta$, where complex roots lie. The side chart shows the error along the flattest and the sharpest direction over 150 steps.

**What moves.** The method (GD, heavy ball, Nesterov), $\kappa$, the step size and the momentum $\beta$.

**What to notice.** Every dot must sit inside the unit circle for the method to converge, and the farthest dot sets the speed. Gradient descent's roots lie on the real axis, and the flattest curvature's root sits near $1$: that is the slow direction. With momentum, the roots move onto the circle of radius $\sqrt\beta$, which can be much smaller than GD's worst root. Raise $\eta$ until a blue dot leaves the unit circle: that is the condition $\eta\lambda < 2(1+\beta)$ failing, and the side chart's blue curve starts to grow.
:::

:::viz valley
:::

:::read How to read this visual
**What is drawn.** A narrow valley $f = \tfrac12(\mu x^2 + 10y^2)$ in 3D, with condition number $\kappa = 10/\mu$; the y-axis is stretched so the valley is visible. Three balls start at $(-1.45, 0.5)$. Violet: GD with its best step $2/(L + \mu)$. Coral: heavy ball. White: Nesterov (next section). The chart shows $f$ on a log scale.

**What moves.** $\kappa$, the step $\eta$ and the momentum $\beta$. The buttons apply Polyak's and Nesterov's classical tunings.

**What to notice.** With the classical tunings, both momentum methods reach $f < 10^{-8}$ in far fewer steps than GD; the readout counts them. Push $\beta$ toward $1$ and the heavy ball swings across the valley for a long time before settling: the long-lived oscillation of the survey's Fig. 4.
:::

## 5. Nesterov: the gradient at a look-ahead point

Nesterov acceleration changes *where* the gradient is evaluated. One common deep-learning form is (survey Eqs. 18 and 19)

$$y_t = \theta_t + \mu(\theta_t - \theta_{t-1}), \qquad \theta_{t+1} = y_t - \eta\nabla L(y_t).$$

First extrapolate along the last movement to $y_t$; then take the gradient there. The heavy ball, by contrast, uses the gradient at the current point plus the accumulated velocity. The look-ahead point exposes some of the change in slope that the velocity is about to meet. It is still a first-order method: it uses gradients and past iterates, never an explicit Hessian. For smooth *convex* objectives, Nesterov's classical analysis gives an $O(1/T^2)$ function-value rate, against $O(1/T)$ for plain GD. That accelerated rate does **not** automatically carry over to non-convex networks.

:::survey Fig. 4 of the survey: momentum integrates directions over time. Across a narrow valley, alternating components may cancel, but excessive momentum can create long-lived oscillations or instability.
public/survey3/fig4.webp
:::

:::read How to read this figure
**(a)** The valley $L = \tfrac12(25x^2 + y^2)$ as a 3D surface (height $\sqrt{L}$), with 40 steps of SGD (blue) and heavy ball with $\mu = 0.8$ (orange), both with $\eta = 0.07$ and the same Gaussian gradient noise, from the white ball. SGD zig-zags across the steep walls and creeps along the floor. The heavy ball travels along the valley much faster, but swings past the bottom.

**(b)** The one-dimensional recurrence above with $\eta\lambda = 0.3$, for $\mu = 0, 0.6, 0.9, 0.97$. Without momentum, $\theta_t$ decays smoothly. With more momentum, it oscillates, and at $\mu = 0.97$ the oscillation lasts for the whole window.

**(c)** Where it settles: over the plane of $\eta\lambda$ and $\mu$, green where every root of the characteristic polynomial above has size below one, so $\theta_t$ settles, and red where one is larger, so $\theta_t$ blows up. The black line $\eta\lambda = 2(1 + \mu)$ is the boundary, the survey's stability condition.
:::

:::problem The next step
Momentum gives the update a memory but still one step size for every coordinate. Chapter 8 gives each coordinate its own.
:::
