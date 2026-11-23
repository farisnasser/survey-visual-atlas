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

@@viz rate@@

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

@@viz echo@@

@@viz valley@@

## 4. Why bigger momentum is not always better

A moving object can overshoot its target, and so can an accumulated direction: it may keep pushing through a minimum after the local gradient has reversed. On the one-dimensional quadratic $L(\theta) = \lambda\theta^2/2$, eliminating the velocity (with $\theta_t - \theta_{t-1} = -\eta v_{t-1}$) gives a two-step recurrence,

$$\theta_{t+1} = (1 + \mu - \eta\lambda)\,\theta_t - \mu\,\theta_{t-1},$$

and trying $\theta_t = r^t$ gives its characteristic polynomial, $r^2 - (1 + \mu - \eta\lambda)\,r + \mu = 0$. Its two roots decide everything: both inside the unit circle means the iterates contract, complex roots mean they oscillate, a root outside means they grow. For $\lambda > 0$ and $0 \le \mu < 1$ the iterates settle exactly when $0 < \eta\lambda < 2(1 + \mu)$. So $\mu$ and $\eta$ must be tuned together, and this is a statement about a fixed quadratic, not about a network's changing Hessian.

@@viz roots@@

The survey's Fig. 4 shows both sides of momentum.

@@survey 4@@

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
