## 1. Back to the zig-zag

Stop 6 solved the saddle problem, but it left the oldest problem of the course standing: in a long, narrow valley gradient descent is slow. Recall stop 3's bowl $f(x, y) = \tfrac12(x^2 + 10y^2)$. Its curvatures are $\mu = 1$ along $x$ and $L = 10$ along $y$, so its [[condition-number|condition number]] is $\kappa = L/\mu = 10$. Stop 4 showed that GD treats each eigen-direction separately and that its best fixed step, $\eta = 2/(L + \mu) = 2/11$, still multiplies the error by $(\kappa - 1)/(\kappa + 1) = 9/11 \approx 0.818$ per step. From $(-1.45, 0.5)$ it needs 48 steps to push $f$ below $10^{-8}$.

Look at what the iterates do. Along the steep direction $y$, the factor $1 - \eta L$ is negative, so $y_t$ **flips sign every step**: the zig-zag. Along the flat direction $x$, the factor $1 - \eta\mu$ is positive, so $x_t$ keeps its sign and creeps toward $0$. One part of the gradient keeps pointing the same way; the other part keeps reversing.

This suggests a fix. If the method **remembered** its recent steps and averaged them, the reversing part would cancel and the steady part would add up. That is what momentum does.

## 2. The heavy ball

:::def Heavy-ball momentum
The **[[heavy-ball|heavy-ball method]]** (Polyak, 1964) adds a fraction $\beta$ of the previous step to the gradient step:

$$x_{t+1} \;=\; x_t - \eta\, \nabla f(x_t) \;+\; \beta\,(x_t - x_{t-1}), \qquad 0 \le \beta < 1 .$$

The number $\beta$ (beta) is the **momentum** coefficient. With $\beta = 0$ this is plain gradient descent.
:::

The name comes from physics. Picture a heavy ball rolling on the landscape with friction: the gradient is the force, $x_t - x_{t-1}$ is the velocity, and $\beta$ says how much velocity survives one step of friction ($\beta = 1$: no friction at all; $\beta = 0$: so much friction that the ball forgets its speed every step). Writing $v_{t+1} = x_{t+1} - x_t$ for the **velocity** gives the form most code uses:

$$v_{t+1} = \beta\, v_t - \eta\, \nabla f(x_t), \qquad x_{t+1} = x_t + v_{t+1}.$$

The general idea, keeping some of the previous motion, is called **[[momentum]]**. It costs one extra vector of memory and nothing else: still one gradient per step.

## 3. Momentum is an average of past gradients

Unroll the velocity recursion, starting from $v_0 = 0$:

$$v_{t+1} \;=\; -\eta\,\big(\nabla f(x_t) + \beta\,\nabla f(x_{t-1}) + \beta^2\,\nabla f(x_{t-2}) + \dots + \beta^t\,\nabla f(x_0)\big).$$

The step is a weighted sum of **all** past gradients, where the gradient from $k$ steps ago gets weight $\beta^k$. Two cases show what this does:

- **A steady gradient** $g$ (the same every step) builds up to $g\,(1 + \beta + \beta^2 + \dots) = g/(1 - \beta)$. With $\beta = 0.9$ that is **10 times** the gradient: the effective step along consistent directions is $\eta/(1 - \beta)$.
- **A flipping gradient** $+a, -a, +a, \dots$ gives $a\,(1 - \beta + \beta^2 - \dots) = a/(1 + \beta)$ in size. With $\beta = 0.9$ that is about $0.526\,a$: the zig-zag is damped, not amplified.

So momentum is a filter: it amplifies what the gradients agree on and suppresses what they disagree on.

The same sum, rescaled so its weights add up to one, is an **[[ema|exponential moving average]]** (EMA):

$$m_t \;=\; \beta\, m_{t-1} + (1 - \beta)\, g_t \;=\; (1 - \beta) \sum_{k \ge 0} \beta^k\, g_{t-k}.$$

Its weights $(1 - \beta)\beta^k$ sum to $1$ and shrink geometrically, so it averages over roughly the last $1/(1 - \beta)$ values; with $\beta = 0.9$ the last 7 values already carry 52% of the weight. Momentum is $v_{t+1} = -\tfrac{\eta}{1 - \beta}\, m_t$, an EMA of the gradients times a bigger step. EMAs come back in stop 9: Adam keeps two of them.

:::viz echo
:::

## 4. Nesterov's look-ahead

In 1983 Yurii Nesterov changed one detail: evaluate the gradient not where the ball **is**, but where momentum is about to **carry** it.

:::def Nesterov's accelerated gradient
First move along the momentum, then take the gradient step from there:

$$y_t = x_t + \beta\,(x_t - x_{t-1}), \qquad x_{t+1} = y_t - \eta\, \nabla f(y_t).$$

This is **[[nesterov|Nesterov's accelerated gradient]]** (NAG). The point $y_t$ is the *look-ahead* point.
:::

The difference with the heavy ball is only where the gradient is measured. If momentum is about to carry the ball up the far wall of the valley, the gradient at $y_t$ already points back, and the step is corrected *before* the overshoot instead of after. That small change is what makes Nesterov's method provably fast on every smooth convex function, not only on quadratics (section 6).

## 5. Memory in a narrow valley

Here are the three methods on the valley $f = \tfrac12(\mu x^2 + 10y^2)$. Gradient descent always gets its best fixed step. The heavy ball and Nesterov share the sliders for $\eta$ and $\beta$, and each has its own "best" setting, derived in the next section:

- **Polyak's tuning** for the heavy ball: $\eta = 4/(\sqrt L + \sqrt\mu)^2$, $\beta = \big((\sqrt L - \sqrt\mu)/(\sqrt L + \sqrt\mu)\big)^2$.
- **Nesterov's tuning**: $\eta = 1/L$, $\beta = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$.

:::viz valley
:::

With $\kappa = 10$, the steps to reach $f < 10^{-8}$ are 48 for GD, 20 for the tuned heavy ball and 31 for tuned Nesterov. With $\kappa = 100$ (the flat curvature drops to $\mu = 0.1$) they are **469, 71 and 100**. Making the valley ten times narrower made GD about ten times slower, but the momentum methods only about three to four times slower: $\sqrt{10} \approx 3.2$. That square root is the whole point of this stop.

Two warnings the visual makes concrete. The tunings are not interchangeable: Nesterov's method with Polyak's heavy-ball tuning ($\eta = 0.3306$, $\beta = 0.6694$ at $\kappa = 100$) diverges. And both tunings need $\mu$ and $L$, which in a real network nobody knows; in practice $\beta = 0.9$ is a common default and $\eta$ is tuned by trial.

## 6. Why the square root: a 2×2 matrix

Stop 4 showed that on a quadratic, GD splits into independent one-dimensional problems, one per eigen-direction. The heavy ball splits the same way, because it only scales and adds gradients and past steps. Take one direction with curvature $\lambda$ and call the coordinate along it $u$ (the minimum is at $u = 0$). Its gradient is $\lambda u$, so the heavy ball becomes

$$u_{t+1} \;=\; (1 + \beta - \eta\lambda)\, u_t \;-\; \beta\, u_{t-1}.$$

The new value depends on the **two** previous values, so the state is the pair $(u_t, u_{t-1})$, and one step is a $2 \times 2$ matrix:

$$\begin{bmatrix} u_{t+1} \\ u_t \end{bmatrix} = \underbrace{\begin{bmatrix} 1 + \beta - \eta\lambda & -\beta \\ 1 & 0 \end{bmatrix}}_{M} \begin{bmatrix} u_t \\ u_{t-1} \end{bmatrix}.$$

After $t$ steps the state is $M^t$ times the start, so the error shrinks like $\lvert r\rvert^t$, where $r$ is the eigenvalue of $M$ with the largest size. The eigenvalues are the roots of

$$r^2 - (1 + \beta - \eta\lambda)\, r + \beta = 0 .$$

For GD the same recipe gives a $1 \times 1$ "matrix" $1 - \eta\lambda$, the factor of stop 3. Now follow three short steps.

1. **The product of the two roots is $\beta$.** (For $r^2 + br + c$ the product of the roots is $c$.) So when the roots are complex, they are conjugates of equal size, and that size is exactly $\sqrt\beta$, **whatever the curvature $\lambda$ is**.
2. **When are they complex?** When the discriminant is negative: $(1 + \beta - \eta\lambda)^2 < 4\beta$, which works out to $(1 - \sqrt\beta)^2 < \eta\lambda < (1 + \sqrt\beta)^2$. Every curvature inside this window contracts at the same rate $\sqrt\beta$.
3. **Fit the window to the landscape.** All curvatures lie between $\mu$ and $L$. Put the window's edges exactly there, $\eta\mu = (1 - \sqrt\beta)^2$ and $\eta L = (1 + \sqrt\beta)^2$. Dividing gives $\sqrt\kappa = (1 + \sqrt\beta)/(1 - \sqrt\beta)$; adding the square roots gives $\sqrt\eta\,(\sqrt L + \sqrt\mu) = 2$. Solving:

$$\sqrt\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}, \qquad \eta = \frac{4}{(\sqrt L + \sqrt\mu)^2}.$$

That is Polyak's tuning, and every direction now contracts at least by $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ per step (at the two edges the root is double, which only adds a factor that grows linearly in $t$). Compare with GD:

| | contraction per step | $\kappa = 10$ | $\kappa = 100$ |
|---|---|---|---|
| GD, step $2/(L + \mu)$ | $(\kappa - 1)/(\kappa + 1) \approx 1 - 2/\kappa$ | 0.818 | 0.980 |
| heavy ball, Polyak's tuning | $(\sqrt\kappa - 1)/(\sqrt\kappa + 1) \approx 1 - 2/\sqrt\kappa$ | 0.519 | 0.818 |
| Nesterov, $\eta = 1/L$ | $1 - 1/\sqrt\kappa$ (on quadratics) | 0.684 | 0.900 |

The number of steps to shrink the error by a fixed factor grows like $\kappa$ for GD and like $\sqrt\kappa$ with momentum. At $\kappa = 100$, a factor of $10^6$ takes about 691 GD steps, 69 heavy-ball steps and 131 Nesterov steps. This change from $\kappa$ to $\sqrt\kappa$ is called **[[acceleration]]**.

:::viz roots
:::

Move $\beta$ in the visual and watch the cloud. With little momentum the roots are real and spread along the axis; the one near $+1$ (the flattest direction, amber) is slow. Raising $\beta$ pulls the roots onto the dashed circle of radius $\sqrt\beta$. Too much $\beta$ and the circle itself is large: every direction spirals slowly. The tuned value is the smallest circle that holds every curvature.

:::theorem What is proved, and where
- **Quadratics** (Polyak 1964; derived above): the tuned heavy ball contracts by $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ per step. The same holds near a minimum where $f$ is close to its [[quadratic-model|quadratic model]], so the guarantee is local.
- **Smooth convex functions** (Nesterov; see [Nesterov 2004](paper.html?id=nesterov2004introductory), Chapter 2): with suitable $\eta$ and $\beta_t$, $f(x_t) - f^\star$ is of order $L\lVert x_0 - x^\star\rVert^2/t^2$, against order $L\lVert x_0 - x^\star\rVert^2/t$ for GD. For $\mu$-[[strongly-convex|strongly convex]] functions the error shrinks like $(1 - 1/\sqrt\kappa)^t$. The same book shows that no method that only uses gradients can beat these orders on every such function, so Nesterov's method is **optimal** among first-order methods there.
- For the heavy ball with Polyak's tuning, the accelerated rate is only guaranteed locally. Nesterov's form keeps acceleration on every smooth convex function, which is why it is the one with the convex theory behind it.
:::

## 7. Momentum overshoots

Momentum methods are **not descent methods**: $f$ can go up from one step to the next. A ball with speed rolls past the bottom of a valley and up the other side before it turns around. Along a direction with complex roots the error spirals in the $(u_t, u_{t-1})$ plane, so $u_t$ changes sign again and again (the side chart of the roots visual shows it).

On a non-convex landscape the overshoot changes **where** the ball ends. Take stop 1's curve $\mathcal{L}(\theta) = 0.1\theta^2 + 0.5\sin 3\theta$ and start at $\theta_0 = 2.4$ with step $\eta = 0.05$:

- $\beta = 0$ (gradient descent) stops in the local minimum at $\theta \approx 1.50$ (loss $-0.264$), and the loss never goes up;
- $\beta = 0.9$ also ends at $1.50$, but the loss rises on 92 of the first 200 steps;
- $\beta = 0.95$ carries the ball over the bump into the global minimum at $-0.50$ (loss $-0.474$);
- $\beta = 0.99$ carries it past the global minimum into the worst valley, at $-2.50$ (loss $0.156$).

:::viz overshoot
:::

:::warn Overshooting is not a feature you can count on
Momentum sometimes helps a ball over a small bump, and that is a common story for why it helps in deep learning. Nothing guarantees it: the same inertia carried the ball *out* of the best valley at $\beta = 0.99$. What momentum provably buys is speed in ill-conditioned valleys (section 6), not better minima.
:::

Because $f$ is not monotone, analyses of momentum track another quantity. [Jin, Netrapalli & Jordan 2018](paper.html?id=jin2018accelerated) use the "energy" $E_t = f(x_t) + \lVert v_t \rVert^2/(2\eta)$, the height plus a kinetic term, and show that it decreases monotonically under their conditions (Lemma 4) even when $f$ alone goes up.

## 8. Acceleration without convexity

Section 6 needed a quadratic, and section 6's theorem needed convexity. Does momentum still help on non-convex losses? Yes, in the worst-case counts:

:::theorem Accelerated rates for non-convex problems
- [Carmon, Duchi, Hinder & Sidford 2018](paper.html?id=carmon2018accelerated): for $f$ with Lipschitz gradient and Lipschitz Hessian, an accelerated method that uses only gradients finds an [[eps-stationary|ε-stationary point]] in $O(\varepsilon^{-7/4}\log(1/\varepsilon))$ gradient evaluations, against $O(\varepsilon^{-2})$ for GD, and the point it returns also satisfies $\nabla^2 f \succeq -O(\varepsilon^{1/2}) I$.
- [Jin, Netrapalli & Jordan 2018](paper.html?id=jin2018accelerated), Theorem 3: perturbed accelerated gradient descent (Nesterov's step, plus a random kick near flat points as in stop 6, plus a "negative curvature exploitation" step) finds an ε-[[sosp|second-order stationary point]] with high probability in $\tilde O(\varepsilon^{-7/4})$ iterations, against $\tilde O(\varepsilon^{-2})$ for [[perturbed-gd|perturbed GD]]. It is the first single-loop, Hessian-free method shown to be faster than GD at this task.
:::

Both need the Hessian to be Lipschitz ([[hessian-lipschitz|constant $\rho$]], stop 5), and the hidden factors include powers of $L$ and $\rho$ and, for Jin et al., logarithms. In numbers: at $\varepsilon = 10^{-4}$, $\varepsilon^{-2} = 10^8$ and $\varepsilon^{-7/4} = 10^7$, ten times fewer, before constants and logarithms. The gain $\varepsilon^{-1/4}$ is real but smaller than the $\sqrt\kappa$ of the convex world; stop 12 shows how close this is to the best possible.

## 9. Lookahead: averaging the trajectory

Momentum averages *gradients*. A different kind of memory averages *positions*.

:::def Lookahead
[[lookahead|Lookahead]] (Zhang, Lucas, Hinton & Ba 2019) keeps two copies of the weights. The **fast weights** $\theta$ run any inner optimizer (SGD, heavy ball, Adam) for $k$ steps, starting from the **slow weights** $\phi$. Then the slow weights move a fraction $\alpha$ of the way toward where the fast weights ended,

$$\phi_{t+1} = \phi_t + \alpha\,(\theta_{t,k} - \phi_t),$$

and the fast weights restart from $\phi_{t+1}$. ("$k$ steps forward, 1 step back.")
:::

Unrolled, the slow weights are an exponential moving average of the endpoints of the inner runs. When the inner optimizer oscillates, the endpoints scatter around the valley floor and their average lands closer to it. Zhang et al. show this on a deterministic quadratic (Sec. 3.2, Fig. 4, with $k = 20$, $\alpha = 0.5$): wrapped around classical momentum, Lookahead **improves the convergence rate when the momentum is under-damped** (oscillating), and is slightly worse when it is over-damped. On a noisy quadratic it lowers the variance of the iterates (Proposition 2; noise is stop 8).

In the visual, the heavy ball has $\beta = 0.99$, far too much momentum for the valley with $\kappa = 100$ ($\eta = 0.1$). Alone it needs 1425 steps to reach $f < 10^{-8}$; wrapped in Lookahead with $k = 20$, $\alpha = 0.5$ it needs 340. Per step, the error along the valley's two curvatures shrinks by $0.9950$ alone and $0.9759$ with Lookahead. With Polyak's tuning instead, Lookahead is slower ($0.980$ per step against $0.818$): it repairs bad momentum, it does not improve good momentum.

:::viz lookahead
:::

:::warn What Lookahead's experiments show
The paper's benefits on real networks are empirical, and the differences are small: on CIFAR-10 its Table 1 reports 95.27% accuracy for Lookahead against 95.23% for SGD and 95.26% for averaging the SGD iterates. Its stated advantages are robustness to the inner optimizer's settings and lower variance, not a new convergence rate.
:::

:::atlas Where this shows up in the Atlas
- [Polyak 1964](paper.html?id=polyak1964some): the heavy-ball method and its $\sqrt\kappa$ rate on quadratics (sections 2 and 6).
- [Sutskever, Martens, Dahl & Hinton 2013](paper.html?id=sutskever2013importance) found that deep and recurrent networks train well with momentum when the initialization is well designed and $\beta$ is raised slowly during training, and markedly worse when momentum is absent or poorly tuned.
- [Jin, Netrapalli & Jordan 2018](paper.html?id=jin2018accelerated) and [Carmon et al. 2018](paper.html?id=carmon2018accelerated): acceleration for non-convex problems (section 8).
- [Zhang et al. 2019](paper.html?id=zhang2019lookahead): Lookahead (section 9).
- [Kingma & Ba 2015](paper.html?id=kingma2015adam): Adam's first moment is the EMA of section 3 (stop 9).
:::

:::problem The next problem
Every method so far evaluates the **full gradient** $\nabla f(x) = \frac1n \sum_i \nabla \ell_i(x)$ at every step: one pass over all $n$ training examples per step. With millions of examples that is far too expensive. **What happens if we use a cheap, noisy estimate of the gradient instead?**
:::
