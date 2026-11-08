## 1. Gradient descent

This chapter follows Section II-E of the survey: the algorithms whose behavior the rest of the survey studies. We begin with the simplest.

**[[gd|Gradient descent]]** (GD) repeats one rule: from the current point $\theta_t$, step against the gradient,

$$\theta_{t+1} = \theta_t - \eta\,\nabla L(\theta_t).$$

The number $\eta > 0$ is the **[[step-size|step size]]**, or learning rate. It is the one knob of GD, and the second half of the survey is about what happens when it is large.

What can GD guarantee? Under Assumption 1 ($\ell$-smoothness) with $\eta = 1/\ell$, GD finds an $\epsilon$-FOSP in $O(\ell\Delta/\epsilon^2)$ iterations, where $\Delta = L(\theta_0) - \inf_\theta L(\theta)$ is how far above the best value it starts (Nesterov 2004). $O(\cdot)$ means "at most a constant times". No first-order method can do better on this class of functions in the worst case (Carmon et al. 2020). Note what is promised: a small gradient, not a minimum.

The effect of $\eta$ is clearest on the simplest loss there is, a parabola $f(x) = \tfrac{a}{2}x^2$ with curvature $a$. One step gives $x_{t+1} = x_t - \eta a x_t = (1 - \eta a)\,x_t$. Every step multiplies $x$ by the same number, $1 - \eta a$. Chapter 5 shows that this simple fact describes GD near *any* critical point.

:::viz parabola
:::

:::read How to read this visual
**What is drawn.** The curve is the parabola $f(x) = \tfrac{a}{2}x^2$; the horizontal axis is $x$ and the vertical axis is $f(x)$. The coral dots are the GD iterates $x_0 = 2, x_1, x_2, \dots$ (up to 25), and the lines join consecutive steps.

**What to try.** Keep the curvature $a = 4$ and slide the step size $\eta$ slowly from $0.01$ to $0.8$. Watch for the two special values: $\eta = 1/a = 0.25$ and $\eta = 2/a = 0.5$.

**What to notice.** The readout shows the factor $1 - \eta a$ and the size of $x_{25}$. Below $1/a$ the factor is between $0$ and $1$: the iterates creep down one side. At exactly $1/a$ the factor is $0$: one step lands on the minimum. Between $1/a$ and $2/a$ the factor is between $-1$ and $0$: the iterates jump from side to side but still shrink. Above $2/a$ the factor is below $-1$: every jump is bigger than the last, and GD explodes. **The threshold $\eta = 2/a$, or equivalently curvature $a = 2/\eta$, is the most important number in Section IV of the survey.**
:::

## 2. Stochastic gradient descent

Computing $\nabla L$ exactly means a pass over all $n$ training examples, which is expensive. **[[sgd|Stochastic gradient descent]]** (SGD), going back to Robbins & Monro (1951), replaces it by a cheap random estimate $g_t = g(\theta_t;\xi_t)$, typically the gradient on a random [[minibatch]] of examples:

$$\theta_{t+1} = \theta_t - \eta\, g_t.$$

Under Assumptions 1 and 3, Ghadimi & Lan (2013) showed that a randomly chosen SGD iterate satisfies $\mathbb{E}\lVert\nabla L(\theta)\rVert^2 \le \epsilon^2$ after $O(\epsilon^{-4})$ stochastic gradients. Arjevani et al. (2023) proved that no general stochastic first-order method can do better. This is the baseline rate every stochastic method is compared against.

SGD has a characteristic behavior with a constant step: it gets close to a minimum quickly, then keeps bouncing around it, because the noise never stops.

:::viz floor
:::

:::read How to read this visual
**What is drawn.** The surface is the loss of fitting a line to 50 points (log height scale), as in the minibatch visual of chapter 3. The chart below plots $f(\theta_t) - f^\star$, the distance above the minimum, against the step $t$, on a log scale.

**What moves.** Three balls start at the same point and run for 4000 steps. Coral: SGD with a constant step $\eta$ and batch size $B$. Violet: SGD whose step shrinks over time, $\eta_t = \eta/(1 + t/200)$. White: full-batch GD with the same $\eta$, which has no noise.

**What to try.** Change $\eta$ and $B$, then *Run again*.

**What to notice.** The coral curve drops, then flattens into a noisy band: the noise floor. Its height grows with $\eta$ and shrinks with $B$, and the readout compares it with the small-step prediction $\eta\sigma^2/(4B)$. The violet curve keeps going down, because its noise shrinks with the step. The white curve has no floor. Remember this: noise is a nuisance near a minimum, but chapter 8 shows that near a saddle it can be exactly what is needed.
:::

## 3. Momentum

Momentum methods give the iterate a memory. The **[[heavy-ball|heavy-ball]]** method (Polyak 1964) adds a fraction $\beta \in [0, 1)$ of the previous step:

$$\theta_{t+1} = \theta_t - \eta\nabla L(\theta_t) + \beta\,(\theta_t - \theta_{t-1}).$$

Nesterov's accelerated gradient is similar, but evaluates the gradient at a look-ahead point. In a long narrow valley, plain GD zig-zags across the steep direction while crawling along the flat one. With momentum, the consistent part of the gradient accumulates, while the part that flips sign at every step cancels.

:::viz valley
:::

:::read How to read this visual
**What is drawn.** A long, narrow valley $f(x, y) = \tfrac12(\mu x^2 + 10y^2)$: curvature $\mu$ along $x$ (flat) and $10$ along $y$ (steep). The ratio $\kappa = 10/\mu$ is the condition number; the $y$-axis is stretched so the valley is visible. The chart plots $f(\theta_t)$ on a log scale against the step $t$.

**What moves.** Three balls start at $(-1.45, 0.5)$. Violet: GD with its best fixed step. Coral: heavy ball. White: Nesterov. The two momentum methods share $\eta$ and $\beta$.

**What to try.** Press *Polyak's tuning* and *Nesterov's tuning*, switch $\kappa$ between 10 and 100, then move $\eta$ and $\beta$ yourself.

**What to notice.** The readout counts steps to $f < 10^{-8}$ for each method. Violet zig-zags and needs many steps. The momentum methods glide along the valley and need far fewer, roughly $\sqrt{\kappa}$ instead of $\kappa$ times a constant. The survey uses momentum in two places: heavy ball escapes the saddle earliest in the survey's Fig. 5 (chapter 8), and momentum changes the stability threshold of chapter 11 from $2/\eta$ to $(2 + 2\beta)/\eta$.
:::

## 4. Adaptive methods: a step size per direction

Adaptive methods rescale each coordinate by statistics of its past gradients. **AdaGrad** (Duchi, Hazan & Singer 2011) divides each coordinate by the square root of the sum of its past squared gradients. **[[adam|Adam]]** (Kingma & Ba 2015) combines momentum with an exponential moving average $v_t$ of squared gradients:

$$\theta_{t+1} = \theta_t - \eta\,\frac{m_t}{\sqrt{v_t} + \varepsilon},$$

where $m_t$ is a moving average of gradients and $\varepsilon$ a small constant. A useful way to see all of them at once: an adaptive method is gradient descent with a **[[preconditioning|preconditioner]]** $P_t$,

$$\theta_{t+1} = \theta_t - \eta\, P_t^{-1} m_t,$$

where $P_t$ is a diagonal matrix built from the past gradients. Its local stability is governed not by the Hessian $\nabla^2 L$ but by the **preconditioned Hessian** $P_t^{-1}\nabla^2 L$. Chapter 14 returns to this: Adam has its own edge of stability, measured on $P^{-1}\nabla^2 L$.

:::viz precond
:::

:::read How to read this visual
**What is drawn.** A map, seen from above, of a narrow quadratic valley $f = \tfrac12 x^\top H x$ with condition number $\kappa$, turned by an angle you choose. The axes are the two coordinates $x_1, x_2$. The green dot is the minimum.

**What moves.** Three paths start at $(-1.3, 1.1)$ and show their first 30 steps. Violet: GD. Coral: GD with a diagonal preconditioner $\mathrm{diag}(H)^{-1}$, which divides each coordinate by its own curvature, the idealized version of what AdaGrad and Adam estimate. White: Newton's method, which uses the full inverse Hessian $H^{-1}$.

**What to try.** Turn the valley from $0°$ to $90°$, and switch $\kappa$.

**What to notice.** At $0°$ or $90°$ the valley is aligned with the coordinates, and the diagonal preconditioner makes it perfectly round: the coral path goes straight in. As the valley turns towards $45°$, a per-coordinate rescaling cannot see the tilt, and the advantage disappears. The readout gives the condition number after rescaling and the steps to $f < 10^{-8}$ for each method. Newton converges in one step at every angle, because it uses the whole Hessian.
:::

## 5. Newton's method, and why it is drawn to saddles

**[[newton|Newton's method]]** uses curvature directly. It jumps to the critical point of the local quadratic model:

$$\Delta = -\big(\nabla^2 L\big)^{-1}\nabla L.$$

On a bowl this is ideal. At a saddle it is a disaster. Along an eigen-direction with eigenvalue $\lambda$, Newton rescales the gradient component by $1/\lambda$. When $\lambda < 0$, this flips the sign: the step that should go *down* the negative-curvature direction goes *up* it, towards the saddle (Dauphin et al. 2014). Newton's method is attracted to saddles rather than repelled. Cubic regularization (Nesterov & Polyak 2006) and related methods repair this, but they are algorithms built to escape saddles, which our survey leaves to Jin et al. (2021). We study what *unmodified* gradient methods do.

:::viz newton
:::

:::read How to read this visual
**What is drawn.** The surface $f = \tfrac12x^2 - \tfrac12y^2 + \tfrac14y^4 + 0.3xy^2$: a strict saddle at the center and two minima (green). Every run starts at $(1.2, y_0)$, slightly off the saddle's stable line.

**What moves.** White: Newton's method (or, with the button, saddle-free Newton, which divides by $|\lambda|$ instead of $\lambda$). Violet: cubic regularization, which adds a penalty $(M/6)\lVert\Delta\rVert^3$ to the quadratic model. Coral: plain gradient descent with $\eta = 1/8$.

**What to try.** Run all three, then switch the white ball to saddle-free Newton, and move $y_0$.

**What to notice.** Newton jumps straight to the saddle and stays there: its quadratic model's critical point *is* the saddle. Saddle-free Newton and cubic regularization leave quickly. Gradient descent also leaves, more slowly, because it simply follows the slope, and along the negative-curvature direction the slope points away from the saddle. Plain GD is never attracted to a strict saddle. Chapter 5 shows why, in one line of algebra.
:::

:::plain How results are counted
Complexity results in the survey are stated in an **oracle model**: we count the calls to a gradient or stochastic-gradient oracle needed to output an approximate FOSP or SOSP. "$O(\epsilon^{-4})$ stochastic gradients" means the number of calls grows like $1/\epsilon^4$ as the tolerance $\epsilon$ shrinks. It ignores the cost of each call, memory and parallelism.
:::

:::problem The next step
Everything in this chapter came back to one pattern: one step multiplies something by a fixed number. Chapter 5 makes this exact near every critical point, and finds that it has three regimes. Two of them are the two halves of the survey.
:::
