## 1. AdaGrad: a learning rate for each coordinate

This chapter follows Sections III-D to III-H of the survey and Appendix A-B. A single learning rate assumes every parameter wants the same scale. Sparse or heterogeneous features break that: a weight attached to a rare feature sees a gradient only now and then. **[[adagrad|AdaGrad]]** accumulates each coordinate's squared gradients (survey Eqs. 20 and 21):

$$s_{t,j} = s_{t-1,j} + \hat g_{t,j}^2, \qquad \theta_{t+1,j} = \theta_{t,j} - \frac{\eta}{\sqrt{s_{t,j}} + \varepsilon}\,\hat g_{t,j}.$$

A coordinate with frequent large gradients builds a large denominator and takes smaller steps. A rarely active coordinate keeps a larger effective step. The survey's example: two features whose cumulative squared gradients are $100$ and $1$ get effective learning rates $\eta/\sqrt{100} = \eta/10$ and $\eta$. The accumulated squares are **not** Hessian entries. They are gradient statistics; the analogy with curvature (chapter 6) is a lens, not an identity. AdaGrad comes from online convex optimization, where its guarantee is a regret bound.

:::viz sparse
:::

:::read How to read this visual
**What is drawn.** The empirical logistic loss of 4000 examples with two features, over the plane of the two weights $(w_1, w_2)$. Feature $x_1$ appears in every example; $x_2$ in only 3% of them. Both balls take one random example per step from $(-2.5, -2.5)$. Violet: SGD with one decaying step $\eta/\sqrt t$. Coral: diagonal AdaGrad. The chart follows the rare feature's weight $w_2$.

**What to notice.** SGD's shared step has shrunk long before the rare weight $w_2$ has received many gradients, so $w_2$ barely moves. AdaGrad divides each coordinate by the root of *its own* accumulated squares: $w_2$, having seen few gradients, keeps a large step and heads to its optimum. This is the original case for per-coordinate learning rates.
:::

AdaGrad's weakness is visible in Eq. (20): $s_{t,j}$ never decreases. Over a long run every effective step shrinks, and the method cannot forget early gradients when the landscape changes.

## 2. RMSProp: forget the distant past

RMSProp replaces the cumulative sum with an exponential moving average (survey Eq. 22):

$$s_t = \rho s_{t-1} + (1 - \rho)\,\hat g_t^{\odot2}, \qquad \theta_{t+1} = \theta_t - \eta\frac{\hat g_t}{\sqrt{s_t} + \varepsilon},$$

where the square $\odot2$, the root and the division act coordinate by coordinate. Old squared gradients are discounted, so the scale can adapt when training enters a new region. The smoothing factor $\rho$ is a new timescale: near one, the denominator averages over a long history and changes slowly; small, it reacts quickly but inherits more minibatch noise. A method that "adjusts learning rates automatically" still has parameters that decide how.

## 3. Adam: momentum plus adaptive scaling

**[[adam|Adam]]** keeps two exponential averages of the stochastic gradient (survey Eqs. 23 and 24): a first moment $m_t$ and a second *raw* moment $v_t$,

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\hat g_t^{\odot2}.$$

Both start at zero, so early on they are too small. Appendix A-B makes this exact. With a constant gradient $g$ and $m_0 = 0$, the recursion gives a finite geometric series (survey Eq. 42):

$$m_t = (1 - \beta)\sum_{k=0}^{t-1}\beta^k g = (1 - \beta^t)\,g.$$

Dividing by $1 - \beta^t$ returns $g$ exactly. Adam applies this **[[bias-correction|bias correction]]** to both averages and then takes the step (survey Eq. 25):

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_{t+1} = \theta_t - \eta\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}.$$

:::viz bias
:::

:::read How to read this visual
**What is drawn.** Feed Adam the same gradient at every step, with $\beta_1 = 0.9$, and plot the size of its step divided by $\alpha$, the learning rate (the survey's $\eta$), against the step number on a log axis. Coral: without bias correction, where the step is $(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$ times $\alpha$. Green: with the correction of Eq. (25).

**What moves.** $\beta_2$: $0.99$, $0.999$ or $0.9999$.

**What to notice.** Without correction the two averages are too small by *different* amounts, so their ratio is wrong. For $\beta_2 = 0.999$ the first step is $3.16\alpha$, and it overshoots to $6.57\alpha$ before settling back to $\alpha$ only after about $1/(1-\beta_2) = 1000$ steps. With the correction the step is exactly $\alpha$ from the first step on. This is why Adam needs *separate* corrections for its two averages.
:::

Two remarks from the survey. First, $\hat v_t$ is $\mathbb{E}[g_j^2]$-like, not a Hessian diagonal: even when Adam's rescaling looks like a curvature correction on some problems, it does not follow that $\hat v_t \approx \partial^2 L/\partial\theta_j^2$. Second, a numerical example. If a coordinate has $\hat m = 0.3$ and $\hat v = 0.09$, its step is $0.3/\sqrt{0.09} = 1$ times $\eta$. With $\hat v = 0.36$ it is $0.3/0.6 = 0.5$ times $\eta$. The *relative* scaling is the practical idea. Because the update is not just $-\eta g$, Adam is not invariant to a change of parameter coordinates.

## 4. Adam's convergence caveat and AMSGrad

Popularity is not a convergence theorem. Reddi, Kale and Kumar built a simple online convex problem on which Adam's exponential weighting fails to converge. Their fix, AMSGrad, keeps a running maximum of the second moment, coordinate by coordinate (survey Eq. 26):

$$\bar v_t = \max(\bar v_{t-1}, v_t), \qquad \Delta_t \propto -\frac{m_t}{\sqrt{\bar v_t} + \varepsilon}.$$

:::viz adamfail
:::

:::read How to read this visual
**What is drawn.** A one-dimensional problem in the spirit of Reddi et al.: minimize $f(\theta) = \theta$ on $[-1, 1]$, whose answer is $\theta = -1$. Each stochastic gradient is $k^2$ with probability $1/k$ and $-k$ otherwise. On average it is $+1$, so it points the right way. The curves are $\theta_t$ over 400 000 steps (log time axis) for Adam (coral), AMSGrad (white) and ADOPT (violet, a later fix).

**What moves.** The rarity $k$ of the big gradient and $\beta_2$.

**What to notice.** With small $\beta_2$, Adam is pushed to the *worst* point, $\theta = +1$. The rare large gradient is quickly forgotten by the second moment, so the frequent small wrong-way gradients get large steps. AMSGrad's running maximum remembers the big gradient and reaches $-1$ for $k = 10$; for $k = 50$ it crawls. The lesson, in the survey's words: not that Adam always diverges, but that smooth heuristic adaptations can have subtle stability conditions. Later analyses control step-size variation, moment schedules or bounded gradients, and their theorems are comparable only when their settings match.
:::

## 5. AdamW: weight decay is not an L2 penalty

With a quadratic penalty $(\lambda/2)\|\theta\|^2$, the penalized gradient is $g + \lambda\theta$. For plain SGD, adding the penalty and shrinking the weights are the same thing (survey Eq. 27):

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta g_t.$$

Under an adaptive scaling $D_t^{-1}$ they are not. Putting $\lambda\theta$ inside the adaptive step gives (survey Section III-H)

$$\theta_{t+1} = \theta_t - \eta D_t^{-1}(g_t + \lambda\theta_t),$$

so the shrinkage of each weight is divided by that weight's own scale. This is no longer uniform weight decay. **AdamW** [[decoupled-decay|decouples]] the shrinkage from the adaptive step (survey Eq. 28):

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}.$$

:::viz adamw
:::

:::read How to read this visual
**What is drawn.** For one step, the fraction of a weight that the decay term removes, against that coordinate's adaptive scale $\sqrt{\hat v}$; both axes are logarithmic. Coral: the L2 penalty inside Adam, which removes $\eta\lambda/(\sqrt{\hat v} + \varepsilon)$ (survey Section III-H). Green: AdamW, which removes $\eta\lambda$ from every weight (Eq. 28). The dashed line at 1 means the decay alone would remove the whole weight. The coral dots mark three example scales, $10^{-4}$, $10^{-2}$ and $1$.

**What moves.** The learning rate $\eta$ and the decay coefficient $\lambda$.

**What to notice.** The green line is flat: AdamW treats every weight alike. The coral line falls with slope $-1$. A coordinate with large gradients (right) is barely regularized, and a coordinate with tiny gradients (left) is regularized hard, until the decay hits the cap. The readout gives the factor between the two ends. This is the "regularization implementation issue" that AdamW corrects; as the survey stresses, it is not a convergence guarantee.
:::

The survey adds a fairness warning that applies to every comparison in this guide. Learning-rate schedules (warm restarts, cyclical rates), initialization and normalization all change the geometry the optimizer sees. A comparison that tunes one optimizer carefully and another barely may be testing a training recipe, not an update rule.

:::survey Fig. 5 of the survey: Adam builds two different gradient statistics. The first moment estimates direction; the second moment controls coordinate-wise scaling. Neither equals the Hessian.
public/survey3/fig5.webp
:::

:::read How to read this figure
**Top.** The stochastic gradient $\hat g_t$ feeds two streams. The upper stream (orange) is the first moment $m_t$ of Eq. (23), a smoothed *direction*. The lower stream (green) is the second raw moment $v_t$ of Eq. (24), built from the elementwise square, a per-coordinate *scale*. Each stream is bias-corrected (Eq. 25). The update divides the corrected first moment by the root of the corrected second moment. The note under the update repeats the survey's warning: $\hat v_t$ is $\mathbb{E}[g_j^2]$-like, not a second derivative.

**Bottom.** Persistent optimizer state in scalars, from Table II, with parameters and gradients excluded: SGD none, momentum $d$, Lion $d$, Adam and AdamW $2d$ (the arrays $m$ and $v$). For one $m\times n$ matrix, Adafactor's factored second moment needs only $m + n$, against $2mn$ for Adam. That is the subject of chapter 9.
:::

:::problem The next step
Adam's two arrays double the memory of momentum. Chapter 9 looks at designs that keep less.
:::
