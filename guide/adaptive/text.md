## 1. AdaGrad: why one learning rate may be wrong

This chapter follows Sections III-D to III-H of the survey. Suppose the gradient is $g = (10, 1)^\top$. The first coordinate is much larger, perhaps consistently, while the second only sees occasional gradients, as with a rare feature. A single learning rate treats both alike and ignores those different histories. **[[adagrad|AdaGrad]]** keeps one accumulated squared gradient per coordinate (survey Eqs. 20 and 21):

$$s_{t,j} = s_{t-1,j} + \hat g_{t,j}^{\,2}, \qquad \theta_{t+1,j} = \theta_{t,j} - \frac{\eta}{\sqrt{s_{t,j} + \varepsilon}}\,\hat g_{t,j}.$$

The index $j$ picks one parameter, and the small $\varepsilon$ prevents division by zero. A coordinate that keeps receiving large gradients builds a large denominator and takes smaller steps.

**One step in numbers.** Start from $s_0 = (0, 0)$ with $\hat g_1 = (10, 1)$. Then $s_1 = (100, 1)$, and, ignoring $\varepsilon$ for the illustration,

$$\frac{\hat g_1}{\sqrt{s_1}} = \Big(\frac{10}{10}, \frac{1}{1}\Big) = (1, 1).$$

AdaGrad has evened out the initially larger gradient. Over time, coordinates with large gradients get smaller effective learning rates, and rarely updated ones keep larger ones, which is what makes it useful for sparse features.

:::viz sparse
:::

:::read How to read this visual
**What is drawn.** The empirical logistic loss of 4000 examples with two features, over the plane of the two weights $(w_1, w_2)$. Feature $x_1$ appears in every example; $x_2$ in only 3% of them. Both balls take one random example per step from $(-2.5, -2.5)$. Violet: SGD with one decaying step $\eta/\sqrt t$. Coral: diagonal AdaGrad. The chart follows the rare feature's weight $w_2$.

**What to notice.** SGD's shared step has shrunk long before the rare weight $w_2$ has received many gradients, so $w_2$ barely moves. AdaGrad divides each coordinate by the root of *its own* accumulated squares: $w_2$, having seen few gradients, keeps a large step and heads to its optimum. This is the original case for per-coordinate learning rates.
:::

**Why AdaGrad can slow down too much.** The accumulator is a sum of nonnegative squares, so it can only grow. Even when training moves to a region where a parameter needs larger steps again, old squared gradients stay in its denominator, and its effective learning rate may become excessively small.

:::key Remember III-D
AdaGrad adapts each coordinate using its whole squared-gradient history. Its weakness is that it never forgets old magnitudes.
:::

## 2. RMSProp: forget old magnitudes gradually

**[[rmsprop|RMSProp]]** changes one detail: instead of summing squared gradients forever, it keeps an exponential moving average (survey Eq. 22):

$$s_t = \rho s_{t-1} + (1 - \rho)\hat g_t^{\odot2}, \qquad \theta_{t+1} = \theta_t - \eta\frac{\hat g_t}{\sqrt{s_t + \varepsilon}}.$$

Here $\rho \in (0, 1)$ is a forgetting factor, $\odot2$ means squaring each entry, and the square root and division also act entry by entry.

**What the moving average does.** In one dimension, with $\rho = 0.9$, $s_0 = 0$ and gradients $g_1 = 10$, $g_2 = 1$:

$$s_1 = 0.9(0) + 0.1(100) = 10, \qquad s_2 = 0.9(10) + 0.1(1) = 9.1.$$

The estimate already starts to fall as the gradient shrinks, and the old large value fades if small gradients continue. AdaGrad's sum would go from $100$ to $101$ and never come down. A large $\rho$ adapts smoothly but slowly; a small $\rho$ reacts quickly but fluctuates more.

:::key Remember III-E
RMSProp keeps adaptive per-coordinate scaling but lets the scale change as training conditions change.
:::

## 3. Adam: momentum plus adaptive scaling

Momentum smooths the direction; RMSProp adapts each coordinate's magnitude. **[[adam|Adam]]** combines both with two moving averages (survey Eqs. 23 and 24):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\hat g_t^{\odot2}.$$

$m_t$ is the first moment, a smoothed direction; $v_t$ is the second *raw* moment, a smoothed squared magnitude per parameter; $\beta_1$ and $\beta_2$ decide how long each remembers. $v_t$ is not the Hessian: squared gradients are not second derivatives of the loss.

**Why bias correction?** Both averages start at zero. With a constant gradient $g = 4$ and $\beta_1 = 0.9$, the first step gives $m_1 = 0.9(0) + 0.1(4) = 0.4$, although the gradient is $4$. The zero start drags early averages toward zero. Adam divides it back out:

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t},$$

so here $\hat m_1 = 0.4/(1 - 0.9) = 4$. The update is (survey Eq. 25)

$$\theta_{t+1} = \theta_t - \eta\frac{\hat m_t}{\sqrt{\hat v_t + \varepsilon}}.$$

The numerator supplies a smoothed direction; the denominator, a per-coordinate scale.

**A first step in numbers.** Let $g_1 = (2, 8)^\top$, $\beta_1 = 0.9$, $\beta_2 = 0.99$ and zero initial states. Then $m_1 = (0.2, 0.8)^\top$ and $v_1 = (0.04, 0.64)^\top$; after correction $\hat m_1 = (2, 8)^\top$ and $\hat v_1 = (4, 64)^\top$, so, ignoring $\varepsilon$, $\hat m_1/\sqrt{\hat v_1} = (1, 1)^\top$. Adam does not always take equal steps: later moments have different histories and the ratio changes as training goes on.

:::viz bias
:::

:::read How to read this visual
**What is drawn.** Feed Adam the same gradient at every step, with $\beta_1 = 0.9$, and plot the size of its step divided by $\alpha$, the learning rate (the survey's $\eta$), against the step number on a log axis. Coral: without bias correction, where the step is $(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$ times $\alpha$. Green: with the correction of Eq. (25).

**What moves.** $\beta_2$: $0.99$, $0.999$ or $0.9999$.

**What to notice.** Without correction the two averages are too small by *different* amounts, so their ratio is wrong. For $\beta_2 = 0.999$ the first step is $3.16\alpha$, and it overshoots to $6.57\alpha$ before settling back to $\alpha$ only after about $1/(1-\beta_2) = 1000$ steps. With the correction the step is exactly $\alpha$ from the first step on. This is why Adam needs *separate* corrections for its two averages.
:::

:::survey Fig. 5 of the survey: Adam builds two different gradient statistics. The first moment estimates direction; the second moment controls coordinate-wise scaling. Neither equals the Hessian.
public/survey3/fig5.webp
:::

:::read How to read this figure
**Top.** The stochastic gradient $\hat g_t$ feeds two streams. The upper stream (orange) is the first moment $m_t$ of Eq. (23), a smoothed *direction*. The lower stream (green) is the second raw moment $v_t$ of Eq. (24), built from the elementwise square, a per-coordinate *scale*. Each stream is bias-corrected (Eq. 25). The update divides the corrected first moment by the root of the corrected second moment. The note under the update repeats the survey's warning: $\hat v_t$ is $\mathbb{E}[g_j^2]$-like, not a second derivative.

**Bottom.** Persistent optimizer state in scalars, from Table II, with parameters and gradients excluded: SGD none, momentum $d$, Lion $d$, Adam and AdamW $2d$ (the arrays $m$ and $v$). For one $m\times n$ matrix, Adafactor's factored second moment needs only $m + n$, against $2mn$ for Adam. That is the subject of chapter 9.
:::

**Adam's cost.** For $d$ parameters it stores two extra length-$d$ vectors, $m$ and $v$: twice momentum's state, with cheap per-coordinate arithmetic. That combination largely explains its popularity.

:::key Remember III-F
Adam combines momentum for direction with squared-gradient history for scale, plus bias correction for its zero-started averages. It is an adaptive first-order method, not a second-order one.
:::

## 4. Adam's convergence caveat and AMSGrad

Adam's second moment can rise or fall as old gradients fade. When it falls, the denominator $\sqrt{\hat v_t}$ shrinks and that coordinate's effective learning rate grows. Reddi, Kale and Kumar built an online convex problem on which the original Adam fails to converge because of exactly this. It does not mean Adam generally fails in practice; it means that popularity and good results do not by themselves establish an unconditional theorem.

**AMSGrad** keeps a running maximum of the second-moment estimates, entry by entry (survey Eq. 26):

$$\bar v_t = \max(\bar v_{t-1}, v_t).$$

If a coordinate's estimates are $v_1 = 9$, $v_2 = 4$, $v_3 = 7$, its running maximum is $9, 9, 9$. The denominator can no longer shrink, which removes one source of growing effective learning rates and supports stronger guarantees under suitable assumptions.

:::viz adamfail
:::

:::read How to read this visual
**What is drawn.** A one-dimensional problem in the spirit of Reddi et al.: minimize $f(\theta) = \theta$ on $[-1, 1]$, whose answer is $\theta = -1$. Each stochastic gradient is $k^2$ with probability $1/k$ and $-k$ otherwise. On average it is $+1$, so it points the right way. The curves are $\theta_t$ over 400 000 steps (log time axis) for Adam (coral), AMSGrad (white) and ADOPT (violet, a later fix).

**What moves.** The rarity $k$ of the big gradient and $\beta_2$.

**What to notice.** With small $\beta_2$, Adam is pushed to the *worst* point, $\theta = +1$. The rare large gradient is quickly forgotten by the second moment, so the frequent small wrong-way gradients get large steps. AMSGrad's running maximum remembers the big gradient and reaches $-1$ for $k = 10$; for $k = 50$ it crawls. The lesson, in the survey's words: not that Adam always diverges, but that smooth heuristic adaptations can have subtle stability conditions. Later analyses control step-size variation, moment schedules or bounded gradients, and their theorems are comparable only when their settings match.
:::

:::key Remember III-G
AMSGrad addresses one specific instability in Adam's adaptive history. The lesson is the distinction between a theorem under assumptions and empirical training success.
:::

## 5. AdamW: regularization is not weight decay inside Adam

To discourage large weights, add an L2 penalty: $L_{\text{reg}}(\theta) = L(\theta) + \tfrac{\lambda}{2}\|\theta\|_2^2$, whose gradient is $g + \lambda\theta$. For plain gradient descent (survey Eq. 27),

$$\theta_{t+1} = \theta_t - \eta(g + \lambda\theta_t) = (1 - \eta\lambda)\theta_t - \eta g:$$

first shrink the weights by $(1 - \eta\lambda)$, then take the ordinary step. For plain GD, the L2 penalty and this **[[decoupled-decay|weight decay]]** are the same thing.

**Why the equivalence fails for Adam.** Adam scales each coordinate by its own history. If $\lambda\theta$ is added to the gradient and fed through Adam, the penalty is scaled too, so the shrinkage is no longer a uniform $(1 - \eta\lambda)$: weights with large gradients are barely regularized. **AdamW** applies the decay separately from the adaptive step (survey Eq. 28):

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta\frac{\hat m_t}{\sqrt{\hat v_t + \varepsilon}}.$$

With $\theta = (10, 1)$, $\eta = 0.1$ and $\lambda = 0.01$, the decay factor is $1 - 0.001 = 0.999$, so decay alone takes the weights to $(9.99, 0.999)$ before the loss step, the same fraction for both, however small either coordinate's Adam denominator happens to be.

:::viz adamw
:::

:::read How to read this visual
**What is drawn.** For one step, the fraction of a weight that the decay term removes, against that coordinate's adaptive scale $\sqrt{\hat v}$; both axes are logarithmic. Coral: the L2 penalty inside Adam, which removes $\eta\lambda/(\sqrt{\hat v} + \varepsilon)$ (survey Section III-H). Green: AdamW, which removes $\eta\lambda$ from every weight (Eq. 28). The dashed line at 1 means the decay alone would remove the whole weight. The coral dots mark three example scales, $10^{-4}$, $10^{-2}$ and $1$.

**What moves.** The learning rate $\eta$ and the decay coefficient $\lambda$.

**What to notice.** The green line is flat: AdamW treats every weight alike. The coral line falls with slope $-1$. A coordinate with large gradients (right) is barely regularized, and a coordinate with tiny gradients (left) is regularized hard, until the decay hits the cap. The readout gives the factor between the two ends. This is the "regularization implementation issue" that AdamW corrects; as the survey stresses, it is not a convergence guarantee.
:::

:::key Remember III-H
AdamW separates weight shrinkage from adaptive gradient scaling. The difference is conceptual and practically important; it is not just a renaming of Adam.
:::

:::problem The next step
Adam stores two numbers for every weight. Chapter 9 looks at designs that keep less: Adafactor, Lion, and a few others.
:::
