## 1. A minibatch gradient is a random estimate

This chapter follows Section II-F of the survey and Appendix C-B. Computing $\nabla L$ exactly means a pass over every training example. Instead, take a random [[minibatch]] $B_t$ of examples and average their gradients (survey Eq. 10):

$$\hat g_t = \frac{1}{|B_t|}\sum_{i \in B_t}\nabla L_i(\theta_t) = g_t + \xi_t, \qquad \mathbb{E}[\xi_t \mid \theta_t] = 0.$$

Here $L_i$ is the loss on example $i$ and $\xi_t$ is the error of the estimate. Saying that $\xi_t$ averages to zero, that $\hat g_t$ is **unbiased**, is an assumption about how the batch is sampled; it is not automatic in every data pipeline. Suppose the per-example gradients have [[variance]] at most $\sigma^2$ around the true gradient and the samples are independent. Averaging $|B_t|$ of them divides the variance by $|B_t|$: the estimate has variance at most $\sigma^2/|B_t|$. Using such estimates is **[[sgd|stochastic gradient descent]]**.

:::viz cloud
:::

:::read How to read this visual
**What is drawn.** The loss of a line fit $wx + b$ to 50 points, over the plane of its two parameters (slope $w$, intercept $b$); the green dot is the minimum. Drag the white point: the white arrow is one full-gradient step of size $\eta = 0.08$ from it. Each coral dot is where a step using a random minibatch of $B$ examples would land instead, for 400 random minibatches. The violet dot is their average.

**What moves.** The point (drag it) and the batch size $B$.

**What to notice.** The coral cloud is centred on the tip of the white arrow: the average minibatch step is the full step (unbiasedness). The cloud's size shrinks as $B$ grows. The readout measures the spread $\mathbb{E}\|\hat g - \nabla f\|^2$ from the 400 samples and compares it with the exact $\sigma^2/B$; the two agree.
:::

A minibatch gradient is far cheaper than a full one, but it perturbs every update. Smaller batches give more updates per pass over the data; larger batches use parallel hardware better. Which is faster in practice depends on hardware and tuning, so the survey keeps "iterations to a given loss" and "time to a given loss" apart.

## 2. What noise does to the descent lemma

Substitute the noisy step $\Delta = -\eta\hat g_t$ into the descent lemma (Eq. 5 of chapter 3) and take the expectation over the batch. Unbiasedness gives $\mathbb{E}\|\hat g_t\|^2 = \|g_t\|^2 + \mathbb{E}\|\xi_t\|^2$: the mean square of the estimate is the true squared gradient plus the noise variance. The result is (survey Eq. 11)

$$\mathbb{E}_t[L_{t+1}] \le L_t - \eta\left(1 - \tfrac{\beta\eta}{2}\right)\|g_t\|^2 + \frac{\beta\eta^2\sigma^2}{2|B_t|}.$$

The first two terms are the deterministic decrease of chapter 3. The last term is new: noise adds a variance term that does not shrink as the gradient shrinks. Near a minimum, where $g_t$ is small, the noise term wins, and SGD with a constant step stops improving at an error floor whose size is set by $\eta$ and the batch size.

:::viz floor
:::

:::read How to read this visual
**What is drawn.** The same line-fit loss, in 3D, from the start $(1.8, -1.2)$. Three runs of 4000 steps. Coral: SGD with a constant step $\eta$. Violet: SGD with a decaying step $\eta_t = \eta/(1 + t/200)$. White: full-gradient descent with the same $\eta$. The chart shows $f(x_t) - f^\star$ on a log scale.

**What moves.** The step size and the batch size ($1$, $5$ or $25$).

**What to notice.** Constant-step SGD gets close fast and then rattles in a cloud: its curve flattens at a [[noise-floor|noise floor]], the last term of Eq. (11). A larger batch or a smaller step lowers the floor. Shrinking the step over time lets SGD keep improving. Full gradients have no noise and no floor.
:::

## 3. An explicit step-size choice (Appendix C-B)

Appendix C-B turns Eq. (11) into a complete bound. Let $\Delta_0 = L(\theta_0) - L_*$ be the initial gap to the lowest possible loss, use a constant step $\eta \le 1/\beta$ and batch size $B$. Then $1 - \beta\eta/2 \ge 1/2$, and Eq. (11) simplifies to (survey Eq. 75)

$$\mathbb{E}_t[L_{t+1}] \le L_t - \frac{\eta}{2}\|g_t\|^2 + \frac{\beta\eta^2\sigma^2}{2B}.$$

Take full expectations, add the inequalities for $t = 0, \dots, T-1$ (the losses telescope), use $L_T \ge L_*$, and divide by $\eta T/2$ (survey Eq. 76):

$$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\|g_t\|^2 \le \frac{2\Delta_0}{\eta T} + \frac{\beta\eta\sigma^2}{B}.$$

The first term shrinks as the step grows; the second grows. Setting the derivative in $\eta$ to zero (survey Eq. 77) balances them at (survey Eq. 78)

$$\eta_{\mathrm{bal}} = \sqrt{\frac{2\Delta_0 B}{\beta\sigma^2 T}}, \qquad \eta = \min\left\{\frac1\beta, \eta_{\mathrm{bal}}\right\}.$$

With this step, both terms are of order $T^{-1/2}$. So, through this particular worst-case bound, reaching an average squared gradient of $\epsilon^2$ (gradient norm $\epsilon$) takes $T = O(\epsilon^{-4})$ iterations. Without noise, chapter 7 shows $O(\epsilon^{-2})$. The survey's caution applies: these are upper bounds under different oracle models, not predictions of how fast SGD runs.

:::viz bound
:::

:::read How to read this visual
**What is drawn.** The right-hand side of Eq. (76) as a function of $\eta$, on log-log axes, with the illustrative constants $\Delta_0 = \beta = \sigma^2 = 1$. Blue dashed: the progress term $2\Delta_0/(\eta T)$. Red dashed: the noise term $\beta\eta\sigma^2/B$. White: their sum, the bound. The amber line and dot mark the chosen step $\min\{1/\beta, \eta_{\mathrm{bal}}\}$; the faint line marks $1/\beta$.

**The side chart.** The value of the bound at the chosen step as $T$ grows (white), against the noiseless bound $2\Delta_0\beta/T$ of deterministic GD with $\eta = 1/\beta$ (blue).

**What to try.** Raise $T$; then raise $B$.

**What to notice.** The bound is lowest where the two dashed lines cross, which is $\eta_{\mathrm{bal}}$. As $T$ grows, the crossing moves to smaller steps: more iterations call for a smaller step. A larger batch moves the crossing to the right. In the side chart, the white line falls with slope $-\tfrac12$ (order $T^{-1/2}$) once $\eta_{\mathrm{bal}} < 1/\beta$, while the noiseless line falls with slope $-1$. That difference in slopes is the $O(\epsilon^{-4})$ versus $O(\epsilon^{-2})$ gap.
:::

:::survey Fig. 12 of the survey: the bound of Eq. (76) with illustrative constants. (a) Its two terms against η; (b) the resulting bound against T.
public/survey3/fig12.webp
:::

:::read How to read this figure
This is the static version of the visual above, for $B = 32$. Panel (a) uses $T = 1000$; there the balanced step is $\eta_{\mathrm{bal}} = \sqrt{2 \cdot 32/1000} = 0.253$, marked by the dotted line. Panel (b) shows the bound at the chosen step (black) falling like $T^{-1/2}$, against the deterministic bound (blue) falling like $T^{-1}$. As the survey's caption says, these are upper bounds, not predicted training curves.
:::

:::problem The next step
The preliminaries end with one formula that connects most optimizers in the survey: a step that minimizes a linear model plus a penalty measured in some metric $M$. That is chapter 6.
:::
