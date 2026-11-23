## 1. What changes when we stop using the full dataset?

This chapter follows Section II-F of the survey. Suppose a dataset has 10,000 examples. Full gradient descent averages 10,000 per-example gradients before every update. **[[sgd|SGD]]** might sample only 100 of them, a **[[minibatch]]** $B_t$, and use (survey Eq. 10)

$$\hat g_t = \frac{1}{|B_t|}\sum_{i\in B_t}\nabla L_i(\theta_t) = g_t + \xi_t.$$

The hat means *estimated*. The decomposition just splits the minibatch gradient into the full gradient $g_t$ and the sampling error $\xi_t$. If $g_t = (4, 2)^\top$ but the minibatch gives $\hat g_t = (4.5, 1.7)^\top$, then $\xi_t = (0.5, -0.3)^\top$.

**What does unbiasedness mean?** The survey assumes that, with appropriate random sampling, the error averages to zero: $\mathbb{E}[\xi_t \mid \theta_t] = 0$. Here $\mathbb{E}$ averages over the possible minibatches while the weights $\theta_t$ are held fixed. If the true gradient of one weight is $4$ and three possible minibatches give $3$, $4$ and $5$, their errors $-1, 0, +1$ average to zero. That is unbiasedness: no systematic tilt, only random wobble. It depends on how the data are sampled; it is not automatic for any data pipeline.

:::idea In plain English
The minibatch arrow points a little differently from the true gradient arrow at every step. Over many samples the deviations cancel on average, but each individual update is still noisy.
:::

@@viz cloud@@

## 2. Why does a larger minibatch reduce the variance?

Take five per-example gradients, $2, 4, 6, 8, 10$. Their average, the full gradient, is $g = 6$. A one-example estimate can be anything from $2$ to $10$. Averages of three, such as $(2 + 8 + 10)/3 \approx 6.67$ or $(4 + 6 + 8)/3 = 6$, typically land closer.

**Variance** measures that spread: $\mathrm{Var}(\hat g) = \mathbb{E}[(\hat g - g)^2] = \mathbb{E}[\xi^2]$. Squaring matters, so that errors of $-2$ and $+2$ both count instead of cancelling.

**Deriving $\sigma^2/B$, one step at a time.** Let $X_1, \dots, X_B$ be independent sample gradients with mean $g$ and variance $\sigma^2$, and $\hat g = (X_1 + \dots + X_B)/B$. Then

$$\mathrm{Var}(\hat g) = \frac{1}{B^2}\,\mathrm{Var}(X_1 + \dots + X_B) = \frac{1}{B^2}\big(\mathrm{Var}(X_1) + \dots + \mathrm{Var}(X_B)\big) = \frac{B\sigma^2}{B^2} = \frac{\sigma^2}{B}.$$

The key step is that variances of independent samples add. For vectors, the survey uses the same bound on the expected squared norm, $\mathbb{E}\|\xi_t\|^2 \le \sigma^2/|B_t|$, where $\sigma^2$ bounds the variance of a single training sample's gradient.

With $\sigma^2 = 100$, a batch of $10$ gives variance $10$ and a batch of $100$ gives variance $1$; the typical noise size (the standard deviation) falls from $\sqrt{10} \approx 3.16$ to $1$. Quadrupling the batch divides the variance by four but the noise size only by two.

:::viz batchvar
:::

:::read How to read this visual
**What is drawn.** The example above: five per-example gradients $2, 4, 6, 8, 10$ (variance $\sigma^2 = 8$ around their mean $6$). The histogram shows *every* possible minibatch estimate of size $B$, drawing with replacement (all $5^B$ equally likely batches), so it is the exact distribution, not a sample. The green line is the true gradient $6$.

**What moves.** The batch size $B$, from $1$ to $5$.

**What to notice.** Every histogram is centred on $6$: the estimate is unbiased at every batch size. The spread shrinks as $B$ grows, and the readout confirms that the variance equals $\sigma^2/B = 8/B$ exactly, while the standard deviation shrinks only like $1/\sqrt{B}$.
:::

Smaller batches process less data per update but give noisier steps; larger batches give steadier steps at more cost per update. Which one trains faster overall also depends on how well the hardware parallelizes the work.

## 3. How noise enters the descent lemma

Return to chapter 3's descent lemma, $L(\theta + \Delta) \le L(\theta) + g^\top\Delta + \tfrac{\beta}{2}\|\Delta\|^2$, and take SGD's step $\Delta = -\eta\hat g_t$. Two gradients now appear: the true $g_t$ in the bound, and the estimated $\hat g_t$ that the algorithm actually follows.

:::fold Derive the survey's Eq. (11), one line at a time | from the descent lemma to the noise penalty
Substitute the step:

$$L_{t+1} \le L_t - \eta g_t^\top\hat g_t + \frac{\beta\eta^2}{2}\|\hat g_t\|^2.$$

Now write $\hat g_t = g_t + \xi_t$ and expand both terms: $g_t^\top(g_t + \xi_t) = \|g_t\|^2 + g_t^\top\xi_t$ and $\|g_t + \xi_t\|^2 = \|g_t\|^2 + 2g_t^\top\xi_t + \|\xi_t\|^2$. Put them back:

$$L_{t+1} \le L_t - \eta\|g_t\|^2 - \eta g_t^\top\xi_t + \frac{\beta\eta^2}{2}\big(\|g_t\|^2 + 2g_t^\top\xi_t + \|\xi_t\|^2\big).$$

Take the expectation over the minibatch, with $\theta_t$ fixed. Because $\mathbb{E}_t[\xi_t] = 0$ and $g_t$ is fixed, $\mathbb{E}_t[g_t^\top\xi_t] = g_t^\top\mathbb{E}_t[\xi_t] = 0$, so every cross term vanishes:

$$\mathbb{E}_t[L_{t+1}] \le L_t - \eta\|g_t\|^2 + \frac{\beta\eta^2}{2}\big(\|g_t\|^2 + \mathbb{E}_t\|\xi_t\|^2\big).$$

Finally use $\mathbb{E}_t\|\xi_t\|^2 \le \sigma^2/|B_t|$ and collect the gradient terms.
:::

The result is the survey's Eq. (11):

$$\mathbb{E}_t[L_{t+1}] \le L_t \underbrace{-\,\eta\Big(1 - \frac{\beta\eta}{2}\Big)\|g_t\|^2}_{\text{descent term}} + \underbrace{\frac{\beta\eta^2\sigma^2}{2|B_t|}}_{\text{noise penalty}}, \qquad L_t = L(\theta_t).$$

**What it means.** Imagine walking downhill in fog. The full gradient would measure the slope perfectly; the minibatch gives a slightly noisy arrow. The first term is the expected progress from the true slope; the second weakens the guarantee because noisy steps can be inaccurate. Near a stationary point $\|g_t\|$ becomes small, so the descent term fades, but the noise penalty does not. That is why constant-step SGD keeps fluctuating near a minimum instead of settling on it. The penalty is an upper bound, not a claim that the loss actually rises by that amount.

@@viz floor@@

## 4. From one step to $T$ steps

Summing Eq. (11) over many steps makes the trade-off explicit. Let $\Delta_0 = L(\theta_0) - L_*$ be the initial gap to the lowest possible loss, take a constant step $\eta \le 1/\beta$ (so $1 - \beta\eta/2 \ge \tfrac12$) and batch size $B$. Taking full expectations, adding the inequalities for $t = 0, \dots, T-1$ (the losses telescope), and using $L_T \ge L_*$ gives

$$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\|g_t\|^2 \le \frac{2\Delta_0}{\eta T} + \frac{\beta\eta\sigma^2}{B}.$$

A bigger step shrinks the first term and grows the second. Balancing them gives

$$\eta_{\mathrm{bal}} = \sqrt{\frac{2\Delta_0 B}{\beta\sigma^2 T}}, \qquad \eta = \min\Big\{\frac1\beta, \eta_{\mathrm{bal}}\Big\},$$

and with this step both terms are of order $T^{-1/2}$. Through this worst-case bound, reaching an average squared gradient of $\epsilon^2$ takes $T = O(\epsilon^{-4})$ iterations; without noise, chapter 7 shows $O(\epsilon^{-2})$. As the survey stresses in Section VI-D, these are upper bounds under different assumptions, not predictions of how fast SGD runs.

@@viz bound@@

:::key What you should remember
With independent unbiased sampling, gradient variance scales like $1/B$. Eq. (11) has the same descent term as gradient descent plus a noise penalty, which is why SGD convergence needs assumptions about gradient variance on top of smoothness.
:::

:::problem The next step
Before the individual optimizers, Section II-G introduces one formula that describes most of them: a step $-\eta M^{-1}g$ for some matrix $M$. Chapter 6 asks what $M$ means.
:::
