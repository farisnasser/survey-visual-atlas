# Adam: A Method for Stochastic Optimization

:::minute The paper in one minute
**Goal:** Deep networks have noisy, nonstationary, and highly uneven gradients. Kingma and Ba introduce Adam (*adaptive moment estimation*), a first-order optimizer that combines an exponential average of gradients (momentum-like behavior) with an exponential average of squared gradients (coordinate-wise scaling).

**Main contribution:** A practical adaptive optimizer that uses bias-corrected first and second gradient moments to combine directional persistence with coordinate-wise rescaling.

**Accuracy note:** "Adam works well in experiments" and "Adam always converges on neural-network objectives" are different claims. The original paper supports the former, not the latter.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. An exponential moving average forgets at a fixed rate

The recurrence $m_t = \beta m_{t-1} + (1 - \beta)g_t$ unrolls to a weighted sum: the gradient from $k$ steps ago has weight $(1 - \beta)\beta^k$. Weights shrink geometrically, and the average "remembers" roughly the last $1/(1 - \beta)$ steps. With Adam's defaults, $\beta_1 = 0.9$ gives a memory of about $10$ steps for the direction, and $\beta_2 = 0.999$ about $1000$ steps for the squared size. A gradient $10$ steps old still carries $0.9^{10} \approx 0.35$ of the weight of the newest one in $m_t$.

### 2. Averaging cancels part of the noise

If the gradients are a fixed signal plus independent noise of variance $\sigma^2$, the moving average keeps the signal and shrinks the noise. In the long run its noise variance is

$$\sigma^2\,\frac{1 - \beta}{1 + \beta},$$

which for $\beta = 0.9$ is $\sigma^2/19$. This is the "momentum-like behavior" of $m_t$: directions that persist across steps add up, directions that flip sign average out.

### 3. Dividing by the root-mean-square removes the gradient's scale

Adam's step is $\alpha\,\hat m_t/\sqrt{\hat v_t}$ per coordinate. Multiplying every gradient by a constant $c$ multiplies $\hat m_t$ by $c$ and $\hat v_t$ by $c^2$, so the ratio does not change. The paper states this invariance, and also that the step size per coordinate is roughly bounded by $\alpha$. Example: a constant gradient of $0.001$ and a constant gradient of $1000$ both give a step of exactly $\alpha$ after bias correction. With plain SGD the two steps would differ by a factor of $10^6$.

### 4. The ratio $\hat m_t/\sqrt{\hat v_t}$ measures signal against noise

The paper calls $\hat m_t/\sqrt{\hat v_t}$ the **signal-to-noise ratio**. If a coordinate's gradient is always $+1$, the ratio is $1$ and the step is $\alpha$. If it is $+1$ or $-1$ at random, $\hat m_t$ hovers near $0$ while $\hat v_t$ stays near $1$, so the step is small. Near an optimum the gradients become mostly noise, the ratio drops, and the steps shrink by themselves: the paper calls this *automatic annealing*.

### 5. Without bias correction the early steps are too large

Both averages start at $0$, so early on $m_t$ is too small by the factor $1 - \beta_1^t$ and $v_t$ by $1 - \beta_2^t$. Because $\beta_2$ is much closer to $1$, the denominator is the worse of the two. For a constant gradient, the uncorrected step is $\alpha(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$: at $t = 1$ that is $0.1/\sqrt{0.001} \approx 3.16$ times $\alpha$, and it peaks near $6.57\alpha$ at $t = 12$ before settling to $\alpha$. The notes' §3 shows why dividing by $1 - \beta^t$ removes exactly this factor.

:::key Putting the five ideas together
Adam keeps two exponential averages (idea 1): the first averages out noise in the direction (idea 2), the second sets a per-coordinate scale so that steps are about $\alpha$ whatever the gradient's size (idea 3). Their ratio shrinks steps automatically when the signal is weak (idea 4), and bias correction keeps the first steps from being several times too large (idea 5). The notes below derive the algorithm and say what the paper does and does not prove.
:::
:::

:::divider Study notes
:::

## 1. Goal: combine momentum with per-coordinate adaptivity

Deep networks have noisy, nonstationary, and highly uneven gradients. Kingma and Ba introduce Adam (*adaptive moment estimation*), a first-order optimizer that combines an exponential average of gradients (momentum-like behavior) with an exponential average of squared gradients (coordinate-wise scaling). Their empirical contribution is the practicality of this combination; the paper's theoretical regret analysis is framed in online convex optimization.

## 2. Derive the algorithm step by step

At iteration $t$, compute stochastic gradient $g_t$. Maintain two running statistics:

$$\begin{aligned} m_t &= \beta_1 m_{t-1} + (1 - \beta_1)g_t, \\ v_t &= \beta_2 v_{t-1} + (1 - \beta_2)g_t^{\odot 2}. \end{aligned}$$

Here $g_t^{\odot 2}$ squares coordinates separately. With $m_0 = v_0 = 0$, these moving averages are initially biased toward zero; use

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}.$$

Then update each parameter:

$$\boxed{\theta_t = \theta_{t-1} - \alpha\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon_{\text{num}}}} \quad \text{(elementwise)}.$$

The small $\varepsilon_{\text{num}}$ prevents division by zero; it is unrelated to the $\epsilon$ used in convergence-to-stationarity theorems.

## 3. Why bias correction is needed (first-principles example)

Suppose the gradient is a constant $g$ and $m_0 = 0$. Expanding the recurrence,

$$m_1 = (1 - \beta_1)g, \qquad m_2 = (1 - \beta_1)(1 + \beta_1)g,$$

so after $t$ steps,

$$m_t = (1 - \beta_1)\sum_{k=0}^{t-1}\beta_1^k g = (1 - \beta_1^t)g.$$

Therefore $\hat m_t = g$. The same calculation with $g^2$ gives $\hat v_t = g^2$ for a constant gradient. Without bias correction, a zero-start moving average initially underestimates its long-run value.

## 4. What the paper actually establishes

The original algorithm is inexpensive and uses $O(d)$ memory. The authors give a regret analysis for the online convex setting under their stated boundedness and parameter conditions, plus experiments showing competitive training behavior. This is *not* a generic theorem that Adam converges to a global minimum of a modern non-convex neural network; subsequent research identified limitations of the original convergence argument and counterexamples for Adam-type updates under certain settings.

:::key Main contribution
A practical adaptive optimizer that uses bias-corrected first and second gradient moments to combine directional persistence with coordinate-wise rescaling.
:::

## 5. Contrast with AdaGrad and SGD

SGD uses $\theta_t = \theta_{t-1} - \alpha g_t$ (or a velocity update with momentum). AdaGrad divides by a cumulative sum of squared gradients. Adam's exponential averages *forget old gradients*, which can adapt rapidly but also means that convergence arguments do not automatically carry over from AdaGrad. The ratio $\hat m_t/(\sqrt{\hat v_t} + \varepsilon_{\text{num}})$ is a *preconditioned direction*, so stability should account for the optimizer-induced metric, not just the Euclidean Hessian.

## 6. Limits and survey connection

The paper does not establish a finite-time $\epsilon$-SOSP guarantee, explicit negative-curvature search, or strict-saddle escape for ordinary Adam. Its online regret bound has assumptions different from later non-convex results, such as our Ref. [42]. We use it for **RQ2** (modern training dynamics) and **RQ4** (what remains unproved about adaptive large-step training).

:::warn Accuracy note
"Adam works well in experiments" and "Adam always converges on neural-network objectives" are different claims. The original paper supports the former, not the latter.
:::

## 7. Our notes

This paper explains the optimizer used in so much modern deep learning: one moving average remembers the direction of gradients, while another rescales each coordinate by recent gradient magnitudes. Bias correction is important at the start. For our survey, Adam is evidence of how real optimization differs from textbook GD, not a proof of saddle escape. **Original source:** [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980). **Source check:** original author manuscript and ICLR publication metadata.

> **Primary source:** Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*, International Conference on Learning Representations (ICLR). [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980).
