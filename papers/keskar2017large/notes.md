# On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima

:::minute The paper in one minute
**Problem.** Why can large-batch SGD yield worse test accuracy than small-batch SGD even when both achieve low training error?

**Method.** The authors compare small- and large-batch training over several networks and tasks; they measure local sharpness of reached solutions and test strategies to reduce the gap.

**Finding.** In their experiments, large-batch methods tend to reach sharper regions and generalize worse; small-batch methods find flatter regions. This is an empirical pattern, not a theorem that batch size uniquely determines generalization.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Large batches are attractive because they cut the number of steps

One **epoch** is one pass over the $n$ training examples. With minibatch size $b$ it takes about $n/b$ steps. On a training set of $n = 50{,}000$ images, $b = 256$ needs $196$ steps per epoch; a batch of $10\%$ of the data, $b = 5{,}000$, needs $10$. Each large step can be split across many processors, so large batches promise faster wall-clock training. The paper uses exactly these two regimes: $256$ examples for small-batch (SB) and $10\%$ of the training set for large-batch (LB) runs.

### 2. A minibatch gradient is an average, so its noise shrinks like $1/\sqrt{b}$

Each example $i$ has its own gradient $\nabla \ell_i(\theta)$. The minibatch gradient averages $b$ of them. Averaging $b$ independent draws divides the variance by $b$, so the standard deviation falls like $1/\sqrt{b}$. If the per-example gradients along some direction have standard deviation $\sigma = 1$, the minibatch gradient has standard deviation $1$ for $b = 1$, $0.25$ for $b = 16$ and $0.0625$ for $b = 256$. A large batch gives an accurate gradient; a small batch gives a noisy one.

### 3. Equal training loss does not mean equal test loss

The training loss is computed on the sample we have; the test loss on fresh data. The two functions are close but not identical. The paper's Figure 1 sketches the consequence: if the test loss is the training loss shifted a little, a sharp minimum pays more for the shift. For a concrete case, compare $L_{\text{flat}}(\theta) = \theta^2$ and $L_{\text{sharp}}(\theta) = 100\theta^2$, both zero at $\theta = 0$. A shift of $0.1$ costs $0.01$ in the first and $1$ in the second. The difference in test performance between two solutions with the same training loss is the **generalization gap** the paper studies.

### 4. Noise can knock an iterate out of a narrow basin

A basin that tolerates a loss increase $\delta$ has half-width about $\sqrt{2\delta/\lambda}$ along a direction of curvature $\lambda$. With $\delta = 0.01$: $0.014$ when $\lambda = 100$, $0.14$ when $\lambda = 1$. One noisy step moves $\theta$ by about $\eta\sigma/\sqrt{b}$. With $\eta = 0.1$ and $\sigma = 1$, that is $0.1$ for $b = 1$: larger than the sharp basin, smaller than the flat one. For $b = 256$ it is $0.006$: small enough to stay in either. This is the intuition behind the paper's explanation; the paper supports it with experiments, not a proof.

:::key Putting the four ideas together
Large batches are fast (idea 1) but remove most gradient noise (idea 2). The paper observes that the noise-free runs settle in sharper minima, which pay more when the test data differ slightly from the training data (idea 3), and proposes that small-batch noise is what keeps iterates out of narrow basins (idea 4). The measurements and their limits are made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Batch gradients and their noise

With $n$ samples and minibatch $B_t$ of size $b$, SGD uses

$$g_t = \frac{1}{b}\sum_{i \in B_t} \nabla \ell_i(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta g_t.$$

If samples are independent with per-example gradient covariance $\Sigma(\theta)$, the minibatch gradient covariance is approximately

$$\operatorname{Cov}(g_t \mid \theta_t) \approx \frac{\Sigma(\theta_t)}{b}.$$

Thus larger $b$ lowers gradient noise. The exact finite-population variance also has a correction for sampling without replacement. Reduced noise is a plausible mechanism affecting which basin training explores, *not proof* that more noise always means a flatter or better solution.

## 2. Sharpness and the local Hessian

Around a local minimum $\theta^*$, Taylor's theorem gives

$$L(\theta^* + u) - L(\theta^*) \approx \tfrac12 u^\top H u.$$

For $\|u\|_2 \le r$, the maximum quadratic increase is about

$$\max_{\|u\| \le r} \big(L(\theta^* + u) - L(\theta^*)\big) \approx \frac12 r^2 \lambda_{\max}(H).$$

This is a helpful *Hessian interpretation* of sharpness. Keskar et al.'s operational sharpness measure instead uses the maximum observed objective increase over a local neighborhood (with coordinatewise scaling), not simply $\lambda_{\max}(H)$. Keep this distinction explicit when discussing their plots.

:::key Teaching example
$L_1(x, y) = x^2 + 100y^2$ and $L_2(x, y) = x^2 + y^2$ both have zero training loss at the origin. But $L_1$ grows far faster for a small $y$ displacement. Local geometry can differ despite matching training objectives; it alone does not determine test loss.
:::

## 3. Empirical results and proposed explanations

The paper reports a small-batch/large-batch generalization gap across the tested tasks and relates it to the sharpness of the solutions found. It argues that minibatch noise may help avoid sharp minimizers. Its authors also explore practical measures to improve large-batch behavior, rather than claiming large-batch training is inherently doomed.

## 4. Limits, follow-up debate and survey use

:::warn Do not universalize
Batch size interacts with learning-rate schedule, training duration, normalization and model scale. Raw sharpness depends on how weights are parameterized; reparameterization counterexamples are an explicit counterpoint in our survey [27]. The results support a correlation in the studied settings, not a universal sharpness $\Rightarrow$ generalization law.
:::

**Survey connection.** RQ2: gradient-noise scale and basin selection; RQ4: empirical/theoretical mismatch. **Our Notes.** This paper popularized the idea that stochastic noise changes the minima we reach, not just the speed of training. Its claim is important, but we need to read it alongside the reparameterization critique before drawing conclusions about real modern training.

> **Source:** [https://openreview.net/pdf?id=H1oyRlYgg](https://openreview.net/pdf?id=H1oyRlYgg)
