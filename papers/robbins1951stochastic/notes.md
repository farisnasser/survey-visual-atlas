# A Stochastic Approximation Method

:::minute The paper in one minute
**Problem.** Suppose an experiment responds with a random value $Y(x)$ when we choose a scalar level $x$. Its mean $M(x) = \mathbb{E}[Y(x)]$ is unknown. We seek the unknown level $\theta$ satisfying $M(\theta) = \alpha$. Only noisy observations of $Y(x)$ are available.

**Method.** Choose a positive gain sequence $\{a_n\}$ and observe $y_n$ with conditional mean $M(x_n)$; update $x_{n+1} = x_n + a_n(\alpha - y_n)$.

**Actual result.** Under the original paper's bounded-response, root-sign/monotonicity and gain/identifiability conditions, $b_n = \mathbb{E}[(x_n - \theta)^2] \to 0$; hence $x_n \to \theta$ *in probability*. It is not presented as a general theorem that SGD converges almost surely to global neural-network minima.

**Terminology check.** "Convergence in probability" is weaker than "almost-sure convergence"; the latter must not be attributed to the original root-finding theorem without a separate argument.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Finding a root you can only measure with noise

Bisection or Newton's method needs exact values of $M(x)$. Here each measurement is $Y(x) = M(x) + \text{noise}$. Take $M(x) = \tanh x$ and target $\alpha = 0.5$, so the root is $\theta = \operatorname{artanh} 0.5 \approx 0.549$, and let the noise be uniform on $[-1, 1]$, with standard deviation $1/\sqrt3 \approx 0.577$: larger than most of the signal. Averaging $100$ measurements at one fixed $x$ shrinks the noise to about $0.058$, but spending $100$ measurements on every trial point is wasteful. Robbins and Monro instead move $x$ a little after every single measurement, so the averaging happens along the way.

### 2. Mean-square convergence implies convergence in probability

The paper tracks $b_n = \mathbb{E}[(x_n - \theta)^2]$, the **mean-square error**. If $b_n \to 0$, then for any tolerance $\varepsilon$ Markov's inequality gives $\Pr(|x_n - \theta| > \varepsilon) \le b_n/\varepsilon^2 \to 0$: the iterate is close to $\theta$ with probability tending to one. For example, $b_n = 0.0001$ and $\varepsilon = 0.1$ give a probability at most $0.01$. This is **convergence in probability**. It is weaker than **almost-sure** convergence, which says that the single sequence $x_1, x_2, \dots$ you actually observe converges.

### 3. The gains must shrink, but not too fast

Run the update on the example of idea 1 from $x_1 = -2$ and average $(x_n - \theta)^2$ over $200$ runs. With $a_n = 2/n$, $b_n$ is about $0.0077$ at $n = 100$, $0.00075$ at $n = 1000$ and $0.00003$ at $n = 20{,}000$: the error keeps falling. With a constant gain $a_n = 0.2$, $b_n$ stays near $0.05$: each step injects fresh noise of the same size, so the iterate never settles. With $a_n = 2/n^2$, $b_n$ stays near $0.51$: the gains sum to a finite total, so the iterate stops moving before it reaches $\theta$. The paper's conditions rule out both failures.

:::key Putting the three ideas together
The method finds a root that can only be measured with noise by moving after every measurement (idea 1). The paper proves that the mean-square error $b_n$ goes to zero, which gives convergence in probability (idea 2), provided the gains shrink fast enough to average out the noise and slowly enough to keep moving (idea 3). The notes below rebuild the paper's argument and connect it to SGD.
:::
:::

:::divider Study notes
:::

## 1. Original problem (not a deep-learning optimization problem)

Suppose an experiment responds with a random value $Y(x)$ when we choose a scalar level $x$. Its mean $M(x) = \mathbb{E}[Y(x)]$ is unknown. We seek the unknown level $\theta$ satisfying

$$\boxed{M(\theta) = \alpha,}$$

where $\alpha$ is a specified target response. Only noisy observations of $Y(x)$ are available; the mean function itself cannot be queried exactly. The original paper proposes a sequential method for estimating this root and proves *convergence in probability* under its assumptions.

## 2. The Robbins–Monro update

Choose a positive gain sequence $\{a_n\}$ and observe $y_n$ with conditional mean $M(x_n)$. The original iteration is

$$\boxed{x_{n+1} = x_n + a_n(\alpha - y_n).}$$

The sign is easy to understand. If $y_n$ exceeds the target $\alpha$, decrease $x$; if the observation is too small, increase $x$. Averaged over the observation noise,

$$\mathbb{E}[x_{n+1} - x_n \mid x_n] = a_n(\alpha - M(x_n)).$$

For a nondecreasing response function, this drift pushes the iterates toward $\theta$.

## 3. Rebuild the original mean-square argument

Write $e_n = x_n - \theta$ and $b_n = \mathbb{E}[e_n^2]$. Expanding the square conditional on $x_n$,

$$\begin{aligned} b_{n+1} &= \mathbb{E}[(e_n - a_n(y_n - \alpha))^2] \\ &= b_n - 2a_n\,\mathbb{E}[e_n(M(x_n) - \alpha)] + a_n^2\,\mathbb{E}[(y_n - \alpha)^2]. \end{aligned}$$

Set $d_n = \mathbb{E}[e_n(M(x_n) - \alpha)]$ and $q_n = \mathbb{E}[(y_n - \alpha)^2]$ so

$$\boxed{b_{n+1} - b_n = -2a_n d_n + a_n^2 q_n.}$$

This is the key balance: the $-2a_n d_n$ term draws the estimate toward the root, while $a_n^2 q_n$ measures the accumulated effect of observation noise. Under the paper's monotonicity and bounded-observation assumptions, $d_n \ge 0$ and $q_n$ is bounded.

## 4. Why step sizes must decrease

The paper assumes $\sum_n a_n^2 < \infty$, and introduces additional conditions controlling the root's local identifiability and the cumulative gains (it gives $a_n = 1/n$ as an important example). Later presentations often summarize the classic gain heuristic as

$$\sum_{n=1}^\infty a_n = \infty, \qquad \sum_{n=1}^\infty a_n^2 < \infty.$$

The first condition prevents the total adjustment from dying out too early; the second keeps squared noise increments summable. *The displayed pair is a common later shorthand, not a complete replacement for Robbins–Monro's original detailed consistency hypotheses.*

:::theorem Actual result
Under the original paper's bounded-response, root-sign/monotonicity and gain/identifiability conditions, $b_n = \mathbb{E}[(x_n - \theta)^2] \to 0$; hence $x_n \to \theta$ *in probability*. It is not presented as a general theorem that SGD converges almost surely to global neural-network minima.
:::

## 5. Connect carefully to modern stochastic gradient descent

If the unknown equation is $M(x) = 0$ and $y_n$ is an unbiased estimate of a derivative, the recursion becomes

$$x_{n+1} = x_n - a_n g_n, \qquad \mathbb{E}[g_n \mid x_n] = \nabla f(x_n).$$

This is the structural ancestor of stochastic gradient descent. In many dimensions and non-convex settings, convergence requires much more analysis: a zero gradient can be a saddle, and the mean-square monotone-root argument cannot be applied directly to arbitrary neural losses.

## 6. Limitations and survey relevance

The original theorem is a *scalar monotone-root* problem with bounded observation variables and specific gain assumptions. Neither high-dimensional non-convex stationarity nor strict-saddle escape is proved. Ref. [46] belongs in the **Preliminaries/RQ2** historical lineage of stochastic optimization, not in a table of efficient SOSP-finding algorithms.

:::warn Terminology check
"Convergence in probability" is weaker than "almost-sure convergence"; the latter must not be attributed to the original root-finding theorem without a separate argument.
:::

## 7. Our notes

This is where the basic idea of learning from noisy observations began. Each update corrects the current guess using a noisy response, with step sizes shrinking so noise does not keep shaking the estimate forever. Its original mathematical target was a monotone scalar root, not a neural-network minimum.

> **Original publication:** [https://doi.org/10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586). Public scan: [https://www.columbia.edu/~ww2040/8100F16/RM51.pdf](https://www.columbia.edu/~ww2040/8100F16/RM51.pdf).
