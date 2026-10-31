# SPIDER: Tracking Gradients with Less Noise

:::minute The paper in one minute
**Problem:** Full gradients are expensive on large data, but individual stochastic gradients are noisy.

**Contribution:** SPIDER builds a recursive estimator that tracks the full gradient using differences evaluated on *the same minibatch*.

**Results:** Near-optimal first-order stationarity complexity; an extended SPIDER-SFO$^+$ method also targets approximate second-order stationarity.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Averaging a minibatch divides the variance by its size

A stochastic gradient $\nabla f_i(x)$, with $i$ drawn uniformly, is **unbiased**: its average over $i$ is $\nabla f(x)$. Its spread around $\nabla f(x)$ is measured by the **variance** $\sigma^2 = \mathbb{E}\|\nabla f_i(x) - \nabla f(x)\|^2$. Averaging $b$ independent draws keeps the mean and divides the variance by $b$: with $\sigma^2 = 4$ and $b = 16$, the minibatch variance is $0.25$. Halving the noise's standard deviation costs four times as many samples, which is why SGD needs many gradients for high accuracy.

### 2. A difference on the same sample can have almost no variance

For two random quantities, $\mathrm{Var}(X - Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) - 2\,\mathrm{Cov}(X, Y)$. Fresh samples at two points are independent, so their difference is *noisier* than either. The same sample at two nearby points is strongly correlated, and the noise mostly cancels. For example, take $f_i(x) = \tfrac12 c_i x^2$ with random curvatures $c_i$. Then $\nabla f_i(x_t) - \nabla f_i(x_{t-1}) = c_i\,(x_t - x_{t-1})$, whose variance is $\mathrm{Var}(c_i)\,(x_t - x_{t-1})^2$. With $\mathrm{Var}(c_i) = 1$ and a step of $0.01$, it is $10^{-4}$, however noisy each gradient is on its own.

### 3. Small errors added step after step grow like a random walk

SPIDER adds one such low-variance difference per step to a running estimate. Independent errors add their variances, so after $k$ steps the variance is $k$ times the per-step amount. The paper's Lemma 1 states this: if each step moves at most $\epsilon_1$ and the per-sample gradients are $L$-Lipschitz in mean square, the estimator's mean squared error grows by at most $kL^2\epsilon_1^2/S_2$ for minibatches of size $S_2$. With $L = 1$, $\epsilon_1 = 0.01$, $S_2 = 10$ and $k = 100$, that is $10^{-3}$, a standard deviation of about $0.032$. Before the error grows too large, SPIDER refreshes the estimate with an accurate gradient.

### 4. Counting gradient evaluations, not iterations

The paper's currency is the number of individual gradients $\nabla f_i$ computed. Gradient descent needs $O(\epsilon^{-2})$ iterations, each costing $n$, so $O(n\epsilon^{-2})$ in total; the paper quotes $O(\min(n\epsilon^{-2}, \epsilon^{-4}))$ for GD and SGD. SPIDER's $O(\min(\sqrt n\,\epsilon^{-2}, \epsilon^{-3}))$ replaces $n$ by $\sqrt n$ and $\epsilon^{-4}$ by $\epsilon^{-3}$. At $\epsilon = 10^{-2}$: for $n = 100$, SPIDER needs $10^5$ against GD's $10^6$; for $n = 10^6$, the online count $\epsilon^{-3} = 10^6$ wins, against $10^{10}$ for GD. The two SPIDER terms cross at $n = \epsilon^{-2}$.

:::key Putting the four ideas together
Minibatches reduce noise slowly (idea 1). Differences on the same sample cancel most of it (idea 2), so a running sum of differences tracks the full gradient with error that grows only linearly and can be refreshed (idea 3). Fewer gradient evaluations per accurate direction give the paper's near-optimal counts (idea 4), made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Start with the finite-sum training problem

For $n$ samples, write

$$f(x) = \frac1n\sum_{i=1}^n f_i(x), \qquad \nabla f(x) = \frac1n\sum_{i=1}^n \nabla f_i(x).$$

GD computes every term at every iteration; SGD uses $g_t = \nabla f_{i_t}(x_t)$ as a cheap but noisy estimate. Noise can dominate when the true gradient is small, so merely taking more SGD steps may not be efficient.

## 2. The estimator: subtract two correlated gradients

At periodic *refresh* steps SPIDER obtains a relatively accurate gradient estimate (full gradient in the finite-sum setting, or a sufficiently large batch in the online setting). Between refreshes it updates

$$\boxed{v_t = v_{t-1} + \frac{1}{|S_t|}\sum_{i \in S_t}\big[\nabla f_i(x_t) - \nabla f_i(x_{t-1})\big]}.$$

The crucial detail: use *the same sampled indices* $S_t$ at both $x_t$ and $x_{t-1}$. If nearby points have similar gradients, the difference has low variance even when either gradient alone is noisy.

## 3. Why cancellation helps — a concrete toy calculation

Suppose a sampled gradient has a shared random component, $g_i(x) = h(x) + \xi_i$. A fresh gradient at each point has noise. But with the same $i$,

$$g_i(x_t) - g_i(x_{t-1}) = h(x_t) - h(x_{t-1}) + (\xi_i - \xi_i),$$

and the shared noise cancels exactly in this teaching model. In real learning, it does not cancel perfectly, but smoothness controls the variance of the gradient *difference* as steps remain small.

## 4. How tracking enters optimization

SPIDER-SFO combines the estimator $v_t$ with a normalized descent rule, schematically

$$x_{t+1} = x_t - \eta_t\frac{v_t}{\|v_t\|} \quad \text{when } v_t \ne 0.$$

The normalization helps control the distance between successive iterates, making the estimator's accumulated error manageable. The actual algorithm specifies adaptive step lengths, refresh schedules and termination conditions.

:::key Core rates — read the tolerance carefully.
For a gradient criterion $\mathbb{E}\|\nabla f(\hat x)\| \lesssim \epsilon$ (or corresponding prescribed stationarity output), the paper reports stochastic-gradient evaluation complexity of order

$$\boxed{O\big(\min\{\sqrt n\,\epsilon^{-2},\ \epsilon^{-3}\}\big)}$$

for its finite-sum/online regimes, under their respective smoothness and stochastic-oracle assumptions. With SPIDER-SFO$^+$ and extra saddle-escape machinery, the second-order target $(\epsilon, O(\sqrt\epsilon))$ has complexity roughly

$$\widetilde{O}\Big(\min\{\sqrt n\,\epsilon^{-2} + \epsilon^{-5/2},\ \epsilon^{-3}\}\Big).$$

The first displayed rate is *FOSP*, not automatically a guarantee of saddle escape.
:::

## 5. Assumptions, limitations and survey connection

The improvement relies on suitable smoothness, randomized sampling, controlled estimator error and refresh batches. Gradient evaluations, not optimizer state memory or GPU runtime, are counted. Second-order guarantees require the extended variant and additional conditions; they are not an automatic consequence of low-variance gradients. **Our takeaway.** SPIDER is important because it reduces the cost of knowing which direction is downhill; when joined with escape machinery it also helps reach a curvature-aware target. Variance reduction and saddle escape are connected but different goals.

> **Source:** [https://papers.nips.cc/paper_files/paper/2018/hash/1543843a4723ed2ab08e18053ae6dc5b-Abstract.html](https://papers.nips.cc/paper_files/paper/2018/hash/1543843a4723ed2ab08e18053ae6dc5b-Abstract.html).
