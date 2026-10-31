# Schedule-Free SGD Is Also Effective for Nonconvex Optimization: General framework for online-to-nonconvex conversion

:::minute The paper in one minute
**Question:** Does the schedule-free SGD method have a rigorous guarantee beyond convex optimization?

**Model/Method:** Transform online learning regret guarantees into a nonsmooth, nonconvex stationarity guarantee; recognize a particular transformation as schedule-free SGD.

**Finding:** With carefully selected and partly randomized interpolation parameters, schedule-free SGD reaches a $(\lambda, \varepsilon)$-stationary point at the optimal oracle-complexity order $O(\sqrt{\lambda}\,\varepsilon^{-7/2})$ (suppressing other problem constants).

**Why we care:** This fills a theoretical gap left by schedule-free optimization's earlier convex analysis, but does *not* establish saddle escape or global optimality.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. On a nonsmooth function the gradient never gets small

Take $F(x) = |x|$. Its minimum is at $0$, yet at every other point the gradient is $+1$ or $-1$. An optimizer that bounces around the minimum at distance $0.01$ still sees gradients of size $1$, so the usual target $\|\nabla F(x)\| \le \varepsilon$ is never met by the iterates. The fix is to look at gradients at *several nearby points* and average them. Put probability $\tfrac12$ on $w = -0.01$ (gradient $-1$) and $\tfrac12$ on $w = 0.03$ (gradient $+1$). Their average location is $x = 0.01$, their average gradient is $0$, and their spread is $\mathbb{E}\|w - x\|^2 = 0.02^2 = 0.0004$. The paper's measure (Definition 1, in the notes) adds the two: $0 + \lambda \times 0.0004$. A small value means "near $x$ there is a cheap way to average the gradients to almost zero", which is the right notion of being close to a stationary point when the function has kinks.

### 2. Regret: what an online learner can promise

In **online learning**, a learner picks $\delta_t$, then is shown a vector $g_t$ and pays $\langle g_t, \delta_t\rangle$. Its **regret** against a fixed comparator $u$ is $\sum_{t=1}^T \langle g_t, \delta_t - u\rangle$. Online gradient descent, $\delta_{t+1} = \delta_t - \gamma g_t$, with $\|g_t\| \le G$, comparators within distance $D$, and $\gamma = D/(G\sqrt{T})$, guarantees regret at most $DG\sqrt{T}$. With $D = G = 1$ and $T = 10\,000$ that is $100$ in total, or $0.01$ per round. The guarantee holds for *any* sequence $g_t$, even one chosen by an adversary. That generality is what lets the paper feed it gradients of a nonconvex function and still get a bound.

### 3. Reading the complexity $\sqrt{\lambda}\,\varepsilon^{-7/2}$

The result counts stochastic-gradient calls $T$ needed to reach $\|\nabla F(x)\|^{[\lambda]} \le \varepsilon$. Halving $\varepsilon$ multiplies $\varepsilon^{-7/2}$ by $2^{3.5} \approx 11.3$. For smoother functions the measure turns into the familiar one (the paper's Proposition 1): if $F$ is $L$-smooth, choosing $\lambda = L^2/\varepsilon$ makes a $(\lambda, \varepsilon)$-stationary point satisfy $\|\nabla F(x)\| \le 2\varepsilon$, and then

$$\sqrt{\lambda}\,\varepsilon^{-7/2} = \sqrt{L^2/\varepsilon}\;\varepsilon^{-7/2} = L\,\varepsilon^{-4},$$

the optimal rate for smooth stochastic nonconvex problems. If $F$ has an $H$-Lipschitz Hessian, $\lambda = H/2$ suffices and the count is of order $\varepsilon^{-3.5}$.

:::key Putting the three ideas together
For nonsmooth losses the paper measures stationarity by averaging gradients over a small cloud of nearby points (idea 1). It obtains such points from an online learner whose regret bound holds for any gradient sequence (idea 2), and the resulting $\sqrt{\lambda}\,\varepsilon^{-7/2}$ count is optimal and reduces to the known smooth rates (idea 3). One particular conversion turns out to be schedule-free SGD, as made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. First, which stationarity notion is being guaranteed?

In smooth problems we often ask for $\|\nabla F(x)\| \le \varepsilon$. For a nonsmooth function, a classical gradient may be unavailable or uninformative. The paper studies a more flexible *distributional* notion: compare gradients at nearby points whose average location is $x$, while penalizing how widely those points spread.

$$\|\nabla F(x)\|^{[\lambda]} = \inf_{\mathbb{E}_p[w] = x} \Big\{ \big\|\mathbb{E}_p[\nabla F(w)]\big\| + \lambda\, \mathbb{E}_p\|w - x\|^2 \Big\}.$$

A point is $(\lambda, \varepsilon)$-stationary when the expression above is at most $\varepsilon$ (Definition 1). *Do not replace this definition by ordinary small-gradient stationarity without extra smoothness conditions.*

## 2. The three schedules hidden inside schedule-free SGD

The algorithm maintains three parameter sequences, not three separate models:

$$\begin{aligned} x_t &= (1 - c_t)\, x_{t-1} + c_t z_t && \text{(average),} \\ y_t &= (1 - \kappa_t)\, z_t + \kappa_t x_t && \text{(gradient point),} \\ g_t \approx \nabla F(y_t), &\qquad z_{t+1} = z_t - \gamma g_t && \text{(base step).} \end{aligned}$$

The paper's novelty is *not* inventing these sequences (Defazio et al. did so), but proving that a particular choice can inherit optimal nonconvex guarantees from an online-learning method.

## 3. Worked example: why $x$, $y$, and $z$ differ

Use the pedagogical one-dimensional loss $F(w) = \tfrac12 (w - 2)^2$; let $x_0 = z_1 = 0$, $c_t = 1/2$, $\kappa_t = 3/4$, and $\gamma = 1/2$. First,

$$x_1 = 0, \quad y_1 = 0, \quad g_1 = -2, \quad z_2 = 1.$$

Then $x_2 = \tfrac12(0) + \tfrac12(1) = 0.5$ and

$$y_2 = \tfrac14(1) + \tfrac34(0.5) = 0.625, \quad g_2 = -1.375, \quad z_3 = 1.6875.$$

The gradient is evaluated at 0.625, neither at the fast iterate 1 nor the average 0.5. This toy calculation teaches the update, not the paper's nonsmooth theorem.

## 4. How an online-learning guarantee becomes nonconvex progress

Online optimization controls *regret*, roughly $\sum_t \langle g_t, \delta_t - u\rangle$ compared with a comparator $u$. Nonconvex optimization instead needs a statement about small local gradients. The authors bridge these goals through a randomized interpolation: evaluate gradients on points between two evolving trajectories and relate the regret bound to the average change in $F$.

### Why the interpolation matters

For a differentiable objective, the fundamental theorem of calculus gives

$$F(b) - F(a) = \int_0^1 \big\langle \nabla F(a + s(b - a)),\, b - a \big\rangle\, ds.$$

Sampling $s$ uniformly makes the *expected* gradient along the line segment exactly relevant to the difference $F(b) - F(a)$. That identity explains the bridge to a nonconvex descent argument even without global convexity.

:::key Conceptual proof path
**Regret control** for an auxiliary online learner $\longrightarrow$ **expected objective-change control** for randomized interpolation $\longrightarrow$ **distributional stationarity** of an output point. This is not a Hessian or negative-curvature argument.
:::

## 5. What the main guarantee really says

Assumption 1 imposes lower boundedness, a suitable line-integral identity, and a globally bounded gradient $\|\nabla F\| \le G$. Assumption 2 requires independent unbiased stochastic gradients with variance at most $\sigma^2$. The authors establish an optimal-order complexity of

$$\boxed{T = O\big(\sqrt{\lambda}\,\varepsilon^{-7/2}\big)}$$

for their stationarity measure, with additional dependence on problem constants. The result is not an $\varepsilon$-second-order stationary-point guarantee.

### Why this is genuinely a schedule-free SGD result

Proposition 2 reconstructs a base trajectory satisfying

$$z_{t+1} - z_t = -\frac{\eta}{1 - \zeta}\, g_t.$$

This exposes a particular online-to-nonconvex conversion as the same three-sequence update used by schedule-free SGD. But the proof needs parameter choices *specific* to the nonconvex regime, including randomized interpolation near the running average; an arbitrary practical Schedule-Free AdamW configuration is not covered.

## 6. Contributions and limitations

**Contributions.** (i) A general conversion from online learning to nonsmooth nonconvex optimization; (ii) recovery of established conversion recipes plus new ones; (iii) an equivalence between one conversion and schedule-free SGD; (iv) an optimal-order stationarity bound under an explicit oracle model. **Limitations.**

- The guarantee targets a distributional first-order measure $\|\nabla F\|^{[\lambda]}$, not global minimization, generalization, or escaping strict saddles.
- The analysis requires globally bounded gradients and an unbiased, bounded-variance stochastic-gradient oracle, and uses parameter choices that may differ from routine implementations.
- No universal convergence theorem is established for Schedule-Free AdamW or arbitrary learning-rate schedules in neural-network training.

## 7. Survey connections

**Defazio et al. (2024):** introduced the three-sequence practical algorithm and analyzed convex optimization; the present work supplies a nonsmooth nonconvex justification. **Daneshmand/Jin:** their targets concern negative curvature or approximate second-order stationarity, whereas this work proves a different, first-order-type criterion. **Lookahead:** both use multiple weight sequences but have different mathematical goals.

## 8. Quick reference

| | |
|---|---|
| **Stationarity** | $\lVert\nabla F(x)\rVert^{[\lambda]} \le \varepsilon$ (Definition 1). |
| **Base iterate** | $z_{t+1} = z_t - \gamma g_t$. |
| **Average iterate** | $x_t = (1 - c_t) x_{t-1} + c_t z_t$. |
| **Gradient iterate** | $y_t = (1 - \kappa_t) z_t + \kappa_t x_t$. |
| **Assumptions** | $\lVert\nabla F\rVert \le G$; unbiased oracle, variance $\le \sigma^2$. |
| **Complexity** | Optimal-order $O(\sqrt{\lambda}\,\varepsilon^{-7/2})$, plus problem constants. |

### Papers Master note

Ahn et al. establish a general online-to-nonconvex conversion and identify a special case as schedule-free SGD, with optimal-order convergence to $(\lambda, \varepsilon)$-stationarity under bounded-gradient/oracle assumptions. This is not a saddle-escape or global-minimum guarantee.

> **Original:** [https://proceedings.mlr.press/v267/ahn25a.html](https://proceedings.mlr.press/v267/ahn25a.html). The pedagogical scalar example is not from the paper.
