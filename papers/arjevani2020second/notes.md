# What Can Second-Order Information Buy?

:::minute The paper in one minute
**Problem:** How many stochastic oracle calls are *unavoidably* necessary for finding approximate stationary points? Does access to Hessian information fundamentally help, and can even higher derivatives do better?

**Method:** Matching algorithms and worst-case lower-bound constructions in stochastic optimization.

**Result:** An $O(\epsilon^{-3})$ optimal stochastic gradient/Hessian-vector rate for first-order stationarity; nearly matching bounds also for second-order targets.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Noise costs a power of $\epsilon$

With exact gradients, gradient descent reaches $\|\nabla F\| \le \epsilon$ in $O(\epsilon^{-2})$ steps on a smooth function. With noisy gradients of variance $\sigma^2$, telling a gradient of size $\epsilon$ apart from the noise takes about $\sigma^2/\epsilon^2$ samples, so the same $\epsilon^{-2}$ steps cost about $\epsilon^{-4}$ samples. This is the rate of plain SGD, and the paper recalls that $\epsilon^{-4}$ is optimal when only stochastic gradients are available. The counting is a heuristic; the rates are the paper's.

### 2. The elbow: one more derivative helps, more do not

The paper's Fig. 1 summarises the optimal counts. With stochastic oracles of order $p$, the count is $\epsilon^{-4}$ for $p = 1$ and drops to $\epsilon^{-3}$ for $p = 2$, with no further gain for $p > 2$. With noiseless oracles it starts at $\epsilon^{-2}$ for $p = 1$ and approaches $\epsilon^{-1}$ as $p$ grows. At $\epsilon = 10^{-2}$ the stochastic counts are $10^8$ for gradients and $10^6$ with Hessian-vector products: a factor of $100$ from one extra derivative, and nothing from the next ones.

### 3. A Hessian-vector product is a derivative of the gradient

The **Hessian-vector product** (HVP) $\nabla^2 F(x)\,v$ is the rate at which the gradient changes when $x$ moves along $v$. For $F(x, y) = x^2 y$, the gradient is $(2xy, x^2)$ and the Hessian is $\begin{bmatrix} 2y & 2x \\ 2x & 0 \end{bmatrix}$. At $(1, 2)$ with $v = (1, 0)$, the HVP is $(4, 2)$. The finite difference $\big(\nabla F(1.001, 2) - \nabla F(1, 2)\big)/0.001 = (4, 2.001)$ agrees up to the step size. Automatic differentiation computes an HVP without forming the Hessian, at a cost the paper describes as roughly that of a stochastic gradient (citing Pearlmutter, 1994).

### 4. Gradient differences are integrals of HVPs

By the fundamental theorem of calculus along the segment from $x'$ to $x$,

$$\nabla F(x) - \nabla F(x') = \int_0^1 \nabla^2 F\big(x' + t(x - x')\big)(x - x')\,dt .$$

A sum over $K$ points approximates the integral. For $F(x) = x^3/3$ (gradient $x^2$, Hessian $2x$) between $x' = 1$ and $x = 1.2$, the exact difference is $0.44$; one point gives $0.40$ and four equally spaced left points give $0.43$. The paper's estimator uses this idea with *stochastic* Hessians, so it never needs two queries with the same random seed, which is what earlier $\epsilon^{-3}$ methods required.

:::key Putting the four ideas together
Noise makes plain SGD pay $\epsilon^{-4}$ (idea 1). HVPs are cheap (idea 3) and, integrated along each step, track how the gradient changes (idea 4), which brings the cost to $\epsilon^{-3}$. The lower bound shows that no stochastic method of any order $p \ge 2$ beats that exponent (idea 2), made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Why study oracle complexity?

In optimization, an *oracle* answers requests about a function. A stochastic gradient oracle returns $g(x; \xi)$; a stochastic Hessian-vector oracle returns a noisy version of $\nabla^2 F(x)v$ without forming the full Hessian matrix. Counting calls abstracts away software and hardware differences and asks what information an algorithm *must* acquire.

$$\text{Goal (FOSP):} \qquad \|\nabla F(\hat x)\| \le \epsilon.$$

This criterion is weaker than an SOSP, because negative curvature can remain.

## 2. From a noisy gradient to the value of curvature

For twice differentiable $F$, Taylor's theorem says

$$\nabla F(x + \delta v) = \nabla F(x) + \delta\nabla^2 F(x)v + O(\delta^2)$$

when the Hessian is sufficiently regular. An HVP reveals how gradients change in direction $v$. Repeated HVPs can detect a direction of negative curvature, but in this paper they also help produce *more accurate first-order progress* using stochastic second-order structure.

## 3. The principal matching result

The authors construct a stochastic second-order method using stochastic gradients and Hessian-vector products that finds an $\epsilon$-FOSP with

$$\boxed{O(\epsilon^{-3}) \text{ stochastic derivative-oracle queries}}$$

up to fixed smoothness/noise/problem constants in the stated oracle model. They prove a matching lower bound for that model. Remarkably, access to derivatives of any fixed order $p \ge 2$ does not improve the worst-case exponent below 3, even with suitable Lipschitz regularity of the first $p$ derivatives.

:::key Meaning of a lower bound.
An *upper bound* constructs an algorithm needing at most a stated number of calls; a *lower bound* constructs hard functions on which every permitted algorithm needs at least that order. When upper and lower exponents match, the rate is optimal *for that oracle/assumption class*. This is not a statement about every training dataset.
:::

## 4. The second-order target is a separate result

They additionally consider

$$\|\nabla F(\hat x)\| \le \epsilon, \qquad \lambda_{\min}(\nabla^2 F(\hat x)) \ge -\gamma.$$

The paper establishes nearly matching upper/lower bounds for this $(\epsilon, \gamma)$-SOSP task. **Crucial distinction:** The headline $O(\epsilon^{-3})$ theorem is about *first-order stationarity*. It would be incorrect to quote that rate without qualification as a universal SOSP guarantee, or as a rate for finding a global minimum.

## 5. Assumptions and boundaries

These are general smooth stochastic objectives with specified stochastic derivative access and regularity/noise controls. Their lower-bound constructions establish worst-case hardness for the allowed algorithm family, not predictions for actual finite neural networks trained on modern accelerators. Hessian-vector evaluations may have different computational costs than ordinary gradient evaluations; memory, parallelism and tuning are not represented by a single oracle count.

:::warn Survey caution.
Never compare $O(\epsilon^{-3})$ from this paper with $\widetilde{O}(\epsilon^{-7/4})$ for deterministic accelerated escape without stating the different *stochastic versus deterministic oracle* and *FOSP versus SOSP* targets.
:::

**Our takeaway.** Curvature information provably improves what can be achieved with noisy optimization, but there is a fundamental limit: arbitrarily higher derivatives do not automatically yield arbitrarily faster convergence.

> **Source:** [https://proceedings.mlr.press/v125/arjevani20a.html](https://proceedings.mlr.press/v125/arjevani20a.html).
