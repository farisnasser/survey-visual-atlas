# Introductory Lectures on Convex Optimization

:::minute The paper in one minute
**Role.** The book develops the mathematical baseline for first-order optimization and oracle complexity, including nonlinear, smooth convex, nonsmooth convex and structural optimization. Our survey cites it as *background*, not as a paper that proves saddle escape in deep neural networks.

**What this reference contributes to the survey.** A baseline framework of smoothness, first-order methods, mathematical rates and oracle-based comparisons. It supports the notation and the elementary $O(\epsilon^{-2})$ first-order stationarity calculation, without asserting second-order stationarity or deep-learning generalization.

**Key lesson.** Which parts survive without convexity: loss decrease and small-gradient guarantees can survive, while "small gradient means global minimum" does not.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the derivations stand on, built from scratch.
### 1. What the Lipschitz constant $L$ measures

$\nabla f$ is **$L$-Lipschitz** when $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$: the slope cannot change faster than $L$ per unit of distance. For a twice-differentiable $f$ this is the same as every Hessian eigenvalue lying in $[-L, L]$. The function $f(x, y) = x^2/4 - \cos(2y)/4$ is not convex (its curvature in $y$ is $\cos 2y$, negative for some $y$), yet its Hessian $\mathrm{diag}(1/2, \cos 2y)$ never has an eigenvalue above $1$ in size, so $L = 1$. Smoothness is about how fast the slope changes, not about the shape being a bowl.

### 2. Sums of decreases telescope

If every step lowers the loss, the decreases add up to the total drop: with losses $10, 7, 5, 4.5$ the decreases $3 + 2 + 0.5$ sum to $10 - 4.5 = 5.5$. Every intermediate value cancels. So if the loss can never go below $f_{\inf}$, the sum of all per-step decreases is at most $\Delta = f(x_0) - f_{\inf}$, however many steps are taken. The notes' Section 3 uses exactly this with per-step decrease $\|\nabla f(x_t)\|^2/(2L)$.

### 3. If the average is small, the best is smaller

At least one of $T$ numbers is no larger than their average. That turns a bound on a sum of squared gradients into a bound on the smallest one, $\min_t \|\nabla f(x_t)\|^2 \le 2L\Delta/T$. Counting is then arithmetic: with $L = 1$, $\Delta = 10$ and target $\epsilon = 0.01$, $T \ge 2L\Delta/\epsilon^2 = 200{,}000$ gradient evaluations suffice. On the function of idea 1 started at $(3, 1)$, where $\Delta \approx 2.60$, twenty steps with $\eta = 1/L$ guarantee $\min_t\|\nabla f\|^2 \le 0.26$; the run actually reaches about $10^{-11}$. The bound is a worst case over all $L$-smooth functions, not a prediction for one.

:::key Putting the three ideas together
A Lipschitz gradient, convex or not (idea 1), guarantees that each gradient step lowers the loss by a fixed fraction of the squared gradient. Those decreases telescope to at most $\Delta$ (idea 2), and the smallest gradient is no larger than the average (idea 3). That is the whole $O(\epsilon^{-2})$ stationarity rate, derived step by step in the notes below together with what it does not say.
:::
:::

:::divider Study notes
:::

## 1. Why this book appears in our non-convex survey

The book develops the mathematical baseline for first-order optimization and oracle complexity, including nonlinear, smooth convex, nonsmooth convex and structural optimization. Our survey cites it as *background*, not as a paper that proves saddle escape in deep neural networks. We need its definitions and the descent calculations that remain valid for smooth *non-convex* functions, while keeping convex-only conclusions separate.

## 2. From first principles: smoothness and a descent step

Suppose $f$ has $L$-Lipschitz gradient:

$$\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|.$$

The smoothness/descent lemma gives for any increment $s$

$$f(x + s) \le f(x) + \langle\nabla f(x), s\rangle + \frac{L}{2}\|s\|^2.$$

Taking the gradient step $s = -\eta\nabla f(x)$,

$$f(x - \eta\nabla f(x)) \le f(x) - \eta\left(1 - \frac{L\eta}{2}\right)\|\nabla f(x)\|^2.$$

For $\eta = 1/L$, this simplifies to

$$f(x_{t+1}) \le f(x_t) - \frac{1}{2L}\|\nabla f(x_t)\|^2.$$

This calculation *does not require convexity*: it only needs global $L$-smoothness.

## 3. Derive the basic non-convex stationarity rate

If $f$ is bounded below by $f_{\inf}$, sum the previous inequality over $T$ steps:

$$\frac{1}{2L}\sum_{t=0}^{T-1}\|\nabla f(x_t)\|^2 \le f(x_0) - f(x_T) \le \Delta, \qquad \Delta = f(x_0) - f_{\inf}.$$

Consequently,

$$\boxed{\min_{0 \le t < T}\|\nabla f(x_t)\|^2 \le \frac{2L\Delta}{T}.}$$

To obtain a gradient norm no larger than $\epsilon$, it suffices to take $T \ge 2L\Delta/\epsilon^2$. This is a *teaching derivation* of the standard first-order bound; it should not be mistaken for a deep-network global-optimality theorem by the book.

## 4. Convexity changes the meaning of the gradient

Convexity provides the extra inequality

$$f(y) \ge f(x) + \langle\nabla f(x), y - x\rangle.$$

Therefore, if $\nabla f(x^\star) = 0$, then $f(y) \ge f(x^\star)$ for every $y$; stationarity is global optimality. Non-convex functions lack that implication. For $f(x) = x^3$ one has $f'(0) = 0$ but $0$ is neither a local nor a global minimum. This simple counterexample explains the transition from convex textbook guarantees to the approximate-stationarity targets of Sections IV–V of our survey.

## 5. Acceleration and complexity: correctly delimit the context

Nesterov-type acceleration is foundational for improving rates of smooth *convex* minimization. It does *not* follow that the same convex proof establishes efficient escape from non-convex saddles. That later claim belongs to specialized works (e.g., our Ref. [10]) whose accelerated updates include additional curvature-sensitive analysis or interventions. The distinction matters when discussing how optimization rates depend on what the algorithm can query.

:::theorem What this reference contributes to the survey
A baseline framework of smoothness, first-order methods, mathematical rates and oracle-based comparisons. It supports the notation and the elementary $O(\epsilon^{-2})$ first-order stationarity calculation, without asserting second-order stationarity or deep-learning generalization.
:::

## 6. Scope and source limitation

The publisher confirms the 2004 monograph and its four substantive chapters. The complete book text was not available for line-by-line checking here. Accordingly, these derivations are *explicitly pedagogical reconstructions of standard optimization facts*, not numbered quotations, verified theorem numbers, or a claim that the full book was read. They can serve as background notes, but a theorem-by-theorem book review would require the text.

:::plain Survey relevance (Preliminaries/RQ4)
Cite this book for classical first-order optimization background, not for a theorem about arbitrary non-convex neural-network minima or efficient saddle escape.
:::

## 7. Our notes

This is the mathematical reference behind the basic gradient descent and smoothness arguments. The key lesson for our survey is which parts survive without convexity: loss decrease and small-gradient guarantees can survive, while "small gradient means global minimum" does not.

> **Publisher:** [https://doi.org/10.1007/978-1-4419-8853-9](https://doi.org/10.1007/978-1-4419-8853-9).
