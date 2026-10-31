# Recent Theoretical Advances in Non-Convex Optimization

:::minute The paper in one minute
**Problem.** A global minimum of a general non-convex objective can be computationally intractable to find. Which meaningful weaker targets have provable algorithms and complexity bounds?

**Method.** Survey of deterministic, stochastic, higher-order, zeroth-order and structured non-convex optimization.

**Main synthesis.** Guarantees depend on the target notion (stationarity or global optimality), assumptions and oracle access.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Each tolerance measures a different thing

Three targets appear throughout: a small gradient $\|\nabla f(x)\| \le \epsilon$, a small expected squared gradient $\mathbb{E}\|\nabla f(x)\|^2 \le \epsilon^2$, and a small optimality gap $f(x) - f_* \le \epsilon$. They are not interchangeable. On $f(x) = x^2/2$, the point $x = 0.1$ has gradient $0.1$ but gap $0.005$: a gradient tolerance $\epsilon$ corresponds here to a gap of $\epsilon^2/2$. On a non-convex function, a small gradient may not bound the gap at all, because the point may be a saddle.

### 2. Exponents of $\epsilon$ hide large factors

Complexities are stated as powers of $1/\epsilon$. At $\epsilon = 10^{-2}$, $\epsilon^{-2} = 10^4$ and $\epsilon^{-4} = 10^8$: the step from the deterministic gradient baseline to the stochastic one is a factor of $10^4$ here, and it grows as $\epsilon$ shrinks. Comparing two rates is meaningful only when both count the same oracle calls for the same tolerance.

### 3. A gradient-dominance condition gives linear convergence without convexity

The review's PŁ condition requires $\tfrac12\|\nabla f(x)\|^2 \ge \mu\,(f(x) - f_*)$. The function $f(x) = x^2 + 3\sin^2 x$ is not convex ($f''(x) = 2 + 6\cos 2x$ reaches $-4$), yet on $[-6, 6]$ the ratio $f'(x)^2/(2f(x))$ never drops below about $0.176$, so PŁ holds there with $\mu \approx 0.176$, and $f''$ is at most $L = 8$. The resulting guarantee $(1 - \mu/L)^k$ promises a $10^6$-fold reduction within about $623$ GD steps; from $x_0 = 5$, GD with step $1/L$ actually needs only $10$. A guarantee is an upper bound, often a loose one.

### 4. Zeroth-order methods pay for dimension

Without gradients, derivatives must be estimated from function values. A forward difference $\big(f(x + \delta e_i) - f(x)\big)/\delta$ approximates one partial derivative: for $f(x) = x^2$ at $x = 1$ with $\delta = 10^{-3}$ it gives $2.001$ instead of $2$. A full gradient in $\mathbb{R}^d$ takes $d + 1$ function values this way, which is why zeroth-order rates carry extra factors of the dimension.

:::key Putting the four ideas together
A rate means something only together with its tolerance (idea 1) and its oracle, and the exponents differ by large factors (idea 2). Structural conditions such as PŁ upgrade stationarity to global convergence (idea 3), and weaker oracles cost extra dimension factors (idea 4). The notes below derive the two baseline rates and show where these distinctions enter.
:::
:::

:::divider Study notes
:::

## 1. First-order stationarity versus global optimality

For $f : \mathbb{R}^d \to \mathbb{R}$ a standard algorithmic target is $\|\nabla f(x)\| \le \epsilon$. It can include saddles: for $f(x, y) = x^2 - y^2$, the origin has zero gradient but negative Hessian curvature.

If $f$ has an $L$-Lipschitz gradient, gradient descent with step $1/L$ satisfies

$$f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2.$$

Summing from $k = 0$ to $T - 1$ gives

$$\frac1T\sum_{k=0}^{T-1}\|\nabla f(x_k)\|^2 \le \frac{2L[f(x_0) - f_*]}{T}.$$

Thus an iterate with gradient norm at most $\epsilon$ exists within $O(L\Delta/\epsilon^2)$ gradient evaluations. This is a *generic baseline*, not a deep-network global-optimality guarantee.

## 2. Stochastic gradient complexity requires a different metric

For $g_k$ with $\mathbb{E}[g_k \mid x_k] = \nabla f(x_k)$ and bounded conditional variance $\mathbb{E}\|g_k - \nabla f(x_k)\|^2 \le \sigma^2$, a standard descent calculation yields

$$\frac1T\sum_{k=0}^{T-1}\mathbb{E}\|\nabla f(x_k)\|^2 \lesssim \frac{\Delta}{\eta T} + L\eta\sigma^2.$$

Choosing $\eta$ on the order of $T^{-1/2}$ gives the familiar $O(T^{-1/2})$ squared-gradient measure and hence an $O(\epsilon^{-4})$ *stochastic-gradient baseline* for $\mathbb{E}\|\nabla f\|^2 \le \epsilon^2$. Variance reduction or stronger oracle assumptions may improve this rate.

## 3. Structural properties may permit much stronger claims

Under a Polyak–Łojasiewicz (PL) condition, for some $\mu > 0$,

$$\frac12\|\nabla f(x)\|^2 \ge \mu\big(f(x) - f_*\big),$$

the same GD descent inequality implies $f(x_{k+1}) - f_* \le (1 - \mu/L)(f(x_k) - f_*)$ for appropriate constants. PL is weaker than convexity but excludes non-optimal stationary points; it must not be assumed for general neural networks.

## 4. Higher-order and zeroth-order methods

A second-order stationary target additionally requires $\lambda_{\min}(\nabla^2 f(x)) \gtrsim -\gamma$. Negative curvature can aid saddle escape, but Hessian/HVP queries and inner subproblem costs cannot be compared directly with a gradient-only oracle count. Zeroth-order methods estimate derivatives from function values and introduce additional dimension and noise dependencies.

:::warn Do not misattribute.
The review organizes results from many independent papers. The GD and SGD calculations above are standard first-principles derivations included for teaching; they are not new theorems due to Danilova et al. A rate is meaningful only alongside its tolerance definition, regularity, and oracle model.
:::

**Survey connection.** Best used to contextualize our Section V complexity table and Section VIII's assumption audit. **Our Notes.** Useful for preventing apples-to-oranges comparisons: two algorithms may have different complexity exponents because they solve different stationarity problems with different information.

> **Source:** [https://arxiv.org/abs/2012.06188](https://arxiv.org/abs/2012.06188)
