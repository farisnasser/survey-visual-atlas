# Stochastic Gradients & Stationarity

:::minute The paper in one minute
**Question:** If our objective is non-convex and our gradients are noisy, can an optimization algorithm still be guaranteed to make progress?

**Answer:** Yes. The authors propose a *randomized stochastic gradient* (RSG) method and bound the work needed to find an *approximately stationary point*. This is *not* a guarantee of reaching a global or local minimum.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. A smooth gradient caps how much one step can go wrong

$f$ is **$L$-smooth** when its gradient changes by at most $L$ times the distance moved: $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$. Integrating the gradient along the segment from $x$ to $y$ turns this into a quadratic upper bound on the loss:

$$f(y) \le f(x) + \langle \nabla f(x),\, y - x\rangle + \tfrac{L}{2}\|y - x\|^2 .$$

The linear term is the first-order Taylor prediction; the $\tfrac{L}{2}\|y-x\|^2$ term is the worst the curvature can add. Example: $f(w) = \sin w$ has $L = 1$. From $x = 0$ to $y = 0.5$ the bound gives $0 + 0.5 + 0.125 = 0.625$, and the true value is $\sin 0.5 \approx 0.479$. Setting $y = w_k - \eta G_k$ gives the notes' "descent inequality" in §3.

### 2. Noise adds a floor to the squared step

The stochastic gradient $G$ equals the true gradient plus zero-mean noise. Expanding the square and using $\mathbb{E}[G - \nabla f] = 0$ splits its second moment into signal and noise:

$$\mathbb{E}\|G\|^2 = \|\nabla f\|^2 + \mathbb{E}\|G - \nabla f\|^2 \le \|\nabla f\|^2 + \sigma^2 .$$

Example: in one dimension with $\nabla f = 2$, let $G$ be $1$ or $3$ with equal probability. Then $\mathbb{E}[G] = 2$ (unbiased), $\mathbb{E}[G^2] = (1 + 9)/2 = 5 = 2^2 + 1$, so $\sigma^2 = 1$. Even at a stationary point, where $\nabla f = 0$, the step still has size $\sigma$ on average. This is the source of the $L\eta^2\sigma^2/2$ term in §3.

### 3. Telescoping turns per-step progress into a total budget

If each step lowers the loss by some amount $d_k$ (up to the noise term), summing over $N$ steps cancels every intermediate value: $\sum_{k=1}^N \big(f(w_k) - f(w_{k+1})\big) = f(w_1) - f(w_{N+1})$. Because $f$ is **bounded below** by $f_{\inf}$, this total is at most $\Delta = f(w_1) - f_{\inf}$. A finite budget spread over $N$ steps forces the *average* progress per step, and hence the average squared gradient, to be small. No convexity is used anywhere.

### 4. The step size balances two errors

The notes' final bound has a term that falls with $\eta$ and one that rises with it. Minimizing $2\Delta/(\eta N) + L\eta\sigma^2$ over $\eta$ gives

$$\eta^\star = \sqrt{\frac{2\Delta}{N L \sigma^2}}, \qquad \text{bound} = 2\sqrt{\frac{2\Delta L \sigma^2}{N}} .$$

Example: $\Delta = L = \sigma^2 = 1$. With $N = 100$ steps, $\eta^\star \approx 0.141$ and the bound is $\approx 0.283$. With $N = 10\,000$, $\eta^\star \approx 0.0141$ and the bound is $\approx 0.0283$: a hundred times more steps buys only ten times less error. This is the $O(N^{-1/2})$ rate, and the reason the step must shrink like $N^{-1/2}$. (Both steps satisfy $\eta \le 1/L$.)

### 5. A small average means a random iterate is good on average

The bound controls the *average* of $\mathbb{E}\|\nabla f(w_k)\|^2$ over the run, not any single iterate. Picking one iterate at random converts the average into a statement about one output. Example: if the squared gradients along a run are $0.9, 0.3, 0.1, 0.1$, their average is $0.35$; an iterate chosen uniformly at random has expected squared gradient exactly $0.35$. This is why RSG returns a randomly selected $w_R$. With a constant step, the paper's selection probabilities are uniform.

:::key Putting the five ideas together
Smoothness bounds the loss after one noisy step (idea 1). Noise adds a $\sigma^2$ floor to that step (idea 2). Summing the steps uses only a finite budget $\Delta$ (idea 3), so the average squared gradient falls like $N^{-1/2}$ once the step size balances the two errors (idea 4), and a randomly selected iterate inherits that average (idea 5). This is the RSG guarantee, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. First principles: what does "stochastic" mean?

**Stochastic = involving randomness.** For a training loss averaged over $n$ examples, full gradient descent uses every example: $f(w) = \frac{1}{n}\sum_{i=1}^n \ell_i(w)$, so $\nabla f(w) = \frac{1}{n}\sum_i \nabla \ell_i(w)$. SGD instead uses a gradient from a randomly sampled example or minibatch, written $G(w,\xi)$. Its update is

$$\boxed{w_{k+1} = w_k - \eta_k G(w_k, \xi_k)} \qquad \boxed{\mathbb{E}[G(w,\xi) \mid w] = \nabla f(w)}\,.$$

The expectation says the stochastic estimate is *unbiased*: each sample may point a little differently, but its average is the true gradient. The random variable $\xi$ describes which data was sampled; $\eta_k$ is the step size. Usually we also assume bounded noise, $\mathbb{E}\|G - \nabla f\|^2 \le \sigma^2$.

## 2. What does convergence mean for a non-convex loss?

A **stationary point** satisfies $\nabla f(w^*) = 0$. Because non-convex surfaces may have minima, maxima, and saddles, a small gradient alone cannot tell us which one we found. Therefore the target is often **approximate first-order stationarity**:

$$\boxed{\mathbb{E}\big[\|\nabla f(w_R)\|^2\big] \le \varepsilon.}$$

Here $\|\nabla f\|$ is the length of the gradient vector; $\varepsilon$ is a chosen tolerance. RSG outputs a randomly selected iterate $w_R$ from the optimization run (with probabilities set using its step-size schedule), rather than simply relying on the last iterate.

## 3. Why does the squared gradient become small? (compact math)

Assume $f$ is **$L$-smooth**: its gradient does not change too sharply, $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$. This gives the standard descent inequality after one stochastic update:

$$f(w_{k+1}) \le f(w_k) - \eta\langle \nabla f(w_k), G_k\rangle + \frac{L\eta^2}{2}\|G_k\|^2.$$

Taking expectations, using the unbiased-gradient and bounded-variance assumptions, and taking $\eta \le 1/L$ yields

$$\frac{\eta}{2}\,\mathbb{E}\|\nabla f(w_k)\|^2 \le \mathbb{E}[f(w_k) - f(w_{k+1})] + \frac{L\eta^2\sigma^2}{2}.$$

Sum over $N$ updates (the losses *telescope*), and let $\Delta = f(w_1) - f_{\inf} \ge 0$:

$$\boxed{\frac{1}{N}\sum_{k=1}^{N} \mathbb{E}\|\nabla f(w_k)\|^2 \le \frac{2\Delta}{\eta N} + L\eta\sigma^2.}$$

**Meaning:** More iterations reduce the first term; reducing the step size controls the noise term. With a suitable $\eta \propto N^{-1/2}$, the bound scales like $O(N^{-1/2})$ when gradient noise is present. This is a *teaching derivation of the standard smooth-SGD bound*; the paper gives its own RSG-specific output rules and complexity results.

:::key Why it matters for our survey
**First-order convergence is not saddle escape.** Ghadimi–Lan establish that we can approach points with small gradients despite noisy updates. Later saddle-escape papers ask the harder question: can we additionally rule out *negative curvature* and avoid strict saddles? **Assumptions:** smoothness, suitable step sizes, unbiased controlled-noise gradients, and an objective bounded below. The paper also analyzes *zeroth-order* methods (using function values without direct gradients).
:::

:::ours Our takeaway
SGD can be theoretically reliable at getting the gradient close to zero, even for non-convex losses. The catch is that "gradient near zero" does not necessarily mean "good minimum."
:::

> **Primary source:** Ghadimi & Lan (2013), *Stochastic First- and Zeroth-Order Methods for Nonconvex Stochastic Programming*, SIAM J. Optim. [doi.org/10.1137/120880811](https://doi.org/10.1137/120880811).
