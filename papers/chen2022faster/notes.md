# Faster Perturbed Stochastic Gradient Methods for Finding Local Minima

:::minute The paper in one minute
**Question:** Can simple stochastic-gradient perturbations match strong second-order rates without a separate negative-curvature oracle?

**Model/Method:** LENA (Last stEp shriNkAge) alternates normalized-descent and escape phases, using recursive SPIDER or STORM gradient estimates and controlled step-length shrinkage.

**Finding:** Under stochastic smoothness and bounded-error assumptions, LENA-SPIDER and LENA-STORM reach an $(\varepsilon, \varepsilon_H)$ approximate second-order point in $\widetilde{O}(\varepsilon^{-3} + \varepsilon_H^{-6})$ stochastic gradients (problem constants hidden).

**Why we care:** It shows that the difficulty in combining perturbations with variance reduction is controlling estimator error *during escape*, not merely injecting noise.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. The oracle: noisy gradients, two points at a time

The objective is an expectation $F(x) = \mathbb{E}_\xi f(x; \xi)$, and the algorithm only sees **stochastic gradients** $\nabla f(x; \xi)$. Cost is the number of such evaluations. LENA, like SPIDER and STORM, uses a slightly stronger oracle: it can evaluate the *same* sample $\xi$ at two different points, which is what lets a difference $\nabla f(x_t; \xi) - \nabla f(x_{t-1}; \xi)$ cancel the sample's own offset.

The target is an **$(\varepsilon, \varepsilon_H)$-approximate local minimum**: $\|\nabla F(x)\| \le \varepsilon$ and $\lambda_{\min}(\nabla^2 F(x)) \ge -\varepsilon_H$. The classical choice is $\varepsilon_H = \sqrt{\varepsilon}$.

### 2. Normalized steps cap how far each iteration moves

A recursive estimator's error grows with the total squared distance travelled since its last reset (the notes' Eq. 4.1). An ordinary step $\eta\, d_t$ can be long when $d_t$ is large. A **normalized step** divides by the length: $x_{t+1} = x_t - \eta\, d_t/\|d_t\|$, so every step has length exactly $\eta$. For $d_t = (3, 4)$ and $\eta = 0.01$ the step is $(0.006, 0.008)$, of length $0.01$, whatever the size of $d_t$. LENA uses this in its descent phase so the estimator's error stays predictable.

### 3. Checking that a gradient is small costs a large batch

Averaging $B$ independent stochastic gradients with spread $\sigma$ gives an error of about $\sigma/\sqrt{B}$. To certify $\|\nabla F\| \le \varepsilon$ this way needs $B \approx \sigma^2/\varepsilon^2$ samples: $10^4$ for $\sigma = 1$ and $\varepsilon = 10^{-2}$.

The paper (Sec. 4) shows why this hurts. If every escape phase starts with such a batch ($\varepsilon^{-2}$ gradients) and buys a decrease of order $\varepsilon^{1.5}$, the total is $\varepsilon^{-1.5} \cdot \varepsilon^{-2} = \varepsilon^{-3.5}$, worse than the $\varepsilon^{-3}$ that SPIDER needs for first-order points. LENA instead reuses its running estimate $d_t$ to decide when to escape, which is only valid if $d_t$ stays accurate.

### 4. STORM sits between SGD and SARAH

The STORM estimator is $d_{t+1} = (1 - a)\big(d_t - \nabla f(x_t; \xi)\big) + \nabla f(x_{t+1}; \xi)$, with the same sample $\xi$ at both points. With $a = 1$ it is plain SGD, $d_{t+1} = \nabla f(x_{t+1}; \xi)$. With $a = 0$ it is the SARAH/SPIDER recursion, $d_t + \nabla f(x_{t+1}; \xi) - \nabla f(x_t; \xi)$. A small $a > 0$ keeps most of the recursion's accuracy while slowly forgetting old error, so STORM needs no periodic full-batch reset: a single loop.

### 5. Comparing with a lower bound

Arjevani et al. (2020) proved that any stochastic first-order method needs $\Omega(\varepsilon^{-3} + \varepsilon_H^{-5})$ gradients for this target. LENA's $\widetilde{O}(\varepsilon^{-3} + \varepsilon_H^{-6})$ has the same $\varepsilon^{-3}$. The two bounds coincide at $\varepsilon^{-3}$ whenever $\varepsilon_H \ge \sqrt{\varepsilon}$, since then $\varepsilon_H^{-6} \le \varepsilon^{-3}$. For smaller $\varepsilon_H$ a gap between $\varepsilon_H^{-6}$ and $\varepsilon_H^{-5}$ remains, which the paper leaves open.

:::key Putting the five ideas together
The algorithm only sees noisy, same-sample gradient pairs (idea 1). Normalized steps and a cap on movement keep the recursive estimate accurate (idea 2), which lets LENA trust that estimate instead of paying for a large verification batch at every escape (idea 3); STORM achieves the same with a single loop (idea 4). The result is $\widetilde{O}(\varepsilon^{-3} + \varepsilon_H^{-6})$, optimal in $\varepsilon$ (idea 5), made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. The problem: an inaccurate direction near a saddle

For a stochastic objective $F(x) = \mathbb{E}_\xi f(x; \xi)$ we can sample $\nabla f(x; \xi)$ but cannot cheaply calculate $\nabla F(x)$. A candidate approximate local minimum satisfies

$$\|\nabla F(x)\| \le \varepsilon, \qquad \lambda_{\min}(\nabla^2 F(x)) \ge -\varepsilon_H.$$

A small true gradient may mean a good minimum or a saddle. Perturbed GD separates these cases by adding a random kick at small gradients; the difficulty for stochastic methods is knowing whether the gradient is *actually* small when the estimator is noisy.

### Build the gradient estimate first

A SPIDER/SARAH recursion compares a sampled component at consecutive points using *the same* sample:

$$d_t = d_{t-1} + \frac{1}{b}\sum_{i \in B_t}\big[\nabla f_i(x_t) - \nabla f_i(x_{t-1})\big].$$

Its error depends on how far the iterates have moved since the reference estimate, not only on how many steps elapsed. In the paper, a key estimate relates squared gradient error to accumulated squared movements:

$$\|d_t - \nabla F(x_t)\|^2 \lesssim \sum_{j=s_t}^{t-1}\|x_{j+1} - x_j\|^2.$$

The suppressed constants, logs and probability conditions are important for the formal proof (Eq. 4.1).

### Worked example: why small steps improve the estimate

Take the illustrative components $f_1(x) = \tfrac12(x-1)^2$, $f_2(x) = \tfrac12(x+1)^2$. The exact gradient of their average is $F'(x) = x$. Start at $x_0 = 2$ with $d_0 = 2$ and move to $x_1 = 1.8$. For either sampled component,

$$f_i'(1.8) - f_i'(2) = -0.2, \qquad d_1 = 2 - 0.2 = 1.8.$$

The recursion reconstructs the exact full gradient here even though the individual gradients differ by 2. In general this exact cancellation does not hold, so restricting cumulative movement matters.

## 2. LENA: shrink the *last step*, not the objective

### The two phases in Algorithm 1

**GD phase:** When $\|d_t\|$ is large, make a normalized update of controlled length $\eta$; this permits progress without letting estimator error grow unpredictably. When the estimated gradient is small, sample a perturbation and begin the escape phase. **Escape phase:** Use the recursive gradient estimator while enforcing a bound on average squared motion. At a trigger involving cumulative displacement, LENA shrinks the *last update* so that total movement remains below its threshold. A sufficiently successful escape produces a noticeable function decrease; otherwise the algorithm returns an approximate second-order stationary candidate.

### Why the shrinkage matters mathematically

Imagine three proposed step lengths $0.03$, $0.04$ and $0.05$. Their total squared motion is

$$0.03^2 + 0.04^2 + 0.05^2 = 0.005.$$

If an estimator-accuracy argument allows only $0.003$, one can shrink the *last* displacement to $\sqrt{0.003 - 0.03^2 - 0.04^2} \approx 0.0224$. This is a pedagogical analogy to the paper’s average-movement safeguard, not its literal parameter rule.

### What the theorems establish

Theorems 5.1 and 5.3 analyze LENA with SPIDER and STORM estimators. Their stochastic-gradient complexity is

$$\widetilde{O}\big(\varepsilon^{-3} + \varepsilon_H^{-6}\big),$$

with objective-gap, noise, gradient/Hessian smoothness and confidence factors suppressed. Under $\varepsilon_H = \sqrt{\varepsilon}$, this becomes $\widetilde{O}(\varepsilon^{-3})$. The STORM version admits a single-loop implementation while SPIDER refreshes its estimator periodically.

:::warn
**Do not conflate guarantees:** These are stochastic *gradient evaluation* counts, not necessarily wall-clock or iteration counts; the result is approximate second-order stationarity, not global optimality.
:::

> Read original Sections 3–6; inspect Algorithm 1, Theorems 5.1 and 5.3, and the proof discussion around Eq. (4.1).

## 3. Contributions, assumptions, limits, survey role

**Contributions.** A simpler perturbed stochastic-gradient framework with near-optimal $\varepsilon^{-3}$ complexity at the standard $\sqrt{\varepsilon}$ curvature tolerance; explicit control of recursive estimator error by limiting iterate displacement; two implementations, SPIDER and STORM, with different loop structure. **Assumptions.** Every sampled loss is $L$-smooth with $\rho$-Lipschitz Hessian (Assumption 3.1); stochastic gradient error is uniformly bounded by $\sigma$ (Assumption 3.2); and the initial objective gap is finite (Assumption 3.3). Perturbation sizes, batches and stopping thresholds must satisfy the theorem choices. **Limitations.** Assumptions on every sample are strong. The escape guarantee can require carefully chosen parameters. A second-order point need not be globally optimal; online oracle complexities are not directly comparable with finite-sum complexities. The empirical comparisons are task-specific.

## Connections to other papers

**Jin (2017)** introduced explicit perturbations for deterministic GD. **SSRGD (Ref 83)** also perturbs a recursive estimator but with a different analysis and $\widetilde{O}(\varepsilon^{-3.5})$ classical tolerance rate. **SpiderBoost (Ref 80)** focuses primarily on first-order convergence; LENA adds second-order escape. **NEON2 (Ref 77)** detects negative curvature explicitly, whereas LENA avoids an additional curvature-finding module.

## Quick reference

| Concept | Interpretation |
|---|---|
| $(\varepsilon, \varepsilon_H)$ point | Gradient $\le \varepsilon$, negative curvature no worse than $-\varepsilon_H$. |
| LENA-SPIDER | Periodically refreshed, recursive gradient estimator. |
| LENA-STORM | Single-loop recursive estimator with momentum-like weights. |
| Step shrinkage | Caps averaged movement to control stochastic estimator error. |
| $\widetilde{O}(\varepsilon^{-3} + \varepsilon_H^{-6})$ | Stochastic-gradient oracle complexity under paper assumptions. |

### Papers Master note

:::plain
Chen–Zhou–Gu’s LENA combines variance-reduced stochastic gradients, occasional perturbations and last-step shrinkage; controlling movement keeps gradient estimates reliable during saddle escape. For $\varepsilon_H = \sqrt{\varepsilon}$, it obtains $\widetilde{O}(\varepsilon^{-3})$ oracle complexity under smoothness and bounded-noise assumptions.
:::

> **Source:** [official ALT 2022 paper](https://proceedings.mlr.press/v167/chen22b/chen22b.pdf), Assumptions 3.1–3.3, Algorithm 1, Theorems 5.1/5.3; example is pedagogical.
