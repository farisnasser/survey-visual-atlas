# SSRGD: Simple Stochastic Recursive Gradient Descent for Escaping Saddle Points

:::minute The paper in one minute
**Question:** Can a near-optimal first-order variance-reduced algorithm gain second-order saddle-escape guarantees simply by adding occasional random perturbations?

**Model/Method:** SSRGD uses a SPIDER/SARAH-like recursive gradient estimator, snapshot resets, and perturbation-triggered “super epochs” when the gradient is small.

**Finding:** For finite sums it achieves $O(n + \sqrt{n}/\varepsilon^2)$ first-order gradients and $\widetilde{O}(\sqrt{n}/\varepsilon^2 + \sqrt{n}/\delta^4 + n/\delta^3)$ for $(\varepsilon, \delta)$ second-order stationarity, with relevant constants suppressed.

**Why we care:** A simple, well-controlled stochastic recursion plus perturbation can substitute for a dedicated negative-curvature detector.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Two tolerances: gradient size and curvature

An **$(\varepsilon, \delta)$-second-order stationary point** has $\|\nabla f(x)\| \le \varepsilon$ and $\lambda_{\min}(\nabla^2 f(x)) \ge -\delta$. Keeping $\varepsilon$ and $\delta$ separate lets a user ask for a loose curvature test and a tight gradient test, or the reverse. The classical choice ties them together as $\delta = \sqrt{\rho\varepsilon}$, where $\rho$ is the Hessian Lipschitz constant.

With that choice the paper's second-order bound becomes $\widetilde{O}(\sqrt{n}/\varepsilon^2 + n/\varepsilon^{1.5})$ in $n$ and $\varepsilon$: the middle term $\sqrt{n}/\delta^4$ turns into $\sqrt{n}/\varepsilon^2$ and merges with the first.

### 2. "Near-optimal" means matching a lower bound

A **lower bound** says that no algorithm of a given kind can do better. Fang et al. (2018) proved that any algorithm using component gradients needs $\Omega(\sqrt{n}/\varepsilon^2)$ of them in the worst case (for $n$ up to order $\varepsilon^{-4}$), just to find an $\varepsilon$-first-order stationary point. SSRGD's first-order cost $O(n + \sqrt{n}/\varepsilon^2)$ meets this bound, and its second-order cost has the same leading term up to logarithms. Escaping saddles costs, in this sense, almost nothing extra.

For $n = 10^6$ and $\varepsilon = 10^{-3}$, ignoring constants, $\sqrt{n}/\varepsilon^2 = 10^9$, against $n^{2/3}/\varepsilon^2 = 10^{10}$ for SVRG-type methods.

### 3. A kick grows geometrically along negative curvature

Near a saddle with curvature $-\gamma$, a gradient step multiplies the unstable coordinate by $1 + \eta\gamma$. Starting from a kick of size $r$, reaching distance $D$ takes about $\ln(D/r)/\ln(1 + \eta\gamma)$ steps. With $\eta\gamma = 0.1$, going from $0.01$ to $0.5$ takes $\ln 50/\ln 1.1 \approx 41$ steps. The count grows only logarithmically as the kick shrinks, and it scales like $1/(\eta\gamma)$. This is why the paper's super epoch has length $t_{\mathrm{thres}} = \widetilde{O}(1/(\eta\delta))$.

### 4. The rate is a budget: decrease per gradient, times the total drop

The paper's proof overview (Sec. 5.2) counts how much $f$ falls per stochastic gradient. Each iteration costs $b + n/m = 2\sqrt{n}$ gradients, with the epoch's full gradient spread over its $m = \sqrt{n}$ steps. With a large gradient, each iteration lowers $f$ by about $\varepsilon^2/L$. Near a saddle, a super epoch of $1/(\eta\delta)$ iterations lowers $f$ by about $\delta^3/\rho^2$, which is $\delta^4/(L\rho^2)$ per iteration. Dividing the total drop $\Delta_f = f(x_0) - f^*$ by these rates and multiplying by $2\sqrt{n}$ gives the first two terms, $\sqrt{n}/\varepsilon^2$ and $\sqrt{n}/\delta^4$. The third term, $n/\delta^3$, comes from accounting for the full gradients exactly instead of spreading them out.

:::key Putting the four ideas together
The target separates gradient and curvature tolerances (idea 1). The recursive estimator already reaches small gradients at the optimal $\sqrt{n}/\varepsilon^2$ cost (idea 2). An occasional uniform kick near a small gradient grows along any negative curvature within $\widetilde{O}(1/(\eta\delta))$ steps (idea 3). Each escape buys a fixed decrease, and the budget of decreases gives $\widetilde{O}(\sqrt{n}/\varepsilon^2 + \sqrt{n}/\delta^4 + n/\delta^3)$ (idea 4), made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Start with the recursive gradient estimator

For $f(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$, SGD’s sampled gradients can vary greatly between examples. SSRGD instead starts each epoch with a full gradient and corrects a running estimate via matched component-gradient differences:

$$v_t = v_{t-1} + \frac{1}{b}\sum_{i \in I_b}\big(\nabla f_i(x_t) - \nabla f_i(x_{t-1})\big), \qquad x_{t+1} = x_t - \eta v_t.$$

The same sampled indices appear at the new and previous point, which often cancels much of the sampling variability. The baseline is refreshed periodically to avoid accumulating uncontrolled error.

### Worked example: why the recursive correction can be accurate

Let $f_1(x) = \tfrac12(x-1)^2$ and $f_2(x) = \tfrac12(x+1)^2$. Thus $f'(x) = x$. Take $x_0 = 2$ and full-gradient estimate $v_0 = 2$. A step of length $0.1$ gives $x_1 = 1.8$. Whichever single component is sampled, its gradient difference is

$$\nabla f_i(1.8) - \nabla f_i(2) = -0.2.$$

So $v_1 = 2 - 0.2 = 1.8 = f'(1.8)$, despite sample gradients $0.8$ and $2.8$ individually being noisy. This toy example is not a general unbiasedness claim for the recursion.

:::plain
**First-order versus second-order:** The recursive estimator reduces gradient noise efficiently. By itself it only supports fast search for small gradients. To escape saddles, SSRGD adds a separate *random perturbation* rule.
:::

## 2. How a simple perturbation handles saddles

### Algorithm 1: trigger a super epoch

At an epoch start, if the full gradient norm falls below threshold $g_{\mathrm{thres}}$ and the method is not already in a super epoch, draw $\xi$ uniformly from a radius-$r$ ball and set $x \leftarrow x + \xi$. Run the same recursive gradient updates, but avoid immediately perturbing again. A super-epoch stopping rule tracks escape over a controlled time window.

### A two-dimensional escape calculation

Take $f(x, y) = \tfrac12 x^2 - \tfrac12 y^2$, a strict saddle at the origin. Ordinary gradient descent at $(0, 0)$ is motionless. Perturb to $(0.01, 0.01)$ and use $\eta = 0.1$:

$$x_{t+1} = 0.9\,x_t, \qquad y_{t+1} = 1.1\,y_t.$$

After 10 idealized full-gradient steps, $x_{10} \approx 0.00349$ while $y_{10} \approx 0.02594$. The negative-curvature direction becomes dominant. SSRGD’s actual stochastic trajectory needs a nontrivial analysis to ensure estimator error does not overwhelm this growth.

### Theorem 1: first-order rate

With $m = b = \sqrt{n}$ and a suitably small step size, the finite-sum method reaches an $\varepsilon$-first-order stationary point in expectation using

$$O\!\left(n + \frac{\sqrt{n}}{\varepsilon^2}\right)$$

stochastic gradients, with the smoothness constant and initial objective gap suppressed.

### Theorem 2: escape / second-order rate

Under the additional Hessian-Lipschitz assumption and calibrated perturbations, it reaches

$$\|\nabla f(x)\| \le \varepsilon, \qquad \lambda_{\min}(\nabla^2 f(x)) \ge -\delta$$

with high probability using

$$\widetilde{O}\!\left(\frac{\sqrt{n}}{\varepsilon^2} + \frac{\sqrt{n}}{\delta^4} + \frac{n}{\delta^3}\right)$$

finite-sum stochastic gradients (again suppressing problem-dependent constants). Theorem 3 and later results separately treat online expectation objectives with their own variance assumptions.

:::plain
**Read:** Algorithm 1 and Table 1, Section 4’s Theorems 1–2, and Section 5’s proof overview. Compare the online guarantees in Tables 1–2 before transferring conclusions across oracle settings.
:::

## 3. Results, assumptions, limitations, and relationships

**Contributions.** A recursive-gradient algorithm with near-optimal finite-sum first-order complexity; a simple uniform-perturbation modification yielding approximate second-order stationarity; online extensions; a simpler route to strong bounds than separate NEON2 modules in the regimes compared. **Assumptions.** Lower-bounded smooth objective, component-gradient Lipschitz continuity, Hessian Lipschitz continuity for saddle-escape theorem, bounded variance for online results, suitable batch size $b$, epoch length $m$, step size, perturbation radius and stopping thresholds. **Limitations.** Full gradients at snapshot epochs can be expensive when $n$ is huge; the complexity bound concerns stochastic gradient evaluations rather than total elapsed time. The guarantee is approximate second-order stationarity, not global optimality; the perturbation schedule must satisfy the theorem’s conditions.

## Cross-paper comparison

**Stabilized SVRG (Ref 86)** uses an explicit gradient *shift* inside super epochs; SSRGD instead uses recursive SPIDER/SARAH gradient differences and a perturbation. **SpiderBoost (Ref 84)** has a related recursive estimator but its main theorems are first-order. **NEON2 (Ref 33)** employs a separate negative-curvature finder; SSRGD directly modifies its existing optimizer. **Jin (2017)** motivates why a random kick can prevent stagnation at strict saddles.

## Quick reference

| Concept | Interpretation |
|---|---|
| Recursive $v_t$ | Correct prior estimate using matched sample-gradient differences. |
| Super epoch | Perturb only occasionally near a low-gradient candidate. |
| $O(n + \sqrt{n}/\varepsilon^2)$ | Finite-sum first-order stochastic-gradient complexity order. |
| $(\varepsilon, \delta)$ point | Small gradient, smallest Hessian eigenvalue at least $-\delta$. |
| $\widetilde{O}(\sqrt{n}/\varepsilon^2 + \sqrt{n}/\delta^4 + n/\delta^3)$ | Second-order finite-sum complexity order (constants hidden). |

### Papers Master note

:::plain
Li’s SSRGD combines a recursive SPIDER/SARAH-style estimator with occasional uniform perturbations, attaining near-optimal first-order finite-sum complexity and a second-order saddle-escape guarantee without a separate eigenvector search. Finite-sum/online rates require different assumptions.
:::

> **Verified source:** [original NeurIPS 2019 PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/addfa9b7e234254d26e9c7f2af1005cb-Paper.pdf), Algorithm 1, Theorems 1–3, Section 5, Tables 1–2.
