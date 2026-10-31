# NEON2: Finding Local Minima via First-Order Oracles

:::minute The paper in one minute
**Question:** Can a method using Hessian-vector products be replaced by a purely first-order method without losing its theoretical oracle performance?

**Model/Method:** NEON2 detects negative curvature with stochastic or full gradient differences; the detector is plugged into existing stationary-point optimizers.

**Finding:** A modular reduction converts GD, SGD, SVRG, SCSG and Natasha2 variants to approximate second-order stationary-point methods, with first-order oracles only.

**Why we care:** Efficient saddle-escape routines need not build Hessians or even evaluate Hessian-vector products; gradients can be sufficient.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A small gradient is not enough: the second-order target

An **oracle** is the only way an algorithm may touch $f$: a **first-order oracle** returns $\nabla f(x)$ (or a stochastic estimate $\nabla f_i(x)$), a **Hessian-vector product** oracle returns $\nabla^2 f(x)\,u$. Complexity is counted in oracle calls. A first-order method naturally stops where $\|\nabla f(x)\| \le \varepsilon$, but $f(x, y) = x^2 - y^2$ has zero gradient at the origin, which is a saddle: its Hessian has eigenvalue $-2$. The paper's target adds a curvature test, $\nabla^2 f(x) \succeq -\delta I$: no eigenvalue below $-\delta$. A point passing both tests is an **$(\varepsilon, \delta)$-approximate local minimum**.

### 2. Two gradients give one Hessian-vector product

Taylor's theorem applied to the gradient gives $\nabla f(x + u) - \nabla f(x) = \nabla^2 f(x)\,u + O(\|u\|^2)$, where the error is controlled by the Lipschitz constant of the Hessian. For $f(x, y) = \tfrac12 x^2 - \tfrac12 y^2 + \tfrac13 y^3$ at the origin, $\nabla^2 f(0) = \mathrm{diag}(1, -1)$. With $u = (0.1, 0.1)$, the gradient difference is $(0.1, -0.09)$ while $\nabla^2 f(0)\,u = (0.1, -0.1)$: the error, $0.01$, is the size of $\|u\|^2$. Keeping $u$ small keeps the approximation good.

### 3. Repeated multiplication finds the most negative direction

Multiply a vector again and again by $I - \eta H$. Each eigen-direction of $H$ with eigenvalue $\lambda$ is scaled by $1 - \eta\lambda$ per step, so the direction with the most negative $\lambda$ grows fastest. With $H = \mathrm{diag}(2, -1)$ and $\eta = 0.2$ the multipliers are $0.6$ and $1.2$; after ten steps the second component has grown $2^{10} = 1024$ times relative to the first. This is the **power method**. If the vector grows past a set radius, its direction has negative curvature; if nothing grows, every eigenvalue is above the threshold.

### 4. Repetition turns "likely" into "almost surely"

A random test that is right with probability $2/3$ can be repeated independently and combined. With a majority vote over $k$ runs, the error probability falls exponentially in $k$; with a test whose positive answers can be verified, a single verified success suffices, and all $k$ runs fail with probability $(1/3)^k$, about $1.7 \times 10^{-5}$ for $k = 10$. Guarantees of the form "with probability at least $1 - p$" cost a factor $\log(1/p)$ in this way.

:::key Putting the four ideas together
The paper's goal is a point that passes the second-order test (idea 1) using gradients only. Gradient differences stand in for Hessian-vector products (idea 2), repeating them is a power method that exposes negative curvature (idea 3), and repetition with verification makes the answer reliable (idea 4). NEON2 packages this as a detector that any stationary-point finder can call, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Why replace the Hessian-vector oracle?

An algorithm may efficiently find points where $\|\nabla f(x)\| \le \varepsilon$, yet return a saddle because $\lambda_{\min}(\nabla^2 f(x)) \ll 0$. Some methods add *Hessian-vector products* to find a negative direction. NEON2 aims to eliminate that additional oracle while retaining comparable guarantees. Its target is

$$\|\nabla f(x)\| \le \varepsilon, \qquad \nabla^2 f(x) \succeq -\delta I.$$

The paper uses the expression *approximate local minimum* for this second-order condition. It does *not* assert that the returned point is an actual local minimum for every nonconvex function.

### The gradient-difference trick

Let $x_0$ be the reference point. For a small displacement $u$ and a sampled component function $f_i$, consider

$$\nabla f_i(x_0 + u) - \nabla f_i(x_0) \approx \nabla^2 f_i(x_0)\,u.$$

Averaging over a random sample index $i$ approximates the full Hessian's action on $u$. This replaces *one Hessian-vector product* by a gradient-difference operation using first-order oracles.

### Worked example: sample-gradient offsets cancel

Take two component losses in $\mathbb{R}^2$:

$$f_1(u) = \tfrac12 u^T H u + (2, -1)^T u, \quad f_2(u) = \tfrac12 u^T H u + (-2, 1)^T u, \quad H = \mathrm{diag}(2, -1).$$

At reference $x_0 = 0$, either sample yields

$$\nabla f_i(u) - \nabla f_i(0) = (Hu + a_i) - a_i = Hu.$$

With $u = (0.1, 0.1)$, this is $(0.2, -0.1)$. A centered step of size $0.2$ gives $u^+ = (0.06, 0.12)$; the negative-curvature component grows. In real stochastic problems the sample Hessians differ, so the cancellation is only of baseline offsets, not all noise. This is a pedagogical illustration.

## 2. NEON2's two principal settings

### Online/stochastic NEON2 (Algorithm 1–2, Theorem 1)

Starting from $x_1 = x_0 + \xi$ with a small random perturbation, the weak detector performs

$$x_{t+1} = x_t - \eta\big(\nabla f_i(x_t) - \nabla f_i(x_0)\big),$$

with fresh sample indices. This is approximately a randomized power iteration on $I - \eta H$; negative eigenmodes expand. A verification stage checks whether the proposed direction truly has sufficiently negative curvature, with the confidence boosted by repetitions. Under the theorem's Lipschitz gradient/Hessian assumptions, the online detector uses $\widetilde O(1/\delta^2)$ stochastic gradients (with hidden smoothness/logarithmic factors) to either return a unit vector $v$ with $v^T H v \le -\delta/2$, or certify $H \succeq -\delta I$ to specified confidence.

### Offline deterministic and finite-sum versions

The deterministic NEON2 uses a different first-order construction based on Chebyshev-style polynomial acceleration, obtaining a bound of order $\widetilde O(1/\sqrt{\delta})$ *full gradients* (or its finite-sum conversion) for negative-curvature search; the complexity includes hidden smoothness and $n$ factors in appropriate oracle units. The paper also describes a finite-sum NEON2 variant for cases where $n$ is moderate.

### The modular reduction

Combine any suitable first-order stationary-point finder with NEON2. When the gradient is large, the base optimizer makes progress. When the gradient is small, call NEON2: either curvature is approximately nonnegative, or use the found direction to decrease the loss. The improvement in the oracle complexity is the *negative-curvature detector*, not a new proof that all nonconvex functions lack spurious local minima.

:::plain Read carefully
Section 1.1 and its Theorems 1–7 (informal overview), Section 3 Algorithms 1–2 / Theorem 1, Section 4 (deterministic), Section 5 (finite-sum). Compare Figure 1 with the independent NEON paper, Ref 76.
:::

## 3. Contribution, assumptions, and limits

**Contributions.** NEON2 is a general-purpose *reduction*: it supplies a negative-curvature detection ingredient compatible with a variety of existing algorithms. The authors analyze online, deterministic and finite-sum oracles, and give improved first-order complexity bounds for multiple combinations (SGD, SCSG, GD, SVRG, Natasha2). **Assumptions.** Component functions have Lipschitz gradients and Hessians as required by the theorems, and the algorithms use randomized initialization, calibrated target curvature $\delta$, confidence parameters and step sizes. Some results require finite-sum or online sampling access. Bounds are in *gradient evaluations*, not wall-clock seconds. **Limitations.** Curvature detection still consumes multiple gradient computations and relies on smoothness. An approximate second-order stationary point is not guaranteed to be globally optimal or even an exact local minimum. The claimed reductions are theoretical complexity results; practical ranking may change with hardware and data.

### Connections to earlier papers

**NEON (Xu–Jin–Yang, Ref 76)** also amplifies negative Hessian eigenmodes using gradients, but NEON2 develops a different stochastic estimator and deterministic Chebyshev-based formulation. **Reddi et al. (Ref 74)** provides a generic first/second-order hybrid architecture that can benefit from cheaper negative-curvature routines. **Daneshmand (2018)** instead assumes stochastic gradients have energy along negative-curvature directions (CNC).

### Quick reference

| Term | Meaning |
|---|---|
| $(\varepsilon, \delta)$-minimum | $\lVert\nabla f\rVert \le \varepsilon$ and $\lambda_{\min}(H) \ge -\delta$. |
| Gradient difference | $\nabla f_i(x + u) - \nabla f_i(x)$ approximates $H_i u$. |
| Online NEON2 | Stochastic gradients; $\widetilde O(1/\delta^2)$ NC detection under assumptions. |
| Deterministic NEON2 | Full gradients; accelerated NC detection of order $\widetilde O(1/\sqrt{\delta})$ in gradient calls. |
| Reduction | Upgrade a first-order stationary-point method with a curvature check. |

### Papers Master note

:::plain
Allen-Zhu and Li introduce NEON2, a gradient-only negative-curvature detector for stochastic, deterministic and finite-sum settings. Plugging it into GD/SGD/SVRG/SCSG/Natasha2 yields approximate second-order stationarity without Hessian-vector oracles, under smoothness and specified oracle-complexity assumptions.
:::

> **Source:** [original NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/d4b2aeb2453bdadaa45cbe9882ffefcf-Paper.pdf), Sections 1–5, Algorithms 1–3, Theorems 1–7.
