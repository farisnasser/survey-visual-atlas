# SASSHA: Sharpness-aware adaptive second-order optimization with stable Hessian approximation

:::minute The paper in one minute
**Question:** Can approximate second-order optimizers retain fast, curvature-aware steps without repeatedly settling in sharp solutions that generalize poorly?

**Model/Method:** Combine a SAM-like worst-neighborhood gradient with an estimated diagonal Hessian; stabilize the denominator using square roots and reuse curvature estimates.

**Finding:** Across tested workloads SASSHA finds flatter solutions with competitive or improved validation performance. Its convergence theorem is conditional on *convexity*, not a global theory for neural networks.

**Why we care:** It distinguishes improving an optimizer's practical generalization from proving saddle escape or global non-convex convergence.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Dividing by curvature fixes ill-conditioning

Take $f(x, y) = \tfrac12(0.01\,x^2 + 4\,y^2)$: one flat direction (curvature $0.01$) and one steep direction (curvature $4$). Plain GD must keep $\eta < 2/4 = 0.5$ to stay stable along $y$. At the largest safe step, $\eta = 0.5$, the flat coordinate is multiplied by $1 - 0.5 \times 0.01 = 0.995$ per step and takes about $139$ steps just to halve. A **Newton step** divides each gradient coordinate by its curvature, $x^+ = x - H^{-1}g$; on this quadratic it lands on the minimum in one step, whatever the curvatures. That is the speed second-order methods are after.

### 2. Negative curvature turns a Newton step around

Newton's step points to the stationary point of the local quadratic model, and that point is not always a minimum. On $f(x) = -\tfrac12 x^2$ (curvature $-1$, a maximum at $0$), from $x = 0.5$ the Newton step $x - f'(x)/f''(x) = 0.5 - (-0.5)/(-1)$ lands at $0$: straight onto the maximum. Dividing by the *absolute* curvature instead gives $0.5 - (-0.5)/1 = 1$, a step downhill and away from it. Neural-network losses have many directions of negative curvature, so a curvature-scaled method that should only go downhill divides by $|H|$ rather than $H$.

### 3. A diagonal Hessian can be estimated with random signs

The full Hessian of a network is far too large to form. Its diagonal can be estimated with **Hutchinson's method**: draw a random vector $z$ with independent $\pm 1$ entries, compute the Hessian-vector product $Hz$ (about the cost of one extra backward pass), and use $z \odot Hz$ (elementwise product). On average the off-diagonal terms cancel:

$$\mathbb{E}\big[z \odot Hz\big] = \operatorname{diag}(H).$$

For $H = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$, the probe $z = (1, 1)$ gives $(3, 4)$ and $z = (1, -1)$ gives $(1, 2)$; their average is $(2, 3)$, the true diagonal. Single probes are noisy, which is why such methods average the estimates over steps, and each probe costs a backward pass, which is why refreshing it less often saves time.

### 4. A square root sits halfway between GD and Newton

Scale each gradient coordinate by $|h|^{-p}$, where $h$ is that coordinate's curvature. $p = 0$ is plain GD, $p = 1$ is the absolute-value Newton step of idea 2, and $p = \tfrac12$ is in between (the same choice Adam makes with $\sqrt{v}$). On the quadratic of idea 1 with $p = \tfrac12$, a step $\eta$ multiplies each coordinate by $1 - \eta\sqrt{h}$. With $\eta = 0.5$ the steep coordinate is multiplied by $0$ and the flat one by $0.95$, so it halves in about $14$ steps instead of $139$. The square root keeps much of Newton's speed-up while dividing by $\sqrt{0.01} = 0.1$ instead of $0.01$ when a curvature estimate is tiny.

:::key Putting the four ideas together
Curvature-scaled steps are fast on ill-conditioned problems (idea 1) but must use absolute curvature to keep going downhill (idea 2). The diagonal curvature comes from cheap, noisy Hessian-vector probes (idea 3), and the square root tempers division by small estimates while keeping most of the speed (idea 4). SASSHA applies this preconditioner to a SAM gradient taken at a perturbed point, as made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. First principles: two objectives conflict

A second-order-style method rescales a gradient by curvature, roughly $x^+ = x - \eta H^{-1} g$. This can accelerate poorly conditioned directions, but a small estimated Hessian entry can yield a very large step. Meanwhile, SAM changes which gradients are followed by first looking for the highest local loss inside a ball:

$$\min_x \max_{\|\epsilon\| \le \rho} f(x + \epsilon).$$

A first-order approximation chooses $\epsilon^* = \rho g / \|g\|$. SASSHA combines these ideas: use the perturbed-point gradient for sharpness awareness, but precondition it with inexpensive curvature information.

## 2. Worked example: why naive inverse curvature can explode

Imagine a two-dimensional parameter vector with gradient $g = (1, 1)^\top$ and diagonal curvature estimate

$$\widehat H = \begin{bmatrix} 0.01 & 0 \\ 0 & 4 \end{bmatrix}.$$

An unregularized inverse-Hessian direction is

$$\widehat H^{-1} g = (100, 0.25)^\top.$$

The first coordinate is enormous solely because its estimated curvature is tiny. A square-rooted absolute diagonal gives instead

$$|\widehat H|^{-1/2} g = (10, 0.5)^\top.$$

The problematic amplification drops from 100 to 10 while retaining the fact that the first direction is flatter. This is an *illustrative scaling calculation*, not the exact SASSHA full update with moving averages, bias correction, and weight decay.

:::key Why absolute values and square roots?
Diagonal curvature estimates can be noisy or negative. Taking their absolute values enables a nonnegative scale; taking a square root softens division by nearly zero estimates. It is not the same as inverting the entire true Hessian.
:::

## 3. How the algorithm is assembled

At step $t$, SASSHA uses a minibatch gradient $g_t = \nabla f_{\mathcal B}(x_t)$, computes the SAM-style perturbation, and evaluates a second gradient:

$$\epsilon_t^* = \rho\,\frac{g_t}{\|g_t\|}, \qquad \widetilde g_t = \nabla f_{\mathcal B}(x_t + \epsilon_t^*).$$

It forms an exponential moving average $m_t$ of $\widetilde g_t$, and every $k$ steps estimates the perturbed-point Hessian diagonal. A running curvature statistic $D_t$ tracks *absolute* Hessian entries and the denominator is proportional to $\sqrt{D_t}$, rather than $D_t$. Between curvature-refresh steps, the old denominator is reused. Algorithm 1 gives the full form, including bias correction and weight decay.

### Why would lazy Hessian updates help?

Computing even a diagonal Hessian approximation repeatedly adds cost. The authors hypothesize that sharpness-aware training visits regions where curvature changes more gradually, so a recently computed estimate remains useful longer. This is a practical hypothesis supported by ablations, not a universal identity about curvature evolution.

## 4. What the mathematical results really guarantee

Theorem 4.4 is important to cite accurately. It assumes a lower-bounded, twice-differentiable *convex*, smooth objective, along with a nonvanishing-gradient technical condition over the specified finite steps and step/radius sequences satisfying

$$\sum_t \eta_t = \infty, \qquad \sum_t \eta_t^2 < \infty, \qquad \sum_t \rho_t^2 \eta_t < \infty.$$

Its conclusion is

$$\boxed{\liminf_{t\to\infty} \|\nabla f(x_t)\| = 0.}$$

This is weaker than asserting convergence of the whole sequence to a unique solution, and it is *not* a general non-convex strict-saddle-escape result. Corollary 4.6 separately studies necessary linear-stability conditions for minima under an idealized stochastic-Hessian model.

:::warn Don't overstate the theorem
Good experimental deep-learning performance and a convex convergence theorem are different pieces of evidence. Neither by itself establishes global optimality for arbitrary ReLU networks.
:::

## 5. Experimental results and contributions

The authors compare SGD, AdaHessian, Sophia-H, Shampoo, and SASSHA using multiple sharpness metrics, loss-surface projections, validation accuracy, stability tests, and wall-clock efficiency. On ResNet-32 / CIFAR-100, their Table 1 reports SASSHA validation accuracy 72.14% versus 69.32% for SGD, with smaller measured maximum Hessian eigenvalue (107 versus 265). These particular numbers are experimental, not a universal accuracy improvement. They also study vision and language workloads. **Contributions.** (i) Put SAM-style neighborhood gradients inside an approximate second-order optimizer; (ii) stabilize the curvature denominator via the square root of absolute diagonal estimates; (iii) reuse Hessian approximations at reduced cost; (iv) examine flatness and stability of selected solutions.

## 6. Assumptions and limitations

- The formal convergence statement uses convexity and carefully decaying step/radius sequences; deep-network training is generally non-convex.
- Diagonal Hessian approximations omit off-diagonal curvature and may be biased/noisy; curvature-refresh intervals introduce a practical tuning trade-off.
- Flatness and validation performance are empirically associated in these experiments, not proved equivalent for all parameterizations.
- There is no general second-order stationary-point or saddle-escape complexity guarantee in this paper.

## 7. Quick reference

| | |
|---|---|
| **Sharpness objective** | $\min_x \max_{\lVert\epsilon\rVert \le \rho} f(x + \epsilon)$. |
| **Perturbation** | $\epsilon^* = \rho g / \lVert g\rVert$. |
| **Gradient used** | $\widetilde g = \nabla f(x + \epsilon^*)$. |
| **Stable preconditioner** | Estimate absolute diagonal Hessian, then use a square-root denominator. |
| **Lazy update** | Recompute curvature every $k$ steps; reuse otherwise. |
| **Guarantee** | Theorem 4.4: $\liminf \lVert\nabla f\rVert = 0$ under convex assumptions. |

### Survey bridge / Papers Master note

SASSHA combines SAM-style sharpness control and approximate second-order preconditioning, using square-rooted absolute diagonal Hessians and lazy updates for stability and speed. Reports flatter, better-generalizing solutions in studied tasks, but its convergence theory assumes convexity; cite as a practical intervention, not a strict-saddle escape theorem.

> **Primary source:** Shin, Lee, Chung & Lee (2025), *SASSHA: Sharpness-aware adaptive second-order optimization with stable Hessian approximation*, ICML 2025. [Original publication](https://proceedings.mlr.press/v267/shin25d.html)
