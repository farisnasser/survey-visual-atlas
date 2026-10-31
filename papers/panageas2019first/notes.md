# First-Order Methods Almost Always Avoid Saddle Points: The Case of Vanishing Step-Sizes

:::minute The paper in one minute
**Research question:** Earlier stable-manifold arguments established that sufficiently small *constant*-step gradient descent (GD) almost surely does not converge to a strict saddle. In actual optimization, step sizes often vary with the iteration. The paper asks whether the same avoidance statement survives this time dependence, including when the step size vanishes.

**Principal result:** Under its smoothness, step-size and initialization hypotheses, the set of initial conditions whose iterates converge to strict saddle points has measure zero. This is an *asymptotic, almost-everywhere* statement, not a bound on how many iterations an escape requires.

**Do not conflate:** "almost surely does not converge to a strict saddle" $\neq$ "escapes every saddle quickly" $\neq$ "finds the global minimum."
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A product of growth factors is controlled by a sum

Along a negative-curvature direction each step multiplies the error by $1 + \gamma\alpha_k$. Taking logarithms turns the product into a sum, and $\log(1 + x) \approx x$ for small $x$, so the total growth is about $\exp(\gamma\sum_k \alpha_k)$. Two schedules with $\gamma = 1$ show the difference exactly: with $\alpha_k = 1/k$, $\prod_{k=1}^{t}(1 + 1/k) = t + 1$, which grows without bound; with $\alpha_k = 1/k^2$, the infinite product $\prod_{k\ge1}(1 + 1/k^2)$ equals $\sinh(\pi)/\pi \approx 3.68$, so the error can grow at most by that factor. This is the notes' §4 in numbers.

### 2. With summable steps, GD can stop where the gradient is not zero

If $\sum_k \alpha_k < \infty$, the iterates move a bounded total distance and may freeze anywhere. Example on the notes' saddle $f = \tfrac12 x^2 - \tfrac12 y^2$: the stable coordinate obeys $x_{k+1} = (1 - \alpha_k)x_k$, so with $\alpha_k = 0.8/(k + 1)^2$ starting from $x_0 = 1.5$ it converges to $1.5\prod_{k\ge0}\big(1 - 0.8/(k+1)^2\big) \approx 0.174$, not to $0$. The paper's Fig. 1 shows the same with $\alpha_k = 1/k^4$. That is why the theorem asks for steps of order $\Omega(1/k)$, whose sum diverges.

### 3. A changing map needs a new stable-manifold theorem

Lee et al. studied one fixed map $g(x) = x - \alpha\nabla f(x)$ applied again and again (an **autonomous** system). With $\alpha_k$ changing, the map $G_k$ changes at every step (a **time-nonhomogeneous** system), and the classical stable-manifold theorem no longer applies. In the quadratic example the answer is easy, because every $G_k$ is diagonal: $x$ and $y$ are multiplied by $1 - \alpha_k$ and $1 + \alpha_k$, and with $\sum_k\alpha_k = \infty$ only starts with $y_0 = 0$ (the $x$-axis, a zero-area set) reach the saddle. For general $f$ the directions mix from step to step, and the paper's Theorem 4.1 is the tool that still yields a measure-zero stable set.

### 4. Mirror descent is gradient descent in a different geometry

Mirror descent replaces the squared distance in the gradient step by a Bregman divergence $D_h$: $x_{k+1} = \arg\min_x \{\alpha_k\langle \nabla f(x_k), x\rangle + D_h(x, x_k)\}$. With $h = \tfrac12\|x\|^2$ it is ordinary GD. With the entropy $h(x) = \sum_i x_i\log x_i$ on the probability simplex it becomes the multiplicative update $x_{k+1,i} \propto x_{k,i}\,e^{-\alpha_k\partial_i f(x_k)}$. Example: $x = (0.5, 0.5)$, gradient $(1, 0)$, $\alpha = 1$ gives $(0.269, 0.731)$. The paper proves the same avoidance result for this family, for proximal point and for GD on manifolds.

:::key Putting the four ideas together
Negative curvature amplifies errors by a product controlled by $\sum_k\alpha_k$ (idea 1), so steps that shrink too fast can stall the method at non-critical points (idea 2). With steps of order $\Omega(1/k)$ the amplification is unbounded, and a stable-manifold theorem for time-varying maps (idea 3) shows that only a measure-zero set of starts reaches a strict saddle, for GD and its mirror, proximal and manifold variants (idea 4). The notes below state the result and its limits.
:::
:::

:::divider Study notes
:::

## 1. Research question: does saddle avoidance survive a decaying step size?

Earlier stable-manifold arguments established that sufficiently small *constant*-step gradient descent (GD) almost surely does not converge to a strict saddle. In actual optimization, step sizes often vary with the iteration. The paper asks whether the same avoidance statement survives this time dependence, including when the step size vanishes. It addresses GD, mirror descent, manifold gradient descent and proximal-point-type methods, not just one Euclidean iteration.

## 2. Rebuild the geometry from first principles

For smooth $f : \mathbb{R}^d \to \mathbb{R}$, GD with a varying step size is

$$x_{k+1} = G_k(x_k) = x_k - \alpha_k\nabla f(x_k), \qquad \alpha_k > 0.$$

At a stationary point $x^*$ with $\nabla f(x^*) = 0$, linearize the update by writing $e_k = x_k - x^*$:

$$e_{k+1} \approx \big(I - \alpha_k H\big)e_k, \qquad H = \nabla^2 f(x^*).$$

An eigenvector $v$ with $Hv = -\gamma v$ ($\gamma > 0$) has scalar error $z_k = v^\top e_k$ evolving as

$$z_{k+1} \approx (1 + \gamma\alpha_k)z_k, \qquad z_t \approx z_0\prod_{k<t}(1 + \gamma\alpha_k).$$

The multiplier is greater than one, so negative curvature locally repels perturbations away from the saddle. This is a *teaching linearization*, not the full global theorem. The remaining issue is whether almost every starting point is repelled when the dynamics change with $k$.

## 3. What method and result does the paper actually provide?

The authors establish a stable-manifold theorem for *time-nonhomogeneous* dynamical systems, since ordinary autonomous stable-manifold theorems cannot be applied directly to the sequence $G_k$. They use it to extend almost-sure avoidance to several first-order methods, including isolated and non-isolated strict-saddle sets (as stated in the NeurIPS version; the authors' arXiv v3 correction adds the assumption that the Hessian at the saddle is invertible, which only isolated saddles satisfy), for suitably controlled vanishing step sizes. An important regime has $\alpha_k = \Omega(1/k)$ in addition to the method-specific small-step/regularity requirements.

:::key Principal result
Under its smoothness, step-size and initialization hypotheses, the set of initial conditions whose iterates converge to strict saddle points has measure zero. This is an *asymptotic, almost-everywhere* statement, not a bound on how many iterations an escape requires.
:::

## 4. Why the step-size condition is meaningful

If $\sum_k \alpha_k = \infty$ and the steps are small, then

$$\log\prod_{k<t}(1 + \gamma\alpha_k) = \sum_{k<t}\log(1 + \gamma\alpha_k) \sim \gamma\sum_{k<t}\alpha_k \to \infty.$$

By contrast, when $\sum_k \alpha_k < \infty$, the product can stay bounded. For a simple quadratic $f(x, y) = \frac{1}{2}x^2 - \frac{1}{2}y^2$, the unstable coordinate obeys $y_{k+1} = (1 + \alpha_k)y_k$: if the total step length is finite, its growth may be limited. This example explains the intuition behind why steps that decrease *too fast* can change the conclusions; it is not a replacement for the paper's nonautonomous manifold proof.

## 5. Assumptions, limitations, and survey relevance

The main arguments assume appropriate differentiability and controlled step sizes and use generic/random initialization. The result does not cover arbitrary discontinuous gradients at ReLU kinks, does not establish avoidance of fully degenerate saddles (where no negative Hessian eigenvalue exists), and does not provide a polynomial hitting time or global optimum guarantee.

:::warn Do not conflate
"almost surely does not converge to a strict saddle" $\neq$ "escapes every saddle quickly" $\neq$ "finds the global minimum."
:::

This belongs to **RQ2** (gradient dynamics), **RQ3** (limits of escape claims), and **RQ4** (smoothness and schedule assumptions).

## 6. Our notes

This paper extends Lee et al.'s strict-saddle avoidance story to learning rates that decay over time. The intuition is that negative-curvature directions still amplify small errors, but proving it needs a time-varying stable-manifold argument. It is useful for our survey because it makes clear that almost-sure avoidance is a statement about where training can end up, not how quickly it gets there. **Original source:** [https://papers.neurips.cc/paper_files/paper/2019/hash/3fb04953d95a94367bb133f862402bce-Abstract.html](https://papers.neurips.cc/paper_files/paper/2019/hash/3fb04953d95a94367bb133f862402bce-Abstract.html). **Verification:** conference record and original manuscript; the eigenmode example is pedagogical.

> **Primary source:** Panageas, Piliouras & Wang (2019), *First-order methods almost always avoid saddle points: The case of vanishing step-sizes*, NeurIPS 2019; corrected version arXiv 1906.07772v3 (2025). [papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2019/hash/3fb04953d95a94367bb133f862402bce-Abstract.html).
