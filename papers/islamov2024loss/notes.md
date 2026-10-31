# Loss Landscape Characterization Without Over-Parametrization

:::minute The paper in one minute
**Question:** The Polyak–Łojasiewicz (PL) condition makes a small gradient imply a small objective gap, but it excludes stationary points with suboptimal loss. The authors seek a structural assumption with convergence consequences that can also accommodate saddle points and does not demand an extreme overparameterization argument for each neural network.

**Paper result:** The proposed condition can include saddle-containing landscapes while supporting meaningful gradient-method bounds. The authors test the condition along trajectories of MLPs, CNNs, ResNets and several larger models, including Transformer workloads.

**Critical distinction:** A bound that approaches a ball around the optimal objective value is not equivalent to exact global convergence. Even checking the condition on many training runs does not establish it over the full parameter space.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. The PL condition rules out every bad stationary point

The **Polyak–Łojasiewicz** condition asks that $\tfrac12\|\nabla f(x)\|^2 \ge \mu\,(f(x) - f^*)$ for some $\mu > 0$ and every $x$. At a stationary point the left side is $0$, so $f(x) = f^*$: every stationary point is a global minimum. A saddle with higher loss violates it. Example: $f(x, y) = \tfrac12 x^2 + \tfrac14 y^4 - \tfrac12 y^2$ has $f^* = -\tfrac14$ (at $y = \pm 1$) but a saddle at the origin with gradient $0$ and $f = 0$: the gap is $\tfrac14$ while the gradient is zero, so no $\mu > 0$ works. This is why the paper looks for something weaker.

### 2. Convex losses point the gradient away from the solution

For a convex $f$, the tangent-plane inequality at $x$, evaluated at a minimizer $x^*$, gives

$$\langle \nabla f(x),\, x - x^* \rangle \ge f(x) - f(x^*) .$$

So the gradient has a positive component pointing away from the solution, and a step against it moves closer. The notes' distance expansion makes this exact: $\|x - \gamma g - x_p\|^2 = \|x - x_p\|^2 - 2\gamma\langle g, x - x_p\rangle + \gamma^2\|g\|^2$. Example: $f(x) = \tfrac12 x^2$, $x = 2$, $\gamma = 0.5$: the squared distance goes from $4$ to $4 - 2(0.5)(4) + 0.25(4) = 1$. Definition 1 with $\beta = 0$ is this kind of alignment (the paper says it then reduces to quasar-convexity around $x^*$ when the solution set is a single point).

### 3. The $\beta$ term is slack paid by components that disagree

Rewrite the right side of Definition 1:

$$\alpha\big(f_i(x) - f_i(x_p)\big) - \beta\big(f_i(x) - f_i^*\big) = (\alpha - \beta)\big(f_i(x) - f_i(x_p)\big) - \beta\big(f_i(x_p) - f_i^*\big).$$

The first term is the convex-style alignment with constant $\alpha - \beta$; the second is a slack, large when example $i$ is far from its *own* best value at the global solution $x_p$. Averaging over $i$ at a stationary point (where the left side is $0$) gives the necessary condition $(\alpha - \beta)(f(x) - f^*) \le \beta\,\sigma^2_{\text{int}}$, with $\sigma^2_{\text{int}} = \mathbb{E}_i[f^* - f_i^*]$ the paper's interpolation error. So a saddle is allowed only if the examples disagree: if every $f_i$ is minimized at a common point ($\sigma^2_{\text{int}} = 0$), every stationary point must be global. Example: $\alpha = 3$, $\beta = 2$, $\sigma^2_{\text{int}} = 0.5$ allow stationary points up to $1.0$ above $f^*$.

### 4. When the examples disagree, SGD keeps a floor

Take two examples $f_1(x) = \tfrac12(x - 1)^2$ and $f_2(x) = \tfrac12(x + 1)^2$. Their average $f(x) = \tfrac12(x^2 + 1)$ has $f^* = 0.5$ at $x = 0$, while each $f_i^* = 0$, so $\sigma^2_{\text{int}} = 0.5$. At $x = 0$ each sampled gradient is $\mp 1$, never zero, so constant-step SGD keeps jumping around the minimizer. With step $\gamma = 0.1$ its average gap $\mathbb{E}f - f^*$ settles near $\gamma/(2(2 - \gamma)) \approx 0.026$. The paper's Theorem 1 has a step-size-independent term $\tfrac{2\beta}{\alpha - \beta}\sigma^2_{\text{int}}$ on top of this: that is the "error floor" of the notes' §4.

:::key Putting the four ideas together
PL is too strong for landscapes with saddles (idea 1). The α–β condition keeps the convex-style pull toward the solution set (idea 2) but lets each example pay a slack, which makes room for saddles exactly when examples disagree (idea 3). The same disagreement leaves SGD a non-vanishing floor (idea 4). The notes below state the condition, the proof's first step and the paper's results.
:::
:::

:::divider Study notes
:::

## 1. The question: can landscape assumptions allow saddles?

The Polyak–Łojasiewicz (PL) condition makes a small gradient imply a small objective gap. That is useful for proving convergence, but it excludes stationary points with suboptimal loss. The authors seek a structural assumption with convergence consequences that can also accommodate saddle points and does not demand an extreme overparameterization argument for each neural network.

## 2. The actual proposed $\alpha$–$\beta$ condition

Write the empirical objective and its component losses as

$$f(x) = \frac{1}{n}\sum_{i=1}^{n} f_i(x), \qquad S = \arg\min f, \qquad x_p \in \mathrm{Proj}(x, S).$$

The central assumption (the paper's Definition 1, Eq. (2)) requires constants $\alpha > \beta > 0$ such that, for every parameter $x$, a projection $x_p$ can be chosen for which, *for all component losses $i$*,

$$\boxed{\langle \nabla f_i(x), x - x_p\rangle \ge \alpha\big(f_i(x) - f_i(x_p)\big) - \beta\big(f_i(x) - f_i^*\big).}$$

Here $f_i^* = \inf_x f_i(x)$. The left-hand side measures how much the individual gradient points *away from* a global solution. The first term asks for useful alignment; the second relaxes that requirement. Unlike PL, this condition can be satisfied by some objectives with saddles.

## 3. How the condition enters the convergence proof

For SGD, $x_{k+1} = x_k - \gamma_k\nabla f_{i_k}(x_k)$, expand the distance to a projected minimizer:

$$\begin{aligned} \|x_{k+1} - x_p\|^2 = {} & \|x_k - x_p\|^2 - 2\gamma_k\langle \nabla f_{i_k}(x_k), x_k - x_p\rangle \\ & + \gamma_k^2\|\nabla f_{i_k}(x_k)\|^2. \end{aligned}$$

Substituting the $\alpha$–$\beta$ condition provides a negative term proportional to the function-value gap plus error/noise terms. The paper uses these estimates to establish bounds for SGD, stochastic Polyak stepsizes (SPS$_{\max}$) and NGN. This display shows the first algebraic step, not the complete proof (which handles projection, sampling and stepsize constraints).

## 4. Main results: convergence with an error floor

Under the assumptions stated in its theorems, the paper derives for SGD a function-value convergence bound of order $O(K^{-1/2})$ *up to a nonvanishing neighborhood*. Its size includes a term proportional to $\beta\sigma^2_{\text{int}}$, where the interpolation-error quantity measures a mismatch among component minima. The bound is *not* a guarantee that arbitrary finite-width deep networks attain a global minimum exactly.

:::key Paper result
The proposed condition can include saddle-containing landscapes while supporting meaningful gradient-method bounds. The authors test the condition along trajectories of MLPs, CNNs, ResNets and several larger models, including Transformer workloads.
:::

## 5. Assumptions and key empirical caveat

The convergence theorems require the specified $\alpha$–$\beta$ property and additional regularity/stepsize conditions. Empirical tests check the inequality only along *sampled optimization trajectories*, not at every point in $\mathbb{R}^d$. Their experimental findings therefore do not prove that the loss landscape globally satisfies the condition. In **RQ4**, this is a rare attempt to bridge model-specific landscape assumptions and practical optimization, but it is not a replacement for unrestricted saddle-escape theory.

:::warn Critical distinction
A bound that approaches a ball around the optimal objective value is not equivalent to exact global convergence. Even checking the condition on many training runs does not establish it over the full parameter space.
:::

## 6. Our notes

This paper matters because many landscape theorems assume something that rules out saddles or requires unrealistic widths. The $\alpha$–$\beta$ condition is more flexible, and the authors actually check it on modern models. What we must emphasize is that the empirical tests are along training paths, and the theory can leave a nonzero error floor, so this does not magically settle global convergence for ordinary neural networks. **Original source:** [https://proceedings.neurips.cc/paper_files/paper/2024/file/52f050499cf82fa8efb588e263f6f3a7-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/52f050499cf82fa8efb588e263f6f3a7-Paper-Conference.pdf). **Verification:** Definition 1, Eq. (2), results in Sec. 4, empirical limitations in Sec. 6 checked.

> **Primary source:** Islamov, Ajroldi, Orvieto & Lucchi (2024), *Loss Landscape Characterization of Neural Networks without Over-Parametrization*, NeurIPS 2024. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/file/52f050499cf82fa8efb588e263f6f3a7-Paper-Conference.pdf).
