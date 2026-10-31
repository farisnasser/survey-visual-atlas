# Why Gradient Descent Avoids Saddles

:::minute The paper in one minute
**Question:** Gradient descent stops moving if the gradient is zero. Why does it not usually end up at saddle points?

**Model/Method:** Deterministic gradient descent, initialized from a continuous random distribution, analyzed using the Stable-Center Manifold Theorem.

**Finding:** Under smoothness and a small fixed step size, the chance of converging to any *particular strict saddle* is zero. With additional conditions, its limit is a local minimum almost surely.

**Why we care:** Noise is *not necessary* for asymptotic strict-saddle avoidance. But this says nothing general about how *quickly* GD leaves saddle neighborhoods.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. Gradient descent is a map from one point to the next

Gradient descent picks a point $z$ and moves against the gradient: $z_{k+1} = z_k - \alpha \nabla f(z_k)$. The gradient points uphill, so a small step against it lowers the loss. To first order,

$$f\big(z - \alpha \nabla f(z)\big) \approx f(z) - \alpha \|\nabla f(z)\|^2 .$$

The decrease is proportional to $\|\nabla f\|^2$, so the method only stops where $\nabla f(z) = 0$: a **critical point**. Nothing in the update tells a minimum apart from any other flat point.

### 2. A zero gradient can hide three different shapes

The gradient only measures slope. The *shape* around a flat point is set by the second derivative, the Hessian $H = \nabla^2 f(z^*)$. Near a critical point,

$$f(z^* + v) \approx f(z^*) + \tfrac{1}{2}\, v^\top H\, v .$$

If every eigenvalue of $H$ is positive, $f$ rises in every direction: a **local minimum** (a bowl). If every eigenvalue is negative, it is a **local maximum** (a dome). If the signs are mixed, $f$ goes down along some directions and up along others: a **saddle**. In deep networks most critical points are of this last kind, which is why the question matters.

### 3. Eigen-directions turn one hard problem into many easy ones

Near $z^*$ the gradient is approximately linear, $\nabla f(z^* + v) \approx H v$, so one GD step sends the offset $v$ to $(I - \alpha H)\, v$. Write $v$ in the eigenvectors of $H$. Each coordinate then evolves on its own, multiplied by the same number at every step:

$$u_{i,k} = (1 - \alpha \lambda_i)^k \, u_{i,0} .$$

A multiplier below one in size shrinks its coordinate; a multiplier above one makes it grow geometrically. A **negative** eigenvalue $\lambda_i < 0$ gives $1 - \alpha\lambda_i > 1$: that direction pushes the iterates *away* from the critical point.

### 4. Why the step size must be small

Let $L$ be the Lipschitz constant of the gradient. Then every Hessian eigenvalue satisfies $|\lambda_i| \le L$. With $0 < \alpha < 1/L$, every multiplier $1 - \alpha\lambda_i$ is positive. No direction is ever flipped, and two different points can never be sent to the same next point. GD can therefore be **run backwards**: every iterate has exactly one predecessor. This is what lets the paper turn a statement about a small neighbourhood of the saddle into a statement about starting points anywhere.

### 5. "Probability zero" means "zero volume"

If the starting point is drawn from any distribution with a density (Gaussian, uniform on a box, and so on), the probability of landing in a set is the integral of that density over the set. A set with zero volume gets probability zero: a line in the plane, or a surface in three dimensions. For example, a uniformly random point in the unit square lands exactly on the line $y = 0.5$ with probability $0$.

:::key Putting the five ideas together
One negative eigenvalue (idea 2) creates a direction that grows under GD (idea 3). Only starts with *exactly zero* component along it can still converge to the saddle, and those starts form a thin, zero-volume set (idea 5). The small step keeps GD invertible (idea 4), so the thinness carries back to starts anywhere in the space. That is the paper's main theorem, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Start with the paradox: a zero gradient need not mean a minimum

Gradient descent repeats $z_{k+1} = z_k - \alpha\nabla f(z_k)$. At any critical point, $\nabla f = 0$: if we initialize *exactly* there, GD stays there. The paper asks a different question: if we initialize at a random point, how likely is it that the entire trajectory eventually *converges* to a strict saddle?

## 2. Worked example: one attractive direction, one repulsive direction

Choose a two-dimensional quadratic with a saddle at the origin:

$$f(x,y) = \tfrac12 x^2 - \tfrac12 y^2, \qquad \nabla f(x,y) = \begin{bmatrix} x \\ -y \end{bmatrix}, \qquad H = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}.$$

The Hessian eigenvalues are $+1$ and $-1$. Applying GD *coordinate by coordinate* gives

$$x_{k+1} = (1-\alpha)x_k, \qquad y_{k+1} = (1+\alpha)y_k.$$

For $\alpha = 0.2$, start at $(x_0, y_0) = (0.40, 0.01)$:

$$(x_1, y_1) = (0.320, 0.012), \qquad (x_2, y_2) = (0.256, 0.0144).$$

After ten steps, $(x_{10}, y_{10}) \approx (0.043, 0.062)$. The $x$-component shrinks toward the saddle, but the $y$-component grows away from it.

:::key Why this example settles the intuition
To converge to $(0,0)$, we need $y_0 = 0$ *exactly*. Every point on the $x$-axis approaches the saddle; any $y_0 \neq 0$ is repelled. The $x$-axis has zero area in $\mathbb{R}^2$, so a continuous random initialization lands on it with probability zero.
:::

## 3. From two coordinates to a high-dimensional Hessian

Near a stationary point $z^*$, diagonalize the symmetric Hessian:

$$H = Q\,\mathrm{diag}(\lambda_1, \dots, \lambda_d)\,Q^\top.$$

For the quadratic approximation, each Hessian eigen-direction behaves like

$$\underbrace{u_{i,k+1}}_{\text{next component}} = \underbrace{(1 - \alpha\lambda_i)}_{\text{GD multiplier}}\;\underbrace{u_{i,k}}_{\text{current component}}.$$

When $0 < \alpha < 1/L$, where $L$ is the gradient Lipschitz constant:

| Hessian direction | Multiplier | Consequence near the point |
|---|:-:|---|
| $\lambda_i > 0$ | $0 < 1 - \alpha\lambda_i < 1$ | Contracts: locally attractive. |
| $\lambda_i < 0$ | $1 - \alpha\lambda_i > 1$ | Expands: unstable direction. |
| $\lambda_i = 0$ | $1$ | Linearization alone does not settle behavior. |

**Definition to remember.** In Lee et al.'s terminology, a *strict saddle* is any critical point $z^*$ with $\lambda_{\min}(H(z^*)) < 0$. It may still have zero eigenvalues. (This definition also includes a local maximum with negative curvature.)

## 4. What does "probability zero" really mean?

:::cols
:::col
For the 2D quadratic, the set of initial points converging to the origin is the entire $x$-axis, not merely the origin. This is called the *stable set*. It is infinitely long but has *zero area*.

In $d$ dimensions, the set of initializations flowing into a strict saddle is geometrically thin: its local stable-center manifold has dimension *less than* $d$.
:::
:::col
:::figure Stable set (the $x$-axis) and unstable motion around the saddle of $f = \tfrac12x^2 - \tfrac12y^2$.
<svg viewBox="0 0 250 160" role="img" aria-label="The x-axis is the stable set: arrows on it point toward the saddle at the origin. Red arrows off the axis curve away from it.">
<defs>
<marker id="ntArB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#7cc4ff"/></marker>
<marker id="ntArR" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#ff5d73"/></marker>
<marker id="ntArM" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#8e99bd"/></marker>
</defs>
<line x1="120" y1="150" x2="120" y2="12" stroke="#8e99bd" stroke-width="1" marker-end="url(#ntArM)"/>
<text x="120" y="9" fill="#8e99bd" font-size="9" text-anchor="middle" font-style="italic">y</text>
<line x1="10" y1="85" x2="226" y2="85" stroke="#7cc4ff" stroke-width="2"/>
<path d="M226 85h6" stroke="#8e99bd" stroke-width="1" marker-end="url(#ntArM)"/>
<text x="236" y="88" fill="#8e99bd" font-size="9" font-style="italic">x</text>
<line x1="30" y1="85" x2="72" y2="85" stroke="#7cc4ff" stroke-width="2" marker-end="url(#ntArB)"/>
<line x1="210" y1="85" x2="168" y2="85" stroke="#7cc4ff" stroke-width="2" marker-end="url(#ntArB)"/>
<circle cx="120" cy="85" r="3.5" fill="#e8ecf8"/>
<text x="112" y="77" fill="#e8ecf8" font-size="9" text-anchor="end">saddle</text>
<text x="44" y="32" fill="#7cc4ff" font-size="9" text-anchor="middle">stable set</text>
<text x="44" y="45" fill="#7cc4ff" font-size="9" text-anchor="middle" font-style="italic">y = 0</text>
<path d="M164 79Q142 68 140 30" fill="none" stroke="#ff5d73" stroke-width="1.8" marker-end="url(#ntArR)"/>
<path d="M76 91Q98 102 100 140" fill="none" stroke="#ff5d73" stroke-width="1.8" marker-end="url(#ntArR)"/>
<text x="188" y="32" fill="#ff5d73" font-size="9" text-anchor="middle">unstable</text>
<text x="188" y="45" fill="#ff5d73" font-size="9" text-anchor="middle">motion</text>
</svg>
:::
:::
:::

## 5. Theorem 4: the actual main result

:::theorem Strict-saddle avoidance (Theorem 4; paper p. 5)
If $f \in C^2$, its gradient is globally $L$-Lipschitz, $z^*$ is a fixed strict saddle, the initialization distribution is absolutely continuous, and $0 < \alpha < 1/L$, then

$$\boxed{\Pr\Big(\lim_{k\to\infty} z_k = z^*\Big) = 0.}$$
:::

**Important precision.** This is a guarantee for *a specified saddle*. To take a union over all strict saddles, the paper's Corollary 9 assumes their set is *at most countable*. To conclude convergence *to a local minimum*, the trajectory must also have a limit (Theorem 11).

## 6. Proof intuition: why a local fact becomes a global guarantee

The local quadratic example alone is not enough: a random start may be arbitrarily far away, and the function can curve nonlinearly. Lee et al. use a dynamical-systems theorem to make the argument rigorous.

1. **Express GD as a map.** Define $g(z) = z - \alpha\nabla f(z)$; its Jacobian is

   $$Dg(z) = I - \alpha\nabla^2 f(z).$$

   At a strict saddle, a negative Hessian eigenvalue produces a Jacobian eigenvalue *greater than one*: at least one direction is unstable.
2. **Characterize the local set that can remain near the saddle.** The *Stable-Center Manifold Theorem* places the non-escaping initial conditions inside a lower-dimensional local manifold. Lower-dimensional smooth manifolds have Lebesgue measure zero.
3. **Show that GD can be inverted when $\alpha < 1/L$.** The paper's Proposition 8 proves that $g$ is a *diffeomorphism* (a differentiable bijection with differentiable inverse). A useful construction is

   $$g^{-1}(y) = \arg\min_x \left\{ \tfrac12 \|x - y\|^2 - \alpha f(x) \right\}.$$

   The minimized function is strongly convex under the step-size condition, so it defines a unique preimage.
4. **Pull the local exceptional set backward.** Any trajectory converging to the saddle eventually enters the local region. Its initialization therefore lies in a preimage $g^{-k}(W^{cs}_{\mathrm{loc}})$ for some finite $k$. Diffeomorphisms preserve null sets under inversion; a countable union of null sets remains null.

   $$\underbrace{W^s(z^*)}_{\text{all starts converging to } z^*} \;\subseteq\; \bigcup_{k=0}^{\infty} \underbrace{g^{-k}\big(W^{cs}_{\mathrm{loc}}\big)}_{\text{measure-zero preimage}}.$$

:::key What to understand, not memorize
The *one negative Hessian eigenvalue* creates instability. The stable-center manifold theorem shows that saddle-convergent trajectories are confined to a thin set. The invertibility of GD lets that local conclusion apply to initializations anywhere in the domain.
:::

## 7. From saddle avoidance to convergence to a local minimum

**Corollary 9 (paper p. 7).** If the set of saddles is at most countable and every saddle is strict, then the probability that GD converges to *any* saddle is zero. Isolated saddles are one sufficient way to obtain countability.

**Theorem 11 (paper p. 7).** Add the assumption that $\lim_{k\to\infty} z_k$ *exists*. The limit is then a local minimizer almost surely.

:::warn Do not confuse these statements
**Avoiding convergence** to a saddle does not automatically imply **convergence** of the iterates, and neither implies convergence to a **global** minimum. This is the critical qualification behind the paper's title.
:::

## 8. What about convergence speed?

The paper also studies the *Łojasiewicz gradient inequality* near a converged critical point:

$$\|\nabla f(z)\| \;\ge\; m\,|f(z) - f(z^*)|^{a}, \qquad 0 \le a < 1.$$

Under its additional conditions, Proposition 14 gives a geometric bound for $0 < a \le 1/2$ and a polynomial bound when $1/2 < a < 1$. These are *local convergence-to-minimizer* results, *not* general finite-time bounds for leaving saddle neighborhoods.

## 9. Main contributions, assumptions and limitations

:::cols
:::col
### Main contributions
- Shows a general almost-sure *strict-saddle avoidance* theorem for unperturbed, deterministic GD.
- Reframes optimization as a discrete dynamical system, using geometry of stable sets.
- Establishes invertibility of the GD map for a sufficiently small constant step.
- Gives conditional *local-minimum convergence* and local-rate consequences.
:::
:::col
### Important assumptions
- Twice continuously differentiable $f$; globally $L$-Lipschitz gradient.
- Fixed step $0 < \alpha < 1/L$.
- Initial point has a distribution absolutely continuous with respect to volume.
- To conclude a local-minimum limit: appropriate strict-saddle/countability and iterate-convergence conditions.
:::
:::

:::warn What this paper does *not* prove
No generic **finite-time saddle escape** bound; no guarantee of a **global minimum**; no direct theorem for **ordinary minibatch SGD**; and no blanket result for **degenerate non-minimizing stationary points without negative Hessian curvature**. Strict saddles *may* have some zero Hessian eigenvalues and are still covered.
:::

## 10. How it fits with the other papers in our survey

| Paper | Main question | What it contributes to our narrative |
|---|---|---|
| **Saxe (2014)** | Why can learning plateau? | Slow dynamics can arise from coupled weights and initialization, not necessarily bad minima. |
| **Dauphin (2014)** | Why are saddles troublesome? | Saddle geometry motivates curvature-based escape interventions. |
| **Lee (2016)** | Can plain GD converge to strict saddles? | Almost-sure *asymptotic avoidance* without added noise. |
| **Jin (2017)** | Can saddle escape be efficient? | Perturbed GD provides a *high-probability finite-time* result under stronger smoothness/parameter conditions. |
| **Li (2018)** | What do loss landscapes look like? | Empirical 2D visualizations and architecture effects; not a substitute for high-dimensional guarantees. |

## 11. Quick reference for revision

| Term | What you need to remember |
|---|---|
| **GD map** | $g(z) = z - \alpha\nabla f(z)$; $Dg(z) = I - \alpha\nabla^2 f(z)$. |
| **Strict saddle** | Critical point with $\lambda_{\min}(\nabla^2 f) < 0$. |
| **Stable set** | $W^s(z^*) = \{z_0 : z_k \to z^*\}$. |
| **Measure zero** | Geometrically thin initialization set; continuous random starts land there with probability zero. |
| **Main theorem** | $\Pr(z_k \to z^*) = 0$ for any specified strict saddle. |
| **Local-minimum claim** | Needs saddle-set condition and existence of $\lim_k z_k$. |

## 12. What to read, and what goes into Papers Master

**Read closely:** Section 3 (quadratic example), Section 4 / Theorem 4, Proposition 8, Corollary 9 and Theorem 11. **Read conceptually:** stable-center manifold proof; Łojasiewicz rate discussion. **No need to memorize:** technical rate proof steps.

:::plain Papers Master — suggested "Our notes" (copy, 2 lines)
Randomly initialized GD with $0 < \alpha < 1/L$ almost surely avoids convergence to any specified strict saddle: the stable-center manifold and invertibility of the update map make its attraction set measure zero. Local-minimum convergence additionally needs conditions ensuring the iterates converge and the saddles can be ruled out collectively; no general finite-time escape bound.
:::

> **Primary source:** Lee, Simchowitz, Jordan & Recht (2016), *Gradient Descent Only Converges to Minimizers*, COLT, pp. 1–12. [proceedings.mlr.press/v49/lee16.pdf](https://proceedings.mlr.press/v49/lee16.pdf). Worked numerical example and diagram are pedagogical adaptations of Section 3.
