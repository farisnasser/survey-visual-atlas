# Special Properties of Gradient Descent with Large Learning Rates

:::minute The paper in one minute
**Question:** Can the step size itself, even without SGD noise, change which minimum GD eventually reaches?

**Model/Method:** Compare large- and small-step full-batch GD in experiments and analyze a structured class of smooth nonconvex functions with sharp spurious minima.

**Finding:** A sufficiently large step may make a sharp local basin repelling; under additional global landscape conditions, GD instead reaches the global minimum.

**Why we care:** A learning rate does more than control speed: it can change the chosen optimization trajectory and final basin.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. With small steps, GD can never climb out of a basin

A function is **$L$-smooth** when its gradient changes by at most $L$ times the distance moved: $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$. This caps how fast the slope can turn, and gives the classic one-step guarantee

$$f\big(x - \eta \nabla f(x)\big) \le f(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\|\nabla f(x)\|^2 .$$

For $0 < \eta < 2/L$ the bracket is positive, so the loss drops at every step. With $L = 10$ and $\eta = 0.1$ the guaranteed drop is $0.05\,\|\nabla f\|^2$. A method whose loss never rises cannot cross a ridge: it stays in the basin it started in. Small-step GD and its limit, gradient flow, are therefore stuck with whatever basin the initialization picks.

### 2. The same step can be stable at a flat minimum and unstable at a sharp one

Near a minimum with curvature $\kappa$, GD multiplies the distance to it by $1 - \eta\kappa$ each step (the notes derive this in Section 2). The minimum attracts only if $\eta < 2/\kappa$. Take $\eta = 0.25$: a flat minimum with $\kappa = 0.3$ gives the multiplier $0.925$, a slow, steady pull inward. A sharp minimum with $\kappa = 10$ gives $-1.5$: GD overshoots further each time and is thrown out. A larger step does not just go faster; it changes *which* minima can hold the iterates.

### 3. One-point strong convexity: every gradient points home

The paper does not assume convexity. It uses a weaker property. A function is **$\mu$-one-point-strongly-convex (OPSC)** with respect to $x^\star$ on a region when, for every $x$ there,

$$\langle \nabla f(x),\, x - x^\star \rangle \ \ge\ \mu\,\|x - x^\star\|^2 .$$

The gradient may bend and the function may be nonconvex, but the negative gradient always has a component pointing toward $x^\star$. If the region is also $L$-smooth (so $\|\nabla f(x)\| \le L\|x - x^\star\|$), one step gives $\|x_{t+1} - x^\star\|^2 \le (1 - 2\eta\mu + \eta^2 L^2)\,\|x_t - x^\star\|^2$. For $\eta \le \mu/L^2$ the factor is below one: every step gets closer to $x^\star$. With $\mu = 1$ and $L = 2$, the step must be at most $0.25$, and at $\eta = 0.25$ the factor is $1 - 0.5 + 0.25 = 0.75$.

### 4. Large-step success needs a window of step sizes

Put ideas 2 and 3 together. To be thrown out of a sharp bad minimum that is $\mu^\dagger$-OPSC around itself (for a quadratic bowl, $\mu^\dagger$ is its curvature), the step must exceed $2/\mu^\dagger$. To keep moving toward the good minimum on the surrounding landscape, it must stay at most $\mu^\star/L^2$. Both hold only when

$$\frac{2}{\mu^\dagger} < \eta \le \frac{\mu^\star}{L^2}, \qquad \text{which needs } \mu^\dagger > \frac{2L^2}{\mu^\star}.$$

With $\mu^\star = 1$ and $L = 2$ the bad minimum must have $\mu^\dagger > 8$; for $\mu^\dagger = 10$ the window is $0.2 < \eta \le 0.25$. Too small a step stays trapped; too large a step loses the guarantee on the rest of the landscape (and can diverge). This is the shape of the step-size condition in the paper's Lemma 1 and Theorem 1.

:::key Putting the four ideas together
Small steps keep GD in its starting basin (idea 1). A large enough step makes a sharp minimum repel GD (idea 2), and a landscape that is one-point strongly convex toward the global minimum everywhere else keeps pulling the iterates home (idea 3), provided the step sits in a window between the two limits (idea 4). That is the contrast the paper's Theorem 1 proves for its function class, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Start with a non-global minimum

Suppose a loss has a sharp, suboptimal valley and a broader, lower valley. At small step sizes, GD can settle into the sharp bad valley. A larger step can overshoot this valley repeatedly and move into a region leading toward the global solution. This mechanism is deterministic: mini-batch noise is not needed to produce it.

The paper emphasizes that the usual smooth-GD theory studies small $\eta$ where loss decreases reliably each step. That regime cannot explain how step size can change which basin is reached.

## 2. Worked example: local instability can reject a valley

Near a local minimizer $x^\dagger$, approximate the loss by $L(x) \approx L(x^\dagger) + \frac{\kappa}{2}(x - x^\dagger)^2$, with $\kappa > 0$. If $e_t = x_t - x^\dagger$, then

$$e_{t+1} \approx (1 - \eta\kappa)\, e_t.$$

Take a sharp valley with $\kappa = 10$. With $\eta = 0.1$, $e_{t+1} = 0$ in this quadratic approximation: it is strongly attracting. With $\eta = 0.25$, $e_{t+1} = -1.5 e_t$: alternating errors grow, so the same point becomes locally repelling.

**Critical warning.** This only proves *local* instability of the sharp valley. It does not prove that the iterate will avoid every other poor region or reach a global minimum. That stronger conclusion requires the global structural assumptions of the paper's Theorem 1.

## 3. What the main theorem actually assumes

Theorem 1 constructs a smooth function class with a sharp local minimum $x^\dagger$ in a special region $M$ and a global minimum $x^\star$. Outside $M$, an *one-point strong-convexity-type* condition directs progress toward $x^\star$. Near $x^\dagger$, local curvature and separation assumptions permit a sufficiently large step to escape. Its point is a contrast:

$$\begin{aligned} \text{small step:}\quad & \text{convergence to } x^\dagger \text{ from a local basin,} \\ \text{large step:}\quad & \text{convergence to } x^\star \text{ under stronger global conditions.} \end{aligned}$$

This is not an all-purpose global convergence theorem for arbitrary neural networks. In particular, escaping a bad basin alone is logically weaker than proving eventual convergence to the best basin.

:::key Why the theorem is a genuine contribution
The authors isolate a trajectory-selection effect of *discretization*. Gradient flow and very small-step GD have different basin behavior from sufficiently large-step GD; stochastic noise cannot fully explain the effect.
:::

## 4. Evidence and interpretation

The paper's experiments include full-batch and stochastic training comparisons that vary learning rate while examining which solutions are reached. The deterministic observations support the argument that noise alone does not explain all advantages of large steps. Figures 1, 3 and 4 illustrate overshooting/avoidance of narrow valleys and the formal neighborhood construction.

**A subtle point:** If $\eta$ is excessively large, the entire algorithm may diverge. The paper advocates analyzing *intermediate* large-step regimes, not choosing the largest imaginable rate.

## 5. Proof structure and experimental interpretation

The local escape lemma uses sharp-basin curvature to show that a large enough step cannot remain nearby indefinitely. The *global* theorem adds structure outside that exceptional region, so excursions ultimately lead toward the global solution. These are logically separate: *repulsion from a bad basin* and *attraction toward the good solution*.

When interpreting plots, compare full-batch GD with stochastic training at matched step sizes. Evidence that full-batch GD benefits from large steps rules out an explanation requiring gradient sampling noise. It does *not* rule out additional SGD-noise effects in other settings or imply that arbitrarily large rates work.

## 5. Assumptions and limitations

**Assumptions:** A globally smooth function; one-point strong-convexity structure relative to $x^\star$ outside a special bad-minimum region; local curvature and geometric separation around $x^\dagger$; suitable step-size interval and initialization distribution.

**Limits:** The theorem's function class is highly structured. Full-batch neural-network experiments cannot establish the universal frequency of this effect, and large steps can cause divergence or poorer performance in other settings. Escaping a sharp minimum is not identical to escaping a strict saddle.

## 6. Connections to the survey

**Safran–Shamir (2018)** establishes that spurious minima can exist; large-step GD offers a possible dynamical way to avoid selected sharp ones. **Li–Wang–Li (2022)** emphasizes curvature oscillation, and **Chen–Bruna (2023)** studies period-two behavior instead of alternative basin selection.

## 7. Quick reference

| Concept | Meaning / equation |
|---|---|
| Local error multiplier | $e_{t+1} \approx (1 - \eta\kappa) e_t$ |
| Sharp-basin instability | $\eta\kappa > 2$ |
| Main contrast | small-step local vs. large-step global basin |
| Global theorem ingredient | one-point convexity outside bad region |
| Original to read | Sections 3–5; Lemma 1, Theorem 1 |

## 8. Papers Master note

:::plain Copy into Our notes
Shows large learning rates can change deterministic GD basin selection, in both experiments and a specially structured smooth nonconvex function class. Local repulsion from a sharp spurious minimum is not a general global-optimality guarantee.
:::

> **Original paper:** [View original research PDF](https://proceedings.mlr.press/v202/mohtashami23a/mohtashami23a.pdf) *Worked numerical examples are pedagogical unless explicitly attributed to the authors.*
