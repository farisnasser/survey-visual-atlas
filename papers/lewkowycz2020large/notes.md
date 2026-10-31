# The Catapult Mechanism at Large Learning Rates

:::minute The paper in one minute
**Problem:** Why can very large learning rates train deep networks successfully, sometimes better than small learning rates, even though the first steps may increase loss?

**Method:** Derive dynamics in a solvable class of wide networks and test predictions in neural-network experiments.

**Result:** A large-but-stable learning-rate regime can cause an initial loss *catapult*, accompanied by a decrease in curvature, followed by successful training in flatter regions. This is a model-dependent mechanism supported by experiments, not a universal guarantee.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. The tangent kernel is the curvature that the output feels

The paper's warm-up model is a linear network $f = n^{-1/2}v^\top u$ of width $n$, trained on one input $x = 1$ with label $0$, so the loss is $L = f^2/2$. One GD step on $u$ and $v$ changes the output to first order by $-\eta\lambda f$, where $\lambda = n^{-1}(\|u\|^2 + \|v\|^2)$ is the **neural tangent kernel** (here a single number). A fixed $\lambda$ would give $f_{t+1} = (1 - \eta\lambda)f_t$: stable for $\eta < 2/\lambda$, divergent above. With $n = 1$, $u = 2$ and $v = 0.35$, the start is $f_0 = 0.7$ and $\lambda_0 = 4.1225$, so this rule forbids any $\eta$ above $2/\lambda_0 \approx 0.485$.

### 2. The exact update lets the kernel move

The paper derives the exact updates for this model, with no terms dropped (its Eqs. (6)–(7)):

$$f_{t+1} = \Big(1 - \eta\lambda_t + \frac{\eta^2 f_t^2}{n}\Big)f_t, \qquad \lambda_{t+1} = \lambda_t + \frac{\eta f_t^2}{n}(\eta\lambda_t - 4).$$

Read the second equation. While $\eta\lambda_t < 4$ and $f_t \ne 0$, the kernel *decreases*. A larger output error makes it decrease faster. When $\eta\lambda_t > 4$ the kernel grows instead, and training blows up. This gives the paper's two thresholds, $2/\lambda_0$ and $4/\lambda_0$.

### 3. The catapult in numbers

Take the same start and $\eta\lambda_0 = 2.6$, so $\eta \approx 0.631$, above the fixed-kernel limit. The loss rises from $0.245$ to $0.484$ after one step. Meanwhile the kernel falls until $\lambda \approx 1.21$, below $2/\eta \approx 3.17$, and from then on the loss goes to zero. The run ends at a minimum with smaller $\lambda$, a flatter one. Just past the upper edge the story changes: with $f_0 = \lambda_0 = 1$ and $n = 10^4$, $\eta\lambda_0 = 3.98$ converges while $\eta\lambda_0 = 4.02$ diverges.

### 4. Width delays the catapult by about $\log n$ steps

The kernel change in Eq. (7) carries a factor $f_t^2/n$. In a wide network the kernel barely moves until the output error has grown large enough to compensate. Since the error grows geometrically, this takes a number of steps proportional to $\log n$. With $f_0 = \lambda_0 = 1$ and $\eta\lambda_0 = 3$, the loss peaks at step $2$, $6$ and $9$ for $n = 10^2$, $10^4$ and $10^6$. This is why the catapult is a short, early spike.

:::key Putting the four ideas together
A fixed kernel forbids steps above $2/\lambda_0$ (idea 1). In the exact dynamics the kernel shrinks whenever $\eta\lambda < 4$ (idea 2), so a step between the two thresholds first raises the loss and then lands in a flatter region where it is stable (idea 3), after a delay of order $\log n$ steps (idea 4). The notes below place this mechanism next to the fixed-quadratic intuition and its limits.
:::
:::

:::divider Study notes
:::

## 1. The standard intuition: too large a step should fail

A local quadratic model with curvature $\lambda > 0$ gives

$$e_{t+1} = (1 - \eta\lambda)e_t.$$

A fixed quadratic becomes unstable for $\eta > 2/\lambda$. If the network behaved like its linearization at initialization forever, then the *initial* curvature $\lambda_0$ would seem to determine which learning rates can work. But neural networks are not fixed quadratics: their tangent kernel and curvature evolve.

## 2. What actually changes during training?

For squared error on training outputs $f_t \in \mathbb{R}^n$, define residuals $r_t = f_t - y$ and the parameter Jacobian $J_t = \partial f_t/\partial\theta$. The empirical tangent-kernel matrix is

$$K_t = J_tJ_t^\top.$$

The usual linearized approximation holds $J_t$ fixed, giving *approximately*

$$r_{t+1} \approx (I - \eta K_0)r_t.$$

If $K_t$ evolves substantially, this fixed-kernel approximation ceases to describe the trajectory. The model in the paper uses this freedom: a large step may initially amplify the output error, but the resulting nonlinear changes can reduce the effective curvature so subsequent steps stabilize.

## 3. The three qualitative learning-rate phases

The authors distinguish a *lazy/small-rate regime*, a *catapult regime*, and a regime with rates so large that stable training fails. In a tractable model, important boundaries are expressed in terms of its initial kernel eigenvalue $\lambda_0$, with catapult behavior starting near $2/\lambda_0$; the upper edge of the stable interval *depends on the model*. Do not assume every architecture has the same upper threshold.

:::key Characteristic trajectory:
$$\underbrace{L_0 \to L_{\mathrm{spike}}}_{\text{transient loss increase}} \;\to\; \underbrace{K_t \text{ decreases}}_{\text{curvature changes}} \;\to\; \underbrace{L_t \text{ falls}}_{\text{stable training}}.$$

The paper reports that larger learning rates can select a flatter solution and sometimes achieve better test performance. The generalized statement "large learning rates always generalize better" is *not* supported.
:::

## 4. First-principles teaching analogy (not the exact paper model)

Take a scalar toy loss $L(w) = \frac12(1 - w^2)^2$. The curvature

$$L''(w) = -2 + 6w^2$$

changes when $w$ changes. Thus a step size incompatible with the *initial* curvature can eventually enter a region with different curvature. This toy merely explains how a moving objective curvature differs from a fixed quadratic; it is not the paper's derived catapult system.

## 5. Limitations and relation to other work

The exact analysis uses a specially solvable neural model and limiting/scaling assumptions; real-architecture evidence is experimental. The phenomenon also differs from Cohen et al.'s edge of stability: a catapult emphasizes an early transient loss spike and curvature reduction, whereas EoS describes sustained training near $2/\eta$. Both show why fixed-kernel, tiny-step reasoning misses important dynamics. **Our takeaway.** A learning rate is not just how far GD moves: a sufficiently large step can reshape the part of the loss landscape encountered during optimization.

> **Source:** [https://arxiv.org/pdf/2003.02218](https://arxiv.org/pdf/2003.02218) (Sections 2–5).
