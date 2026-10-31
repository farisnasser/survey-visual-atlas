# Beyond the Edge of Stability via Two-step Gradient Updates

:::minute The paper in one minute
**Question:** Why can GD with a step size beyond the usual stability threshold remain bounded and settle into oscillatory training?

**Model/Method:** Analyze the *two-step* GD map, higher derivatives around minima, quartic toy losses, teacher–student networks and matrix factorization.

**Finding:** Under explicit local derivative conditions, the two-step map can have stable fixed points representing a period-two GD orbit; selected models converge to these cycles.

**Why we care:** Convergence of *two-step behavior* need not mean that individual iterates settle at a stationary point.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A minimum is a fixed point of the GD map, with a multiplier

One GD step is a function of the current point, $F_\eta(x) = x - \eta f'(x)$. A minimizer $x^*$ is a **fixed point**: $f'(x^*) = 0$, so $F_\eta(x^*) = x^*$. Whether nearby points are pulled in depends on the derivative of the map there, the **multiplier** $F_\eta'(x^*) = 1 - \eta f''(x^*)$. A small error $e$ becomes about $F_\eta'(x^*)\, e$ after one step, so the fixed point attracts when $|1 - \eta f''(x^*)| < 1$ and repels when it is larger.

For the quartic $f(x) = \tfrac14 (x^2 - 1)^2$ used in the notes, $f''(1) = 2$. At $\eta = 1.05$ the multiplier is $1 - 2.1 = -1.1$: an error of $0.01$ becomes about $-0.011$, then about $0.0121$. The minimum itself is unstable.

### 2. Away from the minimum the curvature changes

A quadratic has the same curvature everywhere. A general loss does not: by Taylor's theorem, $f''(x) \approx f''(x^*) + f'''(x^*)(x - x^*)$, so the third derivative measures how fast the curvature changes as $x$ crosses the minimum. For the quartic, $f''(x) = 3x^2 - 1$ is $0.92$ at $x = 0.8$ but $2.646$ at $x = 1.1024$. A step taken on the flat side moves the point much less than one taken on the steep side. That asymmetry is what the paper's derivative conditions measure.

### 3. "Converging to a cycle" means the even and odd iterates converge separately

GD can stay bounded without converging. If $F_\eta(a) = b$ and $F_\eta(b) = a$ with $a \ne b$, the iterates alternate $a, b, a, b, \dots$: a **period-two orbit**. Run GD on the quartic with $\eta = 1.05$ from $x_0 = 0.8$ and the even and odd iterates settle on two different numbers, about $0.8729$ and $1.0911$, one on each side of the minimum. The loss alternates between about $0.0142$ and $0.0091$; it never reaches $0$. Over two steps the map $F_\eta \circ F_\eta$ multiplies a small error by about $0.595$, so this orbit attracts even though the minimum repels. The notes give the product formula behind that number.

### 4. Larger steps can break the cycle in turn

The same story can repeat. As $\eta$ grows, the two-step multiplier of the orbit itself eventually passes $-1$, the 2-cycle becomes unstable and a 4-cycle appears; then 8, and eventually chaotic motion. This cascade is called **period doubling**. In our simulation of the quartic ($\mu = 1$) the 2-cycle lasts until about $\eta \approx 1.237$, beyond the range the paper's Theorem 2 guarantees ($\eta < 1.121$): that range is sufficient, not the exact edge.

:::key Putting the four ideas together
Beyond $\eta = 2/f''(x^*)$ the minimum stops attracting GD (idea 1). If the curvature differs on the two sides of the minimum (idea 2), GD can lock into a bounce between two points, a period-two orbit that attracts over two steps (idea 3), until even larger steps break it too (idea 4). The paper turns the condition on $f'''$ and $f''''$ into theorems for the quartic and related models, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. What changes beyond the edge of stability?

A standard quadratic analysis says the minimizer becomes unstable when $\eta f''(x^*) > 2$. But a nonlinear loss can flatten on either side of its minimizer. Then GD can bounce across it, with the curvature on successive sides compensating. A trajectory may approach a *two-cycle*, moving $a \to b \to a$, rather than a single parameter value.

Write one step as $F_\eta(x) = x - \eta f'(x)$; two steps give $F^2_\eta(x) = F_\eta(F_\eta(x))$. If $F^2_\eta(a) = a$ but $F_\eta(a) = b \ne a$, the original GD iteration oscillates forever between $a$ and $b$. That is not the same as divergence or convergence to the minimizer.

## 2. Worked example: a quartic with a period-two opportunity

Let

$$f(x) = \tfrac14 (x^2 - 1)^2, \qquad f'(x) = x(x^2 - 1), \qquad f''(x) = 3x^2 - 1.$$

At the minimum $x^* = 1$, $f''(1) = 2$, so the quadratic edge is $\eta = 2/2 = 1$. Choose $\eta = 1.05$ and $x_0 = 0.8$. Then

$$x_1 = 0.8 - 1.05\,[0.8(0.64 - 1)] = 1.1024.$$

The next step is $x_2 = 1.1024 - 1.05[1.1024(1.1024^2 - 1)] \approx 0.8532$. The iterates cross the minimizer, but the unequal curvatures on the two sides need not make the cycle explode.

*Important:* Two displayed iterates illustrate the nonlinear bounce; they do *not* by themselves prove a stable orbit. Theorem 2 supplies a rigorous period-two convergence result for this quartic family in a stated step-size range.

## 3. Why study the second iterate?

For a general one-dimensional smooth loss with a local minimizer $\bar x$, Theorem 1 identifies the higher-order condition

$$f^{(3)}(\bar x) \ne 0, \qquad 3\big[f^{(3)}(\bar x)\big]^2 - f''(\bar x)\, f^{(4)}(\bar x) > 0.$$

Under its local hypotheses and a suitable step slightly *above* $2/f''(\bar x)$, two-step fixed points can exist. The derivative conditions measure how the curvature *changes* as the iterate crosses the minimum; constant-curvature quadratics have no such stabilizing mechanism.

:::key Check the worked quartic
For $f(x) = \tfrac14 (x^2 - 1)^2$, at $x = 1$ we have $f'' = 2$, $f^{(3)} = 6$, $f^{(4)} = 6$, so $3(6)^2 - 2(6) = 96 > 0$. The quartic therefore has the right nonlinear geometry for the condition. Theorem 2 is the paper's stronger convergence statement for this model.
:::

## 4. Results to distinguish

**Theorem 1:** existence of fixed points of the two-step map under derivative conditions in 1-D.

**Theorem 2:** for a special quartic $\tfrac14 (x^2 - \mu)^2$, $\eta = K/\mu$ with $1 < K < \sqrt{4.5} - 1 \approx 1.121$, and specified initialization, GD converges to a period-two orbit (apart from exceptional starts).

**Theorem 3 / Proposition 3:** a related bilinear teacher–student loss, $\tfrac12 (xy - \mu)^2$, can reduce toward a balanced situation and display such a cycle under additional assumptions.

**Further evidence:** Results in one-neuron models and matrix factorization show oscillatory behavior in more dimensions, but not a universal theorem for deep neural networks.

## 5. Stability of the two-step orbit

A useful calculation is

$$(F^2_\eta)'(a) = F_\eta'(b)\, F_\eta'(a) = \big[1 - \eta f''(b)\big]\big[1 - \eta f''(a)\big], \qquad b = F_\eta(a).$$

The pair $(a, b)$ can be locally attractive when the absolute value of this *product* is below one, even if one step expands. That is why nonlinear curvature on both sides of a minimizer matters. For a pure quadratic, $f''(a) = f''(b) = \kappa$ and the product becomes $(1 - \eta\kappa)^2 > 1$ beyond EoS; there is no compensation.

The results in Sections 5–6 apply this idea to specific neural and matrix-factorization settings; their hypotheses are not identical to the one-dimensional quartic theorem.

## 5. Assumptions and limitations

**Assumptions:** Deterministic fixed-step GD, smooth local derivatives, specific higher-order curvature signs, special initialization and learning-rate intervals for the global quartic theorem; population-loss assumptions in the teacher–student examples.

**Limitations:** A stable two-cycle is not convergence of parameters to a minimizer. Proofs in low-dimensional examples do not imply that arbitrary networks exhibit stable oscillation above EoS; higher-dimensional observations are more exploratory. Some results are existence statements rather than attraction-basin characterizations.

## 6. Connections to the survey

**Li–Wang–Li (2022)** analyzes how sharpness moves around EoS; this paper asks what orbit GD approaches beyond it. **Kalra et al. (2025)** connects increased learning rates to period doubling and chaos in a simplified UV model. **Galli (2026)** studies adaptive nonmonotone steps but proves a different first-order guarantee.

## 7. Quick reference

| Concept | Meaning / equation |
|---|---|
| One-step map | $F_\eta(x) = x - \eta f'(x)$ |
| Two-step fixed point | $F^2_\eta(a) = a$, but $F_\eta(a) \ne a$ |
| Local condition | $3(f^{(3)})^2 - f'' f^{(4)} > 0$ |
| Quartic condition | $1 < K < \sqrt{4.5} - 1$ |
| Original to read | Sections 4–6; Theorems 1–3 |

## 8. Papers Master note

:::plain Copy into Our notes
Proves existence of stable period-two GD behavior beyond classical quadratic EoS in selected smooth low-dimensional models, using two-step maps and higher derivatives. Do not equate two-cycle convergence with parameter convergence or universal deep-network stability.
:::

> **Original paper:** [View original research PDF](https://proceedings.mlr.press/v202/chen23b/chen23b.pdf) *Worked numerical examples are pedagogical unless explicitly attributed to the authors.*
