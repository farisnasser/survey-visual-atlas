# Cubic Regularization with Momentum for Nonconvex Optimization

:::minute The paper in one minute
**Question:** Can momentum make cubic-regularized Newton practical faster without breaking its second-order convergence guarantee?

**Model/Method:** CRm: a cubic model step, an adaptive momentum extrapolation, then a monotone choice between candidates.

**Finding:** The method retains the worst-case $O(\varepsilon^{-3/2})$ iteration bound for approximate second-order stationarity and gains local quadratic convergence under an error-bound condition; experiments show acceleration.

**Why we care:** Momentum can be combined with curvature-aware updates, but empirical speed gains are distinct from improved worst-case rate exponents.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Momentum extrapolates along the last move, and can overshoot

A **momentum** step does not look at the landscape: it continues the most recent displacement, $v = y_{\text{new}} + \beta\,(y_{\text{new}} - y_{\text{old}})$. When successive steps keep pointing the same way, this saves iterations. On a nonconvex function it can also jump past a minimum. On $f(x) = -x^2/2 + x^4/4$, suppose the cubic step lands exactly on the minimizer $y_{\text{new}} = 1$, $f(1) = -0.25$, coming from $y_{\text{old}} = 0.5$, and $\beta = 0.5$. Then $v = 1.25$ and $f(1.25) \approx -0.171$, worse than staying. A rule that compares $f(y_{\text{new}})$ with $f(v)$ and keeps the smaller one never pays for such an overshoot; the notes show a case where the extrapolation wins.

### 2. Two kinds of speed: worst-case iteration counts and local quadratic convergence

A worst-case bound counts iterations until an $\varepsilon$-second-order stationary point: $\|\nabla f\| \le \varepsilon$ and $\lambda_{\min}(\nabla^2 f) \ge -\sqrt{\varepsilon}$. Gradient descent needs order $\varepsilon^{-2}$ iterations even for the first condition; cubic regularization needs order $\varepsilon^{-3/2}$. For $\varepsilon = 10^{-3}$ that is about $10^6$ versus $3.2 \times 10^4$ (constants omitted).

**Quadratic convergence** is a local statement: once close, the error is squared at every step, $e_{k+1} \le C e_k^2$. With $C = 1$ and $e_0 = 0.1$, the errors are $10^{-2}, 10^{-4}, 10^{-8}$: the number of correct digits doubles each step.

### 3. An error bound ties distance to the gradient

Quadratic convergence needs some regularity near the solutions. A **local error bound** asks $\mathrm{dist}(x, X) \le \kappa\|\nabla f(x)\|$ near the solution set $X$: a small gradient must mean being close. For $f(x) = (x^2 - 1)^2/4$ near $x = 1$, $f'(x) = x(x^2 - 1) \approx 2(x - 1)$, so the bound holds with $\kappa$ about $\tfrac12$ (at $x = 1.01$: distance $0.01$, gradient $\approx 0.0203$). For $f(x) = x^4$ at $0$ it fails: at $x = 0.1$ the distance is $0.1$ but the gradient is only $0.004$, and the ratio grows without bound as $x \to 0$.

:::key Putting the three ideas together
CRm adds a momentum candidate to each cubic step (idea 1) and keeps whichever candidate has the lower objective, so the cubic step's guarantee is never lost. That preserves the $O(\varepsilon^{-3/2})$ worst-case order of cubic regularization, not a better exponent (idea 2), and under a local error bound (idea 3) the iterates still converge quadratically. The notes below give the algorithm and the theorems.
:::
:::

:::divider Study notes
:::

## 1. Start with cubic-regularized Newton

At a current point $x_k$, form a local model of an objective $f$:

$$m_k(s) = \nabla f(x_k)^T s + \frac12 s^T \nabla^2 f(x_k)\, s + \frac{M}{6}\|s\|^3.$$

The quadratic part captures gradient and curvature; the positive cubic term restrains the step when the Hessian is indefinite. Standard cubic regularization (CR) chooses a global minimizer $s_{k+1}$ and sets $y_{k+1} = x_k + s_{k+1}$. This can exploit negative curvature at saddles, but each cubic step may be computationally expensive. Wang et al. ask whether adding momentum after such a step improves progress while retaining guarantees.

### Worked example: how the cubic term escapes a saddle

Let $f(x) = -x^2/2 + x^4/4$, so $f'(0) = 0$, $f''(0) = -1$. At $x_0 = 0$, ordinary GD has update zero and cannot move. The local cubic model with $M = 2$ is

$$m_0(s) = -\tfrac12 s^2 + \tfrac13|s|^3, \qquad m_0'(s) = -s + s|s|.$$

Its minima occur at $s = \pm 1$; choosing $s = 1$ takes us to $y_1 = 1$ where $f(1) = -1/4$, lower than $f(0) = 0$. The method uses negative curvature, even though the gradient vanishes. This is a toy illustration, not an experiment reported by the authors.

:::key A useful distinction
CRm's saddle escape comes from its cubic second-order model. Momentum is intended to accelerate the resulting sequence; it is not by itself a generic guarantee against saddle points.
:::

## 2. The three steps of CRm (Algorithm 1)

**Step 1: Cubic model.** Solve the cubic subproblem to obtain $y_{k+1} = x_k + s_{k+1}$. **Step 2: Adaptive momentum.** Let $y_k$ be the previous cubic candidate and choose

$$\beta_{k+1} = \min\{\rho, \|\nabla f(y_{k+1})\|, \|y_{k+1} - x_k\|\}, \qquad 0 < \rho < 1,$$

$$v_{k+1} = y_{k+1} + \beta_{k+1}(y_{k+1} - y_k).$$

If the current point is still far from stationarity, a larger motion is allowed. Near a stationary point both gradient norm and step norm shrink, so the extrapolation becomes small. **Step 3: Monotone safeguard.** Accept whichever candidate has less actual objective value:

$$x_{k+1} = \operatorname*{argmin}_{z \in \{y_{k+1},\, v_{k+1}\}} f(z).$$

The safeguard is essential: arbitrary momentum may increase a nonconvex objective and invalidate a descent argument.

### A tiny numerical check of the safeguard

Suppose the cubic step proposes $y_{k+1} = 0.8$, the previous cubic candidate was $y_k = 0.5$, and the adaptive coefficient is $\beta_{k+1} = 0.2$. Then $v_{k+1} = 0.8 + 0.2(0.3) = 0.86$. On $f(x) = -x^2/2 + x^4/4$:

$$f(0.8) = -0.2176, \qquad f(0.86) \approx -0.2330.$$

The method would accept $0.86$, which has a lower objective. If extrapolation had a higher loss, it would instead accept $0.8$. The values illustrate Algorithm 1's choice, not its parameter-setting theorem.

## 3. What is actually proved

**Theorem 1:** Under Assumption 1 and suitable $M$, CRm finds an $\varepsilon$-second-order stationary point in $O(\varepsilon^{-3/2})$ iterations. The relevant conditions are $\|\nabla f(x)\| \le \varepsilon$ and smallest Hessian eigenvalue bounded below on the order of $-\sqrt{\varepsilon}$. **Theorem 2:** Under an additional local error bound $\mathrm{dist}(x, X) \le \kappa\|\nabla f(x)\|$ near the stationary-point set $X$, the iterates converge locally at a quadratic rate. **Theorem 3:** A carefully inexact cubic subproblem (inexact Hessian, Assumption 3) still attains the same $O(\varepsilon^{-3/2})$ outer iteration order.

## 4. Assumptions, limitations and contributions

Assumption 1 requires $f$ bounded below, bounded sublevel sets, and Lipschitz gradient/Hessian. The local quadratic theorem additionally needs the error-bound property, which is weaker than global strong convexity but does not hold for every nonconvex objective. The inexact version imposes explicit Hessian-accuracy conditions.

- **Contribution:** A carefully safeguarded momentum variant of cubic regularization with its global-order convergence guarantee preserved.
- **Empirical finding:** The paper's experiments report substantially faster decrease in objective value than conventional cubic regularization on tested problems.
- **Not proved:** A strictly better worst-case $\varepsilon$ exponent; global minima in arbitrary neural networks; negligible cost per iteration. Cubic model minimization can remain expensive.

### Survey connections

Carmon–Duchi (Ref 76) shows GD can solve cubic subproblems to useful accuracy: one route to making each CR step cheaper. Dauphin (2014) also uses Hessian curvature to address saddles, but with a different update. Galli/Kalra (2025–26) treat learning-rate and sharpness dynamics, not cubic Newton's local model. This paper fits Section V as a curvature-based intervention with a momentum extension.

### Quick reference

| Term | Meaning |
|---|---|
| $M\lVert s\rVert^3/6$ | Stabilizes an indefinite quadratic Taylor model. |
| $\beta_{k+1}$ | Adaptive momentum, limited by the gradient norm and cubic step length. |
| Monotone choice | Compare $f(y_{k+1})$ and $f(v_{k+1})$ before accepting. |
| $O(\varepsilon^{-3/2})$ | Worst-case outer iteration order, not total wall-clock runtime. |
| Local error bound | Converts control of gradient norm into distance from the solution set. |

:::plain Papers Master note
Wang et al. extend cubic-regularized Newton with adaptive momentum and a monotone safeguard. The method retains $O(\varepsilon^{-3/2})$ second-order-stationarity iteration complexity, gains local quadratic convergence under an error bound, and improves empirical optimization speed; cubic steps remain expensive.
:::

> **Source:** original UAI 2020 paper, Algorithm 1; Theorems 1–3. Wang, Zhou, Liang & Lan (2020), *Cubic Regularization with Momentum for Nonconvex Optimization*, UAI 2020 (PMLR 115). [proceedings.mlr.press/v115/wang20c/wang20c.pdf](https://proceedings.mlr.press/v115/wang20c/wang20c.pdf)
