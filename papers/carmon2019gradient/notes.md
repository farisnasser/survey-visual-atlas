# Gradient Descent Finds the Cubic-Regularized Nonconvex Newton Step

:::minute The paper in one minute
**Question:** Can plain gradient descent solve the nonconvex cubic-regularized quadratic subproblem used inside Newton-type methods?

**Model/Method:** Study a possibly indefinite quadratic plus a positive cubic norm term; analyze GD in the eigenbasis of its Hessian.

**Finding:** Under specified initialization and step-size conditions, GD approaches the *global* subproblem optimum at a quantified rate; a small random perturbation handles a hard alignment case.

**Why we care:** A second-order saddle-escaping Newton step can itself be approximated with simple matrix-vector/gradient computations.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. The global minimizer of the cubic model has a one-number certificate

For $q(s) = b^\top s + \tfrac12 s^\top A s + \tfrac{\rho}{3}\|s\|^3$, the gradient is $As + b + \rho\|s\|s$. A point $s^*$ is the **global** minimizer exactly when

$$(A + \rho\|s^*\|\,I)\, s^* = -b \qquad\text{and}\qquad A + \rho\|s^*\|\,I \succeq 0 .$$

The first condition is $\nabla q = 0$; the second says the shift $\rho\|s^*\|$ is at least $-\lambda_{\min}(A)$. Take $A = \mathrm{diag}(-1, 2)$, $b = (0.1, 0.2)$, $\rho = 1$. Solving the scalar equation $\lambda = \rho\|(A + \lambda I)^{-1}b\|$ with $\lambda \ge 1$ gives $\lambda^* \approx 1.092$ and $s^* \approx (-1.090, -0.065)$, with $q(s^*) \approx -0.278$. Other critical points exist (the model is nonconvex), but only this one passes both tests.

### 2. Gradient descent needs a step matched to the curvature it will meet

The Hessian of $q$ grows with $\|s\|$, so a safe step depends on how far GD can travel. The paper bounds $\|s^*\| \le R = \tfrac{\beta}{2\rho} + \sqrt{(\tfrac{\beta}{2\rho})^2 + \tfrac{\|b\|}{\rho}}$ with $\beta = \|A\|$, and asks for $\eta \le 1/(4(\beta + \rho R))$ (its Assumption A), starting from $s_0 = -r\,b/\|b\|$ (Assumption B). For the example above, $\beta = 2$, $R \approx 2.106$ and $\eta \le 0.061$; GD from $s_0 = 0$ with that step converges to $s^*$. (The notes' one-dimensional example uses $\eta = 0.25$, above that example's bound $0.125$; it still converges there, and the notes label it pedagogical.)

### 3. If $b$ misses the most negative direction, GD never finds it

In the eigenbasis, GD updates each coordinate as $s^{(i)} \leftarrow (1 - \eta\lambda_i - \eta\rho\|s\|)\,s^{(i)} - \eta b^{(i)}$. If $b^{(1)} = 0$ for the most negative eigenvalue and $s_0$ is a multiple of $b$, coordinate 1 starts at $0$ and stays at $0$ forever. With $b = (0, 0.2)$ in the example, GD stops at $(0, -0.095)$ with $q \approx -0.0097$, while the true minimum is about $-0.173$. This is the **hard case**; a tiny random change to $b$ gives coordinate 1 something to grow from.

:::key Putting the three ideas together
The cubic subproblem's global solution is certified by a shift condition (idea 1). With a step sized to the curvature in the ball where $s^*$ lives and a start along $-b$, GD's negative-curvature component grows and the iterates settle on that global solution (idea 2), unless $b$ has no component along the most negative direction (idea 3). The notes below state the rates and the randomized fix.
:::
:::

:::divider Study notes
:::

## 1. Start with the task Newton's method leaves unsolved

A standard Newton step minimizes a quadratic Taylor approximation. When the Hessian has negative eigenvalues, that quadratic is unbounded below: move far enough along a negative-curvature direction and its value keeps decreasing. Cubic regularization adds a stabilizing term to make the subproblem bounded below:

$$q(s) = b^\top s + \frac12 s^\top A s + \frac{\rho}{3}\|s\|^3, \qquad A = A^\top, \ \rho > 0.$$

Here $b$ acts like the gradient of the original objective, $A$ its Hessian and $s$ the proposed update. Although the cubic term keeps $q$ from decreasing without bound, $q$ is generally *still nonconvex*. The question is why the cheapest possible solver—GD—can nevertheless find its global optimum.

### Worked example: a scalar negative-curvature model

Choose $A = -1$, $b = 0$, $\rho = 1$, with a single coordinate $s$:

$$q(s) = -\tfrac12 s^2 + \tfrac13|s|^3, \qquad q'(s) = -s + s|s|.$$

Stationary points are $s = 0, \pm 1$. The origin is a local *maximum* ($q''(0) = -1$), whereas $q(\pm 1) = -1/6 < q(0) = 0$. Run GD with $\eta = 0.25$ from $s_0 = 0.2$:

$$s_1 = 0.2 - 0.25(-0.2 + 0.04) = 0.24, \qquad s_2 = 0.24 - 0.25(-0.24 + 0.0576) = 0.2856.$$

The iterate moves away from the negative-curvature stationary point and toward $s = +1$. Starting *exactly* at zero would leave GD stuck; this elementary case foreshadows why initialization matters.

:::warn Important precision
This is a pedagogical one-dimensional example, not proof of the full theorem. The general problem may have many dimensions, unequal eigenvalues, saddle points and a bad local minimum.
:::

## 2. Why the eigenvectors explain gradient-descent dynamics

Differentiate the actual cubic objective. Since $\nabla(\|s\|^3/3) = \|s\|s$, the full update is

$$\nabla q(s) = As + b + \rho\|s\|s, \qquad s_{t+1} = s_t - \eta(As_t + b + \rho\|s_t\|s_t).$$

Diagonalize $A = V\Lambda V^\top$. In eigen-coordinate $i$, this gives the recurrence

$$s^{(i)}_{t+1} = \big(1 - \eta\lambda_i(A) - \eta\rho\|s_t\|\big)\,s^{(i)}_t - \eta b^{(i)}.$$

A negative $\lambda_i$ initially *amplifies* the corresponding component. Once $\|s_t\|$ grows enough, the cubic term changes the multiplier so the trajectory can contract toward the global minimizer. The paper's proof separates these two stages: growth in the most negative eigendirection, then convergence in the regularized region.

### What Theorems 3.1 and 3.2 establish

**Theorem 3.1 (deterministic).** For a sufficiently small fixed step, the prescribed initialization aligned with $-b$, and nonzero projection of $b$ onto a smallest-eigenvalue direction, GD reaches objective error $q(s_t) - q(s_*) \le \varepsilon$ after an explicit number of steps. The analysis exhibits different accuracy/conditioning regimes: roughly $\widetilde{O}(1/\varepsilon)$ in a difficult regime, an intermediate $1/\sqrt{\varepsilon}$ behavior, and logarithmic dependence on $1/\varepsilon$ in the favorable local regime. **Theorem 3.2 (randomized hard case).** If $b$ is orthogonal to the most-negative eigenspace, the deterministic alignment argument can fail. Slightly perturbing $b$ before running GD makes the needed projection nonzero with high probability, at a cost involving logarithms of the dimension and failure probability.

### A useful contrast with ordinary Newton's method

Ordinary Newton solves $As = -b$, which is not a minimization guarantee when $A$ is indefinite. Cubic regularization instead minimizes a *stabilized* nonlinear model. Crucially, this paper studies a solver for *that inner model*, not convergence of unconstrained GD on every original nonconvex objective.

:::plain What to read
Original Sections 2–3 (assumptions, Proposition 2.5, Theorems 3.1–3.2); Section 6 for the outer-method application; Section 5 only for line-search nuances. Proof Sections 4 and appendix can be skimmed for the two-stage argument.
:::

## 3. Contribution, qualifications, and survey fit

### What is new

- A finite-iteration analysis of plain GD on an indefinite quadratic plus cubic regularizer, yielding a surprisingly strong global subproblem guarantee.
- A treatment of the "hard case," in which the linear term has no component in a most-negative eigendirection, via a randomized perturbation.
- An outer cubic-Newton application using Hessian-vector products rather than forming or inverting a dense Hessian.

### Assumptions and limitations

The primary analysis uses symmetric $A$, $\rho > 0$, carefully limited constant step size (Assumption A), and special initialization $s_0 = -rb/\|b\|$ (Assumption B); randomized results modify the hard case. The outer optimization application requires regularity such as a Lipschitz Hessian. None of this proves that arbitrary initialization, arbitrary learning rates or ordinary neural-network GD always finds global minima. The subproblem can have spurious local minima even though its special GD trajectory reaches a global solution.

### Cross-paper synthesis

Dauphin (2014) also uses curvature to motivate saddle-aware optimization; Carmon–Duchi focuses on efficiently solving a stabilizing cubic model. Wang et al. (2020), Ref 77 adds momentum to the outer cubic-regularization method, not to precisely this convergence proof. Jin (2017) uses gradient perturbations directly on the objective instead of solving cubic subproblems.

### Quick reference

| Object | Meaning |
|---|---|
| $\frac12 s^T A s + b^T s + \frac{\rho}{3}\lVert s\rVert^3$ | Cubic-regularized Newton model: second-order Taylor terms plus stabilizer. |
| $As + b + \rho\lVert s\rVert s$ | Gradient used in inexpensive inner GD steps. |
| $\lambda_{\min}(A) < 0$ | Negative curvature makes the unregularized quadratic nonconvex. |
| $\varepsilon$-accurate step | Objective gap to global subproblem optimum, not merely small gradient. |
| Hard case | Linear term orthogonal to the least-eigenvalue eigenspace; perturbation resolves alignment. |

:::plain Papers Master note
Carmon–Duchi show that GD, under controlled initialization/step sizes, globally solves cubic-regularized indefinite quadratic subproblems to stated accuracy. Random perturbation treats the hard alignment case. This provides a matrix-free inner solver for second-order saddle-escaping methods, not a global neural-network GD theorem.
:::

> **Source:** original full text, Sections 2–3 and 6; SIAM J. Optim. 29(3), 2019. Carmon & Duchi, *Gradient Descent Finds the Cubic-Regularized Nonconvex Newton Step* (preprint [arXiv:1612.00547](https://arxiv.org/abs/1612.00547)).
