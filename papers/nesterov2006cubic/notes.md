# Cubic Regularization of Newton Method

:::minute The paper in one minute
**Problem.** If $H$ is indefinite, then $s^THs$ can be negative along a direction with negative eigenvalue, making the unconstrained quadratic model unbounded below. A standard Newton step $-H^{-1}g$ is therefore not automatically a descent or saddle-escape method.

**Mechanism.** Regularize by a cubic norm: unlike an indefinite quadratic, the cubic term grows positively as $\|s\| \to \infty$; the model has a global minimizer. A globally minimizing step naturally takes account of negative curvature rather than treating it as a numerical error.

**Documented result.** The original article establishes global worst-case complexity results, across classes including non-convex objectives, and studies local convergence of cubic-regularized Newton steps.

**Trade-off.** A harder subproblem and Hessian-related cost per iteration.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Newton's method jumps to the nearest critical point, saddle or not

Newton's method minimizes the quadratic model $f(x) + g^Ts + \tfrac12 s^THs$ by setting its gradient to zero: $s = -H^{-1}g$. On a quadratic it lands exactly on the critical point in one step, whatever kind of point that is. Take $f(x, y) = x^2 - y^2$, a saddle at the origin, and start at $(1, 0.5)$: $g = (2, -1)$, $H = \mathrm{diag}(2, -2)$, so $s = (-1, -0.5)$ and the next iterate is $(0, 0)$, the saddle. Along $y$ the step went *uphill*. The paper's introduction lists exactly this failure: the Newton scheme may converge to a saddle point or even a local maximum.

### 2. A Lipschitz Hessian puts a cubic cap on the Taylor error

If the Hessian changes by at most $M\|s\|$ over a step $s$, the quadratic Taylor model is wrong by at most $\tfrac{M}{6}\|s\|^3$, so

$$f(x + s) \le f(x) + g^Ts + \tfrac12 s^THs + \tfrac{M}{6}\|s\|^3.$$

The bound is tight. For $f(x) = x^3/6$ the second derivative is $x$, Lipschitz with $M = 1$; at $x = 0$ the quadratic model is identically $0$, and at $s = 2$ the true value $8/6 \approx 1.33$ equals the cap $\tfrac16 \cdot 2^3$. Adding the cubic term turns the local model into an **upper bound** on $f$: minimizing it can only lower $f$.

### 3. At a saddle the cubic model still says where to go

Take one direction where the gradient is zero and the curvature is negative: $g = 0$, $H = -1$, $M = 1$. Newton's step is $s = 0$: it stays put. The cubic model $m(s) = -\tfrac12 s^2 + \tfrac16|s|^3$ is minimized where $-s + \tfrac12|s|s = 0$, at $|s| = 2$, with $m = -2 + \tfrac43 \approx -0.67$. By idea 2, $f$ drops by at least $0.67$. The cubic step moves away from the saddle along the negative-curvature direction without any random noise, and its length, $2|H|/M$, grows with how negative the curvature is.

:::key Putting the three ideas together
Plain Newton can be pulled into saddles because its quadratic model has no floor (idea 1). A Lipschitz Hessian lets us add a cubic term that makes the model an upper bound on the true function (idea 2), and minimizing that bound moves downhill even where the gradient vanishes (idea 3). The notes below write the model and its first-order condition, and say what the paper proves.
:::
:::

:::divider Study notes
:::

:::warn Source blocker
The original full article/proofs were not available for full inspection in this session. Bibliography, abstract-level contribution and the canonical cubic model are source-backed; the explanatory derivations below are independently checked mathematical background, *not* a reconstruction of the paper's exact proof or all its constants. Do not mark the complete paper verified.
:::

## 1. Why ordinary Newton steps are troublesome at saddles

Let $g = \nabla f(x)$ and $H = \nabla^2 f(x)$. Newton's second-order model is

$$m_N(s) = f(x) + g^Ts + \frac12 s^THs.$$

If $H$ is indefinite, then $s^THs$ can be negative along a direction with negative eigenvalue, making the unconstrained quadratic model unbounded below. A standard Newton step $-H^{-1}g$ is therefore not automatically a descent or saddle-escape method.

## 2. The proposed mechanism: regularize by a cubic norm

Under a Hessian-Lipschitz condition

$$\|\nabla^2 f(x + s) - \nabla^2 f(x)\| \le M\|s\|,$$

Taylor's theorem bounds the remainder by $M\|s\|^3/6$. This suggests the *cubic model*

$$\boxed{m_x(s) = f(x) + g^Ts + \tfrac12 s^THs + \tfrac{M}{6}\|s\|^3.}$$

Unlike an indefinite quadratic, the cubic term grows positively as $\|s\| \to \infty$; the model has a global minimizer. A globally minimizing step naturally takes account of negative curvature rather than treating it as a numerical error.

## 3. From the model to an update

For $s \ne 0$, the first-order condition for the model is

$$0 = \nabla_s m_x(s) = g + Hs + \frac{M}{2}\|s\|s.$$

Thus the step solves a shifted Newton-like system

$$\left(H + \frac{M\|s\|}{2}I\right)s = -g.$$

The shift depends on the resulting step length. This is why the step is *not* simply Newton's update with a fixed regularizer. Finding a global solution of the subproblem also requires more than merely solving the first-order equation.

:::theorem Documented result
The original article establishes global worst-case complexity results, across classes including non-convex objectives, and studies local convergence of cubic-regularized Newton steps. Standard modern interpretations connect this framework to approximate second-order stationarity, with rates depending on the exact tolerance definition and oracle model.
:::

## 4. What can and cannot be concluded

The core analytical premise is regularity of the Hessian and the ability to solve or accurately approximate cubic subproblems. Counting *outer iterations* is different from counting Hessian calculations, linear algebra operations or wall-clock time. The method does not say that arbitrary nonsmooth ReLU networks are globally optimized or that a second-order stationary point is necessarily the global optimum.

:::plain Survey relevance (RQ3)
The method is a foundational second-order escape framework: modify the local quadratic so negative curvature does not destroy the model and analyze global complexity. Any theorem-numbered claims or exact constants should be checked after acquiring the full source.
:::

## 5. Our notes

The cubic term prevents the indefinite Newton model from running off to minus infinity and makes curvature-aware steps mathematically manageable. The trade-off is a harder subproblem and Hessian-related cost per iteration. Source access is incomplete, so detailed original proofs are not asserted here.

> **Original bibliographic record:** [https://doi.org/10.1007/s10107-006-0706-8](https://doi.org/10.1007/s10107-006-0706-8).
