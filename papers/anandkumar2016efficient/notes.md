# Escaping Higher-Order Saddle Points

:::minute The paper in one minute
**Question:** Perturbed gradient descent and negative-curvature methods ask whether the minimum Hessian eigenvalue is significantly negative. But for a degenerate saddle, even the Hessian can be positive semidefinite. The authors ask whether third derivatives provide a principled escape method, and whether the same idea can continue indefinitely.

**Paper result:** There is a provable algorithm for reaching third-order local optimality, but no comparable unrestricted polynomial-time method for fourth-order local optimality unless standard complexity assumptions fail.

**Scope:** The NP-hardness claim concerns general fourth-order local optimality, not the impossibility of training all deep neural networks. Structured neural-network degeneracies may be more tractable.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Local optimality comes in orders

The paper (Definition 1) calls $x$ a **$p$-th order local minimum** if, near $x$, $f(y) \ge f(x) - C\|y - x\|^{p+1}$ for some constant $C$: any decrease must be smaller than order $p + 1$ in the distance. Example: $h(x, y) = x^2 + y^3$ at the origin. Along $y = -t$ it equals $-t^3$, which satisfies $-t^3 \ge -C t^3$ with $C = 1$, so the origin passes the second-order test. It fails the third-order test, which would need $-t^3 \ge -C t^4$ for all small $t$: at $t = 0.1$ that already requires $C \ge 10$, and as $t \to 0$ no constant works.

### 2. Higher derivatives are directional derivatives of higher order

The third-derivative tensor applied to a direction $u$ is the third derivative along a line,

$$\nabla^3 f(x)[u, u, u] = \frac{d^3}{dt^3} f(x + tu)\Big|_{t=0} .$$

For $h = x^2 + y^3$ at the origin it is $+6$ along $u = (0, 1)$ and $-6$ along $u = (0, -1)$. An odd-order term changes sign with $u$, so whenever it is nonzero along a flat direction, one of $\pm u$ goes down. That is why the paper's third-order condition asks the cubic term to vanish on the Hessian's null space, and why it can be checked and acted on efficiently.

### 3. At fourth order the sign no longer flips, and checking becomes hard

A fourth-order term $q(u) = \nabla^4 f(x)[u, u, u, u]$ is even: $q(-u) = q(u)$. Deciding descent now means deciding whether a quartic form can be negative somewhere, which is a genuine search, not a sign check. For two variables it is easy to see: $(u_1^2 - u_2^2)^2$ is never negative, while $u_1^4 + u_2^4 - 3u_1^2u_2^2$ equals $-1$ at $u = (1, 1)$. In many variables, deciding whether a quartic form is nonnegative is NP-hard, and the paper reduces fourth-order local optimality to that question.

### 4. Deep linear networks produce exactly these degenerate saddles

For the notes' toy $L(a, b, c) = \tfrac12(1 - abc)^2$, every term of the expansion at the origin below order three vanishes, and along $(t, t, t)$ the third derivative is $-6$: a third-order method can escape. With four scalar layers, $L(t, t, t, t) = \tfrac12(1 - t^4)^2 = \tfrac12 - t^4 + \tfrac12 t^8$: the first visible decrease is at fourth order. Depth pushes the escape direction to higher orders, into the regime the paper proves hard in general. (A toy; the paper's hardness result is about general functions, not these products.)

:::key Putting the four ideas together
Optimality can be tested order by order (idea 1). Third-order terms are odd, so a nonzero one always reveals a way down and can be used algorithmically (idea 2): that is the paper's efficient method. Fourth-order terms are even and checking them is NP-hard in general (idea 3), which is why the program stops there, even though deep products create such saddles naturally (idea 4). The notes below state these results.
:::
:::

:::divider Study notes
:::

## 1. What goes wrong with Hessian-based saddle escape?

Perturbed gradient descent and negative-curvature methods ask whether the minimum Hessian eigenvalue is significantly negative. But for a degenerate saddle, even the Hessian can be positive semidefinite. The authors ask whether third derivatives provide a principled escape method, and whether the same idea can continue indefinitely.

## 2. Rebuild the problem using a simple mathematical example

Consider the two-variable function

$$f(x, y) = x^2 - y^4.$$

At $(0, 0)$, the gradient vanishes and the Hessian is $\mathrm{diag}(2, 0)$; no negative quadratic curvature is visible. Yet $f(0, y) = -y^4 < 0$ when $y \neq 0$. This *fourth-order* example is for intuition: a second-order test can miss descent even when the function is not locally minimized. For a smooth function and a displacement $u$, Taylor expansion shows the additional information available at third order:

$$f(x + u) = f(x) + \nabla f(x)^\top u + \tfrac{1}{2}u^\top\nabla^2 f(x)u + \tfrac{1}{6}\nabla^3 f(x)[u, u, u] + R_4.$$

The tensor $\nabla^3 f(x)[u, u, u]$ is a cubic directional derivative. It can detect escape directions unavailable to the gradient/Hessian pair. A simple *third-order* saddle example is $h(x, y) = x^2 + y^3$: at the origin, the Hessian is semidefinite, but the $y$-direction changes sign across the origin.

## 3. Actual algorithmic and hardness results

The paper constructs an efficient procedure to find an approximate *third-order local optimum* under its smoothness and oracle assumptions. It combines first-/second-order information with tests using higher-order derivatives to avoid certain degenerate stationary regions. Crucially, the authors show that extending this program to finding a *fourth-order local optimum* is NP-hard in the general class they study. This separates two tasks:

$$\underbrace{\text{second-order stationarity}}_{\text{no strong negative Hessian direction}} \quad\not\Rightarrow\quad \underbrace{\text{third-order local optimality}}_{\text{also rules out relevant cubic descent}}.$$

:::key Paper result
There is a provable algorithm for reaching third-order local optimality, but no comparable unrestricted polynomial-time method for fourth-order local optimality unless standard complexity assumptions fail.
:::

## 4. Why this matters for deep linear networks

For a toy deep linear network $\hat{y} = abc$ and target 1, take

$$L(a, b, c) = \tfrac{1}{2}(1 - abc)^2.$$

At $(0, 0, 0)$, both the gradient and Hessian vanish, but along $(t, t, t)$:

$$L(t, t, t) = \tfrac{1}{2} - t^3 + \tfrac{1}{2}t^6.$$

The cubic directional term makes the loss smaller for small $t > 0$. This *teaching example* links Kawaguchi's degenerate deep-linear saddles to the paper's central idea; it is not presented as one of Anandkumar and Ge's experiments.

## 5. Assumptions, limits and relevance

These are general smooth, higher-order oracle results with regularity and accuracy parameters; they do not yield a cheap practical training routine for large Transformers, where third-order derivative calculations and activation kinks are challenging. A third-order escape guarantee also does not imply global optimality. The paper is central to **RQ1**, **RQ3** and **RQ4** because it exposes a limitation of the usual strict-saddle paradigm.

:::warn Scope
The NP-hardness claim concerns general fourth-order local optimality, not the impossibility of training all deep neural networks. Structured neural-network degeneracies may be more tractable.
:::

## 6. Our notes

I like this paper as the logical follow-up to Kawaguchi. If a saddle has no negative Hessian eigenvalue, an optimizer looking only at curvature might think it is safe to stop. Third derivatives can reveal a descent direction, but trying to extend this to every higher order quickly hits a complexity barrier. It explains why our survey shouldn't equate SOSP with a true minimum. **Original source:** [https://proceedings.mlr.press/v49/anandkumar16.html](https://proceedings.mlr.press/v49/anandkumar16.html). **Verification:** official COLT abstract and original manuscript checked; toy examples labelled explicitly.

> **Primary source:** Anandkumar & Ge (2016), *Efficient Approaches for Escaping Higher Order Saddle Points in Non-Convex Optimization*, COLT 2016 (PMLR 49). [proceedings.mlr.press/v49/anandkumar16.html](https://proceedings.mlr.press/v49/anandkumar16.html).
