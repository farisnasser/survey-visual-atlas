# Non-Convex Optimization for Machine Learning

:::minute The paper in one minute
**Problem:** How can nonconvex algorithms have rigorous guarantees when arbitrary nonconvex functions admit poor minima and difficult saddles?

**Approach:** A *monograph* organizing tractable machine-learning structures—sparsity, low rank, projections and strict-saddle geometry—and their optimization algorithms.

**Main message:** Strong results come from *structural assumptions on a problem class*, not from nonconvexity itself being harmless.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A set is convex when it contains every segment between its points

A set $\mathcal{C}$ is **convex** if, for any two of its points, every point on the segment between them is also in $\mathcal{C}$. Sparse vectors fail this test. With $s = 1$ in $\mathbb{R}^3$, both $(1, 0, 0)$ and $(0, 1, 0)$ have one nonzero entry, but their midpoint $(0.5, 0.5, 0)$ has two. Matrices of rank at most $r$ fail it the same way. So a convex loss restricted to such a set is a nonconvex problem, even with no bad shape in the loss itself.

### 2. Projecting onto these nonconvex sets is easy

The closest point in a set is its **projection**. For vectors with at most $s$ nonzeros, keep the $s$ entries of largest magnitude and zero the rest: projecting $z = (0.3, -2, 1.1, 0.05)$ with $s = 2$ gives $(0, -2, 1.1, 0)$. For matrices of rank at most $r$, keep the top $r$ singular values of the SVD: the best rank-one approximation of $\mathrm{diag}(3, 1)$ is $\mathrm{diag}(3, 0)$, at distance $1$. Easy projections are what make projected gradient descent practical on these sets.

### 3. Curvature only needs to hold along the directions the method uses

For least squares $f(w) = \tfrac12\|Aw - b\|^2$ the Hessian is $A^\top A$. Take $A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}$. Then $A^\top A$ has eigenvalues $0, 1, 3$: the loss is flat along $(1, 1, -1)$, so it is not strongly convex. But that direction has three nonzeros. Over directions with at most two nonzeros, $h^\top A^\top A h/\|h\|^2$ stays between $0.382$ and $2.618$. This is **restricted strong convexity and smoothness**, with constants $\alpha \approx 0.382$ and $\beta \approx 2.618$ on that set of directions.

### 4. Random measurements nearly preserve the length of sparse vectors

A $50 \times 200$ matrix with independent $\mathcal{N}(0, 1/50)$ entries has a null space of dimension at least $150$, so it destroys most vectors. Yet on 20 000 random 3-sparse vectors $w$, the ratio $\|Aw\|^2/\|w\|^2$ stayed between about $0.38$ and $1.97$, and $98\%$ of the ratios fell between about $0.60$ and $1.51$. A **restricted isometry**, $\|Aw\|^2 \approx \|w\|^2$ for all sparse $w$, is exactly the kind of data property that supplies the restricted constants of idea 3 in sparse recovery and matrix sensing.

:::key Putting the four ideas together
The difficulty sits in nonconvex constraint sets (idea 1), whose projections are nevertheless cheap (idea 2). Convergence then only needs curvature along the structured directions (idea 3), which suitable data provide (idea 4). For example, the monograph's Theorem 3.3 proves that generalized projected gradient descent with step $1/\beta$ reaches $f(x^*) + \epsilon$ in $O\big(\tfrac{\alpha}{2\alpha - \beta}\log\tfrac1\epsilon\big)$ steps when $\beta/\alpha < 2$. The notes below place this structural view next to saddles and neural networks.
:::
:::

:::divider Study notes
:::

## 1. A nonconvex constraint is not the same as a bad objective

Consider least squares with sparsity:

$$\min_{w \in \mathbb{R}^d}\ \frac12\|Aw - b\|^2 \qquad \text{subject to} \qquad \|w\|_0 \le s.$$

The quadratic loss is convex in $w$, but the feasible set of vectors with at most $s$ nonzero entries is nonconvex. Taking the average of two sparse vectors can yield more than $s$ nonzeros. The optimization challenge here lies partly in the *constraint geometry*.

## 2. Projected gradient descent from first principles

A projected step takes a gradient descent move and maps it back into the feasible set $\mathcal{C}$:

$$\boxed{w_{t+1} = \Pi_{\mathcal{C}}\big(w_t - \eta\nabla f(w_t)\big)}, \qquad \Pi_{\mathcal{C}}(z) \in \arg\min_{u \in \mathcal{C}}\|u - z\|.$$

For the sparse set $\mathcal{C} = \{w : \|w\|_0 \le s\}$, Euclidean projection keeps the $s$ coordinates of largest magnitude and zeros the rest. For a rank-$r$ matrix constraint, projection keeps the top $r$ singular components (a truncated SVD). These are exact, concrete examples in which nonconvex projections are computationally manageable.

## 3. Why restricted curvature appears

An unrestricted loss may be badly conditioned in some directions. Yet the algorithm only moves along differences of feasible points. A useful restricted-geometry condition for low-complexity directions $h$ has the form

$$\alpha\|h\|^2 \le h^\top\nabla^2 f(w)h \le \beta\|h\|^2 \quad \text{for relevant structured } h.$$

This is a *schematic reminder* of restricted strong convexity/smoothness: the set of relevant directions and constants depends on the particular sparse/low-rank problem. It explains why convergence can be proved without global strong convexity.

## 4. Saddles and the role of curvature

A smooth critical point has $\nabla f(w) = 0$. If the Hessian has a negative eigenvalue, $\lambda_{\min}(\nabla^2 f(w)) < 0$, a second-order Taylor expansion implies a direction of local decrease. The monograph reviews strict-saddle conditions and algorithms that inject randomness or exploit curvature to avoid undesirable stationary points. For general deep ReLU networks, neither such geometric structure nor smoothness is automatic.

:::key Why this matters for RQ4.
Matrix sensing or sparse recovery can admit strong global-convergence statements because their data operators satisfy restricted isometry/incoherence-like properties. We should not transport those guarantees directly to arbitrary neural-network losses.
:::

## 5. Scope and limitations

This is an instructional synthesis of several research lines, not a single original theorem. Its positive convergence results depend on the specific problem, feasible set, sampling conditions, initialization and regularity assumptions. It is valuable for interpreting how *special geometry* rescues nonconvex optimization, while reminding us that broad guarantees for realistic neural nets remain substantially harder. **Our takeaway.** The main lesson is to ask "which structural property makes this nonconvex problem tractable?" before claiming that gradient descent solves nonconvex problems generally.

> **Source:** [https://www.cse.iitk.ac.in/users/purushot/papers/ncom-book.pdf](https://www.cse.iitk.ac.in/users/purushot/papers/ncom-book.pdf) (Parts I–III, especially Chapters 3 and 6).
