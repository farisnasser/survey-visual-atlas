# Hardness in Quadratic and Nonlinear Programming

:::minute The paper in one minute
**Question:** How difficult is it, in the worst case, to decide properties of non-convex quadratic or nonlinear optimization problems?

**Approach:** Complexity-theoretic reductions connecting finite combinatorial decision problems to continuous optimization questions.

**Finding:** Several deceptively simple optimization decision problems are computationally intractable in the NP-hard/NP-complete sense.

**Why we care:** Non-convex optimization is difficult in general, but this result is *not* a theorem about the average neural-network training run.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. "Hard" is a statement about a whole family of instances

Complexity theory does not grade one problem instance. It grades a **family** of instances by how the running time grows with the input size $n$ (the number of variables plus the bits needed to write the data). An algorithm is **efficient** when its time grows like a polynomial in $n$. Trying every 0/1 choice of $n$ variables takes $2^n$ cases: $2^{20} = 1\,048\,576$ is fine, but $2^{60} \approx 1.15 \times 10^{18}$ is not. **NP-hard** means that an efficient algorithm for the family would give an efficient algorithm for every problem in NP. Nobody knows such an algorithm, and most researchers believe none exists.

### 2. A reduction turns a discrete question into a continuous one

To prove a continuous problem hard, one writes a known hard discrete problem *as a special case of it*. The paper's Example 2 uses **Subset Sum**: given positive integers $d_0; d_1, \dots, d_n$, is there a choice $y_j \in \{0, 1\}$ with $\sum_j d_j y_j = d_0$? It sets up the box-constrained quadratic program

$$\min_{0 \le y_j \le 1}\; \Big(\sum_{j=1}^n d_j y_j - d_0\Big)^2 + \sum_{j=1}^n y_j (1 - y_j).$$

Both terms are non-negative, and the second is zero only at $y_j \in \{0, 1\}$. So the minimum is $0$ exactly when the Subset Sum instance has a solution. With $d = (3, 5)$: for $d_0 = 8$ the point $y = (1, 1)$ gives $0$; for $d_0 = 4$ no subset of $\{3, 5\}$ sums to $4$, and the minimum over the box is $5/32$, at $y = (1, 3/16)$. A fast method for the global minimum of such programs would therefore solve Subset Sum fast.

### 3. Eigenvalues do not decide local optimality once there are constraints

Without constraints, the origin is a local minimum of $x^\top Q x$ exactly when $Q \succeq 0$ (all eigenvalues non-negative). On the orthant $x \ge 0$ the right condition is weaker: $Q$ must be **copositive**, meaning $x^\top Q x \ge 0$ for every $x \ge 0$. For example, $Q = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$ has eigenvalues $3$ and $-1$, so it is indefinite. Yet all its entries are non-negative, so $x^\top Q x \ge 0$ on the orthant, and the origin is a local (here global) minimum there. The eigenvalue test gives the wrong answer; the real question is a sign condition over a cone.

### 4. Testing a candidate point can itself be the hard part

Gradient methods return a point, and one would like to certify it. The paper's abstract states the two questions it studies: whether a given feasible solution is **not** a local minimum, and whether the objective is **not** bounded below on the feasible set. For a special class of indefinite quadratic programs with simple constraints and integer data, both are NP-complete. As a corollary, checking that an integer square matrix is *not* copositive is NP-complete.

:::key Putting the four ideas together
Hardness is a worst-case statement about a growing family (idea 1). Reductions plant Subset Sum inside a smooth quadratic program (idea 2). With constraints, local optimality depends on copositivity, not eigenvalues (idea 3), and deciding it is NP-complete (idea 4). This is the background the notes below place around deep learning, made precise in their scope.
:::
:::

:::divider Study notes
:::

## 1. First principles: convex versus indefinite quadratic problems

Start with the quadratic objective

$$q(x) = \tfrac12 x^\top Q x + c^\top x, \qquad x \in \mathbb{R}^d, \quad Q = Q^\top.$$

Its gradient and Hessian are

$$\nabla q(x) = Qx + c, \qquad \nabla^2 q(x) = Q.$$

If $Q \succeq 0$ and the feasible set is convex, the objective is convex: local minima are global. If $Q$ has a negative eigenvalue, the objective has a direction of downward curvature. For example, $q(x, y) = x^2 - y^2$ has Hessian $\mathrm{diag}(2, -2)$ and a saddle at the origin. This is a *teaching example*, not the paper's reduction.

## 2. What does NP-complete mean in this context?

A *decision problem* asks for a yes/no answer, e.g. whether a feasible point satisfying a certain objective threshold exists. To show NP-hardness, one maps instances of a known NP-hard problem to instances of the target decision problem in polynomial time, preserving yes/no answers. NP-completeness additionally requires membership in NP for the specified encoded decision problem. This is about *worst-case instances* as input dimension grows, not a claim that every instance is difficult.

:::key Why even local optimality is subtle.
A zero gradient is necessary at an unconstrained differentiable local minimizer, but not sufficient. For $q(x, y) = x^2 - y^2$, $\nabla q(0, 0) = 0$ while $q(0, t) = -t^2 < q(0, 0)$. On constrained regions, feasibility and active constraints make local-optimality questions still more delicate.
:::

## 3. The paper's method and result — careful scope

Murty and Kabadi establish computational hardness for particular quadratic and nonlinear programming decision problems. A central line of their work connects combinatorial hardness to properties of indefinite quadratic programs and local optimality; the precise reductions, input encodings, and problem statements matter. The work is cited in our survey as *background worst-case hardness*, not as a statement that SGD necessarily struggles on typical deep-learning losses.

:::warn Verification limit.
The publication and an author-hosted original PDF were located, but the full reduction proof and exact theorem wording could not be extracted and checked in this environment. Consequently these notes deliberately do *not* attribute a particular reduction formula, decision direction, or theorem number to the authors. The mathematics on this page is first-principles teaching background.
:::

## 4. Connection to our survey and assumptions

**RQ4:** This is a boundary result: general non-convex optimization can be hard, yet structural assumptions in deep-linear, low-rank, or wide-network analyses can make selected training problems tractable. Hardness of arbitrary quadratic programs does not imply that training the particular architectures in later chapters is NP-complete. **Our takeaway.** Non-convexity can hide genuinely hard decisions even in fairly simple mathematical objectives. That motivates studying which additional structures—rank, width, data assumptions, noise geometry—make neural-network training easier.

> **Source:** [https://doi.org/10.1007/BF02592948](https://doi.org/10.1007/BF02592948); author copy: [https://public.websites.umich.edu/~murty/np.pdf](https://public.websites.umich.edu/~murty/np.pdf).
