# When a Small Krylov Space Can Solve a Nonconvex Model

:::minute The paper in one minute
**Question:** How quickly can matrix-vector products solve trust-region and cubic-regularized quadratic subproblems?

**Model/Method:** Globally minimize the model over $\mathrm{span}\{b, Ab, \dots, A^{t-1}b\}$, computed efficiently by Lanczos; randomize to uncover hidden negative curvature.

**Finding:** Theorems give exponential error decay with shifted conditioning and a sublinear $t^{-2}$ bound with an alignment logarithm. Constructed lower bounds show the rates are sharp in their oracle setting.

**Why we care:** The inner cubic solve is a major part of second-order optimization cost. A gradient-seeded Krylov method can miss a negative eigenspace completely in the hard case.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A ball constraint makes an indefinite quadratic solvable

Minimizing $q(x) = \tfrac12 x^\top A x + b^\top x$ has no answer when $A$ has a negative eigenvalue: $q$ falls forever along it. Restricting to $\|x\| \le R$, the **trust-region problem**, fixes that. Its global minimizers are exactly the points with a multiplier $\lambda^* \ge 0$ such that

$$(A + \lambda^* I)\, s^* = -b, \qquad A + \lambda^* I \succeq 0, \qquad \lambda^*(R - \|s^*\|) = 0 .$$

In one dimension with $A = -1$, $b = 0.5$, $R = 1$: the minimizer is $s^* = -1$ on the boundary, with $q(-1) = -1$ (the other end gives $q(1) = 0$). The condition $(-1 + \lambda^*)(-1) = -0.5$ gives $\lambda^* = 1.5$, and $A + \lambda^* = 0.5 \ge 0$ as required. The shift $\lambda^*$ is what turns the nonconvex problem into a convex one.

### 2. A Krylov space is what $t$ matrix-vector products can reach

Starting from $b$ and multiplying by $A$ repeatedly gives $b, Ab, \dots, A^{t-1}b$; their span is the **Krylov space** $\mathcal{K}_t(A, b)$. Each new direction costs one product $Av$, and the matrix is never stored. **Lanczos** builds an orthonormal basis of this space as it grows. For $A = \mathrm{diag}(1, 2, 3)$ and $b = (1, 1, 1)$, three products already span all of $\mathbb{R}^3$: the matrix $[b, Ab, A^2b]$ is a Vandermonde matrix with determinant $(2-1)(3-1)(3-2) = 2 \ne 0$.

### 3. Exponential rates come with the square root of a condition number

Polynomials in $A$ can approximate the inverse of the shifted matrix $A + \lambda^* I$ well when its eigenvalues are spread over a narrow range. The relevant ratio is the shifted **condition number** $\kappa = (\lambda_{\max} + \lambda^*)/(\lambda_{\min} + \lambda^*)$, and the error falls like $e^{-ct/\sqrt{\kappa}}$, the same $\sqrt{\kappa}$ as in conjugate gradients. With the paper's constants, $36\,e^{-4t/\sqrt{\kappa}}$ for $\kappa = 100$ is about $0.66$ at $t = 10$ and about $2.2 \times 10^{-4}$ at $t = 30$, as a fraction of the initial gap.

### 4. A random vector almost never misses a direction

A vector drawn uniformly from the unit sphere in $\mathbb{R}^d$ has a component along any fixed unit vector $u$ that is nonzero with probability one, of typical size $1/\sqrt{d}$. The paper uses $|u^\top v|^2 \ge \delta^2/d$ with probability at least $1 - \delta$. The price shows up only inside a logarithm: for $d = 10^6$ and $\delta = 0.1$, the factor $\log^2(2\sqrt{d}/\delta) = \log^2(20{,}000) \approx 98$.

:::key Putting the four ideas together
The trust-region and cubic models are solved through a shift $\lambda^*$ (idea 1). A Krylov method searches the space that $t$ products reach (idea 2), and its error falls at a rate set by the shifted condition number (idea 3). When $b$ has no component along the most negative eigenvector, that space never sees it; a random vector supplies one at a logarithmic cost (idea 4). The notes below work through the failure and the fix.
:::
:::

:::divider Study notes
:::

## 1. Start from the model that must be solved

$$q(x) = \tfrac12 x^\top A x + b^\top x, \qquad A = A^\top.$$

The trust-region problem minimizes $q$ under $\|x\| \le R$. Cubic regularization instead minimizes $q(x) + (\rho/3)\|x\|^3$. $A$ can be indefinite, but the radius or cubic term prevents unbounded descent. In an outer Newton-type method, $b$ is a local gradient and $A$ a local Hessian. Solving this model globally is not the same as globally minimizing the original nonlinear training loss.

A huge dense $A$ may be impossible to store, yet $Av$ may be affordable. A Krylov method constructs $\mathcal{K}_t(A, b) = \mathrm{span}\{b, Ab, \dots, A^{t-1}b\}$, then solves the model over this small space. Lanczos builds an orthonormal basis $Q_t$ whose projected matrix $Q_t^\top A Q_t$ is tridiagonal. The expensive operations are roughly $t$ products with $A$; the small projected model must still be solved.

## 2. Why polynomials appear

Every vector in $\mathcal{K}_t$ is $p(A)b$ for a polynomial $p$ of degree at most $t - 1$. In an eigenbasis of $A$, its $i$th coordinate is $p(\lambda_i)b_i$. Thus a polynomial can rescale a component that exists in $b$, but it cannot create an eigenvector component whose coefficient $b_i$ is exactly zero. This simple fact explains both rapid approximation and the hard-case failure.

:::warn Fixed matrix, adaptive directions
All vectors in one Krylov sequence are built using the same $A$. Replacing $A$ by a freshly sampled Hessian on every product changes the method and is not covered by these subspace bounds.
:::

## 3. Worked example: a hidden negative eigendirection

*Pedagogical illustration, constructed for these notes.*

Let $A = \mathrm{diag}(-2, 1)$, $b = (0, 1)$, and $R = 1$. Then $q(x, y) = -x^2 + y^2/2 + y$. The first coordinate has negative curvature, but $b$ has no component there.

$A^j b = b$ for every $j \ge 0$, so $\mathcal{K}_t(A, b) = \mathrm{span}\{(0, 1)\}$ for every $t$. Restricting $x = 0$ gives $q(0, y) = y^2/2 + y$, minimized on $|y| \le 1$ at $y = -1$, with value $-1/2$. More products cannot improve this space.

For a full-space minimum choose the trust-region multiplier $\lambda^* = 2$. The conditions $(A + \lambda^* I)s^* = -b$ and $\|s^*\| = 1$ give $3y = -1$ and $x^2 = 1 - 1/9 = 8/9$.

$$s^* = \big(\pm 2\sqrt{2}/3,\; -1/3\big), \qquad q(s^*) = -8/9 + 1/18 - 1/3 = -7/6.$$

The restricted gap is $(-1/2) - (-7/6) = 2/3$ forever. The shifted matrix $\mathrm{diag}(0, 3)$ is positive semidefinite, so the stated global optimality conditions hold. A local projected solve is correct within its space yet globally wrong for the original model.

If an extra vector $v$ has a nonzero first coordinate, $\mathrm{span}\{b, v\}$ is all of $\mathbb{R}^2$. A continuously sampled random direction has this property with probability one, so the joint space can represent $s^*$. This small example demonstrates why adding a random direction repairs the missing-component problem.

The hard case is more specific than a poorly conditioned positive-definite problem: $u_{\min}^\top b = 0$ and $\lambda^* = -\lambda_{\min}(A)$. Here both hold exactly. Near the hard case, a tiny but nonzero component is available; the nonconvex sublinear bound tracks the difficulty through an alignment logarithm rather than requiring a uniformly large initial projection.

## 4. The two convergence bounds

Theorem 1 bounds the objective gap of the exact minimizer over $\mathcal{K}_t$ for the trust-region problem. Write $\Delta_0 = q(0) - q(s^*)$ and $\kappa = (\lambda_{\max} + \lambda^*)/(\lambda_{\min} + \lambda^*)$ when the denominator is positive. One bound is

$$q(s_t) - q(s^*) \le 36\,\Delta_0 \exp\!\big(-4t/\sqrt{\kappa}\big).$$

This is geometric error reduction in the subspace order, with a condition number for the shifted problem, not for $A$ alone. A second bound is proportional to $(\lambda_{\max} - \lambda_{\min})\|s^*\|^2/t^2$, multiplied in the nonconvex case by a logarithmic factor in $\|b\|/|u_{\min}^\top b|$. Both deterministic bounds can be vacuous in the exact hard case, as the worked example confirms.

## 5. What makes the proof work

Krylov spaces are invariant under scalar shifts: $\mathcal{K}_t(A, b) = \mathcal{K}_t(A + \lambda I, b)$. Shifting to a positive-semidefinite matrix permits convex approximation estimates. The gap decomposes into a shifted convex-model error plus a norm discrepancy. Polynomial inverse approximation controls the first term; a good extremal-eigenvector approximation or a small rescaling controls the second. Chebyshev polynomial bounds explain the $\exp(-4t/\sqrt{\kappa})$ and $t^{-2}$ scales.

## 6. Randomization has a cost and a choice

Corollary 2 augments the space with a random direction and its matrix powers. With probability at least $1 - \delta$, the alignment difficulty becomes a logarithmic factor in dimension and failure probability. Counting both sequences matters: about half the products are spent on the extra vector. Corollary 3 instead perturbs $b$ by $\sigma v$. It avoids that doubling but adds an objective-error term $2\sigma R$. Choosing $\sigma$ too large imposes a high error floor; too small can delay discovery of negative curvature.

Theorem 5 constructs difficult instances whose bounds match the stated scales up to constants. A resisting-oracle argument extends appropriate lower bounds to deterministic matrix-vector methods with sufficient dimension. This is not a lower bound against all randomized methods; randomization explicitly helps the nonconvex alignment obstruction.

## 7. Transfer to cubic regularization

For the cubic convention $\rho\|x\|^3/3$, global optimality requires $(A + \rho\|s^*\|I)s^* = -b$ and $A + \rho\|s^*\|I$ positive semidefinite. The equivalent trust-region instance has $R = \|s^*\|$ and $\lambda^* = \rho\|s^*\|$. At feasible points the cubic objective gap is no larger than the corresponding trust-region gap; Corollary 4 transfers the convergence analysis. This equivalence is analytical: the algorithm is not assumed to know $\|s^*\|$ beforehand.

## 8. What was tested

Section 5 generates 5,000 random diagonal cubic problems of dimension one million for each of three prescribed finite condition numbers. Rotation invariance makes diagonal instances useful for testing subspace behavior while keeping products cheap. Figure 1 supports the two regimes in these ensembles. Additional hard-case experiments compare joint spaces with perturbing $b$; fixed perturbations reach an error floor while joint spaces can continue improving.

Supplement E specifies uniform interior eigenvalues, controlled extrema and scaling to unit initial gap. Hard-case instances deliberately set $b_1 = 0$; the supplementary figure varies eigen-gap. These are generated quadratic-model experiments, not neural-network training comparisons. Claims about outer ARC complexity on main page 8 are expressly discussed without a full outer analysis in this paper.

## 9. Limits and survey use

The theory analyzes exact minimizers of projected problems and mathematical Krylov spaces. Finite precision can destroy orthogonality; exact $d$-step termination is not a reliable floating-point promise. Inner reduced solves, storage and possible basis regeneration also cost work. The randomization bounds concern a fixed symmetric matrix, not arbitrary changing stochastic products. This paper supplies an inner-solver ingredient, not an unconditional end-to-end saddle-escape theorem for neural networks.

## 10. Quick reference

Agarwal2017 also studies efficient cubic solves, through shifted systems and eigenvectors. Here their ingredients already lie in an appropriate Krylov space. Tripuraneni2018 additionally controls stochastic derivative errors and outer stopping; substituting this inner method would require proving that its particular approximation contract is met.

| Term | Meaning |
|---|---|
| $\kappa$ | Conditioning after the optimal regularization shift. |
| Hard case | Missing minimum-eigenvector component and singular optimal shift. |
| Random $b$ perturbation | Improves alignment but introduces controllable bias. |

:::plain Papers Master - Our notes (copy)
Krylov trust-region/cubic subproblem solves have geometric and $t^{-2}$ objective-gap bounds controlled by shifted conditioning and eigenvector alignment. The exact hard case can stall forever; random augmentation or perturbation repairs alignment, with extra products or bias. Results concern fixed-matrix inner models, not global neural-network training.
:::

Reading recommendation: Study the Krylov mechanism, hard-case example and randomization. Keep this inner-subproblem result separate from an outer neural-network convergence theorem.

> **Source:** main PDF 1-9, Theorem 1, Corollaries 2-4, Theorem 5; all 12 supplement pages (printed 12-23). Trust-region example constructed and checked independently. Carmon & Duchi (2018), *Analysis of Krylov Subspace Solutions of Regularized Non-Convex Quadratic Problems*, NeurIPS 2018. [papers.nips.cc (paper)](https://papers.nips.cc/paper_files/paper/2018/file/349f36aa789af083b8e26839bd498af9-Paper.pdf)
