## 1. Two accumulators, one per side of the matrix

This chapter follows Section V-C of the survey. **[[shampoo|Shampoo]]** approaches matrix structure from a different direction than K-FAC. Instead of approximating a Fisher block, it builds matrix accumulators directly from the gradients. For a weight gradient $G_t \in \mathbb{R}^{m\times n}$, define (survey Eqs. 61 and 62)

$$L_t = \epsilon I_m + \sum_{s=1}^t G_sG_s^\top, \qquad R_t = \epsilon I_n + \sum_{s=1}^t G_s^\top G_s.$$

$L_t$ is $m\times m$ and summarizes how the gradient varies across *rows* (outputs); $R_t$ is $n\times n$ and summarizes the *columns* (inputs). The small $\epsilon I$ keeps both positive definite. The two-sided matrix Shampoo step uses inverse fourth roots (survey Eq. 63):

$$\Delta W_t = -\eta\,L_t^{-1/4}\,G_t\,R_t^{-1/4}.$$

## 2. What a matrix power means

The powers are spectral functions. If $L = U\,\mathrm{diag}(\lambda_i)\,U^\top$ is the eigendecomposition of the positive-definite $L$, then

$$L^{-1/4} = U\,\mathrm{diag}(\lambda_i^{-1/4})\,U^\top.$$

That is: rotate into $L$'s eigenbasis, divide each direction by the fourth root of its accumulated gradient energy, rotate back. Directions in which the gradient has been large get smaller steps, exactly like AdaGrad, except that the "directions" are eigenvectors of a matrix, not individual coordinates. The two factors together rescale the gradient along statistically important row *and* column directions.

Why the fourth root? For a vector parameter, full-matrix AdaGrad would use the inverse square root $(\sum gg^\top)^{-1/2}$. Shampoo applies a root on each of two sides, and the two quarter-powers together play the role of that one square root. The exponent $-1/4$ is for the matrix case; for higher-order tensors Shampoo uses other exponents.

:::viz rotate
:::

:::read How to read this visual
**What is drawn.** A $1\times2$ weight matrix on a narrow quadratic valley with curvatures $8$ and $0.2$, in 3D; green marks the minimum. All three methods use step $0.5$ and start at the same point *in the valley's own axes*, for every angle. Violet: diagonal AdaGrad. Coral: Shampoo, which accumulates the full $2\times2$ matrix $R = \sum g^\top g$ and applies $R^{-1/4}$ (and the scalar $L^{-1/4}$). White: full-matrix AdaGrad. The chart plots the loss of each on a log scale.

**What moves.** The rotation of the valley.

**What to notice.** Diagonal AdaGrad is fast when the valley is aligned with the axes and slows down as it tilts: its per-coordinate scale cannot see a rotated valley (chapter 4 again). Shampoo's and full-matrix AdaGrad's runs are the same at every angle, because their accumulators rotate with the problem. For a $1\times2$ matrix, Shampoo *is* a full-matrix method, with power $-1/4$ instead of $-1/2$. Its memory advantage only appears for real matrices, which is the next visual.
:::

## 3. Memory and cost

Shampoo avoids a full $(mn)^2$ preconditioner, but it stores $m^2 + n^2$ numbers per matrix before any blocking. Every inverse root needs an eigendecomposition or an iterative method, which can be expensive.

The survey warns against a common misreading: it is incorrect to assume that $m^2 + n^2$ is always smaller than $mn$. For a square $n\times n$ matrix it is $2n^2$ versus $n^2$: twice one of Adam's moment arrays, and as much as both of them together. Structured methods are economical relative to a full $(mn)^2$ layer curvature matrix, not necessarily relative to Adam's diagonal state.

:::viz matrix
:::

:::read How to read this visual
**What is drawn.** Matrix least squares, $\tfrac12\|AWB - C\|^2$, with an $8\times8$ weight $W$ and ill-conditioned $A$ and $B$. The curvature of this problem is itself a Kronecker product, the structure Shampoo assumes. The main plot is the loss (log scale) over 1500 deterministic steps for GD (grey), diagonal AdaGrad (violet), Shampoo (coral) and full-matrix AdaGrad (white), each with its best step from a small grid. The side chart compares how many numbers each preconditioner stores.

**What to notice.** Shampoo's two $8\times8$ statistics, $8^2 + 8^2 = 128$ numbers, capture the row and column structure, and it converges far faster than GD or diagonal AdaGrad. Full-matrix AdaGrad also does well, but its preconditioner has $64\times64 = 4096$ numbers. This toy illustrates the structure only. The paper's guarantee is an online convex regret bound, not a statement about networks.
:::

## 4. What Shampoo guarantees

The original Shampoo analysis proves convergence in a stochastic *convex* setting, as a regret bound. That theorem cannot be quoted as a non-convex deep-network guarantee. Scalable implementations (Anil et al., 2020) amortize and distribute the matrix operations: the roots are recomputed only every so often, and large matrices are split into blocks. The key trade-off, in the survey's words, is that a mathematically richer direction may need fewer optimizer steps while costing more arithmetic or memory per step.

Section V-F's SOAP, which runs Adam inside Shampoo's eigenbasis, is covered with Muon in chapter 16.

:::problem The next step
Shampoo keeps matrices. Section V-D of the survey goes the other way: estimate only the diagonal of the Hessian, but estimate the real thing. Chapter 15.
:::
