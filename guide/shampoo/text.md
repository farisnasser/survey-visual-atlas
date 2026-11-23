## 1. Adaptive scaling for whole rows and columns

This chapter follows Section V-C of the survey. **[[shampoo|Shampoo]]** looks like K-FAC because it also transforms a matrix gradient from the left and the right, but its matrices come from accumulated *gradients*, not from a Fisher factorization. It can be understood with AdaGrad and a little linear algebra.

Suppose a layer's gradient is $G = \mathrm{diag}(2, 4)$. GD steps $-\eta G$, twice as far along the second direction. If one direction keeps producing large gradients, we may want to shrink its effective scale, as AdaGrad and Adam do per coordinate. Shampoo extends that idea from single weights to the row and column dimensions of the matrix.

**First matrix, $GG^\top$.** Take $G = \begin{bmatrix}1 & 2\\ 2 & 4\end{bmatrix}$. Then

$$GG^\top = \begin{bmatrix}1 & 2\\ 2 & 4\end{bmatrix}\begin{bmatrix}1 & 2\\ 2 & 4\end{bmatrix} = \begin{bmatrix}5 & 10\\ 10 & 20\end{bmatrix}.$$

The off-diagonal entries are nonzero because the rows are related (the second is twice the first): $GG^\top$ captures how gradient *rows* relate, not just four isolated magnitudes. Likewise $G^\top G$ captures relationships among *columns*. For an $m\times n$ gradient, $GG^\top$ is $m\times m$ and $G^\top G$ is $n\times n$, and Shampoo accumulates them over steps:

$$L_t = \epsilon I_m + \sum_{s=1}^{t} G_sG_s^\top, \qquad R_t = \epsilon I_n + \sum_{s=1}^{t} G_s^\top G_s.$$

These are running summaries of the row-side and column-side gradient statistics; the small $\epsilon$ keeps them positive definite. One minibatch can be noisy, but accumulating over time reveals persistent directions.

## 2. The inverse fourth roots

Shampoo's representative step is (survey Eq. 37)

$$\Delta W_t = -\eta\,L_t^{-1/4}\,G_t\,R_t^{-1/4}.$$

**What is an inverse fourth root?** For a number, $16^{1/4} = 2$ and $16^{-1/4} = \tfrac12$: a larger number gets a smaller scale. For a diagonal matrix, $L = \mathrm{diag}(16, 81)$ gives $L^{-1/4} = \mathrm{diag}(\tfrac12, \tfrac13)$. For a general positive-definite matrix, use its eigendecomposition, $L = U\Lambda U^\top$ and $L^{-1/4} = U\Lambda^{-1/4}U^\top$: express a direction in the eigenvector basis, rescale each eigen-direction, transform back. That is why Shampoo can do more than scale each original weight.

**Why a fourth root and not a square root?** With a single weight, both accumulators are the same number $v$, and the step becomes $-\eta\,v^{-1/4}g\,v^{-1/4} = -\eta g/\sqrt{v}$: AdaGrad's inverse square root, split evenly between the two sides. For a real matrix the two factors act along different dimensions and cannot be collapsed into one denominator.

**A complete update.** Let $G = \mathrm{diag}(2, 4)$ and, for one step, drop $\epsilon$. Then $L = GG^\top = \mathrm{diag}(4, 16)$ and $R = G^\top G = \mathrm{diag}(4, 16)$, with $L^{-1/4} = R^{-1/4} = \mathrm{diag}(1/\sqrt2, 1/2)$. Now $L^{-1/4}G = \mathrm{diag}(\sqrt2, 2)$ and

$$L^{-1/4}\,G\,R^{-1/4} = \begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix}.$$

So this one-step Shampoo update is $-\eta I$, where GD would take $-\eta\,\mathrm{diag}(2, 4)$: the large direction was reduced more. Real Shampoo does not turn every gradient into the identity; history and regularization change the result.

:::viz rotate
:::

:::read How to read this visual
**What is drawn.** A $1\times2$ weight matrix on a narrow quadratic valley with curvatures $8$ and $0.2$, in 3D; green marks the minimum. All three methods use step $0.5$ and start at the same point *in the valley's own axes*, for every angle. Violet: diagonal AdaGrad. Coral: Shampoo, which accumulates the full $2\times2$ matrix $R = \sum g^\top g$ and applies $R^{-1/4}$ (and the scalar $L^{-1/4}$). White: full-matrix AdaGrad. The chart plots the loss of each on a log scale.

**What moves.** The rotation of the valley.

**What to notice.** Diagonal AdaGrad is fast when the valley is aligned with the axes and slows down as it tilts: its per-coordinate scale cannot see a rotated valley (chapter 4 again). Shampoo's and full-matrix AdaGrad's runs are the same at every angle, because their accumulators rotate with the problem. For a $1\times2$ matrix, Shampoo *is* a full-matrix method, with power $-1/4$ instead of $-1/2$. Its memory advantage only appears for real matrices, which is the next visual.
:::

:::viz matrix
:::

:::read How to read this visual
**What is drawn.** Matrix least squares, $\tfrac12\|AWB - C\|^2$, with an $8\times8$ weight $W$ and ill-conditioned $A$ and $B$. The curvature of this problem is itself a Kronecker product, the structure Shampoo assumes. The main plot is the loss (log scale) over 1500 deterministic steps for GD (grey), diagonal AdaGrad (violet), Shampoo (coral) and full-matrix AdaGrad (white), each with its best step from a small grid. The side chart compares how many numbers each preconditioner stores.

**What to notice.** Shampoo's two $8\times8$ statistics, $8^2 + 8^2 = 128$ numbers, capture the row and column structure, and it converges far faster than GD or diagonal AdaGrad. Full-matrix AdaGrad also does well, but its preconditioner has $64\times64 = 4096$ numbers. This toy illustrates the structure only. The paper's guarantee is an online convex regret bound, not a statement about networks.
:::

## 3. What it costs, and what it promises

When the factors have off-diagonal entries, their eigenvectors are tilted relative to the axes, so Shampoo scales *combinations* of weights, which helps when the geometry is coupled. The price: for an $m\times n$ weight the two factors hold about $m^2 + n^2$ numbers, and inverse roots cost far more than elementwise arithmetic. As the survey warns, $m^2 + n^2$ is not always smaller than Adam's storage: for a square $n\times n$ matrix it is $2n^2$, exactly Adam's two moment arrays, and for a thin matrix it is more. Practical implementations reduce how often the roots are recomputed. Shampoo's original analysis gives convergence guarantees for stochastic *convex* problems, which do not automatically extend to non-convex network training.

:::key Remember V-C
Shampoo accumulates gradient outer products along each matrix dimension and rescales the gradient from both sides with inverse matrix roots, capturing relationships among rows and columns. Unlike K-FAC, its factors are gradient statistics, not a Fisher approximation: adaptive scaling extended from coordinates to matrix directions.
:::

:::problem The next step
Matrix factors are costly. Chapter 15 goes back to diagonal scaling, but with curvature: AdaHessian, Apollo and Sophia.
:::
