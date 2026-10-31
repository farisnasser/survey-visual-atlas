# Shampoo: Preconditioned stochastic tensor optimization

:::minute The paper in one minute
**Question:** Can we precondition very large tensor-valued parameters without storing an enormous full gradient-covariance matrix?

**Model/Method:** Maintain a small preconditioner for each tensor mode, using accumulated gradient contractions rather than flattening everything.

**Finding:** Shampoo obtains regret guarantees in online stochastic convex optimization and shows competitive neural-network training performance in experiments.

**Why we care:** Gradient geometry can be exploited through tensor structure at far lower memory cost than an unrestricted full-matrix preconditioner.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Regret: the yardstick of online convex optimization

In **online convex optimization** a learner picks weights $x_t$, then sees a convex loss $f_t$ and pays $f_t(x_t)$, for $t = 1, \dots, T$. Its **regret** compares the total paid with the best single fixed choice in hindsight:

$$\mathrm{Regret}_T = \sum_{t=1}^T f_t(x_t) - \min_x \sum_{t=1}^T f_t(x).$$

A bound that grows like $\sqrt{T}$ means the *average* regret per round shrinks like $1/\sqrt{T}$: after $T = 10\,000$ rounds a regret of $100$ is $0.01$ per round. Stochastic training fits this frame by letting $f_t$ be the loss on the $t$-th minibatch. Shampoo's theorems are regret bounds of this kind, so they hold for convex losses.

### 2. AdaGrad, diagonal and full-matrix

**AdaGrad** divides each step by the square root of the accumulated squared gradients. The diagonal version keeps one number per coordinate. The **full-matrix** version keeps $H_t = \big(\sum_{s \le t} g_s g_s^\top\big)^{1/2}$ and steps by $-\eta H_t^{-1} g_t$, which also adapts to how coordinates move *together*: it can rescale along a tilted direction, not only along the axes. The price is size. For a $1000 \times 1000$ weight matrix the flattened gradient has $d = 10^6$ entries, so $H_t$ has $10^{12}$ entries and computing its inverse root costs on the order of $d^3 = 10^{18}$ operations.

### 3. Fractional powers of a matrix act on its eigenvalues

For a symmetric positive semidefinite matrix $A = U\,\mathrm{diag}(\lambda_1, \dots, \lambda_n)\,U^\top$, the power $A^p$ is defined as $U\,\mathrm{diag}(\lambda_1^p, \dots, \lambda_n^p)\,U^\top$: keep the eigenvectors, raise the eigenvalues. It is not the same as raising each entry. For

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \ (\text{eigenvalues } 3, 1), \qquad A^{-1/2} \approx \begin{bmatrix} 0.789 & -0.211 \\ -0.211 & 0.789 \end{bmatrix},$$

while raising each entry to the power $-\tfrac12$ would give $\begin{bmatrix} 0.707 & 1 \\ 1 & 0.707 \end{bmatrix}$, a different matrix. When the notes write $L^{-1/4}$ they mean the spectral power.

### 4. A Kronecker product acts on a matrix from both sides

The **Kronecker product** $L \otimes R$ of an $m \times m$ and an $n \times n$ matrix is an $mn \times mn$ matrix. Applied to a flattened matrix it never needs to be formed, because (the paper's Lemma 4, with row-major flattening)

$$(L \otimes R^\top)\,\mathrm{vec}(G) = \mathrm{vec}(L\,G\,R).$$

So a preconditioner for the $mn$-vector that has Kronecker form costs only an $m \times m$ and an $n \times n$ matrix: $m^2 + n^2$ numbers instead of $m^2 n^2$. Powers also split, $(L \otimes R)^p = L^p \otimes R^p$, so the two quarter-powers $L^{-1/4}$ and $R^{-1/4}$ together play the role of the full-matrix inverse square root of idea 2.

:::key Putting the four ideas together
Full-matrix AdaGrad has strong regret guarantees (idea 1) but its preconditioner is far too large for neural-network weights (idea 2). Shampoo keeps one small statistic per side of the weight matrix, takes spectral quarter-powers of each (idea 3), and applies them from the left and right, which is a Kronecker-structured preconditioner on the flattened gradient (idea 4). The paper proves this structured version keeps an $O(\sqrt{T})$ regret bound, as made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Why coordinatewise adaptive scaling is not the whole story

SGD uses a scalar step $W^+ = W - \eta G$. AdaGrad-style diagonal scaling adjusts coordinates separately, but misses how gradient coordinates vary together. A full preconditioner might capture these correlations; for a matrix $W \in \mathbb{R}^{m \times n}$, however, flattening it creates a vector of length $mn$ and a covariance-like matrix with $(mn)^2$ entries. Shampoo uses the fact that gradients *already have matrix or tensor shape*. It stores statistics along the left (row) and right (column) directions instead of forming the enormous flattened matrix.

## 2. Worked example: a $2 \times 2$ gradient

Let one gradient be

$$G = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}.$$

Ignore the small identity damping term for this illustration and use one step of statistics:

$$L = GG^\top = \mathrm{diag}(4, 1), \qquad R = G^\top G = \mathrm{diag}(4, 1).$$

Shampoo's two-sided preconditioned update uses $L^{-1/4} G R^{-1/4}$. Since $4^{-1/4} = 1/\sqrt{2}$, we get

$$L^{-1/4} G R^{-1/4} = \begin{bmatrix} 1/\sqrt{2} & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} & 0 \\ 0 & 1 \end{bmatrix} = I_2.$$

The raw gradient's first entry was twice the second, but the structured preconditioner balances them in this example. It *does not* imply all matrix gradients become identity matrices.

:::key The important memory saving
For a $1000 \times 1000$ weight matrix, a dense full flattened preconditioner would have $10^{12}$ entries. The two Shampoo factors together have about $2 \times 10^6$ entries, before implementation optimizations.
:::

## 3. The matrix update, step by step

Algorithm 1 initializes $L_0 = \epsilon I_m$ and $R_0 = \epsilon I_n$. On each stochastic gradient $G_t$, accumulate

$$L_t = L_{t-1} + G_t G_t^\top, \qquad R_t = R_{t-1} + G_t^\top G_t,$$

then update

$$\boxed{W_{t+1} = W_t - \eta L_t^{-1/4} G_t R_t^{-1/4}.}$$

Why the negative quarter powers? Both left and right factors contribute to the overall scaling. When interpreted through vectorization and a Kronecker-product preconditioner, the combination behaves as a structured analogue of the inverse square-root scaling in matrix AdaGrad. These fractional powers are spectral matrix powers, not elementwise powers.

## 4. Extension from matrices to tensors

For a $k$-way tensor with mode lengths $n_1, \dots, n_k$, Shampoo maintains one $n_i \times n_i$ preconditioner per mode. Each factor is formed by contracting gradient tensors over all the other modes. This brings memory closer to $\sum_i n_i^2$ instead of $(\prod_i n_i)^2$. The algorithm's tensor-mode contractions and fractional powers are designed to respect the parameter's structure.

### What makes the proof work?

Theoretical results use online mirror descent and regret bounds in a *convex* setting. The paper's matrix analysis uses a Kronecker-product argument and matrix inequalities (especially Lemmas 8–9) to relate the structured factors to a more expensive full-matrix update. The structure is not arbitrary: the relevant preconditioners must be positive definite (the $\epsilon I$ initialization helps), and tensor ranks enter the regret analysis.

:::warn Different use of the word "second order"
Shampoo uses accumulated outer products of *gradients*, not the exact loss Hessian. It is a structured adaptive preconditioner, not a Newton step or a guaranteed negative-curvature detector.
:::

## 5. Results, contributions, and limitations

**Contributions.** (i) Tensor-mode preconditioning with manageable state; (ii) an elegant two-sided matrix update; (iii) convex online regret guarantees grounded in matrix trace inequalities; and (iv) experiments on modern-for-2018 neural-network training tasks. **Empirical result.** The paper reports faster optimization than standard first-order baselines in selected neural-network workloads, with practical per-step costs that can be comparable under its implementations. Those empirical observations are separate from the convex regret guarantees.

## 6. Important assumptions and caveats

- The formal guarantee is online/stochastic convex optimization; it does not prove global convergence for arbitrary deep neural networks.
- A dense eigendecomposition or matrix-root computation for each mode can be expensive; actual efficiency depends on update frequency, block sizes, and system implementation.
- Tensor modes encode useful structure, but a Kronecker approximation cannot capture every correlation in a flattened gradient.
- Neither strict-saddle escape nor improved test accuracy is guaranteed by the foundational theorem.

## 7. Quick reference

| | |
|---|---|
| **Matrix parameter** | $W \in \mathbb{R}^{m \times n}$; gradient $G$ has same shape. |
| **Left statistic** | $L_t = \epsilon I + \sum_{s \le t} G_s G_s^\top$. |
| **Right statistic** | $R_t = \epsilon I + \sum_{s \le t} G_s^\top G_s$. |
| **Update** | $W^+ = W - \eta L^{-1/4} G R^{-1/4}$. |
| **Memory** | Two factors: $O(m^2 + n^2)$ versus $O(m^2 n^2)$. |
| **Theory scope** | Convex online regret; deep-network speedups are experimental. |

### Survey bridge / Papers Master note

Shampoo uses row/column or tensor-mode gradient statistics to build memory-efficient structured preconditioners. The original theory establishes convex online regret guarantees, while neural-network gains are empirical. Cite in the adaptive/preconditioned dynamics discussion, not as a general saddle-escape theorem.

> **Primary source:** Gupta, Koren & Singer (2018), *Shampoo: Preconditioned stochastic tensor optimization*, ICML 2018. [Original publication](https://proceedings.mlr.press/v80/gupta18a.html)
