# Neural Networks and Principal Component Analysis

:::minute The paper in one minute
**Question:** If we train a neural network, might it get trapped in a bad local minimum?

**The simpler model:** A network with *two linear layers* compresses data, then reconstructs it.

**Finding:** Its best reconstruction is connected to *PCA*. In the paper's analyzed setting, the nonoptimal critical configurations are saddles, not inferior local minima.

**Why we care:** *Non-convex* does not automatically mean *many bad minima*.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A symmetric matrix stretches along perpendicular axes

A covariance matrix $\Sigma$ is symmetric. Every symmetric matrix has orthonormal eigenvectors $u_1, \dots, u_d$ with real eigenvalues $\lambda_1 \ge \dots \ge \lambda_d$, and it can be written as $\Sigma = \sum_i \lambda_i u_i u_i^T$ (the **spectral theorem**). For example,

$$\Sigma = \begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix} = 9\,u_1u_1^T + 1\,u_2u_2^T, \qquad u_1 = \tfrac{1}{\sqrt2}(1, 1)^T,\; u_2 = \tfrac{1}{\sqrt2}(1, -1)^T.$$

Along $u_1$ the data spread with variance $9$; along $u_2$ with variance $1$. The eigenvectors are the natural axes of the data cloud.

### 2. The variance along a unit direction is an average of eigenvalues

For a unit vector $u$, write $c_i = u_i^T u$. Then $u^T \Sigma u = \sum_i \lambda_i c_i^2$, and the weights $c_i^2$ add up to $1$. So $u^T\Sigma u$ is a weighted average of the eigenvalues: it lies between $\lambda_d$ and $\lambda_1$, and it equals $\lambda_1$ only along the top eigenvector. In the example, $u = (1, 0)^T$ has $c_1^2 = c_2^2 = \tfrac12$ and $u^T\Sigma u = 5$, halfway between $1$ and $9$. This is why the notes' "maximize $u^T\Sigma u$" picks PC1.

### 3. The trace is the total variance

The **trace** $\mathrm{tr}(\Sigma)$, the sum of the diagonal entries, equals the sum of the eigenvalues: $5 + 5 = 9 + 1 = 10$. Keeping the subspace spanned by a set of eigenvectors preserves exactly their eigenvalues, and the reconstruction error is what is left. The paper's eq. (10) states this for every full-rank critical point: the error is the total variance minus the eigenvalues kept. Keeping $u_1$ loses $1$; keeping $u_2$ loses $9$.

### 4. Convex in each factor, not in both together

Fix one factor and the loss becomes ordinary least squares in the other. With $L(a, b) = (1 - ab)^2$ and $a = 2$ held fixed, $L = (1 - 2b)^2$ is a parabola in $b$ with its minimum at $b = \tfrac12$. The paper's Facts 1 and 2 say the same for the matrices: for fixed $A$ the error is convex in $B$, and for fixed $B$ it is convex in $A$. Jointly it is not convex, because the factors multiply. The difficulty lives only in how the two factors interact.

:::key Putting the four ideas together
Each critical point keeps the span of some set of eigenvectors (ideas 1 and 3) and pays the eigenvalues it leaves out. Rotating a kept direction toward a better unused eigenvector raises the preserved variance (idea 2), so every choice except the top-$p$ set has a descent direction: a saddle. Each factor alone is an easy least-squares problem (idea 4); the landscape result below explains why their product is benign too.
:::
:::

:::divider Study notes
:::

## 1. Start with a concrete network

Picture a data point with **three features**: $x = (x_1, x_2, x_3)^T$. We want to compress those three numbers into a smaller code, and then reconstruct them. There are no nonlinear activations—only matrix multiplication.

$$\underbrace{h = Bx}_{\text{compress}}, \qquad \underbrace{\hat x = Ah}_{\text{reconstruct}} \quad\Longrightarrow\quad \boxed{\hat x = ABx}.$$

:::figure
<svg viewBox="0 0 460 60" role="img" aria-label="Diagram: x with 3 features goes through B to h, the small code, which goes through A to x-hat, the reconstruction.">
<defs><marker id="ntBhAr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#8e99bd"/></marker></defs>
<rect x="6" y="14" width="104" height="32" rx="7" fill="none" stroke="#7cc4ff" stroke-width="1.3"/>
<text x="58" y="34" fill="#e8ecf8" font-size="9" text-anchor="middle"><tspan font-style="italic">x</tspan>: 3 features</text>
<line x1="110" y1="30" x2="176" y2="30" stroke="#8e99bd" stroke-width="1.3" marker-end="url(#ntBhAr)"/>
<text x="143" y="24" fill="#e8ecf8" font-size="9" text-anchor="middle" font-style="italic">B</text>
<rect x="178" y="14" width="104" height="32" rx="7" fill="none" stroke="#ffc857" stroke-width="1.3"/>
<text x="230" y="34" fill="#e8ecf8" font-size="9" text-anchor="middle"><tspan font-style="italic">h</tspan>: small code</text>
<line x1="282" y1="30" x2="348" y2="30" stroke="#8e99bd" stroke-width="1.3" marker-end="url(#ntBhAr)"/>
<text x="315" y="24" fill="#e8ecf8" font-size="9" text-anchor="middle" font-style="italic">A</text>
<rect x="350" y="14" width="104" height="32" rx="7" fill="none" stroke="#7cc4ff" stroke-width="1.3"/>
<text x="402" y="34" fill="#e8ecf8" font-size="9" text-anchor="middle"><tspan font-style="italic">x̂</tspan>: reconstruction</text>
</svg>
:::

**Notation:** $x$ is the original *column vector*; $B$ makes a shorter representation $h$; $A$ maps it back to the original dimension; $\hat x$ ("x-hat") is the reconstruction. If $x \in \mathbb{R}^d$ and the code has $p < d$ entries, then $B \in \mathbb{R}^{p \times d}$ and $A \in \mathbb{R}^{d \times p}$.

**Training objective:** Make the reconstructed points as close as possible to the originals. For $N$ samples,

$$\boxed{E(A, B) = \frac{1}{N}\sum_{i=1}^N \|x_i - ABx_i\|_2^2.}$$

$\|v\|_2^2 = v^Tv$ is squared distance. Each term measures how much information was lost when reconstructing one data point.

## 2. Why a *linear* network has a non-convex loss

Even though $x \mapsto ABx$ is linear, we optimize the *separate* matrices $A, B$, whose entries multiply each other. For a one-input, one-output illustration, set $\hat y = ab$ and target $y = 1$:

$$L(a, b) = (1 - ab)^2 = 1 - 2ab + a^2b^2.$$

This is *not jointly convex* in $(a, b)$. Contrast: if we optimized one unconstrained linear map $W$ directly, squared error would be convex in $W$.

:::key Why it matters for your survey (RQ1)
*Non-convex* does not automatically mean *bad local minima*. It tells us the parameter-space surface is not a single convex bowl; we still have to classify its stationary points.
:::

## 3. Stationary points, minima, and saddles — from calculus

The **gradient** contains first derivatives: $\nabla L = (\partial L/\partial a, \partial L/\partial b)^T$. A stationary point has $\nabla L = 0$; this is only a *candidate*, not proof of a minimum.

$$L(a, b) = (1 - ab)^2, \qquad \frac{\partial L}{\partial a} = -2b(1 - ab), \qquad \frac{\partial L}{\partial b} = -2a(1 - ab).$$

For both derivatives to be zero: **(i)** $ab = 1$ (loss 0: global minima), or **(ii)** $a = b = 0$ (loss 1: stationary). Is the origin a bad minimum?

$$\begin{aligned} L(\varepsilon, \varepsilon) &= (1 - \varepsilon^2)^2 < 1, \\ L(\varepsilon, -\varepsilon) &= (1 + \varepsilon^2)^2 > 1 \qquad (0 < \varepsilon < 1). \end{aligned}$$

Moving in one direction *decreases* the loss; in another it *increases*. So $(0, 0)$ is a **saddle point**.

### 3.1 The Hessian: a curvature test

The **Hessian** $H$ collects second partial derivatives. At a stationary point, the local approximation is $L(\theta + \delta) \approx L(\theta) + \frac12 \delta^T H \delta$. Eigenvalues of $H$ indicate curvature along its eigenvector directions.

$$H(0, 0) = \begin{bmatrix} 0 & -2 \\ -2 & 0 \end{bmatrix}, \qquad \lambda_1 = 2,\; \lambda_2 = -2.$$

Positive curvature in one direction, negative in another $\Rightarrow$ saddle. A **strict saddle** has a negative Hessian eigenvalue. Zero eigenvalues need further analysis: second-order information can be inconclusive.

:::warn Teaching example vs. the actual theorem
The scalar example above illustrates the logic but is *not* the paper's full classification. The authors analyze the matrix-valued squared-error objective and show its connection to covariance eigenvectors under specified assumptions.
:::

## 4. PCA from first principles: what are $u$, $u^T$, and $uu^T$?

**The problem:** We have a point $x \in \mathbb{R}^2$, but want to store it using *one number* and then reconstruct it as well as possible. We must first choose a direction $u$ (a column vector). Make $u$ a **unit vector**, $u^Tu = 1$, so its length does not distort the measurement. For example, $u = (1, 0)^T$ chooses the horizontal axis.

**Transpose means turn a column into a row.** For $x = (3, 2)^T$, we have $u = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $u^T = \begin{bmatrix} 1 & 0 \end{bmatrix}$. The product $u^Tx$ is the **dot product**: row $\times$ column $\to$ a scalar. It extracts how far $x$ extends along the chosen unit direction:

$$\underbrace{z = u^Tx}_{\text{compress: 2 numbers}\to 1} = \begin{bmatrix} 1 & 0 \end{bmatrix}\begin{bmatrix} 3 \\ 2 \end{bmatrix} = 3.$$

To **reconstruct**, use that scalar to scale the direction: $\hat x = uz = (1, 0)^T 3 = (3, 0)^T$. We kept the horizontal component and discarded the vertical one. Substituting $z = u^Tx$ into $\hat x = uz$ yields $\hat x = u(u^Tx) = (uu^T)x$.

**Why $uu^T$ is a matrix, not a number:** column $\times$ row produces an *outer product*, unlike the dot product $u^Tu$ (a scalar):

$$\underbrace{uu^T}_{P \text{ (projection matrix)}} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}\begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \qquad Px = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}\begin{bmatrix} 3 \\ 2 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}.$$

The **order matters**: $u^Tx$ gives one coordinate; $uz$ turns it back into a vector; $uu^T$ is the single matrix that performs both steps. For a different unit direction, the same construction projects onto a different line.

**Which $u$ should we choose?** PCA picks the direction that loses the *least* information across all data points. For centered data, let $\Sigma = \frac1N \sum_i x_i x_i^T$ be the covariance matrix. Expanding squared reconstruction error and using $u^Tu = 1$ gives

$$\underbrace{\frac1N \sum_i \|x_i - uu^Tx_i\|^2}_{\text{error after compression}} = \underbrace{\mathrm{tr}(\Sigma)}_{\text{total variance}} - \underbrace{u^T\Sigma u}_{\text{variance preserved}}.$$

The first term is constant, so **minimize error $\Leftrightarrow$ maximize** $u^T\Sigma u$ subject to $u^Tu = 1$. From the eigenvalue equation $\Sigma u = \lambda u$, the maximizing direction is the eigenvector with the largest eigenvalue (PC1). With $p$ retained dimensions, use the top $p$ eigenvector directions.

## 5. Connect the mathematics to the paper's result

Take a centered 2D dataset with covariance

$$\Sigma = \begin{bmatrix} 9 & 0 \\ 0 & 1 \end{bmatrix}.$$

PC1 is horizontal (variance 9); PC2 is vertical (variance 1). If we preserve one direction $u(\theta) = (\cos\theta, \sin\theta)^T$, then

$$\begin{aligned} E(\theta) &= \mathrm{tr}(\Sigma) - u(\theta)^T\Sigma u(\theta) \\ &= 10 - (9\cos^2\theta + \sin^2\theta) = \boxed{1 + 8\sin^2\theta}. \end{aligned}$$

**At $\theta = 0$:** retain PC1; loss $= 1$ (best). **At $\theta = \pi/2$:** retain PC2; loss $= 9$ (worse). The second direction is stationary with respect to rotation, but rotating toward PC1 improves reconstruction. On the one-dimensional space of projection directions it is a maximum; in the relevant full-weight network setting, the nonoptimal full-rank configurations are saddle points.

:::theorem Paper's key landscape conclusion
In the auto-associative linear-network case, covariance eigenvectors characterize the critical projection subspaces. The projector onto the **leading $p$-dimensional eigenspace** minimizes the reconstruction error; the paper classifies the other nondegenerate critical configurations as saddles rather than suboptimal local minima.
:::

## 6. One subtlety: a unique solution is not unique weights

The *learned mapping* $AB$ can be the same for many weights. For any invertible hidden-space matrix $C$,

$$\boxed{AB = (AC)(C^{-1}B)}.$$

Thus even if the optimal *projection* is unique when the relevant eigenvalues are distinct, there are many equivalent $(A, B)$ pairs. This parameter symmetry creates flat directions and helps explain why neural-network Hessians may have zero eigenvalues.

## 7. What the result does *not* say

- **Architecture:** it concerns *linear* networks; the same "no poor local minima" claim does not extend automatically to nonlinear ReLU networks.
- **Loss/data:** the central PCA characterization depends on quadratic error and the paper's rank/eigenvalue hypotheses; non-distinct eigenvalues and rank-deficient cases require care.
- **Optimization:** describing all critical points is not the same as proving that a particular GD/SGD run escapes saddles quickly or converges globally.

## 8. Connection to our survey and next paper

:::key Takeaway — Geometry (RQ1), limits (RQ4)
$$\boxed{\nabla L(\theta^*) = 0 \;\not\Rightarrow\; \theta^* \text{ is a minimum.}}$$

A *non-convex* problem can still possess a favorable landscape: global minima and saddles, but no poor local minima in the analyzed regime. **Baldi–Hornik tells us what critical points look like.** Saxe et al. (2014) is the natural next reading for *how gradient-based training moves* through a deep linear network over time.
:::

> **Reference:** Baldi, P., & Hornik, K. (1989). *Neural Networks and Principal Component Analysis: Learning from Examples without Local Minima.* *Neural Networks*, 2(1), 53–58. [doi:10.1016/0893-6080(89)90014-2](https://doi.org/10.1016/0893-6080%2889%2990014-2). *Worked scalar and $2 \times 2$ examples are explanatory additions, not direct reproductions of the authors' proofs.*
