# When Non-Convexity Has No Bad Minima

:::minute The paper in one minute
**Question:** Can a non-convex problem have *no suboptimal local minima*, so every genuine local optimum solves the original task?

**Model/Method:** Recover a low-rank matrix from measurements, missing entries, or sparse corruptions. Factor it as $M = UV^\top$ and analyze gradients and Hessian curvature.

**Finding:** For the paper's regularized matrix sensing, completion, and robust-PCA formulations, under explicit assumptions, all local minima are global; nonoptimal stationary points have escaping curvature (or a smooth-upper-bound analogue).

**Why we care:** Non-convexity alone does not force bad minima. Combine a *benign landscape* with a saddle-escaping optimizer and global recovery can become possible.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Curvature along one direction is a second derivative along a line

The quantity $\langle \Delta, \nabla^2 F(U)\Delta\rangle$ in Lemma 7 is the second derivative of $F$ along the straight line through $U$ in direction $\Delta$:

$$\langle \Delta, \nabla^2 F(U)\Delta\rangle = \frac{d^2}{dt^2} F(U + t\Delta)\Big|_{t=0} .$$

A single direction with a negative value proves that $U$ is not a local minimum. In the notes' $2 \times 2$ example at $U = 0$, the aligned direction is $\Delta = 0 - u^* R = (\mp 1, 0)^\top$, and $F(t\Delta) = \tfrac12(t^2 - 1)^2$ has second derivative $-2$ at $t = 0$. Lemma 7's idealized formula gives the same number: $\|\Delta\Delta^\top\|_F^2 - 3\|M - M^*\|_F^2 = 1 - 3 = -2$, below the bound $-\|M - M^*\|_F^2 = -1$.

### 2. Rescaling the factors changes nothing, unless a penalty says so

For $M = UV^\top$, the pair $(cU, V/c)$ gives the same $M$ for every $c > 0$. The data-fit loss is therefore flat along a whole curve, and the factors can drift to wildly different scales, which spoils the curvature argument. The balancing penalty $\tfrac18\|U^\top U - V^\top V\|_F^2$ is zero exactly when the two factors are balanced. Example with scalars and $uv = 2$: $(u, v) = (1, 2)$ pays $\tfrac18(1 - 4)^2 = 9/8$, while $(\sqrt2, \sqrt2)$ pays $0$.

### 3. Sampling entries only works if the information is spread out

Matrix completion observes random entries. If $M^* = e_1 e_1^\top$ in a $100 \times 100$ matrix, all its information sits in one of the $10\,000$ entries: sampling $10\%$ of the entries misses it with probability $0.9$, and nothing can recover it. A matrix whose entries are all equal to $1/100$ has the same rank, but every sampled entry carries information about it. **Incoherence** ($\mu$ in Theorem 4) measures how spread out the singular vectors are, and the paper's regularizers keep the factors in that spread-out region.

### 4. A smooth upper bound that touches can stand in for a Hessian

The robust-PCA loss is not twice differentiable, so it has no Hessian everywhere. If a smooth function $g$ lies *above* the loss and *touches* it at the current point, then any direction in which $g$ goes down also lowers the loss. Example: $|x| \le \tfrac12 x^2 + \tfrac12$, with equality at $x = \pm 1$. Showing that $g$ has negative curvature at the touching point then certifies an escape direction for the original loss. This is the "pseudo" in Theorem 5's pseudo-strict-saddle property.

:::key Putting the four ideas together
The proof certifies non-minimality with the curvature along one well-chosen direction, the aligned difference $\Delta$ (idea 1). Balancing (idea 2) and incoherence (idea 3) are the conditions that make that certificate work for sensing and completion, and a touching smooth upper bound replaces the Hessian for robust PCA (idea 4). The notes below state the identity and the three theorems.
:::
:::

:::divider Study notes
:::

## 1. First understand the task: recover a simple matrix

Consider a large matrix $M^*$ that is **low rank**: its rows or columns contain redundant information. For example, in a recommendation matrix, user preferences may depend on only a few underlying factors. Three recovery problems ask us to infer $M^*$ from different observations:

| | |
|---|---|
| **Matrix sensing** | Observe linear measurements $b_i = \langle A_i, M^*\rangle$. |
| **Matrix completion** | Observe a randomly sampled subset of the entries of $M^*$. |
| **Robust PCA** | Observe $M_{\text{obs}} = M^* + S^*$ with sparse corruptions $S^*$. |

The rank constraint makes recovery non-convex. A standard trick replaces the matrix with two smaller factors:

$$M = UV^\top, \qquad U \in \mathbb{R}^{d_1\times r}, \quad V \in \mathbb{R}^{d_2\times r}.$$

Even if the data-fit function $f(M)$ is convex *in $M$*, the composition $f(UV^\top)$ is generally **non-convex in $U, V$**. The paper asks whether that non-convexity creates harmful minima.

:::key Keep the scope straight
This is a theorem about *structured low-rank recovery objectives*, not a theorem stating that arbitrary deep neural networks have no bad local minima. The lesson transfers conceptually, not automatically.
:::

## 2. Worked example: a saddle with zero gradient

Take a tiny symmetric rank-one target, with a rank-one model $M = uu^\top$:

$$M^* = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \qquad u = \begin{bmatrix} a \\ b \end{bmatrix}, \qquad F(a, b) = \frac{1}{2}\|uu^\top - M^*\|_F^2.$$

**Step 1 — Expand the loss.** Because $uu^\top = \left[\begin{smallmatrix} a^2 & ab \\ ab & b^2 \end{smallmatrix}\right]$, sum the squared entrywise errors:

$$F(a, b) = \frac{1}{2}\Big[(a^2 - 1)^2 + 2a^2b^2 + b^4\Big] = \frac{1}{2}\Big[(a^2 + b^2)^2 - 2a^2 + 1\Big].$$

**Step 2 — Calculate gradient and stationary points.**

$$\frac{\partial F}{\partial a} = 2a(a^2 + b^2 - 1), \qquad \frac{\partial F}{\partial b} = 2b(a^2 + b^2).$$

Both derivatives are zero at $(a, b) = (0, 0)$, and also at $(\pm 1, 0)$. **Step 3 — Is $(0, 0)$ a minimum?** The Hessian there is

$$\nabla^2 F(0, 0) = \begin{bmatrix} -2 & 0 \\ 0 & 0 \end{bmatrix}, \qquad \lambda_{\min} = -2 < 0.$$

It has a *negative-curvature direction*. Along $b = 0$:

$$F(a, 0) = \tfrac{1}{2}(a^2 - 1)^2; \qquad F(0, 0) = 0.5, \qquad F(0.5, 0) = 0.28125, \qquad F(1, 0) = 0.$$

**Step 4 — Identify the true minima.** At $u = (\pm 1, 0)^\top$, we recover $M^*$ exactly, so $F = 0$. These are global minima; $(0, 0)$ is a *strict saddle*, not a bad local minimum.

:::key What this teaches
The loss is non-convex in the factor $u$, and it has a zero-gradient point with positive loss. Yet that point has an escaping curvature direction. *Zero gradient does not equal local minimum.* This calculation is a pedagogical full-observation example, not a proof of the paper's general theorems.
:::

## 3. The theorem in plain English

Under their hypotheses, any putative local minimum of the relevant factorized objectives must reconstruct $M^*$. Otherwise the authors can exhibit a direction that lowers the objective; in particular, a nonoptimal stationary point has a negative-curvature direction (or a suitable generalization). Thus **no spurious local minima** and **strict-saddle geometry** complement each other.

## 4. Why the geometry is favorable: the proof's central trick

The unifying idea is that the data-fit loss is quadratic in the matrix $M$ and has a Hessian operator $\mathcal{H}$ that approximately preserves squared Frobenius lengths *on relevant low-rank directions*. For a symmetric factorization $M = UU^\top$, first align the current factor with the true factor:

$$\Delta = U - U^*R, \qquad R \text{ orthogonal, chosen to minimize } \|U - U^*R\|_F.$$

Why align? $U$ and $UR$ represent the *same* matrix for orthogonal $R$; comparing raw factors without alignment can mistake equivalent solutions for different ones. **Lemma 7 (main identity):** For $M = UU^\top$ and the regularized objective, the Hessian quadratic form in the aligned direction $\Delta$ decomposes as

$$\begin{aligned} \langle \Delta, \nabla^2 F(U)\Delta\rangle = {} & \langle \Delta\Delta^\top, \mathcal{H}(\Delta\Delta^\top)\rangle - 3\langle M - M^*, \mathcal{H}(M - M^*)\rangle \\ & + 4\langle \nabla F_{\text{data}}(U), \Delta\rangle + \text{regularizer correction}. \end{aligned}$$

**Read the signs, not every symbol:** the first term is positive, but the *minus three times reconstruction error* term is negative. Under the norm-preservation and regularizer conditions, at a nonoptimal stationary point the negative part wins, proving negative curvature.

:::key Idealized case $\mathcal{H} = I$, no regularizer
At a stationary point, Lemma 7 and Lemma 6 yield

$$\langle \Delta, \nabla^2 F(U)\Delta\rangle = \|\Delta\Delta^\top\|_F^2 - 3\|M - M^*\|_F^2 \le -\|M - M^*\|_F^2.$$

If $M \neq M^*$, the right side is negative: the point cannot be a local minimum. This is the essence of the unified proof.
:::

## 5. RIP and the regularizer: what are they doing?

**Restricted Isometry Property (RIP).** For measurement matrices $A_i$, the paper assumes a nearly length-preserving observation map on low-rank matrices:

$$(1 - \delta)\|X\|_F^2 \le \frac{1}{m}\sum_{i=1}^{m}\langle A_i, X\rangle^2 \le (1 + \delta)\|X\|_F^2 \qquad (\mathrm{rank}\,X \le r).$$

**Intuition:** Two sufficiently different low-rank matrices should not look identical after measurement. Otherwise the observations would hide a direction of the solution space.

For asymmetric factors, the balancing penalty

$$Q_{\text{bal}}(U, V) = \tfrac{1}{8}\|U^\top U - V^\top V\|_F^2$$

discourages arbitrarily mismatched factor scales (e.g., replacing $U$ by $cU$ and $V$ by $V/c$). In matrix completion, *additional* regularizers keep factors in incoherent regions where observation geometry remains well behaved.

## 6. What is actually proved, for which problems?

| Problem / result | Key assumptions | Conclusion in the paper |
|---|---|---|
| **Matrix sensing, Theorem 3** | $(2r, 1/20)$-RIP for the measurement map; balanced asymmetric factors | Every local minimum has $UV^\top = M^*$; quantitative strict-saddle property. |
| **Matrix completion, Theorem 4** | $\mu$-incoherent target, sufficiently many uniformly sampled entries, regularizers and conditioning | With high probability, no spurious local minima and quantitative strict-saddle geometry. |
| **Robust PCA, Theorem 5** | Incoherent low-rank target, sufficiently sparse corruptions and bounded corruption assumptions | All local minima recover the low-rank component; *pseudo*-strict-saddle property. |

**Why "pseudo" for robust PCA?** The loss minimizes over a sparse correction $S$ and need not be twice differentiable in $(U, V)$. The authors use a smooth upper-bounding function touching the loss locally and test its negative Hessian curvature. Don't incorrectly describe Theorem 5 as an ordinary smooth strict-saddle theorem.

## 7. What the paper does not claim

- **Not universal:** remove the measurement/incoherence/sparsity assumptions and bad minima may appear; this is not a result about arbitrary neural-network losses.
- **Not "GD from any point always works":** at an exact saddle such as $U = 0$, unmodified GD can remain stationary. Polynomial-time global recovery relies on combining the landscape with suitable saddle-escaping algorithms and their conditions.
- **Not generalization:** recovery of the specified matrix is different from generalization guarantees for neural networks.
- **Not a new optimizer:** the contribution is chiefly a *unified landscape proof*, not a new SGD or perturbed-GD update.

## 8. Survey connections: the question this paper resolves

| Paper | How to connect them without overstating either result |
|---|---|
| **Safran–Shamir (2018)** | Some simple ReLU objectives *do* have bad local minima; the low-rank models here show an important favorable counterexample class. |
| **Lee (2016)** | Avoidance of strict saddles is useful if all remaining local minima are globally optimal. This paper supplies that geometry for specific models. |
| **Jin (2017)** | Finite-time saddle escape + benign strict-saddle landscape can imply efficient global recovery under additional regularity conditions. |
| **Du / Allen-Zhu (2019)** | Strong global training-loss results also require structured assumptions, but for wide neural-network architectures rather than matrix recovery. |

:::key Survey takeaway
The hard question is not "Is the objective non-convex?" but "What kinds of stationary points does its structure permit?" Ge et al. identify models with global local minima and escaping saddles; Safran–Shamir identify neural models with genuine bad minima.
:::

## 9. Quick reference

| Concept | Meaning in one line |
|---|---|
| Factorization | $M = UV^\top$ turns rank-restricted recovery into unconstrained, non-convex optimization. |
| No spurious minima | Every local minimum reconstructs $M^*$ under the stated assumptions. |
| Strict-saddle | Away from local minima: sizeable gradient or negative Hessian curvature. |
| RIP | Measurements approximately preserve the Frobenius norm of low-rank changes. |
| Incoherence | Matrix information is spread across coordinates, making entry sampling informative. |
| Main proof tool | Lemma 7: aligned-factor displacement creates a negative-curvature certificate. |

## 10. Papers Master note (purple column)

:::plain
Ge et al. (2017) prove no spurious local minima and (pseudo-)strict-saddle geometry in low-rank sensing, completion and robust PCA under structural assumptions. Lemma 7 gives a curvature certificate; this does not directly cover neural networks.
:::

> **Primary source:** Ge, Jin & Zheng (2017), *No Spurious Local Minima in Nonconvex Low Rank Problems: A Unified Geometric Analysis*, ICML 2017 (PMLR 70). [proceedings.mlr.press/v70/ge17a/ge17a.pdf](https://proceedings.mlr.press/v70/ge17a/ge17a.pdf).
