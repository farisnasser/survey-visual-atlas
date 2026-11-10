## 1. SOAP: Adam in a rotated matrix basis

This chapter follows Sections V-F, V-G and V-H of the survey, with Appendices A-C and C-C. **[[soap|SOAP]]** builds on a connection between Shampoo's eigenbases and familiar adaptive updates. Suppose Shampoo's left and right factors (chapter 14) have eigendecompositions $L = U\Lambda U^\top$ and $R = V\Gamma V^\top$. The gradient expressed in their rotated coordinates is (survey Eq. 66)

$$\widetilde G = U^\top GV.$$

SOAP accumulates Adam-style moments of $\widetilde G$ in this evolving coordinate system, takes an adaptive step there, and rotates the step back to the original weight matrix. It is not merely "Adam plus an SVD": the update depends on the preconditioner's eigenbasis and on how that basis is tracked across steps. Its motivation is visible in chapter 4. A diagonal method fails on a rotated valley; in a basis aligned with the valley, a diagonal method is exactly right. SOAP lets Shampoo choose the basis and Adam do the per-coordinate scaling in it.

In the cited language-model experiments (360M and 660M parameters, large batch), SOAP's authors report over 40% fewer iterations and over 35% lower wall-clock time than AdamW. The two percentages differ, and the survey reads that as a lesson: faster per-step progress does not translate one-for-one into elapsed time. The basis is updated only periodically, and that frequency becomes another hyperparameter.

## 2. Muon: orthogonalize the update

**Muon** is the survey's most instructive modern case: it uses a matrix-valued transformation, yet it is not a Hessian approximation. It applies momentum to a hidden layer's weight gradients, then approximately orthogonalizes the resulting update matrix. Write the momentum matrix's singular value decomposition (survey Eq. 67), $M = U\Sigma V^\top$. The semi-orthogonal **[[polar-factor|polar factor]]** is (survey Eq. 68)

$$\mathrm{msign}(M) = UV^\top.$$

Where $M$ has nonzero singular values, this replaces all of them by one while keeping the left and right singular directions. So Muon keeps *which directions* the update acts in, and throws away *how strongly* it acts in each. That is different from elementwise normalization, and different from Adam's coordinate scaling.

## 3. Why the polar factor? Steepest descent in the spectral norm

The linearized change of the loss under a matrix step $\Delta$ is $\langle G, \Delta\rangle_F = \mathrm{tr}(G^\top\Delta)$. Now limit the step by its **[[spectral-norm|spectral norm]]** $\|\Delta\|_2$, its largest singular value. Von Neumann's trace inequality gives (survey Eq. 69)

$$\min_{\|\Delta\|_2 \le \eta}\langle G, \Delta\rangle_F = -\eta\sum_i\sigma_i(G),$$

attained by $\Delta = -\eta UV^\top$ when $G = U\Sigma V^\top$. Appendix A-C states the same fact through trace duality: $|\langle G, \Delta\rangle_F| \le \|G\|_*\|\Delta\|_2$, where $\|G\|_* = \sum_i\sigma_i(G)$ is the nuclear norm. In words, the polar factor is steepest descent for the linearized loss *when step size is measured by the spectral norm*. This is a different geometry from Euclidean steepest descent ($-G$), and a different mechanism from multiplying by $H^{-1}$. It is a fact about one linearized step, not a convergence theorem for networks.

:::viz spectral
:::

:::read How to read this visual
**What is drawn.** One fixed $6\times5$ gradient matrix $G$ with singular values $\sigma_i = 2i^{-p}$. Three candidate updates, each scaled so that its spectral norm is exactly $1$. The bar height is how much each lowers the linearized loss, $-\langle G, \Delta\rangle$.
- Violet: the scaled gradient $-G/\|G\|_2$.
- Amber: the elementwise sign, $-\mathrm{sign}(G)/\|\mathrm{sign}(G)\|_2$, the kind of step Lion or Adam's first step takes.
- Coral: the polar factor $-UV^\top$.

The dashed line is the bound $\sum_i\sigma_i(G)$ of Eq. (69).

**What moves.** The spectrum decay $p$.

**What to notice.** The coral bar always reaches the dashed line, and no other update does: Eq. (69) checked numerically. When the singular values are very unequal (large $p$), the scaled gradient wastes most of its budget on the top direction, and its bar falls far below. When they are all equal ($p = 0$), $G$ is already a multiple of $UV^\top$, so the first and third bars agree. The elementwise sign respects the coordinates, not the singular directions, so it is never the steepest step in this geometry. The note gives the Frobenius-norm counterpart: there $-G/\|G\|_F$ would win. Which step is "steepest" depends on how step size is measured.
:::

## 4. Newton–Schulz: orthogonalizing with matrix products only

An exact SVD at every step would be expensive. Muon approximates the polar factor with Newton–Schulz iterations, which use only matrix multiplications. Appendix C-C explains the basic version. Normalize the matrix, $X_0 = M/\|M\|_F$, so that every singular value lies in $(0, 1]$, and iterate (survey Eq. 79)

$$X_{k+1} = \tfrac12X_k(3I - X_k^\top X_k).$$

If $X_k = U\,\mathrm{diag}(s_i)\,V^\top$, then $X_k^\top X_k = V\,\mathrm{diag}(s_i^2)\,V^\top$, and the iteration keeps $U$ and $V$ and changes each singular value by a scalar map (survey Eq. 80):

$$s_{i,k+1} = \tfrac12s_{i,k}(3 - s_{i,k}^2).$$

The fixed point $s = 1$ is attractive: the map's derivative, $\tfrac32(1 - s^2)$, vanishes at $s = 1$. So errors near one shrink very fast, quadratically. This is why matrix-multiplication iterations can approximate polar factors without an SVD.

:::viz ns
:::

:::read How to read this visual
**What is drawn.** The 12 singular values of a $16\times12$ momentum matrix, after dividing by its Frobenius norm (grey bars, from $0.687$ down to $0.021$), and the singular values after $k$ Newton–Schulz steps (coloured bars). Because Eq. (80) acts on each singular value separately, the coloured bars are computed exactly by iterating the scalar map on the spectrum. The dashed line at $1$ is the exact polar factor $UV^\top$.

**What moves.** The number of steps $k$, and the iteration: the cubic of Eq. (79), or the quintic polynomial with the coefficients used by the Muon implementation.

**What to notice.**
- **The cubic.** Large singular values reach $1$ within a few steps. Small ones grow by only about $\times1.5$ per step while they are small, so they lag: after 5 steps the smallest is still well below $1$. After about a dozen steps all are essentially $1$; the readout's distance to $UV^\top$ goes to zero. Each step costs two matrix multiplications.
- **The quintic.** It is tuned to lift small singular values much faster. After 5 steps all of them are already in a band around $1$, but it does not converge to exactly $1$. Muon accepts an approximate orthogonalization in exchange for fewer steps.

The survey stresses that Eq. (79) is a didactic example, not Muon's production code. Muon uses its own coefficients, scaling, iteration counts and safeguards.
:::

:::survey Fig. 8 of the survey: matrix geometry in Muon. Orthogonalization changes the singular spectrum of the proposed weight update without estimating the model's Hessian.
public/survey3/fig8.webp
:::

:::read How to read this figure
**Top.** Muon's pipeline: the gradient $G_t$ feeds a momentum $M_t = \beta M_{t-1} + G_t$, which is orthogonalized (by SVD or Newton–Schulz) into the update $\Delta W = -\eta UV^\top$.

**Bottom.** The static version of the visual above, for the same $16\times12$ matrix. Grey: the normalized singular values of $M$. Blue: the cubic iteration after 5 steps. Light blue: after 12 steps. Pink: the quintic after 5 steps. The dashed line is the exact polar factor. As in the visual, the cubic lags on small singular values after 5 steps and is essentially exact after 12, and the quintic spreads them in a band around $1$.
:::

:::survey Fig. 13 of the survey: the scalar Newton–Schulz map of Eq. (80). (a) Cobweb iterations from 0.2 and 1.62, both drawn to the fixed point 1. (b) Distance to one: slow while small, then very fast.
public/survey3/fig13.webp
:::

:::read How to read this figure
**(a)** The map $s \mapsto \tfrac12s(3 - s^2)$ (black) and the diagonal $s_{k+1} = s_k$ (grey). A cobweb diagram traces the iteration: go up to the curve, across to the diagonal, repeat. From $s_0 = 0.2$ (blue) the staircase climbs to the fixed point $1$; from $s_0 = 1.62$ (vermillion) the first step drops low, then climbs too. Starting values must stay below $\sqrt3$, where the map turns negative.

**(b)** $|s_k - 1|$ on a log scale for five starting values. Each curve is flat at first (slow growth, about $\times1.5$ per step while $s$ is small) and then plunges once $s_k$ is near $1$: the quadratic convergence that the zero slope at $s = 1$ predicts.
:::

## 5. What Muon is not

The name "Newton–Schulz" evokes Newton's method, but it does not turn Muon into Newton optimization with the network's Hessian. The orthogonalization is applied to the *momentum matrix*, an operation on the update, not to the loss's curvature. Two practical cautions follow:

- **Rank deficiency.** For a matrix whose rank is lower than its smaller dimension, the polar factor is not unique outside the supported subspace, and the iteration needs care near zero singular values.
- **Parameter groups.** Implementations apply Muon to suitable hidden weight matrices and keep AdamW for embeddings, biases, gains and output heads. Not every parameter is eligible for a matrix operation.

## 6. Inverted, estimated, or normalized?

Section V-H closes the structured methods with a warning against one box labelled "advanced second-order optimization". Each method does something different:

| method | what it uses | operation |
|---|---|---|
| K-FAC | Fisher blocks, as Kronecker factors | inverts factors |
| Shampoo | accumulated gradient products | applies inverse fourth roots |
| SOAP | Adam moments in Shampoo's eigenbasis | adaptive step in a rotated basis |
| Sophia | diagonal Hessian estimates | clipped division |
| Muon | the momentum matrix | normalizes singular values |

Some keep an $O(d)$ diagonal, some $O(m^2 + n^2)$ factors, and some keep no inverse-root matrices but spend matrix multiplications at every step. There is also a distinction between *preconditioning*, which multiplies the gradient by a matrix meant to rescale directions, and *normalization*, which replaces magnitudes by a prescribed norm or sign. Both change the update, but they optimize different local models and need different convergence analyses.

:::problem The next step
With every family in place, Section VI of the survey compares them by memory, cost, guarantees and measured runs. Chapter 17.
:::
