## 1. SOAP: Adam-style adaptation in a rotated basis

This chapter follows Sections V-F, V-G and V-H of the survey. **[[soap|SOAP]]** combines two familiar ideas: Adam adapts coordinates from gradient history, and Shampoo finds informative matrix directions. Its question: what if Adam's per-coordinate adaptation were done in a better coordinate system?

In chapter 4's rotated valley the important directions are diagonal combinations of the weights, while Adam's second moments live on the original axes. A change of basis does not change the gradient information, only the numbers that describe it, like describing the same walk in east/north coordinates or along a diagonal road.

**Where the basis comes from.** Shampoo's factors are symmetric, so $L = U\Lambda U^\top$ and $R = V\Gamma V^\top$, with eigenvectors $U$ (row side) and $V$ (column side). SOAP expresses the gradient in that basis (survey Eq. 38),

$$\widetilde G_t = U^\top G_t V,$$

runs Adam-style moment updates on $\widetilde G_t$, and rotates the resulting update $\widetilde D$ back: $D = U\widetilde DV^\top$ and $W \leftarrow W - \eta D$. Gradient, rotate, adapt, rotate back, update.

**A change of basis in numbers.** Let $G = \begin{bmatrix}3 & 1\\ 1 & 3\end{bmatrix}$ and $U = V = \tfrac{1}{\sqrt2}\begin{bmatrix}1 & 1\\ 1 & -1\end{bmatrix}$ (orthogonal, and here $U^\top = U$). Then

$$\widetilde G = U^\top G V = \begin{bmatrix}4 & 0\\ 0 & 2\end{bmatrix}:$$

a gradient with off-diagonal entries became diagonal in the new basis. (SOAP's basis comes from accumulated statistics, so it does not diagonalize every gradient.) If the adaptation there produces, say, $\widetilde D = \mathrm{diag}(1, 2)$, rotating back gives

$$D = U\widetilde DV^\top = \begin{bmatrix}1.5 & -0.5\\ -0.5 & 1.5\end{bmatrix},$$

an update that mixes the original weights even though it was diagonal in the rotated basis.

**Not simply Shampoo.** Shampoo applies the inverse roots directly, $L^{-1/4}GR^{-1/4}$; SOAP uses the eigenbasis of similar statistics and applies Adam's moments there. Updating the basis has a cost, and the stored moments must be handled consistently when it changes. In the original study, SOAP reported more than 40% fewer iterations and over 35% less wall-clock time than AdamW in large-batch language-model settings, results tied to that setup.

:::key Remember V-F
SOAP changes *where* coordinate-wise adaptation happens: in a basis defined by Shampoo-like statistics, at the cost of tracking that basis.
:::

## 2. Muon: what an update matrix's singular values are

**[[muon|Muon]]** does not approximate the Hessian or the Fisher matrix. It changes the singular-value structure of the proposed update. Start with $G = \mathrm{diag}(4, 1)$: GD with $\eta = 0.1$ takes $-0.1G = \mathrm{diag}(-0.4, -0.1)$, four times more along the first direction. Muon's question is different from Newton's: should the singular values of the update decide how strongly each matrix direction is used?

**Orthogonality.** A square orthogonal matrix has $Q^\top Q = I$ and preserves lengths: $Q = \begin{bmatrix}0 & -1\\ 1 & 0\end{bmatrix}$ turns $(3, 4)$ into $(-4, 3)$, both of length $5$.

**The singular value decomposition.** Every real matrix can be written (survey Eq. 39)

$$M = U\Sigma V^\top.$$

Picture the matrix turning a circle into an ellipse: $V$ gives the special input directions, $\Sigma$ the nonnegative stretches along them, $U$ the output directions. For $M = \mathrm{diag}(4, 1)$ the singular values are $4$ and $1$. Muon's idealized step replaces every nonzero singular value by one (survey Eq. 40):

$$\operatorname{msign}(M) = UV^\top,$$

the **[[polar-factor|polar factor]]**; here $U = V = I$, so it is $I$. Applied to $G$ above, the step becomes $-\eta I = \mathrm{diag}(-0.1, -0.1)$ instead of $\mathrm{diag}(-0.4, -0.1)$: not a smaller learning rate, but a change in the *relative* sizes of the singular directions. Real Muon orthogonalizes a momentum matrix, $M_t = \beta M_{t-1} + (1 - \beta)G_t$, and steps by $-\eta\operatorname{msign}(M_t)$, with extra choices such as Nesterov momentum, scaling and weight decay.

:::survey Fig. 8 of the survey: matrix geometry in Muon. Orthogonalization changes the singular spectrum of the proposed weight update without estimating the model's Hessian; Newton–Schulz iterations are a numerical device for the matrix operation.
public/survey3/fig8.webp
:::

:::read How to read this figure
**Top.** Muon's pipeline: the gradient $G_t$ feeds a momentum $M_t = \beta M_{t-1} + G_t$, which is orthogonalized (by SVD or Newton–Schulz) into the update $\Delta W = -\eta UV^\top$.

**Bottom.** The static version of the visual above, for the same $16\times12$ matrix. Grey: the normalized singular values of $M$. Blue: the cubic iteration after 5 steps. Light blue: after 12 steps. Pink: the quintic after 5 steps. The dashed line is the exact polar factor. As in the visual, the cubic lags on small singular values after 5 steps and is essentially exact after 12, and the quintic spreads them in a band around $1$.
:::

## 3. Why equalizing singular values can be useful

For a matrix update the linear model is $L(W + \Delta W) \approx L(W) + \langle G, \Delta W\rangle_F$, where $\langle\cdot,\cdot\rangle_F$ multiplies matching entries and adds them. With $G = \mathrm{diag}(4, 1)$ that is $4\Delta w_{11} + \Delta w_{22}$. Now limit the **[[spectral-norm|spectral norm]]**, the largest singular value, to $\|\Delta W\|_2 \le 0.1$. Both $\Delta W_A = \mathrm{diag}(-0.1, 0)$ and $\Delta W_B = \mathrm{diag}(-0.1, -0.1)$ fit the budget, but they predict changes of $-0.4$ and $-0.5$: the second uses both directions for a larger decrease without exceeding the largest-singular-value limit. In general, for $G = U\Sigma V^\top$,

$$\min_{\|\Delta W\|_2 \le \eta}\langle G, \Delta W\rangle_F = -\eta\sum_i\sigma_i(G), \qquad \text{attained by } \Delta W = -\eta\,UV^\top.$$

That is the spectral-norm reading of Muon's step. It is exact for the current gradient's SVD in one linearized step; practical Muon orthogonalizes momentum, so it is not a guarantee for every real step.

:::viz spectral
:::

:::read How to read this visual
**What is drawn.** One fixed $6\times5$ gradient matrix $G$ with singular values $\sigma_i = 2i^{-p}$. Three candidate updates, each scaled so that its spectral norm is exactly $1$. The bar height is how much each lowers the linearized loss, $-\langle G, \Delta\rangle$.
- Violet: the scaled gradient $-G/\|G\|_2$.
- Amber: the elementwise sign, $-\mathrm{sign}(G)/\|\mathrm{sign}(G)\|_2$, the kind of step Lion or Adam's first step takes.
- Coral: the polar factor $-UV^\top$.

The dashed line is the bound $\sum_i\sigma_i(G)$ above.

**What moves.** The spectrum decay $p$.

**What to notice.** The coral bar always reaches the dashed line, and no other update does: the bound above, checked numerically. When the singular values are very unequal (large $p$), the scaled gradient wastes most of its budget on the top direction, and its bar falls far below. When they are all equal ($p = 0$), $G$ is already a multiple of $UV^\top$, so the first and third bars agree. The elementwise sign respects the coordinates, not the singular directions, so it is never the steepest step in this geometry. The note gives the Frobenius-norm counterpart: there $-G/\|G\|_F$ would win. Which step is "steepest" depends on how step size is measured.
:::

## 4. Newton–Schulz: orthogonalizing with matrix products only

A full SVD at every step would be expensive. Muon instead approximates the polar factor with polynomial matrix iterations of the **[[newton-schulz|Newton–Schulz]]** type, using only matrix multiplications. A classical cubic version, after suitable scaling, is

$$X_{k+1} = \tfrac12X_k\big(3I - X_k^\top X_k\big),$$

which acts on each singular value separately as $\sigma_{k+1} = \tfrac12\sigma_k(3 - \sigma_k^2)$: $0.5$ becomes $0.6875$ and $0.9$ becomes $0.9855$, both moving toward $1$. Practical Muon uses different polynomial coefficients tuned for fast, approximate orthogonalization. Despite the name, this is not Newton's optimization method: Newton models the loss with a Hessian, while this iteration is only a numerical device for transforming the update matrix. Muon is meant for suitable matrix weights, such as hidden layers; other parameters are often trained with AdamW.

:::viz ns
:::

:::read How to read this visual
**What is drawn.** The 12 singular values of a $16\times12$ momentum matrix, after dividing by its Frobenius norm (grey bars, from $0.687$ down to $0.021$), and the same values after $k$ Newton–Schulz steps (coral). The dashed line at $1$ is the polar factor's target.

**What moves.** The number of steps $k$, and the iteration: the classical cubic above, or the quintic polynomial with Muon's coefficients.

**What to notice.** The cubic pulls the large singular values to $1$ quickly but the small ones slowly: after $5$ steps the smallest is still far from $1$, and after $12$ all lie within $0.002$ of it. Muon's quintic moves every value into a band around $1$ (about $0.68$ to $1.14$) in $5$ steps, without converging exactly. That is the trade the survey points to: approximate orthogonalization with a handful of matrix products.
:::

:::key Remember V-G
Muon changes the shape of the update matrix, not a curvature estimate: it keeps the singular directions and sets the singular values toward one. Newton–Schulz is matrix arithmetic, not Newton's method.
:::

## 5. What is being inverted, estimated, or normalized?

Every method still produces $\theta_{t+1} = \theta_t + \Delta_t$, but the transformations do not mean the same thing:

- **Curvature estimation.** Newton solves with the true Hessian; on $\tfrac12(100x^2 + y^2)$ its inverse scales the directions by $0.01$ and $1$. AdaHessian and Sophia seek cheaper diagonal Hessian information; BFGS and Apollo learn it from changes in weights and gradients. K-FAC also uses inverse-like scaling, but of a Fisher block, which is why not every inverse-matrix update is "Newton's method".
- **Historical gradient statistics.** Shampoo's $L_t$ and $R_t$ are second moments of observed gradients, not second derivatives of the loss; SOAP adapts with Adam's moments in their eigenbasis.
- **Matrix normalization.** Muon sends $M = U\Sigma V^\top$ toward $UV^\top$. Ordinary vector normalization sends $(3, 4)$ to $(0.6, 0.8)$, keeping the direction and fixing the length; Muon's polar step does the matrix version, acting on the singular values.

A *preconditioner* rescales, and can rotate or mix, gradient directions through an operator built from information; a *normalization* imposes a prescribed scale or spectrum. Both change the update, with different mathematics.

| optimizer | information source | main transformation |
|---|---|---|
| Newton | exact loss Hessian | solve a curvature system |
| BFGS / L-BFGS | weight and gradient differences | quasi-Newton inverse curvature |
| K-FAC | approximate Fisher factors | two-sided Fisher-style preconditioning |
| Shampoo | accumulated gradient products | two-sided inverse-root preconditioning |
| SOAP | factor eigenspaces plus adaptive moments | rotate, adapt, rotate back |
| AdaHessian / Sophia | approximate Hessian diagonal | per-coordinate curvature scaling; Sophia clips |
| Muon | momentum matrix | polar-factor normalization |

Costs belong in the comparison: a dense curvature matrix has $O(d^2)$ entries, diagonal estimates $O(d)$, two layer factors $O(m^2 + n^2)$ plus inverse roots and decompositions, and Muon adds repeated matrix products. Fewer iterations do not automatically mean less wall-clock time.

:::key The lesson of Section V
The defining question is not whether an optimizer uses a matrix. It is what that matrix represents, why transforming the gradient with it should help, and whether the benefit justifies the cost.
:::

:::problem The next step
Section VI compares all these families on equal terms: memory, cost, guarantees, stability, and a reproducible protocol. Chapter 17.
:::
