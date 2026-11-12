## 1. Diagonal curvature, honestly estimated

This chapter follows Sections V-D and V-E of the survey. Full dense curvature is not the only way to use second-order information. One class of methods estimates only the *diagonal* of the Hessian and combines it with an adaptive update:

- **AdaHessian** uses cheap stochastic estimates of the Hessian diagonal, smoothed over time, as its adaptive scale.
- **Apollo** learns diagonal quasi-Newton information from parameter and gradient changes (secant pairs, chapter 11) rather than from second derivatives.

The two differ both in what they estimate (sampled Hessian information versus secant updates) and in the guarantees attached to them. A diagonal approximation brings memory down to $O(d)$, but it misses off-diagonal interactions. The rotated valley of chapter 4 makes this concrete: even the *exact* Hessian diagonal, $50.5$ in both coordinates, cannot see the coupling $-49.5$. Conversely, a diagonal method may be the better choice when large structured factors are a bottleneck, or when noisy estimates of full curvature are unreliable. A more elaborate matrix should earn its place by a measurable benefit.

## 2. Hutchinson's estimator

How can the diagonal of $H$ be estimated without forming $H$? Draw a random **Rademacher** vector $z \in \{-1, 1\}^d$, with independent entries that are $\pm1$ with equal probability, and form (survey Section V-E)

$$\hat h = z \odot (Hz).$$

This is **[[hutchinson|Hutchinson's estimator]]**. It is unbiased because $\mathbb{E}[z_iz_j] = \mathbf{1}\{i = j\}$: the entries are uncorrelated and each squares to one. So

$$\mathbb{E}[z_i(Hz)_i] = \sum_j H_{ij}\,\mathbb{E}[z_iz_j] = H_{ii}.$$

Each probe needs one [[hvp|Hessian–vector product]], never the matrix. Two cautions from the survey. The estimate is unbiased for the diagonal of the *Hessian under the stated randomness*; it is not an estimate of a positive curvature matrix. And Hessian diagonals can be negative in non-convex regions, so clipping, regularization and the details of the estimator matter.

:::viz hutch
:::

:::read How to read this visual
**What is drawn.** A fixed $12\times12$ symmetric matrix $H$, standing in for a Hessian, with some negative diagonal entries. For each coordinate $i$, the white outline is the true $H_{ii}$ and the filled bar is Hutchinson's estimate averaged over the first $k$ probes (coral where positive, red where negative). The side chart shows the relative error of the whole estimated diagonal as $k$ grows (log-log), for one fixed sequence of probes. The dashed line is the exact expected error.

**What moves.** The number of probes $k$, from $1$ to $1000$.

**What to notice.** With one probe the estimate is rough, and several entries can even have the wrong sign; the readout counts them. The estimate is unbiased, so averaging more probes always helps. The error falls like $1/\sqrt k$, the dashed slope. The constant comes from the off-diagonal entries: one probe's error in coordinate $i$ has variance $\sum_{j\ne i}H_{ij}^2$, so a strongly coupled matrix needs more probes. The negative entries are real curvature, not noise. A method that divides by this estimate must deal with them, which is what Sophia's clipping does.
:::

## 3. Sophia: curvature scaling with explicit control of step size

**[[sophia|Sophia]]** targets efficient language-model pretraining. It keeps a momentum estimate $m_t$ and an occasional, lightweight diagonal Hessian estimate $h_t$, and takes an elementwise *clipped* preconditioned step. In the survey's simplified notation (survey Section V-E):

$$\Delta\theta_t = -\eta_t\,\mathrm{clip}\!\left(\frac{m_t}{\max\{h_t, \epsilon\}},\,-\rho,\,\rho\right),$$

where the maximum and the clipping act coordinate by coordinate. The two ingredients are *curvature scaling*, dividing by an estimate of the real second derivative, and *explicit control of step magnitude*, so no coordinate moves more than $\eta_t\rho$. The $\max\{h_t, \epsilon\}$ is the guard against the non-positive diagonal entries seen above. Unlike Adam, the denominator is not an average of squared gradients.

:::viz clip
:::

:::read How to read this visual
**What is drawn.** Why capping a step's size helps, in its simplest form. The loss $f(x, y) = x^2/2 + x^4/4 + y^2/2$ is steep far out along $x$ and gentle near the minimum (green), in 3D with a log height scale. Violet: gradient descent with one fixed step, which must be tiny to survive the steep start. Coral: clipped gradient descent, which takes the full step $\eta = 1$ near the minimum but at most a length $\gamma = 0.5$ when the gradient is large.

**What moves.** The start $x_0$ and the fixed step of the violet run.

**What to notice.** The fixed step is either too large for the steep region or too small for the flat one; the clipped step is safe in both. Sophia applies the same idea per coordinate, to a curvature-scaled step: where the curvature estimate is tiny or negative, $m_t/\max\{h_t, \epsilon\}$ would be huge, and the clip caps it at $\rho$. This visual clips plain gradient descent, not Sophia itself; it shows why the cap matters.
:::

## 4. What Sophia's results do and do not say

Sophia's original study reports about a two-fold improvement over its Adam baseline in GPT pretraining runs of 125M to 1.5B parameters, counting steps, compute and wall-clock time to a matched target. The survey calls this an **attributed experimental result**: it is not a guarantee that Sophia halves the cost of every model. The frequency of Hessian estimates, the clipping threshold, the precision and the model class all affect the conclusion. Chapter 17 runs a small Sophia-H on a real task next to six other optimizers.

:::problem The next step
The last two structured methods of the survey work on whole matrices again: SOAP rotates Adam into Shampoo's basis, and Muon orthogonalizes the update. Chapter 16.
:::
