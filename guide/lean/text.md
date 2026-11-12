## 1. Adafactor: factor the second moment

This chapter follows Sections III-I, III-J and III-K of the survey. An $m \times n$ weight matrix has $mn$ parameters, and Adam stores $mn$ first-moment and $mn$ second-moment values on top of them. **Adafactor** cuts the dominant [[optimizer-state|optimizer state]] by storing only the row and column sums of the second-moment matrix. If $V$ is a nonnegative $m\times n$ second-moment matrix, its [[adafactor-factored|factored reconstruction]] is (survey Section III-I)

$$\widetilde V_{ij} = \frac{r_i c_j}{\sum_k r_k}, \qquad r_i = \sum_j V_{ij}, \qquad c_j = \sum_i V_{ij}.$$

Why this formula? If $V$ were exactly a product $V_{ij} = u_i w_j$, then $r_i = u_i\sum_j w_j$, $c_j = w_j\sum_i u_i$ and $\sum_k r_k = \sum_i u_i\sum_j w_j$. Substituting gives $\widetilde V_{ij} = u_i w_j = V_{ij}$: the reconstruction is exact. For any other $V$ it is the best guess of that rank-one form. It is a reconstruction of the *second-moment scale*, not a factorization of the weight matrix or of the Hessian. Storing $r \in \mathbb{R}^m$ and $c \in \mathbb{R}^n$ takes $m + n$ numbers instead of $mn$.

:::viz factor
:::

:::read How to read this visual
**What is drawn.** Three colour maps, one cell per weight of an $8\times12$ matrix. Left: a nonnegative second-moment matrix $V$. Middle: Adafactor's reconstruction $\widetilde V$ from only its 8 row sums and 12 column sums. Right: the ratio $\widetilde V/V$ of the two, cell by cell, on a log colour scale from $\tfrac12$ to $2$; the middle colour means equal.

**What moves.** $s$, the amount of structure that is not a simple row-times-column product.

**What to notice.** At $s = 0$, $V$ is exactly rank one; the two maps are identical and the ratio map is uniform (the readout's error is zero). As $s$ grows, $V$ gets cell-by-cell detail that no row-times-column product can reproduce, and the reconstruction keeps only the overall pattern. The storage drops from 96 numbers to 20 at every $s$. How faithful the reconstruction is depends on the matrix's structure, which is the survey's caveat.
:::

Practical Adafactor adds more choices, such as update clipping and relative step sizes, and drops momentum in its lean configuration. The saving depends on shape: the ratio $(m+n)/(mn) = 1/m + 1/n$ is small for large matrices, but vectors cannot be factored two-sidedly at all. "Sublinear memory" describes the second-moment storage for large matrices, not zero extra memory overall.

## 2. Lion: a sign update with one state

**Lion** was not derived by approximating a Hessian; it was found by a symbolic search over programs. A simplified form of its update is (survey Eqs. 29 to 31)

$$c_t = \beta_1 m_{t-1} + (1 - \beta_1)\hat g_t, \qquad \theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta\,\mathrm{sign}(c_t), \qquad m_t = \beta_2 m_{t-1} + (1 - \beta_2)\hat g_t.$$

The two coefficients play different roles. $\beta_1$ blends the momentum with the fresh gradient to decide the **sign** of the step. $\beta_2$ updates the momentum itself. Except at zero, every coordinate moves by exactly $\eta$ (before weight decay). The gradient's magnitude influences the sign through the momentum, but it never scales the step. Lion keeps one momentum array instead of Adam's two.

:::viz lion
:::

:::read How to read this visual
**What is drawn.** One coordinate over 120 steps. The white dots are noisy gradients whose true mean is $0.3$ (green dashed line) but whose noise is four times larger. Violet: the momentum $m_t$ of Eq. (31). Coral: the blend $c_t$ of Eq. (29), whose sign Lion uses. The bars are the steps actually taken, in units of $\eta$: green when the step goes the right way (against the mean gradient), red when it goes the wrong way. Every bar has height exactly one.

**What moves.** $\beta_1$ and $\beta_2$.

**What to notice.** The bars never change height, whatever the gradient's size: Lion's update magnitude is fixed. The momentum averages away much of the noise, so the sign of $c_t$ is right far more often than the sign of the raw gradient. The readout counts both. Set $\beta_1 = 0$: the blend becomes the raw gradient and the count drops to the sign-SGD count. Set $\beta_2$ small: the momentum forgets quickly and becomes noisy too. Because the step size is $\eta$ in every coordinate, Lion's learning rate and weight decay must be tuned for that magnitude, not copied from AdamW.
:::

The Lion authors report gains on particular vision, multimodal and diffusion settings, and tasks where gains were small or inconclusive. So, the survey concludes, it would be misleading to call Lion categorically better than AdamW. Its memory saving is an accounting identity; its runtime and success depend on the model, hardware and tuning.

## 3. Other first-order designs

Optimizer development is not one monotone sequence. The survey names three more designs, each changing something other than the scaling:

- **D-adaptation** sets the learning rate automatically, with guarantees in specific convex analyses.
- **Schedule-free training** combines interpolation and averaging, so no stopping-time-dependent schedule is needed.
- **Lookahead** separates fast inner steps from a slower averaged state.

:::viz lookahead
:::

:::read How to read this visual
**What is drawn.** A narrow valley $f = \tfrac12(0.1x^2 + 10y^2)$ seen from above, with $\eta = 0.1$ and the start $(-1.45, 0.5)$. Coral: an over-eager heavy ball (high momentum), whose last 60 steps swing across the valley. Faint white: Lookahead's current inner run of $k$ fast steps (the same heavy ball). White dots: Lookahead's slow weights, which move only a fraction $\alpha$ of the way toward where each inner run ended. The chart shows $f$ on a log scale.

**What moves.** The momentum $\beta$, the number of fast steps $k$, and the slow step $\alpha$.

**What to notice.** The fast weights oscillate like the plain heavy ball, but the slow weights, which Lookahead reports, average over each inner run and move smoothly down the valley. Lookahead changes *which iterate* is kept, not how the gradient is scaled.
:::

:::viz three
:::

:::read How to read this visual
**What is drawn.** Schedule-free SGD on a noisy quadratic $\tfrac12(2x^2 + 0.25y^2)$, in 3D. Three sequences: the base sequence $z$ (violet) takes plain SGD steps with a constant learning rate and keeps jittering; the reported sequence $x$ (coral) is the running average of all the $z$'s; gradients are taken at $y$ (white), a mix $y = (1 - \beta)z + \beta x$. The chart compares the loss of $x$ and of $z$.

**What moves.** The mixing coefficient $\beta$.

**What to notice.** No learning-rate schedule and no stopping time are set in advance, yet the averaged $x$ keeps getting closer to the minimum while $z$ stays noisy. $\beta = 0$ is classical Polyak–Ruppert averaging.
:::

None of these designs reduces to "newer methods converge faster". An optimizer is also only one part of the training system: throughput depends on batch size, communication, precision and where parameters live.

## 4. A synthesis of the first-order families

Section III-K summarizes RQ1 by what each design stores:

| method | stored history | how the gradient is transformed |
|---|---|---|
| SGD | nothing beyond the iterate | scaled by one step size |
| momentum | one vector | discounted sum of past gradients |
| Adam | two vectors | momentum divided by a per-coordinate scale |
| Adafactor | row and column sums per matrix | divided by a factored scale |
| Lion | one vector | sign of a momentum blend |

All of them use only gradients, but their geometry and memory differ. Their success raises the question of RQ2: what more can be gained from true curvature, or from interactions between coordinates?

:::problem The next step
Section IV of the survey takes the second derivative seriously, starting with Newton's method.
:::
