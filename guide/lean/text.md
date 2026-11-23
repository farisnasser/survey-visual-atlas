## 1. Adafactor: Adam's memory problem for large matrices

This chapter follows Sections III-I, III-J and III-K of the survey. Take a layer with weight matrix $W \in \mathbb{R}^{m\times n}$. Adam keeps a second-moment value for every weight, $mn$ numbers for that statistic alone: a $1000\times1000$ matrix needs a million. **[[adafactor-factored|Adafactor]]** asks whether that matrix of squared-gradient scales can be approximated by two small vectors, one along the rows and one along the columns.

**A toy example.** Take the squared-gradient matrix

$$Q = \begin{bmatrix}4 & 8 & 12\\ 2 & 4 & 6\end{bmatrix}.$$

Its row sums are $r = (24, 12)^\top$, its column sums $c = (6, 12, 18)^\top$, and its total is $36$. A simple rank-one reconstruction is

$$\tilde Q = \frac{r\,c^\top}{\sum_i r_i} = \frac{1}{36}\begin{bmatrix}24\\12\end{bmatrix}\begin{bmatrix}6 & 12 & 18\end{bmatrix} = \begin{bmatrix}4 & 8 & 12\\ 2 & 4 & 6\end{bmatrix}.$$

It rebuilds this matrix exactly, because its rows follow a rank-one pattern. Real squared-gradient matrices are not rank one, so in practice this is an approximation, with its own normalization and stabilization details.

**What was saved?** Instead of $mn$ statistics, about $m + n$: for a $1000\times1000$ matrix, roughly $2000$ numbers instead of a million for that statistic. Practical versions add update clipping and scale-aware learning rates, and may drop the first moment to save even more, so the real saving depends on the configuration and the shapes of the tensors.

:::viz factor
:::

:::read How to read this visual
**What is drawn.** Three colour maps, one cell per weight of an $8\times12$ matrix. Left: a nonnegative second-moment matrix $V$. Middle: Adafactor's reconstruction $\widetilde V$ from only its 8 row sums and 12 column sums. Right: the ratio $\widetilde V/V$ of the two, cell by cell, on a log colour scale from $\tfrac12$ to $2$; the middle colour means equal.

**What moves.** $s$, the amount of structure that is not a simple row-times-column product.

**What to notice.** At $s = 0$, $V$ is exactly rank one; the two maps are identical and the ratio map is uniform (the readout's error is zero). As $s$ grows, $V$ gets cell-by-cell detail that no row-times-column product can reproduce, and the reconstruction keeps only the overall pattern. The storage drops from 96 numbers to 20 at every $s$. How faithful the reconstruction is depends on the matrix's structure, which is the survey's caveat.
:::

:::key Remember III-I
Adafactor is Adam-like adaptive scaling with a memory-efficient row and column approximation to the squared-gradient statistics. It is not a Hessian estimate, and it trades precision of the scale for memory.
:::

## 2. Lion: sign-based momentum with one main state

Adam carries two moment states. **[[lion|Lion]]** goes the other way and simplifies the update itself: it keeps momentum-like history and uses only the *sign* of a blended direction (survey Eqs. 29 to 31):

$$c_t = \beta_1 m_{t-1} + (1 - \beta_1)\hat g_t, \qquad \theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta\,\operatorname{sign}(c_t), \qquad m_t = \beta_2 m_{t-1} + (1 - \beta_2)\hat g_t.$$

The two coefficients play different roles: $\beta_1$ blends for the current step, $\beta_2$ updates the stored state.

**What taking the sign does.** Suppose $c_t = (8, -0.2, 3)^\top$. Then $\operatorname{sign}(c_t) = (1, -1, 1)^\top$, and with $\eta = 0.01$ (ignoring decay) the step is $(-0.01, +0.01, -0.01)^\top$. Every nonzero coordinate moves by exactly $\eta$; only the direction differs. Gradient magnitudes still shape the accumulated direction, but not the size of the final step. This is neither ordinary gradient normalization nor Hessian preconditioning: it deliberately discards magnitude information.

:::viz lion
:::

:::read How to read this visual
**What is drawn.** One coordinate over 120 steps. The white dots are noisy gradients whose true mean is $0.3$ (green dashed line) but whose noise is four times larger. Violet: the momentum $m_t$ of Eq. (31). Coral: the blend $c_t$ of Eq. (29), whose sign Lion uses. The bars are the steps actually taken, in units of $\eta$: green when the step goes the right way (against the mean gradient), red when it goes the wrong way. Every bar has height exactly one.

**What moves.** $\beta_1$ and $\beta_2$.

**What to notice.** The bars never change height, whatever the gradient's size: Lion's update magnitude is fixed. The momentum averages away much of the noise, so the sign of $c_t$ is right far more often than the sign of the raw gradient. The readout counts both. Set $\beta_1 = 0$: the blend becomes the raw gradient and the count drops to the sign-SGD count. Set $\beta_2$ small: the momentum forgets quickly and becomes noisy too. Because the step size is $\eta$ in every coordinate, Lion's learning rate and weight decay must be tuned for that magnitude, not copied from AdamW.
:::

Lion was found by symbolic program search, and its original study reported gains in several settings. A compact state is not automatically better, though: the fixed step size changes how the learning rate and the weight decay must be tuned, and results depend on the task.

:::key Remember III-J
Lion uses momentum to decide the sign of each move instead of keeping Adam's separate per-coordinate scale. One state vector, but magnitudes are thrown away.
:::

## 3. Other modern designs and a synthesis

First-order design did not stop there. D-Adaptation aims to reduce manual learning-rate selection in specific theoretical settings; schedule-free optimization reduces reliance on predefined learning-rate schedules; Lookahead combines fast weights with a slowly moving copy. Variants such as RAdam, AdaBelief and AdaBound change how gradient history is used for stability or scaling. None of these needs to be treated as a separate mathematical framework: they change what is remembered, forgotten, or scaled.

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

**The complete first-order roadmap.**

| method | what it remembers | why it exists | key limitation |
|---|---|---|---|
| GD | nothing | follow the full gradient | full-gradient cost; conditioning |
| SGD | nothing | cheaper minibatch updates | sampling noise |
| momentum | a running direction | use persistent movement | overshoot; one state vector |
| Nesterov | the previous movement | look-ahead gradient | tuning; guarantees need assumptions |
| AdaGrad | cumulative squares | per-coordinate scales | old history never fades |
| RMSProp | moving squares | scales follow recent history | noise and tuning |
| Adam | first and second moments | direction plus scaling | two states; convergence caveats |
| AMSGrad | running maximum of $v$ | control one scaling instability | not a universal guarantee |
| AdamW | Adam states, decoupled decay | separate weight shrinkage | still adaptive-scaling dependent |
| Adafactor | factored second moment | less optimizer memory | approximation and tuning |
| Lion | one momentum state | simple sign update | magnitudes discarded |

:::idea In plain English
Section III is the story of making better use of gradients without ever computing the loss's Hessian. Each optimizer decides what to remember, what to forget, what to scale, and how much memory to spend doing it.
:::

:::key What this section prepares us for
Every method so far uses gradients and their history. None describes how the gradient *changes* across directions, which is exactly the information that matters in narrow or rotated valleys. That is curvature, and Section IV turns to it.
:::

:::problem The next step
Chapter 10 begins Section IV with Newton's method: what the Hessian adds, and what it costs.
:::
