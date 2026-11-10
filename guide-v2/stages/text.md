## 1. Training starts at a saddle

This chapter follows Section III-E of the survey, the last part of RQ1. It changes the meaning of a saddle.

Escape time matters most at initialization. Networks are usually initialized with small weights, and the origin of weight space, all weights zero, is a critical point of most architectures. For example, in every deep linear network the gradient vanishes when all weights are zero. **Training from small initialization therefore begins near a saddle.**

Saxe, McClelland & Ganguli (2014) derived exact solutions for this regime in deep linear networks, $f(x) = W_2W_1x$, with whitened inputs (inputs rescaled so that their covariance is the identity). Write the input–output correlation matrix through its singular value decomposition $\Sigma_{yx} = USV^\top$, with singular values $s_1 > s_2 > \cdots$. Each singular value is the strength of one input–output "mode", one pattern in the data. For a network with one hidden layer initialized in a suitably aligned way, learning decouples into independent modes. Each mode $k$ has a strength $a_k$, the product of the corresponding input and output weights, and in continuous time it evolves as

$$\tau\frac{da_k}{dt} = 2a_k(s_k - a_k), \qquad a_k(t) = \frac{s_k\,e^{2s_kt/\tau}}{e^{2s_kt/\tau} - 1 + s_k/a_k(0)},$$

where $\tau$ is the inverse learning rate. This is survey Eq. (16). Each mode follows a **sigmoid**. It stays near its small initial value for a time of order $(\tau/2s_k)\log(s_k/a_k(0))$, then rises rapidly to $s_k$. Three consequences:

- **Stronger modes are learned first**, since the waiting time scales like $1/s_k$.
- **Smaller initializations mean longer plateaus**, since the waiting time grows like $\log(1/a_k(0))$. This is the escape-time formula of chapter 5 again: a small start is a small $\delta$.
- Saxe et al. also showed that random **orthogonal** initializations can make learning time nearly independent of depth, an early form of the dynamical-isometry principle.

:::viz hyperbolas
:::

:::read How to read this visual
**What is drawn.** The loss of a single mode, $\tfrac12(s - ab)^2$ with $s = 1$, over the plane of its two weights: $a$ for the first layer and $b$ for the second. The green curve $ab = s$ is the valley of global minima. The faint gray curves are hyperbolas $a^2 - b^2 = \text{const}$. The origin is a saddle.

**What moves.** *Drop from small init* releases balls near the origin, at a distance set by *init scale*, and runs gradient descent.

**What to notice.** Every ball slides along its own hyperbola into the valley. Gradient flow conserves $a^2 - b^2$, a consequence of the symmetry $a \to ca$, $b \to b/c$ that leaves the loss unchanged. The readout checks this conservation for one ball. The smaller the init scale, the longer the ball lingers near the origin before it leaves.
:::

:::viz plateaus
:::

:::read How to read this visual
**What is drawn.** The mode strength $u(t)$ against time, for four modes of different strengths $s$. The lines are the exact solution of survey Eq. (16). The dots are discrete gradient descent on the same loss, started at $a = b = \sqrt{u_0}$ with step $0.01$.

**What to try.** Move the initial strength $u_0$ from $10^{-1}$ down to $10^{-6}$.

**What to notice.** Each curve is a sigmoid: a long flat plateau, then a sudden jump to its target $s$. The strongest mode jumps first and the weakest last. Smaller $u_0$ shifts every jump later by the same logarithmic amount. The readout gives the predicted learning times $(1/2s)\ln(s/u_0)$, which match where the jumps happen. The dots sit on the lines: the continuous solution describes the discrete algorithm.
:::

## 2. From one saddle to the next

Equation (10) produces **[[saddle-to-saddle]] dynamics**. The network sits near the saddle at the origin, escapes along the direction of the strongest mode, then lingers near a new saddle that corresponds to a rank-one solution (only the first mode learned), escapes again along the second mode, and so on. The loss curve becomes a staircase.

Zhang, Saxe & Latham (2026) placed this phenomenon in a general framework that covers fully connected, convolutional and attention-based architectures. Their key observation is that the fixed points and invariant manifolds of a wide network include those of narrower networks. A wide network can therefore behave, for a while, exactly like a narrower one. Gradient descent evolves near such an invariant manifold, approaches a saddle, and switches to the next manifold. As a result:

- linear networks learn solutions of increasing **rank**;
- ReLU networks learn solutions with an increasing number of **kinks**;
- convolutional networks learn an increasing number of **kernels**;
- attention models learn an increasing number of **heads**.

The framework separates two mechanisms. **Data-induced** saddle-to-saddle dynamics, ordered by the strength of the data's modes, lead to low-rank weights. **Initialization-induced** dynamics, ordered by which unit happened to start largest, lead to sparse weights. It predicts how the data distribution and the initialization affect the number and duration of the plateaus. For shallow ReLU-like networks, Wu, Şimşek & Ged (2025) refined the description of this process from vanishing initialization by ruling out escape types that earlier analyses had not excluded.

:::viz manifold
:::

:::read How to read this visual
**What is drawn.** The exact loss of a two-layer linear network, restricted to an invariant manifold with one coordinate per data mode: $a_1$ for the strong mode ($s_1 = 2$) and $a_2$ for the weak one ($s_2$, your slider). Balanced, aligned weights stay balanced and aligned under gradient flow (their Theorem 3), so this two-dimensional surface is the true loss along the path, not a slice. The white points are saddles: the origin (width 0, nothing learned) and the point where only the strong mode is learned (width 1). Green: the global minima. The red and blue tubes are the negative and positive curvature directions at the saddles. The chart plots the loss along the run.

**What moves.** *Restart from tiny weights* drops the coral ball next to the origin and runs gradient descent.

**What to try.** Change the weak mode $s_2$ and restart.

**What to notice.** The ball first leaves the origin along the strong mode, pauses at the width-1 saddle, then leaves along the weak mode to the minimum. On the chart this is a staircase: plateau, drop, plateau, drop. The readout shows $a_1^2$ reaching $s_1 = 2$ long before $a_2^2$ reaches $s_2$. A weaker second mode makes the middle plateau longer.
:::

:::viz growth
:::

:::read How to read this visual
**What is drawn.** Two networks trained from small Gaussian initialization. The main plot is a linear network learning three data modes whose strengths follow a power law $s \propto n^{-\theta}$; each curve is one mode's strength over time. The chart below is a network that is quadratic in its weights, $f = \sum_i v_i(u_i\cdot x)^2$, a model of linear self-attention. Each curve is one unit's size $\lVert(v_i, u_i)\rVert$ over time.

**What to try.** Change the power-law exponent $\theta$ and the init scale, then *Re-run*.

**What to notice.** On the left, the modes are learned one at a time **in order of their strength**: the data decides the order. On the right, units grow one at a time **in order of their initial size**: the initialization decides, a rich-get-richer effect. Both produce stage-like loss curves. With $\theta = 0$, all modes are equally strong and the linear network's plateaus disappear, but the quadratic network's do not. A larger init scale weakens the plateaus in both.
:::

## 3. Obstacle or mechanism?

This line of work changes the meaning of a saddle. In the escape literature of chapters 7 and 8, a saddle is an obstacle to be left as quickly as possible. In the saddle-to-saddle view, **the sequence of saddles visited is the mechanism** by which networks acquire progressively more complex functions: first the strongest pattern, then the next. This is a form of **[[simplicity-bias|simplicity bias]]**: the network tries simple solutions before complex ones.

An optimizer that escaped every saddle instantly might change which solution is found. The two views are in tension, and chapter 16 returns to this as an open problem: is there a trade-off between escape speed and the simplicity bias that incremental learning provides?

:::survey Survey Fig. 6. Original simulation of saddle-to-saddle learning in linear networks trained by GD from small random initialization, on a target map with singular values (3, 1.6, 0.8, 0.4).
public/survey/saddle_to_saddle.webp
:::

:::read How to read this figure
**Panel (a).** The training loss against the GD iteration, for a 2-layer (solid blue) and a 3-layer (dashed purple) linear network. The loss falls in a staircase. Each plateau sits at a saddle at which the end-to-end map has rank 0, 1, 2 or 3: rank 0 before anything is learned, rank 1 after the first mode, and so on.

**Panel (b).** For the two-layer network, the four singular values of the end-to-end map $W_2W_1$ over the iterations. Each rises in a sigmoid from almost zero to its target, $s_1 = 3.0$ (red), $s_2 = 1.6$ (yellow), $s_3 = 0.8$ (green), $s_4 = 0.4$ (teal), shown as dotted lines. They are learned one at a time, strongest first, following the sigmoidal law of survey Eq. (16). The weakest mode takes several times longer than the strongest; their strengths differ by a factor $3/0.4 = 7.5$, and the waiting time scales like $1/s_k$.
:::

:::key What to keep from this chapter
- From small initialization, training starts at a saddle. In linear networks each data mode is learned as a sigmoid: a plateau whose length grows like $\log(1/\text{init})/s_k$, then a jump.
- The plateaus form a staircase of saddles: rank by rank, kink by kink, head by head. The order is set by the data or by the initialization.
- Saddles are not only obstacles. The sequence of saddles visited implements a simplicity bias.
:::

:::problem The next step
This ends RQ1, the left side of the multiplier line. Everything so far assumed a step size small enough that GD follows the gradient flow. The right side of the line, curvature above $2/\eta$, is where practical training actually lives. Section IV begins with what happens to SGD when the step size is too large for the curvature.
:::
