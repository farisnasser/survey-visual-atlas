## 1. The other side of the multiplier line

This chapter opens Section IV of the survey and RQ2: what happens when the step size exceeds the classical stability limit. It follows Sections IV-A and IV-B.

Everything in chapters 7 to 9 shared one implicit assumption: the step size is small enough that gradient descent approximately follows the gradient flow and decreases the loss at every step. Under Assumption 1 that means $\eta < 1/\ell$, or at least $\eta < 2/\ell$. Recall from chapter 5 that a minimum whose sharpness $\lambda_{\max}$ exceeds $2/\eta$ is linearly unstable under GD: the right side of the multiplier line, $1 - \eta\lambda < -1$. Classical analyses therefore choose $\eta$ small relative to the curvature. **Practice does the opposite.** This chapter covers the early work that took that seriously.

## 2. Which minima can SGD stay in? Dynamical stability

Wu, Ma & E (2018) asked which global minima can be reached by SGD with a given learning rate and batch size, and studied the question from the perspective of dynamical stability. Their idea: a minimum that SGD cannot *stay* in is a minimum SGD will not *end* in. They introduced a measure of **non-uniformity**, how much the curvature differs from one training example to another. Together with the sharpness, it characterizes whether a global minimum is stable for a given SGD algorithm. Their analysis shows that **the learning rate and the batch size play different roles** in this selection.

Here is the condition in its simplest form, as stated by Jastrzębski et al. (2020), who cite Wu et al. for its proof. Look at the loss along the top Hessian eigenvector $e_1$, and suppose it is quadratic along that line. Each training example $i$ has its own curvature $h_i$ along $e_1$; their average is the sharpness $\lambda_1$, and their spread (variance over the examples) is $s^2$. A minibatch of $S$ examples, drawn without replacement from $N$, has curvature $\bar h_B$, the average over the batch. One SGD step multiplies the distance $\psi$ along $e_1$ by $1 - \eta\bar h_B$, a random multiplier. SGD is stable along $e_1$, meaning $\psi^2$ shrinks on average, exactly when

$$(1 - \eta\lambda_1)^2 + s^2\,\frac{\eta^2(N - S)}{S(N - 1)} \;\le\; 1.$$

This is survey Eq. (17). Read it in two parts.

- The first term is the full-batch multiplier of chapter 5, squared. For full-batch GD, $S = N$ and the second term vanishes: the condition is $|1 - \eta\lambda_1| \le 1$, the familiar $\eta\lambda_1 \le 2$.
- The second term is the variance of the random multiplier. It is positive whenever the examples disagree ($s > 0$) and the batch is smaller than the data set. It adds to the left side, so the **condition is stricter for smaller batches**. SGD can stay only where the curvature is smaller, or more uniform across examples.

A worked example with $N = 100$, $\eta\lambda_1 = 1.5$ and $\eta s = 1$: full batch gives $(1 - 1.5)^2 = 0.25 \le 1$, stable. A batch of $S = 16$ gives $0.25 + 1 \cdot 84/1584 = 0.303$, still stable. A batch of $S = 1$ gives $0.25 + 1 \cdot 99/99 = 1.25 > 1$, unstable. The same minimum, with the same step size, can hold full-batch GD and reject SGD with batch size one.

:::viz sgdstab
:::

:::read How to read this visual
**What is drawn.** A map of possible minima, each described by two numbers along its sharpest direction. The horizontal axis is $\eta\lambda$, the step size times the average curvature: the quantity of chapter 5. The vertical axis is $\eta s$, the step size times the spread of the per-example curvatures. Each colored curve is the boundary of survey Eq. (17) for one batch size. Below its curve, SGD with that batch size is stable. The full-batch boundary is the pair of vertical lines $\eta\lambda = 0$ and $\eta\lambda = 2$: for full batches the spread does not matter. The lightly shaded region is the stable region of the batch size you selected. The $S = 64$ boundary peaks at $eta s pprox 13.3$, above the drawn area, so its curve only shows near the two ends.

**What moves.** The dot is one minimum; drag it. The chart below runs real SGD at that point: 100 examples whose curvatures have exactly that mean and spread, batches drawn without replacement, $\eta = 1$. It plots the average of $\psi_t^2$ over 250 runs (dots) on a log scale, against the exact mean-square law, the factor of Eq. (17) to the power $t$ (white line).

**What to try.** Put the dot at $(1.5, 1.0)$ and switch the batch size from 100 down to 1. Then drag the dot upwards, increasing the spread, at a fixed $\eta\lambda$.

**What to notice.** The stable regions are nested: every smaller batch has a smaller region, inside the full-batch strip. At $(1.5, 1.0)$ the dot is stable for $S = 100$ and $S = 16$ and unstable for $S = 1$, and the chart's dots rise instead of fall. The dots follow the white line: the formula is exact for this model. A minimum with uneven curvature across examples ("non-uniform", in Wu et al.'s word) is rejected by small-batch SGD even when its average curvature is moderate. This is one precise sense in which **SGD prefers flatter and more uniform minima**.
:::

## 3. The break-even point

Jastrzębski, Szymczak, Fort, Arpit, Tabor, Cho & Geras (2020) applied this condition to the early phase of training. In a simplified model, they assume that the sharpness grows during training until training becomes unstable along $e_1$. They call the first point where the left side of Eq. (17) reaches one the **[[break-even|break-even point]]**. For full batches it is reached exactly when $\lambda_1 = 2/\eta$.

They conjectured, and supported with experiments on several classification tasks, the following.

:::plain Jastrzębski et al. 2020, Conjectures 1 and 2 (in words)
Along the SGD trajectory, a **larger learning rate or a smaller batch size** used before the break-even point leads to **smaller maximal values** of the sharpness and of the spectral norm of the gradient covariance (the size of the gradient noise), and to **better conditioning** of both.
:::

They also found that a low learning rate leads to badly conditioned regions even in networks with batch normalization. The early choice of step size shapes the curvature of the regions that training visits later. This is the first hint of the main finding of the next chapter: training does not avoid sharp regions by staying below $2/\eta$. It is pushed to the edge, and the step size sets where the edge is.

## 4. Catapults

Lewkowycz, Bahri, Dyer, Sohl-Dickstein & Gur-Ari (2020) studied wide networks trained with the squared loss and identified a **[[catapult]] phase**. Let $\lambda_0$ be the curvature at initialization.

- For $\eta < 2/\lambda_0$, the **lazy phase**: training converges smoothly, and for very wide networks the curvature barely changes.
- For $2/\lambda_0 < \eta < \eta_{\max}$, the **catapult phase**: the loss first *grows* sharply, the curvature then *decreases* until the dynamics become stable, and training converges in a flatter region. In their solvable model $\eta_{\max} = 4/\lambda_0$; for ReLU networks they report it larger, near $12/\lambda_0$.
- For $\eta > \eta_{\max}$: divergence.

In their solvable model the update equations for the network output and the curvature are exact, with no higher-order terms dropped. The loss grows for a number of steps that scales with the logarithm of the width. These dynamics lie outside the lazy, kernel description of wide networks, and in their experiments the best test accuracy was often reached in the catapult phase. **A catapult is the simplest instance of the right side of the multiplier line:** the iterate is expelled from a region that is too sharp, and the expulsion itself reduces the sharpness.

:::viz spike
:::

:::read How to read this visual
**What is drawn.** The loss $\tfrac12(uv)^2$ of the smallest two-layer linear model: one hidden unit, weights $u$ and $v$, one example with input $1$ and label $0$ (the paper's Sec. 2.1 model). Both axes $u = 0$ and $v = 0$ are minima (green). The curvature at a point of the axes is $\lambda = u^2 + v^2$: far from the origin the axes are sharp, near it they are flat. The charts below plot the loss and the curvature $\lambda$ over the steps.

**What moves.** The coral ball starts near a sharp point, with curvature $\lambda_0$, and runs gradient descent. The slider sets the step through the product $\eta\lambda_0$.

**What to try.** Try $\eta\lambda_0 = 1.5$, then $3$, then $4.2$.

**What to notice.** At $1.5$ the ball slides into the nearest minimum. At $3$, between $2$ and $4$, the loss first grows (the readout reports the peak against the start), the ball is thrown towards the origin, and it lands on a flatter part of the axes, where the readout shows $\lambda$ below $2/\eta$. At $4.2$ it diverges. With only one unit the lazy phase is not perfectly lazy: even below $2/\lambda_0$ the curvature moves a little.
:::

:::viz phases
:::

:::read How to read this visual
**What is drawn.** For the width-$n$ version of the model (the paper's Eqs. 6–7), the final curvature after training, $\lambda_\infty/\lambda_0$, against the step size $\eta\lambda_0$, one point per step size. The dashed curve is $2/(\eta\lambda_0)$, the largest curvature that is stable at that step size.

**What to try.** Change the width $n$ from $10$ to $10^8$.

**What to notice.** Three phases. Below $\eta\lambda_0 = 2$, the final curvature stays at $\lambda_0$: the lazy phase, which becomes exact as the width grows. Between $2$ and $4$, the final curvature drops below the dashed line: training ends in a flatter minimum, flat enough to be stable at its step size. Above $4$, training diverges. The readout reports where divergence first happens. The step size does not just set the speed: it selects how flat the final minimum is.
:::

:::key What to keep from this chapter
- With minibatches, the stability condition becomes $(1 - \eta\lambda)^2 + (\eta s)^2(N - S)/(S(N - 1)) \le 1$: smaller batches tolerate less curvature and less non-uniformity.
- Larger learning rates and smaller batches early in training lead to flatter, better-conditioned regions (the break-even point).
- Between $2/\lambda_0$ and $\eta_{\max}$, training catapults: the loss spikes, the curvature drops, and it converges somewhere flatter.
:::

:::problem The next step
Catapults happen when training *starts* too sharp. Cohen et al. (2021) found something more surprising: during ordinary training with a fixed step size, the sharpness *rises* until it reaches $2/\eta$, and then stays there.
:::
