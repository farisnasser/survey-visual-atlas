## 1. Average the oscillation away

This chapter follows Section IV-E of the survey. The mechanisms of chapter 11 analyze the edge of stability near a single unstable direction, or in small models. Real networks at the edge can have several Hessian eigenvalues at $2/\eta$ at once, oscillating in several directions with chaotic-looking iterates. Cohen, Damian, Talwalkar, Kolter & Lee (2025) proposed a different strategy: **do not track the oscillating iterates; track their time average.**

Write the GD iterate as an average position plus an oscillation along the top eigenvector:

$$\theta_t = \bar\theta_t + x_t\,u_t, \qquad \text{with } x_t \text{ averaging to zero over a few steps.}$$

The average position moves with the *average* gradient. Average the Taylor expansion of chapter 11 (survey Eq. 12) over the oscillation. The term $x_t S u_t$ averages to zero because $x_t$ does; the term $\tfrac{x_t^2}{2}\nabla S$ does not, because $x_t^2 \ge 0$. Writing $\sigma^2(t)$ for the average of $x_t^2$, the variance of the oscillation, this gives a differential equation for the average path:

$$\frac{d\theta}{dt} = -\eta\Big[\nabla L(\theta) + \tfrac12\,\sigma^2(t)\,\nabla S(\theta)\Big].$$

This is survey Eq. (13), the **[[central-flow|central flow]]** for one unstable direction. It is a gradient flow with an implicit **sharpness penalty**, whose strength is the variance of the oscillation.

## 2. The variance is not a free parameter

The flow has one unknown, $\sigma^2(t)$, and it is fixed by what we observe: at the edge, the sharpness stays at $2/\eta$. Along the flow, the sharpness changes as $dS/dt = \langle\nabla S, d\theta/dt\rangle$. Setting this to zero gives a unique value:

$$\sigma^2(t) = \frac{2\,\langle\nabla S, -\nabla L\rangle}{\lVert\nabla S\rVert^2}.$$

In words, the oscillation is exactly as large as it needs to be for its sharpness-reducing push to cancel the sharpening push of the gradient. Substituting this back, the central flow becomes a **projected gradient flow**:

$$\frac{d\theta}{dt} = -\eta\,\Pi^\perp_{\nabla S}\,\nabla L(\theta), \qquad \Pi^\perp_v = I - \frac{vv^\top}{\lVert v\rVert^2}.$$

It takes the ordinary gradient and removes its component along $\nabla S$, the direction that would increase the sharpness. **At the edge of stability, GD moves like gradient flow constrained to the edge.** This nearly recovers the projected dynamics of self-stabilization (chapter 11) by a simpler, non-rigorous argument.

The general central flow allows any number of Hessian eigenvalues to be at the edge. It models the covariance of the oscillations as the solution of a semidefinite complementarity problem, a kind of convex program, and it reduces to the ordinary gradient flow when no eigenvalue is at the edge.

## 3. A toy you can watch

The visual below builds the smallest example where all of this can be seen, and checks it numerically. It extends Damian et al.'s toy with one more coordinate, $z$, which plays the role of "everything else the network is learning". The toy has three coordinates, $(x, y, z)$:

- $x$ is the direction GD oscillates along. Its curvature is $S = 2/\eta + y$, so the line $y = 0$ is the edge $S = 2/\eta$, and $y$ measures how far above the edge the current point is.
- $(y, z)$ are the slow coordinates. The rest of the loss, $V(y, z)$, pulls them towards a point with $y > 0$: the best point of the loss lies in the region that is **too sharp** for the step size.

So gradient flow, which ignores step sizes, walks straight into the too-sharp region. GD cannot follow it there. The central flow predicts what GD does instead.

:::viz centralflow
:::

:::read How to read this visual
**What is drawn.** The main plot is the plane of the two slow coordinates seen from above: $z$ (progress) horizontally, $y$ (sharpness excess $S - 2/\eta$) vertically. The thin curves are contours of $V$. The amber line $y = 0$ is the edge $S = 2/\eta$, and the red-tinted region above it is too sharp for the step size. The violet dot is the minimum of $V$, inside the too-sharp region. The white dot is the best point on the edge, the minimum of $V$ along $y = 0$.

**What moves.** Three trajectories, drawn as time advances. Violet: gradient flow. White dashed: the central flow (drawn on top, so before the edge, where all three coincide, you see it over the others). Coral dots: actual gradient descent, every third step, projected onto $(z, y)$; its oscillation along $x$ is not drawn. Three charts below follow the same three over time: the sharpness excess, the loss on a log scale, and GD's oscillation $x_t^2$ with the central flow's predicted variance $\sigma^2$ as a dashed line.

**What to try.** *Replay*, at slow speed first.

**What to notice.**
- Until step 123 all three move together. Below the edge, GD *is* gradient flow, and they reach $y = 0$ at the same step.
- Gradient flow then crosses into the too-sharp region and goes to the violet point, with sharpness $2/\eta + 0.5$.
- The central flow stops at the edge and slides along it to the white point. Its loss decreases monotonically.
- GD first **overshoots** into the too-sharp region, by up to $0.242$ at step 293. While it waited below the edge its oscillation decayed to about $3\times10^{-11}$, and it needs time to grow back. Then the oscillation grows, self-stabilization pulls the sharpness back, and from step 600 on GD sits on the central flow to within $0.002$.
- At the end, GD's oscillation settles at $x^2 = 0.01429$, the central flow's $\sigma^2 = 0.01429$ exactly. GD's loss is higher than the central flow's ($0.0161$ against $0.0018$) and non-monotone, because it includes the energy of the oscillation itself. The central flow predicts the average *position*, not the bouncing.
:::

The overshoot in this toy is not a defect of the toy. It is the "blow-up" stage of chapter 11, and Cohen et al. note the same effect in real networks: when the sharpness first reaches $2/\eta$, the central flow starts regularizing immediately, while discrete GD has not yet developed its oscillation.

## 4. What central flows explain

Two properties make the central flow a useful tool (Cohen et al., Propositions 1 and 2):

- **Its loss decreases monotonically.** So the time-averaged loss is a valid potential function, even though the loss of GD itself goes up and down. Training at the edge *is* making steady progress, measured the right way.
- **It decreases the loss more slowly than gradient flow.** The projection removes part of the gradient. The oscillations cost progress through their implicit curvature penalty.

Cohen et al. derived analogous flows for scalar and per-coordinate RMSProp (Adam without momentum). These flows show how such optimizers implicitly adapt their step sizes to the local curvature, and how they steer towards regions where they can take larger steps. Empirically, the central flows predicted the long-term trajectories of GD on a ResNet and a Transformer, and of RMSProp on a Mamba model, with a high degree of numerical accuracy.

:::warn Limitations, stated by the authors
The derivation is informal: a heuristic time-averaging argument, validated by experiments rather than by theorems. It applies to deterministic (full-batch) training without momentum. It becomes less accurate for very large learning rates, on some cross-entropy problems, and around large loss spikes. The authors note that, because of the randomness, it is not clear whether such a deterministic description even exists for stochastic optimizers. Chapter 15 returns to this as an open problem.
:::

:::key What to keep from this chapter
- Do not track the oscillating iterate; track its average. The average follows a gradient flow with an implicit sharpness penalty, $\tfrac12\sigma^2\nabla S$.
- Holding the sharpness at $2/\eta$ fixes $\sigma^2$ uniquely, and the flow becomes gradient flow projected onto the edge.
- The central-flow loss falls monotonically, more slowly than gradient flow. The prediction is accurate on real networks, but the derivation is non-rigorous and covers deterministic training only.
:::

:::problem The next step
Real training uses Adam, warmup, minibatches and sometimes sharpness-aware updates. Do these have their own edges of stability? And does living at the edge change *where* training ends?
:::
