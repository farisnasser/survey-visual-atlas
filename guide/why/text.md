## 1. The puzzle of the edge

This chapter follows the "Mechanisms" part of Section IV-D of the survey. At the edge of stability, the sharpness sits at or slightly above $2/\eta$. The linear analysis of chapter 5 says that along the top eigenvector the multiplier is then at or below $-1$: the oscillation should grow forever, and training should diverge. It does not. Several works explain why. The common answer is that the higher-order terms the linear map discards, the ones chapter 5 said "decide what happens next", act as a brake.

## 2. Phases, and an indicator

Li, Wang & Li (2022) analyzed a two-layer linear network and divided the trajectory into **four phases**, according to whether the sharpness is below or above $2/\eta$ and whether it is rising or falling:

- **I.** Sharpness below $2/\eta$ and rising (progressive sharpening).
- **II.** Above $2/\eta$: the oscillation along the top eigenvector grows.
- **III.** Still above $2/\eta$, but falling; the loss may rise.
- **IV.** Back below $2/\eta$: the oscillation decays.

They identified the norm of the output-layer weights, $\lVert A\rVert^2$, as an indicator that tracks the sharpness dynamics most of the time.

:::viz trace
:::

:::read How to read this visual
**What is drawn.** An offline replication: a two-layer linear network $f(x) = A^\top Wx/\sqrt m$ of width $64$, fitting 32 labels $\pm1$ with full-batch GD. The step size is chosen so that its threshold $2/\eta$ lies between the initial sharpness and the sharpness gradient flow would reach. The main plot shows the sharpness (measured every step by power iteration) and the output-layer norm $\lVert A\rVert^2$ against the GD step, with $2/\eta$ marked. Shaded bands are the steps where the sharpness is above $2/\eta$. The chart below shows the loss on a log scale.

**What to try.** Switch the window between the first 60 steps, the first 300, and the whole run.

**What to notice.** The sharpness climbs past $2/\eta$ (phase I into II), overshoots, falls back (III into IV), and keeps oscillating around $2/\eta$ while the loss drops non-monotonically. Eventually it converges just below the edge. $\lVert A\rVert^2$ rises and falls with the sharpness: watching one tells you about the other. The readout reports the initial, peak and final sharpness against $2/\eta$.
:::

## 3. Self-stabilization

Damian, Nichani & Lee (2023) gave a general mechanism, **[[self-stabilization]]**, based on a third-order Taylor expansion. Let $u$ be the top Hessian eigenvector and $S(\theta)$ the sharpness. When $S > 2/\eta$, the iterate oscillates along $u$ with growing amplitude $x$. Expand the gradient at the displaced point $\theta + xu$ to second order in $x$:

$$\nabla L(\theta + xu) \approx \nabla L(\theta) + x\,S(\theta)\,u + \frac{x^2}{2}\nabla S(\theta).$$

This is survey Eq. (12). The three terms mean three things.

- $\nabla L(\theta)$: the ordinary gradient, which drives progress and, in the progressive-sharpening phase, pushes the sharpness up.
- $x\,S(\theta)\,u$: the oscillation itself. It flips sign with $x$, so it averages out.
- $\tfrac{x^2}{2}\nabla S(\theta)$: the new term. It does *not* flip sign, because $x^2 \ge 0$. A gradient step therefore picks up a drift $-\eta\tfrac{x^2}{2}\nabla S$, which moves the iterate **in the direction that decreases the sharpness**, and does so harder the bigger the oscillation.

So the instability brakes itself. The oscillation grows, the drift pushes $S$ below $2/\eta$, and the oscillation then decays. The net effect, which Damian et al. prove under their assumptions for a long but finite horizon, is that GD at the edge of stability approximately follows **projected gradient descent onto the stable set** $\{\theta : S(\theta) \le 2/\eta\}$. This accounts for the loss, the sharpness and the deviation from gradient flow.

The amplification here is the same one that drives escape from a saddle in chapter 7, with alternating sign. The third-order term turns it into a restoring force.

:::viz stages
:::

:::read How to read this visual
**What is drawn.** Damian et al.'s own toy loss from their Appendix B: $L(x, y) = (2/\eta + y)\,x^2/2 - \alpha y$, with $\eta = 1$ and $\alpha = 0.002$. Across the trough ($x$) the curvature is $S = 2/\eta + y$. Along it ($y$) the loss slopes gently downhill towards larger $y$, so plain descent keeps sharpening. The amber line is where $S = 2/\eta$; beyond it the trough is too sharp for the step. The white lines are the period-2 orbit $x = \pm\delta$. The charts below follow the sharpness excess $S - 2/\eta$ and the oscillation size $|x|$ over the steps.

**What moves.** The coral ball runs exact gradient descent, which for this loss is exactly Damian et al.'s Eq. (3).

**What to try.** Change the initial oscillation size $|x|/\delta$ and restart.

**What to notice.** Four stages repeat. The ball slides downhill and the sharpness rises past $2/\eta$ (1: sharpening). The zig-zag across the trough grows (2: blow-up). The $x^2$ term pulls $y$ back and the sharpness drops (3: self-stabilization). The zig-zag dies out below the edge (4). Then the cycle starts again. The readout names the stage and gives the running mean of $S - 2/\eta$, which stays close to zero. The run is 450 steps, about four cycles: the discrete map slowly spirals outward, which is consistent with the paper's theorem holding over a finite horizon.
:::

:::viz orbit
:::

:::read How to read this visual
**What is drawn.** The same dynamics in a phase plane. The horizontal axis is the oscillation size $|x|$; the vertical axis is the sharpness excess $S - 2/\eta$. The gray closed curves are orbits of the continuous version, Damian et al.'s Eq. (4), a predator–prey (Lotka–Volterra) system, around the fixed point $(\delta, 0)$ with $\delta = \sqrt{2\alpha/\beta}$. The dots are the actual GD iterates, one per step.

**What to try.** Click anywhere in the plane to start a new run there.

**What to notice.** Sharpness and oscillation chase each other like predator and prey. High sharpness feeds the oscillation; a large oscillation eats the sharpness. GD's dots follow the gray loops for a few cycles, slowly spiraling outward. A smaller starting oscillation means a bigger loop (their Corollary 1). The readout tracks the conserved quantity $g$ of their Lemma 3 and the mean sharpness excess.
:::

:::survey Survey Fig. 9. Self-stabilization (schematic).
public/survey/self_stabilization.webp
:::

:::read How to read this figure
**Panel (a), valley dynamics.** A curved valley seen from above; its green floor is the valley's lowest line. The dashed curves mark where the sharpness equals $2/\eta$: outside them, the valley is too sharp. The black zig-zag is GD bouncing across the valley along the top eigenvector $u$ with growing amplitude $x$ (red arrow). The purple arrow is the drift $-\tfrac{\eta x^2}{2}\nabla S$ produced by the third-order term, pointing towards the flatter region, after which the bouncing decays.

**Panel (b), sharpness and oscillation.** A schematic of time on the horizontal axis. The blue curve is the sharpness $S(\theta_t)$. The dashed red line is $2/\eta$. The brown curve is the normalized oscillation size $|x|$. The sharpness rises (progressive sharpening), overshoots $2/\eta$ (blow-up: the oscillation grows), is pulled back (self-stabilization), and settles at $2/\eta$ (edge of stability) while the oscillation decays. This is a drawing of the mechanism, not a simulation.
:::

## 4. Provable entry into the edge

Arora, Li & Panigrahi (2022) gave settings in which GD **provably** operates at the edge of stability. They studied two algorithms on a smooth loss $L$ whose minimum value is zero:

- **normalized GD**, $\theta \leftarrow \theta - \eta\nabla L/\lVert\nabla L\rVert$, which can be viewed as GD with the varying step size $\eta/\lVert\nabla L(\theta_t)\rVert$, a step that grows without bound as the gradient vanishes;
- **GD on $\sqrt{L}$** with a constant step, a loss that is not smooth at its minimizers.

Under regularity conditions, in particular a manifold of minimizers on which the Hessian has constant rank, both provably enter the edge of stability. Their iterates then oscillate around the manifold of minimizers while following a deterministic flow *along* it that decreases the sharpness $\lambda_1(\nabla^2 L)$. This is an implicit regularization that relies neither on infinitesimal steps nor on gradient noise. Instability itself selects flatter minima.

:::viz normgd
:::

:::read How to read this visual
**What is drawn.** The example of Arora et al.'s Fig. 1: $L(x, y) = (1 + x^2)\,y^2$, drawn with height $\sqrt L$ and the $y$-axis stretched so the valley is visible. Every point of the green line $y = 0$ is a minimum with zero loss. But the sharpness there, $2(1 + x^2)$, differs along the line: $20$ at $x = 3$, and its smallest value $2$ at $x = 0$, labeled "flattest minimum". The first chart plots the sharpness of the nearest minimizer, $2(1 + x_t^2)$, against the step. The second plots the distance across the valley, $|y_t|$, on a log scale.

**What moves.** The coral ball starts at $(3, 0.4)$ and runs the method you choose for up to 3000 steps.

**What to try.** Compare the three methods: *GD on √L* ($\eta = 0.1$), *normalized GD* ($\eta = 0.1$), and *plain GD, small step* ($\eta = 0.02$).

**What to notice.** Plain GD with a small step drops straight into the nearest minimum. Its $|y|$ vanishes, and it stops near $x = 2.97$ with sharpness $19.6$. It never moves along the valley. GD on $\sqrt L$ never settles across the valley: $|y|$ stays of the order of the step, and the run ends in a period-2 bounce between $y = 0.027$ and $y = -0.073$. While it bounces, it slides along the valley: $x = 0.24$ after 500 steps and $0.02$ after 1000, so the sharpness falls from $20$ to $2$. Normalized GD does the same, more slowly ($x = 2.13$ after 500 steps, $1.01$ after 1000, below $0.001$ by step 3000). The oscillation is what moves the iterate towards the flattest minimum.
:::

## 5. Beyond the edge, and a route to chaos

Chen & Bruna (2023) went **beyond the edge of stability**. They analyzed the two-step map $\theta \mapsto G(G(\theta))$, two GD steps at once, and gave local conditions involving third-order derivatives under which GD converges to a **stable period-two orbit** even when single-step stability fails. They established these conditions in representative learning problems, including teacher–student settings. A quadratic can never satisfy them, because its third derivative is zero: on a quadratic, past $2/\eta$ there is only divergence.

:::viz bif
:::

:::read How to read this visual
**What is drawn.** A bifurcation diagram. For each value of $K = \eta\mu$ on the horizontal axis (step size times the curvature at the minimum), GD runs 3000 steps on the quartic $\tfrac14(x^2 - 1)^2$, and the next 48 iterates are plotted as dots above that $K$. One dot per column means GD converged to a single point; two dots means a period-2 orbit; many dots means chaos.

**What to notice.** Below $K = 1$ (the edge, $\eta = 2/\lambda$ at this minimum) there is one dot: GD converges to the minimum. Just above, the column splits into two: a stable period-2 orbit, exactly the behavior Chen & Bruna's Theorem 2 guarantees, up to $K \approx 1.121$. In this simulation the orbit survives until $K \approx 1.237$, then splits into 4, then 8, then chaos. The readout gives these values. The theorem's range is sufficient, not sharp.
:::

Kalra, He & Barkeshli (2025) showed that a two-layer linear "UV" model trained on a single example already exhibits the full sharpness phenomenology: early sharpness reduction, progressive sharpening, the edge of stability, and a **[[period-doubling]] route to chaos** on the edge-of-stability manifold as the learning rate increases. They analyze it through the dynamics of two numbers, the residual and the sharpness, and give the conditions needed for the edge of stability to occur at all.

:::viz chaos
:::

:::read How to read this visual
**What is drawn.** For the UV model with effective width $1$ (the paper's Eqs. 1–2), each column is one step size $\eta$ on the horizontal axis. Above it are the last 120 sharpness values of a 3000-step run. This is the paper's Fig. 3b. The critical step is $\eta_c = 0.5$.

**What to notice.** Below $\eta_c$, each column is a single point: training converges. Above $\eta_c$, the sharpness no longer settles. It oscillates around $2/\eta$, splits $1 \to 2 \to 4 \to \dots$, and turns chaotic, until training finally diverges. The readout reports where divergence first happens. The edge of stability is not one behavior but a family of increasingly complex ones, and the step size moves the system through them.
:::

:::survey Survey Fig. 8. Original simulation of the route to chaos in a two-layer linear (UV) model, f = uᵀv/√n with n = 8, trained by GD on a single example with target 4 from small initialization, for 1400 step sizes.
public/survey/route_to_chaos.webp
:::

:::read How to read this figure
**Panel (a).** For each step size, plotted on the horizontal axis as $\eta\lambda_\star$ ($\lambda_\star$ is the sharpness of the flattest global minimum), the 128 outputs $f(\theta_t)$ after 4000 burn-in steps. Left of $\eta\lambda_\star = 2$ the model converges to the target $4$. Just right of it, the outputs alternate between two values: period 2. Further right, the band widens into chaos.

**Panel (b).** The time-averaged $\eta\lambda_{\max}(\nabla^2 L)$ for the same runs. While a stable minimum exists ($\eta\lambda_\star < 2$), it equals $\eta\lambda_\star$: a straight line of slope one. Beyond that point no stable minimum exists, and the average stays pinned close to the threshold $2$ before rising. This is the edge of stability in its cleanest form: the sharpness is held at the threshold for a whole range of step sizes. Compare with the phase diagrams of Kalra et al.
:::

:::key What to keep from this chapter
- At the edge, the third-order term of the Taylor expansion produces a drift $-\eta\tfrac{x^2}{2}\nabla S$ that lowers the sharpness. The instability brakes itself (self-stabilization), and GD tracks projected descent onto $\{S \le 2/\eta\}$.
- Normalized GD and GD on $\sqrt L$ provably enter the edge, and their oscillation drives them along the valley to flatter minima.
- Past the edge, third-order terms allow stable period-2 orbits; further on, period doubling and chaos.
:::

:::problem The next step
These mechanisms explain the edge one unstable direction at a time, or in small models. Is there a description of the whole trajectory of a real network at the edge of stability, with any number of unstable directions?
:::
