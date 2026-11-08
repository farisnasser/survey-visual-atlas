## 1. Adaptive methods have their own edge

This chapter follows Sections IV-F and IV-G of the survey. Practical training rarely uses plain full-batch GD. Does the edge of stability survive the move to Adam, warmup schedules, sharpness-aware updates and minibatches?

Recall from chapter 4 that an adaptive method is GD with a preconditioner $P$: $\theta \leftarrow \theta - \eta P^{-1}m$. Along the preconditioned directions, the multiplier of chapter 5 involves the eigenvalues of the **preconditioned Hessian** $P^{-1}\nabla^2 L$, not of $\nabla^2 L$. Cohen, Ghorbani, Krishnan, Agarwal et al. (2022) showed empirically that adaptive methods have their own edge of stability.

:::plain The adaptive edge of stability (Cohen et al. 2022)
During full-batch training, the largest eigenvalue of the preconditioned Hessian $P^{-1}\nabla^2 L$ typically **equilibrates at the stability threshold** of the corresponding update. For Adam with step size $\eta$ and momentum $\beta_1$, that threshold is $(2 + 2\beta_1)/((1 - \beta_1)\eta)$, which is $38/\eta$ for the default $\beta_1 = 0.9$. Similar effects occur during minibatch training, especially as the batch size grows.
:::

The behavior differs from GD in one important respect. Non-adaptive methods at the edge are blocked from entering high-curvature regions. Adaptive methods at this **[[adaptive-eos|adaptive edge of stability]]** keep advancing into high-curvature regions, while adapting the preconditioner to compensate. The raw sharpness can keep rising; what stays at the threshold is the sharpness *as seen through the preconditioner*.

:::viz adam
:::

:::read How to read this visual
**What is drawn.** The largest eigenvalue of Adam's preconditioned Hessian, $\lambda_{\max}(P^{-1}H)$, on a log scale, over the first 300 steps of training a 200-dimensional quadratic. Its curvatures range from $0.1$ to $10$, and the starting coordinates are spread over four decades. Here $P_t = (1 - \beta_1^t)\,\mathrm{diag}(\sqrt{\hat v_t}) + \varepsilon$, with $\eta = 10^{-3}$, $\beta_1 = 0.9$ and $\beta_2 = 0.999$. The red dashed line is Adam's threshold $(2 + 2\beta_1)/((1 - \beta_1)\eta)$. The blue dashed line is the raw sharpness $\lambda_{\max}(H) = 9.87$, for comparison.

**The two curves.** Coral: standard Adam. Violet: GI-Adam, which initializes the second-moment estimate with the first squared gradient, $v_0 = g_0^2$.

**What to notice.** At step 1, Adam's $\hat v$ is just $g_0^2$, so coordinates with tiny gradients get a tiny $P$, and $P^{-1}H$ is enormous, far above the threshold. Adam starts deep in the unstable region. GI-Adam starts lower by a factor of $\sqrt{1 - \beta_2} \approx 1/32$; the readout confirms the ratio $0.0317$ against $0.0316$. Kalra & Barkeshli show that this acts as a built-in warmup: GI-Adam is Adam with the step size scaled by $\sqrt{1 - \beta_2^t}$ in the early steps.
:::

## 2. Warmup through the same lens

Kalra & Barkeshli (2024) explained learning-rate [[warmup]] through the stability lens. Its main benefit is to let training tolerate a larger target learning rate, by steering it towards flatter, better-conditioned regions. As the learning rate rises during warmup, each time $\eta_t\lambda$ crosses 2 a small loss catapult (chapter 9) reduces the sharpness and restores stability. They distinguish two regimes:

- a **sharpness-reduction** regime, for instance large standard-parameterization initializations, in which warmup helps considerably;
- a **progressive-sharpening** regime, for instance $\mu$P or pre-LayerNorm Transformers, in which it helps little.

For Adam the relevant quantity is the preconditioned sharpness, which typically starts high. That is why GI-Adam, in the visual above, behaves like Adam with a warmup. Its initialization lowers the starting preconditioned sharpness by a factor of $\sqrt{1 - \beta_2}$, which the authors describe as around two orders of magnitude.

:::viz warm
:::

:::read How to read this visual
**What is drawn.** The surface $f = uv$ fitted to one input $x = 1$, with labels $+0.02$ and $-0.02$, used one per step: SGD on the smallest linear network. Its minima are the axes (green), with sharpness $\lambda = u^2 + v^2$, so points far out on the axes are sharp. The charts follow the step size $\eta_t$ against $2/\lambda$, and the loss.

**What moves.** The coral ball runs SGD towards a target step of $6/\lambda_0$, three times what the starting point tolerates ($2/\lambda_0$). The control sets the length of a linear warmup from $\eta = 0$ to the target: none, 20, 50, 200 or 400 steps.

**What to try.** Compare *none* with *200*.

**What to notice.** Without warmup, the very first steps throw the ball off the surface: divergence. With a warmup, $\eta_t$ grows slowly. Every time $\eta_t\lambda$ reaches 2, a small catapult lowers the sharpness, so $\lambda$ tracks $2/\eta_t$ downwards and training continues at the full target step. The small label noise matters: it keeps the residual from vanishing. With exactly zero loss, GD would just sit still and could not catapult.
:::

## 3. Sharpness-aware minimization

The same analysis applies to **[[sam|sharpness-aware minimization]]** (SAM), which takes its gradient step from a nearby, deliberately worse point. For its unnormalized variant, $\theta \leftarrow \theta - \eta\nabla L\big(\theta + \rho\nabla L(\theta)\big)$, Agarwala & Dauphin (2023) showed that on a quadratic each eigen-component is multiplied by $1 - \eta(\lambda + \rho\lambda^2)$. The stability condition becomes

$$\eta\,\lambda\,(1 + \rho\lambda) < 2$$

for the curvature eigenvalues $\lambda$, where $\rho$ is the perturbation radius. Training then settles at a smaller largest eigenvalue than GD with the same step size. This eigenvalue regularization arises from the dynamics, not from an explicit penalty in the loss.

:::viz sam
:::

:::read How to read this visual
**What is drawn.** The largest stable curvature for unnormalized SAM, $\lambda_{\rm SAM} = (-1 + \sqrt{1 + 8\rho/\eta})/(2\rho)$, the solution of $\eta\lambda(1 + \rho\lambda) = 2$, against the learning rate, with GD's $2/\eta$ for comparison. The dot marks your choice of $\eta$ and $\rho$.

**What to try.** Move the radius $\rho$ from $0$ upwards, at a large and a small $\eta$.

**What to notice.** At $\rho = 0$ SAM is GD, and its edge is $2/\eta$. Any $\rho > 0$ lowers the edge, more so for small $\eta$. The readout compares $\lambda_{\rm SAM}$ with $2/\eta$, and reproduces the worked example $\eta = 0.5$, $\rho = 0.2$, where $\lambda_{\rm SAM} \approx 2.62$ against $2/\eta = 4$. SAM is an edge-of-stability method with a lower edge.
:::

## 4. Minibatch SGD: the edge of stochastic stability

For SGD, the sharpness of the full-batch loss does **not** lock onto $2/\eta$; it typically stays lower. Andreyev & Beneventano (2024) showed that minibatch SGD trains in a different regime, which they call the **edge of stochastic stability**. In it, the quantity that stabilizes at $2/\eta$ is the **[[batch-sharpness|batch sharpness]]**: the expected curvature of each minibatch Hessian along its own minibatch gradient,

$$\mathrm{BS}(\theta) = \mathbb{E}_B\left[\frac{\nabla L_B(\theta)^\top\,\nabla^2 L_B(\theta)\,\nabla L_B(\theta)}{\lVert\nabla L_B(\theta)\rVert^2}\right].$$

This is survey Eq. (14). Read it as: draw a batch $B$; compute its gradient direction; measure the curvature of *that batch's* loss along *that* direction; average over batches. It is a curvature of the minibatch landscape, the one SGD actually steps on.

Because the full-batch sharpness is generally smaller than the batch sharpness, it is suppressed below $2/\eta$. This fits the long-standing observation that smaller batches and larger step sizes favor flatter minima. The evidence is empirical, for vision classification tasks with the squared loss. It is complemented by a theorem on the local quadratic approximation: crossing $2/\eta$ by any fixed margin forces a catapult. Together with the stability condition of chapter 9, this suggests that **for SGD the relevant notion of stability is stability on the minibatch landscape.**

:::plain Why there is no live visual of batch sharpness here
Batch sharpness only becomes interesting in a model whose curvature changes during training, that is, a non-linear network trained by SGD long enough to reach the edge. We have not built a check-scripted replication of that experiment, and the Atlas only shows what it can recompute. The SGD stability visual of chapter 9 shows the same principle in the simplest exact setting: minibatch noise makes the curvature threshold stricter than $2/\eta$.
:::

| Optimizer | Quantity that stabilizes | Threshold | Source |
|---|---|---|---|
| GD | $\lambda_{\max}(\nabla^2 L)$ | $2/\eta$ | Cohen et al. 2021 |
| Heavy ball | $\lambda_{\max}(\nabla^2 L)$ | $(2 + 2\beta)/\eta$ | Cohen et al. 2021 |
| Nesterov | $\lambda_{\max}(\nabla^2 L)$ | $(2 + 2\beta)/(\eta(1 + 2\beta))$ | Cohen et al. 2021 |
| Adam | $\lambda_{\max}(P^{-1}\nabla^2 L)$ | $(2 + 2\beta_1)/((1 - \beta_1)\eta)$, i.e. $38/\eta$ at $\beta_1 = 0.9$ | Cohen et al. 2022; Kalra & Barkeshli 2024 |
| SAM (unnormalized) | $\lambda(1 + \rho\lambda)$ at $\lambda = \lambda_{\max}$ | $2/\eta$ | Agarwala & Dauphin 2023 |
| SGD | batch sharpness, Eq. (14) | $2/\eta$ | Andreyev & Beneventano 2024 |
| SGD, quadratic model | condition (11) | stricter than $2/\eta$ | Wu et al. 2018; Jastrzębski et al. 2020 |

This is the survey's Table IV: the stability thresholds of the main optimizers. In each case the listed quantity rises during training and then stabilizes near its threshold.

## 5. Large steps change the destination

Living at the edge does not only change the path. It changes where training ends.

Mohtashami, Jaggi & Stich (2023) argued that the benefits usually attributed to SGD noise are partly benefits of the large step size itself. They showed that these effects persist in noiseless full-batch GD. They also proved that on certain non-convex function classes, GD with a large step follows a qualitatively different trajectory from small-step GD and can reach a global rather than a local minimum.

:::viz escape
:::

:::read How to read this visual
**What is drawn.** A wide bowl around the global minimum (green, curvature $0.3$) with a narrow, higher spurious well (red, curvature $15.5$). This is a toy built like the paper's function class, not the paper's networks. The chart plots the gradient norm $\lVert\nabla f\rVert$ along the large-step run, on a log scale.

**What moves.** Two runs of deterministic full-batch GD from the same start. Violet: small step, $\eta = 0.05$. Coral: the step you choose.

**What to try.** Move the large step from below $2/15.5 \approx 0.13$ to above it.

**What to notice.** The small-step ball slides into the narrow well and stays. With a step above $0.13$, the narrow well is linearly unstable, its multiplier below $-1$, so the coral ball is thrown out of it. Its gradient norm briefly grows on the chart. It then converges to the global minimum, because the wide bowl is stable for any step below $2/0.3 \approx 6.7$. The readout reports where each run ended. No noise is involved: the step size alone selects the minimum.
:::

Galli, Fox, Bartolomaeus, Schmidt & Rauhut (2026) asked how large a step can be when the loss is not globally $\ell$-smooth. They gave a unifying definition of "large" step sizes that needs only local Lipschitz or Hölder continuity of the gradient, and designed adaptive first-order methods that provably produce such steps. These methods operate at the edge of stability **from the start of training**, with step size times sharpness above 2 throughout. Using them, the sharpness can be driven all the way down to its global minimum.

:::viz linesearch
:::

:::read How to read this visual
**What is drawn.** A curved valley, $f = \tfrac14(x^2 - 1)^2 + 3(y - 0.3x^2)^2$, with minima in green. The first chart plots $\eta_k\cdot\lambda_{\max}(\nabla^2 f(w_k))$, the step times the local sharpness, at each iteration; 2 is the edge. The second plots $f(w_k)$ on a log scale.

**What moves.** Two balls, both using an equality line search that keeps growing the step while it is accepted. White: monotone acceptance, where every step must lower $f$. Coral: nonmonotone acceptance, where a step only has to beat a running average of recent losses (the Zhang–Hager rule).

**What to notice.** The monotone search stays below the edge. The nonmonotone search accepts occasional loss increases, bounces across the valley, and sits at $\eta\lambda_{\max} \approx 2$ or above: the edge of stability, from the first iterations. It still converges, with $\lVert\nabla f\rVert \to 0$, which is what their convergence theorem guarantees. The readout counts the accepted loss increases and gives the median $\eta\lambda_{\max}$.
:::

Contrary to expectation, Galli et al. found that **reaching globally flat regions too early can slow convergence and harm generalization**. By exploiting self-stabilization to let GD enter slightly sharper valleys, they turned unsuccessful training runs into successful ones. The slogan "flatter is better" must be qualified.

:::viz flatland
:::

:::read How to read this visual
**What is drawn.** The top Hessian eigenvalues of a tiny classifier (2 inputs, 4 tanh hidden units, $K$ outputs) trained with the squared loss averaged over its $K$ outputs, on 30 random points with random labels. The red line is $2/K$. For this loss the sharpness can never drop below $2/K$, because the output biases alone contribute a curvature of $2/K$.

**What to try.** Switch between *random weights*, *after GD*, and the *flat point*, and change the number of classes $K$.

**What to notice.** At the flat point the hidden layer is silent and the outputs equal the class frequencies. There the sharpness is exactly $2/K$, the global minimum of the sharpness, the gradient is zero, and there are negative eigenvalues. The readout reports all three. The flattest point of the whole landscape is a saddle, with near-random loss: a "Flatland" that a method driving the sharpness down too eagerly can get stuck in.
:::

:::key What to keep from this chapter
- Adam's edge is on the preconditioned sharpness, $(2 + 2\beta_1)/((1 - \beta_1)\eta) = 38/\eta$ at $\beta_1 = 0.9$, and adaptive methods keep advancing into sharper regions while the preconditioner compensates.
- Warmup lets training reach a large step by catapulting down the sharpness as the step grows.
- SAM has a lower edge, $\eta\lambda(1 + \rho\lambda) = 2$. For minibatch SGD, what locks to $2/\eta$ is the batch sharpness, which keeps the full-batch sharpness below $2/\eta$.
- Large steps change the destination, towards flatter minima, but flatness reached too early can hurt.
:::

:::problem The next step
That completes both halves of the survey. The next chapter puts them side by side: one instability of the gradient-descent map, two faces, and the assumptions that separate what is proved from what is observed.
:::
