## 1. Many minima, different shapes

Stop 9 ended with a question: a network's loss has many minima, so **which one does training find, and does it matter?** Stop 1 already showed that the starting point can decide the valley. This stop shows that the *step size* decides it too, and that the deciding quantity is how sharply a minimum bends.

Near a minimum $x^\star$ the gradient is zero, so the landscape is described by its curvature, the [[hessian|Hessian]] $H = \nabla^2 f(x^\star)$ (stop 4). Its largest [[eigen|eigenvalue]] is the curvature of the steepest direction out of the minimum. It has a name of its own.

:::def Sharpness
The **[[sharpness]]** of $f$ at a point $x$ is the largest eigenvalue of the Hessian there:

$$S(x) \;=\; \lambda_{\max}\big(\nabla^2 f(x)\big).$$

At a minimum all eigenvalues are $\ge 0$, and $S$ says how fast the loss climbs when we leave the minimum in the worst direction: $f(x^\star + r u) \approx f^\star + \tfrac{S}{2} r^2$ for the top eigenvector $u$ and a small distance $r$.
:::

A minimum with small sharpness sits at the bottom of a wide, gently curved basin: a **[[flat-min|flat minimum]]**. One with large sharpness sits in a narrow pit: a **sharp minimum**. In one dimension, $f(x) = \tfrac{a}{2}x^2$ has sharpness $a$, the curvature of stop 3.

Two minima can be equally deep and still very different. The landscape of the next visual has a sharp minimum with $f \approx -1.008$ and sharpness $20.8$, and a flat one with $f \approx -1.007$ and sharpness $1.97$, about 10.6 times smaller. The loss value alone cannot tell them apart.

## 2. A step size only fits minima flatter than $2/\eta$

Close to a minimum, $f$ is its [[quadratic-model|quadratic model]] $f(x) \approx f^\star + \tfrac12 (x - x^\star)^\top H\, (x - x^\star)$. Stop 4 showed what GD does on it: write the error $x_t - x^\star$ in the eigenvectors of $H$; the component $u_i$ along the eigenvector with eigenvalue $\lambda_i$ evolves on its own,

$$u_i \;\leftarrow\; (1 - \eta \lambda_i)\, u_i .$$

All components shrink if and only if $\lvert 1 - \eta\lambda_i \rvert < 1$ for every $i$, that is $0 < \eta\lambda_i < 2$. The binding one is the largest eigenvalue, so

$$\boxed{\;\text{GD with step } \eta \text{ can settle at a minimum only if its sharpness } S < 2/\eta.\;}$$

If $S > 2/\eta$, the tiniest offset along the top eigenvector is multiplied by $1 - \eta S < -1$ at every step: it flips sign and grows. With $\eta = 0.11$, a direction of curvature $20$ has factor $1 - 2.2 = -1.2$ (the offset grows by 20% per step), while a direction of curvature $2$ has factor $0.78$ and shrinks. This is the parabola rule of stop 3, applied to the steepest direction of each minimum.

Read the other way round, a step size is a **filter**: GD with step $\eta$ can only end in minima whose sharpness is below $2/\eta$. Sharp minima are invisible to large steps.

:::viz wells
:::

In the visual, the thresholds are $2/20.8 \approx 0.096$ for the sharp well and $2/1.97 \approx 1.01$ for the flat one. For $\eta$ below $0.096$, 87 of the 357 starting points settle in the sharp well and 270 in the flat one. For $\eta$ between $0.1$ and $0.2$, the 87 starts that fall into the sharp well can no longer settle; they keep bouncing inside it, held in by its walls, where the quadratic model no longer applies. From $\eta = 0.25$ to $1.0$, every one of the 357 starts ends in the flat well. Above $2/1.97$, neither minimum is stable, and nothing settles.

:::warn What the stability rule does and does not say
It is a *local* statement about one minimum: it says which minima GD **cannot** converge to. It does not say which of the remaining ones GD reaches (that still depends on the start), and it is about plain GD. Noise (stop 8) and momentum (stop 7) change the threshold; for heavy-ball momentum it becomes $(2 + 2\beta)/\eta$ (Cohen et al. 2021, Eq. 1).
:::

## 3. Inside a valley of minima, the step size picks the flat end

Neural networks do not just have a few separate minima. Because of symmetries (stop 5, and stop 11), their minima come in connected families. The smallest example is the valley of stop 5,

$$f(u, v) = \tfrac12 (uv - 1)^2 ,$$

whose minima are the whole curve $uv = 1$. At a minimum $(a, 1/a)$ the Hessian is

$$\nabla^2 f = \begin{bmatrix} 1/a^2 & 1 \\ 1 & a^2 \end{bmatrix}, \qquad \text{eigenvalues } 0 \text{ and } a^2 + \frac{1}{a^2}.$$

The zero eigenvalue points along the curve (moving along it keeps $f = 0$). The other one is the sharpness, and it changes along the curve: $2$ at $(1, 1)$, $4.25$ at $(2, 0.5)$, $16.06$ at $(4, 0.25)$. All of these points fit the data equally well.

:::viz family
:::

GD starts at $(2.5, 0.05)$, where the sharpness is $6.34$. With $\eta = 0.05$ it slides to the nearest point of the curve and stops at sharpness $6.54$. With $\eta = 0.3$ it ends at $5.42$; with $\eta = 0.4$ at $2.54$; with $\eta = 0.44$ at $2.00$, the flattest point $(1, 1)$. Every run ends below its own $2/\eta$, as section 2 requires. The larger steps first bounce across the valley, where the curvature is above $2/\eta$, and only come to rest once they reach a part of the valley flat enough to hold them.

Lewkowycz et al. 2020 analyze this effect exactly in a two-layer linear model trained on one example. Writing $\lambda_0$ for the curvature at the start, they find three phases: for $\eta < 2/\lambda_0$ the curvature barely changes during training (the *lazy* phase); for $2/\lambda_0 < \eta < 4/\lambda_0$ the loss first grows, the curvature drops below $2/\eta$, and training converges in a flatter region (the **catapult** phase); above $4/\lambda_0$ it diverges. In their experiments with wide ReLU networks the upper edge is larger, about $12/\lambda_0$, and the best test accuracy is often found in the catapult phase (observed).

## 4. The edge of stability

Section 2 predicts something about real training. If the sharpness ever exceeds $2/\eta$, GD should become unstable. Cohen et al. 2021 measured the sharpness during full-batch GD on many networks (fully connected, VGG, ResNet, Transformers) and found two things. Their paper is empirical: it reports what training does, with no convergence theorem.

:::def Progressive sharpening and the edge of stability
**[[progressive-sharpening|Progressive sharpening]]**: while $S < 2/\eta$, full-batch GD has a strong tendency to *increase* the sharpness.

**[[edge-of-stability|Edge of stability]]**: once $S$ reaches $2/\eta$, it stops rising and hovers at or just above $2/\eta$. The loss keeps decreasing over long periods, but no longer at every step.
:::

The second finding is the surprise. A quadratic with $S > 2/\eta$ diverges; a network sitting at $S \approx 2/\eta$ does not, and keeps training. The visual replays a run of this kind on a small network (an Atlas replication, not the paper's own runs).

:::viz eos
:::

All four runs start at sharpness $4.05$ and loss $0.444$. With $\eta = 0.1$, the sharpness rises to $7.8$, far below $2/\eta = 20$, and the loss never goes up. With $\eta = 0.3$, $0.36$ and $0.42$, it overshoots to $7.33$, $6.84$ and $5.84$, comes back, and then sits at an average of $6.66$, $5.55$ and $4.75$ over the last 1000 steps, against $2/\eta = 6.67$, $5.56$ and $4.76$. The loss rises in 73, 42 and 96 of the 300 ten-step windows, yet ends below $10^{-5}$ in all three.

Cohen et al. report variations: with momentum the sharpness settles at the momentum threshold $(2 + 2\beta)/\eta$; with SGD it does not flatline at one value; with the cross-entropy loss the sharpness drops near the end of training.

## 5. Self-stabilization: why GD does not blow up

Damian, Nichani and Lee 2023 explain the edge of stability with one extra term of the Taylor expansion. Let $u$ be the top eigenvector, $x$ the displacement of the iterate along $u$, and $y = S - 2/\eta$ how far the sharpness is past the edge. Two numbers describe the landscape around the trajectory: $\alpha$, how fast the sharpness grows as GD descends (progressive sharpening, $\alpha > 0$), and $\beta = \lVert \nabla S \rVert^2$, how fast the sharpness changes when we move. Their simplified dynamics (Eq. 3) are

$$x_{t+1} = -(1 + \eta\, y_t)\, x_t, \qquad y_{t+1} = y_t + \eta\Big(\alpha - \frac{\beta}{2}\, x_t^2\Big).$$

The first equation is the stability rule: the oscillation along $u$ flips sign every step and grows when $y > 0$ ($S > 2/\eta$), shrinks when $y < 0$. The second has two forces on the sharpness: $\alpha$ pushes it up, and the oscillation pushes it down by $\tfrac{\beta}{2} x^2$. That last term comes from the third derivative: bouncing with amplitude $x$ across the top direction adds $\tfrac{x^2}{2} \nabla S$ to the average gradient, so the step also moves *downhill on the sharpness*. The two forces balance at amplitude $\delta = \sqrt{2\alpha / \beta}$. One cycle has four stages:

1. **Progressive sharpening**: $S < 2/\eta$, $x$ small, $S$ rises.
2. **Blow-up**: $S > 2/\eta$, the oscillation grows.
3. **Self-stabilization**: $\lvert x \rvert > \delta$, the oscillation drags $S$ back down.
4. **Return to stability**: $S < 2/\eta$ again, the oscillation shrinks.

:::viz stabilize
:::

The visual uses the toy landscape from the paper's App. B, on which GD follows these two equations exactly. With $\alpha = 0.002$ (and $\beta = 1$), $\delta = 0.063$. Starting at $S = 1.95$ and $x = 0.02$, the sharpness crosses $2/\eta$ at step 27 and then stays between $1.80$ and $2.16$, with $\lvert x \rvert \le 0.29$; its average over steps 100–600 is $1.99$, the edge $2/\eta = 2$.

Their main result, Theorem 1 (under five assumptions), says that for $O(\varepsilon^{-1})$ steps GD follows a *constrained trajectory*: projected gradient descent on the set where $S \le 2/\eta$ (and the gradient has no component along the top eigenvector). The edge of stability is GD automatically doing gradient descent with the constraint $S \le 2/\eta$, without being told to. The statement is over a finite horizon, and in the discrete toy the cycles slowly grow over long runs.

:::key The idea to keep
At large step sizes, GD does not just tolerate the curvature it finds: it **shapes** it. It cannot stay where $S > 2/\eta$, and the oscillations it makes there move it to where $S \le 2/\eta$. So the step size acts as a regularizer on sharpness.
:::

## 6. Do flat minima generalize better?

Everything so far is about the training loss. What we care about is how the model does on new data, its **[[generalization]]**: the *test loss* on data it was not trained on. The *generalization gap* is the test loss minus the training loss.

The classic intuition: the test loss landscape is a slightly shifted, slightly deformed version of the training landscape. In a flat basin a small shift costs little; at the bottom of a sharp pit it can cost a lot. This is an intuition, not a theorem, and the literature around it is a debate.

- **Hochreiter & Schmidhuber 1997** define a flat minimum as a large connected region of weight space where the error stays roughly constant. Their argument is from minimum description length: a flat minimum can be described with low-precision weights, that is, with few bits, and a simpler description overfits less. They give an algorithm, Flat Minimum Search, that looks for such regions.
- **Keskar et al. 2017** train six networks with Adam using small batches (256 examples) and large ones (10% of the data). The large-batch runs generalize worse, and the minima they reach are sharper by the authors' measure. Their explanation: the gradient noise of small batches pushes the iterate out of sharp basins (observed, not proved).
- **Dinh et al. 2017** point out that sharpness depends on how the network is parameterized. In a ReLU network, multiplying one layer's weights by $c$ and dividing the next layer's by $c$ computes exactly the same function, but the Hessian changes: they show that its largest eigenvalue can be made arbitrarily large (Theorem 4), so a "sharp" minimum can compute the same function as a "flat" one.

Section 3 shows Dinh's point directly. A ReLU unit with input weight $u > 0$ and output weight $v$ computes $v \cdot \mathrm{ReLU}(u\,z) = uv\, z$ for $z \ge 0$, so every point of the curve $uv = 1$ is the same function. Its sharpness is $2$ at $(1, 1)$ and $100.01$ at $(10, 0.1)$.

:::warn Flatness is not a property of the function
Sharpness is measured in a coordinate system, and rescaling the coordinates changes it without changing the network's predictions. Any claim "flat minima generalize better" needs a measure of flatness that does not change under such rescalings (the filter normalization of Li et al. 2018, stop 11, is one attempt). Flatness and generalization are correlated in many experiments; a general causal law is not established.
:::

## 7. Sharpness-aware minimization

If flat minima are preferable, why not ask for them directly? **[[sam|Sharpness-aware minimization]]** (SAM; Foret et al. 2021) replaces the loss by its worst value in a small ball:

$$\min_x \; \max_{\lVert \epsilon \rVert \le \rho} f(x + \epsilon) .$$

Here $\rho$ (rho) is the radius of the ball, a hyper-parameter; it is *not* the Hessian-Lipschitz constant of stop 5. A point scores well only if its whole neighbourhood is low, which a sharp pit is not. The motivation in the paper is a PAC-Bayes generalization bound.

The inner maximum has no formula, so SAM approximates it to first order: the worst direction is uphill, $\hat\epsilon = \rho\, \nabla f(x) / \lVert \nabla f(x) \rVert$. The gradient of the outer problem is then approximated by the gradient at the perturbed point (dropping second-order terms), which gives the update

:::idea SAM, one step
$$x_{t+1} = x_t - \eta\, \nabla f\big(x_t + \hat\epsilon_t\big), \qquad \hat\epsilon_t = \rho\, \frac{\nabla f(x_t)}{\lVert \nabla f(x_t) \rVert}.$$

First climb a distance $\rho$ uphill, then step with the gradient found there. Cost: two gradients per step, about twice the price of GD. Foret et al. report lower test error on CIFAR-10/100, ImageNet and fine-tuning tasks, and a lower Hessian spectrum for a network trained with SAM (observed).
:::

Agarwala & Dauphin 2023 analyze the **unnormalized** version $x_{t+1} = x_t - \eta \nabla f\big(x_t + \rho \nabla f(x_t)\big)$. On a quadratic $f = \tfrac12 x^\top H x$ it reads $x_{t+1} = x_t - \eta (H + \rho H^2) x_t$, so along an eigenvalue $\lambda$ the factor is $1 - \eta\lambda(1 + \rho\lambda)$, and the stability condition becomes

$$\eta\, \lambda\, (1 + \rho \lambda) < 2 .$$

Their Theorem 2.2 proves this rule near a zero-loss point for squared-error models. SAM's edge of stability is therefore *lower* than GD's: with $\eta = 0.5$ and $\rho = 0.2$, GD's edge is $2/\eta = 4$, SAM's is $\lambda \approx 2.62$. The Atlas page on their paper replicates this on a small network: with SAM, the sharpness settles at SAM's edge to within 0.1%.

:::viz sam
:::

From $(2.2, 0.1)$ with $\eta = 0.3$, GD ends at sharpness $4.91$. Unnormalized SAM ends at $4.20$, $2.96$ and $2.00$ for $\rho = 0.05$, $0.1$ and $0.2$, below its edges $5.28$, $4.57$ and $3.79$. Normalized SAM behaves differently: its perturbation always has length $\rho$, even at a minimum, so it never stops moving. With $\rho = 0.1$ it drifts along the valley to near $(1, 1)$ and keeps jittering there (average sharpness of the last 200 steps $1.94$, $\lvert uv - 1 \rvert = 0.085$ at the end).

:::warn What SAM does not guarantee
The one-step update is an approximation of the min-max objective, not its exact solution, and the Atlas page on Foret et al. finds toy starts where the two disagree. SAM's benefits on test error are experimental; the PAC-Bayes motivation is a bound on a related quantity, not a proof that the algorithm generalizes better.
:::

## 8. Warmup: start small, end large

A common recipe is **[[warmup|learning-rate warmup]]**: start with a small learning rate and increase it to the target $\eta_{\text{trgt}}$ over the first few hundred or thousand steps. Sections 2–5 suggest why it helps. At initialization the sharpness may be far above $2/\eta_{\text{trgt}}$; jumping straight to $\eta_{\text{trgt}}$ then means divergence, or a violent catapult. Raising $\eta$ gradually lets the self-stabilization of section 5 lower the sharpness step by step, so that it always stays near $2/\eta_t$.

Kalra & Barkeshli 2024 study this directly and conclude that the main benefit of warmup is to let the network tolerate a larger target learning rate, by moving it to flatter, better-conditioned regions. Their model is the dynamics of section 5 with a time-dependent $\eta_t$. Whether training first sharpens or first flattens depends on the initialization and parameterization.

For Adam (stop 9), the relevant curvature is that of the *preconditioned* Hessian. Late in training the threshold becomes $(2 + 2\beta_1)/\big((1 - \beta_1)\eta\big)$, which is $38/\eta$ for $\beta_1 = 0.9$. Early in training, Adam's second-moment estimate starts at zero, which makes the preconditioned sharpness huge. Kalra & Barkeshli propose initializing it with the first squared gradient (GI-Adam); this lowers the initial preconditioned sharpness by a factor $\sqrt{1 - \beta_2}$, about $1/31.6$ for $\beta_2 = 0.999$ (the paper rounds this to "around two orders of magnitude"), and is equivalent to Adam with an automatic warmup $\eta_t = \eta_{\text{trgt}} \sqrt{1 - \beta_2^t}$.

:::atlas Where this shows up in the Atlas
- [Cohen et al. 2021](paper.html?id=cohen2021gradient): progressive sharpening and the edge of stability (section 4), with the run replayed above.
- [Lewkowycz et al. 2020](paper.html?id=lewkowycz2020large): the catapult phase between $2/\lambda_0$ and $4/\lambda_0$ (section 3).
- [Damian, Nichani & Lee 2023](paper.html?id=damian2023self): self-stabilization and the constrained trajectory (section 5).
- [Kalra, He & Barkeshli 2025](paper.html?id=kalra2025universal): a minimal model that shows sharpening, the edge of stability and the route to chaos.
- [Hochreiter & Schmidhuber 1997](paper.html?id=hochreiter1997flat), [Keskar et al. 2017](paper.html?id=keskar2017large) and [Dinh et al. 2017](paper.html?id=dinh2017sharp): the flat-versus-sharp debate (section 6).
- [Foret et al. 2021](paper.html?id=foret2021sharpness) and [Agarwala & Dauphin 2023](paper.html?id=agarwala2023sam): SAM and its edge of stability (section 7).
- [Kalra & Barkeshli 2024](paper.html?id=kalra2024warmup): why warmup works (section 8).
- [Galli et al. 2026](paper.html?id=galli2026flatland): line searches whose accepted steps have $\eta \cdot \lambda_{\max}$ around 2 or above from early in training.
:::

:::problem The next problem
We have seen where GD ends among many minima. But we have quietly assumed that the minima it finds are *good* ones, low enough to be useful. For a non-convex function nothing guarantees that. **Network losses are non-convex, yet training usually works: what is special about their landscapes?**
:::
