# SAM Operates Far from Home: Eigenvalue regularization as a dynamical phenomenon

:::minute The paper in one minute
**Question:** Why does sharpness-aware minimization (SAM) suppress large Hessian eigenvalues *during training*, not only at a final minimum?

**Model/Method:** Analyze unnormalized SAM in quadratic regression; study stability of its discrete update and compare predicted sharpness with neural-network experiments.

**Finding:** SAM has a modified edge-of-stability condition, $\eta\lambda(1 + \rho\lambda) \simeq 2$, which can stabilize curvature below the GD threshold.

**Why we care:** SAM's sharpness effect is partly a training-dynamics phenomenon; analyzing only final minima misses the mechanism.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. SAM takes the gradient at a nearby worse point

SAM was designed to prefer flat regions by minimizing the *worst* loss in a small ball of radius $\rho$ around the weights, $\max_{\|\epsilon\| \le \rho} L(\theta + \epsilon)$. To first order the worst direction is the gradient itself, so SAM perturbs the weights uphill, $\theta + \delta$, and descends with the gradient measured there. On $L(x) = \tfrac12 \lambda x^2$ with $\lambda = 3$ at $x = 1$ ($L = 1.5$, gradient $3$): the normalized step $\delta = \rho\, g/\|g\| = 0.2$ lands at $x = 1.2$, where the loss is $2.16$ and the gradient is $3.6$. The unnormalized step $\delta = \rho g = 0.6$ lands at $x = 1.6$, where the gradient is $4.8 = (\lambda + \rho\lambda^2)\,x$. In both cases the step SAM takes is larger than GD's, and the extra amount grows with the curvature.

### 2. A stability threshold acts as a ceiling on sharpness

On a quadratic, GD with step $\eta$ diverges along any direction with curvature above $2/\eta$ (the notes derive this in Section 2). In neural-network training this has a striking consequence, documented by Cohen et al. (2021) and called the **edge of stability**: the sharpness $\lambda_{\max}$ tends to rise during training until it reaches about $2/\eta$, then stays there while the loss keeps falling. The threshold is not only a limit on the step size; it ends up setting the curvature the network sits at. Change the threshold and the sharpness follows. This paper asks what threshold SAM has.

### 3. For squared loss, the Jacobian's Gram matrix plays the role of curvature

With mean squared error the paper tracks the **residuals** $z = f(\theta) - y$ and the Jacobian $J = \partial z / \partial \theta$. One GD step moves the weights by $-\eta J^\top z$ and so moves the residuals by about $-\eta\, J J^\top z$:

$$z_{t+1} \approx \big(I - \eta\, J J^\top\big)\, z_t .$$

The eigenvalues $\lambda_i$ of the Gram matrix $JJ^\top$ (the neural tangent kernel) therefore play the role of curvatures, and GD is stable when every $\eta\lambda_i < 2$. Unnormalized SAM adds the perturbation $\rho J^\top z$ before taking the gradient, which turns the multiplier into $I - \eta JJ^\top(I + \rho JJ^\top)$ when $J$ is held fixed. Each eigenvalue $\lambda$ is replaced by $\lambda(1 + \rho\lambda)$.

### 4. Normalized and unnormalized SAM are different dynamical systems

The usual SAM implementation normalizes, $\delta = \rho\, g/\|g\|$: the perturbation keeps length $\rho$ even when the gradient is tiny. On the 1-D quadratic this gives $x_{t+1} = (1 - \eta\lambda)\,x_t - \eta\lambda\rho\,\mathrm{sign}(x_t)$, which is not linear in $x$. With $\lambda = 3$, $\eta = 0.5$, $\rho = 0.2$, a start at $x = 0.05$ jumps to $-0.325$, and the iterates settle on a bounce between $+0.6$ and $-0.6$; they never reach the minimum. Unnormalized SAM, $\delta = \rho g$, gives a linear map with a clean multiplier. That is why the paper's stability formulas are stated for the unnormalized version. This toy is ours, not the paper's.

:::key Putting the four ideas together
SAM descends with a gradient taken at a nearby uphill point (idea 1). In training, the stability threshold of the update sets the sharpness the network ends up at (idea 2). For unnormalized SAM and squared loss, the threshold moves from $\eta\lambda = 2$ to $\eta\lambda(1 + \rho\lambda) = 2$ (idea 3), which is met at a smaller $\lambda$; the normalized variant behaves differently (idea 4). That shifted edge of stability, and evidence that networks sit at it, is made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Start with the update: what does SAM do?

Ordinary gradient descent uses the current gradient $g = \nabla L(\theta)$. SAM first takes a small step *toward higher loss*, then evaluates the gradient at that perturbed point. With a normalized perturbation, $\delta = \rho g / \|g\|$, this gives $\theta^+ = \theta - \eta \nabla L(\theta + \delta)$. The paper's clean stability derivation instead uses *unnormalized* SAM, $\delta = \rho g$. Distinguishing these two variants matters: the simple equation below is exact for the unnormalized quadratic case, not a universal equation for every SAM implementation.

## 2. Worked example: large curvature becomes less stable

Consider a one-dimensional quadratic loss $L(x) = \tfrac12 \lambda x^2$, with curvature $\lambda > 0$. Its gradient is $\lambda x$, so ordinary GD updates

$$x^+ = (1 - \eta\lambda)\, x.$$

The distance to zero shrinks exactly when $|1 - \eta\lambda| < 1$, or $0 < \eta\lambda < 2$. That is the familiar quadratic edge-of-stability boundary. For unnormalized SAM, first perturb $x$ by $\rho\lambda x$. Evaluate the gradient there:

$$g_{\mathrm{SAM}} = \lambda(x + \rho\lambda x) = (\lambda + \rho\lambda^2)\, x.$$

The resulting update is therefore

$$\boxed{x^+ = \big[1 - \eta(\lambda + \rho\lambda^2)\big]\, x.}$$

SAM is stable on this quadratic exactly if

$$0 < \eta(\lambda + \rho\lambda^2) < 2.$$

:::key Read the equation, not just the symbols
SAM replaces effective curvature $\lambda$ by $\lambda + \rho\lambda^2$. A large Hessian eigenvalue therefore hits the stability boundary *earlier* than under ordinary GD. As training adapts, this can create pressure toward lower sharpness.
:::

## 3. A concrete edge-of-stability calculation

Set learning rate $\eta = 0.5$ and unnormalized SAM radius $\rho = 0.2$. Under GD, the critical curvature is $\lambda_{\mathrm{GD}} = 2/0.5 = 4$. SAM instead reaches its boundary when

$$0.5(\lambda + 0.2\lambda^2) = 2 \quad\Longleftrightarrow\quad 0.2\lambda^2 + \lambda - 4 = 0.$$

The positive root is

$$\lambda_{\mathrm{SAM}} = \frac{-1 + \sqrt{4.2}}{0.4} \approx 2.62.$$

At $\lambda = 3$, ordinary GD has multiplier $1 - 0.5(3) = -0.5$ (stable oscillation), while SAM's multiplier is $1 - 0.5(3 + 1.8) = -1.4$ (unstable on this fixed quadratic). Thus SAM is selective against high curvature at a level GD could still tolerate. This calculation is a *pedagogical quadratic example*, not a neural-network convergence theorem.

## 4. What the paper actually analyzes

The authors use a quadratic regression model $f(\theta) = y + G^\top\theta + \tfrac12 Q(\theta, \theta)$. Its prediction residuals and Jacobian evolve together: a changing model representation changes effective curvature. The paper studies the Jacobian Gram eigenvalues (an NTK-related quantity) and derives the SAM stability condition near a zero-residual solution in Theorem 2.2.

$$\boxed{\eta\lambda_{\max}(1 + \rho\lambda_{\max}) \approx 2.}$$

This is a *local stability boundary*. Its experimentally observed persistence during training is an additional dynamical finding, not a global theorem about all neural-network trajectories.

### Why do batch size and normalization matter?

Theorem 2.1 shows weak initial suppression for full-batch training in their simplified random quadratic model. Theorem 2.3 shows minibatch effects can strengthen regularization and introduce dependence on the sampling structure rather than on the full-batch Hessian alone. A normalized SAM perturbation is not identical to the unnormalized calculation above.

:::plain Connection to the previous EoS papers
Li et al. (2022) describe progressive sharpening; Chen–Bruna (2023) analyze nonlinear dynamics beyond the usual threshold. Agarwala–Dauphin show that SAM itself shifts the relevant stability boundary.
:::

## 5. Evidence, contributions, and careful interpretation

**Contributions.** (i) A dynamical rather than merely final-minimum account of SAM's curvature regularization; (ii) a modified edge-of-stability prediction for unnormalized SAM; (iii) analysis of batch-size effects; and (iv) experiments showing that the modified threshold predicts sharpness in selected networks. **Experimental picture.** In Figure 3, ordinary GD on a fully connected MSE model stabilizes near $\eta\lambda_{\max} = 2$, whereas SAM reaches a smaller leading eigenvalue consistent with $\eta(\lambda_{\max} + \rho\lambda_{\max}^2) \approx 2$. Experiments on WideResNet / CIFAR-10 provide additional evidence that the effect can appear in realistic training.

## 6. Important assumptions and limitations

- The exact simple threshold is derived for *unnormalized* SAM and a quadratic approximation. Theorem 2.2 concerns a smooth model near a zero-residual solution with nondegenerate Gram spectrum.
- Sharpness stabilization away from minima is an observed phenomenon and supported by simplified-model analysis, not a universal finite-time convergence theorem.
- Batch size can change the dynamics substantially; full-batch reasoning cannot simply be imported into arbitrary minibatch SAM runs.
- A lower Hessian eigenvalue does not by itself prove superior generalization or global optimality.

## 7. Quick reference

| | |
|---|---|
| **GD stability** | $0 < \eta\lambda < 2$ for a positive quadratic eigenmode. |
| **Unnormalized SAM** | $\delta = \rho\nabla L$, then $\theta^+ = \theta - \eta\nabla L(\theta + \delta)$. |
| **SAM effective curvature** | $\lambda + \rho\lambda^2$ in the fixed-quadratic example. |
| **SAM–EoS** | $\eta\lambda_{\max}(1 + \rho\lambda_{\max}) \approx 2$. |
| **Result type** | Local stability theorem + empirical dynamical evidence; no universal saddle escape guarantee. |

### Survey bridge / Papers Master note

SAM's eigenvalue regularization can arise throughout training via a shifted edge-of-stability threshold, not only from penalizing final sharp minima. The exact threshold relies on an unnormalized local quadratic model, while experiments support related dynamics in networks. Useful alongside Foret (2021), Li (2022), and Kalra (2025).

> **Primary source:** Agarwala & Dauphin (2023), *SAM operates far from home: Eigenvalue regularization as a dynamical phenomenon*, ICML 2023. [Original publication](https://proceedings.mlr.press/v202/agarwala23a.html)
