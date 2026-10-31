# Why Gradient Clipping Accelerates Training

:::minute The paper in one minute
**Problem.** Classical proofs choose step sizes from a global gradient-Lipschitz constant. Deep-network training can have strongly varying local smoothness, and large gradients often appear together with high curvature. Zhang et al. ask whether a different regularity assumption can explain why *normalizing or clipping updates* speeds up training.

**Model.** A relaxed condition $\|\nabla^2 f(x)\| \le L_0 + L_1\|\nabla f(x)\|$: smoothness can increase with gradient magnitude.

**Main result.** Under the relaxed smoothness model and the paper's other conditions, clipping and normalized-gradient methods admit faster convergence guarantees than fixed-stepsize GD; the authors give examples with arbitrarily large theoretical separation and test the explanation empirically.

**Limits.** This is an adaptivity/first-order training-dynamics result, *not* a theorem guaranteeing escape from all saddles.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. A global smoothness constant buys a safe step

A gradient is **$L$-Lipschitz** if $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$; for twice-differentiable $f$ this means every Hessian eigenvalue lies in $[-L, L]$. Then one GD step obeys the **descent lemma**

$$f(x - \eta g) \le f(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\|g\|^2, \qquad g = \nabla f(x),$$

so any $\eta < 2/L$ is guaranteed to lower the loss, and $\eta = 1/L$ lowers it by at least $\|g\|^2/(2L)$. Classical convergence rates for GD are built on this one inequality. The catch: the step must be safe for the *largest* curvature anywhere along the path.

### 2. Curvature that grows with the gradient

Take $f(x) = \tfrac12 x^2 + \tfrac14 x^4$. Its curvature is $f''(x) = 1 + 3x^2$, which is unbounded, so no global $L$ exists. But the gradient $f'(x) = x + x^3$ grows even faster, and $f''(x) \le 1 + 3|f'(x)|$ for every $x$. At $x = 5$: $f'' = 76$ and $f' = 130$, and indeed $76 \le 1 + 3 \cdot 130$. This is the paper's **$(L_0, L_1)$-smoothness** with $L_0 = 1$ and $L_1 = 3$: curvature is large only where the gradient is large. The paper shows that polynomials of degree at least 3 behave this way (its Lemma 2).

### 3. Clipping matches the step to the local curvature

A fixed step must suit the steepest point it will visit. Starting at $x = 5$, where the curvature is $76$, the rule $\eta < 2/L$ with the local $L$ gives $\eta < 2/76 \approx 0.026$. Near the minimum the curvature is only $1$, so that same step shrinks the error by a factor of just $0.974$ per iteration. **Clipped GD** uses $h = \min\{\eta_c, \gamma\eta_c/\|g\|\}$ instead. With $\eta_c = 1$ and $\gamma = 0.5$: at $x = 5$, $h = 0.5/130 \approx 0.0038$, so $h \cdot f''(5) \approx 0.29$, well inside the safe range, and the move is only $\gamma\eta_c = 0.5$ long; near the minimum, where $\|g\| < 0.5$, $h = 1$, the full step suited to curvature $1$. One rule, two regimes, no tuning to the worst point.

:::key Putting the three ideas together
Classical analysis ties the step to one global curvature bound (idea 1). Neural-network losses look more like functions whose curvature grows with the gradient (idea 2), and for them a fixed step is forced to be tiny everywhere while a clipped step is small only where the gradient is large (idea 3). The paper turns this into convergence bounds in which clipped and normalized GD can be arbitrarily faster than fixed-step GD, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Problem: why does clipping help optimization?

Classical proofs choose step sizes from a global gradient-Lipschitz constant: $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$. Deep-network training can have strongly varying local smoothness, and large gradients often appear together with high curvature. Zhang et al. ask whether a different regularity assumption can explain why *normalizing or clipping updates* speeds up training.

## 2. The new smoothness model

A canonical form of the relaxed condition is

$$\|\nabla^2 f(x)\| \le L_0 + L_1\|\nabla f(x)\|,$$

where $L_0, L_1 \ge 0$. The important difference is that smoothness can increase with gradient magnitude; it need not be globally bounded by a constant independent of $x$. The paper motivates the condition with empirical measurements along neural-network trajectories.

## 3. Contrast the updates

Full gradient descent takes

$$x_{t+1} = x_t - \eta\nabla f(x_t).$$

For large $\|\nabla f(x_t)\|$, this may be a long step exactly when local curvature is large. A typical clipped direction is

$$\widetilde g_t = g_t \min\left\{1, \frac{c}{\|g_t\|}\right\}, \qquad x_{t+1} = x_t - \eta\widetilde g_t,$$

while normalized gradient descent uses $x_{t+1} = x_t - \eta g_t/\|g_t\|$ (with a convention at $g_t = 0$). Clipping bounds the update norm without necessarily changing small gradients.

## 4. First-principles reason for adaptivity

In the simple scalar teaching function $f(x) = e^x$, one has $f'(x) = f''(x) = e^x$. Curvature is proportional to gradient magnitude; no finite global $L$ exists on $\mathbb{R}$. For large $x$, a fixed GD step of size $\eta e^x$ can be enormous. A normalized step has length $\eta$, independent of $e^x$. This is an *illustration of the proposed mechanism*, not an experiment reported in the paper.

:::theorem Main result
Under the relaxed smoothness model and the paper's other conditions, clipping and normalized-gradient methods admit faster convergence guarantees than fixed-stepsize GD; the authors give examples with arbitrarily large theoretical separation and test the explanation empirically. This is an adaptivity/first-order training-dynamics result, *not* a theorem guaranteeing escape from all saddles.
:::

## 5. Limits and importance for the survey

The bound is a model assumption, not an established universal law of all neural-network objectives. Theoretical acceleration depends on the function class, choice of step/clipping parameters and stationarity target. ReLU non-differentiability and heavy-tailed noise require their own treatments. More recent adaptive-optimizer analyses borrow the motivation but need not share the same theorems.

:::plain Survey relevance (RQ2, RQ4)
This paper is evidence that constant global smoothness is often an unrealistic lens. A clipped or normalized step responds to local gradient magnitude and curvature, which helps explain empirical robustness. Do not reinterpret its first-order rates as second-order or global guarantees.
:::

## 6. Our notes

Clipping is not just a trick to prevent exploding gradients: it can be helpful because steps should shrink in regions where big gradients are accompanied by high curvature. The paper proves improvements under a relaxed smoothness condition; whether that condition holds for a given training setup still needs checking.

> **Original source:** [https://sls.csail.mit.edu/publications/2020/TianxingHe_ICLR-2020.pdf](https://sls.csail.mit.edu/publications/2020/TianxingHe_ICLR-2020.pdf).
