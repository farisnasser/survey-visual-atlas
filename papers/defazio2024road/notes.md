# The Road Less Scheduled: Schedule-free optimization through interpolation and iterate averaging

:::minute The paper in one minute
**Question:** Can an optimizer perform competitively without choosing a learning-rate schedule around a known stopping time $T$?

**Model/Method:** Couple a fast base sequence, a running-average sequence, and a gradient-evaluation sequence through a momentum-like interpolation.

**Finding:** Schedule-Free SGD obtains optimal convex stochastic rates under its assumptions, and Schedule-Free AdamW is competitive in broad deep-learning experiments without a prespecified decay schedule.

**Why we care:** Schedules are not the only way to obtain stable, good final iterates: where gradients are evaluated matters as much as how iterates are averaged.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. A constant step leaves SGD bouncing in a noise ball

Run SGD on $f(w) = \tfrac12 w^2$ with gradient noise of variance $\sigma^2 = 1$: $w_{t+1} = w_t - \gamma (w_t + \xi_t)$. The mean goes to $0$, but every step adds fresh noise, and the spread settles at the stationary variance $\gamma^2/\big(1 - (1 - \gamma)^2\big) = \gamma/(2 - \gamma)$. At $\gamma = 0.5$ that is $1/3$; at $\gamma = 0.05$ it is about $0.026$. A large step makes fast early progress but ends in a wide cloud; a small step ends in a tight cloud but gets there slowly. A **learning-rate schedule** (step decay, cosine decay) starts large and shrinks the step as training approaches its planned end $T$, which is why it needs $T$ in advance.

### 2. Averaging the iterates removes noise, but lags

Another way to shrink the cloud is to keep the step constant and report the **average** of the iterates. Averaging $t$ roughly independent noisy points divides their variance by about $t$: the mean of $100$ independent draws with variance $1$ has variance $0.01$. This is **Polyak–Ruppert averaging**. Its weakness is memory of the past. If SGD starts at $w = 10$ and reaches the solution $0$ on its next step, the uniform average after $10$ iterates is still $10/10 = 1$; after $100$ it is $0.1$. Early, poor iterates keep dragging the average, which is why plain averaging often trails a tuned schedule in practice.

### 3. The convex benchmark: $DG/\sqrt{T}$

For a convex loss whose stochastic gradients are bounded by $G$ (a **$G$-Lipschitz** loss), starting at distance $D$ from a minimizer, the classic result is that SGD with step $\gamma = D/(G\sqrt{T})$, reported as the average of its iterates, has expected suboptimality at most $DG/\sqrt{T}$. No first-order method can do better in the worst case for this class, up to constants. With $D = G = 1$ and $T = 10\,000$ steps that is $0.01$. The step in this result depends on $T$, which is the dependence a schedule-free method tries to remove in practice.

:::key Putting the three ideas together
A constant step leaves SGD in a noise ball (idea 1); averaging shrinks the ball but lags behind the moving iterates (idea 2); and the target for convex problems is the optimal $DG/\sqrt{T}$ rate (idea 3). Schedule-Free keeps a fast base sequence, a running average, and evaluates gradients at a point between them, which reaches that optimal rate for every interpolation weight $\beta$, as made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Why schedules are awkward

Cosine decay and related schedules usually need a planned training horizon $T$. If training stops early or continues longer, the intended step-size trajectory may be inappropriate. Plain constant-step SGD can be good for early progress, while averaging its iterates reduces variance but may lag behind the changing solution. The paper unifies these ideas with *three* sequences, which must not be mixed up:

$$\begin{aligned} y_t &= (1 - \beta) z_t + \beta x_t && \text{(where the gradient is measured),} \\ z_{t+1} &= z_t - \gamma \nabla f(y_t; \zeta_t) && \text{(base update),} \\ x_{t+1} &= (1 - c_{t+1}) x_t + c_{t+1} z_{t+1} && \text{(reported average).} \end{aligned}$$

With $c_{t+1} = 1/(t+1)$, $x$ is a running average of $z$. Parameter $\beta$ interpolates between gradients at the base point ($\beta = 0$) and the averaged point ($\beta = 1$).

## 2. Worked example: the three sequences really differ

Take $f(w) = \tfrac12 (w - 2)^2$, start at $x_1 = z_1 = 0$, choose $\gamma = 0.5$ and $\beta = 0.8$. At step 1:

$$y_1 = 0, \quad g_1 = y_1 - 2 = -2, \quad z_2 = 0 - 0.5(-2) = 1, \quad x_2 = \tfrac12(0) + \tfrac12(1) = 0.5.$$

At step 2, the new gradient is evaluated not at $z_2 = 1$ or $x_2 = 0.5$ but at

$$y_2 = 0.2(1) + 0.8(0.5) = 0.6.$$

Therefore $g_2 = -1.4$, $z_3 = 1 + 0.7 = 1.7$, and $x_3 = \tfrac23(0.5) + \tfrac13(1.7) = 0.9$. The base moves fast; the average lags and stabilizes; the gradient point interpolates between them.

:::key How to think about the algorithm
The algorithm is not merely "SGD with a constant learning rate." Its defining feature is *where* the gradient is evaluated and *how* the resulting base steps are averaged.
:::

## 3. Theorem: what is guaranteed and what is not

Theorem 1 studies an i.i.d. stochastic convex problem with $G$-Lipschitz losses, a minimizer $x^\star$, and $D = \|x_1 - x^\star\|$. For the theorem's horizon-dependent choice $\gamma = D/(G\sqrt{T})$, any $\beta \in [0, 1]$ yields

$$\boxed{\mathbb{E}\big[F(x_T) - F(x^\star)\big] \le \frac{DG}{\sqrt{T}}.}$$

This is a stochastic *convex* guarantee for the averaged sequence $x_T$; it is not a proof that arbitrary deep neural networks converge to a global minimum. It may seem ironic that the theorem chooses $\gamma$ using $T$. The answer is that this gives a clean worst-case proof, while the practical schedule-free motivation and algorithms explore using a fixed rate without specifying a decay horizon.

## 4. Why averaging and the gradient location must be coupled

Polyak–Ruppert averaging uses an average $x_t$ of base iterates $z_t$ but computes gradients at $z_t$. In the paper's notation, this is $\beta = 0$. Primal averaging computes gradients at $x_t$, corresponding to $\beta = 1$. Schedule-Free uses intermediate $\beta$ values to gain some averaging stability while remaining responsive to new gradients. Theorem 2 unifies these choices through an online-to-batch conversion bound: a weighted average's optimization error can be controlled by an online-gradient regret term. This is the theoretical bridge between scheduling, averaging, and momentum.

### What happens in real optimizers?

Schedule-Free AdamW adapts the mechanism to an Adam-like base optimizer and evaluates/test-uses the proper averaged sequence. The paper also discusses how momentum allows aggressive step sizes on quadratic models. These implementation details are separate from Theorem 1's abstract convex setting.

:::warn Practical caveat
During training, gradients may be evaluated at $y_t$, but evaluation typically uses the averaged weights $x_t$. Accidentally reporting the wrong sequence changes what algorithm is being assessed.
:::

## 5. Findings and main contributions

**Contributions.** (i) A three-sequence schedule-free update; (ii) an online-to-batch theory unifying interpolation and averaging; (iii) convex convergence guarantees independent of the particular fixed interpolation parameter $\beta$; and (iv) broad empirical comparisons showing strong training results for Schedule-Free AdamW. **Evidence.** Figures 2–4 examine optimization behavior across tasks, including language modeling and the AlgoPerf suite. The authors report competitiveness with tuned schedule-based baselines and describe Schedule-Free AdamW as a major part of their winning 2024 AlgoPerf self-tuning submission. These are empirical comparisons, not universal guarantees.

## 6. Assumptions and limitations

- The flagship $DG/\sqrt{T}$ theorem requires convex Lipschitz losses and i.i.d. sampling; non-convex deep-learning outcomes are empirical.
- A fixed learning rate still must be selected and may need warmup or tuning. "Schedule free" means no prescribed decay schedule, not no learning-rate hyperparameters.
- Theory's choice $\gamma = D/(G\sqrt{T})$ uses a training horizon; practical schedule-free behavior is a separate design claim.
- The work is not principally a strict-saddle-escape method, nor does it prove global optimization or generalization for all networks.

## 7. Quick reference

| | |
|---|---|
| **Base weights** | $z_t$: updated using the sampled gradient. |
| **Average weights** | $x_t$: the running, reported/evaluation sequence. |
| **Gradient point** | $y_t = (1 - \beta) z_t + \beta x_t$. |
| **Step** | $z_{t+1} = z_t - \gamma \nabla f(y_t; \zeta_t)$. |
| **Convex guarantee** | $\mathbb{E}[F(x_T) - F(x^\star)] \le DG/\sqrt{T}$ (Theorem 1). |
| **Not proved** | General non-convex global convergence or saddle escape. |

### Survey bridge / Papers Master note

Schedule-Free SGD/AdamW replaces a predefined learning-rate decay with coupled base, average and gradient-evaluation sequences. Theorem 1 provides an optimal convex stochastic rate, and broader neural-network benefits are empirical. Useful when contrasting learning-rate schedules with dynamic update modifications and Lookahead.

> **Primary source:** Defazio, Yang, Mehta, Mishchenko, Khaled & Cutkosky (2024), *The Road Less Scheduled*, NeurIPS 2024. [Original publication](https://proceedings.neurips.cc/paper_files/paper/2024/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html)
