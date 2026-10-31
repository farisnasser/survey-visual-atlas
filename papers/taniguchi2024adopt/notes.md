# ADOPT: Modified Adam can converge with any β2 with the optimal rate

:::minute The paper in one minute
**Question:** Why can Adam fail to converge in theory for fixed choices of its second-moment decay $\beta_2$?

**Model/Method:** Reorder moment estimation so the current stochastic gradient is normalized with the previous second-moment estimate, and apply momentum after normalization.

**Finding:** ADOPT attains an $O(T^{-1/2})$ first-order convergence measure for any allowed $\beta_2$, under bounded second moments and smoothness, without assuming almost-sure bounded stochastic gradients.

**Why we care:** A subtle dependence between current noise and adaptive denominators can matter as much as the optimization landscape itself.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Adam gives every coordinate its own step size

Adam keeps two running averages per coordinate: the **first moment** $m_t$, an exponential average of gradients, and the **second moment** $v_t$, an exponential average of squared gradients. It steps by $-\alpha\, m_t / \sqrt{v_t}$ (plus a small $\epsilon$). Dividing by $\sqrt{v_t}$ removes the gradient's scale: if one coordinate's gradient is a constant $c$, then after bias correction $m_t = c$ and $v_t = c^2$, so the step is $-\alpha\,\mathrm{sign}(c)$ whether $c = 0.01$ or $c = 100$. Coordinates with large, noisy gradients get smaller steps; quiet ones get larger steps. The size of $v_t$ is controlled by $\beta_2$, the decay of the second-moment average.

### 2. A scale that sees the current sample can point the step the wrong way

A stochastic gradient $g_t$ is **unbiased** when its average, given everything that happened before step $t$, equals the true gradient $\nabla f$. If the step multiplies $g_t$ by a scale $s$ computed from the past only, the average step is $s\,\nabla f$: a descent direction. If $s$ also depends on $g_t$, that is no longer true. Use the paper's toy gradient (its Eq. 17 with $k = 10$): $f(\theta) = \theta$, so $\nabla f = 1$, but the sample is $g = 100$ with probability $0.1$ and $g = -10$ otherwise (mean $10 - 9 = 1$). With $v_{t-1} = 1$ and $\beta_2 = 0.9$:

| | $g = 100$ (prob. 0.1) | $g = -10$ (prob. 0.9) | average |
|---|:-:|:-:|:-:|
| past scale, $g / \sqrt{v_{t-1}}$ | $100$ | $-10$ | $+1$ |
| same-step scale, $g / \sqrt{v_t}$ | $3.16$ | $-3.03$ | $-2.41$ |

The same-step denominator shrinks the rare large sample far more than the common small one, so the average normalized gradient has the *wrong sign*: subtracting it moves $\theta$ uphill on average. This single-step calculation is a simplified illustration of the failure mode the paper analyzes.

### 3. What the convergence measure says, and what it does not

For nonconvex problems the target is a point with a small gradient, and the standard yardstick for stochastic methods is how fast the best iterate's gradient shrinks with the number of steps $T$. A rate $O(T^{-1/2})$ means $100$ times more steps buy a $10$ times smaller bound; for SGD this rate is known to be the best possible up to constants. The paper measures the gradient as $\big(\mathbb{E}\|\nabla f\|^{4/3}\big)^{3/2}$. By Jensen's inequality this is never larger than $\mathbb{E}\|\nabla f\|^2$, and it can be smaller: if $\|\nabla f\|$ is $0$ or $2$ with equal probability, the first is $(0.5 \times 2^{4/3})^{3/2} \approx 1.41$ while the second is $2$. So a bound on the paper's measure is not a bound on the squared gradient norm.

:::key Putting the three ideas together
Adam scales each coordinate by a running estimate of its gradient size (idea 1). When that estimate already contains the current sample, the scaled step can be biased, even toward ascent (idea 2). ADOPT normalizes the fresh gradient by the *previous* estimate and only then applies momentum, which restores an unbiased descent argument and yields the $O(T^{-1/2})$ rate in the paper's $4/3$-moment measure (idea 3) for any $\beta_2$, as made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Understand the dependency problem

Adam uses first-moment momentum $m_t$ and a second-moment scale $v_t$ based on squared recent gradients. Roughly, it moves like $-m_t/\sqrt{v_t}$. But $v_t$ already contains the same random sample $g_t$ whose effect it rescales. Thus the random direction and random denominator are coupled, complicating a guaranteed descent argument. Think of estimating a student's exam score and then using that same score to choose how much to scale it. The scaling is now statistically dependent on the observation. The algebraic issue here is conditional dependence, not that the use of $g_t$ in a denominator is always intrinsically bad.

## 2. Worked example: the ordering really changes the step

Suppose the previous second-moment estimate is $v_{t-1} = 1$, the new scalar gradient is $g_t = 4$, and $\beta_2 = 0.9$ (ignore $\epsilon$ and momentum for illustration). A same-step denominator gives

$$v_t = 0.9(1) + 0.1(4^2) = 2.5, \qquad \frac{g_t}{\sqrt{v_t}} = \frac{4}{\sqrt{2.5}} \approx 2.53.$$

ADOPT instead divides the current gradient by the previous denominator:

$$\frac{g_t}{\sqrt{v_{t-1}}} = \frac{4}{1} = 4.$$

The point is not that the ADOPT step should always be bigger. It is that $v_{t-1}$ is measurable using past samples, so it does not depend on the fresh noise $g_t$ conditional on the history.

:::key Key first-principles connection
If $g_t$ is unbiased conditionally on past information, then multiplying it by a past-computed scale preserves an analyzable conditional expectation. A denominator that already uses $g_t$ breaks this simple separation.
:::

## 3. Why simply lagging the denominator is not enough

For RMSProp without momentum, the lagged denominator already eliminates the most direct current-sample coupling. With Adam-style momentum, however, the momentum vector itself is an accumulation of earlier gradients that are also embedded in the adaptive scale. The authors therefore change the *order* of the operations:

$$\boxed{m_t = \beta_1 m_{t-1} + (1 - \beta_1)\,\frac{g_t}{\max\{\sqrt{v_{t-1}}, \epsilon\}}},$$

$$\theta_t = \theta_{t-1} - \alpha_t m_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2.$$

The componentwise formula above captures the important dependency structure; Algorithm 1 specifies initialization and the precise ordering used in the implementation. The fresh gradient is normalized using history, and its normalized version is accumulated into momentum. By comparison, Adam forms momentum first and then normalizes it.

## 4. What is the actual mathematical guarantee?

Theorem 4.1 assumes a lower-bounded objective, unbiased stochastic gradients, $L$-smoothness, and a finite *second moment* $\mathbb{E}\|g_t\|^2 \le G^2$ (Assumptions 2.1–2.3, 2.5). With $\alpha = \Theta(T^{-1/2})$ it controls a first-order stationarity measure:

$$\min_{1 \le t \le T} \Big(\mathbb{E}\|\nabla f(\theta_{t-1})\|^{4/3}\Big)^{3/2} = O(T^{-1/2}).$$

The exponent and expectation are part of the theorem, so it should not silently be rewritten as a simple bound on $\mathbb{E}\|\nabla f\|^2$. Importantly, the authors remove the stronger almost-sure assumption $\|g_t\| \le G$; they *do not* remove all noise or moment assumptions.

:::plain Why this belongs in the survey
It addresses convergence of stochastic adaptive updates, not second-order stationarity. The paper does not prove that ADOPT escapes strict saddles in finite time or finds global minima.
:::

## 5. Evidence, contributions, and limitations

**Contributions.** (i) Diagnose the problematic dependence between the current stochastic gradient and adaptive scaling; (ii) design ADOPT's lagged normalization-before-momentum update; (iii) establish a fixed-$\beta_2$ $T^{-1/2}$ convergence result under finite second-moment conditions; (iv) compare against Adam-type baselines across image classification, generative, language, and reinforcement-learning tasks. **Experimental reading.** The paper reports improvements on several benchmarks, but the exact gains depend on task and hyperparameters. When citing these results, distinguish "reported better performance" from the mathematical convergence theorem.

## 6. Important assumptions and cautions

- The stochastic gradient must be unbiased, with bounded second moment; this is less restrictive than almost-sure bounded gradients but is still a condition.
- Smoothness and a lower-bounded objective support the descent analysis; the rate concerns a first-order measure.
- "Any $\beta_2$" is not a promise about arbitrary learning rates, optimizer settings, or batch sequences.
- ADOPT does not provide a general saddle-escape theorem, generalization guarantee, or superiority over Adam on every model.

## 7. Quick reference

| | |
|---|---|
| **Adam coupling** | Current $g_t$ contributes to both numerator and current $v_t$. |
| **ADOPT normalization** | Normalize $g_t$ with $v_{t-1}$ first; then update momentum. |
| **Moment assumption** | $\mathbb{E}\lVert g_t\rVert^2 \le G^2$ (not almost-sure bounded $g_t$). |
| **Rate** | Theorem 4.1: $O(T^{-1/2})$ for the specified gradient-norm measure. |
| **Guarantee class** | Approximate first-order stationarity, not saddle escape. |

### Survey bridge / Papers Master note

ADOPT modifies Adam by normalizing the fresh gradient with the previous second moment before applying momentum. It proves an $O(T^{-1/2})$ first-order stationarity measure for any fixed $\beta_2$ under smoothness and finite-second-moment assumptions. Useful for adaptive stochastic dynamics, not a direct second-order saddle-escape result.

> **Primary source:** Taniguchi et al. (2024), *ADOPT: Modified Adam can converge with any β2 with the optimal rate*, NeurIPS 2024. [Original publication](https://papers.nips.cc/paper_files/paper/2024/hash/84d286e32bbee8fa3a86ee9c50e00081-Abstract-Conference.html)
