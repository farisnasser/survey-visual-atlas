# Adaptive Subgradient Methods for Online Learning and Stochastic Optimization

:::minute The paper in one minute
**Problem:** In sparse learning, a useful feature might appear in only a few training examples, while other coordinates receive gradients all the time. Ordinary SGD treats all coordinates with the same nominal learning rate.

**Method:** Duchi et al. introduce a family of proximal and adaptive subgradient methods that uses *past gradient geometry* to adjust each coordinate's effective step.

**Main result:** Coordinate-wise and full-matrix adaptive proximal methods exploit the geometry of earlier gradients and enjoy data-dependent regret bounds in convex online optimization. They are especially effective when features are sparse or heterogeneous.

**Limitation:** Low *regret* in online convex prediction is not the same as finding a global minimum, a local minimum, or an $\epsilon$-SOSP of a non-convex neural-network loss.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. Regret compares you with the best fixed answer in hindsight

In **online learning** a player picks $x_t$, then a convex loss $f_t$ is revealed and the player pays $f_t(x_t)$. Nothing is assumed about where the losses come from. **Regret** $R_T$ is the total paid minus the total the best single fixed point would have paid. A method is useful when $R_T$ grows slower than $T$: then the *average* regret $R_T/T$ goes to zero. Example: $R_T = \sqrt{T}$ gives average regret $1/\sqrt{T}$, which is $0.01$ after $T = 10\,000$ rounds.

### 2. A subgradient is a slope that stays below the function

The paper allows losses with kinks, such as the hinge loss $[1 - y\langle z, x\rangle]_+$. At a kink there is no gradient, but there are **subgradients**: vectors $g$ with

$$f(y) \ge f(x) + \langle g,\, y - x\rangle \quad \text{for all } y .$$

Example: $f(x) = |x|$ at $x = 0$ accepts every $g \in [-1, 1]$. Where $f$ is differentiable, the only subgradient is $\nabla f(x)$. "Subgradient methods" use any such $g_t$ in place of the gradient, which is why the notes write $g_t$, not $\nabla f(x_t)$.

### 3. One learning rate starves rare features

Take two coordinates that each receive gradient $1$ when active. One is active in every round, the other in $1\%$ of rounds. After $T = 10\,000$ rounds the accumulated squares are $G = 10\,000$ and $G = 100$, so AdaGrad's effective steps are $\eta/100$ and $\eta/10$: the rare feature moves ten times further per occurrence. Plain SGD gives both the same $\eta$, so a rate small enough for the busy coordinate leaves the rare one barely trained.

### 4. Sparse gradients make the adaptive bound much smaller

The notes' bound sums, over coordinates, the root of each coordinate's accumulated squared gradients. The paper's §1.3 example lets feature $j$ appear with probability $p_j = \min\{1, c\,j^{-\alpha}\}$. With $c = 1$ and $\alpha = 2$, the expected sum is at most $\sqrt{T}\sum_j \sqrt{p_j} = \sqrt{T}\sum_j 1/j$, a harmonic number:

$$d = 10: \ 2.93 \ \text{vs}\ \sqrt{d} = 3.16; \qquad d = 10^4: \ 9.79 \ \text{vs}\ \sqrt{d} = 100 .$$

The $\sqrt{d}$ column is the factor of a non-adaptive bound of order $\sqrt{dT}$. The adaptive term grows like $\log d$. (The full comparison also involves the domain constants; the paper does it for the box $\|x\|_\infty \le 1$.)

### 5. Online regret becomes a stochastic-optimization guarantee only under convexity

If the losses $f_t$ are independent samples of one convex objective $F$, the average of the iterates $\bar{x}_T$ satisfies $\mathbb{E}F(\bar{x}_T) - \min F \le \mathbb{E}R_T/T$. This **online-to-batch conversion** is why the paper's title names stochastic optimization too. Both steps use convexity: Jensen's inequality for the average, and comparison with one global minimizer. Neither holds for a non-convex network loss, which is the notes' warning in §6.

:::key Putting the five ideas together
The paper works in the online convex game measured by regret (idea 1), with subgradients so that kinked losses are allowed (idea 2). Scaling each coordinate by its own gradient history stops rare features from being starved (idea 3), and on sparse data the resulting regret bound grows like $\log d$ where the non-adaptive one grows like $\sqrt{d}$ (idea 4). Convexity converts this into a stochastic-optimization rate, but only for convex losses (idea 5). The notes below derive the update and state the regret result.
:::
:::

:::divider Study notes
:::

## 1. Problem: one learning rate is not right for every feature

In sparse learning, a useful feature might appear in only a few training examples, while other coordinates receive gradients all the time. Ordinary SGD treats all coordinates with the same nominal learning rate. Duchi et al. introduce a family of proximal and adaptive subgradient methods that uses *past gradient geometry* to adjust each coordinate's effective step. Their central theoretical framework is online *convex* optimization and regret, not non-convex saddle-point escape.

## 2. First principles: coordinate-wise scaling

For a gradient sequence $g_t \in \mathbb{R}^d$, accumulate squared gradients for coordinate $j$:

$$G_{t,j} = \sum_{s=1}^{t} g_{s,j}^2.$$

The familiar diagonal AdaGrad update (ignoring optional projection or composite regularization) is

$$\boxed{x_{t+1,j} = x_{t,j} - \frac{\eta\, g_{t,j}}{\sqrt{G_{t,j}} + \delta}, \quad \delta > 0.}$$

A feature with repeated large gradients accumulates a large denominator and receives smaller steps. If a coordinate has received few gradients, its effective step stays comparatively large. The $\delta$ is a numerical stabilizer; the paper develops adaptive *proximal* updates more generally, including full-matrix and diagonal choices.

## 3. Where does this update come from?

A quadratic regularizer determines what "distance" the optimizer considers costly. At iteration $t$, imagine minimizing a linear approximation to the current loss, while penalizing the move in an adaptive metric $H_t \succ 0$:

$$x_{t+1} = \arg\min_x \left\{ g_t^\top(x - x_t) + \frac{1}{2\eta}(x - x_t)^\top H_t (x - x_t) \right\}.$$

Differentiate this *teaching proximal surrogate* with respect to $x$:

$$g_t + \frac{1}{\eta}H_t(x_{t+1} - x_t) = 0 \quad\Longrightarrow\quad x_{t+1} = x_t - \eta H_t^{-1} g_t.$$

Choosing $H_t$ diagonal, with entries related to $\sqrt{\sum_{s\le t} g_{s,j}^2}$, gives the coordinate-wise AdaGrad intuition. For constrained or regularized problems, replace the explicit update by the appropriate proximal step.

## 4. The paper's formal contribution: adaptive regret

In online convex learning, the player selects $x_t$ and suffers $f_t(x_t)$. Compare cumulative loss with the best fixed feasible point chosen in hindsight:

$$R_T = \sum_{t=1}^{T} f_t(x_t) - \min_{x\in\mathcal{X}} \sum_{t=1}^{T} f_t(x).$$

The authors give regret bounds that adapt to observed gradient sequences and compete with favorable hindsight choices of proximal geometry. An illustrative diagonal form includes a term of order

$$\boxed{R_T \;\lesssim\; D_\infty \sum_{j=1}^{d} \sqrt{\sum_{t=1}^{T} g_{t,j}^2}}$$

with problem-specific domain constants and tuning, rather than simply $O(\sqrt{dT})$ in a worst-case isotropic bound. This displayed expression is a schematic consequence under appropriate bounded-domain assumptions, not a verbatim complete theorem with all constants.

:::key Main result
Coordinate-wise and full-matrix adaptive proximal methods exploit the geometry of earlier gradients and enjoy data-dependent regret bounds in convex online optimization. They are especially effective when features are sparse or heterogeneous.
:::

## 5. Distinguish AdaGrad from Adam and saddle escape

AdaGrad's accumulated second moments $G_t$ grow without forgetting; Adam (Ref. [51]) instead uses exponentially moving first and second moments and bias corrections. AdaGrad's 2011 regret guarantee does *not* certify avoidance of strict saddles in deep non-convex loss landscapes. Such avoidance requires separate assumptions and analysis (our Ref. [66]).

## 6. Assumptions, limitations and survey relevance

The core proofs use convex losses, bounded feasible sets or suitable regularization, and controlled subgradients. Full-matrix adaptivity can be expensive in large dimensions. The algorithm can shrink effective steps excessively after a long history of large gradients. Relevant to **RQ2** (optimizer dynamics) and **RQ4** (convex guarantee versus non-convex applications).

:::warn Do not transfer guarantees
Low *regret* in online convex prediction is not the same as finding a global minimum, a local minimum, or an $\epsilon$-SOSP of a non-convex neural-network loss.
:::

## 7. Our notes

AdaGrad is easiest to understand as a change in the geometry of an update: coordinates with many big gradients are treated differently from rarely active ones. The interesting proof is about data-adaptive regret, not about avoiding saddles. For our paper, it motivates why the optimizer itself changes training dynamics, but we must use later results for non-convex guarantees. **Original source:** [https://jmlr.org/papers/v12/duchi11a.html](https://jmlr.org/papers/v12/duchi11a.html). **Source check:** journal manuscript.

> **Primary source:** Duchi, Hazan & Singer (2011), *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*, Journal of Machine Learning Research 12, 2121–2159. [jmlr.org/papers/v12/duchi11a.html](https://jmlr.org/papers/v12/duchi11a.html).
