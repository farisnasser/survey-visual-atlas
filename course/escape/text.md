## 1. Near a saddle, every step multiplies the way out

Stop 5 ended with gradient descent caught at a saddle when it started on exactly the right line, and lingering for thousands of steps near a flat one. This stop asks whether it is ever truly stuck, and what the field built to get out faster.

Start with the quadratic model at a strict saddle, in its eigenvector coordinates (stop 4, section 4). One direction curves up with $\lambda_1 > 0$; one curves down with $\lambda_2 = -\gamma < 0$, where $\gamma$ (gamma) is the strength of the negative curvature. A GD step multiplies each coordinate by $1 - \eta\lambda_i$:

$$u_1 \leftarrow (1 - \eta \lambda_1)\, u_1 \quad \text{(shrinks)},$$

$$u_2 \leftarrow (1 + \eta \gamma)\, u_2 \quad \text{(grows)}.$$

The coordinate $u_2$ along the [[negative-curvature|negative-curvature direction]] is multiplied by a number larger than 1 at every step. If it starts at a tiny offset $\delta$, after $t$ steps it is $\delta (1 + \eta\gamma)^t$, which reaches a fixed size $c$ after

$$t \;=\; \frac{\ln(c/\delta)}{\ln(1 + \eta\gamma)} \;\approx\; \frac{\ln(c/\delta)}{\eta\gamma} \quad \text{steps}.$$

Two lessons. The escape time grows only like the **logarithm** of $1/\delta$: every factor of 100 in $\delta$ costs the same fixed number of extra steps. And it grows like $1/\gamma$: weak negative curvature is slow.

On $f(x, y) = \tfrac12 x^2 + \tfrac14 y^4 - \tfrac12 y^2$ (the saddle of stop 5 with curvatures $1$ and $-1$) with $\eta = 0.1$, starting at $(1, \delta)$, GD needs $43$, $91$, $139$, $188$ and $284$ steps to reach $\lvert y \rvert > 0.5$ for $\delta = 10^{-2}, 10^{-4}, 10^{-6}, 10^{-8}, 10^{-12}$: about $\ln 100 / \ln 1.1 = 48.3$ more steps per factor of 100. Weakening the curvature to $-\gamma$ (the function $\tfrac12 x^2 - \tfrac{\gamma}{2}y^2 + \tfrac14 y^4$, from $y = 10^{-3}$ to $\lvert y \rvert > \sqrt{\gamma}/2$) gives $67$, $524$ and $4{,}058$ steps for $\gamma = 1, 0.1, 0.01$.

:::viz escape
:::

## 2. Is gradient descent ever stuck? The stable set

Only a start with $\delta = 0$ exactly stays forever. The set of starting points from which gradient descent converges to a given critical point is that point's **[[stable-set|stable set]]**. For the saddle above it is the line $y = 0$. In general it is a curved, lower-dimensional surface through the saddle, tangent to the directions that do not grow. A line in the plane, or any lower-dimensional surface in $\mathbb{R}^d$, has zero volume: it has **[[measure-zero|measure zero]]**. A starting point drawn at random from any distribution with a density (Gaussian, uniform on a box, and so on) lands in a measure-zero set with probability zero.

:::theorem Lee, Simchowitz, Jordan & Recht 2016, Theorem 4
Let $f$ be twice continuously differentiable with an $L$-Lipschitz gradient, let $x^\star$ be a strict saddle, and run gradient descent with a constant step $0 < \eta < 1/L$ from a random start with a density. Then the probability that the iterates converge to $x^\star$ is zero.
:::

So gradient descent avoids strict saddles **[[almost-surely|almost surely]]**. Two ingredients make the proof work. The step $x \mapsto x - \eta \nabla f(x)$ is a smooth map with a smooth inverse when $\eta < 1/L$ (their Proposition 8), so it never folds a set of positive volume onto a set of zero volume. And the center-stable manifold theorem from dynamical systems ([Shub 1987](paper.html?id=shub1987global)) says that near a strict saddle the points that converge to it lie on a manifold of dimension strictly less than $d$. With countably many saddles, all strict, the same holds for all of them at once (Corollary 9), and if the iterates converge at all, their limit is a local minimum almost surely (Theorem 11).

In the visual, every start in the box is tinted by the minimum it reaches. The saddle's stable set is the thin amber curve between the two tints; of $2{,}000$ random starts, none end at the saddle.

:::viz stable
:::

:::warn What the theorem does not say
It is **asymptotic**: it says where GD ends up, not how long it takes, and section 1 already showed that a start close to the stable set can take a long time to leave. It needs the strict-saddle property (degenerate saddles such as stop 5's monkey saddle are not covered) and a step below $1/L$. Later work extends it: [Panageas, Piliouras & Wang 2019](paper.html?id=panageas2019first) allow step sizes that decay no faster than order $1/k$ (their corrected arXiv v3 needs an invertible Hessian at the saddle), and [Mertikopoulos et al. 2020](paper.html?id=mertikopoulos2020almost) prove that SGD with suitable noise avoids strict saddles with probability one.
:::

## 3. But it can take exponentially long

"Almost surely escapes" leaves room for "escapes after an absurdly long time". [Du et al. 2017](paper.html?id=du2017gradient) show that this really happens. Their Theorem 4.1 builds a smooth function on $\mathbb{R}^d$, with smoothness constants polynomial in $d$, on which gradient descent from a random start in $[-1, 1]^d$ stays far from every local minimum for $e^{\Omega(d)}$ steps ($\Omega(\cdot)$ means "at least a constant times"), while perturbed gradient descent (next section) reaches one in time polynomial in $d$ and $1/\varepsilon$.

The construction is a chain of $d$ saddles that GD visits one after another, and each wait is longer than the last by a constant factor (about $(L + \gamma)/\gamma$ in their analysis), so the waits multiply up to an exponential. A strict saddle cannot trap gradient descent forever, but a sequence of them can delay it for longer than any practical training run.

## 4. Perturbed gradient descent

The fix is to make sure that GD never sits close to a stable set: add a little randomness exactly when the gradient becomes small.

:::idea Perturbed gradient descent (Jin, Ge, Netrapalli, Kakade & Jordan 2017, Algorithm 2)
Run gradient descent with step $\eta = c/L$. Whenever $\lVert \nabla f(x_t) \rVert$ falls below a threshold and there has been no kick for a while, add a random vector drawn uniformly from a ball of small radius $r$. If, some fixed number of steps after a kick, $f$ has not dropped by a set amount, stop and return the point before the kick: it is an approximate second-order stationary point.
:::

:::theorem Jin et al. 2017, Theorem 3
If $f$ is $L$-smooth and $\rho$-Hessian-Lipschitz, then with probability at least $1 - \delta$, **[[perturbed-gd|perturbed gradient descent]]** outputs an $\varepsilon$-SOSP (stop 5) within

$$O\!\left( \frac{L\, \Delta_f}{\varepsilon^2} \log^4 \frac{d\, L\, \Delta_f}{\varepsilon^2 \delta} \right)$$

iterations, where $\Delta_f = f(x_0) - f^\star$.
:::

This is stop 3's $O(\varepsilon^{-2})$ rate for first-order points, now for **second-order** points, at the price of a factor polylogarithmic in the dimension $d$. Papers write such bounds as $\tilde O(\varepsilon^{-2})$: the tilde ([[soft-o|soft-O]]) hides logarithmic factors. "With probability at least $1 - \delta$" is the meaning of **[[whp|with high probability]]**: the user picks a small failure probability $\delta$, and it enters the bound only through a logarithm.

Why a random kick works (their Section 5): the kicks that fail to escape are those that land too close to the stable set. Near the saddle that region is a thin band (a "pancake" in three dimensions) around the stable directions, and they show that two points separated along the negative-curvature direction by a small distance cannot both fail. So the failing band is thin, and a uniform kick lands in it with small probability.

In the visual both balls start on the stable line. Plain GD stops at the saddle with $f = 0$. Perturbed GD with $r = 0.1$ is kicked at step $33$, escapes to the minimum, is kicked again there at step $114$, sees no further decrease, and stops at step $194$, returning the minimum with $f = -2$.

:::viz pgd
:::

The version in [Jin et al.'s 2021 journal paper](paper.html?id=jin2021nonconvex) adds Gaussian noise at every step instead of only when the gradient is small, with the same $\tilde O(\varepsilon^{-2})$ rate. Adding momentum gives $\tilde O(\varepsilon^{-7/4})$ ([Jin, Netrapalli & Jordan 2018](paper.html?id=jin2018accelerated); stop 7).

## 5. Finding the way down with gradients only

Perturbed GD finds negative curvature by accident: gradient descent near a saddle is itself a power iteration. Since $\nabla f(x + v) - \nabla f(x) \approx \nabla^2 f(x)\, v$ for small $v$ (stop 4, section 7), the iteration

$$v \;\leftarrow\; v - \eta\,\big(\nabla f(x + v) - \nabla f(x)\big)$$

is approximately $v \leftarrow (I - \eta H)\, v$ with $H = \nabla^2 f(x)$: it multiplies $v$ by the matrix $I - \eta H$, whose eigenvalues are $1 - \eta\lambda_i$. The largest of them, $1 + \eta\gamma$, belongs to the most negative eigenvalue $-\gamma$, so after enough steps $v$ points along the negative-curvature direction. This is the **[[power-method|power method]]**, run with nothing but gradients. Doing it on purpose, at a point where the gradient is small, turns "escape by luck" into "find the way down, or certify there is none".

- **NEON** ([Xu, Jin & Yang 2018](paper.html?id=xu2018first)) runs exactly this iteration from a random start on a small sphere around $x$; an accelerated variant, NEON+, needs fewer iterations, in the way Lanczos beats the power method.
- **NEON2** ([Allen-Zhu & Li 2018](paper.html?id=allenzhu2018neon2), Theorem 1) does it with *stochastic* gradients (the same random example at both points) and, with high probability, either returns a unit $v$ with $v^\top \nabla^2 f(x)\, v \le -\delta/2$ or certifies $\nabla^2 f(x) \succeq -\delta I$, after $\tilde O(L^2/\delta^2)$ stochastic gradients. Plugged into first-order methods that find $\varepsilon$-FOSPs (SGD, GD, SVRG and others), it turns them into methods that find $\varepsilon$-SOSPs.

In $d = 30$ dimensions with one negative eigenvalue $-\gamma$, the number of iterations until $v^\top H v \le -\gamma/2$ is $5$, $24$, $195$ and $1{,}939$ for $\gamma = 1, 0.1, 0.01, 0.001$: it grows like $1/\gamma$, as the factor $1 + \gamma$ per step predicts.

:::viz power
:::

## 6. Using the Hessian: Newton, saddle-free Newton, cubic regularization

If curvature is the problem, why not use it directly? **[[newton|Newton's method]]** jumps to the critical point of the quadratic model:

$$x_{t+1} \;=\; x_t - \big[\nabla^2 f(x_t)\big]^{-1} \nabla f(x_t).$$

In the eigenvector basis it divides the gradient component along $v_i$ by $\lambda_i$. Near a minimum (all $\lambda_i > 0$) this is excellent: it rescales every direction to the same speed, so the condition number of stop 4 disappears. Near a saddle it is a disaster. Along a negative-curvature direction, dividing by $\lambda_i < 0$ flips the sign of the step, so Newton moves *toward* the saddle along exactly the direction that should lead away. [Dauphin et al. 2014](paper.html?id=dauphin2014identifying) make this point (their section 4): on $5x^2 - y^2$ one exact Newton step from any point lands on the saddle; from $(0.9, 0.12)$ it goes to $(0, 0)$. **Newton's method is attracted to saddles** as much as to minima.

:::idea Saddle-free Newton (Dauphin et al. 2014)
**[[saddle-free-newton|Saddle-free Newton]]** divides by $\lvert \lambda_i \rvert$ instead of $\lambda_i$: $\Delta x = -\lvert H \rvert^{-1} \nabla f$, where $\lvert H \rvert$ has the same eigenvectors as $H$ and eigenvalues $\lvert \lambda_i \rvert$. Directions of positive curvature behave as in Newton; along negative curvature the step now goes downhill, and it is fastest where the curvature is weakest. On $5x^2 - y^2$ the step from $(0.9, 0.12)$ goes to $(0, 0.24)$, away from the saddle. Their large-scale version works in a low-dimensional Krylov subspace with damping; the method is supported by experiments, not by a convergence theorem.
:::

:::idea Cubic regularization (Nesterov & Polyak 2006)
Trust the quadratic model only nearby. Each step minimizes the model plus a cubic penalty on the step length,

$$x_{t+1} = x_t + \operatorname*{arg\,min}_{\Delta}\, m_t(\Delta),$$

where $m_t(\Delta) = \nabla f(x_t) \cdot \Delta + \tfrac12 \Delta^\top \nabla^2 f(x_t)\, \Delta + \tfrac{M}{6} \lVert \Delta \rVert^3$, with $M$ at least the Hessian-Lipschitz constant $\rho$. The penalty makes the model an upper bound on $f$ (stop 5, section 6), so every step decreases $f$, the descent lemma one level up. At a saddle the best step leaves along the negative-curvature direction, even when the gradient is exactly zero.
:::

[[cubic-reg|Cubic regularization]] comes with the guarantees that Newton lacks. Nesterov and Polyak prove that it cannot be attracted to a non-degenerate saddle or maximum, even when started on one (Lemma 6), that its limit points satisfy $\nabla f = 0$ and $\nabla^2 f \succeq 0$ (Theorem 2), and that the smallest gradient after $k$ steps is $O(k^{-2/3})$ (Theorem 1), versus $O(k^{-1/2})$ for gradient descent. That translates into $O(\varepsilon^{-3/2})$ iterations to an $\varepsilon$-SOSP. Each iteration must solve the cubic subproblem, which needs the Hessian or many Hessian-vector products; later papers in the Atlas solve it approximately ([Carmon & Duchi 2019](paper.html?id=carmon2019gradient) by gradient descent on the model, [Tripuraneni et al. 2018](paper.html?id=tripuraneni2018stochastic) with stochastic Hessian-vector products).

In the visual, from $(1.2, 0.03)$ on $f = \tfrac12 x^2 - \tfrac12 y^2 + \tfrac14 y^4 + 0.3xy^2$: Newton converges to the saddle in $4$ steps; saddle-free Newton reaches a minimum in $7$, cubic regularization ($M = 10$) in $12$, and gradient descent ($\eta = 1/8$) in $167$. Start exactly on the stable line, $y_0 = 0$: Newton and gradient descent both end at the saddle, and cubic regularization still leaves, reaching a minimum in $13$ steps.

:::viz newton
:::

## 7. The scoreboard

What the Atlas papers prove about reaching second-order stationary points, with the oracle each method uses (stop 8 introduces stochastic gradients). Rates count iterations or oracle calls; $\tilde O$ hides logarithmic factors.

| Method | Uses | Guarantee | Paper |
|---|---|---|---|
| gradient descent | gradients | $\varepsilon$-FOSP in $O(\varepsilon^{-2})$ | stop 3 |
| gradient descent | gradients | avoids strict saddles almost surely; no time bound | Lee et al. 2016, Thm 4 |
| gradient descent | gradients | can need $e^{\Omega(d)}$ steps | Du et al. 2017, Thm 4.1 |
| perturbed GD | gradients | $\varepsilon$-SOSP in $\tilde O(\varepsilon^{-2})$, w.h.p. | Jin et al. 2017, Thm 3 |
| perturbed accelerated GD | gradients | $\varepsilon$-SOSP in $\tilde O(\varepsilon^{-7/4})$, w.h.p. | Jin et al. 2018, Thm 3 |
| cubic regularization | Hessians | $\varepsilon$-SOSP in $O(\varepsilon^{-3/2})$ iterations | Nesterov & Polyak 2006, Thm 1 |
| perturbed SGD | stochastic gradients | $\varepsilon$-SOSP in $\tilde O(\varepsilon^{-4})$ | Jin et al. 2021 |
| stochastic cubic regularization | stochastic gradients and HVPs | $\varepsilon$-SOSP in $\tilde O(\varepsilon^{-3.5})$ | Tripuraneni et al. 2018 |
| SPIDER-SFO+ | stochastic gradients | $(\varepsilon, O(\sqrt{\varepsilon}))$-SOSP in $\tilde O(\varepsilon^{-3})$ | Fang et al. 2018 |

The pattern: second-order points cost about as much as first-order ones, up to logarithms, once the method injects noise or searches for negative curvature. Whether these rates can be beaten is the subject of stop 12.

:::atlas Where this shows up in the Atlas
- [Lee et al. 2016](paper.html?id=lee2016gradient), [Shub 1987](paper.html?id=shub1987global), [Panageas et al. 2019](paper.html?id=panageas2019first), [Mertikopoulos et al. 2020](paper.html?id=mertikopoulos2020almost): the stable set and almost-sure avoidance (section 2).
- [Du et al. 2017](paper.html?id=du2017gradient): exponential escape time for plain GD (section 3).
- [Jin et al. 2017](paper.html?id=jin2017escape), [Jin et al. 2021](paper.html?id=jin2021nonconvex), [Jin et al. 2018](paper.html?id=jin2018accelerated): perturbed and accelerated escape (section 4, stop 7).
- [Xu et al. 2018](paper.html?id=xu2018first), [Allen-Zhu & Li 2018](paper.html?id=allenzhu2018neon2), [Zhang & Li 2021](paper.html?id=zhang2021escape), [Reddi et al. 2018](paper.html?id=reddi2018generic): negative curvature from gradients and frameworks that alternate gradient steps with curvature steps (section 5).
- [Dauphin et al. 2014](paper.html?id=dauphin2014identifying), [Nesterov & Polyak 2006](paper.html?id=nesterov2006cubic), [Carmon & Duchi 2019](paper.html?id=carmon2019gradient), [Tripuraneni et al. 2018](paper.html?id=tripuraneni2018stochastic), [Wang et al. 2020](paper.html?id=wang2020cubic): second-order methods (section 6).
:::

:::problem The next problem
Escaping saddles is, in theory, solved: perturbed gradient descent reaches an $\varepsilon$-SOSP almost as fast as plain GD reaches a small gradient. But all these rates still carry stop 4's condition number, and in a long narrow valley gradient descent still zig-zags. **Can the optimizer remember its past steps, so the zig-zags cancel and the slow direction builds up speed?**
:::
