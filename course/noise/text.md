## 1. A full gradient costs a pass over the data

Stop 7 ended on the cost of a single step. Recall from stop 1 that the training loss is an average over the $n$ training examples:

$$f(x) \;=\; \frac{1}{n} \sum_{i=1}^{n} f_i(x),$$

where $f_i(x)$ is the loss on example $i$ alone (in stop 1's notation, $f_i(x) = \ell(\hat y_i, y_i)$ with parameters $x$). A loss of this shape is called a **finite sum**. Its gradient is the average of the $n$ example gradients, $\nabla f(x) = \frac1n \sum_i \nabla f_i(x)$, so one exact gradient costs $n$ backpropagations: one full pass over the data. Every method of stops 3 to 7 pays that price at every step. For a few dozen examples that is nothing; for millions of images it means one step per pass, and training needs many thousands of steps.

The running example of this stop is small enough to compute exactly: a line $w\,x + b$ fitted by squared error to $n = 50$ points (the same kind of landscape as stop 1). Its minimum is at $w^\star \approx 0.800$, $b^\star \approx -0.488$, with $f^\star \approx 0.180$.

## 2. One random example is an unbiased guess

Instead of averaging all $n$ example gradients, pick one example $i$ uniformly at random and use its gradient alone, $g = \nabla f_i(x)$. This guess is wrong at every single step, but it is right **on average**: each example has probability $1/n$, so

$$\mathbb{E}[g] \;=\; \sum_{i=1}^{n} \frac1n\, \nabla f_i(x) \;=\; \nabla f(x).$$

$\mathbb{E}$ is the **expected value**, the average over the random choice. An estimate whose expected value equals the true quantity is **[[unbiased]]**. How far a single guess typically lands from the truth is measured by its **[[variance]]**,

$$\sigma^2(x) \;=\; \mathbb{E}\,\lVert g - \nabla f(x) \rVert^2 \;=\; \frac1n \sum_{i=1}^n \lVert \nabla f_i(x) - \nabla f(x) \rVert^2 ,$$

the average squared distance from the mean ($\sigma$, sigma, is the typical size of the error). Papers usually *assume* a bound $\sigma^2(x) \le \sigma^2$ for all $x$; this is the "bounded variance" assumption.

The noise can be turned down by averaging. A **[[minibatch]]** is a random set of $B$ examples, and its gradient $g_B = \frac1B \sum_{i \in \text{batch}} \nabla f_i(x)$ is still unbiased. If the $B$ examples are drawn independently, the errors of the $B$ terms are independent and their average has variance

$$\mathbb{E}\,\lVert g_B - \nabla f(x) \rVert^2 \;=\; \frac{\sigma^2(x)}{B}.$$

(Drawing without replacement gives the slightly smaller $\frac{n - B}{n - 1}\cdot\frac{\sigma^2(x)}{B}$, which is $0$ at $B = n$.) Ten times the batch buys ten times less variance, at ten times the cost: the error's typical size only shrinks like $1/\sqrt B$.

At the point $(1.8, -1.2)$ of the line fit, $\nabla f = (7.93, 2.58)$ and $\sigma^2 = 75.25$. With $B = 5$ the exact variance is $15.05$; the 400 minibatches in the visual measure $15.02$, and their average $(7.77, 2.56)$ sits next to the true gradient.

:::viz cloud
:::

## 3. Stochastic gradient descent

:::def Stochastic gradient descent
**[[sgd|Stochastic gradient descent]]** (SGD) is gradient descent with the exact gradient replaced by an unbiased random estimate $g_t$, usually a minibatch gradient:

$$x_{t+1} = x_t - \eta_t\, g_t, \qquad \mathbb{E}[g_t \mid x_t] = \nabla f(x_t).$$

The step size may change with $t$ (section 5). "$\mid x_t$" means: averaged over the new random batch, with $x_t$ held fixed.
:::

Each step now costs $B$ example gradients instead of $n$. To compare methods fairly, papers count **example-gradient evaluations** (an *incremental first-order oracle*, IFO, call each), not steps. One full pass over the data is an **[[epoch]]**: $n/B$ SGD steps, or one GD step. In the line fit with $B = 5$, one epoch is 10 SGD steps.

In deep learning the minibatch is drawn by shuffling the data and walking through it, and "SGD" almost always means minibatch SGD, often with momentum (stop 7).

## 4. The noise floor

What does the noise do to the iterates? Take one direction with curvature $a$ and pretend the error of the gradient is independent noise $\xi_t$ with variance $s^2$. One step gives

$$x_{t+1} = (1 - \eta a)\, x_t \;-\; \eta\, \xi_t .$$

The first term pulls toward the minimum, exactly as in stop 3. The second term pushes randomly, every step, forever. The two balance when the variance $V$ of $x_t$ stops changing: $V = (1 - \eta a)^2 V + \eta^2 s^2$, so

$$V = \frac{\eta\, s^2}{a\,(2 - \eta a)}, \qquad \mathbb{E}[f] - f^\star = \frac{a}{2} V = \frac{\eta\, s^2}{2\,(2 - \eta a)} \;\approx\; \frac{\eta\, s^2}{4}.$$

With a constant step SGD does **not converge to the minimum**. It converges to a cloud around it, and the cloud's height, the **[[noise-floor|noise floor]]**, is proportional to $\eta$. Summing over directions and using minibatches gives about $\eta\,\sigma_\star^2/(4B)$ for small $\eta$, where $\sigma_\star^2$ is the gradient variance at the minimum. In the line fit $\sigma_\star^2 = 4.81$. With $\eta = 0.005$ and $B = 1$ the measured floor is $0.0062$ against the formula's $0.0060$. With $\eta = 0.02$ it is $0.031$ against $0.024$: the formula is a small-step approximation, and real minibatch noise also changes with $x$.

:::viz floor
:::

The same trade-off holds without convexity. [Bottou, Curtis & Nocedal 2018](paper.html?id=bottou2018optimization) (Theorem 4.8) show that SGD with a fixed step $\bar\alpha$ on a smooth non-convex function keeps the average squared gradient below a term proportional to $\bar\alpha$, the floor, plus a term that shrinks like $1/(K\bar\alpha)$ over $K$ steps. A larger step reaches the floor sooner and the floor is higher.

## 5. Shrinking steps: the Robbins–Monro conditions

The fix is to shrink the step over time: large steps early to travel, small steps late to average the noise away. How fast? Two requirements pull in opposite directions:

:::def Robbins–Monro conditions
$$\sum_{t} \eta_t = \infty, \qquad \sum_t \eta_t^2 < \infty .$$

The first says the steps never run out of distance, so the method can reach a minimum from anywhere. The second says the total noise injected, which scales with $\sum \eta_t^2$, stays finite. $\eta_t = c/t$ satisfies both; a constant step fails the second; $\eta_t = c/t^2$ fails the first.
:::

These are the **[[robbins-monro|Robbins–Monro conditions]]**, named after the 1951 paper that started stochastic approximation. [Robbins & Monro](paper.html?id=robbins1951stochastic) solved $M(x) = \alpha$ when every evaluation of $M$ is noisy, with steps $a_n$ such that $\sum a_n^2 < \infty$ plus a condition that implies $\sum a_n = \infty$, and proved that $\mathbb{E}(x_n - \theta)^2 \to 0$, so $x_n$ converges in probability to the root $\theta$ (Theorems 1 and 2; almost-sure convergence came in later work).

A one-line example shows all three behaviours. Minimize $f(x) = x^2/2$ with noisy gradients $x_t + \xi_t$, $\xi_t$ standard normal, from $x_0 = 2$. The mean square obeys $\mathbb{E}x_{t+1}^2 = (1 - \eta_t)^2\, \mathbb{E}x_t^2 + \eta_t^2$ exactly.

- **Constant $\eta = 0.1$**: stalls at $\eta/(2 - \eta) \approx 0.0526$.
- **$\eta_t = 1/(t + 2)$**: then $x_t = \big(x_0 - \sum_{s<t} \xi_s\big)/(t + 1)$, the running average of the noise, so $\mathbb{E}x_t^2 = (x_0^2 + t)/(t + 1)^2 \approx 1/t$: $0.0102$ at $t = 100$ and $0.0001$ at $t = 10^4$.
- **$\eta_t = 1/(t + 2)^2$**: the steps add up to only $\pi^2/6 - 1 \approx 0.645$, and the deterministic part shrinks by $\prod_{k \ge 2}(1 - 1/k^2) = \tfrac12$ at most. The method stops short: $\mathbb{E}x_t^2 \approx 1.04$ forever.

:::viz schedules
:::

In practice, deep-learning schedules (step decay, cosine decay) do not satisfy these conditions literally; they run a fixed number of steps and decay the step toward a small value. Warmup, which *increases* the step at the start, is a different story (stop 10).

## 6. How many gradients? The $\varepsilon^{-4}$ rate

Stop 3 counted GD's steps to an [[eps-stationary|ε-stationary point]]: $O(\varepsilon^{-2})$ steps. On a finite sum each step costs $n$ example gradients, so GD needs about $n\,\varepsilon^{-2}$ of them. SGD's count does not depend on $n$ at all:

:::theorem Ghadimi & Lan 2013, Corollary 2.2
Let $f$ be $L$-smooth and bounded below, and let the stochastic gradients be unbiased with variance at most $\sigma^2$. Run SGD for $N$ steps with step $\gamma = \min\{1/L,\; \tilde D/(\sigma\sqrt N)\}$ and output a randomly chosen iterate $x_R$. Then

$$\tfrac{1}{L}\, \mathbb{E}\,\lVert \nabla f(x_R) \rVert^2 \;\le\; \frac{L D_f^2}{N} + \Big(\tilde D + \frac{D_f^2}{\tilde D}\Big) \frac{\sigma}{\sqrt N}, \qquad D_f = \sqrt{2\,(f(x_1) - f^\star)/L}.$$

The noise term falls only like $1/\sqrt N$. Asking for $\mathbb{E}\lVert \nabla f \rVert^2 \le \varepsilon^2$ therefore needs $N = O(\varepsilon^{-4})$ stochastic gradients (of order $L\Delta\sigma^2/\varepsilon^4$ when the noise dominates, with $\Delta = f(x_1) - f^\star$).
:::

([Ghadimi & Lan](paper.html?id=ghadimi2013stochastic) write $\gamma$ for the step and $N$ for the number of steps.) Two remarks. The output is a **random** iterate, a device that turns a bound on the average gradient into a bound on one point. And the guarantee is first-order only: $x_R$ may be a saddle (stop 5).

This rate cannot be improved in general. [Arjevani et al. 2023](paper.html?id=arjevani2023lower) (Theorem 1) prove that every method that only sees such noisy gradients needs $\Omega(\Delta L \sigma^2 \varepsilon^{-4})$ of them on some function, so SGD is optimal under these assumptions (stop 12).

Orders of growth, for $n = 10^6$ examples and $\varepsilon = 10^{-2}$, constants ignored:

| method | example gradients | here |
|---|---|---|
| GD | $n\,\varepsilon^{-2}$ | $10^{10}$ |
| SGD | $\varepsilon^{-4}$ | $10^{8}$ |
| SVRG | $n + n^{2/3}\varepsilon^{-2}$ | $1.01 \times 10^{8}$ |
| SPIDER / SpiderBoost, finite sum | $n + n^{1/2}\varepsilon^{-2}$ | $1.1 \times 10^{7}$ |
| SPIDER / SpiderBoost, online | $\varepsilon^{-3}$ | $10^{6}$ |

The last three rows are the subject of the next section.

## 7. Variance reduction

SGD's floor exists because its noise does not shrink as the iterates approach the minimum: at $x^\star$ the full gradient is zero but the example gradients are not ($\sigma_\star^2 = 4.81$ in the line fit). **[[variance-reduction|Variance reduction]]** builds estimates whose noise does shrink.

:::idea SVRG: correct the cheap gradient with an old full one
Every so often, at a **snapshot** point $\tilde x$, pay for one full gradient $\nabla f(\tilde x)$. In between, use

$$g_t \;=\; \nabla f_i(x_t) - \nabla f_i(\tilde x) + \nabla f(\tilde x).$$

It is unbiased: the last two terms average to zero over $i$. And when $x_t$ and $\tilde x$ are both near the minimum, $\nabla f_i(x_t) - \nabla f_i(\tilde x)$ is small, so $g_t$ is nearly exact. Each inner step costs two example gradients.
:::

For non-convex finite sums, [Reddi et al. 2016](paper.html?id=reddi2016stochastic) prove that SVRG reaches $\mathbb{E}\lVert \nabla f \rVert^2 \le \varepsilon$ with $O(n + n^{2/3}/\varepsilon)$ example gradients (Corollary 4), that is $O(n + n^{2/3}\varepsilon^{-2})$ for $\lVert \nabla f \rVert \le \varepsilon$: better than GD's $n\varepsilon^{-2}$ by a factor $n^{1/3}$.

**SPIDER** ([Fang et al. 2018](paper.html?id=fang2018spider)) updates the estimate recursively instead, $v_t = \nabla f_S(x_t) - \nabla f_S(x_{t-1}) + v_{t-1}$ with the same minibatch $S$ at both points, and refreshes it with a large batch every $q$ steps. It reaches an ε-stationary point with $O(\min(n + n^{1/2}\varepsilon^{-2},\; \varepsilon^{-3}))$ stochastic gradients (Theorems 1 and 2), and Theorem 3 shows that the finite-sum count cannot be improved when $n$ is not too large ($n \le O(\varepsilon^{-4})$). **SpiderBoost** ([Wang et al. 2019](paper.html?id=wang2019spiderboost)) keeps the same estimator with a much larger constant step $1/(2L)$ and the same $O(n + \sqrt n\, \varepsilon^{-2})$ count (Theorem 1). The $\varepsilon^{-3}$ rates need a stronger assumption than SGD's: each example's gradient (not only the average) must be Lipschitz, or at least smooth in mean square. Under that assumption $\varepsilon^{-3}$ is also a lower bound ([Arjevani et al. 2023](paper.html?id=arjevani2023lower), Theorem 2).

In the visual, the line fit is a strongly convex finite sum, where SVRG converges linearly: after 60 epochs of gradient evaluations it is at the minimum to machine precision, GD with step $1/L$ is at $2.7 \times 10^{-3}$, and SGD with the same step as SVRG is still at its floor.

:::viz vr
:::

:::warn Variance reduction in deep learning
These rates are worst-case counts under assumptions that real networks only roughly meet, and they certify first-order stationary points only (SpiderBoost, for one, proves nothing about saddles; escape variants such as Ge et al. 2019 and Li 2019 add perturbations). Networks are mostly trained with SGD with momentum or with adaptive methods (stop 9); the variance-reduced methods here serve mainly as the benchmark for what stochastic gradients can achieve in theory.
:::

## 8. Noise near saddles: help, and its limits

Noise is not only a nuisance. Stop 6 showed that gradient descent started exactly on a saddle's stable set converges to the saddle, and that random perturbations fix this. SGD's own noise is a random perturbation at every step, so it often does the same job for free.

Take $f(x, y) = \tfrac12 x^2 + \tfrac14 y^4 - \tfrac12 y^2$, with a saddle at the origin (curvature $+1$ along $x$, $-1$ along $y$) and minima at $(0, \pm 1)$. Start at $(1.5, 0)$, on the line $y = 0$ that leads into the saddle. Make it a two-example finite sum, $f \pm s\,y$: each SGD gradient is the true one plus $\pm s$ along $y$. With $s = 0.3$ and step $0.1$, SGD leaves the saddle region ($\lvert y \rvert > 0.5$) after 23 steps in the default run. GD never does. But if the two examples are $f \pm s\,x$ instead, the noise only shakes $x$, the $y$-coordinate stays exactly $0$, and SGD converges to the saddle like GD.

:::viz escape
:::

That is the whole condition in miniature: noise helps only if it has a component along the direction of negative curvature.

:::theorem What is proved about SGD noise and saddles
- **Correlated negative curvature (CNC).** [Daneshmand et al. 2018](paper.html?id=daneshmand2018escaping) assume (Assumption 1) that the stochastic gradient has, at every point, variance at least $\gamma > 0$ along the most negative curvature direction. Under it, SGD with an occasional larger step reaches an $(\varepsilon, \varepsilon^{2/5})$-SOSP (Theorems 1 and 2) with no explicit dependence on the dimension $d$. Isotropic noise added by hand, by contrast, spreads its variance $r^2$ over all $d$ directions and puts only $r^2/d$ on any given one.
- **Plain SGD.** [Fang, Lin & Zhang 2019](paper.html?id=fang2019sharp) show that SGD, with a simple restart-and-average rule and under a "dispersive noise" assumption, finds an approximate SOSP with $\tilde O(\varepsilon^{-3.5})$ stochastic gradients.
:::

Real SGD noise is also not the well-behaved Gaussian of these pictures. [Gürbüzbalaban, Şimşekli & Zhu 2021](paper.html?id=gurbuzbalaban2021heavy) show that even for linear regression, SGD's iterates can have a **heavy-tailed** stationary distribution, with tail index $\alpha$ that rises with the batch size $b$ and falls with the step, the data variance and the dimension (Theorems 2 and 4). For Gaussian data they give a critical step $\eta_{\text{crit}} = 2b/\big(\sigma^2(d + b + 1)\big)$ (Proposition 5): above it, $\alpha < 2$ and the iterates have **infinite variance** (for $d = b = \sigma^2 = 1$, $\eta_{\text{crit}} = 2/3$). Such tails make occasional very long jumps, which can help leave a region, but they break the bounded-variance assumption that the rates of section 6 rest on.

:::atlas Where this shows up in the Atlas
- [Robbins & Monro 1951](paper.html?id=robbins1951stochastic): stochastic approximation and the step-size conditions (section 5).
- [Ghadimi & Lan 2013](paper.html?id=ghadimi2013stochastic): the $O(\varepsilon^{-4})$ SGD rate (section 6); [Arjevani et al. 2023](paper.html?id=arjevani2023lower): the matching lower bound.
- [Bottou, Curtis & Nocedal 2018](paper.html?id=bottou2018optimization): fixed versus diminishing steps, and noise-reduction methods.
- [Reddi et al. 2016](paper.html?id=reddi2016stochastic), [Fang et al. 2018](paper.html?id=fang2018spider), [Wang et al. 2019](paper.html?id=wang2019spiderboost), and the escape variants [Ge et al. 2019](paper.html?id=ge2019stabilized) and [Li 2019](paper.html?id=li2019ssrgd): variance reduction.
- [Daneshmand et al. 2018](paper.html?id=daneshmand2018escaping), [Fang et al. 2019](paper.html?id=fang2019sharp), [Gürbüzbalaban et al. 2021](paper.html?id=gurbuzbalaban2021heavy): what SGD's noise does near saddles (section 8).
- [Zhang et al. 2019](paper.html?id=zhang2019lookahead): Lookahead (stop 7) lowers the variance of SGD's iterates on a noisy quadratic.
:::

:::problem The next problem
Look again at the line fit: its curvature is $0.45$ along one direction of $(w, b)$ and $12.33$ along another, a condition number of $27.4$. Every method so far, with or without noise, uses **one step size for all coordinates**, so the step must respect the steepest direction while crawling along the flattest. **What if each parameter got its own step size?**
:::
