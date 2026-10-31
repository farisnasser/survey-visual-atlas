## 1. Flat is not finished

Stop 4 ended with a warning: where the gradient is zero, the quadratic model is only $\tfrac12 \Delta^\top H \Delta$, and a negative eigenvalue of $H$ is a way down that the gradient cannot see. Stop 3's guarantee only promised a small gradient. This stop sorts out what a small gradient can mean.

A point $x$ with $\nabla f(x) = 0$ is a **[[critical-point|critical point]]** (also called a stationary point). Gradient descent stops moving there: $x - \eta \cdot 0 = x$. At a critical point the first-order term of the quadratic model vanishes, and

$$f(x + \Delta) \;\approx\; f(x) + \tfrac12\, \Delta^\top \nabla^2 f(x)\, \Delta .$$

So near $x$ the landscape is shaped by the Hessian alone. Write $\Delta$ in the eigenvector basis, $\Delta = \sum_i c_i v_i$; then $\tfrac12 \Delta^\top H \Delta = \tfrac12 \sum_i \lambda_i c_i^2$ (stop 4, section 3). The signs of the eigenvalues decide everything, and this is the **second-derivative test**:

| Eigenvalues of $\nabla^2 f(x)$ at a critical point | What $x$ is |
|---|---|
| all $> 0$ | a strict local minimum: every direction goes up |
| all $< 0$ | a strict local maximum: every direction goes down |
| some $> 0$, some $< 0$ | a saddle point: up in some directions, down in others |
| none negative, at least one $= 0$ | undecided: higher-order terms decide (section 3) |
| some negative, some $= 0$ | a saddle or a maximum: it cannot be a minimum |

A **[[saddle|saddle point]]** is a critical point that is neither a local minimum nor a local maximum, like a mountain pass between two valleys. The last row holds because a negative eigenvalue $\lambda_i < 0$ gives $f(x + t v_i) = f(x) + \tfrac12 \lambda_i t^2 + O(t^3) < f(x)$ for small $t$, so $x$ is not a local minimum, whatever the zero eigenvalues do.

## 2. The zoo

Six functions with a critical point at the origin, one of each kind. In each case $\nabla f(0) = 0$; only the Hessian differs:

| $f(x, y)$ | $\nabla^2 f(0)$ | eigenvalues | what the origin is |
|---|---|---|---|
| $x^2 + y^2$ | $\operatorname{diag}(2, 2)$ | $2, 2$ | minimum |
| $-x^2 - y^2$ | $\operatorname{diag}(-2, -2)$ | $-2, -2$ | maximum |
| $x^2 - y^2$ | $\operatorname{diag}(2, -2)$ | $2, -2$ | saddle |
| $x^3 - 3xy^2$ | $0$ | $0, 0$ | "monkey saddle": three ways down |
| $x^2 + y^4$ | $\operatorname{diag}(2, 0)$ | $2, 0$ | minimum, invisible to the Hessian |
| $x^2 - y^4$ | $\operatorname{diag}(2, 0)$ | $2, 0$ | saddle, invisible to the Hessian |

The number of negative eigenvalues at a critical point is its **[[index]]**: how many independent downhill directions it has. A minimum has index 0, the saddle $x^2 - y^2$ has index 1, the maximum in two dimensions has index 2. In $d$ dimensions the index can be anything from $0$ to $d$; every index strictly between is a kind of saddle.

Now drop twelve gradient-descent balls from the unit circle onto each surface. The minimum catches all 12 and the maximum none. The saddle $x^2 - y^2$ catches exactly 2: the two balls that start on the $x$-axis, where the downhill coordinate $y$ is exactly zero and stays zero forever. The monkey saddle catches 3, the balls that start on its three ridges.

:::viz zoo
:::

## 3. Strict saddles and degenerate points

The last two rows of the zoo show the limit of the second-derivative test. $x^2 + y^4$ and $x^2 - y^4$ have the same Hessian at the origin, $\operatorname{diag}(2, 0)$, yet one is a minimum and the other a saddle: only the fourth-order term tells them apart. A critical point whose Hessian has a zero eigenvalue is **[[degenerate|degenerate]]**. Optimization theory separates the saddles that the Hessian can see from those it cannot:

:::def Strict saddle
A critical point $x$ is a **[[strict-saddle|strict saddle]]** if $\lambda_{\min}\big(\nabla^2 f(x)\big) < 0$: there is at least one direction of strictly negative curvature. A function has the **strict-saddle property** if every critical point that is not a local minimum is a strict saddle.
:::

Lee et al. 2016 (Definition 2) use exactly this definition; it allows zero eigenvalues as long as one is negative, and it also covers local maxima. The saddle $x^2 - y^2$ is strict. The monkey saddle and $x^2 - y^4$ are not: their Hessians have no negative eigenvalue, and the way down is a higher-order effect. That makes them slow to leave. On $x^2 - y^4$ the downhill coordinate grows by $y \leftarrow y + 4\eta y^3$, a growth that stalls as $y$ shrinks: with $\eta = 0.1$, starting from $y = 0.05$ takes $504$ steps to reach $\lvert y \rvert = 1$.

:::warn Telling a minimum from a saddle is hard in general
For non-degenerate points the Hessian decides. For degenerate ones it does not, and no efficient general test is known: deciding whether a given point is a local minimum is NP-hard, even for polynomials of degree four whose Hessian at the point is zero ([Murty & Kabadi 1987](paper.html?id=murty1987some), via the restatement of Ahmadi and Zhang). [Anandkumar & Ge 2016](paper.html?id=anandkumar2016efficient) show that a third-order condition can still be checked and reached in polynomial time, while finding a fourth-order local minimum is NP-hard (their Theorem 10). This is why the guarantees in the papers target strict saddles and second-order conditions (section 6), not "a local minimum".
:::

## 4. Saddles made by symmetry

Where do saddles in neural networks come from? The smallest example is a network with one weight per layer: input $1$, hidden value $u \cdot 1$, output $v \cdot u$, target $1$, squared error

$$f(u, v) = (1 - uv)^2,$$

$$\nabla f(u, v) = \big(-2v(1 - uv),\; -2u(1 - uv)\big).$$

Every point with $uv = 1$ is a global minimum ($f = 0$): a whole curve of them, the hyperbola, because $(cu, v/c)$ computes the same function for every $c \ne 0$ (stop 11 returns to such continuous families).

Now the symmetry. Flipping both signs, $(u, v) \to (-u, -v)$, leaves $f$ unchanged. A function with $f(-p) = f(p)$ has $\nabla f(-p) = -\nabla f(p)$ (differentiate both sides), and at $p = 0$ this says $\nabla f(0) = -\nabla f(0)$, so $\nabla f(0) = 0$. **The point fixed by the symmetry is forced to be critical.** Here it is a saddle:

$$\nabla^2 f(0) = \begin{bmatrix} 0 & -2 \\ -2 & 0 \end{bmatrix},$$

with eigenvalue $-2$ along $(1, 1)/\sqrt2$ (downhill) and $2$ along $(1, -1)/\sqrt2$ (uphill). It is a strict saddle, with $f(0) = 1$. The symmetry also explains why such losses are not convex (stop 4): the minima $(1, 1)$ and $(-1, -1)$ are swapped by the symmetry, and the chord between them passes through the origin, where $f = 1 > 0$.

Real networks have many such symmetries. Swapping two hidden units (with their incoming and outgoing weights) never changes the function, so every minimum comes with copies, and the points where two units coincide are fixed by the swap. [Zhang, Saxe & Latham 2026](paper.html?id=zhang2026saddle) show that critical points of a narrower network reappear as critical points of a wider one (their Theorem 1, for example by duplicating a unit and splitting its output weight) and that units which start identical stay identical under gradient flow (Theorem 3). For deep linear networks, [Baldi & Hornik 1989](paper.html?id=baldi1989neural) (one hidden layer) and [Kawaguchi 2016](paper.html?id=kawaguchi2016deep) (any depth, Theorem 2.3) prove, under rank and distinct-eigenvalue assumptions on the data, that every local minimum is global and every other critical point is a saddle.

Depth makes the saddles flatter. In the chain $(1 - abc)^2$ with three weights, the Hessian at the origin is the zero matrix: a degenerate saddle. Kawaguchi's Corollary 2.4 makes this precise: with one hidden layer every saddle of a deep linear network has a negative eigenvalue, while with two or more there are saddles with none. Starting gradient descent at $(s, 0.6s)$ near the origin with $\eta = 0.05$, the one-unit network reaches $f < 0.01$ in $37$, $61$ and $85$ steps for $s = 0.1$, $0.01$, $0.001$ (growing like $\log(1/s)$), while the deeper chain needs $85$, $833$ and $8{,}301$ (growing like $1/s$). Long, flat stretches of this kind are the **plateaus** seen in training curves.

:::viz sym
:::

## 5. Why saddles outnumber minima in high dimensions

In two dimensions, minima and saddles seem equally common. [Dauphin et al. 2014](paper.html?id=dauphin2014identifying) argue that in high dimensions they are not, and their argument (their section 2) goes like this.

1. They cite a result of Bray and Dean (2007) for **random Gaussian fields**, a model of a "typical" smooth function in $N$ dimensions. For such functions the critical points concentrate on a monotonically increasing curve in the plane of (index fraction $\alpha$, error $\varepsilon$): the higher the value of $f$ at a critical point, the larger the fraction $\alpha$ of its Hessian eigenvalues that are negative. The eigenvalues at a critical point follow Wigner's **semicircle law**, shifted by an amount set by its error.
2. So for large $N$, most critical points are saddles, and critical points with high error are almost never minima.
3. They measure the same qualitative trend at critical points found in small neural networks trained on downsampled MNIST and CIFAR-10 (their Figure 1).

A random-matrix experiment shows why "all eigenvalues positive" is so demanding. Take a random symmetric $N \times N$ matrix (the Gaussian orthogonal ensemble, variance $1/N$ off the diagonal) as a stand-in for a Hessian. With no shift, the chance that all $N$ eigenvalues are positive is about $0.50$ for $N = 1$, $0.145$ for $N = 2$, $0.021$ for $N = 3$, $0.0015$ for $N = 4$, and none of $2{,}000$ samples succeed for $N \ge 5$. That is even faster than the $2^{-N}$ of independent coin flips ($2^{-4} = 0.0625$). A $100 \times 100$ sample has $50$ negative eigenvalues: index 50. Shifting the spectrum right by $m = 1$ (lower error, in Bray and Dean's picture) leaves $19$ negative ones, close to the semicircle law's $0.196$, and still makes all-positive spectra rare as $N$ grows.

:::viz random
:::

:::warn What this argument is, and is not
Bray and Dean's result is a theorem about random Gaussian fields, not about neural networks; the network evidence in Dauphin et al. is empirical, on small networks. The random-matrix visual is an illustration with a shift we choose by hand. None of this proves that a given network has no bad local minima (stop 11 collects what has been proved), but it explains why the field turned its attention from "local minima" to "saddles".
:::

## 6. First-order and second-order stationary points

Back to guarantees. Stop 3 showed that gradient descent finds an **[[eps-stationary|$\varepsilon$-stationary point]]**, $\lVert \nabla f(x) \rVert \le \varepsilon$. Papers call this a **[[fosp|first-order stationary point]]** ($\varepsilon$-FOSP): it constrains the gradient and says nothing about curvature, so saddles and maxima qualify. The natural stronger target adds the Hessian:

:::def Second-order stationary point
$x$ is a **[[sosp|second-order stationary point]]** (SOSP) if $\nabla f(x) = 0$ and $\nabla^2 f(x) \succeq 0$: no gradient and no negative curvature. Strict saddles are excluded by definition. A local minimum is always an SOSP; an SOSP can still be a degenerate saddle such as $x^2 - y^4$.
:::

The approximate version needs a second tolerance for the curvature, and the papers tie it to $\varepsilon$ through one more smoothness constant.

:::def Hessian-Lipschitz
$f$ is **[[hessian-lipschitz|$\rho$-Hessian-Lipschitz]]** if $\lVert \nabla^2 f(x) - \nabla^2 f(y) \rVert \le \rho\, \lVert x - y \rVert$ for all $x, y$, where the norm of a symmetric matrix is its largest eigenvalue in absolute value. It is stop 3's $L$-smoothness one level up: the curvature cannot change faster than $\rho$ (rho) times the distance moved. It also bounds the error of the quadratic model: $\big\lvert f(x + \Delta) - f(x) - \nabla f(x) \cdot \Delta - \tfrac12 \Delta^\top \nabla^2 f(x) \Delta \big\rvert \le \frac{\rho}{6} \lVert \Delta \rVert^3$ (Nesterov & Polyak 2006, Lemma 1).
:::

Following [Jin et al. 2017](paper.html?id=jin2017escape) (Definition 6), $x$ is an **$\varepsilon$-second-order stationary point** if

$$\lVert \nabla f(x) \rVert \le \varepsilon \quad \text{and} \quad \lambda_{\min}\big(\nabla^2 f(x)\big) \ge -\sqrt{\rho \varepsilon}.$$

Why $\sqrt{\rho\varepsilon}$? Look at the units. A gradient is a curvature times a distance, and $\rho$ is a curvature per distance, so $\rho \varepsilon$ is a curvature squared, and $\sqrt{\rho \varepsilon}$ is the one curvature that can be built from $\rho$ and $\varepsilon$ alone. It is the scale at which the two conditions are equally hard to meet, and it keeps rates comparable across papers. Some papers use a separate tolerance $\gamma$ for the curvature and write $(\varepsilon, \gamma)$-SOSP.

A worked example: $f(x, y) = \tfrac12 x^2 + \tfrac14 y^4 - \tfrac12 y^2$ has a strict saddle at the origin ($\lambda_{\min} = -1$) and minima at $(0, \pm 1)$. Its Hessian is $\operatorname{diag}(1, 3y^2 - 1)$, which changes at rate at most $6\lvert y \rvert$, so $\rho = 9$ on the strip $\lvert y \rvert \le 1.5$. With $\varepsilon = 0.01$ the curvature allowance is $\sqrt{9 \cdot 0.01} = 0.3$, so the saddle is an $\varepsilon$-FOSP but not an $\varepsilon$-SOSP. Only for $\varepsilon \ge 1/9 \approx 0.111$ does the allowance reach 1 and let the saddle through: a loose tolerance can certify a saddle.

:::viz sosp
:::

:::key The target of the next stop
Under the strict-saddle property, an $\varepsilon$-SOSP with small $\varepsilon$ is close to a local minimum, and for many problems (matrix sensing, matrix completion, robust PCA; stop 11) every local minimum is global. So "find an $\varepsilon$-SOSP fast" became the central question of non-convex optimization theory.
:::

:::atlas Where this shows up in the Atlas
- [Lee et al. 2016](paper.html?id=lee2016gradient) define strict saddles (Definition 2) and prove that gradient descent almost never converges to one (stop 6).
- [Jin et al. 2017](paper.html?id=jin2017escape) define the $\varepsilon$-SOSP with $\sqrt{\rho\varepsilon}$ (Definition 6) used throughout the escape literature.
- [Kawaguchi 2016](paper.html?id=kawaguchi2016deep) and [Baldi & Hornik 1989](paper.html?id=baldi1989neural): in deep linear networks every local minimum is global; deeper ones have saddles without negative curvature.
- [Ge, Jin & Zheng 2017](paper.html?id=ge2017no) and [Bhojanapalli et al. 2016](paper.html?id=bhojanapalli2016global) prove the strict-saddle property and "all local minima are global" for matrix problems (stop 11).
- [Dauphin et al. 2014](paper.html?id=dauphin2014identifying): the saddle-point argument of section 5, and a method designed to escape saddles (stop 6).
- [Murty & Kabadi 1987](paper.html?id=murty1987some) and [Anandkumar & Ge 2016](paper.html?id=anandkumar2016efficient): the hardness of certifying a local minimum at degenerate points.
:::

:::problem The next problem
Gradient descent stops at any critical point, saddles included, and the zoo shows that a start exactly on the right line ends at a saddle. Near a degenerate saddle it can linger for thousands of steps. **Is gradient descent ever truly stuck at a saddle, and how do we get out fast?**
:::
