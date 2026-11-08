## 1. Why saddles matter

This chapter follows Sections III-A and III-B of the survey: the first half of RQ1. The left side of the multiplier line, $\lambda < 0$, is about saddles. Before asking whether gradient descent escapes them, we ask why they deserve a whole section.

Dauphin et al. (2014) gave the answer its modern form. Think of the Hessian at a critical point in $d$ dimensions as having $d$ eigenvalues. The point is a local minimum only if *all* of them are positive. If each sign were an independent coin flip, that would happen with probability $2^{-d}$: essentially never for a network with millions of parameters. Real Hessians are not coin flips, and Dauphin et al. made the picture precise using results on random Gaussian fields from statistical physics (Bray & Dean 2007). The [[index]] of a critical point, the fraction of negative Hessian eigenvalues, is strongly correlated with its loss. Critical points with high loss are overwhelmingly saddles, and local minima concentrate at loss values close to the global minimum. Dauphin et al. confirmed this correlation empirically for small networks on MNIST and CIFAR-10, and showed that the plateaus around saddles slow gradient descent down.

If bad local minima are rare, the practical obstacle is the time spent near saddles, and the key quantity is the curvature of the escape direction: the $|\lambda|$ in the escape-time formula of chapter 5.

:::viz random
:::

:::read How to read this visual
**What is drawn.** The histogram shows the eigenvalues of one random symmetric $100\times100$ matrix, a stand-in for the Hessian at a critical point. Red bars are negative eigenvalues (downhill directions); blue bars are positive ones. The white curve is the semicircle law that such random spectra follow. The slider shifts the whole spectrum to the right. In Bray and Dean's picture, a lower loss corresponds to a spectrum shifted to the right. The chart below plots, on a log scale, the probability that all $N$ eigenvalues of a random $N\times N$ matrix are positive, for $N = 1$ to $10$, estimated from 2,000 matrices each.

**What to try.** Slide the shift $m$ from $0$ to the right.

**What to notice.** At $m = 0$ about half of the eigenvalues are negative: a typical critical point is a saddle with many downhill directions. Shifting right removes negative eigenvalues; only a large shift, meaning a low loss, leaves none. The chart falls steeply with $N$: even at $N = 10$, having all eigenvalues positive is already rare. The shift is our slider, not a fitted model. It illustrates the mechanism, not a measurement of real networks.
:::

Depth makes the obstacle worse, because it creates saddles with no downhill curvature at all. For deep linear networks, Kawaguchi (2016) proved, under full-rank conditions on the data and distinct eigenvalues, that for any depth and any widths every local minimum is a global minimum and every other critical point is a saddle. At saddles where the product of the hidden-layer weights has full rank, the Hessian has at least one strictly negative eigenvalue. But networks with two or more hidden layers also have **"bad" saddles whose Hessian has no negative eigenvalue at all**, such as the origin. Minima are benign; some saddles are degenerate.

:::viz bad
:::

:::read How to read this visual
**What is drawn.** The surface is a slice of the loss of the smallest deep linear network: three scalar weights in a chain, loss $(1 - w_1w_2w_3)^2$, restricted to $w_3 = w_2$. The green curve is the set of global minima. The amber tubes through the origin are the Hessian's eigen-directions there. Both are amber because both curvatures are zero: the Hessian at the origin is the zero matrix.

**What moves.** *Nudge ball* places a ball near the origin and runs gradient descent.

**What to notice.** The top is a flat plateau. There is no red tube, no direction that curves down, so the multiplier of chapter 5 is exactly $1$ in every direction. The ball can only leave through higher-order terms, and it creeps before it finds the way out. This is the kind of saddle that every escape guarantee in this chapter excludes.
:::

:::viz race
:::

:::read How to read this visual
**What is drawn.** Two surfaces side by side. Left: one hidden layer, loss $(1 - w_1w_2)^2$, whose origin is a strict saddle with one red (downhill) direction. Right: two hidden layers, the degenerate saddle of the previous visual, with only amber (flat) directions.

**What moves.** *Restart race* drops one ball on each surface at the same small distance from the origin, $(0.05, 0.04)$, with the same step size.

**What to notice.** The left ball slides off quickly along the red direction: its multiplier is larger than one. The right ball crawls for a long time, because near a degenerate saddle the slope away grows only like a higher power of the distance. Same nudge, same step size, and depth alone turns a quick slide into a long crawl.
:::

## 2. Asymptotic avoidance: GD almost never converges to a strict saddle

The foundational result is due to Lee, Simchowitz, Jordan & Recht (2016).

:::theorem Lee et al. 2016, informal (survey Theorem 1)
Let $L$ be twice continuously differentiable with an $\ell$-Lipschitz gradient, let $0 < \eta < 1/\ell$, and suppose every saddle point of $L$ is strict. If $\theta_0$ is drawn from any distribution with a density, then GD converges to a strict saddle with probability zero. Consequently, whenever GD converges, it converges to a local minimizer [[almost-surely|almost surely]].
:::

"Any distribution with a density" covers every usual random initialization: Gaussian, uniform on a box, and so on. The proof is a short and elegant use of dynamical systems theory, and each of its steps is visible in the visuals below.

1. **The GD step is a smooth map.** Write one step as $G(\theta) = \theta - \eta\nabla L(\theta)$. Its Jacobian (chapter 2's matrix of first derivatives) is $DG = I - \eta\nabla^2 L(\theta)$, with eigenvalues $1 - \eta\lambda_j$: the multipliers of chapter 5. When $\eta < 1/\ell$, every $|\eta\lambda_j| < 1$, so every multiplier is positive, $DG$ is invertible, and $G$ is a **diffeomorphism**: a smooth map with a smooth inverse. It never folds the plane onto itself.
2. **At a strict saddle, one multiplier exceeds one.** $DG$ has the eigenvalue $1 + \eta|\lambda_{\min}| > 1$: the left region of the multiplier line.
3. **Few points flow into the saddle.** By the center-stable manifold theorem (Shub 1987), the points that converge to such a saddle lie, locally, on a manifold of dimension strictly less than $d$: a curve in the plane, a surface in 3D. Such a set has **[[measure-zero|Lebesgue measure zero]]**: zero length, zero area, zero volume.
4. **Pulling back keeps it thin.** The set of *all* initializations that eventually converge to the saddle is obtained by pulling this thin set back through $G$ again and again. Since $G$ is a diffeomorphism, the preimage of a measure-zero set has measure zero. A random start lands in it with probability zero.

:::viz stable
:::

:::read How to read this visual
**What is drawn.** The surface $f = x^2/2 + y^4/4 - y^2/2$: a strict saddle at the center, with curvature $-1$ along $y$ (red) and $+1$ along $x$ (blue), and minima at $(0, \pm1)$ (green). The white line is the $x$-axis, the saddle's stable set: from any point on it, gradient descent rolls straight into the saddle.

**What moves.** *Drop 40 random balls* runs gradient descent from 40 random starts. *Start on the stable set* drops one ball exactly on the white line.

**What to notice.** The readout counts where the balls end. Every random ball ends at a minimum, none at the saddle. Only the deliberately placed ball, started exactly on the line, reaches the saddle. The line has zero area, so a random start never lands on it. This is step 3 of the proof in two dimensions.
:::

:::viz map
:::

:::read How to read this visual
**What is drawn.** The faint grid is a square grid of the plane. The blue grid is the same grid after one gradient-descent step $g(x) = x - \alpha\nabla f(x)$ has been applied to every point, on the same $f$ as above. Red cells mark where the map folds, where the Jacobian determinant $\det Dg = (1 - \alpha)\big((1 + \alpha) - 3\alpha y^2\big)$ is negative.

**What to try.** Slide the step $\alpha$ from small to large.

**What to notice.** For small $\alpha$, the blue grid is a smooth deformation of the original: no cell is flipped, the map is a diffeomorphism, and a thin set stays thin when pulled back (step 4 of the proof). The readout gives $1/L$ on the drawn grid. Once $\alpha$ passes it, red folds appear, the map is no longer invertible, and the proof's argument no longer applies. The theorem needs a small step size, and chapter 11 shows that real training uses larger ones.
:::

## 3. How the avoidance result was extended

The argument has been extended in several directions. Each extension keeps the conclusion and changes the setting.

- **One theorem for many methods.** Lee, Panageas, Piliouras, Simchowitz, Jordan & Recht (2019) proved a single statement: for any first-order method whose update map has an invertible Jacobian everywhere, the set of initial points that converge to an unstable fixed point has measure zero. A countable-cover argument removes the need for the saddles to be isolated. They then checked the conditions for GD, the proximal point method, coordinate and block coordinate descent, manifold gradient descent, and mirror descent.
- **Step sizes that shrink.** Panageas, Piliouras & Wang (2019) proved a stable-manifold theorem for time-varying systems. They showed that GD, mirror descent, the proximal point method and manifold GD avoid strict saddles almost surely when the step sizes decay, but no faster than order $1/k$. The corrected version of their proof assumes that the Hessian at the saddle is invertible. Step sizes that decay too fast, so that their sum is finite, can stop the iterates at points that are not even critical.
- **Adaptive step sizes.** Antonakopoulos, Mertikopoulos, Piliouras & Wang (2022) showed that the AdaGrad family, with scalar, diagonal or full-matrix preconditioning, avoids saddles from almost every initialization, for deterministic gradients. This was not a routine extension: AdaGrad's preconditioner depends on the whole gradient history, so the earlier results do not apply directly.
- **Stochastic gradients.** Mertikopoulos, Hallak, Kavis & Cevher (2020) showed that SGD with step sizes decaying like $1/n^p$, a family that includes but goes beyond the Robbins–Monro schedule, converges almost surely to the set of critical points. Under bounded and uniformly exciting noise, it avoids strict saddles with probability one, including whole manifolds of non-isolated saddles.
- **No global smoothness.** Apidopoulos, Lytras & Mertikopoulos (2026) extended almost-sure convergence and saddle avoidance to "tamed" variants of SGD under the $(L_0, L_1)$-smoothness of chapter 3, removing the globally Lipschitz gradient.
- **Real (small) networks.** Cheridito, Jentzen & Rossmannek (2024) proved a center-stable manifold theorem with weaker regularity, and applied it to shallow ReLU and leaky-ReLU networks with scalar input, trained on affine targets. GD avoids most saddle points, and if the loss along the run converges to a value below an explicit threshold, the limit is a global minimum.

:::viz ridge
:::

:::read How to read this visual
**What is drawn.** A loss that depends only on the distance $r$ from the center: $f = r^4/4 - r^3 + r^2$. It has a minimum in the middle, a ring of minima at $r = 2$ (green), and between them a ridge at $r = 1$ (red) made entirely of strict saddles: curvature $-1$ across the ridge, $0$ along it. This is a manifold of non-isolated saddles, the case Mertikopoulos et al. cover.

**What moves.** *Release balls* starts plain GD (white) and eight SGD runs with decreasing steps (coral) exactly on the ridge. The SGD noise is bounded and points in every direction with some probability: it is "uniformly exciting".

**What to notice.** GD started on the ridge stays there forever: it is on the stable set, so it has no reason to leave. Every SGD run falls off the ridge, either into the middle or out to the ring. The readout counts where they end and reports an offline experiment with 200 SGD runs from the ridge. Noise turns "almost every start" into "every run, with probability one".
:::

:::warn What these theorems do not say
Together, these results settle the asymptotic question under strict-saddle conditions. They say nothing about **how long** convergence takes, and nothing about **degenerate saddles**, which the stable-manifold argument does not cover. A start very close to the stable set takes a long time to leave (chapter 5's escape-time formula). The next chapter shows that "eventually" can mean "after an exponentially long time".
:::

:::key What to keep from this chapter
- In high dimension, critical points with high loss are mostly saddles, and depth adds degenerate saddles with no downhill curvature.
- From a random start, GD with $\eta < 1/\ell$ converges to a strict saddle with probability zero, because the GD map is a diffeomorphism and the points that flow into a saddle form a measure-zero set.
- The result extends to many first-order methods, to decaying steps, to AdaGrad, to SGD with exciting noise, to tamed SGD, and to small ReLU networks. It is asymptotic, and it needs strict saddles.
:::

:::problem The next step
"Converges to a saddle with probability zero" leaves room for "spends an absurdly long time near saddles first". Can that really happen, and what fixes it?
:::
