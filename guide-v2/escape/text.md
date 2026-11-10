## 1. Avoidance is not efficiency

This chapter follows Sections III-C and III-D of the survey: the second part of RQ1, about how long escape takes and what makes it fast.

The escape-time formula of chapter 5 already shows why "avoids saddles" and "escapes saddles quickly" are different statements. A start at distance $\delta$ from the stable manifold escapes in time proportional to $\log(1/\delta)$: finite, but unbounded. Du, Jin, Lee, Jordan, Póczos & Singh (2017) turned this observation into a striking negative result.

:::theorem Du et al. 2017, Theorem 4.1 (informal)
There is a smooth function on $\mathbb{R}^d$, with smoothness constants polynomial in $d$, on which gradient descent from a random start in $[-1, 1]^d$ stays far from every local minimum for $e^{\Omega(d)}$ steps. Perturbed gradient descent reaches a local minimum in time polynomial in $d$. The same holds for any initialization concentrated in a ball, such as a Gaussian (their Corollary 4.3).
:::

$\Omega(d)$ means "at least a constant times $d$", so $e^{\Omega(d)}$ is exponential in the dimension. The construction is a chain of saddles, arranged so that the trajectory leaving one saddle arrives very close to the stable manifold of the next. While GD escapes one saddle, the next escape direction still curves up, so the offset $\delta$ along it shrinks. Each wait is longer than the last by a constant factor, about $(L + \gamma)/\gamma$ in their analysis, where $L$ is the upward curvature and $\gamma$ the downward one. The waits multiply up to an exponential. **Almost-sure avoidance is compatible with arbitrarily slow escape.**

:::viz chain
:::

:::read How to read this visual
**What is drawn.** A surface with two saddles in a row: the origin, where GD must first grow $x$, and the points $(\pm1, 0)$, where it must then grow $y$. The minima are at $(\pm1, \pm1)$ (green), and the red tubes are the escape directions. This is our own two-saddle chain built on the same mechanism, not the paper's $d$-dimensional construction.

**What moves.** Coral: gradient descent from $(0.001, 0.05)$. Violet: perturbed GD, which adds a small random kick whenever the gradient is small.

**What to try.** Switch the upward curvature $L$ between 1, 2 and 4, and release the balls each time.

**What to notice.** While GD escapes the first saddle along $x$, the $y$ direction still curves up, so $y$ shrinks. When GD arrives at the second saddle, its offset $y$ is tiny, and the second escape takes much longer. The readout gives the step at which each ball escapes. Raising $L$ makes $y$ shrink faster and the second wait longer. Perturbed GD never starts the second escape from an absurdly small $y$, because its kick resets the offset. Chain $d$ such saddles together and the waits multiply.
:::

## 2. The minimal fix: perturbed gradient descent

The fix is to make sure GD never sits close to a stable set: add a little randomness exactly when the gradient becomes small. This is the one algorithm "designed to escape saddles" that the survey includes, because it is the smallest possible change to GD.

:::idea Perturbed gradient descent (Jin, Ge, Netrapalli, Kakade & Jordan 2017)
Run gradient descent. Whenever the gradient is small and no perturbation has been added recently, add a perturbation drawn uniformly from a small ball. If the loss has not decreased enough after a fixed number of further steps, stop and return the point before the perturbation: it is an approximate second-order stationary point.
:::

:::theorem Jin et al. 2017, Theorem 3
Under Assumptions 1 and 2 ($\ell$-smooth, $\rho$-Hessian-Lipschitz), with probability at least $1 - \delta$, [[perturbed-gd|perturbed GD]] outputs an $\epsilon$-SOSP within
$$O\!\left(\frac{\ell\,\Delta}{\epsilon^2}\cdot\log^4\frac{d\,\ell\,\Delta}{\epsilon^2\,\delta}\right)$$
iterations, where $\Delta = L(\theta_0) - \inf L$.
:::

Compare this with chapter 4: plain GD needs $O(\ell\Delta/\epsilon^2)$ iterations to find an FOSP. Perturbed GD finds an SOSP in the same number, up to a factor that is only poly-logarithmic in the dimension $d$. **Escaping strict saddles is almost free in iteration count.**

Why does one random kick work so reliably? The perturbations that fail to escape form a thin region around the stable manifold, a "pancake" in 3D. A random kick lands outside it with high probability, at a typical distance from the stable manifold, so the $\delta$ in the escape-time formula is no longer tiny.

:::viz kick
:::

:::read How to read this visual
**What is drawn.** The surface $f = x^2/2 + \cos y - 1$, whose gradient and Hessian are both 1-Lipschitz. It has a strict saddle at the center, with curvature $-1$ along $y$ (red) and $+1$ along $x$ (blue), and minima where $f = -2$ (green).

**What moves.** Both balls start on the saddle's stable set. White: plain GD. Coral: perturbed GD. The ring shows the kick radius $r$.

**What to try.** Change the kick radius $r$ and *Run again*.

**What to notice.** Plain GD rolls into the saddle and stops: its gradient is exactly zero there. Perturbed GD arrives at the same place, sees the gradient fall below the threshold, takes one random kick, and gradient descent then carries it down to a minimum. The readout shows each ball's loss, its gradient norm, and the number of kicks.
:::

:::viz pancake
:::

:::read How to read this visual
**What is drawn.** A cloud of 3D kick points sampled in a ball around a strict saddle in three dimensions, with eigenvalues $(1, -1, 1)$. Amber points are kicks that do not escape within $t_{\rm thres}$ gradient steps, meaning the loss drops by less than $0.05$. Faint blue points are kicks that do escape. The red line is the negative-curvature direction $y$.

**What to try.** Change $t_{\rm thres}$, the number of steps a kick is given to escape.

**What to notice.** The stuck kicks are not scattered at random. They form a thin, curved slab across the negative-curvature direction: the "pancake" of Jin et al.'s proof. The readout gives the fraction of stuck kicks, which is small and shrinks as $t_{\rm thres}$ grows. A random kick fails only if it lands in the slab, so it fails with small probability.
:::

## 3. Noise as an escape mechanism

Plain GD needs luck in its initialization to escape quickly. SGD has a built-in source of randomness, and the central question of this section is whether that randomness is enough on its own.

**Isotropic noise.** Ge, Huang, Jin & Yuan (2015) gave the first polynomial-time guarantee. Under the strict-saddle property, noisy SGD, which adds noise drawn uniformly from the unit sphere to each stochastic gradient, reaches a neighborhood of a local minimum in a number of iterations polynomial in $d$ and the problem parameters. They noted that the added noise is unnecessary if the stochastic gradients already have non-negligible variance in every direction. Isotropic noise, noise that is equally strong in every direction, guarantees a component along the escape direction. But its strength along any one fixed direction scales like $1/\sqrt d$, so the guarantees inherit a polynomial dependence on the dimension.

**Correlated negative curvature.** Daneshmand, Kohler, Lucchi & Hofmann (2018) asked whether the noise of plain SGD already has the component that is needed. They proposed the **[[cnc|correlated negative curvature]]** (CNC) condition: there is a $\gamma > 0$ such that, at every $\theta$,

$$\mathbb{E}_\xi\big[\langle v_\theta,\, g(\theta;\xi)\rangle^2\big] \ge \gamma,$$

where $v_\theta$ is a unit eigenvector of the smallest Hessian eigenvalue at $\theta$. In words, the stochastic gradients have a non-vanishing spread along the way down. They showed empirically, for several models including neural networks, that this spread does not shrink with the dimension, whereas the spread of isotropic noise along a fixed direction does. Under CNC, a variant of SGD that periodically takes a larger step reaches an $(\epsilon, \epsilon^{2/5})$-SOSP in $\tilde O(\epsilon^{-4})$ iterations with high probability. A variant that runs GD and takes a single larger stochastic step when the gradient is small does so in $\tilde O(\epsilon^{-2})$. Neither rate depends explicitly on $d$. ($\tilde O$ hides logarithmic factors.)

:::viz cnc
:::

:::read How to read this visual
**What is drawn.** The surface $f = x^2/2 - y^2/2 + y^4/4$, written as the average of 64 components $f_i = f + a_iy + c_ix$, each with a random tilt. Its saddle is at the center, the escape direction is $y$ (red), and the minima are at $(0, \pm1)$ (green). All balls start on the saddle's stable set $y = 0$.

**What moves.** White: plain GD. Coral: CNC-PGD (the paper's Algorithm 1), which runs GD and, when the gradient is small, takes one stochastic-gradient step with a larger step $r$. Violet: CNC-SGD (Algorithm 2), which runs SGD and takes a larger step every $t_{\rm thres}$ iterations. No artificial noise is added to either.

**What to try.** Switch *component noise* between "has a y-part (CNC holds)" and "x only (CNC fails)".

**What to notice.** With CNC, the component gradients have a spread along $y$, the readout reports the CNC constant, and both coral and violet escape. With noise only along $x$, the CNC constant is zero and both stay stuck with GD: noise that never points along the way down cannot help, however large it is. The guaranteed methods use these periodic larger steps; this is not a theorem about constant-step SGD.
:::

Fang, Lin & Zhang (2019) sharpened this picture with a **dispersive noise** assumption. Under it, and with Lipschitz gradients and Hessians, SGD with a simple averaging and restart rule finds an $\epsilon$-SOSP in $\tilde O(\epsilon^{-3.5})$ stochastic gradients. This improves on the $\tilde O(\epsilon^{-4})$ rate of perturbed SGD with isotropic noise, without injecting any noise at all.

## 4. Heavy tails and global exploration

Gradient noise in deep learning is often far from Gaussian. Gürbüzbalaban, Şimşekli & Zhu (2021) showed that **[[heavy-tail|heavy-tailed]]** behavior emerges in the SGD iterates even for linear regression with Gaussian data, through the multiplicative nature of the noise: each step multiplies the iterate by a random matrix. The resulting tail index $\alpha$ describes how fast $P(|x| > t)$ decays like $t^{-\alpha}$. It depends on the step size, the batch size, the data variance and the dimension, and larger step sizes and smaller batches produce heavier tails. Heavy tails mean occasional very large jumps, which can carry the iterate between basins.

:::viz tail
:::

:::read How to read this visual
**What is drawn.** A log–log plot of $P(|x_1| > t)$, the fraction of 6,000 independent SGD runs on streaming least-squares regression whose first coordinate exceeds $t$ after 500 steps. The dashed line has slope $-\alpha$, the tail index predicted by the paper's Theorem 3. On log–log axes a power law $t^{-\alpha}$ is a straight line, and a heavy tail is a line that falls slowly.

**What to try.** Change the step $\eta$, the batch size $b$ and the dimension $d$.

**What to notice.** The empirical tail follows the predicted slope. Raising $\eta$ or lowering $b$ makes the slope shallower: heavier tails. The readout reports $\alpha$, the critical step $\eta_{\rm crit}$ at which $\alpha = 2$, and quantiles of $\lVert x \rVert$. Above $\eta_{\rm crit}$, $\alpha < 2$, and the stationary variance is infinite even though every moment of the data is finite. The bounded-variance Assumption 3 of chapter 3 then fails, for the simplest model there is.
:::

Azizian, Iutzeler, Malick & Mertikopoulos (2025) quantified this global behavior using large-deviations theory. Modeling SGD as a randomly perturbed dynamical system, they derived matching upper and lower bounds on the expected time to reach the global minimizers. That time scales as $\exp(E/\eta)$, where the "energy" $E$ is set by the barriers that SGD might have to climb on its way. In landscapes without spurious minima, $E = 0$ and convergence is subexponential. Spurious minima do not merely exist: they control how long noisy training takes.

:::viz camel
:::

:::read How to read this visual
**What is drawn.** The paper's own test function (its Fig. 1), with three minima and two saddles. The global minimum $p_1$ is green; the local minima $p_3$ (middle) and $p_5$ (right) are white; the saddles $p_2$ and $p_4$ are red. The height is drawn as $\log(f + 7.6)$ so that all basins are visible.

**What moves.** The coral ball is SGD with Gaussian gradient noise, started in the middle basin $p_3$. It runs until it comes within $0.1$ of the global minimum $p_1$.

**What to try.** Change the step $\eta$ and *Run again* a few times.

**What to notice.** The ball usually falls first into the right basin $p_5$, which is cheaper to enter, and must then climb back out of it before it can reach $p_1$. That rare detour sets the exponent $E$ of the mean hitting time. The readout counts the crossings towards $p_5$ and gives the theory's prediction $\log\mathbb{E}[\tau] \approx E/\eta$. A smaller step makes each barrier crossing exponentially rarer.
:::

:::warn What the noise results assume
These analyses show that noise can provide escape without deliberate perturbation, but only under assumptions on its geometry relative to the Hessian, such as CNC or dispersivity. They have been checked empirically only on a limited range of models and training stages. Whether minibatch noise in modern architectures satisfies them throughout training is largely open (chapter 16).
:::

## 5. The survey's simulation of this chapter

The survey's Fig. 5 puts GD, perturbed GD, SGD and heavy-ball momentum on one saddle.

:::survey Survey Fig. 5. Original simulation of saddle escape on L = ½θ₁² − ½θ₂² + ¼θ₂⁴ (η = 0.1, start (1.5, 10⁻⁷)).
public/survey/saddle_escape.webp
:::

:::read How to read this figure
**Panel (a).** A contour map of $L = \tfrac12\theta_1^2 - \tfrac12\theta_2^2 + \tfrac14\theta_2^4$, with the strict saddle at the origin (×) and the minima at $(0, \pm1)$ (★). All four methods start at the white circle $(1.5, 10^{-7})$, a distance $10^{-7}$ from the stable manifold $\theta_2 = 0$. The colored lines are their paths: gray GD, red perturbed GD, teal SGD, yellow heavy ball.

**Panel (b).** The suboptimality $f(\theta_t) - f^\star$ on a log scale against the iteration. The dotted line marks the saddle's level. GD (gray) sits on the saddle level for about 150 iterations before it drops. Perturbed GD (red) escapes as soon as its gradient is small, and the arrow marks where it stops, at an $\epsilon$-SOSP, once a perturbation no longer decreases the loss. Its curve is flat from then on because the algorithm has halted, not because it is stuck. SGD (teal) escapes quickly but rattles at a noise floor. Heavy-ball momentum (yellow) escapes earliest.

**Panel (c).** The number of iterations until $|\theta_2| > 0.5$, against the initial offset $\delta$ from the stable manifold, on a log axis. GD's points lie on a straight line: the $\log(1/\delta)$ law of the escape-time formula. Perturbed GD's median is flat, insensitive to $\delta$, because one random perturbation resets the offset.
:::

| Result | Method | Key assumption | Guarantee |
|---|---|---|---|
| Lee et al. 2016 | GD | $\eta < 1/\ell$, random init. | converges to a strict saddle w.p. 0 (asymptotic) |
| Du et al. 2017 | GD | constructed chain of saddles | escape can take time exponential in $d$ |
| Jin et al. 2017 | perturbed GD | Lipschitz gradient and Hessian | $\epsilon$-SOSP in $\tilde O(\ell\Delta/\epsilon^2)$, poly-log in $d$ |
| Ge et al. 2015 | noisy SGD | strict-saddle property | near a local minimum in poly($d$) iterations |
| Daneshmand et al. 2018 | SGD with occasional larger steps | CNC | $(\epsilon, \epsilon^{2/5})$-SOSP in $\tilde O(\epsilon^{-4})$; no explicit $d$ |
| Fang et al. 2019 | SGD with averaging | dispersive noise | $\epsilon$-SOSP in $\tilde O(\epsilon^{-3.5})$ |
| Azizian et al. 2025 | SGD | large-deviations regime | time to global minimizers $\approx \exp(E/\eta)$ |

This is a condensed version of the survey's Table III.

:::key What to keep from this chapter
- Avoiding saddles is not escaping them quickly: a chain of saddles can delay GD exponentially in the dimension.
- One random kick when the gradient is small fixes this. Perturbed GD finds an SOSP at nearly the cost of an FOSP.
- SGD's own noise can do the same, if it has a component along the escape direction (CNC) or is dispersive. Heavy tails and large deviations describe how noise moves SGD between basins.
:::

:::problem The next step
So far a saddle has been an obstacle to leave as fast as possible. But training from small initialization *starts* at a saddle, and visits one after another. The next chapter shows why that sequence of saddles is how networks learn.
:::
