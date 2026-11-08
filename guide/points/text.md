## 1. Flat points: where gradient descent can stop

This chapter follows Sections II-C and II-D of the survey. Gradient descent moves against the gradient, so it can only come to rest where the gradient is zero. Those are the points we must understand.

:::def Critical point (survey Definition 1)
A point $\theta$ is a **[[critical-point|critical point]]** (or first-order stationary point) if $\nabla L(\theta) = 0$. Since an algorithm never lands exactly on such a point, we also use a tolerance $\epsilon > 0$: $\theta$ is an **$\epsilon$-first-order stationary point** ([[fosp|$\epsilon$-FOSP]]) if $\lVert\nabla L(\theta)\rVert \le \epsilon$.
:::

A critical point can be a local minimum (every nearby point is at least as high), a local maximum, or a **[[saddle]]**: neither, because some nearby points are lower and some are higher. The Hessian often tells which, but not always.

:::def Strict and degenerate saddles (survey Definition 2)
A saddle is **[[strict-saddle|strict]]** if $\lambda_{\min}(\nabla^2 L(\theta)) < 0$: at least one direction curves strictly down. A saddle at which $\nabla^2 L(\theta) \succeq 0$, meaning no eigenvalue is negative, is **[[degenerate]]** (or non-strict). For a degenerate point, the Hessian cannot decide between minimum and saddle; only third- or higher-order derivatives can.
:::

The survey's Fig. 2 shows four flat points at the origin.

:::survey Survey Fig. 2. The four prototypical critical points at the origin, with the Hessian spectrum above each surface.
public/survey/critical_points.webp
:::

:::read How to read this figure
**What is drawn.** Four surfaces over the plane $(\theta_1, \theta_2)$, each with a critical point at the origin (red dot). Above each surface is its Hessian spectrum $\lambda(\nabla^2 f)$ at the origin: the two curvatures. The contour lines are drawn on the floor of each box.

**Panel by panel.** (a) $\theta_1^2 + \theta_2^2$, spectrum $\{2, 2\}$: both curvatures positive, a local minimum. (b) $\theta_1^2 - \theta_2^2$, spectrum $\{2, -2\}$: one negative curvature, a strict saddle, with a downhill direction gradient methods can use. (c) $\theta_1^2 - \theta_2^4$, spectrum $\{2, 0\} \succeq 0$: no negative curvature, so every second-order test is passed, yet the surface falls along $\theta_2$. This is a degenerate saddle. (d) $\theta_1^3 - 3\theta_1\theta_2^2$, the monkey saddle, spectrum $\{0, 0\}$: the Hessian is zero, and the surface has three valleys and three ridges.

**What to notice.** Only case (b) has a negative eigenvalue. The efficient-escape guarantees of Section III exist for (b), and not, in general, for (c) and (d).
:::

The visual below lets you visit these points, and a few more, and watch which ones catch gradient descent.

:::viz zoo
:::

:::read How to read this visual
**What is drawn.** Each choice is a surface whose gradient is zero at the center. The two buttons pick the function. *all λ ≠ 0* holds the non-degenerate ones: a minimum, a maximum, and a strict saddle. *some λ = 0* holds the degenerate ones: the monkey saddle, a flat minimum $x^2 + y^4$, and the flat saddle $x^2 - y^4$ of the survey's Fig. 2(c). The tubes through the center are the Hessian's eigen-directions, colored blue (curves up), red (curves down) or amber (flat to second order). The readout lists the Hessian at the center, its index (the number of negative eigenvalues), and the kind of point.

**What moves.** *Drop 12 balls* starts gradient descent ($\eta = 0.1$, 400 steps) from twelve points on the unit circle around the center.

**What to notice.** A minimum catches every ball. The strict saddle catches only balls that start exactly on its upward axis, a set of zero width, and all others roll away along the red tube. The degenerate saddle $x^2 - y^4$ has no red tube at all: its Hessian $\begin{bmatrix}2&0\\0&0\end{bmatrix}$ looks like a minimum's, yet the loss still decreases along $y$. Balls leave it, but slowly, because along $y$ the slope is only $-4y^3$. This is the survey's warning example: every second-order test is passed, and the point is still not a minimum.
:::

## 2. What counts as success

Because a small gradient can still mean a saddle, the escape literature asks for more than first-order stationarity. Two definitions make this precise.

:::def Strict-saddle property (Ge et al. 2015; survey Definition 3)
A twice-differentiable $L$ is **$(\alpha, \gamma, \epsilon, \delta)$-strict saddle** if every point $\theta$ satisfies at least one of: (i) $\lVert\nabla L(\theta)\rVert \ge \epsilon$; (ii) $\lambda_{\min}(\nabla^2 L(\theta)) \le -\gamma$; or (iii) $\theta$ is within distance $\delta$ of a local minimum $\theta^\star$, and $L$ is $\alpha$-strongly convex on the $2\delta$-neighborhood of $\theta^\star$.
:::

In words: every point is either steep, or has a clear downhill curvature direction, or is already close to a good minimum. The property excludes degenerate saddles, and it is the central structural assumption of the escape theory in chapters 6 and 7.

:::def Approximate second-order stationarity (survey Definition 4)
$\theta$ is an **$(\epsilon, \gamma)$-second-order stationary point** ($(\epsilon,\gamma)$-[[sosp|SOSP]]) if
$$\lVert\nabla L(\theta)\rVert \le \epsilon \quad\text{and}\quad \lambda_{\min}\big(\nabla^2 L(\theta)\big) \ge -\gamma.$$
When the Hessian is $\rho$-Lipschitz (Assumption 2 below), the standard choice is $\gamma = \sqrt{\rho\epsilon}$, and the point is simply called an $\epsilon$-SOSP.
:::

Why $\sqrt{\rho\epsilon}$? A gradient of size $\epsilon$ and a negative curvature of size $\sqrt{\rho\epsilon}$ each allow a decrease of order $\epsilon^{3/2}/\sqrt{\rho}$ in one well-chosen step, so the two conditions are balanced. If $L$ is strict saddle with thresholds larger than these tolerances, every $\epsilon$-SOSP must lie near a local minimum. This is how algorithms that find SOSPs become algorithms that find local minima.

:::warn Three outcomes that must not be confused
An **FOSP** has a small gradient. An **SOSP** additionally has no significant negative curvature. A **global minimum** has loss no larger than any other point. In general, reaching one does not guarantee the next. The survey's Fig. 3 nests them.
:::

:::survey Survey Fig. 3. Hierarchy of solution concepts, and which kind of method guarantees which set.
public/survey/solution_hierarchy.webp
:::

:::read How to read this figure
**What is drawn.** Nested boxes, from the largest set to the smallest. The outer box is every parameter point $\theta \in \mathbb{R}^d$. Inside it is the set of $\epsilon$-FOSPs, the points with a small gradient. Inside that is the set of $\epsilon$-SOSPs, which additionally have $\lambda_{\min}(\nabla^2 L) \ge -\sqrt{\rho\epsilon}$. Inside that are the local minima, and inside those the global minima.

**The crosses.** Each cross marks what can hide in a gap between two boxes. A strict saddle with $\lambda_{\min} < -\sqrt{\rho\epsilon}$ is an FOSP but not an SOSP. A degenerate saddle with $\lambda_{\min} = 0$ is an SOSP but not a local minimum.

**The arrows on the right.** These say which methods reach which box. GD and SGD guarantee an FOSP. Escape methods, such as perturbed GD (chapter 7), guarantee an SOSP. Reaching the local or global minima needs additional structure, such as the strict-saddle property.
:::

:::viz sosp
:::

:::read How to read this visual
**What is drawn.** A map seen from above of $f = \tfrac12x^2 + \tfrac14y^4 - \tfrac12y^2$, which has a saddle at the origin (red) and minima at $(0, \pm1)$ (white). The thin lines are contours of $f$. The axes are $x$ and $y$.

**The colors.** Amber marks every point with $\lVert\nabla f\rVert \le \epsilon$: the $\epsilon$-FOSPs. Green marks the part of the amber region that also has $\lambda_{\min}(\nabla^2 f) \ge -\sqrt{\rho\epsilon}$: the $\epsilon$-SOSPs. Here $\rho = 9$ is the Hessian-Lipschitz constant of $f$ on the drawn strip.

**What to try.** Move the tolerance $\epsilon$ from small to large.

**What to notice.** For small $\epsilon$, there are three amber patches, one around each critical point, but only the two around the minima are green. The SOSP test throws out the saddle, because the saddle's curvature $-1$ is far below the allowance $-\sqrt{\rho\epsilon}$. The readout says what happens when $\epsilon$ is too loose: once $\epsilon \ge 1/\rho \approx 0.111$, the allowance reaches $-1$, and the saddle itself counts as an $\epsilon$-SOSP.
:::

## 3. What the theorems assume

Every convergence guarantee needs quantitative regularity of the loss. The survey uses four standard assumptions, and chapter 14 checks each against practice.

:::def Assumption 1: Lipschitz gradient ($\ell$-smoothness)
$\lVert\nabla L(\theta) - \nabla L(\theta')\rVert \le \ell\,\lVert\theta - \theta'\rVert$ for all $\theta, \theta'$. For twice-differentiable $L$ this is the same as $-\ell I \preceq \nabla^2 L(\theta) \preceq \ell I$: no curvature, up or down, exceeds $\ell$ anywhere. A loss with this property is called [[l-smooth|$\ell$-smooth]].
:::

The word "Lipschitz" means "never changes faster than a fixed rate". The first visual shows the idea for a function itself; Assumption 1 applies the same idea to the gradient.

:::viz lipschitz
:::

:::read How to read this visual
**What is drawn.** The blue curve is $f(x) = 0.8\sin 2x + 0.3x$. The red wedges form a double cone placed at the point $x_0$ of the graph. They are the regions the graph would have to enter to rise or fall faster than slope $\pm K$ from that point.

**What to try.** Slide the cone along the graph with *cone at $x_0$*, and change $K$.

**What to notice.** $f$ is $K$-Lipschitz exactly when the graph never enters the red regions, wherever the cone is placed. The coral marks show where the rule breaks. The steepest slope of this $f$ is $1.9$, so the rule holds everywhere exactly when $K \ge 1.9$. For Assumption 1, replace "the function" by "the gradient": the gradient of an $\ell$-smooth loss never changes faster than rate $\ell$.
:::

Why does $\ell$ matter so much? Because it makes a single gradient step safe. If $L$ is $\ell$-smooth, then at every point $L$ lies below the parabola $L(\theta) + \nabla L(\theta)^\top\Delta + \tfrac{\ell}{2}\lVert\Delta\rVert^2$. The step $\eta = 1/\ell$ jumps to the lowest point of that parabola and is guaranteed to decrease the loss by at least $\lVert\nabla L\rVert^2/(2\ell)$. This is the [[descent-lemma|descent lemma]], and almost every saddle-escape theorem rests on it, with a step size $\eta \le 1/\ell$. Chapter 10 shows that practical training runs far outside this regime.

:::viz upper
:::

:::read How to read this visual
**What is drawn.** The blue curve is $f(x) = 0.4\cos 2x + 0.1x^2$, whose curvature $f''$ ranges from $-1.4$ to $1.8$, so its smoothness constant is $1.8$. The violet parabola is the upper bound $f(x) + f'(x)(y - x) + \tfrac{L}{2}(y - x)^2$ built at the point $x$ with a constant $L$ you choose. The coral dot is the gradient step with $\eta = 1/L$, which lands at the bottom of the parabola.

**What to try.** Move the point $x$, then lower $L$ below $1.8$ near $x = 1.57$, where $f$ bends most sharply.

**What to notice.** With $L \ge 1.8$ the parabola stays above $f$ everywhere, and the readout confirms that the actual decrease is at least the guaranteed $f'^2/(2L)$. With $L$ below the true constant, the parabola cuts through $f$ (red), and the guarantee can fail. A step size too large for the curvature breaks the descent guarantee. That sentence is the seed of Section IV of the survey.
:::

:::def Assumption 2: Lipschitz Hessian
$\lVert\nabla^2 L(\theta) - \nabla^2 L(\theta')\rVert \le \rho\,\lVert\theta - \theta'\rVert$: the curvature itself changes at a bounded rate $\rho$ ([[hessian-lipschitz|Hessian-Lipschitz]]). It lets the theory trust a negative curvature measured at one point over a small neighborhood.
:::

:::def Assumption 3: stochastic gradients with bounded variance
The algorithm sees a random estimate $g(\theta;\xi)$ of the gradient that is right on average, $\mathbb{E}_\xi[g(\theta;\xi)] = \nabla L(\theta)$, with bounded spread $\mathbb{E}_\xi\lVert g(\theta;\xi) - \nabla L(\theta)\rVert^2 \le \sigma^2$. The canonical example is the [[minibatch]] gradient: the gradient of the average loss over a random batch of examples instead of all $n$.
:::

:::viz cloud
:::

:::read How to read this visual
**What is drawn.** A map of the parameter plane for fitting a line $wx + b$ to 50 points: slope $w$ horizontally, intercept $b$ vertically. The green dot is the minimum.

**The arrows and dots.** At the white point (drag it), the white arrow is one full gradient step of size $\eta = 0.08$ using all 50 examples. Each coral dot is where the same step would land if the gradient were computed on a random minibatch of $B$ examples instead; there are 400 of them. The violet dot is their average.

**What to try.** Change the batch size $B$ from 1 to 50, and drag the point.

**What to notice.** The violet average sits on the tip of the white arrow: minibatch gradients are right on average (unbiased). Their spread shrinks as $B$ grows, and the readout compares the measured spread with the exact value $\sigma^2/B$. This is Assumption 3 in a picture: SGD takes the right step plus random noise. Chapters 7, 9 and 13 show that this noise is not only an error. It changes which saddles are escaped and which curvatures are tolerated.
:::

:::def Assumption 4: $(L_0, L_1)$-smoothness (Zhang et al. 2020)
$\lVert\nabla^2 L(\theta)\rVert \le L_0 + L_1\lVert\nabla L(\theta)\rVert$. The curvature may grow with the gradient norm instead of being globally bounded ([[l0l1-smooth]]). It was proposed after Zhang et al. observed that the local smoothness of neural-network losses correlates with the gradient norm along training trajectories, and it underlies recent analyses of Adam and of "tamed" SGD.
:::

:::viz clip
:::

:::read How to read this visual
**What is drawn.** The surface $f(x, y) = x^2/2 + x^4/4 + y^2/2$ on a log height scale (the $y$-axis is stretched for visibility). Far out along $x$ the quartic term makes it very steep, and the curvature grows with the gradient. It is not $\ell$-smooth for any single $\ell$, but it is $(L_0, L_1)$-smooth.

**What moves.** Violet: gradient descent with one fixed step $h$, which must be tiny to survive the steep start. Coral: clipped gradient descent, which takes steps of length at most $\gamma = 0.5$ while the gradient is large, and full steps $\eta = 1$ near the minimum (green).

**What to notice.** The readout counts iterations until $\lVert\nabla f\rVert \le 10^{-4}$. The fixed step crawls, because a step safe for the steepest region is far too small near the minimum. Clipping adapts the step to the local curvature. Assumption 4 is the property that makes this provably better (Zhang et al. 2020, Theorems 3 and 4), and it is the setting of the tamed-SGD avoidance result in chapter 6.
:::

:::key What to keep from this chapter
- A zero gradient can be a minimum, a strict saddle (some $\lambda < 0$), or a degenerate saddle (no $\lambda < 0$, yet not a minimum).
- FOSP, SOSP, local minimum and global minimum are four different targets. Each result guarantees one of them, and the gaps between them are where the difficulty lies.
- Most theorems assume $\ell$-smoothness with a step $\eta \le 1/\ell$, a Lipschitz Hessian, and bounded-variance noise. Remember these assumptions: the large-step half of the survey starts where the first one is violated.
:::

:::problem The next step
We have the landscape and the targets. Next come the algorithms themselves: what GD, SGD, momentum, Adam and Newton's method do in one step, and why Newton's method is drawn to saddles.
:::
