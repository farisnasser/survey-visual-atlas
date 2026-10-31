## 1. The second derivative: how fast the slope changes

Stop 3 ended with a zig-zag: in a long valley, gradient descent bounces across the steep walls and creeps along the flat floor, and the single smoothness constant $L$ cannot explain why. $L$ only says how sharply the landscape bends *at most*. To see the zig-zag coming we need to measure the bend itself, in every direction.

Start again in one dimension. The derivative $f'(x)$ is the slope. The slope itself changes as we move, and its rate of change is the **second derivative**:

$$f''(x) \;=\; \lim_{h \to 0} \frac{f'(x + h) - f'(x)}{h}.$$

Its sign says which way the curve bends. If $f''(x) > 0$ the slope is increasing, so the curve bends **up**, like a cup. If $f''(x) < 0$ the slope is decreasing and the curve bends **down**, like a cap. Its size says how sharply. This number is the **curvature** we met on stop 3's parabola $\tfrac{a}{2}x^2$, where $f'' = a$ everywhere.

Adding the bend to the tangent line gives the second-order Taylor approximation, a parabola that touches $f$ at $x$:

$$f(x + h) \;\approx\; f(x) + f'(x)\, h + \tfrac12 f''(x)\, h^2 .$$

Take stop 1's loss $f(x) = 0.1x^2 + 0.5\sin 3x$. Its second derivative is $f''(x) = 0.2 - 4.5 \sin 3x$. At $x = -0.5$ it is $4.689$ (bends up); at $x = 0.5$ it is $-4.289$ (bends down). Over the whole line $f''$ ranges from $-4.3$ to $4.7$, so this $f$ is $L$-smooth with $L = 4.7$ (stop 3). Looking ahead by $h = 0.3$ from $x = -0.5$, the tangent line misses $f$ by $0.194$; the parabola misses by only $0.017$.

:::viz bend
:::

## 2. The Hessian: all the second derivatives at once

With $d$ coordinates the slope is the gradient, a vector, and "how the slope changes" becomes a table: how the $i$-th entry of the gradient changes when we nudge the $j$-th coordinate. That table is the **[[hessian|Hessian]]**, written $\nabla^2 f(x)$ or $H$:

$$\big(\nabla^2 f(x)\big)_{ij} \;=\; \frac{\partial^2 f}{\partial x_i\, \partial x_j}(x) \;=\; \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_j}\right)(x).$$

It is a $d \times d$ matrix. The diagonal entry $\partial^2 f / \partial x_i^2$ is the ordinary second derivative along coordinate $i$. The off-diagonal entries measure *twist*: how the slope in one coordinate changes as we move in another. When the second partial derivatives are continuous, the order of differentiation does not matter (Schwarz's theorem), so $H_{ij} = H_{ji}$: **the Hessian is symmetric**.

A worked example. For $f(x, y) = 2x^2 + 2xy + 2y^2$:

$$\nabla f = \begin{pmatrix} 4x + 2y \\ 2x + 4y \end{pmatrix}, \qquad \nabla^2 f = \begin{bmatrix} 4 & 2 \\ 2 & 4 \end{bmatrix},$$

the same at every point, because $f$ is quadratic.

**Curvature along any direction.** Walk from $x$ along a unit vector $u$: the height along this line is $\varphi(t) = f(x + t u)$, a function of one variable. Applying the chain rule twice gives its second derivative,

$$\varphi''(0) \;=\; u^\top \nabla^2 f(x)\, u \;=\; \sum_{i,j} u_i\, H_{ij}\, u_j .$$

So the Hessian holds the curvature of $f$ along *every* direction at once: feed it a direction, get back a number. In the example, along the $x$-axis $u = (1, 0)$ and $u^\top H u = 4$.

The **[[quadratic-model|quadratic model]]** is the $d$-dimensional version of the parabola of section 1:

$$f(x + \Delta) \;\approx\; f(x) + \nabla f(x) \cdot \Delta + \tfrac12\, \Delta^\top \nabla^2 f(x)\, \Delta .$$

The first two terms are stop 2's tangent plane; the last one bends it into a bowl, a dome or a saddle. The error of this approximation shrinks like $\lVert \Delta \rVert^3$ when the Hessian changes smoothly (stop 5 makes this precise). Almost every result in the rest of the course is a statement about this model.

## 3. Eigenvalues: the principal curvatures

A $d \times d$ table of numbers is hard to picture. Luckily a symmetric matrix always has a much simpler description. A nonzero vector $v$ is an **[[eigen|eigenvector]]** of $H$ if $H$ only stretches it,

$$H v = \lambda v,$$

and the stretch factor $\lambda$ (lambda) is its **eigenvalue**. Along an eigenvector the curvature is $v^\top H v = \lambda$ (for $\lVert v \rVert = 1$). The **spectral theorem** says that a symmetric $d \times d$ matrix has $d$ real eigenvalues $\lambda_1, \dots, \lambda_d$ with eigenvectors $v_1, \dots, v_d$ that are **perpendicular** to each other and have length 1. These are the **principal directions** of bending, and the eigenvalues are the **principal curvatures**.

In the example, $H = \begin{bmatrix} 4 & 2 \\ 2 & 4 \end{bmatrix}$ has $\lambda = 6$ along $v = (1, 1)/\sqrt{2}$ and $\lambda = 2$ along $v = (1, -1)/\sqrt{2}$. Check the first: $H (1, 1) = (6, 6) = 6 \cdot (1, 1)$.

Every other direction is a blend. Write a unit vector in the eigenvector basis, $u = \sum_i c_i v_i$ with $c_i = u \cdot v_i$ and $\sum_i c_i^2 = 1$. Then

$$u^\top H u \;=\; \sum_{i=1}^{d} \lambda_i\, c_i^2 ,$$

a weighted average of the eigenvalues. It is never larger than the largest eigenvalue $\lambda_{\max}$ and never smaller than the smallest, $\lambda_{\min}$. In the example the $x$-axis makes a 45° angle with both eigenvectors, so $c_1^2 = c_2^2 = \tfrac12$ and $u^\top H u = \tfrac12 \cdot 6 + \tfrac12 \cdot 2 = 4$, as before.

This is also the precise meaning of stop 3's smoothness constant: for a twice-differentiable $f$, $f$ is $L$-smooth exactly when every eigenvalue of every Hessian lies between $-L$ and $L$. One number $L$ bounds the sharpest bend; the eigenvalues tell us all of them.

The signs of the eigenvalues decide the shape of the quadratic model: all positive is a **bowl**, all negative a **dome**, mixed signs a **saddle**, and a zero eigenvalue a flat **trough** along its eigenvector. Try them all:

:::viz quad
:::

## 4. Gradient descent, one eigen-direction at a time

Now the zig-zag. Take a quadratic with its minimum at $x^\star$:

$$f(x) = f^\star + \tfrac12 (x - x^\star)^\top H\, (x - x^\star), \qquad \nabla f(x) = H (x - x^\star).$$

Near any minimum of a smooth function, the quadratic model has exactly this form (the gradient is zero there), so what we learn here describes every landscape close to its minima. One GD step gives

$$x_{t+1} - x^\star \;=\; x_t - x^\star - \eta H (x_t - x^\star) \;=\; (I - \eta H)\,(x_t - x^\star),$$

where $I$ is the identity matrix. Measure the position in the eigenvector basis: $u_i = v_i \cdot (x - x^\star)$ is the coordinate along the $i$-th principal direction. Since $H v_i = \lambda_i v_i$, the matrix $I - \eta H$ just multiplies each coordinate by a number:

$$\boxed{\;u_i \;\leftarrow\; (1 - \eta \lambda_i)\, u_i \qquad \text{for each } i = 1, \dots, d.\;}$$

In the eigenvector basis, GD on a $d$-dimensional quadratic is **$d$ separate copies of stop 3's parabola**, one per principal direction, all forced to share the same $\eta$. Stop 3's table now applies to each direction:

- Stability needs $\lvert 1 - \eta \lambda_i \rvert < 1$ for every $i$, so $\eta < 2/\lambda_{\max}$: the sharpest direction sets the largest allowed step.
- Progress along the flattest direction shrinks by only $1 - \eta \lambda_{\min}$ per step, which is close to 1 when $\lambda_{\min}$ is small.

Push $\eta$ close to $2/\lambda_{\max}$ to speed up the flat direction and the sharp coordinate's factor approaches $-1$: it flips sign every step. That sign flip, seen in the original coordinates, is the zig-zag. With $\mu = 1$, $L = 10$ and $\eta = 0.18$, the factors are $0.82$ and $-0.8$, and the run below needs 38 steps to bring $f$ under $10^{-6}$:

:::viz zigzag
:::

## 5. The condition number: why narrow valleys are slow

Write $\mu = \lambda_{\min} > 0$ for the flattest curvature and $L = \lambda_{\max}$ for the sharpest. Their ratio is the **[[condition-number|condition number]]**,

$$\kappa \;=\; \frac{L}{\mu} \;\ge\; 1 .$$

$\kappa$ (kappa) measures how elongated the valley is: $\kappa = 1$ is a round bowl, $\kappa = 1000$ is a long, thin canyon. A problem with large $\kappa$ is called **ill-conditioned**.

With the safe step $\eta = 1/L$, the sharp direction is solved in one step (factor $0$), and the flat direction shrinks by $1 - \mu/L = 1 - 1/\kappa$ per step. Since $1 - 1/\kappa \le e^{-1/\kappa}$, after $t$ steps that coordinate has shrunk by at least $e^{-t/\kappa}$, so

$$t \;\ge\; \kappa \,\ln(1/\varepsilon) \quad \text{steps shrink every coordinate by the factor } \varepsilon .$$

The best single step balances the two extremes, $\lvert 1 - \eta\mu \rvert = \lvert 1 - \eta L \rvert$, which gives $\eta = 2/(L + \mu)$ and the factor $(\kappa - 1)/(\kappa + 1) \approx 1 - 2/\kappa$: only twice as fast. With $\kappa = 10$ and $\varepsilon = 10^{-6}$: $\eta = 1/L$ has factor $0.9$ and needs $132$ steps, the best step has factor $9/11$ and needs $69$. With $\kappa = 100$, $\eta = 1/L$ needs $1{,}375$ steps; with $\kappa = 1000$, $13{,}809$. **The cost of gradient descent grows linearly with the condition number**, and no choice of a single step size fixes that. On quadratics, giving the method memory (stop 7) brings it down to $\sqrt{\kappa}$; giving each direction its own step (stop 9) attacks $\kappa$ itself.

:::viz kappa
:::

## 6. Convex, strongly convex, positive semidefinite

The signs of the eigenvalues deserve names, because the papers state their assumptions with them.

:::def Positive semidefinite
A symmetric matrix $H$ is **[[psd|positive semidefinite]]** (PSD), written $H \succeq 0$, if $v^\top H v \ge 0$ for every vector $v$; equivalently, all its eigenvalues are $\ge 0$. It is **positive definite**, $H \succ 0$, if all eigenvalues are $> 0$. We write $A \succeq B$ when $A - B \succeq 0$, so $H \succeq \mu I$ means every eigenvalue of $H$ is at least $\mu$.
:::

:::def Convex function
$f$ is **[[convex]]** if the straight segment (chord) between any two points of its graph never lies below the graph:

$$f\big(\theta x + (1 - \theta) y\big) \;\le\; \theta f(x) + (1 - \theta) f(y)$$

for all $x, y$ and all $0 \le \theta \le 1$.

For a twice-differentiable $f$ this is equivalent to $\nabla^2 f(x) \succeq 0$ at every point: no direction ever bends down.
:::

Convexity removes the trouble of stop 1's many valleys: **every local minimum of a convex function is a global minimum**. The proof is two lines. Suppose $x$ is a local minimum and some $y$ has $f(y) < f(x)$. On the chord from $x$ to $y$, convexity gives $f\big(x + \theta(y - x)\big) \le f(x) + \theta\big(f(y) - f(x)\big) < f(x)$ for every $\theta > 0$, even tiny ones. So there are points arbitrarily close to $x$ that are lower, and $x$ was not a local minimum after all.

:::def Strongly convex
$f$ is **[[strongly-convex|$\mu$-strongly convex]]** ($\mu > 0$) if $\nabla^2 f(x) \succeq \mu I$ everywhere: every direction bends up with curvature at least $\mu$. Together with $L$-smoothness, every curvature lies between $\mu$ and $L$, and $\kappa = L/\mu$ is the condition number of the whole function, not just of one quadratic.
:::

The quadratic picture of section 5 extends to every strongly convex, $L$-smooth function, with a short proof that reuses stop 3. Curvature at least $\mu$ means $f$ lies above a bowl touching it at $x$ (the descent-lemma argument with the inequality reversed):

$$f(y) \;\ge\; f(x) + \nabla f(x) \cdot (y - x) + \frac{\mu}{2}\, \lVert y - x \rVert^2 \quad \text{for all } y.$$

The right side is smallest at $y = x - \nabla f(x)/\mu$, where it equals $f(x) - \lVert \nabla f(x) \rVert^2 / (2\mu)$. The left side is never below $f^\star$, so

$$\lVert \nabla f(x) \rVert^2 \;\ge\; 2\mu\, \big(f(x) - f^\star\big) :$$

far above the minimum, the gradient must be large. Plug this into stop 3's descent lemma with $\eta = 1/L$, $f(x_{t+1}) \le f(x_t) - \lVert \nabla f(x_t) \rVert^2/(2L)$, and subtract $f^\star$:

$$f(x_{t+1}) - f^\star \;\le\; \Big(1 - \frac{\mu}{L}\Big)\big(f(x_t) - f^\star\big),$$

and repeating this $t$ times,

$$f(x_t) - f^\star \;\le\; \Big(1 - \frac{1}{\kappa}\Big)^{t} \big(f(x_0) - f^\star\big).$$

This is called **linear convergence** (the error shrinks by a constant factor per step): about $\kappa \ln(1/\varepsilon)$ steps, far fewer than the $O(\varepsilon^{-2})$ of stop 3, which is all that holds without convexity. The inequality $\lVert \nabla f \rVert^2 \ge 2\mu (f - f^\star)$ is useful on its own: it has a name, the Polyak–Łojasiewicz (PL) condition, and returns on stop 11, because it can hold without convexity.

:::warn Neural-network losses are not convex
Stop 1's loss has $f''(0.5) = -4.289 < 0$, so it is not convex, and network losses with hidden layers are not convex either. One reason (stop 5): swapping two hidden units gives a second minimum with the same loss, while the minima of a convex function form a convex set, so the whole chord between the two would have to be minima too, and it usually is not. The convex results above are a benchmark and the local picture near a minimum, not a description of the whole landscape.
:::

## 7. Hessian-vector products: curvature without the matrix

The Hessian is the right object, but it is huge. A network with $d = 10^8$ parameters has a Hessian with $d^2 = 10^{16}$ entries: at 4 bytes each, 40 petabytes. Nobody forms it. What methods use instead is the **[[hvp|Hessian-vector product]]** $\nabla^2 f(x)\, v$ for one chosen vector $v$, and there are two ways to get it.

**From two gradients.** By the definition of the derivative applied to the gradient,

$$\nabla^2 f(x)\, v \;=\; \lim_{h \to 0} \frac{\nabla f(x + h v) - \nabla f(x)}{h},$$

so one extra gradient gives an approximation. Its error has two sources pulling in opposite directions: for large $h$ the curvature changes between $x$ and $x + hv$ (an error proportional to $h$), and for tiny $h$ the two gradients agree in almost every digit and the subtraction loses accuracy to rounding. In the visual the forward difference is best around $h \approx 10^{-8}$, with error about $10^{-8}$; the **central** difference $\big(\nabla f(x + hv) - \nabla f(x - hv)\big)/(2h)$ has error proportional to $h^2$ and does better, about $10^{-11}$ near $h \approx 10^{-6}$.

**Exactly, by automatic differentiation.** The number $\nabla f(x) \cdot v$ is a scalar function of $x$, and its gradient is $\nabla^2 f(x)\, v$ (the Hessian is symmetric). So running backpropagation (stop 2) through the gradient computation once more gives the exact product, at a cost of a small constant multiple of one gradient. Deep-learning libraries do this routinely.

:::viz hvp
:::

Hessian-vector products are enough to explore curvature. Repeatedly multiplying a vector by $H$ (the **power method**) or by a cleverer polynomial in $H$ (**Lanczos**) finds the extreme eigenvalues and their eigenvectors, using nothing but products. Stop 6 uses exactly this to find the directions that lead away from saddle points.

:::atlas Where this shows up in the Atlas
- [Dauphin et al. 2014](paper.html?id=dauphin2014identifying) analyse optimizers in the Hessian's eigenvector basis, as in section 4: along eigenvector $i$ a gradient step moves by an amount proportional to $\lambda_i$, so directions of small curvature are slow, and they build a method that rescales each direction by $1/\lvert\lambda_i\rvert$ (stop 6).
- [Carmon & Duchi 2018](paper.html?id=carmon2018analysis) analyse Krylov (Lanczos) methods that solve the quadratic model with Hessian-vector products only, with rates that depend on $\lambda_{\min}$ and $\lambda_{\max}$.
- [Xu, Jin & Yang 2018 (NEON)](paper.html?id=xu2018first) and [Allen-Zhu & Li 2018 (NEON2)](paper.html?id=allenzhu2018neon2) replace Hessian-vector products by differences of gradients, as in section 7, to find negative curvature (stop 6).
- [Tripuraneni et al. 2018](paper.html?id=tripuraneni2018stochastic) and [Arjevani et al. 2020](paper.html?id=arjevani2020second) build stochastic second-order methods that only ever query gradients and Hessian-vector products.
:::

:::problem The next problem
The quadratic model is a bowl only when every eigenvalue is positive. Where the gradient is zero, the model is just $\tfrac12 \Delta^\top H \Delta$, and a negative eigenvalue means a way down that the gradient cannot see. **A zero gradient can still hide downward curvature: which flat points are traps?**
:::
