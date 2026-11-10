## 1. From data to a non-convex objective

This chapter follows Sections II-A and II-B of the survey. Given training examples $(x_i, y_i)$, $i = 1, \dots, n$, a model $f_\theta$ and a per-example [[loss]] $\ell$, the **empirical risk** is the average loss (survey Eq. 1):

$$L(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell(f_\theta(x_i), y_i), \qquad \theta \in \mathbb{R}^d.$$

$\theta$ collects every weight and bias into one long vector, even though software stores each layer as a matrix. Two cases make the shape of $L$ concrete:

- **A linear model with squared loss**, $f_\theta(x) = x^\top\theta$: $L$ is a convex quadratic, a single bowl.
- **A multilayer network**: $f_\theta$ multiplies weight matrices together, with activations in between. Products of weights are enough to make $L$ non-convex. For example, the one-number network $f(x) = w_2 w_1 x$ fitting $y = x$ has loss $(1 - w_2 w_1)^2$, which is minimized on the whole curve $w_2 w_1 = 1$ and has a saddle at the origin.

So a descent method need not find a *global* minimizer, and the curvature can change sign from place to place. Keep one more distinction in mind: $L$ is the *training* loss. How well the network does on new data, its generalization, is a different question, which the survey keeps separate throughout (chapter 17).

## 2. The gradient and the first-order model

The [[gradient]] $g(\theta) = \nabla L(\theta)$ has coordinates $g_j = \partial L / \partial\theta_j$, the slopes along each parameter. For a small displacement $\Delta$, the first-order [[taylor|Taylor]] model is (survey Eq. 2)

$$L(\theta + \Delta) \approx L(\theta) + g^\top\Delta.$$

Which direction lowers this model fastest? By the Cauchy–Schwarz inequality, $g^\top\Delta \ge -\|g\|_2\|\Delta\|_2$, with equality when $\Delta$ points along $-g$. So among all steps of the same Euclidean length, $-g$ gives the largest *first-order* decrease. That is all "steepest descent" means. It does not promise that a finite step decreases the true loss, because the true loss bends away from the plane.

:::viz plane
:::

:::read How to read this visual
**What is drawn.** A surface $f(x, y) = 0.3x^2 + 0.2y^2 + 0.4\sin(1.5x)\cos y$, a small curved landscape, with heights to scale. The violet patch is the tangent plane $f(x) + \nabla f(x)^\top\Delta$: the first-order model of Eq. (2) at the white point. The coral arrow is the gradient direction.

**What moves.** Move the point with the position sliders, and grow the size of the patch.

**What to notice.** Right at the point, plane and surface agree: the gradient captures the slope exactly. As the patch grows, the surface curves away from the plane. The readout compares, for an uphill step of length $r$, the value the plane predicts with the true value; the error grows with $r$. That gap is what the second-order model below accounts for, and what limits how far a gradient step can be trusted (chapter 3).
:::

## 3. The Hessian and the quadratic model

The **[[hessian|Hessian]]** $H(\theta) = \nabla^2 L(\theta)$ is the $d \times d$ matrix of second derivatives, $H_{jk} = \partial^2 L/(\partial\theta_j\partial\theta_k)$. It is symmetric. Adding it gives the local quadratic model (survey Eq. 3):

$$L(\theta + \Delta) \approx L(\theta) + g^\top\Delta + \tfrac12\Delta^\top H\Delta.$$

The quadratic term answers a question the gradient cannot: *how fast will the slope change as we move?* The cleanest way to read it is through the [[eigen|eigenvectors]] of $H$. If $Hv = \lambda v$ with $\|v\|_2 = 1$, a step $\Delta = a v$ contributes $\tfrac12\lambda a^2$:

- $\lambda > 0$: the loss curves **up** along $v$, like a bowl.
- $\lambda < 0$: it curves **down**, so moving along $v$ lowers the loss faster than the plane predicts.
- $\lambda \approx 0$: the model is **flat** along $v$.

At a stationary point, where $g = 0$, one negative eigenvalue rules out a local minimum: the point is a saddle or a maximum. All eigenvalues nonnegative ($H \succeq 0$) is necessary for a local minimum, but not sufficient on its own.

:::viz quad
:::

:::read How to read this visual
**What is drawn.** The surface $\tfrac12\Delta^\top H\Delta$ for a $2\times2$ symmetric Hessian built from the sliders: two eigenvalues $\lambda_1$, $\lambda_2$ and the angle of the first eigenvector. The default is the worked example $H = \begin{bmatrix}4 & 2\\ 2 & 4\end{bmatrix}$, with eigenvalues $6$ and $2$ along the two diagonals. The two eigen-directions are drawn as tubes: blue where the surface bends up, red where it bends down, amber where it is flat. The coral tube is any other direction $u$ you choose with the last slider.

**What to try.** Make one eigenvalue negative; then turn the rotation slider.

**What to notice.** The eigen-directions are always perpendicular, and the curvature along any other direction, $u^\top H u$ in the readout, is a blend of the two eigenvalues. With mixed signs the surface becomes a saddle. The readout also prints the matrix $H$: as soon as the eigenvectors are rotated away from the axes, $H$ gets off-diagonal entries. That is the rotated valley of chapter 4, the case where a per-coordinate step size breaks down.
:::

The survey's Fig. 2 shows the four local shapes that the rest of the survey keeps returning to.

:::survey Fig. 2 of the survey: loss geometry in two dimensions. Gradients indicate local slope; Hessian eigenvectors indicate directions of principal curvature.
public/survey3/fig2.webp
:::

:::read How to read this figure
**What is drawn.** Four quadratics $L = \tfrac12\theta^\top H\theta$, each as contour lines (curves of equal loss). (a) A round **bowl**, $H = 2I$: every direction curves equally. (b) A **narrow valley**, $H = \mathrm{diag}(16, 1)$. (c) The same valley **rotated** by $45^\circ$. (d) A **saddle**, $H = \mathrm{diag}(1, -1)$; dashed contours are negative losses.

**The arrows.** Blue arrows are $-\nabla L$ at a grid of points, all drawn the same length so only their direction shows. The double-headed arrows through the centre are the Hessian eigenvectors, labelled with their eigenvalues: orange for the larger positive curvature, green for the smaller, vermillion for negative curvature.

**What to notice.** In the bowl, $-\nabla L$ points straight at the minimum. In both valleys it points mostly across the valley, toward the nearest wall, not along the valley floor where the minimum lies; chapter 4 shows what that does to gradient descent. In the rotated valley the eigenvectors no longer line up with the axes, even though the eigenvalues are the same as in (b). In the saddle, the arrows point away from the centre along the vermillion eigenvector: negative curvature pushes gradient descent off the saddle. These are local models only, not claims about the global landscape of a network.
:::

:::problem The next step
The quadratic model says how the slope changes. Chapter 3 turns this into a bound on how far a gradient step can safely go.
:::
