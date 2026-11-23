## 1. What exactly are we minimizing?

This chapter follows Sections II-A and II-B of the survey. Imagine a network that predicts house prices. It receives $n$ training pairs $(x_i, y_i)$: an input and its target. Its prediction is $f_\theta(x_i)$, where $\theta$ collects every trainable weight and bias, and a [[loss|loss function]] $\ell$ measures the mistake on one example. The survey's Eq. (1) averages those mistakes:

$$L(\theta) = \frac{1}{n}\sum_{i=1}^{n}\ell\big(f_\theta(x_i), y_i\big), \qquad \theta \in \mathbb{R}^d.$$

Read it as: predict for every example, measure each mistake, add them up, divide by $n$. With four examples whose losses are $1, 3, 2, 6$, the training loss is $(1 + 3 + 2 + 6)/4 = 3$. Changing the weights changes all four predictions, and so the average. Optimization means choosing $\theta$ so that this one number goes down.

**Why is $\theta$ a vector when the network stores matrices?** A tiny layer with weights $W = \begin{bmatrix} w_{11} & w_{12}\\ w_{21} & w_{22}\end{bmatrix}$ and biases $b = (b_1, b_2)$ can be listed as $\theta = (w_{11}, w_{12}, w_{21}, w_{22}, b_1, b_2)^\top \in \mathbb{R}^6$. Nothing about the model changes; we only list its numbers in one vector, which is what the theory needs. Section V will deliberately keep the matrix shape, because some optimizers use it.

## 2. Convex versus non-convex, with the smallest possible model

Take one weight and squared error, $L(w) = (w - 2)^2$. This is a bowl with a single minimum at $w = 2$: the derivative is $2(w - 2)$ and the second derivative is $2 > 0$ everywhere. For a linear model with squared error, the loss is always such a convex quadratic, and every local minimum is a global one.

Now stack two linear "layers" with no activation at all. For input $1$ the prediction is $uv$, and the target is $1$:

$$L(u, v) = (uv - 1)^2.$$

The settings $(u, v) = (1, 1)$ and $(2, \tfrac12)$ both give zero loss, and so does every point on the curve $uv = 1$. In between, at the origin, the gradient is zero but the loss is $1$: that point is a saddle. Holding $v = 0$ gives $L(u, 0) = 1$ for every $u$, a completely flat line. The parameters interact through their product, and that alone makes the landscape non-convex. This is the survey's point that multiplying the weight matrices of different layers can produce a non-convex objective even without nonlinear activations.

:::viz uvloss
:::

:::read How to read this visual
**What is drawn.** The loss $L(u, v) = (uv - 1)^2$ of the two-weight "network" above as a 3D surface (height $\sqrt{L}$, so the floor is visible). The green curves are the two branches of $uv = 1$, where the loss is exactly zero. The red marker is the origin, where the gradient vanishes but the loss is $1$.

**What moves.** The direction from which the ball starts, on a circle of radius $1.6$ around the origin. "Roll" runs gradient descent with $\eta = 0.05$.

**What to try.** Start from several directions, including one close to the line $v = -u$.

**What to notice.** Every run ends with zero loss, but at a different point of the green curve: the minimum is not unique, and which one you reach depends on where you start. Starts near $v = -u$ first drift toward the origin before turning away, because the origin is a saddle (curvature $-2$ along $u = v$, $+2$ along $u = -v$). Even this two-weight model has valleys, a saddle, and many equivalent solutions. That is why an optimizer generally cannot promise the globally best network from a random start.
:::

:::idea In plain English
A deep network's output is built by composing many simple parameterized operations. Even when each one looks harmless, the combined loss can have valleys, flat directions, saddles, and many equivalent parameter settings.
:::

**Training loss is not generalization.** $L(\theta)$ in Eq. (1) is measured on the training set. Good optimization means reducing it; good generalization means predicting well on new examples. One does not guarantee the other, and the survey keeps this distinction when it compares optimizers in Section VI.

## 3. What does the gradient tell us?

For two weights $\theta = (x, y)^\top$, the **[[gradient]]** is the vector of partial derivatives,

$$g = \nabla L(\theta) = \begin{bmatrix}\partial L/\partial x\\ \partial L/\partial y\end{bmatrix}.$$

Each component says how the loss changes if we nudge one weight and hold the others still. If $g = (3, 4)^\top$, raising $x$ by $0.1$ changes the loss by about $0.3$, and raising $y$ by $0.1$ changes it by about $0.4$. This is a local statement: it holds for small changes.

**Deriving the first-order Taylor model.** In one variable, $f(x + \delta) \approx f(x) + f'(x)\delta$. With several weights, the derivative is a vector and the step $\Delta$ is a vector, so (survey Eq. 2)

$$L(\theta + \Delta) \approx L(\theta) + g^\top\Delta \quad\Longleftrightarrow\quad \underbrace{L(\theta + \Delta) - L(\theta)}_{\text{change in loss}} \approx g^\top\Delta.$$

The dot product predicts uphill ($> 0$) or downhill ($< 0$). With $g = (3, 4)^\top$ and $\Delta = (-0.3, -0.4)^\top$, $g^\top\Delta = 3(-0.3) + 4(-0.4) = -2.5$: a predicted decrease of about $2.5$. It is a prediction for a small step, not a guarantee for any finite one.

@@viz plane@@

## 4. Why is $-g$ the direction of steepest descent?

We want $g^\top\Delta$ as negative as possible. But without a limit on the step's length, we could make $\Delta$ longer and longer, so first fix the length: $\|\Delta\|_2 = r$. Here $r$ is just a distance, not the learning rate; $\Delta = (3, 4)^\top$, for example, has length $\sqrt{3^2 + 4^2} = 5$. The dot product of two vectors is

$$g^\top\Delta = \|g\|_2\,\|\Delta\|_2\cos\phi = \|g\|_2\, r\cos\phi,$$

where $\phi$ is the angle between them. Since $\cos\phi \ge -1$, we get the survey's Cauchy–Schwarz bound

$$g^\top\Delta \ge -\|g\|_2\, r,$$

with equality exactly when $\phi = 180^\circ$, that is, when the step points straight against the gradient. To give that direction length $r$, normalize and rescale: $\Delta = -r\,g/\|g\|_2$. Gradient descent uses $\Delta = -\eta g$, the same direction with length $r = \eta\|g\|_2$.

:::viz steepest
:::

:::read How to read this visual
**What is drawn.** Left: the gradient $g = (3, 4)^\top$ (blue arrow) at a point, and the circle of all steps of length $r$. The coral arrow is one candidate step $\Delta$ on that circle. Right: the predicted change $g^\top\Delta = \|g\|\,r\cos\phi$ for every direction on the circle, against the angle $\phi$ between $\Delta$ and $g$; the coral dot is the current step.

**What moves.** The direction of the step, and its length $r$.

**What to try.** Turn the step all the way around, with $r = 0.5$.

**What to notice.** The predicted change is largest ($+\|g\|r$) along $g$ and most negative ($-\|g\|r$) exactly opposite it. With $r = 0.5$ the best step is $\Delta = (-0.3, -0.4)^\top$, the example above, and the readout gives $g^\top\Delta = -2.5$. Perpendicular steps ($\phi = 90^\circ$) are predicted to change nothing: they move along a contour.
:::

## 5. The Hessian and the second-order model

The gradient says how steep the loss is here; as we move, the slope itself changes. In one variable, $f(x + \delta) \approx f(x) + f'(x)\delta + \tfrac12 f''(x)\delta^2$. The multivariable version uses the **[[hessian|Hessian]]** $H = \nabla^2 L(\theta)$ (survey Eq. 3):

$$L(\theta + \Delta) \approx L(\theta) + g^\top\Delta + \tfrac12\Delta^\top H\Delta, \qquad H = \begin{bmatrix} L_{xx} & L_{xy}\\ L_{yx} & L_{yy}\end{bmatrix}.$$

The diagonal entries measure curvature along single coordinates; the off-diagonal entries measure how moving one coordinate changes the slope in the other. With $H = \begin{bmatrix}2 & 3\\ 3 & 5\end{bmatrix}$ and $\Delta = (a, b)^\top$,

$$\Delta^\top H\Delta = \begin{bmatrix} a & b\end{bmatrix}\begin{bmatrix} 2a + 3b\\ 3a + 5b\end{bmatrix} = 2a^2 + 6ab + 5b^2,$$

and the cross term $6ab$ is exactly that coupling.

**What do the eigenvalues mean?** Take a unit eigenvector, $Hv = \lambda v$ with $\|v\|_2 = 1$, and move along it, $\Delta = av$:

$$\tfrac12\Delta^\top H\Delta = \tfrac12 a^2\, v^\top H v = \tfrac12 a^2\, v^\top(\lambda v) = \tfrac12\lambda a^2.$$

So $\lambda$ is the curvature along $v$: upward if $\lambda > 0$, downward if $\lambda < 0$, nearly flat if $\lambda \approx 0$.

@@viz quad@@

## 6. Why this matters at stationary points

At a **stationary point** the gradient is zero, and the model reduces to $L(\theta + \Delta) - L(\theta) \approx \tfrac12\Delta^\top H\Delta$. If $H$ has a negative eigenvalue, stepping along its eigenvector gives $\tfrac12\lambda a^2 < 0$: a nearby downhill direction, so the point cannot be a local minimum. The converse is weaker: $H \succeq 0$ (no negative eigenvalues) is necessary for a local minimum but not sufficient, because higher-order terms can still decide.

The survey's Fig. 2 shows the local shapes the rest of the survey keeps returning to.

@@survey 2@@

:::key What you should remember
Eq. (1) defines the target. The gradient answers "which way is downhill right now?"; the Hessian answers "how is the slope changing, and in which directions?". With ordinary Euclidean length, $-g$ is the steepest first-order descent direction. Both Taylor models are local, which is why the next chapter needs one more assumption.
:::

:::problem The next step
The quadratic model says how the slope changes. Chapter 3 turns this into a guaranteed bound on how far a gradient step can safely go.
:::
