## 1. The derivative: the slope at one point

Start with one parameter. We stand at $x$ and want to know whether $f$ goes up or down if we move a little. Take a small step $h$ and measure the change:

$$\frac{f(x + h) - f(x)}{h} \quad = \quad \frac{\text{rise}}{\text{run}} .$$

This is the slope of the straight line through the two points $\big(x, f(x)\big)$ and $\big(x + h, f(x + h)\big)$, a *secant*. As $h$ shrinks toward $0$, the secant turns into the line that just touches the curve at $x$, the *tangent*, and its slope settles to one number, the **[[derivative]]**:

$$f'(x) \;=\; \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}.$$

"$\lim_{h \to 0}$" (the **limit**) means "the value this ratio approaches as $h$ gets as close to $0$ as we like". For example, for $f(x) = x^2$:

$$\frac{(x + h)^2 - x^2}{h} = \frac{2xh + h^2}{h} = 2x + h \;\longrightarrow\; 2x, \qquad \text{so } f'(x) = 2x.$$

:::viz secant
:::

## 2. The sign of the slope says which way is down

If $f'(x) > 0$, the curve rises to the right, so a small move to the *left* lowers $f$. If $f'(x) < 0$, the curve falls to the right, so we move *right*. Both cases are one rule: move against the slope,

$$x_{\text{new}} = x - \eta\, f'(x),$$

where $\eta > 0$ (the Greek letter eta) is a small number that sets the step length. The size of $f'(x)$ matters too: a steep slope means a small move changes $f$ a lot. Where $f'(x) = 0$ the curve is flat, and the rule stops moving.

## 3. Many knobs: partial derivatives and the gradient

Now $x = (x_1, \dots, x_d)$ has $d$ coordinates. The simplest question we can ask is: what happens if we move **one** coordinate and freeze the others? That is the **[[partial-derivative|partial derivative]]**, written with a curly $\partial$:

$$\frac{\partial f}{\partial x_j}(x) \;=\; \lim_{h \to 0} \frac{f(x_1, \dots, x_j + h, \dots, x_d) - f(x)}{h}.$$

To compute it, treat every other coordinate as a constant and take the ordinary derivative. For $f(x, y) = x^2 + 3y^2$:

$$\frac{\partial f}{\partial x} = 2x, \qquad \frac{\partial f}{\partial y} = 6y.$$

Stack all $d$ partial derivatives into one vector and you have the **[[gradient]]**, written $\nabla f$ ("nabla f" or "grad f"):

$$\nabla f(x) \;=\; \left( \frac{\partial f}{\partial x_1}(x),\; \dots,\; \frac{\partial f}{\partial x_d}(x) \right). \qquad \text{Here } \nabla f(x, y) = (2x,\; 6y).$$

The gradient has as many entries as there are parameters: for a network with $d$ weights, $\nabla f$ is a list of $d$ numbers, one per weight, each saying how the loss reacts to nudging that weight.

## 4. The gradient points straight uphill

Partial derivatives only look along the coordinate axes. What about any other direction? Pick a direction as a **unit vector** $u$ (length $\lVert u \rVert = 1$). The rate of change of $f$ when moving from $x$ along $u$ is the **[[directional-derivative|directional derivative]]**, and it turns out to be built from the gradient:

$$D_u f(x) \;=\; \lim_{h \to 0} \frac{f(x + h u) - f(x)}{h} \;=\; \nabla f(x) \cdot u .$$

The dot is the **[[dot-product|dot product]]**: multiply matching entries and add, $a \cdot b = a_1 b_1 + \dots + a_d b_d$. Geometrically, $a \cdot b = \lVert a \rVert\,\lVert b \rVert \cos\varphi$, where $\varphi$ (phi) is the angle between the two vectors. So

$$D_u f(x) = \lVert \nabla f(x) \rVert \cos\varphi ,$$

and three facts fall out at once:

1. **Steepest ascent.** $\cos\varphi$ is largest ($= 1$) when $u$ points along $\nabla f$. The gradient is the direction in which $f$ climbs fastest, and $\lVert \nabla f \rVert$ is that fastest rate.
2. **Steepest descent.** $\cos\varphi = -1$ when $u$ points along $-\nabla f$. That is the direction in which $f$ falls fastest.
3. **Level directions.** $\cos\varphi = 0$ when $u$ is perpendicular to $\nabla f$: moving that way does not change $f$ at all, to first order.

A **[[contour]]** (or level set) is a curve along which $f$ has one constant value, like the height lines on a hiking map. Walking along a contour does not change $f$, so by fact 3 **the gradient is perpendicular to the contour** through every point.

:::viz compass
:::

## 5. The tangent plane: a first-order picture of $f$

Fact 1 says that, near $x$, $f$ changes like a tilted plane. Written as a formula, this is the **first-order [[taylor|Taylor approximation]]**:

$$f(x + \Delta) \;\approx\; f(x) + \nabla f(x) \cdot \Delta ,$$

where $\Delta$ (delta) is a small displacement vector. "First-order" means it keeps the terms that are linear in $\Delta$ and drops everything smaller, such as $\Delta_1^2$ or $\Delta_1 \Delta_2$. The error of the approximation shrinks like $\lVert \Delta \rVert^2$ as $\Delta \to 0$ (for functions with continuous second derivatives), so close to $x$ the plane is excellent and farther away it drifts. What decides *how fast* it drifts is the curvature of $f$, the subject of stop 4.

:::viz plane
:::

This approximation is the whole justification of gradient descent. Taking $\Delta = -\eta \nabla f(x)$ gives

$$f\big(x - \eta \nabla f(x)\big) \;\approx\; f(x) - \eta\, \lVert \nabla f(x) \rVert^2 ,$$

a decrease, as long as $\eta$ is small enough that the approximation holds.

## 6. Functions with many outputs: the Jacobian

A layer of a neural network maps a vector to a vector: $g : \mathbb{R}^n \to \mathbb{R}^m$, with outputs $g_1, \dots, g_m$. Each output has its own gradient. Stacking these gradients as rows gives the **[[jacobian|Jacobian]]** matrix, $m$ rows by $n$ columns:

$$J_g(x) = \begin{bmatrix} \dfrac{\partial g_1}{\partial x_1} & \cdots & \dfrac{\partial g_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial g_m}{\partial x_1} & \cdots & \dfrac{\partial g_m}{\partial x_n} \end{bmatrix}, \qquad (J_g)_{ij} = \frac{\partial g_i}{\partial x_j}.$$

It plays the role of the derivative for vector-valued maps: $g(x + \Delta) \approx g(x) + J_g(x)\,\Delta$. Up close, every smooth map looks like a matrix, and the matrix is the Jacobian. Its **determinant**, $\det J_g$, says how much the map scales small areas (volumes in higher dimensions) near $x$.

:::viz jacobian
:::

## 7. The chain rule: how the gradient flows through layers

A network is a chain of maps: input → layer 1 → layer 2 → … → loss. To get the gradient of the loss we need the derivative of a composition. In one dimension, for $f(x) = g\big(h(x)\big)$, the **[[chain-rule|chain rule]]** says

$$f'(x) = g'\big(h(x)\big)\; h'(x):$$

the rates of change multiply. For example, $f(w) = (2w - 1)^2$ is $g(u) = u^2$ applied to $u = h(w) = 2w - 1$, so $f'(w) = 2u \cdot 2 = 4(2w - 1)$; at $w = 1$ this is $4$.

With vectors, the rates of change are Jacobians and they multiply as matrices: $J_{g \circ h}(x) = J_g\big(h(x)\big)\, J_h(x)$. For a network with layers $h_1, \dots, h_k$ followed by a scalar loss, the gradient is a product of all the layers' Jacobians. Evaluating that product from the loss backwards is **[[backprop|backpropagation]]**. Its key property is cost: the whole gradient, all $d$ entries, costs only a small constant multiple of computing the loss once. That is why nearly every method in the Atlas uses gradients, and why methods that need second derivatives work hard to avoid forming them (stops 4 and 6).

:::atlas Where this shows up in the Atlas
- Almost every result in the survey is about **first-order methods**: algorithms that only ever ask for $f(x)$ and $\nabla f(x)$, such as gradient descent and SGD.
- [Saxe et al. 2014](paper.html?id=saxe2014exact) analyze deep *linear* networks, where the layer Jacobians are just the weight matrices and the chain rule can be followed by hand.
- [Allen-Zhu & Li 2018 (NEON2)](paper.html?id=allenzhu2018neon2) and [Xu et al. 2018](paper.html?id=xu2018first) obtain second-order information using gradients alone, the cheap object of section 7.
:::

:::problem The next problem
We now know **which way** is down: $-\nabla f(x)$. But section 5 warned that the first-order picture only holds for small steps. **How far should we step?**
:::
