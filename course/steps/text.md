## 1. Gradient descent

Stop 2 gave us the direction: $-\nabla f(x)$ is the steepest way down. Repeating "take a step that way" gives the most important algorithm in this whole course, **[[gd|gradient descent]]** (GD):

$$x_{t+1} \;=\; x_t - \eta\, \nabla f(x_t), \qquad t = 0, 1, 2, \dots$$

Here $x_t$ is the point after $t$ steps (an **iteration** is one step), $x_0$ is where we start, and $\eta > 0$ is the **[[step-size|step size]]**, called the **learning rate** in machine learning. Everything else in the algorithm is fixed by the landscape, so $\eta$ is the one choice we have to make. It turns out to matter enormously.

## 2. The whole story in one parabola

Take the simplest valley there is, a parabola in one dimension:

$$f(x) = \tfrac{a}{2}\, x^2, \qquad f'(x) = a\,x, \qquad a > 0.$$

The number $a$ is the **curvature**: how sharply the valley bends (it is $f''(x)$, the second derivative; stop 4). One GD step gives

$$x_{t+1} = x_t - \eta\, a\, x_t = (1 - \eta a)\, x_t, \qquad \text{so} \qquad x_t = (1 - \eta a)^t\, x_0 .$$

Every step multiplies the distance to the minimum by the same factor $1 - \eta a$. So everything depends on the product $\eta a$:

| $\eta a$ | factor $1 - \eta a$ | what GD does |
|---|---|---|
| between 0 and 1 | between 0 and 1 | slides down one side, slowly if $\eta a$ is tiny |
| exactly 1 | 0 | lands on the minimum in one step |
| between 1 and 2 | between −1 and 0 | jumps across the valley each step, but converges |
| exactly 2 | −1 | bounces between $x_0$ and $-x_0$ forever |
| above 2 | below −1 | jumps farther out every step: **diverges** |

With $a = 4$ and $x_0 = 1$: $\eta = 0.1$ gives factor $0.6$ and $x_{10} \approx 0.006$; $\eta = 0.4$ gives factor $-0.6$, the same size but with the sign flipping each step; $\eta = 0.6$ gives factor $-1.4$ and $x_{10} \approx 28.9$. The rule to remember:

$$\boxed{\;\text{GD on a valley of curvature } a \text{ is stable only if } \eta < 2/a.\;}$$

Too small a step wastes iterations; too large a step makes the method explode. The best step here is $\eta = 1/a$, but it depends on $a$, which we do not know in advance.

:::viz parabola
:::

## 3. Real landscapes bend differently in different places

In a parabola the curvature is the same everywhere. A real loss bends gently in some places and sharply in others, and in $d$ dimensions also differently in different directions. The same step can be safe in one valley and explosive in another:

:::viz bowl
:::

To say anything general about GD we need *some* guarantee about how sharply $f$ can bend. The standard guarantee is a Lipschitz condition.

:::def Lipschitz continuity
A function $g$ is **[[lipschitz|Lipschitz]]** with constant $K$ if, for all points $u, v$,

$$\lVert g(u) - g(v) \rVert \;\le\; K\, \lVert u - v \rVert .$$

In words: the output can never change faster than $K$ times the input. For a function of one variable, this says that no secant slope is steeper than $K$: the graph stays inside a double cone of slopes $\pm K$ placed at any of its points.
:::

:::viz lipschitz
:::

Optimization papers apply this condition not to $f$ but to its **gradient**:

:::def L-smooth (gradient Lipschitz)
$f$ is **[[l-smooth|$L$-smooth]]** if its gradient is Lipschitz with constant $L$:

$$\lVert \nabla f(x) - \nabla f(y) \rVert \;\le\; L\, \lVert x - y \rVert \quad \text{for all } x, y.$$

The slope cannot change faster than $L$ times the distance moved. For a twice-differentiable $f$ this is the same as saying that **every curvature lies between $-L$ and $L$**: in one dimension, $\lvert f''(x) \rvert \le L$ everywhere; in $d$ dimensions, the same for every eigenvalue of the Hessian (stop 4).
:::

For the parabola, $L = a$. For $f(x) = 0.4\cos 2x + 0.1x^2$ (used below), $f''(x) = -1.6\cos 2x + 0.2$ ranges from $-1.4$ to $1.8$, so $f$ is $L$-smooth with $L = 1.8$. "Smoothness" in the papers almost always means this condition, and $L$ almost always means this constant.

## 4. The descent lemma: why $\eta = 1/L$ is safe

Smoothness turns the vague "the tangent is good near $x$" into a hard inequality. In one dimension, the gap between $f$ and its tangent line at $x$ is

$$f(y) - f(x) - f'(x)(y - x) \;=\; \int_x^y \big(f'(s) - f'(x)\big)\, ds \;\le\; \int_x^y L\,\lvert s - x \rvert\, ds \;=\; \frac{L}{2}(y - x)^2$$

(for $y > x$; the case $y < x$ is the same), because the slope $f'(s)$ can differ from $f'(x)$ by at most $L \lvert s - x\rvert$. The same argument works in $d$ dimensions and gives the **quadratic upper bound**:

$$f(y) \;\le\; f(x) + \nabla f(x) \cdot (y - x) + \frac{L}{2}\, \lVert y - x \rVert^2 .$$

The right-hand side is a parabola that touches $f$ at $x$ and stays above it everywhere. Now put in the GD step $y = x - \eta \nabla f(x)$:

$$f(x_{t+1}) \;\le\; f(x_t) - \eta \Big(1 - \frac{L \eta}{2}\Big) \lVert \nabla f(x_t) \rVert^2 .$$

This is the **[[descent-lemma|descent lemma]]**. For any $0 < \eta < 2/L$ the bracket is positive, so **every step lowers $f$** unless the gradient is zero. The decrease is largest at $\eta = 1/L$, which gives

$$f(x_{t+1}) \;\le\; f(x_t) - \frac{1}{2L}\, \lVert \nabla f(x_t) \rVert^2 .$$

That is exactly the step that lands at the bottom of the upper-bound parabola. Note the two thresholds again: $2/L$ is the edge of stability for the sharpest valley (section 2), and $1/L$ is the safe, standard choice.

:::viz upper
:::

## 5. How many steps? ε-stationary points and big-O

GD cannot promise to find a minimum of a non-convex function (stop 5 shows why), but the descent lemma does promise that it finds a point where the gradient is small. Add the inequality up over the first $T$ steps. Each step lowers $f$ by at least $\frac{1}{2L}\lVert\nabla f(x_t)\rVert^2$, and $f$ can never go below its lowest value $f^\star$, so

$$\frac{1}{2L} \sum_{t=0}^{T-1} \lVert \nabla f(x_t) \rVert^2 \;\le\; f(x_0) - f^\star \quad\Longrightarrow\quad \min_{t < T}\; \lVert \nabla f(x_t) \rVert \;\le\; \sqrt{\frac{2L\,\big(f(x_0) - f^\star\big)}{T}} .$$

A point with $\lVert \nabla f(x) \rVert \le \varepsilon$ is called an **[[eps-stationary|ε-stationary point]]** (ε, epsilon, is a small tolerance such as $10^{-3}$). The bound reaches $\varepsilon$ once

$$T \;\ge\; \frac{2L\,\big(f(x_0) - f^\star\big)}{\varepsilon^2}.$$

With $L = 1$, $f(x_0) - f^\star = 1$ and $\varepsilon = 0.01$, that is $T = 20{,}000$ steps. Halving $\varepsilon$ quadruples the guarantee.

Papers summarize statements like this with **[[big-o|big-O notation]]**: GD needs $O(\varepsilon^{-2})$ iterations. $O(g)$ means "at most a constant times $g$", where the constant may depend on $L$ and $f(x_0) - f^\star$ but not on $\varepsilon$. The count of steps (or of gradient evaluations) needed to reach a tolerance is the method's **[[iteration-complexity|iteration complexity]]**, and comparing these counts is how the papers rank algorithms. Two things to keep in mind:

1. This is a **worst-case guarantee**. On a particular function GD is often much faster; the visual shows the actual gradient norm far below the bound.
2. A small gradient is not the same as a minimum. Flat points include saddles and maxima. Telling them apart is the job of curvature (stops 4 and 5).

:::viz rate
:::

## 6. Why one step size is not enough

Section 2 found the best step $1/a$ for one valley. In $d$ dimensions there is a curvature *per direction*, and a single $\eta$ has to respect the sharpest one ($\eta < 2/L$) while crawling along the flattest. The bowl visual above already shows the symptom: with a step that is safe for the steep direction, GD zig-zags across the valley and inches along its floor. The rest of the course is, to a large extent, a list of answers to this one problem:

- understand the curvature itself (stop 4),
- give the method **memory**, so the zig-zags cancel and the slow direction builds up speed (momentum, stop 7),
- give **each direction its own step size** (AdaGrad, Adam, stop 9),
- change the step size over time (schedules and warmup, stops 8 and 10).

:::atlas Where this shows up in the Atlas
- [Nesterov 2004](paper.html?id=nesterov2004introductory) and [Bottou, Curtis & Nocedal 2018](paper.html?id=bottou2018optimization) build their analyses on $L$-smoothness and the descent lemma.
- [Zhang et al. 2020](paper.html?id=zhang2020gradient) argue that real network losses are not globally $L$-smooth: the curvature grows with the gradient, an "$(L_0, L_1)$" condition under which clipping the gradient helps (stop 9).
- [Cohen et al. 2021](paper.html?id=cohen2021gradient) observe that in full-batch training the sharpest curvature rises until it reaches about $2/\eta$ and then hovers there, the **edge of stability** of section 2 (stop 10).
- [Carmon et al. 2020](paper.html?id=carmon2020lower) prove that no first-order method can beat the $\varepsilon^{-2}$ rate of section 5 on all $L$-smooth functions (stop 12).
:::

:::problem The next problem
One number $L$ hides how differently the landscape bends in different directions, and that difference is what makes GD zig-zag. **What does curvature look like in many dimensions?**
:::
