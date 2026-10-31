# All Local Minima Are Global

:::minute The paper in one minute
Can adding many *linear* layers create bad local minima even when the loss on the combined linear map is convex? **Under two assumptions, no: every local minimum in the separate layer weights is globally optimal.** This is a result about *landscape geometry*, not a guarantee that gradient descent trains quickly.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. For a convex function, a zero gradient is enough

A differentiable function $f$ is **convex** when it lies above every tangent plane:

$$f(B) \ge f(A) + \langle \nabla f(A),\, B - A\rangle \quad \text{for all } A, B .$$

If $\nabla f(A^*) = 0$ the right side is just $f(A^*)$, so $f(B) \ge f(A^*)$ for every $B$: the critical point is a global minimum. Example: $f(A) = (A - 4)^2$ has $f'(A) = 2(A - 4)$, which vanishes only at $A = 4$, the global minimum. This one line is why the paper only needs to show $\nabla f(A^*) = 0$.

### 2. Writing the map as a product breaks convexity

Convexity means the loss at a midpoint is never above the average of the losses at the two ends. With the notes' example $F(w_1, w_2) = (w_2 w_1 - 4)^2$, the points $(2, 2)$ and $(-2, -2)$ both have loss $0$, but their midpoint $(0, 0)$ has loss $16$. So $F$ is not convex, even though $f$ is. The non-convexity comes only from the product, which also creates whole curves of equally good solutions ($w_2 w_1 = 4$).

### 3. A critical point of the product can be a saddle

The notes' §3 shows that $(0, 0)$ is stationary. Its Hessian tells what kind of point it is:

$$\nabla^2 F(0, 0) = \begin{bmatrix} 0 & -8 \\ -8 & 0 \end{bmatrix}, \qquad \text{eigenvalues } +8 \text{ and } -8 .$$

Mixed signs make it a saddle: moving along $w_1 = w_2$ lowers the loss, moving along $w_1 = -w_2$ raises it. The theorem says nothing against such saddles; it rules out bad *minima* only.

### 4. A thin layer limits which maps the product can reach

The product of a $3 \times 1$ and a $1 \times 3$ matrix always has rank at most $1$. If the input and output have dimension $3$ but a hidden layer has width $1$, the network can only represent rank-one maps, and the best rank-one map may be worse than the best map overall. The paper's width condition $d_k \ge \min(d_0, d_L)$ makes the narrowest layer the input or output layer, so the product can reach every $d_L \times d_0$ matrix and no such bottleneck exists.

### 5. At a kink, a linear cost can beat a linear gain

Differentiability matters because at a kink the loss can rise *linearly* in every direction. The paper's counterexample uses $f(x, y) = |x| + (1 - y)_+ - 1$, with $A = (x, y) = W_2 W_1$, $W_1$ a scalar and $W_2 \in \mathbb{R}^2$. At $W_1 = 0$, $W_2 = (1, 0)$ the loss is $0$, while the best value is $-1$. A small move such as $W_1 = \varepsilon$, $W_2 = (1, \delta)$ gives $f = |\varepsilon| - \varepsilon\delta$: the $|x|$ term costs $|\varepsilon|$, the gain is only $\varepsilon\delta$, so for small $\delta$ the loss goes up. Example: $\varepsilon = 0.1$, $\delta = 0.5$ gives $0.05 > 0$. The point is a bad local minimum, which a differentiable loss would not allow.

:::key Putting the five ideas together
With a convex loss, proving $\nabla f(A^*) = 0$ is enough for global optimality (idea 1). The product of layers makes the problem non-convex (idea 2) and creates saddles (idea 3), but the paper shows that at every *local minimum* of the layer weights the combined map still has zero gradient, provided no layer is a bottleneck (idea 4) and the loss is differentiable (idea 5). That is Theorem 3, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. First principles: what are the two optimization problems?

With one linear transformation, write $A$ for the combined map and minimize $f(A)$. A deep linear network replaces $A$ by a product $A = W_L W_{L-1} \cdots W_1$, so it minimizes

$$\boxed{F(W_1, \dots, W_L) = f(W_L \cdots W_1).}$$

**Why can $F$ be non-convex?** Even if $f$ is convex in $A$, multiplying separately trainable weights makes $F$ nonlinear in those weights. For one input and two scalar layers, with target 4:

$$f(A) = (A - 4)^2 \quad \text{(convex parabola)}, \qquad F(w_1, w_2) = (w_2 w_1 - 4)^2 \quad \text{(non-convex)}.$$

A **local minimum** means no *sufficiently small* change of the weights lowers the loss; a **global minimum** means *no change anywhere* can lower it. There may be many different optimal weight settings: $(2, 2)$ and $(1, 4)$ both give $w_2 w_1 = 4$ and zero loss.

## 2. Visual intuition: many valleys, but none is a bad valley

:::figure A ball can reach different valley bottoms; both have the same lowest loss.
<svg viewBox="0 0 340 150" role="img" aria-label="A loss curve over a schematic weight axis with two valley bottoms at the same lowest loss, both labelled global min, joined by a dashed line at that height; a ball sits in the left valley.">
<defs>
<marker id="ntLvA" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#8e99bd"/></marker>
</defs>
<line x1="24" y1="118" x2="24" y2="10" stroke="#8e99bd" stroke-width="1" marker-end="url(#ntLvA)"/>
<text x="24" y="7" fill="#8e99bd" font-size="9" text-anchor="middle">loss</text>
<line x1="24" y1="118" x2="326" y2="118" stroke="#8e99bd" stroke-width="1" marker-end="url(#ntLvA)"/>
<text x="326" y="144" fill="#8e99bd" font-size="9" text-anchor="end">weight configuration (schematic)</text>
<line x1="30" y1="110" x2="310" y2="110" stroke="#ffc857" stroke-width="1" stroke-dasharray="4 3"/>
<path d="M30.0 46.0 L33.5 53.8 L37.0 61.0 L40.5 67.5 L44.0 73.4 L47.5 78.8 L51.0 83.6 L54.5 88.0 L58.0 91.8 L61.5 95.2 L65.0 98.2 L68.5 100.8 L72.0 103.0 L75.5 104.8 L79.0 106.3 L82.5 107.5 L86.0 108.5 L89.5 109.2 L93.0 109.7 L96.5 109.9 L100.0 110.0 L103.5 109.9 L107.0 109.7 L110.5 109.4 L114.0 108.9 L117.5 108.4 L121.0 107.8 L124.5 107.2 L128.0 106.6 L131.5 105.9 L135.0 105.2 L138.5 104.6 L142.0 104.0 L145.5 103.4 L149.0 102.9 L152.5 102.4 L156.0 102.0 L159.5 101.6 L163.0 101.4 L166.5 101.2 L170.0 101.1 L173.5 101.1 L177.0 101.2 L180.5 101.4 L184.0 101.6 L187.5 102.0 L191.0 102.4 L194.5 102.9 L198.0 103.5 L201.5 104.1 L205.0 104.8 L208.5 105.4 L212.0 106.1 L215.5 106.8 L219.0 107.5 L222.5 108.2 L226.0 108.8 L229.5 109.3 L233.0 109.7 L236.5 109.9 L240.0 110.0 L243.5 109.9 L247.0 109.6 L250.5 109.0 L254.0 108.1 L257.5 106.8 L261.0 105.2 L264.5 103.2 L268.0 100.7 L271.5 97.6 L275.0 94.0 L278.5 89.8 L282.0 84.9 L285.5 79.3 L289.0 72.9 L292.5 65.6 L296.0 57.4 L299.5 48.2 L303.0 37.9 L306.5 26.6 L310.0 14.0" fill="none" stroke="#7cc4ff" stroke-width="2"/>
<text x="170" y="52" fill="#e8ecf8" font-size="9" text-anchor="middle">A ball can reach different valley bottoms;</text>
<text x="170" y="64" fill="#e8ecf8" font-size="9" text-anchor="middle">both have the same lowest loss.</text>
<circle cx="100.0" cy="104.5" r="5" fill="#ff7a59"/>
<circle cx="240.0" cy="110" r="3.5" fill="#5fe0b0"/>
<text x="100.0" y="130" fill="#5fe0b0" font-size="9" text-anchor="middle">global min</text>
<text x="240.0" y="130" fill="#5fe0b0" font-size="9" text-anchor="middle">global min</text>
</svg>
:::

**Important:** This sketch is conceptual, *not* a literal cross-section of the paper's high-dimensional surface. Saddles and flat regions can still exist: "no bad local minima" $\neq$ "optimization is easy."

## 3. The crucial theorem: why can they be sure?

**What needs proving?** It is *not* enough that each layer gradient vanishes. By the chain rule, $\partial F/\partial w_1 = w_2 f'(A)$ and $\partial F/\partial w_2 = w_1 f'(A)$; at $(0, 0)$ they both vanish although $f'(0) = -8$. That point is stationary but not a local minimum.

:::theorem The actual bridge established by the paper (Theorem 3)
For a *local minimum* $(W_1^*, \dots, W_L^*)$ of the factorized objective, the combined matrix $A^* = W_L^* \cdots W_1^*$ satisfies $\nabla f(A^*) = 0$, provided the loss is differentiable and the layers are sufficiently wide. Since $f$ is also convex, $\nabla f(A^*) = 0$ implies $f(A^*) \le f(A)$ for *all* $A$. Thus the original local minimum is global.

$$\boxed{\text{local minimum of } F \;\Longrightarrow\; \nabla f(A^*) = 0 \;\overset{f\ \text{convex}}{\Longrightarrow}\; \text{global minimum.}}$$
:::

## 4. What assumptions make this true?

**(i)** The outer loss $f$ is **convex and differentiable**. **(ii)** Every hidden-layer width $d_k$ satisfies $d_k \ge \min(d_0, d_L)$ (the thinnest layer is an input or output layer). This avoids a rank bottleneck that would restrict which combined maps can be represented. The paper constructs a *convex, Lipschitz but nondifferentiable* loss for which a bad local minimum exists—so differentiability really matters.

:::warn Limits / survey connection
The theorem does *not* cover arbitrary nonlinear networks, nonconvex outer losses, or narrow bottleneck architectures; nor does it promise fast convergence or escape from saddles. **RQ1:** non-convex parameterizations can have benign minima. **RQ4:** that claim depends strongly on loss smoothness and network width.
:::

:::ours Takeaway
Adding linear layers changes the shape of the optimization problem but, under the theorem's assumptions, does not create suboptimal local minima.
:::

> **Source:** Laurent & von Brecht, "Deep Linear Networks with Arbitrary Loss: All Local Minima Are Global," ICML (PMLR 80), 2018, Theorems 1–3. [proceedings.mlr.press/v80/laurent18a.html](https://proceedings.mlr.press/v80/laurent18a.html)
