## 1. The worst case versus the cases we train

Stop 10 ended with a puzzle. Nothing so far guarantees that the minima GD finds are *good*: a non-convex function can have local minima far above its global one (stop 1), and finding a global minimum is NP-hard in general (stop 12). Yet networks with millions of parameters are trained to near-zero training loss every day, from random starting points, with plain gradient methods. **What is special about their landscapes?**

The honest answer is that there is no single theorem. There is a collection of results, each for a model simpler than a real network, that together explain a good part of the picture:

1. symmetries make minima come in many equivalent copies, joined by saddles (section 2);
2. some problems provably have no bad local minima at all (section 3), though small networks can (section 4);
3. when a landscape satisfies a gradient-domination inequality, GD converges fast without any convexity (section 5);
4. very wide networks behave like that near their initialization, and GD provably reaches zero training loss (sections 6–8).

Throughout, keep the stop 5 vocabulary in mind: a [[critical-point|critical point]] is a minimum, a maximum or a [[saddle]], and a [[strict-saddle|strict saddle]] has a direction of negative curvature that GD escapes along (stop 6).

## 2. Symmetry: every minimum comes in copies

Take a network with one hidden layer and swap two of its hidden neurons, together with their incoming and outgoing weights. The network computes exactly the same function, so the loss is the same. This **[[symmetry]]** means that every minimum has copies: with $k$ hidden neurons in a layer, up to $k!$ (k factorial: $k \cdot (k-1) \cdots 1$) of them, all equally good.

Copies of a minimum sit at different points of parameter space, and the loss has to rise somewhere on the way from one copy to another. The points halfway between them are typically saddles. The smallest example has two hidden units with weights $w_1, w_2$ and no output weights:

$$\mathcal{L}(w_1, w_2) = \frac{1}{9}\sum_{i=1}^{9} \Big(\tanh(w_1 x_i) + \tanh(w_2 x_i) - y_i\Big)^2 ,$$

with nine inputs $x_i$ between $-2$ and $2$ and targets made by units $1.6$ and $0.4$. The swap is the reflection $(w_1, w_2) \mapsto (w_2, w_1)$ across the diagonal. In the square $[-0.5, 2.5]^2$ there are exactly three critical points: the two global minima $(1.6, 0.4)$ and $(0.4, 1.6)$, and a saddle on the diagonal at $(0.736, 0.736)$ with loss $0.0118$, whose Hessian eigenvalues are $-0.106$ (across the diagonal) and $0.775$ (along it).

:::viz mirror
:::

The diagonal is special in a second way. If the two units start identical, they receive identical gradients and stay identical forever, so GD started on the diagonal can never leave it and ends at the saddle. Zhang, Saxe & Latham 2026 prove such statements in general (their Theorem 3: identical units stay identical, and similar *invariant manifolds* exist for zero or proportional units), and show that the critical points of a narrower network reappear inside every wider one (Theorem 1). Symmetry is a large source of saddles. Off the diagonal, the smallest offset decides the side: in the visual, $d = \pm 0.001$ already sends the violet ball to one minimum or the other, and the 24 random starts split 12 and 12.

The other common symmetry is rescaling, met in stop 10: in a ReLU network, multiplying one layer's weights by $c > 0$ and dividing the next layer's by $c$ changes nothing. It turns isolated minima into continuous curves of minima, like $uv = 1$, with a zero Hessian eigenvalue along the curve. So in networks, minima are rarely isolated points with positive definite Hessians; they are families.

## 3. Landscapes with no bad minima

A [[spurious-min|spurious local minimum]] is a local minimum that is not global. The best possible situation for a local method is a landscape without them.

:::def Benign landscape
A landscape is **[[benign]]** if every local minimum is a global minimum and every other critical point is a strict saddle.
:::

On a benign landscape, the results of stop 6 finish the job: GD from a random start avoids strict saddles with probability one (Lee et al. 2016, Theorem 4), so whenever it converges, it converges to a global minimum. Several model problems are proved to be benign.

- **Linear networks.** Baldi & Hornik 1989: a linear network with one hidden layer and squared loss has, up to the symmetries above, a single local minimum, which is global (the projection onto the top principal components of the data); all other critical points are saddles. Kawaguchi 2016 (Theorem 2.3) extends "every local minimum is global, every other critical point is a saddle" to deep linear networks of any depth and widths, under rank assumptions on the data, and shows that with two or more hidden layers some saddles have no negative Hessian eigenvalue (Corollary 2.4), so they are not strict. Laurent & von Brecht 2018 (Theorem 1) prove that all local minima are global for deep linear networks with any convex differentiable loss, provided no hidden layer is narrower than the smaller of the input and output layers.
- **Matrix problems.** Recovering a low-rank matrix from linear measurements (*matrix sensing*) by fitting a factored form $UU^\top$ is non-convex. Bhojanapalli, Neyshabur & Srebro 2016 prove that, if the measurements satisfy a restricted isometry property (RIP: they nearly preserve the norm of low-rank matrices, which random Gaussian measurements do once there are enough of them), every local minimum is global in the noiseless case and every other critical point is a strict saddle. Ge, Jin & Zheng 2017 prove the same for matrix sensing, matrix completion and robust PCA in one framework.

The smallest matrix problem fits a $2 \times 2$ matrix $M = \operatorname{diag}(1, s)$ by $uu^\top$ with $u \in \mathbb{R}^2$:

$$f(u) = \tfrac12 \lVert u u^\top - M \rVert_F^2 ,$$

where $\lVert A \rVert_F$ (the *Frobenius norm*) is the square root of the sum of the squared entries. Its critical points are the origin, a maximum (Hessian $-2M$); the two minima $\pm(1, 0)$, both global with loss $s^2/2$ (Hessian eigenvalues $2(1 - s)$ and $4$); and the two strict saddles $\pm(0, \sqrt{s})$, with eigenvalues $2(s - 1) < 0$ and $4s$. For $s = 0.5$: loss $0.125$ at the minima, $0.5$ at the saddles, saddle eigenvalues $-1$ and $2$.

:::viz benign
:::

For every $s$ on the slider, all 24 random starts reach a global minimum (13 and 11 on the two sides at $s = 0.5$). The one start placed exactly on the vertical axis, the saddle's stable set (stop 6), ends at the saddle. That set has measure zero, so a random start never lands on it.

## 4. Bad minima exist

Benign results are for linear networks and matrix problems. Real networks have nonlinear activations, and then spurious minima can appear.

The smallest example (Sun et al. 2020, their Fig. 2) is one neuron $v \cdot \varphi(w x)$ fitting one example $(x, y) = (1, 1)$, with loss $F(v, w) = \big(1 - v\,\varphi(w)\big)^2$. With the activation $\varphi(z) = -(z - 1)^2/8$, the line $w = 1$ is a trap: there $\varphi(1) = 0$ is the *maximum* of $\varphi$, so for $v > 0$ any change of $w$ makes $v\,\varphi(w)$ negative and the loss larger than $1$, and changing $v$ alone does nothing. Every point $(v, 1)$ with $v > 0$ is a local minimum with loss $1$ (Hessian eigenvalues $0$ and $v/2$), while the global minimum is $0$. Their Proposition IV.1 gives the rule for one neuron: there are no spurious minima exactly when no zero of $\varphi$ is a local maximum or minimum of $\varphi$. For $\tanh$, whose only zero is not an extremum, there are none.

:::viz neuron
:::

Of the 30 random starts, 21 reach zero loss and 9 get stuck at loss 1 with the quadratic activation; with $\tanh$, all 30 reach zero. From a $41 \times 41$ grid of starts, 30.4% get stuck.

For real ReLU networks, Safran & Shamir 2018 study a two-layer *teacher–student* problem: a student network with $n$ ReLU units learns a teacher with $k$ units, Gaussian inputs, squared loss. Their Theorem 1 proves (with a computer-assisted certificate) that spurious local minima exist for $k = n$ from 6 to 20, and for $n = k + 1$ with $10 \le k \le 19$. They also ran GD 1000 times per size from random initialization and counted how often it stopped at a spurious minimum (their Table 1, $k = n$):

| teacher = student width | 6 | 7 | 8 | 9 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| runs ending at a spurious minimum | 0.3% | 5.5% | 12.6% | 21.8% | 34.6% | 58.5% | 80.3% | 94% |

With two or more extra student units ($n \ge k + 2$) their runs found no spurious minima for $k$ up to 20, an observation, not a theorem. This is the pattern that the next sections make precise: **extra width removes bad minima from the path of GD**.

:::warn What the benign results do not say
"All local minima are global" is proved for linear networks, matrix problems and special architectures, each under assumptions (rank conditions, RIP, convex differentiable losses). It is false for small ReLU networks. Even on benign landscapes, the results say nothing about *how long* GD takes; that is the job of the next section.
:::

## 5. The PL condition: convergence without convexity

Stop 3 showed that GD finds a point with a small gradient. Stop 4 showed that, on a strongly convex function, it converges geometrically. There is an inequality between the two that gives the fast rate without convexity.

:::def Polyak–Łojasiewicz (PL) condition
$f$ satisfies the **[[pl|PL condition]]** with constant $\mu > 0$ if, at every point,

$$\tfrac12\, \lVert \nabla f(x) \rVert^2 \;\ge\; \mu\, \big(f(x) - f^\star\big).$$

In words: wherever the loss is above its minimum, the gradient is not small. A small gradient forces a small loss gap.
:::

Two consequences follow at once. First, at a critical point $\nabla f = 0$, so $f = f^\star$: **every critical point is a global minimum**, with no saddles and no spurious minima. Second, combine PL with the descent lemma of stop 3, $f(x_{t+1}) \le f(x_t) - \eta(1 - L\eta/2)\lVert \nabla f(x_t) \rVert^2$:

$$f(x_{t+1}) - f^\star \;\le\; \Big(1 - 2\mu\eta\big(1 - \tfrac{L\eta}{2}\big)\Big)\big(f(x_t) - f^\star\big) \qquad \text{for } 0 < \eta < 2/L .$$

With $\eta = 1/L$ the factor is $1 - \mu/L$, the rate GD has on a $\mu$-strongly convex function (stop 4). Every strongly convex function satisfies PL with the same $\mu$, but PL functions need not be convex. The condition is named after Polyak and Łojasiewicz, who introduced it in 1963.

The function $f(x) = x^2 + 3\sin^2 x$ is not convex: $f''(x) = 2 + 6\cos 2x$ ranges from $-4$ to $8$. But its only critical point is $x = 0$ ($\lvert f' \rvert \ge 1.54$ wherever $\lvert x \rvert > 0.5$), and the ratio $\tfrac12 f'^2 / f$ never drops below $\mu = 0.1755$ (reached at $\lvert x \rvert = 2.20$; it tends to 8 near the minimum and to 2 far away). With $L = 8$ and $\eta = 1/L$, PL guarantees a factor $0.978$ per step.

:::viz pl
:::

From $x_0 = 5.5$, where $f = 31.7$, the guarantee reaches a gap of $10^{-10}$ by step 1194; the real run gets there in 11 steps. As in stop 3, the guarantee is a worst case.

## 6. Wide networks: GD provably reaches zero loss

The strongest results about real (nonlinear) networks are about **[[overparameterization]]**: networks much wider than the number of training examples $n$. To see the mechanism, write the predictions on the training set as a vector $f(\theta) \in \mathbb{R}^n$, the residual $r = f(\theta) - y$, and the loss $\tfrac12 \lVert r \rVert^2$. The gradient is $J^\top r$, where $J$ is the $n \times d$ [[jacobian|Jacobian]] of the predictions with respect to the parameters (stop 2). One GD step changes the predictions by, to first order,

$$f(\theta_{k+1}) - f(\theta_k) \;\approx\; J\,(\theta_{k+1} - \theta_k) \;=\; -\eta\, J J^\top r_k , \qquad \text{so} \qquad r_{k+1} \approx (I - \eta\, G_k)\, r_k ,$$

with the $n \times n$ **Gram matrix** $G = J J^\top$. Its entries $G_{ij} = \nabla_\theta f(x_i) \cdot \nabla_\theta f(x_j)$ measure how much a step that helps example $i$ also moves example $j$. If $G$ stayed fixed with smallest eigenvalue $\lambda_0 > 0$, every component of the residual would shrink geometrically, exactly like GD on a quadratic (stop 4). Equivalently, $\lVert \nabla \mathcal{L} \rVert^2 = r^\top G\, r \ge 2\lambda_0 \cdot \tfrac12\lVert r \rVert^2$: the loss satisfies the PL condition of section 5 with $\mu = \lambda_0$.

The question is whether $G$ stays put during training. For wide networks it does: each parameter has to move only a little to change the predictions, so the Jacobian barely changes. This is what the theorems prove.

:::theorem Du, Lee, Li, Wang & Zhai 2019, Theorem 5.1 (deep fully connected networks)
Squared loss, $n$ unit-norm inputs, smooth non-polynomial activations, Gaussian initialization, full-batch GD. If every layer has width $m = 2^{O(H)} \cdot \mathrm{poly}(n, 1/\lambda_0, \log(Hn/\delta))$ for depth $H$, and the step is $\eta = O\big(\lambda_0 / (n^2 2^{O(H)})\big)$, then with probability at least $1 - \delta$

$$\mathcal{L}(\theta_k) \;\le\; \Big(1 - \frac{\eta \lambda_0}{2}\Big)^k \mathcal{L}(\theta_0),$$

where $\lambda_0 > 0$ is the smallest eigenvalue of a limiting Gram matrix. Their Theorem 6.1 gives the same rate for residual networks with width polynomial in $n$ and $H$ (the exponential factor for plain networks comes from their analysis, not necessarily from reality).
:::

Allen-Zhu, Li & Song 2019 (Theorem 1) prove the same for deep ReLU networks: with width $m \ge \mathrm{poly}(n, L, \delta^{-1}) \cdot d$, where $\delta$ is the minimum distance between training inputs, GD reaches loss $\varepsilon$ in $\mathrm{poly}(n, L)/\delta^2 \cdot \log(1/\varepsilon)$ steps, and SGD as well (Theorem 2). Their key lemma (Theorem 3) is a PL-type inequality that holds near the initialization: $\lVert \nabla F \rVert^2 \ge \Omega\big(\delta m / (d n^2)\big) \cdot F$. Zou & Gu 2019 reduce the required width from $n^{24}$ to $n^8$ (times other factors). These are worst-case widths, far beyond practice; the point is the mechanism.

:::viz wide
:::

The visual trains the same small network at five widths. Every width from 16 up reaches a loss below $10^{-6}$ in 1000 steps; width 4 is still at $0.064$. The relative distance the weights travel falls from $4.09$ (width 4) through $0.838$ (64) to $0.174$ (1024), roughly like $1/\sqrt{m}$. The smallest eigenvalue of $G$ changes by 96.5% during training at width 4 and by 0.9% at width 1024. Every run stays below Du et al.'s rate $(1 - \eta\lambda_0/2)^k$, computed with the network's own $\lambda_0$; at width 1024, $\lambda_0 = 0.0085$, so that rate is slow and the real run is far faster, as in the PL example.

## 7. Lazy training and the neural tangent kernel

When $G$ barely changes, the network is effectively its own first-order Taylor expansion around the initialization $\theta_0$:

$$f(x; \theta) \;\approx\; f(x; \theta_0) + \nabla_\theta f(x; \theta_0) \cdot (\theta - \theta_0).$$

This is *linear in $\theta$*, so training it with squared loss is a convex problem. The kernel that appears,

$$K(x, x') = \nabla_\theta f(x; \theta_0) \cdot \nabla_\theta f(x'; \theta_0),$$

is the **[[ntk|neural tangent kernel]]** (NTK); $G$ above is $K$ evaluated on the training inputs. Arora, Du, Hu, Li & Wang 2019 (Theorem 4.1) make the picture exact for wide two-layer ReLU networks: the residual after $k$ steps is, up to a small error, $\sqrt{\sum_i (1 - \eta\lambda_i)^{2k} (v_i^\top y)^2}$, where $\lambda_i, v_i$ are the eigenvalues and eigenvectors of the limiting kernel matrix. Labels aligned with the top eigenvectors are learned fast, others slowly. Their Theorem 5.1 bounds the test error by a quantity that does not grow with the width.

Chizat, Oyallon & Bach 2019 show that this behaviour, which they call **[[lazy-training|lazy training]]**, is not special to networks: any model scaled by a large factor $\alpha$ (or initialized at a large scale) trains while its parameters move by only $O(1/\alpha)$ (Theorem 2.2), staying close to its linearization. Their experiments find that lazy training *generalizes worse* than the non-lazy regime in two-layer networks and convolutional networks on CIFAR-10. And when the model is too small to fit the data (under-parameterized), the lazy limit only reaches a local minimum (Theorem 2.5).

:::warn Lazy training explains optimization, not deep learning
In the NTK regime the features computed by the hidden layers essentially do not change: the network is a fixed kernel method. Real networks learn features, and kernel methods provably cannot do everything networks do: Li, Ma & Zhang 2020 (Theorem 1.2) show a target that every polynomial-size kernel method, NTK included, fits with error $\Omega(1/d)$, while a two-layer network trained by a variant of GD fits it far better.
:::

## 8. Beyond lazy: the mean-field picture

A second limit keeps the features moving. Write a two-layer network with $m$ units as an *average* over its units, $f(x) = \frac{1}{m} \sum_{j=1}^{m} \phi(x; \theta_j)$ (note $1/m$, not the $1/\sqrt{m}$ of the NTK scaling). Then each unit $\theta_j$ is a particle, the whole network is a cloud of particles, and as $m \to \infty$ GD becomes a flow of a probability distribution over parameter space. In this **[[mean-field|mean-field]]** limit, the loss is a *convex* function of the distribution, even though it is non-convex in the individual weights.

- Chizat & Bach 2018 (Theorems 3.3 and 3.5) prove that if the initial distribution of the units is spread over all directions (a condition on its support) and the flow converges, its limit is a global minimizer. The result is qualitative: it gives no rate and no finite width.
- Mei, Misiakiewicz & Montanari 2019 (Theorem 1) prove that SGD on a finite network stays close to the mean-field flow, with an error that shrinks with the number of units at a rate that does not depend on the input dimension.

Unlike the lazy regime, the units travel far from where they started: the network learns features. The price is that the theory is mostly about two-layer networks and about limits, not rates.

## 9. Architecture, and what landscape pictures show

Some of the help comes from the architecture rather than the width. He, Zhang, Ren & Sun 2016 observed a *degradation problem*: deeper plain networks reached higher *training* error than shallower ones, so the problem was optimization, not overfitting (and with batch normalization, the authors did not attribute it to vanishing gradients). Adding identity shortcuts, $y = F(x) + x$, fixed it: on ImageNet (top-1 error, their Table 2) the 34-layer plain network had 28.54% against 27.94% for 18 layers, while the 34-layer residual network had 25.03% against 27.88%.

Li, Xu, Taylor, Studer & Goldstein 2018 drew 2-D slices of real network losses, along two random directions rescaled filter by filter (*filter normalization*, which removes the rescaling symmetry of section 2 from the picture). Networks without shortcuts become chaotic and non-convex-looking as depth grows, and their test error grows with depth (CIFAR-10: 8.18%, 13.31%, 16.44% at 20, 56 and 110 layers), while residual networks keep smooth, nearly convex-looking slices and improve (7.37%, 5.89%, 5.79%). Goodfellow, Vinyals & Saxe 2015 looked at the loss along the straight line from the initialization to the trained solution and found that it often decreases monotonically, with no visible barrier; the paper itself shows exceptions, and a barrier between two different solutions.

:::warn A slice is not the landscape
A 2-D slice of a function of millions of variables can look convex while the function has saddles and negative curvature in other directions (Li et al. say so themselves). These pictures are evidence about the region training visits, not proofs about the landscape.
:::

:::key The idea to keep
Network landscapes are non-convex, but not adversarial. Symmetry multiplies good minima and creates saddles that GD escapes; some model problems have no bad minima at all; and when the network is wide enough, the loss near the initialization satisfies a PL-type inequality, so GD reaches zero training loss at a geometric rate. None of this covers every network, and the strongest theorems need widths far beyond practice.
:::

:::atlas Where this shows up in the Atlas
- Symmetry and saddles: [Zhang, Saxe & Latham 2026](paper.html?id=zhang2026saddle), [Wu, Simsek & Ged 2025](paper.html?id=wu2025loss).
- No bad minima: [Baldi & Hornik 1989](paper.html?id=baldi1989neural), [Kawaguchi 2016](paper.html?id=kawaguchi2016deep), [Laurent & von Brecht 2018](paper.html?id=laurent2018deep), [Bhojanapalli et al. 2016](paper.html?id=bhojanapalli2016global), [Ge, Jin & Zheng 2017](paper.html?id=ge2017no), [Nguyen & Hein 2017](paper.html?id=nguyen2017loss) (one layer with at least $N - 1$ units for $N$ examples, under conditions).
- Bad minima: [Safran & Shamir 2018](paper.html?id=safran2018spurious), [Sun et al. 2020](paper.html?id=sun2020global).
- PL and gradient domination: [Danilova et al. 2022](paper.html?id=danilova2022recent) (which uses the example of section 5).
- Wide networks and NTK: [Du et al. 2019](paper.html?id=du2019gradient), [Allen-Zhu, Li & Song 2019](paper.html?id=allenzhu2019convergence), [Zou & Gu 2019](paper.html?id=zou2019improved), [Arora et al. 2019](paper.html?id=arora2019fine), [Chizat, Oyallon & Bach 2019](paper.html?id=chizat2019lazy), [Li, Ma & Zhang 2020](paper.html?id=li2020learning).
- Mean field: [Chizat & Bach 2018](paper.html?id=chizat2018global), [Mei, Misiakiewicz & Montanari 2019](paper.html?id=mei2019mean).
- Architecture and pictures: [He et al. 2016](paper.html?id=he2016deep), [Li et al. 2018](paper.html?id=li2018visualizing), [Goodfellow, Vinyals & Saxe 2015](paper.html?id=goodfellow2015qualitatively).
:::

:::problem The next problem
The results of this stop are *upper bounds*: under assumptions, a method needs at most so many steps. Every stop has quoted rates, $\varepsilon^{-2}$, $\varepsilon^{-7/4}$, $\varepsilon^{-4}$. **What is the best any algorithm could possibly do, and how close do the methods in the Atlas come?**
:::
