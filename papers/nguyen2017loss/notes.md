# Why Width Can Make Bad Minima Exceptional

:::minute The paper in one minute
**Question:** Does a wide neural network still have suboptimal local minima, or does extra width make its loss geometry more favorable?

**Model/Method:** Fully connected networks with smooth analytic activations; analyze rank of hidden-layer features, backpropagation gradients, and non-degenerate Hessian blocks.

**Finding:** With a layer of at least $N - 1$ units for $N$ distinct training examples and suitable downstream rank and non-degeneracy conditions, the covered critical points—including the covered local minima—are globally optimal.

**Why we care:** Architecture can rule out *certain bad stationary points* without requiring a new optimizer. Width alone is not a universal no-bad-minima theorem.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. Full row rank means every target can be fitted

An $N \times m$ matrix $A$ has **full row rank** when its $N$ rows are linearly independent. Two equivalent facts follow. The system $A\beta = y$ has a solution for *every* target $y \in \mathbb{R}^N$. And $A^\top r = 0$ forces $r = 0$. Full row rank needs at least $N$ columns: with $n_k$ hidden features plus one bias column, that is $n_k + 1 \ge N$, the paper's width $n_k \ge N - 1$. Example: three examples and two features *without* a bias give a $3 \times 2$ matrix, rank at most $2$, so some target vectors cannot be fitted; adding the bias column makes the notes' $3 \times 3$ example solvable.

### 2. A stationary point of the loss sees the loss only through its residuals

The training loss is a sum of per-example losses, $\sum_i l(a_i)$, where $a_i$ is the error on example $i$. By the chain rule, every parameter gradient is built from the vector of derivatives $l'(a_i)$. The paper assumes that $l'(a) = 0$ only at a global minimum of $l$ (true for the squared loss $l(a) = a^2$, where $l'(a) = 2a$). So if the rank argument forces every $l'(a_i)$ to be zero, every example sits at the bottom of its own loss, and the total loss is globally minimal.

### 3. Analytic functions vanish only on thin sets

A polynomial that is not identically zero vanishes only on a lower-dimensional set: $p(w_1, w_2) = w_1 w_2 - 1$ is zero on a curve, which has zero area. The same holds for **analytic** functions (functions equal to their Taylor series, such as $\tanh$ or the sigmoid), and determinants built from them are analytic in the weights. Hence a single parameter setting with full feature rank (Lemma 4.3) implies that rank-deficient settings have measure zero (Lemma 4.4): a random choice of lower-layer weights has full feature rank with probability one.

### 4. A non-degenerate Hessian block lets a stationary point move smoothly

Suppose $\partial_u G(u, v) = 0$ at $(u_0, v_0)$ and the second derivative $\partial^2_{uu} G$ is invertible there. The **implicit function theorem** says that for every $v$ near $v_0$ there is a $u(v)$ near $u_0$ with $\partial_u G(u(v), v) = 0$, changing smoothly with $v$. Without invertibility this can fail. Example: $G(u, v) = u^3/3 - v u$ has $\partial_u G = u^2 - v$ and $\partial^2_{uu} G = 2u = 0$ at the origin. For $v = 0.01$ the stationary points are $u = \pm 0.1$, but for $v = -0.01$ there are none: the stationary point cannot be tracked. The paper uses the invertible block exactly to track upper-layer stationarity while it nudges lower layers.

### 5. Full column rank stops error information from being lost

A matrix $W$ with **full column rank** is one-to-one: $W z = 0$ only for $z = 0$. Backpropagation passes error signals backwards through the weight matrices, so a full-column-rank matrix cannot turn a nonzero error into zero. Example: $W = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ has full column rank, while $W = \begin{bmatrix} 1 & 2 \end{bmatrix}$ cannot (it sends $(2, -1)$ to $0$). A matrix can only have full column rank if it has at least as many rows as columns, which is why the notes' condition forces a pyramidal tail.

:::key Putting the five ideas together
If the features at a wide layer have full row rank (idea 1), a stationary readout must have zero residual derivatives, which means a global minimum (idea 2). Full rank holds for almost all lower-layer weights (idea 3). The proof moves a critical point slightly to such weights while keeping it stationary, using the non-degenerate Hessian block (idea 4), and full column rank above carries the error signal down to the wide layer (idea 5). That is Theorem 3.8, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Start with the question: why can extra width help?

A network can fit the training examples poorly even though its parameters are at a stationary point ($\nabla\Phi = 0$). The authors ask whether a sufficiently expressive hidden layer prevents this *at regular stationary points*. They focus on the **geometry of the training loss**, not on how many gradient-descent iterations are needed.

Let $N$ be the number of training samples, $n_k$ the width of hidden layer $k$, and $F_k \in \mathbb{R}^{N\times n_k}$ the matrix whose $i$th row is the hidden representation of example $i$. Append a bias column $\mathbf{1}$:

$$A = [F_k\ \mathbf{1}] \in \mathbb{R}^{N\times(n_k+1)}.$$

If $\mathrm{rank}(A) = N$, its rows are independent: there is enough linear freedom at that layer to distinguish all $N$ training examples.

## 2. Worked example: three examples, only two hidden units

This is a *teaching example of the feature-rank mechanism*, with a linear readout for transparent algebra. The paper proves a more general statement for its specified multilayer networks and activations.

Take inputs $x = -1, 0, 1$ and two tanh units, $h_1(x) = \tanh(x)$ and $h_2(x) = \tanh(2x + 1)$. The augmented feature matrix is

$$A \approx \begin{bmatrix} -.762 & -.762 & 1 \\ 0 & .762 & 1 \\ .762 & .995 & 1 \end{bmatrix}, \qquad \det A \approx -0.982 \neq 0.$$

For desired targets $y = (1, 0, 2)^\top$, solving $A\beta = y$ gives

$$\beta \approx (3.339, -2.326, 1.772)^\top, \qquad A\beta \approx y.$$

So two hidden features *plus a bias* can independently fit three labels. This illustrates why the width threshold is $n_k \ge N - 1$, not necessarily $n_k \ge N$.

:::key Why this example matters
In a simple squared-loss readout, $\nabla_\beta \frac{1}{2}\|A\beta - y\|^2 = A^\top(A\beta - y)$. If $A$ has full row rank, $A^\top r = 0$ forces $r = 0$. Thus a stationary readout cannot keep nonzero training residuals. The paper extends this intuition through backpropagation and rank conditions on later layers.
:::

## 3. The main theorem: exactly what is guaranteed?

**Theorem 3.8 (informal but faithful).** Assume the paper's analytic-activation and loss conditions. Suppose some hidden layer $k$ has

$$n_k \ge N - 1.$$

A *critical point* is a global minimizer if it additionally satisfies:

1. A Hessian **block** involving parameters from an upper-layer subset that *includes layer $k + 1$* is non-singular (the paper's block non-degeneracy condition).
2. Every weight matrix from layer $k + 2$ through the output has **full column rank**: $\mathrm{rank}(W_\ell) = n_\ell$.

Because a matrix can have full column rank only if it has at least as many rows as columns, condition 2 implicitly requires a **pyramidal tail**: $n_{k+1} \ge n_{k+2} \ge \cdots \ge n_L$. *Corollary 3.9* gives the corresponding statement for non-degenerate local minima with full-rank upper layers.

:::warn Be precise about "almost all"
The authors prove that the set of *hidden-layer parameters producing deficient augmented feature rank* has Lebesgue measure zero (Lemma 4.4). Their main global-optimality conclusion still requires Hessian-block non-degeneracy and upper-layer full rank. They explicitly do **not** exclude all degenerate or rank-deficient spurious local minima. Do not paraphrase this as "every local minimum is global."
:::

## 4. Understand the conditions rather than memorize them

| Condition | Intuition / why the proof uses it |
|---|---|
| $N$ distinct examples; analytic, strictly monotone activation | Makes full-rank feature representations possible except for exceptional parameter choices; sigmoid, tanh, softplus are covered. |
| $n_k \ge N - 1$ | The $n_k$ features plus the bias can span $N$ independent training-example directions. |
| Full column rank after $k + 1$ | The downstream network does not destroy error information through a rank-deficient map. |
| Non-singular upper-layer Hessian block | Lets the implicit function theorem track stationary upper-layer parameters under small changes to earlier layers. |
| Appropriate smooth loss; minimum attained | Critical loss residuals must correspond to global minima of the per-example loss. |

**Not included in the core guarantee:** nonsmooth ReLU itself, convolutions with sparse connectivity, cross-entropy without checking the assumptions, and any generic GD/SGD convergence rate.

## 5. Why use the Hessian if this is not a saddle-escape paper?

Here the Hessian is *not* used to design Newton updates. Its invertible upper-layer block is a proof tool: it permits upper-layer stationary solutions to vary smoothly when earlier layers change. This is different from the negative-eigenvalue arguments in saddle-escape papers.

## 6. Proof intuition: four logical moves

The proof is advanced, but you should retain its structure rather than its technical details.

1. **Rank implies zero training error at a suitable critical point.** Lemma 3.5 says that if $[F_k, \mathbf{1}]$ has rank $N$, the later matrices have full column rank, and the gradient at layer $k + 1$ vanishes, then the point is globally optimal.
2. **Width makes full feature rank generic.** Analyticity implies that a nontrivial determinant-type expression does not vanish on a positive-volume parameter region. Lemmas 4.3–4.4 establish that rank-deficient feature parameters form a measure-zero set.
3. **Move slightly while remaining stationary above.** The non-singular upper Hessian block permits an application of the implicit function theorem: small lower-layer changes can be paired with adjusted upper-layer parameters that retain upper-layer stationarity.
4. **Transfer global optimality back to the original point.** Nearby full-feature-rank stationary configurations are globally optimal; continuity and local stationary structure show that the original covered critical point is global too.

:::key Think of a locked door
A poorly represented set of examples might seem to block progress; width says the representation can be nudged to distinguish all examples. Full downstream rank carries that information forward, and Hessian non-degeneracy keeps the stationarity equations well behaved under that nudge. The original point cannot then be a regular bad local minimum.
:::

## 7. A second result you should not confuse with the first

**Theorem 5.3: linear separability of learned features.** The authors also study a *different class-dependent loss*, akin to squared hinge loss. If examples have become linearly separable at a layer and later matrices have full column rank, then a critical point is globally optimal for that classification objective. This does *not* automatically mean that linear separability alone gives zero squared regression loss, nor is it the same statement as Theorem 3.8.

### What does "pyramidal" mean?

Suppose a network has layer widths $[3, 80, 25, 8, 2]$. From the 80-unit layer onward, widths decrease. A chain like $[3, 80, 10, 40, 2]$ is *not* pyramidal after the wide layer; rank preservation in the expanding $10 \to 40$ step is impossible if its weight matrix is required to have full column rank.

### What the authors leave open

The theorem does not show that *standard gradient descent* or SGD actually reaches the regular critical points, how quickly it does so, or whether its generalization is good. It also leaves possible degenerate and low-rank suboptimal minima unruled out. The discussion explicitly raises sparse-connectivity extensions as future work.

## 8. Main contributions, assumptions, and limitations

| Contribution | What we can honestly say in the survey |
|---|---|
| Benign wide-network geometry | A broad class of stationary points in sufficiently wide, smooth, fully connected networks is globally optimal, given additional rank and non-degeneracy conditions. |
| Generic feature rank | With $n_k \ge N - 1$, deficient augmented hidden-feature rank occurs only on a measure-zero set of lower-layer parameters. |
| Multilayer extension | Hidden-layer feature rank and full-rank downstream propagation extend simpler single-hidden-layer intuitions. |
| Feature separability result | A separate classification result covers critical points at which learned hidden features are linearly separable. |

:::warn Important limits
No universal "all minima are global" theorem; no proof that degenerate/rank-deficient bad minima cannot exist; smooth analytic activations rather than ReLU itself; an often demanding width threshold $n_k \ge N - 1$; the result is about loss geometry, not training time, SGD guarantees, or generalization.
:::

## 9. Connection to papers we already studied

| Paper | The distinction to remember |
|---|---|
| Safran & Shamir (2018) | Finds certified spurious minima in selected finite-width teacher–student ReLU population losses; Nguyen–Hein proves good geometry for a different, wide, smooth empirical-loss setting. No contradiction. |
| Ge (2017); Bhojanapalli (2016) | Obtain benign landscapes for *structured matrix recovery*; Nguyen–Hein analyzes deep neural-network architectures directly. |
| Du (2019); Allen-Zhu (2019) | Prove optimization convergence for specified overparameterized networks. Nguyen–Hein mainly classifies critical points; a benign landscape does not itself supply a rate or successful algorithm. |
| Lee (2016); Jin (2017) | Study avoidance or finite-time escape from saddles; Nguyen–Hein characterizes which regular stationary points can be globally optimal. |

## 10. Quick reference and reading path

| Term | Meaning / what to remember |
|---|---|
| $N$; $n_k$ | Training examples; width of hidden layer $k$. Main width condition: $n_k \ge N - 1$. |
| $F_k$; $[F_k, \mathbf{1}]$ | Hidden-feature matrix; augmented matrix must have *row rank $N$* for the simple gradient argument. |
| Critical point | All parameter gradients vanish; may be a minimum, saddle, or maximum. |
| Non-degenerate Hessian block | An invertible Hessian submatrix for selected upper-layer parameters. |
| Theorem 3.8 / Corollary 3.9 | Under the full set of conditions, critical points / non-degenerate local minima are global. |
| Theorem 5.3 | Separate result for classification loss and linear separability of learned features. |

**Must read:** Abstract, Section 3.2 (Assumptions 3.2, Lemma 3.5, Theorem 3.8, Corollary 3.9), Section 4 (Lemmas 4.3–4.4 and proof idea), Sections 5–6. **Skim:** long implicit-function and matrix-rank proof details.

:::plain Papers Master — Our notes (copy)
Nguyen and Hein (2017) show that in fully connected networks with analytic monotone activations, a layer with at least $N - 1$ units makes augmented hidden features generically full rank. Under upper-layer full-rank and Hessian-block non-degeneracy assumptions, the covered critical points are global minima. Does not rule out degenerate/rank-deficient bad minima or prove GD/SGD convergence rates.
:::

> **Source:** [Nguyen & Hein, ICML 2017, original PDF](https://proceedings.mlr.press/v70/nguyen17a/nguyen17a.pdf). The 3-example interpolation is a pedagogical adaptation, not a reported experiment.
