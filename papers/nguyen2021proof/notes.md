# On the Proof of Global Convergence of Gradient Descent for Deep ReLU Networks with Linear Widths

:::minute The paper in one minute
**Question:** Must *every* hidden layer be extremely wide to guarantee GD fits deep ReLU training data?

**Model/Method:** Standard-parameterized deep ReLU networks trained with squared-error GD. Track the last hidden-layer feature matrix rather than controlling the full neural tangent kernel.

**Finding:** Under appropriate feature-rank and initialization conditions, GD converges geometrically to zero training loss; a single sufficiently wide hidden layer can suffice, with linear, quadratic or cubic width conditions depending on the initialization setting.

**Why we care:** Global training-loss convergence can follow from preserving one informative feature matrix, not necessarily from tracking all activation patterns in an enormous network.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. The smallest singular value measures whether features can separate examples

Stack the last hidden-layer features of the $N$ training examples as the rows of $F_{L-1}$. Its **smallest singular value** $\sigma_{\min}$ is the least amount by which $F_{L-1}^\top$ can stretch a unit vector: $\|F_{L-1}^\top e\| \ge \sigma_{\min}\|e\|$ for every residual $e \in \mathbb{R}^N$. If $\sigma_{\min} > 0$, every nonzero residual produces a nonzero last-layer gradient $F_{L-1}^\top e$.

This needs at least $N$ features. With $N = 3$ examples and only $2$ features, $F_{L-1}$ has rank at most $2$, so some residual $e \ne 0$ satisfies $F_{L-1}^\top e = 0$: the last layer cannot reduce that error at all. This is why Theorem 2.2 asks for a last hidden width $n_{L-1} \ge N$.

### 2. A gradient that grows with the loss gives a geometric rate

If $\|\nabla\Phi\|^2 \ge 2\mu\,\Phi$ along the whole run, a small enough GD step lowers the loss by a fixed fraction of itself, and $\Phi(\theta_t) \le (1 - c\,\eta\mu)^t\,\Phi(\theta_0)$ for a constant $c$. This is a **Polyak–Łojasiewicz-type** inequality: it requires no convexity, only that the gradient cannot be small while the loss is large. Idea 1 gives it with $\mu = \sigma_{\min}(F_{L-1})^2$. With a per-step factor $1 - \eta\alpha_0^2/8 = 0.99$, the loss falls below $1\%$ of its start after $459$ steps.

### 3. Small weight changes cause small feature changes, and small feature changes keep $\sigma_{\min}$

ReLU is **1-Lipschitz**: $|\mathrm{relu}(a) - \mathrm{relu}(b)| \le |a - b|$. Chaining this through the layers bounds how much $F_{L-1}$ can move when the weights move (the paper's Lemma 2.1), without asking which neurons switch on or off. Singular values are also stable: by **Weyl's inequality**, $|\sigma_{\min}(A + E) - \sigma_{\min}(A)| \le \|E\|_2$. For $A = \mathrm{diag}(1, 2)$ and any perturbation with $\|E\|_2 = 0.4$, $\sigma_{\min}(A + E) \ge 0.6$. So if the features drift by less than $\alpha_0/2$, the smallest singular value stays at least $\alpha_0/2$.

### 4. Scaling the hidden weights trades off the conditions

ReLU is **positively homogeneous**: $\mathrm{relu}(\beta z) = \beta\,\mathrm{relu}(z)$ for $\beta > 0$. Multiplying every hidden weight matrix by $\beta$ therefore multiplies $F_{L-1}$, and with it $\alpha_0$, by $\beta^{L-1}$. The paper's Section 3.1 uses this: with the output layer set to zero, the left side of its condition (7) grows like $\beta^{2(L-1)}$ while the right side grows only like $\beta^{L-1}$, so a large enough $\beta$ satisfies it (and similarly (8)–(9)). For $L = 3$, doubling $\beta$ multiplies $\alpha_0^2$ by $16$ and the right side by at most $4$.

:::key Putting the four ideas together
A last hidden layer with at least $N$ well-conditioned features turns any residual into a usable gradient (idea 1), which gives a geometric rate as long as it lasts (idea 2). ReLU's Lipschitz property and Weyl's inequality show the conditioning lasts if the weights stay close to initialization (idea 3), and scaling the initialization makes the conditions hold with a last-layer width of only $N$ (idea 4). Theorem 2.2 combines these in an induction, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Start from the final hidden-layer representation

Let $N$ examples pass through a deep ReLU network. Write $F_{L-1} \in \mathbb{R}^{N\times m}$ for their features at the last hidden layer and $W_L \in \mathbb{R}^{m\times k}$ for the final linear weight matrix. The predicted training outputs are

$$F_L = F_{L-1}W_L, \qquad \Phi(\theta) = \frac12\|F_L - Y\|_F^2.$$

The key property is $\sigma_{\min}(F_{L-1}) > 0$: the feature rows separate the training examples. Holding the features temporarily fixed,

$$\nabla_{W_L}\Phi = F_{L-1}^\top(F_{L-1}W_L - Y), \qquad \|\nabla_{W_L}\Phi\|_F^2 \ge 2\sigma_{\min}(F_{L-1})^2\,\Phi.$$

A nonzero training error cannot have zero last-layer gradient while this smallest singular value stays positive.

### Worked example: two examples, two independent features

Take $F_{L-1} = \mathrm{diag}(1, 2)$, one output, $W_L = (0, 0)^\top$, and target $Y = (1, 1)^\top$. The initial error is $e = (-1, -1)^\top$, and

$$\nabla_{W_L}\Phi = F^\top e = \begin{pmatrix} -1 \\ -2 \end{pmatrix}.$$

A GD step of size $\eta = 0.1$ gives $W_L^+ = (0.1, 0.2)^\top$, and predicted outputs $(0.1, 0.4)^\top$. The squared loss falls from 1 to $\tfrac12(0.9^2 + 0.6^2) = 0.585$. The example holds hidden features fixed; the paper’s theorem additionally controls their evolution when *all layers* train.

:::key
**Key conceptual point:** One full-row-rank feature matrix can make the final linear layer powerful enough to correct the training residual. The hard part is proving the features remain sufficiently nondegenerate throughout learning.
:::

## 2. What makes the proof simpler than NTK tracking?

### Main argument: keep the last hidden features stable

Standard NTK proofs often attempt to control the full Gram matrix of all parameter derivatives and activation-pattern changes. Nguyen instead tracks $F_{L-1}$, using ReLU’s Lipschitz property and bounds on changes in layer matrices. If $\alpha_0 = \sigma_{\min}(F_{L-1}^{(0)}) > 0$ and the accumulated feature drift is less than roughly $\alpha_0/2$, then

$$\sigma_{\min}(F_{L-1}^{(t)}) \ge \alpha_0/2$$

by a singular-value perturbation bound. The last-layer gradient remains useful.

### Theorem 2.2: a conditional geometric rate

With explicit initial feature-rank, layer-norm, learning-rate and data-dependent inequalities (conditions (7)–(9) of the paper), GD training all layers satisfies a geometric decrease of the form

$$\Phi(\theta_t) \le \left(1 - \frac{\eta\alpha_0^2}{8}\right)^t\Phi(\theta_0).$$

Do not quote this as a universal bound: the initial conditions ensure the feature rank survives full-network updates.

### How width depends on initialization

The paper’s Section 3 shows several ways to satisfy these conditions. Carefully chosen initial points can require only a last hidden width of order $N$; a conventional LeCun-style random initialization for two-layer networks yields a sufficient order $N^2$ under additional data assumptions. A deep-network concentration setting yields other polynomial width requirements, including cubic regimes. All theorems use *sufficient*, not necessary, widths.

:::warn
**Do not conflate two Nguyen papers:** Nguyen–Hein (2017) characterizes regular stationary points of wide networks, whereas Nguyen (2021) proves an actual GD training trajectory can attain zero training loss under initialization and feature-conditioning assumptions.
:::

> Read Theorem 2.2, its induction proof, Section 3.1 (linear width), Section 3.2 (LeCun initialization), Section 3.3 (deeper case).

## 3. Contributions, assumptions, limits

**Contributions.** A shorter proof of global *training-loss* convergence for standard-parameterized deep ReLU networks; only one wide hidden layer rather than necessarily all; a feature-matrix stability condition replacing full NTK tracking; linear/quadratic/cubic sufficient width regimes under different initializations. **Assumptions.** Squared empirical loss, standard ReLU hidden layers plus linear output, initial nonzero smallest feature singular value, controlled norms and learning rate, and data/distributional conditions for particular random-initialization corollaries. **Limitations.** No generalization guarantee; not an unconditional result for all networks or arbitrary starting points. The feature-conditioning and initialization requirements are crucial, and width scaling varies substantially by construction. There is no general saddle-escape bound.

## Connections across the survey

**Nguyen–Hein (2017)** uses wide features to characterize favorable stationary points; **Nguyen (2021)** adds a GD trajectory theorem. **Allen-Zhu (2019)** and **Zou–Gu (Ref 90)** require sufficiently broad overparameterization throughout the deep architecture; this paper explores a different structural simplification. **Du (2019)** studies selected deep and residual networks under other activation and width assumptions.

## Quick reference

| Object | Interpretation |
|---|---|
| $F_{L-1}$ | Last hidden-layer features of the training samples. |
| $\alpha_0 = \sigma_{\min}(F_{L-1}^{(0)})$ | Conditioning measure at initialization. |
| $\nabla_{W_L}\Phi = F_{L-1}^\top(F_L - Y)$ | Gradient used to link positive error to progress. |
| Feature stability | Small drift preserves positive smallest singular value. |
| Geometric convergence | Training loss contracts at each GD step under conditions. |

### Papers Master note

:::plain
Nguyen (2021) proves GD convergence to zero squared training loss for certain deep ReLU networks by maintaining a full-row-rank last-hidden feature matrix, without controlling the entire NTK. Only one sufficiently wide hidden layer may be needed; linear/quadratic/cubic sufficient widths depend on initialization and data assumptions.
:::

> **Source:** [original ICML 2021 PDF](https://proceedings.mlr.press/v139/nguyen21a/nguyen21a.pdf), Theorem 2.2 and Section 3.
