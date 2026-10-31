# Why Very Wide ReLU Networks Can Be Trained

:::minute The paper in one minute
**Question:** How can ordinary GD and SGD fit training data with deep ReLU networks when their objective is non-convex *and* non-smooth?

**Model/Method:** Sufficiently wide networks initialized at random. Prove that near initialization, large loss forces a nonzero gradient and a semi-smoothness bound controls each update.

**Finding:** Under data-separation and width assumptions, GD and minibatch SGD reach arbitrarily small squared training loss in polynomially many iterations, with logarithmic dependence on accuracy. Certain CNNs and ResNets are covered too.

**Why we care:** Architecture and width can make the region actually visited by training benign; an optimizer need not explicitly search for Hessian negative-curvature directions.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A gradient lower bound is enough for fast convergence, even without convexity

Suppose a loss $F \ge 0$ satisfies $\|\nabla F(W)\|^2 \ge 2\mu F(W)$ everywhere: whenever the loss is positive, the gradient is large. This is a **Polyak–Łojasiewicz (PL) inequality**. It does not need convexity. For example, $F(x) = x^2 + 3\sin^2 x$ has $F''(x) = 2 + 6\cos 2x$, which reaches $-4$, so $F$ is not convex; yet $F'(x)^2/(2F(x))$ never drops below about $0.17$, so the inequality holds with $\mu \approx 0.17$. Together with a descent lemma, PL makes every GD step remove a fixed fraction of the remaining loss. The paper's Theorem 3 is a local version of this inequality.

### 2. ReLU kinks break the usual descent lemma

The textbook guarantee $F(W + \Delta) \le F(W) + \langle\nabla F(W), \Delta\rangle + \tfrac{L}{2}\|\Delta\|^2$ needs a gradient that changes at most linearly. The ReLU $\phi(z) = \max\{z, 0\}$ has slope $0$ just left of zero and $1$ just right of it. Between $z = -0.1$ and $z = 0.1$ the slope changes by $1$ over a distance $0.2$, a ratio of $5$; between $\pm0.001$ the ratio is $500$. No single constant $L$ works. The paper's Theorem 4 therefore adds a first-order error term to the descent bound.

### 3. In a wide layer, a small perturbation flips few ReLUs

A neuron's pre-activation at random initialization behaves like a Gaussian $z \sim \mathcal{N}(0, 1)$. A perturbation of size $\varepsilon$ can flip its ReLU only if $|z| < \varepsilon$, which has probability about $2\varepsilon/\sqrt{2\pi} \approx 0.8\,\varepsilon$. For $\varepsilon = 0.01$ that is about $0.8\%$ of the neurons. In a wide layer most activation patterns stay fixed, so the network behaves almost linearly near its initialization. This calculation is an illustration of step 2 of the paper's proof, not its statement.

### 4. Geometric decrease gives $\log(1/\varepsilon)$ steps and a bounded path

If each step multiplies the loss by $1 - \gamma$, then $F_T \le (1 - \gamma)^T F_0$, and reaching $F_T \le \varepsilon$ takes about $\log(F_0/\varepsilon)/\gamma$ steps: $1375$ for $\gamma = 0.01$, $F_0 = 1$, $\varepsilon = 10^{-6}$. The same geometric decay bounds the distance travelled. Step lengths shrink like $\sqrt{F_t} \propto (\sqrt{1 - \gamma})^t$, and the sum of this geometric series is $1/(1 - \sqrt{1 - \gamma}) \approx 200$ times the first step for $\gamma = 0.01$, however long training runs. A bounded path is how the iterates stay in the region where ideas 1–3 apply.

:::key Putting the four ideas together
Near initialization, positive loss forces a large gradient (idea 1). Few ReLUs flip under small moves (idea 3), so a modified descent bound still holds (idea 2). Each step then removes a fixed fraction of the loss, which keeps the total movement bounded (idea 4). The notes below show how Theorems 3 and 4 make this precise.
:::
:::

:::divider Study notes
:::

## 1. Start with the difficulty, not the theorem

A deep neural network multiplies and composes weight matrices, and uses kinks like $\phi(z) = \max\{z, 0\}$. Its training objective is generally non-convex and not differentiable everywhere. *Having enough parameters to fit the data* does not by itself show that *GD starting randomly will find those parameters*.

The authors ask for both *efficient trainability* and a dependence on depth $L$ that is polynomial rather than exponential. Their central idea is local: we need good optimization geometry *near the random initialization and training trajectory*, not across the entire parameter space.

## 2. The model and two essential conditions

For the fully connected main analysis, let $W_1, \dots, W_L$ be the trainable hidden-layer matrices, $A$ the input map, and $B$ the output map. The forward pass is

$$h_{i,0} = \phi(Ax_i), \qquad h_{i,\ell} = \phi(W_\ell h_{i,\ell-1}), \qquad \widehat{y}_i = Bh_{i,L}.$$

The square-loss training objective is $F(W) = \frac12\sum_{i=1}^n \|\widehat{y}_i - y_i^*\|_2^2$. Two conditions matter:

- **Nondegenerate data:** distinct normalized training inputs have pairwise separation at least $\delta > 0$. Identical inputs with conflicting labels cannot both be fitted.
- **Overparameterization:** each hidden layer has sufficiently large width $m$, polynomial in the sample count $n$, depth $L$, and inverse separation $1/\delta$ (plus problem-specific dimensions/factors).

:::warn Do not overread "global minimum."
In the square-loss result, $F(W) \le \varepsilon$ means the nonnegative training objective becomes arbitrarily small. It is not a guarantee of test accuracy, and it does not say that every stationary point everywhere is benign.
:::

## 3. Worked example: why a large gradient can force the loss down

*A teaching example of the local linearized prediction map, not the exact network or proof in the paper.* Suppose two training residuals are $e = \widehat{y} - y^* \in \mathbb{R}^2$, and the Jacobian of the predictions with respect to three parameters is

$$J = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}, \qquad K = JJ^\top = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}.$$

**Step 1 — The loss and gradient.** With $F = \frac12\|e\|_2^2$, the parameter gradient is $\nabla_\theta F = J^\top e$. Therefore,

$$\|\nabla_\theta F\|_2^2 = e^\top JJ^\top e = e^\top Ke.$$

**Step 2 — Why distinct prediction directions matter.** The eigenvalues of $K$ are 1 and 3. Thus for every residual $e$,

$$\|\nabla_\theta F\|_2^2 = e^\top Ke \ge \|e\|_2^2 = 2F.$$

If $F > 0$, the gradient cannot be zero. This is the intuition behind a *gradient-dominance-like lower bound*.

**Step 3 — A numerical GD update.** In the exactly linearized model, a parameter step of size $\eta$ updates residuals by $e^+ = (I - \eta K)e$. Take $\eta = \frac14$ and $e = (1, -1)^\top$, an eigenvector with eigenvalue 1:

$$e^+ = \tfrac34 e, \qquad F_0 = \tfrac12\big(1^2 + (-1)^2\big) = 1, \qquad F_1 = \left(\tfrac34\right)^2 = 0.5625.$$

The residual is not trapped at a nonoptimal stationary point; the training loss decreases in this model.

:::key What transfers to the paper:
overparameterization and separated inputs are used to prevent the different sample gradients from canceling each other near initialization. The actual deep ReLU proof is significantly harder: the Jacobian changes with the weights and activation signs can flip.
:::

## 4. Theorem 3: no bad critical points near initialization

The paper proves (with high probability and within its specified neighborhood of random initialization) a lower bound of the form

$$\|\nabla F(W)\|_F^2 \;\ge\; c\,F(W), \qquad c = \Omega\!\left(\frac{\delta m}{d\,n^2}\right) > 0,$$

where $d$ denotes the output dimension in the paper's notation. There is also an upper gradient-norm bound. The authors call this a *no critical point* property: positive loss cannot coexist with zero gradient *in the controlled neighborhood*. It resembles a Polyak–Łojasiewicz inequality but is derived for the composite network locally, not assumed globally.

## 5. Why the ReLU kink creates a second obstacle

The gradient bound alone does not prove that GD decreases the loss at every step. The classic smooth descent lemma assumes a Lipschitz gradient, but ReLU changes slope abruptly at zero. For example, $\phi(-0.1) = 0$ and $\phi(0.2) = 0.2$; across zero the derivative jumps from 0 to 1. The paper therefore proves a different *semi-smoothness* inequality controlling the loss after a small weight perturbation.

## 6. Theorem 4: semi-smoothness replaces ordinary smoothness

Let $\Delta$ be a small update and $W$ remain near initialization. The paper's bound has the schematic form

$$F(W + \Delta) \le F(W) + \langle\nabla F(W), \Delta\rangle + C_2\|\Delta\|^2 + C_1\|\Delta\|\sqrt{F(W)}.$$

*The constants and precise matrix norms are in Theorem 4; this displays its structure.* Unlike the textbook smoothness lemma, the bound contains an additional term $\|\Delta\|\sqrt{F}$ reflecting nonsmooth activation changes. With enough width and a suitably small step, the authors show these error terms can be controlled.

## 7. Combine the two results: why convergence becomes geometric

**Step 1.** A GD update $\Delta = -\eta\nabla F(W)$ makes the linear term $-\eta\|\nabla F(W)\|^2$ negative.

**Step 2.** The semi-smoothness result controls the terms that could otherwise spoil the loss decrease.

**Step 3.** The gradient lower bound replaces $\|\nabla F\|^2$ by a multiple of $F$; the proof obtains a contraction of the form

$$F(W_{t+1}) \le (1 - \gamma)F(W_t), \qquad 0 < \gamma < 1, \qquad\Longrightarrow\qquad F(W_T) \le (1 - \gamma)^T F(W_0).$$

**Step 4.** Because the loss contracts fast enough, weights do not wander outside the initialization neighborhood where Theorems 3 and 4 apply. This closes the argument.

:::key Why this is more than "wide networks have no saddles."
The theorem controls a high-probability, *local* region and the particular GD/SGD trajectories. It does not characterize every critical point of the full non-convex landscape. The proof uses both geometry (no bad local critical points in the region) and dynamics (a guaranteed useful update).
:::

## 8. Main formal guarantees: what is actually proved

| Result | Meaning under the theorem's specified assumptions |
|---|---|
| Theorem 1 | GD reduces squared training loss to $\varepsilon$ in $T = \mathrm{poly}(n, L, 1/\delta)\log(1/\varepsilon)$ iterations, with sufficiently large $m$ and small step size. |
| Theorem 2 | Minibatch SGD also attains logarithmic dependence on $1/\varepsilon$; the stated iteration bound additionally depends on batch size $b$ and logarithmic width factors. |
| Theorems 3–4 | Local lower gradient bound (no positive-loss critical point) and semi-smoothness, the two technical ingredients behind convergence. |
| Theorems 5–7 | Conditional extensions to other output losses, a specified CNN family, and a simple residual-network architecture. |

A caveat about SGD: geometric convergence is not automatic for arbitrary stochastic objectives; here the training-data interpolation structure and problem assumptions are crucial.

## 9. How the proof works, without reproducing the appendices

1. **At random initialization:** concentration controls forward and backward signal propagation across $L$ layers without allowing exponential blow-up in depth.
2. **After small perturbations:** show that only a controlled number of ReLU activation signs change and hidden representations remain close to their initialized values.
3. **Lower-bound gradients:** distinct data points give sufficiently separated hidden representations, preventing contributions from many examples from canceling completely.
4. **Control every update:** prove semi-smoothness and combine it with the gradient bound so GD/SGD decreases loss and remains in the trustworthy region.

## 10. What the figure and experiments support

Figure 1 shows empirical 2D loss slices around points along an SGD trajectory on CIFAR-10, using gradient and a negative-curvature direction after smoothing. It illustrates why following the negative gradient can still provide substantial loss decrease even though negative curvature exists. *This is an illustration consistent with the theory, not a proof that all practical training stays inside the theorem's neighborhood.* The authors explicitly acknowledge practical weight movement can exceed their bounds.

## 11. Important assumptions and limitations

- **Nondegenerate data and large width:** the main results require sample separation $\delta$ and potentially very conservative polynomial width requirements.
- **Specific initialization and training setup:** Gaussian initialization, carefully tuned learning rates, and staying close to initialization; the main formulation holds $A, B$ fixed while training hidden matrices $W_\ell$.
- **Training not generalization:** arbitrarily small squared training error or perfect *training* classification accuracy does not imply low test error.
- **Not every arbitrary loss has the same guarantee:** the nonconvex output-loss extension distinguishes gradient-dominant, convex, and general nonconvex cases.
- **Architecture claims have scope:** CNN and ResNet extensions cover particular specified constructions, not every modern architecture.

## 12. Connections to our survey

| Paper | How it differs from Allen-Zhu et al. |
|---|---|
| Lee (2016) | General asymptotic avoidance of strict saddles under smoothness; no global training-fit guarantee. |
| Jin (2017) | Perturbations yield approximate second-order stationary points in finite time; here local geometry can yield *global training fit* without explicit escape. |
| Du (2019) | Another global-fit theory for wide deep nets; emphasizes smooth activations and GD, while Allen-Zhu treats ReLU, SGD, and polynomial depth dependence for fully connected nets. |
| Wu (2025) | Analyzes nonsmooth stationary points and escape neurons in shallow ReLU-like nets rather than wide-network training convergence. |

## 13. Quick reference and Papers Master note

| Term | Meaning |
|---|---|
| Overparameterized | $m$ large enough (polynomial in $n, L, 1/\delta$, with other factors) for local high-probability control. |
| No-critical-point bound | $\lVert\nabla F\rVert^2 \ge cF$: positive training loss forces a nonzero parameter gradient. |
| Semi-smoothness | Descent bound with $\lVert\Delta\rVert^2$ and $\lVert\Delta\rVert\sqrt{F}$ correction terms. |
| Geometric rate | $F_T \le (1 - \gamma)^T F_0$, hence $T = O(\log(1/\varepsilon))$ for fixed problem parameters. |

:::plain Our notes (copy into Papers Master):
Wide randomly initialized deep ReLU networks can be trained by GD/SGD to arbitrarily small squared training loss in polynomial time under data-separation and width assumptions. The proof combines a local gradient lower bound (no positive-loss critical points) and semi-smoothness, keeping iterates near initialization. Results concern training fit, not generalization; width bounds can be conservative.
:::

> **Primary source:** Allen-Zhu, Li and Song, *ICML* (2019), Theorems 1–7 and Sections 3–6. [https://proceedings.mlr.press/v97/allen-zhu19a/allen-zhu19a.pdf](https://proceedings.mlr.press/v97/allen-zhu19a/allen-zhu19a.pdf). Worked Jacobian example is pedagogical, not from the original paper.
