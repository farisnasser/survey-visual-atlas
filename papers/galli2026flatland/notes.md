# Flatland: Large-Step GD

:::minute The paper in one minute
**Question.** How can GD use extremely large learning rates yet converge, and must the flattest neural-network regions be desirable?

**Model / Method.** First-order adaptive step sizes via nonmonotone *equality line search*; theory uses local smoothness along the gradient direction.

**Finding.** Large steps often put training at the edge of stability (EoS), with oscillating loss. In the MSE-classification experiments, some runs reach maximally flat *saddle* regions early and fail to learn; limiting step size improves them.

**Why we care.** Learning-rate choice, curvature, and trainability are inseparable. Neither monotone loss decrease nor maximal flatness is always a useful training objective.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. A line search tests a step before taking it

Instead of fixing $\eta$, a **line search** tries a step and accepts it only if the loss drops enough: $f(w - \eta g) \le f(w) - c\,\eta\|g\|^2$ with $g = \nabla f(w)$ and a small $c \in (0, 1)$ (the **sufficient-decrease** or Armijo test). On $f(x) = \tfrac12 x^2$ at $x = 1$ the test reads $\tfrac12(1 - \eta)^2 \le \tfrac12 - c\eta$, which holds exactly for $0 < \eta \le 2(1 - c)$: with $c = 0.01$, every step up to $1.98$ passes. More generally, on a quadratic of curvature $\lambda$ the boundary is $2(1 - c)/\lambda$, just below the stability edge $2/\lambda$. A line search that pushes its step up to the boundary, halving or doubling the trial step, therefore lands within a factor $2$ below it, at or above $(1 - c)/\lambda$, near the edge of stability. This is the shape of the paper's Proposition 2.5.

### 2. A nonmonotone test compares with a running average of past losses

A **nonmonotone** line search replaces $f(w_k)$ on the right of the test by a reference value $R_k \ge f(w_k)$. The Zhang–Hager reference is a weighted average of past losses, with weights shrinking by a factor $\xi \in (0, 1)$ per step. With $\xi = 0.85$ and losses $1.0$ then $0.5$, the reference after the second step is $(0.85 \cdot 1.0 + 0.5)/1.85 \approx 0.73$, well above the current loss $0.5$. A step may raise the loss as long as it stays below this average, which is what lets GD oscillate at the edge of stability and still converge.

### 3. A small sharpness does not mean a minimum

The sharpness $\lambda_{\max}(H)$ only looks at the most positive curvature. For any unit vector $v$, $\lambda_{\max}(H) \ge v^\top H v$, so the largest eigenvalue of any diagonal block of $H$ is a lower bound on $\lambda_{\max}(H)$. Take $H = \begin{pmatrix} 0.2 & 0.3 \\ 0.3 & -0.5 \end{pmatrix}$: its eigenvalues are about $0.31$ and $-0.61$. The sharpness is small, at least the block value $0.2$, yet the second eigenvalue is negative: this point is a saddle. A method that drives sharpness down can stop on such a point.

:::key Putting the three ideas together
An equality line search keeps the step near the largest acceptable size, which on a quadratic sits just below $2/\lambda$ (idea 1), and a nonmonotone reference lets the loss go up temporarily (idea 2), so GD runs at the edge of stability. The paper finds that in MSE classification this can drive the sharpness to its lowest possible value, a bias-block bound, at points that are saddles (idea 3). The notes below give the definitions, the proved guarantee and the experiments.
:::
:::

:::divider Study notes
:::

## 1. Start with the mystery: large steps do not always fail

Traditional GD uses $w_{k+1} = w_k - \eta_k\nabla f(w_k)$. A globally $L$-Lipschitz gradient suggests using a small step $\eta < 2/L$ to secure descent. But a deep network need not have one finite global $L$, and practitioners often train with larger steps. This paper asks how to choose such steps without requiring global smoothness.

### A worked example: why the edge threshold is 2

Take the *teaching example* $f(x) = \tfrac12\lambda x^2$ with $\lambda = 4$, so $f'(x) = 4x$. GD becomes

$$x_{k+1} = x_k - \eta\lambda x_k = (1 - 4\eta)x_k.$$

To contract toward $x = 0$, we need $|1 - 4\eta| < 1$, i.e. $0 < \eta\lambda < 2$. For $\eta = 0.40$, the multiplier is $-0.6$: each step crosses the optimum, but gets closer. At $\eta = 0.50$, it is $-1$: it oscillates forever. For $\eta = 0.55$, the multiplier is $-1.2$, so *fixed-step GD on this quadratic diverges.*

:::warn Read this carefully
The empirical EoS diagnostic is approximately $\eta_k\lambda_{\max}(\nabla^2 f(w_k)) \simeq 2$, with possible excursions above 2 in nonquadratic training. This is *not* a claim that holding $\eta\lambda > 2$ fixed is safe on every objective.
:::

### What "sharpness" means here

$\lambda_{\max}(H(w))$ is the largest eigenvalue of $H(w) = \nabla^2 f(w)$. It measures the strongest *positive* local curvature. A low sharpness value does not by itself imply a low loss, a local minimum, or good test accuracy.

## 2. What the authors actually change: the line search

Instead of knowing a global Lipschitz constant, the paper measures how much the gradient changes *along a proposed GD segment*:

$$\ell(w, \eta) = \sup_{0 < t \le \eta} \frac{\|\nabla f(w) - \nabla f(w - t\nabla f(w))\|}{\|t\nabla f(w)\|}.$$

This **local segment smoothness** need only be finite locally. The paper also analyzes Hölder versions with gradient differences of order $\|u - v\|^\nu$, $0 \le \nu \le 1$.

### A precise definition of a "large" step

For a fixed window length $P$, a sequence is large if each $\eta_k \ge 1/\ell(w_k, \eta_{w_k})$ and at least once within the next $P$ steps, $\eta_{k+p} > 2/\ell(w_{k+p}, \eta_{w_{k+p}})$. The point is *persistent aggressive stepping*, not merely a large numerical learning rate.

### Nonmonotone equality line search (NLS)

The acceptance test is

$$f(w_k - \eta_k g_k) \le R_k - c\eta_k\|g_k\|^2, \quad g_k = \nabla f(w_k), \quad R_k \ge f(w_k), \; 0 < c < 1.$$

1. Start with a candidate step. If it fails, **backtrack** (shrink it).
2. If it succeeds, **extrapolate** (expand it) to approach the acceptance boundary.
3. Set $w_{k+1} = w_k - \eta_k g_k$ and repeat, updating the reference $R_k$.

Since $R_k$ can exceed today's loss, an accepted step can temporarily increase $f$.

### Worked example: accepting a temporary loss increase

Let $f(x) = \tfrac12 x^2$, $x = 1$, $g = 1$, and propose $\eta = 2.1$. Then

$$x^+ = 1 - 2.1 = -1.1, \qquad f(x^+) = 0.605 > f(1) = 0.5.$$

A *monotone* rule would reject the step. But with past-loss reference $R = 0.70$ and $c = 0.01$, the nonmonotone right-hand side is $0.70 - 0.01(2.1) = 0.679$. Since $0.605 \le 0.679$, it is accepted. This illustrates the acceptance mechanism, *not* a guarantee that repeated $\eta = 2.1$ is safe.

:::theorem Theorem 2.9 (what is proved)
If $f \in C^1$ is bounded below, its initial level set is compact, and the specified Zhang–Hager-type nonmonotone equality search is used, then $\|\nabla f(w_k)\| \to 0$. This says **first-order stationarity**; it does not assert convergence to a global optimum or avoidance of all saddles.
:::

### Additional theoretical properties

Proposition 2.5 gives a *local smoothness-aware lower bound* $\eta_k \ge 2\delta(1 - c)/\ell(w_k, \eta_{w_k})$. Proposition 2.8 extends adaptivity to Hölder segment-smooth losses. Corollary 2.10 gives an asymptotic bound involving Hessian eigenvalue *magnitudes*, assuming iterates converge to a stationary point.

## 3. The surprising finding: the flattest region may be wrong

In the authors' $K$-class **mean-squared-error** (MSE) classification setting, consider a network output $z = Wh + b$ with an unconstrained final-layer bias $b \in \mathbb{R}^K$, and loss normalized as the mean squared error across $K$ outputs.

### A short derivation of the $2/K$ sharpness bound

For a single example, $f(b) = K^{-1}\sum_{j=1}^K (z_j - y_j)^2$. Differentiating once in the last-layer bias,

$$\frac{\partial f}{\partial b_j} = \frac{2}{K}(z_j - y_j), \qquad \frac{\partial^2 f}{\partial b_j \partial b_l} = \frac{2}{K}\mathbf{1}_{j=l}.$$

Thus the bias block of the full Hessian is

$$H_{bb} = \frac{2}{K}I_K \quad\Longrightarrow\quad \lambda_{\max}(H_{\text{full}}) \ge \frac{2}{K}.$$

The inequality follows by restricting the Rayleigh quotient to directions changing only $b$. It is a *global lower bound on sharpness*, **not a lower bound on training loss**. For $K = 10$, the threshold is $0.2$; for $K = 100$, it is $0.02$.

:::warn Crucial distinction
A point minimizing $\lambda_{\max}(H)$ globally is not necessarily a minimum of $f$. The experiments find points whose sharpness reaches $2/K$ yet whose Hessian also has negative eigenvalues: nearly stationary **saddles** where useful feature learning has stopped.
:::

### How the experiments reveal the problem

The authors run full-batch GD on 5,000-image subsets of CIFAR-10, CIFAR-100, SVHN and EMNIST, with MLP, CNN, ResNet, VGG, DenseNet and other architectures across main and supplementary experiments. The main results use MSE; other settings appear in appendices.

| Method / phenomenon | What the paper reports |
|---|---|
| NLS / PoNLS | Large adaptive steps, EoS-like loss oscillations; some runs approach sharpness $2/K$ very early. |
| Unsuccessfully flat | Sharpness hits $2/K$ early, but loss remains near random-guess performance; hidden-layer gradients are tiny. |
| Successfully flat | Slightly greater sharpness, much better training loss and often higher test accuracy. |
| NLS-ub | Caps the step size below $K$ for this $K$-class MSE setup; empirically avoids the pathological globally-flat region in tested runs. |

### Why cap $\eta_k < K$?

A self-stabilization *heuristic* suggests limiting sharpness near $2/\eta_k$. If $\eta_k \approx K$, this points toward the smallest possible sharpness $2/K$; capping $\eta_k < K$ leaves room for a slightly sharper valley. This is a setting-specific intervention motivated by the experiments, *not* a universal saddle-escape theorem.

## 4. Contributions, boundaries, and survey synthesis

### What this paper contributes

1. A definition of *large* GD steps based on local segment smoothness (including Hölder generalizations), not global $L$-smoothness.
2. A practical first-order equality line search combining backtracking and extrapolation, with a convergence-to-stationarity result under explicit assumptions.
3. Empirical evidence that such adaptive steps can operate in EoS-like regimes from early training, and can uncover globally least-sharp but unsuccessful saddle regions.
4. A $2/K$ lower bound on Hessian sharpness in the stated MSE/final-bias setting, and an experimentally motivated step-size cap to avoid premature flatness.

### Important assumptions and limitations

- **Theorem vs. practice:** The main convergence theorem gives $\|\nabla f(w_k)\| \to 0$ with $C^1$, lower-bounded objective, compact starting level set, and the specified nonmonotone reference; no global-minimum or second-order guarantee.
- **Local smoothness:** Stronger step-size bounds need segment/Hölder smoothness and algorithm parameters; experimental choices $c = 10^{-4}$ and $\delta = 0.5$ do not automatically satisfy every theoretical "large" condition.
- **Geometry:** The $2/K$ sharpness bound depends on the MSE normalization and an unconstrained linear final layer with bias. It is not a universal numerical sharpness threshold.
- **Evidence:** Major experiments use 5,000 training images; the authors explicitly note that larger-scale comparisons are needed. Sharpness is a proxy, and compute/runtime dominance is not proved.
- **SGD:** Additional SGD experiments are presented, but a corresponding general convergence theorem is left open.

### Running cross-paper comparison: what each answers

| Paper / key | Tags | Link to Flatland |
|---|---|---|
| Saxe (2014) `saxe2014exact` | D, A / F, B | Initialization and depth can produce long plateaus without bad minima. |
| Dauphin (2014) `dauphin2014identifying` | G, E / F, B | Saddle curvature causes slow training; curvature-aware intervention. |
| Lee (2016) `lee2016gradient` | G, D / F | Avoids convergence to strict saddles under conditions, not necessarily quick passage near them. |
| Jin (2017) `jin2017escape` | D, E / F, B | Perturbations provide finite-time second-order stationarity; Flatland targets step-size dynamics. |
| Li (2018) `li2018visualizing` | G, A / F, B | Loss flatness is difficult to compare and interpret visually; Flatland adds failure cases. |
| Du (2019) `du2019gradient` | D, A / F, B | Overparameterization gives a specialized global training-loss convergence result. |
| **Galli (2026)** `galli2026flatland` | **D, G, E / R, B** | Large steps can yield EoS and problematic globally flat saddles; cap steps in tested setting. |

### Quick reference and Papers Master note

**GD:** $w^+ = w - \eta\nabla f$ · **EoS diagnostic:** $\eta\lambda_{\max}(H) \approx 2$ · **Line search:** $f(w^+) \le R - c\eta\|\nabla f\|^2$ · **Sharpness lower bound (MSE):** $\lambda_{\max}(H) \ge 2/K$ · **Guarantee:** $\|\nabla f(w_k)\| \to 0$.

:::plain Our notes (copy to sheet)
Galli et al. (2026) define locally smoothness-aware large GD steps and construct nonmonotone equality line searches that converge to first-order stationarity under compact-level-set assumptions. In MSE neural-network experiments, aggressive steps can reach EoS and globally least-sharp ($2/K$) saddle regions prematurely, stalling training; capping steps below $K$ improves tested runs. No global-minimum or general SGD guarantee.
:::

> **Original:** [PMLR 306 (ICML 2026)](https://proceedings.mlr.press/v306/galli26a.html). Full text used: [arXiv:2606.06722](https://arxiv.org/html/2606.06722). Numerical scalar examples are pedagogical adaptations.
