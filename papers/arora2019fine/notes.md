# Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks

:::minute The paper in one minute
**Question:** Why do neural nets fit *real* labels faster than random labels, and which data property predicts generalization?

**Model/Method:** A wide, randomly initialized two-layer ReLU network; its training predictions are approximated by a linear dynamical system controlled by a data-dependent kernel/Gram matrix.

**Finding:** Different eigen-directions of the kernel learn at different rates; label alignment explains training-speed variation. A data-dependent complexity term $y^\top(H^\infty)^{-1}y$ underlies a width-independent generalization bound.

**Why we care:** Zero training loss alone does not explain training speed or generalization. The label–kernel alignment adds information hidden by a single worst-case convergence rate.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. The ReLU Gram matrix measures how similar two inputs look to a random neuron

The paper's network is $f(x) = \frac{1}{\sqrt{m}}\sum_r a_r\,\mathrm{relu}(w_r^\top x)$ with unit-norm inputs, random $w_r$ and fixed signs $a_r = \pm1$. Its **Gram matrix** has entries $H^\infty_{ij} = \mathbb{E}_w\big[x_i^\top x_j\,\mathbb{1}\{w^\top x_i \ge 0,\, w^\top x_j \ge 0\}\big]$: the inner product of two inputs, counted only when a random neuron is active on both. It has the closed form $x_i^\top x_j\,(\pi - \arccos x_i^\top x_j)/(2\pi)$ (paper Eq. 3).

Example: two unit inputs at $60°$ have $x_1^\top x_2 = 0.5$ and $\arccos 0.5 = \pi/3$, so $H^\infty_{12} = 0.5 \cdot (2\pi/3)/(2\pi) = 1/6$. On the diagonal $H^\infty_{ii} = 1/2$. Hence $H^\infty = \begin{pmatrix} 1/2 & 1/6 \\ 1/6 & 1/2 \end{pmatrix}$.

### 2. A very wide network's predictions follow a linear recursion

One GD step on the first-layer weights changes each prediction by the chain rule: $u(k+1) - u(k) \approx -\eta\,H(k)\,(u(k) - y)$, where $H(k)$ is the Gram matrix of the network's current gradients. When the width $m$ is large and the initialization small, each neuron moves very little, so $H(k)$ stays close to $H^\infty$ for the whole run. The predictions then follow the fixed linear system $\widetilde{u}(k+1) = \widetilde{u}(k) - \eta H^\infty(\widetilde{u}(k) - y)$, which the paper uses as its proxy (Eq. 7).

### 3. Eigenvectors split the labels into fast and slow parts

A linear recursion is solved mode by mode. For the $60°$ example, $H^\infty$ has eigenvalue $2/3$ along $(1, 1)/\sqrt2$ and $1/3$ along $(1, -1)/\sqrt2$. Labels $y = (1, 1)$, equal on the two similar inputs, lie entirely in the fast mode. Labels $y = (1, -1)$, opposite on similar inputs, lie entirely in the slow mode. With $\eta = 1$ the residual shrinks by $1/3$ per step in the first case and by $2/3$ per step in the second. Labels that disagree on similar inputs, as random labels tend to, are learned slowly.

### 4. How far the weights travel bounds how complex the learned function is

A network whose weights stay close to their initialization can only represent a limited family of functions, and the paper bounds the **Rademacher complexity** of that family (Lemma 5.4), a standard measure of how well a function class can fit random noise. The distance travelled is controlled by the labels: $\|W(k) - W(0)\|_F \le \sqrt{y^\top (H^\infty)^{-1}y}$ plus small terms (Lemma 5.3). For the $60°$ example, $y^\top(H^\infty)^{-1}y$ is $3$ for $y = (1, 1)$ and $6$ for $y = (1, -1)$. Labels in slow modes force the weights to travel further, and the generalization bound grows with them.

:::key Putting the four ideas together
The ReLU Gram matrix $H^\infty$ encodes how similar the inputs look to random neurons (idea 1). In the wide regime, predictions evolve by the linear map $I - \eta H^\infty$ (idea 2), so each eigen-direction of the labels is learned at its own rate (idea 3). The same matrix, through $y^\top(H^\infty)^{-1}y$, bounds how far the weights move and therefore how well the trained network generalizes (idea 4), made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. From global convergence to *which directions learn fast?*

For a sufficiently wide two-layer ReLU network, the empirical output vector $u(t)$ evolves approximately according to a kernel-gradient update. If $H^\infty$ is the limiting positive-definite Gram matrix, then the linearized model gives

$$\widetilde{u}_{t+1} = \widetilde{u}_t - \eta H^\infty(\widetilde{u}_t - y), \qquad e_{t+1} = (I - \eta H^\infty)\,e_t,$$

where $e_t = \widetilde{u}_t - y$ and $y$ is the training-label vector. A lower bound on the smallest eigenvalue ensures eventual contraction, but it hides how different label directions behave.

### Worked example: fast and slow label components

Let the toy kernel and learning rate be

$$H^\infty = \begin{pmatrix} 2 & 0 \\ 0 & 0.2 \end{pmatrix}, \qquad \eta = 0.25, \qquad y = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \qquad \widetilde{u}_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}.$$

The residual update is $e_{t+1} = \mathrm{diag}(0.5, 0.95)\,e_t$, so

$$e_1 = \begin{pmatrix} -0.5 \\ -0.95 \end{pmatrix}, \qquad e_2 = \begin{pmatrix} -0.25 \\ -0.9025 \end{pmatrix}, \qquad \|e_t\|^2 = 0.5^{2t} + 0.95^{2t}.$$

The first label direction is learned quickly; the second persists. Thus the same optimizer and network can display very different loss curves depending on which eigen-directions the target labels occupy.

:::plain
**Interpretation:** The paper explains why meaningful labels can train faster than randomized labels *in the observed settings*, through the spectrum of $H^\infty$ and projections $v_i^\top y$; it does not claim every random labeling is slower for every dataset.
:::

## 2. Spectral training law and generalization quantity

Let $H^\infty = \sum_i \lambda_i v_iv_i^\top$ with positive eigenvalues. With a zero initial output in the linearized proxy,

$$\|e_t\|^2 = \sum_{i=1}^n (1 - \eta\lambda_i)^{2t}\,(v_i^\top y)^2.$$

Theorem 4.1 shows that the actual neural-network prediction error stays close to this expression (with an additive tolerance), under sufficiently large width, small step size, kernel nondegeneracy, and high-probability initialization conditions. This sharper result improves on an analysis using only $\lambda_{\min}$.

### Generalization: the inverse-kernel cost

Theorem 5.1 produces a bound depending on a data-dependent quantity of the form

$$\sqrt{\frac{2\,y^\top(H^\infty)^{-1}y}{n}} \quad \text{plus smaller confidence terms,}$$

under its sampling, label and loss assumptions. This term is *not* a guarantee that every fitted network generalizes. Its significance is that it does not explicitly grow with network width.

### Worked intuition: when is a label expensive to learn?

For our toy $H^\infty = \mathrm{diag}(2, 0.2)$, the kernel cost for $y = (1, 1)$ is

$$y^\top(H^\infty)^{-1}y = 1/2 + 1/0.2 = 5.5.$$

For $y = (1, 0)$ it is only $0.5$. Placing energy in a low-eigenvalue direction is expensive for the complexity bound and slow for optimization. Theorem 5.1 also contains assumptions and constants not reflected in this two-dimensional demonstration.

:::warn
**Separate these statements:** Theorem 4.1 concerns training *dynamics*; Theorem 5.1 concerns population/test performance under statistical assumptions. One does not follow automatically from the other.
:::

> Read Section 4 (Theorem 4.1), Section 5 (Theorem 5.1), Figures 1–2, and the data-dependent complexity discussion.

## 3. Main contributions, assumptions, limits

**Contributions.** A mode-by-mode, kernel-spectral description of training error; an explanation for slower fitting of randomized labels in studied datasets; a bound on generalization controlled by label/kernel alignment rather than raw parameter count; learning guarantees for a broad class of sufficiently smooth target functions in the paper’s setting. **Assumptions.** Two-layer ReLU network with specified random initialization and fixed output signs, squared-error GD, sufficiently large width and small learning rate, a positive-definite limiting kernel $H^\infty$, and additional distributional/label and bounded-loss conditions for generalization. **Limitations.** Kernel-regime approximation may not capture rich feature learning when weights move far from initialization. Empirical real-versus-random label comparisons are dataset-specific. This paper’s generalization results are not general theorems for arbitrary deep networks and optimizers.

## Cross-paper connections

**Du and Allen-Zhu (2019)** establish that sufficiently wide networks can fit training data; Arora et al. clarify *which directions fit first* and what controls a particular generalization bound. **Saxe (2014)** likewise decomposes learning into data modes, but in deep linear rather than wide ReLU kernel dynamics. **Zou–Gu (Ref 86)** sharpens worst-case convergence requirements, while Arora et al. add mode-specific predictions.

## Quick reference

| Concept | Interpretation |
|---|---|
| $H^\infty$ | Data-dependent limiting prediction-gradient Gram matrix. |
| $\lambda_i, v_i$ | Kernel eigenvalues and orthogonal data modes. |
| $(1 - \eta\lambda_i)^t$ | Per-mode training-error decay in the kernel proxy. |
| $v_i^\top y$ | How strongly the labels align with mode $i$. |
| $y^\top(H^\infty)^{-1}y$ | Data-dependent complexity controlling the bound. |

### Papers Master note

:::plain
Arora et al. refine wide two-layer ReLU optimization analysis using the eigenstructure of a limiting Gram matrix: label components in high-eigenvalue modes converge faster, and $y^\top(H^\infty)^{-1}y$ helps control a width-independent generalization bound under additional statistical assumptions.
:::

> **Source:** [original ICML 2019 PDF](https://proceedings.mlr.press/v97/arora19a/arora19a.pdf), Sections 3–5, Theorems 4.1 and 5.1. Examples are pedagogical.
