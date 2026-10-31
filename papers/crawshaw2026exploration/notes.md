# Muon and Its Many Variants: An exploration of non-Euclidean gradient descent

:::minute The paper in one minute
**Question:** Which geometry defines the steepest update for a whole neural network, including Muon-trained matrices and Adam-trained parameters?

**Model/Method:** Choose a norm per layer, combine those norms into one product-space norm, and derive constrained or regularized steepest-descent steps. Test Muon variants with model-truncation momentum (Momo).

**Finding:** The framework unifies practical Muon+Adam updates and motivates MuonMax, whose Momo version is substantially less sensitive to learning-rate choice in the tested GPT-2 tasks.

**Why we care:** Optimizer design involves a choice of geometry, not only a learning rate. Better tuning robustness is an empirical finding, not a theorem guaranteeing saddle escape.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Different norms, different unit balls

A **norm** measures the size of a step, and its **unit ball** is the set of steps of size at most one. For vectors, the Euclidean norm $\|v\|_2$ has a round ball and the max-norm $\|v\|_\infty = \max_i |v_i|$ has a square ball. For matrices there are more choices: the **Frobenius norm** $\|\Delta\|_F$ treats the matrix as a long vector, while the **spectral norm** $\|\Delta\|_{2\to2}$ is its largest singular value, the most it can stretch any input vector. The identity $I_2$ has spectral norm $1$ but Frobenius norm $\sqrt{2}$: it is "size one" in one geometry and larger in the other. Which steps count as small depends on which norm we pick.

### 2. Steepest descent in a norm, and its dual norm

To first order a step $\Delta$ changes the loss by $\langle g, \Delta\rangle$. The **steepest-descent** step in a norm is the unit-ball step that makes this most negative, and the best value it reaches is minus the **dual norm** of $g$:

$$\min_{\|\Delta\| \le 1} \langle g, \Delta\rangle = -\|g\|_* .$$

For $g = (3, -1)$: in the Euclidean ball the best step is $-g/\|g\|_2$ and the decrease is $\sqrt{10} \approx 3.16$ (the Euclidean norm is its own dual). In the max-norm ball the best step is $-\mathrm{sign}(g) = (-1, 1)$, which moves every coordinate by a full unit, and the decrease is $|3| + |-1| = 4 = \|g\|_1$. So sign descent is steepest descent in the max-norm, and the $\ell_1$ norm is the max-norm's dual.

### 3. The polar factor sets every singular value to one

Every matrix has a singular value decomposition $G = U \Sigma V^\top$: rotations $U$, $V$ and nonnegative stretches $\Sigma$. Its **polar factor** is $\mathrm{polar}(G) = U V^\top$: same directions, every stretch replaced by $1$. It is the matrix analogue of $\mathrm{sign}$, and $-\mathrm{polar}(G)$ is the steepest-descent step in the spectral-norm ball. The decrease it achieves is the **nuclear norm**, the sum of singular values. For $G = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ the singular values are $1.618$ and $0.618$, so

$$\mathrm{polar}(G) \approx \begin{bmatrix} 0.894 & 0.447 \\ -0.447 & 0.894 \end{bmatrix}, \qquad \langle G, \mathrm{polar}(G)\rangle = 1.618 + 0.618 \approx 2.236 = \|G\|_{\mathrm{nuc}} .$$

Muon's update is (approximately, via a cheap Newton–Schulz iteration) the polar factor of the momentum matrix.

### 4. Constrained versus regularized: how long is the step?

Steepest descent comes in two forms. **Constrained** steepest descent (CSD) takes a step of fixed size $\eta$ in the best direction, whatever the gradient's size. **Regularized** steepest descent (RSD) minimizes $\langle g, \Delta\rangle + \frac{1}{2\eta}\|\Delta\|^2$, whose solution has the same direction but length $\eta\,\|g\|_*$, so it grows with the gradient. In the Euclidean norm with $g = (3, 4)$ and $\eta = 0.1$: CSD moves a distance $0.1$ (normalized gradient descent), RSD moves $0.1 \times 5 = 0.5$ (plain gradient descent, $-\eta g$). Muon normalizes like CSD; the MuonMax variant in the notes is an RSD method.

:::key Putting the four ideas together
A step's size depends on the chosen norm (idea 1); each norm has its own steepest direction and dual norm (idea 2); for weight matrices under the spectral norm that direction is the polar factor and the dual is the nuclear norm (idea 3); and the step length is either fixed or scaled by the dual norm (idea 4). The paper builds one such norm for the whole network, recovers Muon+Adam as CSD and derives MuonMax as RSD, as made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. First principles: "steepest" depends on the norm

For a small update $\Delta$, the first-order model is

$$f(W + \Delta) \approx f(W) + \langle G, \Delta\rangle.$$

Under an Euclidean constraint $\|\Delta\|_F \le \eta$, the steepest update is proportional to $-G$. But a matrix has many legitimate norms. Under a spectral-norm constraint, the direction is related to the matrix's polar factor rather than its entrywise values.

$$\Delta_{\mathrm{CSD}} = \arg\min_{\|\Delta\| \le \eta} \langle G, \Delta\rangle, \qquad \Delta_{\mathrm{RSD}} = \arg\min_\Delta \Big\{ \langle G, \Delta\rangle + \frac{1}{2\eta}\|\Delta\|^2 \Big\}.$$

These are constrained and regularized steepest descent (CSD and RSD). Their directions and step lengths need not coincide.

## 2. Worked matrix example: why Muon is not just SGD

Take the illustrative gradient $G = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$ and a unit trust-region radius. With Frobenius norm, the steepest unit update is

$$\Delta_F = -\frac{1}{\sqrt{10}} \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}, \qquad \langle G, \Delta_F\rangle = -\sqrt{10}.$$

With spectral norm $\|\Delta\|_{2\to2} \le 1$, its linear minimizer is $\Delta_{\mathrm{sp}} = -I$, yielding

$$\langle G, \Delta_{\mathrm{sp}}\rangle = -4.$$

The spectral trust region permits full unit steps in both singular directions. More generally, if $G = U\Sigma V^\top$, the minimizing matrix is $-UV^\top$ (up to zero singular directions); the dual of spectral norm is the nuclear norm, the sum of singular values.

:::key One norm, a different algorithm
A matrix-orthogonalized update can arise from optimizing the same linearized loss under a different trust-region geometry. It is not generally a Newton or inverse-Hessian update.
:::

## 3. Combine the layers, not just the individual matrices

A network has hidden matrices $W^1, \dots, W^L$ and other parameters $\theta$. The paper uses a norm for each component and aggregates them, for example:

$$\|W\|_\infty = \max_i \|W^i\|_{(i)}, \qquad \|W\|_2 = \Big(\sum_i \|W^i\|_{(i)}^2\Big)^{1/2}.$$

Proposition 3.6 interprets practical Muon+Adam as constrained steepest descent for a suitable *product norm* with a spectral norm on hidden matrices and an adaptive norm on other parameters.

## 4. From Muon to MuonMax

Choosing a different product norm changes how layerwise updates are scaled relative to one another. MuonMax arises as a regularized steepest-descent method under a hybrid norm. Its matrix update includes a scale tied to the *sum of nuclear norms of the layerwise momenta*:

$$W^\ell_{t+1} = W^\ell_t - \eta \Big(\sum_j \|M^j_t\|_{\mathrm{nuc}}\Big)\, \mathrm{polar}(M^\ell_t) \qquad \text{(schematic matrix part).}$$

In the full method, learning rates and non-matrix parameter rules must also be respected. The layer coupling is important: unlike normalizing each matrix independently, the update scale can reflect the collection of layers.

### Why "stale" nuclear norms?

Computing polar factors and their nuclear norms for every layer before updating anything can create memory or runtime overhead. The authors reuse previous-iteration dual norms as an approximation. In their ablations, this reduces the overhead of MuonMax–Momo to about 5% per step relative to MuonAdam, with little effect on measured loss.

## 5. Model-based momentum: the Momo ingredient

A plain Taylor model can underestimate loss if a chosen step is overly aggressive. Momo adds a known or estimated lower bound $F_*$ for the loss to a truncated first-order model, combining current and historical gradient/loss information. In the paper, integrating Momo with Muon variants improves robustness to step-size tuning.

:::warn Do not confuse two questions
The paper *derives a geometric interpretation* of existing and novel optimizers; its preference for MuonMax–Momo comes from language-model experiments. It does not establish a general nonconvex global-convergence or second-order saddle-escape theorem for Muon.
:::

## 6. Experimental evidence and caveats

The authors sweep 36 optimizer variants on GPT-2 Small trained on FineWeb (1B tokens) and evaluate selected methods on GPT-2 Large with SlimPajama (1B and 6B tokens). Figure 1 and Section 5 report that MuonAdam and Scion can be sensitive to the learning rate, while the Momo variants retain competitive validation loss across a substantially wider range. These are task- and tuning-protocol-dependent results. **Contributions.** (i) A product-norm steepest-descent framework; (ii) an exact geometric interpretation of practical MuonAdam; (iii) new MuonMax updates and efficient stale-norm approximation; (iv) Momo combinations and controlled comparisons.

## 7. Assumptions and limitations

- Norm equivalences and first-order subproblem derivations assume well-defined linear minimization oracles, dual norms, and the relevant momentum/preconditioning quantities.
- Empirical conclusions come from the selected GPT-2/data/token budgets; they do not prove universal superiority or generalization benefits across architectures.
- The method does not directly solve for Hessian negative-curvature directions. Calling it a guaranteed saddle-escape algorithm would be inaccurate.

## 8. Quick reference

| | |
|---|---|
| **Euclidean steepest step** | $\Delta \propto -G$ under a Frobenius-norm constraint. |
| **Spectral step** | $\Delta \propto -\mathrm{polar}(G)$ for matrix weights. |
| **Norm duality** | $\lVert G\rVert_{\mathrm{nuc}}$ is dual to spectral norm. |
| **CSD vs RSD** | Fixed norm-ball radius versus quadratic norm penalty. |
| **MuonAdam** | CSD under a mixed spectral/adaptive product norm. |
| **MuonMax** | RSD with layer-coupled dual-norm scaling. |
| **Momo** | Model truncation uses a loss lower bound. |

### Papers Master note

Crawshaw et al. interpret Muon+Adam as non-Euclidean steepest descent on network product spaces and introduce MuonMax; Momo improves LR robustness in GPT-2 experiments. Geometric optimizer design and empirical stability are the contributions, not a general saddle-escape theorem.

> **Original:** [https://proceedings.mlr.press/v306/crawshaw26a.html](https://proceedings.mlr.press/v306/crawshaw26a.html). The $2 \times 2$ matrix is a pedagogical illustration.
