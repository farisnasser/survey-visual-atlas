# An Improved Analysis of Training Over-parameterized Deep Neural Networks

:::minute The paper in one minute
**Question:** Can we prove training convergence of deep ReLU networks with much smaller required widths than earlier theory?

**Model/Method:** GD and minibatch SGD on randomly initialized deep fully connected ReLU networks, analyzing local gradient lower bounds and weight displacement from initialization.

**Finding:** A sharper gradient bound and telescoping control of trajectory length reduce the sufficient GD width scaling in the sample count from roughly $n^{24}$ to $n^8$ (with other parameter factors) and improve the iteration complexity.

**Why we care:** The theorem does not change GD itself; it changes how much overparameterization a proof needs to explain successful training.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A gradient lower bound gives linear convergence

Suppose each GD step lowers the loss by at least $\tfrac{\eta}{2}\|\nabla\mathcal{L}\|^2$ (true for a small enough step), and suppose the gradient can never be small while the loss is large: $\|\nabla\mathcal{L}(W)\|^2 \ge \mu\,\mathcal{L}(W)$. Then $\mathcal{L}(W_{t+1}) \le (1 - \eta\mu/2)\,\mathcal{L}(W_t)$, so the loss shrinks by a fixed factor every step: **linear convergence**. Reaching loss $\varepsilon$ takes about $\frac{2}{\eta\mu}\log\frac{\mathcal{L}_0}{\varepsilon}$ steps. For example, a factor $0.99$ per step needs $1375$ steps to cut the loss by $10^6$.

The paper's two ingredients have exactly this shape. Its gradient lower bound (Lemma 4.1) gives $\mu$ of order $m\phi/(kn^2)$, and its step size is $\eta$ of order $k/(L^2m)$. The product $\eta\mu$ is of order $\phi/(n^2L^2)$, independent of the width $m$, which gives Theorem 3.3's $T = O(n^2L^2\log(1/\varepsilon)/\phi)$.

### 2. Wide networks barely move their weights

The gradient lower bound only holds near the random initialization. The proof must therefore show that training never leaves a small ball around $W^{(0)}$. In wide networks this is plausible: many weights share the work, so each moves little. The paper bounds the movement of every layer by $O\big(\sqrt{kn^2\log(n)/(m\phi)}\big)$ (Lemma 4.2). Quadrupling the width halves this bound.

The required width comes from making the movement bound smaller than the radius of the ball where Lemma 4.1 holds. The sharper the movement bound, the smaller the width that suffices. That is where the telescoping argument in the notes pays off.

### 3. Separated inputs are needed to fit every label

The paper assumes unit-norm inputs that are pairwise at least $\phi$ apart (Assumptions 3.1–3.2). Without separation, zero training loss can be impossible: if $x_1 = x_2$ but $y_1 = 1$ and $y_2 = -1$, any network predicts the same value $u$ for both, and the squared loss $\tfrac12\big((u - 1)^2 + (u + 1)^2\big)$ is at least $1$ (at $u = 0$). A small $\phi$ means nearly duplicate inputs with possibly different labels, which is why the bounds grow as $\phi$ shrinks.

### 4. Polynomial width conditions are sufficient, not necessary

A width condition like $m = \widetilde{\Omega}(kn^8L^{12}/\phi^4)$ says: *at least this wide is enough for the proof*. It does not say narrower networks fail. The exponents show how loose such bounds are: for $n = 1000$ examples, $n^8 = 10^{24}$ and $n^{24} = 10^{72}$, both far beyond any real network. The improvement by a factor $n^{16} = 10^{48}$ is a better proof, not a change in practice.

:::key Putting the four ideas together
Near initialization, a lower bound on the gradient turns GD into linear convergence at a rate independent of width (idea 1), but only while the weights stay near where they started (idea 2), and only for separated data that can be fitted exactly (idea 3). Zou and Gu bound the movement more tightly, so a smaller width keeps the iterates inside the good region. The resulting width condition drops from $n^{24}$ to $n^8$ while remaining a sufficient condition (idea 4), made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. The puzzle: wide networks work, but how wide must they be?

Prior work shows sufficiently wide ReLU networks can reduce square training loss under random initialization. However, earlier proof techniques required hidden widths with extremely high-degree polynomial dependence on the number $n$ of examples. Such requirements are theoretical sufficient conditions, often far beyond real architectures. Zou and Gu ask whether a better analysis can shrink the proven width needed. Their object is the empirical regression loss $\mathcal{L}(W)$ for a deep ReLU network, with all relevant hidden layers wide and weights initialized with scaled Gaussian entries.

### The two proof obligations

**(i) A gradient lower bound:** If training loss is positive and the weights stay in a controlled neighborhood of initialization, the gradient must be large enough to make progress. **(ii) A movement bound:** To keep this gradient property valid, the parameter trajectory must not travel too far from initialization. This creates a loop: good geometry implies loss reduction; loss reduction in turn bounds movement.

### Worked example: telescoping a learning trajectory

Consider a stylized loss sequence $L_t = q^tL_0$ with $0 < q < 1$, and suppose an analysis controls each update by

$$\|W_{t+1} - W_t\| \le C\big(\sqrt{L_t} - \sqrt{L_{t+1}}\big).$$

Summing $t = 0, \dots, T-1$ makes the middle terms cancel:

$$\sum_{t<T}\|W_{t+1} - W_t\| \le C\sum_{t<T}\big(\sqrt{L_t} - \sqrt{L_{t+1}}\big) = C\big(\sqrt{L_0} - \sqrt{L_T}\big) \le C\sqrt{L_0}.$$

The total path length remains bounded *independently of the number of iterations*. This pedagogical inequality illustrates the actual proof’s telescoping idea in Eq. (4.1); its constants and conditions are different.

:::key
**Why the proof improves:** Bounding each gradient step separately can accumulate too much error. A telescoping square-root-loss bound controls the *entire path* more tightly, reducing the required width.
:::

## 2. Results and what the theorem really says

### Theorem 3.3: deep-network GD

Assumptions 3.1–3.2 require normalized inputs, a common positive last coordinate (a bias-like condition), and pairwise input separation $\|x_i - x_j\| \ge \phi > 0$. Under sufficiently large hidden width, Gaussian initialization, and a carefully scaled learning rate, GD reaches training loss at most $\varepsilon$ in

$$T = O\!\left(\frac{n^2L^2}{\phi}\log\frac{1}{\varepsilon}\right)$$

iterations (paper notation; problem-dependent constants suppressed). Table 1 gives sufficient width

$$m = \widetilde{\Omega}\!\left(\frac{k\,n^8L^{12}}{\phi^4}\right),$$

compared with $\widetilde{\Omega}(kn^{24}L^{12}/\phi^8)$ in the cited previous work.

### Theorem 3.8: minibatch SGD

The stochastic theorem uses a different width and batch-size condition; Table/statement indicate polynomial $n^{17}$, dependence on batch size $B^{-4}$ and separation $\phi^{-8}$. Thus the GD width improvement does *not* transfer unchanged to SGD. Theorem 3.10 gives tighter special-case bounds for two-layer networks.

### How the proof is organized

Near initialization, ReLU changes are controlled by a semi-smoothness estimate; the gradient lower bound quantifies descent opportunity. The argument combines descent with the telescoping movement bound so that the iterates remain where the inequality is valid, closing an induction.

:::warn
**What it does not prove:** These are guarantees on achieving small *training loss*, not test generalization or a theorem that all ReLU landscapes lack spurious minima. Their architecture and width assumptions remain stringent.
:::

> Read Sections 3–4, Theorems 3.3/3.8/3.10, Table 1 and Eq. (4.1). Use the original PDF to verify any constants before citing a rate.

## 3. Contributions, limits, and comparisons

**Contributions.** Improved GD and SGD convergence analysis for deep overparameterized ReLU networks; tighter gradient lower bounds; an improved trajectory-length argument; improved specialization to two-layer networks. **Assumptions.** Squared loss, normalized and separated data, Gaussian initialization, sufficiently wide hidden layers, small learning rates, and a local neighborhood around initialization where gradient and semi-smoothness estimates apply. **Limitations.** The sufficient widths remain large and data-dependent; the result is about optimization rather than generalization. It does not apply unchanged to arbitrary activations, architectures, adaptive methods, or losses; for SGD the sufficient width may be much larger than for GD.

## Connection to prior papers

**Allen-Zhu (2019)** supplies the earlier semismoothness and gradient-lower-bound framework this work sharpens. **Du (2019)** proves zero training-loss convergence for selected sufficiently wide smooth-activation architectures. **Safran (2018)** finds spurious minima for smaller, different ReLU settings, so these conditional positive guarantees do not contradict it. **Nguyen (Ref 92)** will reduce a different assumption: how many layers must be wide.

## Quick reference

| Concept | Meaning |
|---|---|
| $\phi$ | Lower bound on pairwise training-input separation. |
| $n$ / $k$ / $L$ / $m$ | Number of samples / output dimension / depth / hidden width. |
| Gradient lower bound | Positive training loss implies usable descent locally. |
| Telescoping path bound | Controls parameter displacement from initialization. |
| $n^8$ versus $n^{24}$ | Exponents in sufficient GD width scaling, other factors omitted. |

### Papers Master note

:::plain
Zou–Gu improve the convergence proof for wide deep ReLU networks via a tighter local gradient lower bound and telescoping control of trajectory length; the sufficient GD width dependence in sample count falls from approximately $n^{24}$ to $n^8$ with other factors retained. SGD requires a separate, more restrictive theorem.
:::

> **Source:** [original NeurIPS 2019 PDF](https://papers.neurips.cc/paper/8479-an-improved-analysis-of-training-over-parameterized-deep-neural-networks.pdf), Table 1, Assumptions 3.1–3.2, Theorems 3.3/3.8 and Eq. (4.1).
