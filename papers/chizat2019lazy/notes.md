# On Lazy Training in Differentiable Programming

:::minute The paper in one minute
**Question:** Is nearly frozen neural-network training a universal consequence of having many parameters, and is it always desirable?

**Model/Method:** Study gradient flow for the scaled objective $F_\alpha(w) = \alpha^{-2} R(\alpha h(w))$ and compare it with training the linearized model at initialization.

**Finding:** Large output scale can force trajectories close to the initial tangent model; the authors give quantitative finite-time bounds and show that lazy training can worsen vision-model performance.

**Why we care:** Fast, easily analyzable convergence may describe a kernel-like regime rather than genuine nonlinear feature learning. Width alone is not the whole explanation; scaling matters.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. A linearization is accurate to second order in the step

Near the starting weights $w_0$, Taylor's theorem gives $h(w_0 + \Delta) = h(w_0) + Dh(w_0)\,\Delta + O(\|\Delta\|^2)$. The straight-line part is the **linearization**; what it leaves out shrinks with the *square* of the displacement. For $h(w) = w + w^2$ at $w_0 = 0$, the linearization is $w$ and the error is exactly $\Delta^2$: $0.01$ at $\Delta = 0.1$, $0.0001$ at $\Delta = 0.01$. Ten times smaller moves make the linear picture a hundred times more accurate. If training only ever moves the weights a tiny distance, the network behaves like its linearization the whole time.

### 2. Training a linear-in-parameters model is kernel regression

Suppose the model is linear in its parameters, $\overline{h}(w) = h(w_0) + J_0 (w - w_0)$ with a fixed Jacobian $J_0$, and the loss is $\tfrac12\|\overline{h}(w) - y\|^2$. Under gradient flow the residual $r = \overline{h}(w) - y$ obeys $\dot r = -J_0 J_0^\top r$. The matrix $K_0 = J_0 J_0^\top$ is a **kernel** (Gram) matrix between training points, fixed for the whole run. Each eigen-component of $r$ decays like $e^{-\lambda t}$: with eigenvalues $2$ and $0.5$, after time $t = 2$ the two components are multiplied by $e^{-4} \approx 0.018$ and $e^{-1} \approx 0.37$. When $K_0$ is positive definite on the data, training converges exponentially, but only the fixed features in $J_0$ are ever used: nothing about the representation is learned.

### 3. Output scale decides how far the weights must move

Multiply the model's output by $\alpha$. To change $\alpha h(w)$ by an amount of order one, the weights only need to move by about $1/\alpha$, and by idea 1 the linearization error over such a move is of order $1/\alpha^2$. The paper summarizes this with a scale-free number (its Eq. 1),

$$\kappa_h(w_0) = \|h(w_0) - y^\star\|\,\frac{\|D^2 h(w_0)\|}{\|Dh(w_0)\|^2},$$

and lazy training is expected when it is much smaller than $1$. Scaling the model by $\alpha$ divides it by $\alpha$ (when the initial output is bounded). For $h(w) = w + w^2$, $w_0 = 0$, target $y^\star = 1$: $Dh = 1$, $D^2 h = 2$, so $\kappa_{\alpha h} = 2/\alpha$, which is $0.02$ at $\alpha = 100$. A large scale, not width as such, is what makes training lazy.

:::key Putting the three ideas together
Small weight moves keep a network close to its linearization (idea 1), and training a linearization is kernel regression with fixed features (idea 2). A large output scale guarantees the moves are small, of order $1/\alpha$ (idea 3), so the trained network tracks the kernel model. The paper proves this closeness and shows the price, lost feature learning, as made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. What is "lazy training"?

A nonlinear predictor $h(w)$ can be approximated around its initial weights $w_0$ by its first-order Taylor expansion:

$$\overline{h}(w) = h(w_0) + Dh(w_0)(w - w_0).$$

*Lazy training* is the regime in which the actual optimization trajectory remains close to the one that would be obtained by training this fixed linearization. The network can still fit the training labels, but its internal features need not change much.

## 2. The scaling parameter changes the game

The paper explicitly studies

$$F_\alpha(w) = \frac{1}{\alpha^2} R\big(\alpha h(w)\big),$$

with $h(w_0) = 0$ for its simplest theorems. For large $\alpha$, the model needs a parameter displacement of order $1/\alpha$ to produce an $O(1)$ output. Over such small displacements, the neglected quadratic terms in the Taylor expansion are of order $1/\alpha^2$.

## 3. Worked example: same nonlinear model, different scale

Use the toy model $h(w) = w + w^2$, $w_0 = 0$, and a scalar target 1 with squared loss. The scaled prediction is $\alpha h(w)$ and the loss is

$$F_\alpha(w) = \frac{1}{2\alpha^2}\big(\alpha(w + w^2) - 1\big)^2.$$

At $\alpha = 100$, an exact fitted parameter satisfies $w + w^2 = 0.01$. Its positive solution is

$$w_* = \frac{-1 + \sqrt{1.04}}{2} \approx 0.00990195.$$

The linearized model $\overline{h}(w) = w$ instead fits at $\overline{w}_* = 0.01$. Their displacement difference is about $9.8 \times 10^{-5}$: tiny relative to the original parameter scale. The neglected scaled term is $\alpha w_*^2 \approx 0.0098$.

:::warn Interpret the example carefully
This compares two fitted solutions of a pedagogical one-parameter model. The paper proves *trajectory* closeness under additional regularity assumptions; the toy does not establish a time-uniform theorem.
:::

## 4. Theorem 2.2: finite-horizon trajectory closeness

With $h(w_0) = 0$ and a fixed time horizon $T$, Theorem 2.2 shows, in the paper's setting, that as $\alpha \to \infty$,

$$\sup_{t \le T} \|w_\alpha(t) - w_0\| = O(\alpha^{-1}),$$

$$\sup_{t \le T} \|w_\alpha(t) - \overline{w}_\alpha(t)\| = O(\alpha^{-2}).$$

The scaled model outputs also differ by $O(\alpha^{-1})$. A crucial qualification: the hidden constants may depend badly on the fixed time horizon. This result does not automatically extend to arbitrary durations or any learning-rate schedule.

## 5. Why this behaves like kernel regression

The linearized predictor is

$$\overline{h}(w) = h(w_0) + J_0 (w - w_0), \qquad J_0 = Dh(w_0).$$

Only the *fixed* features in $J_0$ matter. In squared-error training, the effective Gram/kernel operator is $K_0 = J_0 J_0^\top$. If this operator is positive definite on the training data, the linearized system can contract training residuals quickly. But this apparent simplicity may hide the fact that the nonlinear network has stopped adapting its features.

### Global and local conclusions have different hypotheses

Theorem 2.4 uses a surjective initialization Jacobian and strong convexity of the output-space loss to obtain a global-type convergence statement in the lazy regime. Theorem 2.5 discusses underparameterized settings with locally constant rank, where training can converge to a *local* rather than global minimum.

:::warn Do not equate lazy convergence with universal success
A guarantee that a fixed tangent model fits well under favorable conditioning does not mean the same regime extracts the best representation or generalizes best on difficult images.
:::

## 6. What the experiments actually suggest

Figure 1 contrasts non-lazy training, where learned features move toward relevant directions, with lazy training that fits the training set while moving much less. Figure 1(c) shows that the test behavior can be worse. Section 3 further investigates CNN settings and reports degraded performance as training becomes lazier. These experiments support a potential practical limitation of the kernel-like regime, not a theorem that all nonlinear feature learning generalizes better.

## 7. Contributions and limitations

**Contributions.** (i) Lazy training arises from scaling choices, not exclusively width; (ii) quantitative trajectory-vs-linearization bounds; (iii) conditions for fast convergence in both sufficiently expressive and rank-restricted cases; (iv) empirical evidence that the lazy regime can reduce the advantages of nonlinear networks. **Assumptions/limitations.**

- The precise bounds require differentiability, controlled local derivatives, the scaled objective, and often $h(w_0) = 0$ or related initialization conditions.
- Theorem 2.2 is stated on fixed time horizons; global convergence results require additional rank or convexity hypotheses.
- This is a lens on optimization regimes, not a proof that all wide networks train lazily or that lazy training always harms test performance.

## 8. Quick reference

| | |
|---|---|
| **Linearized model** | $\overline{h}(w) = h(w_0) + Dh(w_0)(w - w_0)$. |
| **Scaled objective** | $F_\alpha = \alpha^{-2} R(\alpha h(w))$. |
| **Displacement** | $\lVert w_\alpha(t) - w_0\rVert = O(\alpha^{-1})$ for fixed $T$. |
| **Path discrepancy** | $\lVert w_\alpha(t) - \overline{w}_\alpha(t)\rVert = O(\alpha^{-2})$. |
| **Fixed tangent kernel** | $K_0 = Dh(w_0) Dh(w_0)^\top$. |
| **Major caveat** | Good training fit need not imply feature learning or generalization. |

### Survey bridge / Papers Master note

Chizat et al. show that scale choices can force nonlinear training to follow the linearized, kernel-like model near initialization; Theorem 2.2 quantifies path closeness, while experiments reveal possible generalization costs. Essential qualification of overparameterization/NTK convergence interpretations; links to Li–Ma–Zhang (2020).

> **Original:** [https://papers.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf](https://papers.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf). Scalar calculation is pedagogical.
