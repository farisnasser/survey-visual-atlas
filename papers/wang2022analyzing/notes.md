# Analyzing Sharpness along GD Trajectory: Progressive Sharpening and Edge of Stability

:::minute The paper in one minute
**Question:** Why does Hessian sharpness rise toward $2/\eta$ and then oscillate near that value while full-batch GD still makes progress?

**Model/Method:** Four-phase analysis of GD trajectories, experiments on neural networks, and conditional theorems for a two-layer *linear* network.

**Finding:** Growth of an output-weight norm is correlated with progressive sharpening; negative feedback from an unstable top residual mode can reduce sharpness after overshooting $2/\eta$.

**Why we care:** Explains edge-of-stability (EoS) as a moving interaction between curvature and discrete updates, not a constant-curvature theorem.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Sharpness belongs to a point, not to a problem

The **sharpness** at weights $\theta$ is the largest eigenvalue of the Hessian $\nabla^2 L(\theta)$: the steepest curvature in any direction. It changes as the weights move. Take the smallest "network", one input and one label, with output $f = a\,w$ (an output weight $a$ times an inner weight $w$) and the paper's squared loss $L(a, w) = (aw - 1)^2$. Every point with $aw = 1$ is a global minimum, and the Hessian there is

$$\nabla^2 L = 2\begin{bmatrix} w^2 & 1 \\ 1 & a^2 \end{bmatrix}, \qquad \text{eigenvalues } 0 \text{ and } 2(a^2 + w^2).$$

At $(a, w) = (1, 1)$ the sharpness is $4$; at $(2, 0.5)$ it is $8.5$. Same loss value, very different curvature. Which minimum GD approaches, and how large the weights are on the way, decides the sharpness.

### 2. For squared loss, curvature is a Gram matrix of gradients

Collect the network's outputs on the $n$ training points in a vector $F$ and the labels in $Y$. The **residual** is $D = F - Y$. The Hessian of $L = \frac1n\|D\|^2$ has two parts: one built from first derivatives of the outputs, one multiplied by the residuals. The first part shares its nonzero eigenvalues with the $n \times n$ **Gram matrix** $M = \frac{2}{n}\,\frac{\partial F}{\partial \theta}\big(\frac{\partial F}{\partial \theta}\big)^\top$ (also called the neural tangent kernel), and it dominates the top eigenvalue. The paper uses $\lambda_{\max}(M)$ as its definition of sharpness. In the toy above, $\partial f / \partial(a, w) = (w, a)$, so $M = 2(a^2 + w^2)$: at a zero-residual point it equals the Hessian's top eigenvalue exactly.

### 3. GD moves the residuals by a matrix multiplier

A small GD step changes the outputs by about $\frac{\partial F}{\partial \theta}\Delta\theta$, and $\Delta\theta = -\eta \nabla L$. Putting the two together gives the first-order rule the paper works with (its Assumption 3.2):

$$D(t + 1) \approx \big(I - \eta M(t)\big)\, D(t).$$

Write $D$ in the eigenvectors of $M$. The component along the top eigenvector $v_1$, the number $D^\top v_1$, is multiplied by $1 - \eta\Lambda$ at each step, where $\Lambda = \lambda_{\max}(M)$. The other components feel smaller eigenvalues. So when $\Lambda$ passes $2/\eta$, only the top component misbehaves; the rest of the residual keeps shrinking.

### 4. The sign of $D^\top F$ decides whether the output layer grows

Write the network as $f(x) = A^\top h(W, x)$: an output weight vector $A$ on top of features $h$. Under gradient flow the squared norm of the output layer changes by a clean rule (the paper's Eq. 2):

$$\frac{d\|A\|^2}{dt} = -\frac{4}{n}\, D^\top F.$$

In the toy model ($n = 1$, $A = a$, $F = aw$): if the output undershoots the label, say $aw = 0.5$, then $D^\top F = (0.5 - 1)(0.5) = -0.25 < 0$ and $a^2$ grows. If it overshoots, $aw = 1.5$, then $D^\top F = 0.75 > 0$ and $a^2$ shrinks. Since $\|A\|$ feeds the Gram matrix (idea 2), a growing output layer tends to mean growing sharpness, and an overshoot tends to pull it back.

:::key Putting the four ideas together
Sharpness is a moving property of the current weights (idea 1), measured through the Gram matrix of output gradients (idea 2). GD multiplies the top residual component by $1 - \eta\Lambda$ (idea 3), so above $2/\eta$ that component grows and the outputs overshoot; an overshoot flips the sign of $D^\top F$, which shrinks the output layer and with it the sharpness (idea 4). That feedback loop is the four-phase cycle, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. The puzzle: a loss can improve even when the local quadratic picture says "unstable"

Sharpness means the *largest Hessian eigenvalue* $\Lambda(t) = \lambda_{\max}(\nabla^2 L(\theta_t))$. For a fixed positive quadratic direction, GD is locally stable only when $\eta\Lambda < 2$. Yet Figure 1 shows neural-network training continuing after $\Lambda$ reaches roughly $2/\eta$. Why? The parameters move, so the Hessian and the meaning of "locally unstable" change every step. The paper follows those changes rather than freezing curvature.

**Four phases (Section 3.1).** I: $\Lambda < 2/\eta$ and progressive sharpening; II: $\Lambda > 2/\eta$ and a top residual component grows; III: curvature drops while that component may still grow; IV: sharpness falls below the threshold, the residual contracts, and the cycle can repeat.

## 2. Worked example: why the number $2/\eta$ appears

Take a one-direction quadratic $L(z) = \frac{\lambda}{2} z^2$ with $\lambda > 0$. Its gradient is $\lambda z$, so one GD step gives

$$z_{t+1} = z_t - \eta\lambda z_t = (1 - \eta\lambda) z_t.$$

The distance shrinks exactly when $|1 - \eta\lambda| < 1$, equivalently $0 < \eta\lambda < 2$. For $\lambda = 5$ and $\eta = 0.3$, the multiplier is $-0.5$: each step changes sign but contracts. For $\lambda = 8$, it becomes $-1.4$: each step changes sign and expands.

**What this does *not* prove.** In a real network, the direction and curvature are not fixed. A step that expands one residual component may also change the weights enough to *lower* subsequent curvature. The paper's four-phase model seeks to explain this feedback.

## 3. The paper's mechanism: residual modes and changing weights

In the squared-loss linear-network setting, the authors write a vector of prediction residuals $D(t)$ and an effective Gram matrix $M^*(t)$, obtaining a discrete evolution of the form

$$D(t + 1) = \big(I - \eta M^*(t)\big) D(t).$$

This resembles the scalar example, except that the matrix itself evolves with the weights. Project $D$ onto the leading eigenvector $v_1$. Once its associated eigenvalue exceeds $2/\eta$, that component may grow geometrically, producing non-monotone loss. The weight updates then push sharpness down; when curvature falls enough, the component contracts again.

:::key Why does the output layer matter?
The empirical observation is that changes in the *output-layer weight norm* often track changes in sharpness. The paper decomposes the network Gram matrix into contributions from output and inner-layer parameters. In its special linear model, this relation can be analyzed precisely; it is not an exact law for all deep nets.
:::

## 4. Theoretical and empirical findings

**Theorem 1 (Section 4.2, informal):** Under a controlled eigengap/spectrum, sufficiently wide linear network, label alignment, and sample-size/width conditions, sharpness increases during an initial progressive-sharpening interval.

**Theorem 2:** With additional bounds on residual growth, sharpness, data spectrum and width, if sharpness rises sufficiently beyond $2/\eta$ it must later fall below a nearby threshold. A leading residual component can grow while above it. This is a *conditional* result for the special model.

**Figures 1–2:** Experiments show characteristic loss/sharpness oscillations and a lower-frequency relationship between output weight norm and sharpness. The authors openly note that fast sharpness oscillations and the roles of inner layers are not fully captured.

## 5. Proof idea: why a growing residual can reduce future sharpness

The top residual component obeys an approximate scalar recursion with multiplier $1 - \eta\Lambda_t$. In a frozen quadratic, $|1 - \eta\Lambda_t| > 1$ would make its magnitude grow forever. The key is that the network is *not frozen*: the same growing component changes the parameter gradients and can reverse the upward movement of sharpness. Theorem 2 formalizes parts of this feedback for a restricted linear model.

**How to read Figure 1:** Compare loss and sharpness timelines jointly. First identify when $\Lambda$ crosses $2/\eta$; then find the delayed sharpness decrease and the nonmonotone loss interval. Oscillating sharpness alone does not show that optimization is improving. The authors highlight unexplained fast oscillations (Section 5).

## 5. Assumptions and limits

**Assumptions:** Full-batch, fixed-step GD; special two-layer linear architecture with symmetric initialization in the proof; restrictions on the spectrum of $X^\top X$, label projections, width/sample size and bounded residual behavior. The analysis uses squared error.

**Limits:** The four-phase interpretation combines experiments and partial heuristics; it is not a universal proof of EoS in nonlinear networks. The theoretical phase results do not establish generalization, optimal step-size choice, or convergence for arbitrary SGD. In particular, the norm/sharpness relationship can differ at high frequency.

## 6. Connections to the survey

**Kalra (2025)** later studies sharpness regimes via low-dimensional fixed points; **Chen–Bruna (2023)** characterizes stable two-step oscillations. This paper instead emphasizes *how sharpness evolves along a trajectory* and the four-phase feedback mechanism. **Kalra–Barkeshli (2024)** uses progressive sharpening and early sharpness reduction to understand warmup.

## 7. Quick reference

| Concept | Meaning / equation |
|---|---|
| Sharpness | $\Lambda = \lambda_{\max}(\nabla^2 L)$ |
| Quadratic boundary | $\eta\Lambda = 2$ |
| Predicted residual dynamics | $D_{t+1} = (I - \eta M^*_t) D_t$ |
| Four phases | below / above / decreasing / below threshold |
| Original to read | Sections 3.1–3.4, 4.2, Figure 1 |

## 8. Papers Master note

:::plain Copy into Our notes
Explains progressive sharpening and EoS as a four-phase curvature/residual feedback cycle. Formal sharpness claims require a restricted two-layer linear model and strong spectral/width assumptions; the full-network account remains partly empirical.
:::

> **Original paper:** [View original research PDF](https://papers.neurips.cc/paper_files/paper/2022/file/40bb79c081828bebdc39d65a82367246-Paper-Conference.pdf) *Worked numerical examples are pedagogical unless explicitly attributed to the authors.*
