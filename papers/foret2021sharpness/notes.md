# Sharpness-Aware Minimization for Efficiently Improving Generalization

:::minute The paper in one minute
**Question:** How can training favor solutions that stay low-loss throughout a neighborhood rather than only at one parameter vector?

**Model/Method:** SAM minimizes the maximum training loss under bounded parameter perturbations, approximately solving the inner problem with a first-order gradient step.

**Finding:** A practical two-gradient update targets neighborhood-wise low loss and improves test performance and label-noise robustness in tested models.

**Why we care:** Optimizer interventions can target *landscape geometry* and generalization, not merely saddle-escape iteration counts.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Two losses: the one we train and the one we want

The **training loss** $L_S(w) = \frac1n\sum_{i=1}^n l(w, x_i, y_i)$ averages the per-example loss $l$ over the training set $S$. The **population loss** $L_D(w)$ is the expected loss on a fresh example from the data distribution $D$. We can only compute $L_S$, but we care about $L_D$. In a non-convex network many weight vectors reach almost the same $L_S$; the paper starts from the observation that they can have very different $L_D$.

### 2. The worst nudge of a linear function points along the gradient

For a fixed vector $g$, the largest value of $\epsilon^\top g$ over all $\epsilon$ with $\|\epsilon\|_2 \le \rho$ is $\rho\|g\|_2$, reached at $\epsilon = \rho\, g/\|g\|_2$ (the Cauchy–Schwarz inequality). With $g = (3, 4)$ and $\rho = 0.1$: the best $\epsilon$ is $(0.06, 0.08)$ and the maximum is $0.5$. Near $w$ the loss is approximately linear in $\epsilon$, so this is the approximate worst perturbation of the weights.

### 3. The slope of a worst case is the slope at the worst point

Let $F(w) = \max_{\|\epsilon\| \le \rho} L(w + \epsilon)$. When the maximizer $\epsilon^*(w)$ is unique, $\nabla F(w) = \nabla L(w + \epsilon^*(w))$: moving $w$ changes the worst case only through the loss at the worst point (Danskin's theorem). In one dimension, take $L(w) = w^2$ and $\rho = 0.1$. Then $F(w) = (|w| + 0.1)^2$, and at $w = 1$ its slope is $2.2$, which is exactly $L'(1.1)$. This is why SAM takes its gradient at the perturbed point $w + \hat\epsilon$; with the first-order $\hat\epsilon$ of idea 2 it is an approximation.

### 4. A generalization bound with a dial

The paper's motivation (its Theorem 1, stated informally) has the form: with high probability over the training set,

$$L_D(w) \le \max_{\|\epsilon\|_2 \le \rho} L_S(w + \epsilon) + h\big(\|w\|_2^2/\rho^2\big),$$

with $h$ increasing. The radius $\rho$ is a dial: a larger $\rho$ makes the worst-case training term larger and the complexity term smaller. SAM minimizes the first term and replaces the second by ordinary weight decay. The bound holds under the paper's technical conditions; it motivates the method, it is not what the experiments test.

:::key Putting the four ideas together
We train on $L_S$ but want low $L_D$ (idea 1), and the paper's bound links $L_D$ to the worst training loss in a ball around the weights (idea 4). The worst nudge is approximately along the gradient (idea 2), and the slope of the worst case is the slope at that nudged point (idea 3). One gradient to find the nudge, one at the nudged point to step: the two-gradient SAM update made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. A low-loss point is not always a robust solution

Two parameter vectors may have the same training loss. But one may lie in a narrow valley: a tiny parameter perturbation causes a large increase in loss. The other lies in a wide, flat region: neighboring parameters also have low loss. SAM trains for the *worst nearby loss*, not just the loss at the current iterate. Its motivation is generalization and robustness, not a strict-saddle escape theorem.

For empirical loss $L_S(w)$ and neighborhood radius $\rho$, the SAM objective (ignoring optional weight decay here) is

$$\min_w \max_{\|\epsilon\|_2 \le \rho} L_S(w + \epsilon).$$

The inner maximization is adversarial in *parameter space*, not adversarial examples in input space.

## 2. Worked example: same minimum value, different neighborhood risk

Compare two local quadratic losses at their respective minima $a$:

$$L_{\text{flat}}(w) = \tfrac{2}{2}(w - a)^2, \qquad L_{\text{sharp}}(w) = \tfrac{20}{2}(w - a)^2.$$

Both have $L(a) = 0$. But for $|\epsilon| \le \rho = 0.1$, the worst perturbation is at the boundary. Hence

$$\max_{|\epsilon| \le 0.1} L_{\text{flat}}(a + \epsilon) = 0.01, \qquad \max_{|\epsilon| \le 0.1} L_{\text{sharp}}(a + \epsilon) = 0.10.$$

The sharp solution looks just as good to ordinary training loss at $a$, yet SAM assigns it a much higher neighborhood penalty.

**Important:** This example illustrates the objective but is *not* a proof that all flatter minima generalize better. Parameter reparameterization and choice of neighborhood metric matter.

## 3. How SAM turns a minimax problem into two gradients

Linearize the inner loss near $w$:

$$L_S(w + \epsilon) \approx L_S(w) + \epsilon^\top g, \qquad g = \nabla L_S(w).$$

With an $\ell_2$ radius, the largest dot product occurs in the gradient direction:

$$\hat\epsilon = \rho\frac{g}{\|g\|_2} \qquad (g \ne 0).$$

Then approximate the outer update by

$$w^+ = w - \eta\nabla L_S(w + \hat\epsilon).$$

Thus one gradient finds the adverse neighboring point; a second evaluates the loss slope *there*. The practical method omits derivatives through the dependence of $\hat\epsilon$ on $w$ (the paper's Equations 2–3).

:::key Why "sharpness-aware" rather than "Hessian minimization"?
SAM uses a *finite-radius worst-case training-loss increase*. This agrees with curvature intuition in a local quadratic model, but is not identical to minimizing the largest Hessian eigenvalue everywhere.
:::

## 4. Main evidence and theoretical motivation

Section 2 motivates the objective using a PAC-Bayes-style generalization inequality involving worst-neighborhood empirical loss and a complexity term. That bound has technical assumptions and does not assert that a smaller Hessian eigenvalue always yields lower test error.

Section 3 studies image-classification datasets (CIFAR-10/100, ImageNet), transfer/fine-tuning and noisy labels. In the tested settings, replacing a base optimizer with SAM often reduces test error and improves robustness to label noise. The method typically requires an extra gradient calculation, approximately doubling the gradient work of a comparable basic update.

## 5. Generalization motivation and figure reading

The bound motivates population risk using worst-neighborhood *empirical* loss and a complexity term under technical conditions. The practical method minimizes a tractable surrogate, not the exact bound. That is why test error and robustness, not just neighborhood training loss, matter.

**Figure 2:** A plain gradient update uses the slope at the current parameters. SAM first moves toward an adversarial nearby parameter vector and then computes a new gradient there. Its final update responds to the nearby loss landscape, which explains the extra gradient cost.

**Caution:** Improved generalization in tested models is not a theorem guaranteeing fast saddle escape or global optimization.

## 5. Assumptions and limitations

**Requirements:** Chosen parameter-space norm $p$ (usually $p = 2$), radius $\rho$, stochastic batches, learning rate and optional weight decay. The generalization motivation relies on distributional/bound conditions; practical implementation uses a first-order inner approximation.

**Limits:** Gains depend on the architecture, batch handling, radius and data. Sharpness is sensitive to parameter scaling, and flatter neighborhoods are not a universal test-accuracy guarantee. SAM is not designed to prove finite-time escape from strict saddles or global optimization.

## 6. Connections to the survey

**Li et al. (2018)** warns that sharpness plots are affected by parameter scaling. **Galli (2026)** shows globally flat regions can still be undesirable for optimization. SAM instead targets *neighborhood-wise loss* and empirically better generalization, a distinct objective from **Jin's** finite-time saddle-escape guarantees.

## 7. Quick reference

| Concept | Meaning / equation |
|---|---|
| SAM objective | $\min_w \max_{\lVert\epsilon\rVert_2 \le \rho} L_S(w + \epsilon)$ |
| Inner perturbation | $\hat\epsilon = \rho g/\lVert g\rVert_2$ |
| Outer update | $w^+ = w - \eta\nabla L_S(w + \hat\epsilon)$ |
| Computational price | about two gradient evaluations / step |
| Original to read | Sections 2–4; Eqs. (1)–(3), Figs. 1–2 |

## 8. Papers Master note

:::plain Copy into Our notes
Introduces SAM, minimizing worst empirical loss within a parameter-radius neighborhood via a two-gradient approximation; reports improved test performance and label-noise robustness. Its flatness/generalization objective is distinct from formal saddle-escape or global-optimality guarantees.
:::

> **Original paper:** [View original research PDF](https://arxiv.org/pdf/2010.01412) *Worked numerical examples are pedagogical unless explicitly attributed to the authors.*
