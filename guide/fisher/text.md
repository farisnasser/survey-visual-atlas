## 1. Three different matrices

This chapter follows Sections IV-H, IV-I and IV-J of the survey and Appendix C-D. Deep-learning papers use "second order" loosely. The survey separates three matrices.

**The Hessian.** $H = \nabla^2L$ is the second derivative of the chosen scalar objective. For a composite loss $L(\theta) = \ell(f_\theta)$, where the network $f_\theta$ produces outputs $f_k$ and the loss $\ell$ scores them, the chain rule splits it in two (survey Section IV-H):

$$H = J^\top H_\ell J + \sum_k \frac{\partial\ell}{\partial f_k}\nabla_\theta^2 f_k.$$

Here $J$ is the Jacobian of the outputs with respect to the parameters and $H_\ell$ is the Hessian of the loss with respect to the outputs.

**The Gauss–Newton matrix.** The first term, $G = J^\top H_\ell J$, is the **[[ggn|generalized Gauss–Newton]]** matrix. When $H_\ell \succeq 0$, as for squared error or cross-entropy, $G \succeq 0$ even if the full Hessian is indefinite. Dropping the second term is an approximation, not an identity. Its usefulness is that $G$ never has negative curvature, so Hessian-free solvers (chapter 11) get well-posed linear systems.

**The Fisher matrix.** For a probabilistic model $p_\theta(y \mid x)$, the **[[fisher|Fisher information]]** is (survey Section IV-H)

$$F(\theta) = \mathbb{E}_{x,\,y \sim p_\theta}\left[\nabla_\theta\log p_\theta(y\mid x)\,\nabla_\theta\log p_\theta(y\mid x)^\top\right].$$

The expectation is over labels drawn *from the model itself*. Replacing them with the observed labels gives the **[[empirical-fisher|empirical Fisher]]**, a gradient outer-product estimate. It does not generally equal $F$, $G$ or $H$. In well-specified likelihood problems some identities between them hold in expectation, but they must not be applied to arbitrary non-convex losses.

| matrix | definition | always positive semidefinite? | relation to the Hessian |
|---|---|---|---|
| Hessian $H$ | $\nabla^2L$ | no | itself |
| Gauss–Newton $G$ | $J^\top H_\ell J$ | yes, when $H_\ell \succeq 0$ | $H$ without the second term of the split above |
| Fisher $F$ | $\mathbb{E}_{p_\theta}[\nabla\log p\,\nabla\log p^\top]$ | yes | related only through identities in well-specified cases |
| empirical Fisher | outer products of observed-label gradients | yes | does not generally equal $F$, $G$ or $H$ |

The survey's added Table VI extends this list to Adam's second moment, Sophia's diagonal estimate, the K-FAC factors and Shampoo's factors (chapters 13 to 16).

## 2. Natural gradient: geometry of predictions

Change the parameter coordinates by a smooth invertible map, say from $\sigma$ to $\log\sigma$. The model and its predictions are the same, but Euclidean distance in the new coordinates is different, so ordinary gradient descent follows a different path. **[[natural-gradient|Natural gradient]]** asks for a step that is small in terms of the *predictions* instead. Under regularity conditions, the Kullback–Leibler divergence between the model's predictions at nearby parameters is (survey Section IV-I)

$$\mathrm{KL}(p_\theta\,\|\,p_{\theta+\Delta}) \approx \tfrac12\Delta^\top F(\theta)\Delta.$$

So the Fisher matrix is exactly the metric that measures how much a step changes the predictions. Use it as $M$ in the variable-metric model of chapter 6 and the step is (survey Eq. 35)

$$\Delta_{\mathrm{NG}} = -\eta F^{-1}g,$$

with damping or a pseudo-inverse if $F$ is singular.

**A derivation from a KL budget (Appendix C-D).** Ordinary GD asks which *Euclidean* step decreases the linear model most. Natural gradient instead limits how much the predictions may change (survey Eq. 52):

$$\min_\Delta\, g^\top\Delta \quad\text{subject to}\quad \tfrac12\Delta^\top F\Delta \le r^2/2.$$

With $F \succ 0$ and the constraint active, the Lagrangian is $\mathcal{L}(\Delta, \lambda) = g^\top\Delta + \tfrac\lambda2(\Delta^\top F\Delta - r^2)$. Stationarity gives $g + \lambda F\Delta = 0$, so $\Delta = -\lambda^{-1}F^{-1}g$. The radius $r$ fixes the scalar $\lambda$; the *direction* is $-F^{-1}g$. Its goal is a small change in the predictions, which is distinct from minimizing a Hessian quadratic model. That is why natural gradient is sometimes called second order but is not simply an inverse-Hessian step.

:::viz natgrad
:::

:::read How to read this visual
**What is drawn.** The plane of the two parameters of a Gaussian model $N(\mu, \sigma^2)$: mean $\mu$ across, standard deviation $\sigma$ up. The background shading is the log of the loss gap: the average negative log-likelihood of data with mean $1$ and spread $0.5$, minus its minimum. The green dot is the optimum $(1, 0.5)$. All four methods start at the white dot $(-1.5, 2.5)$ with the same step size.
- Violet: GD on the parameters $(\mu, \sigma)$.
- White: GD on $(\mu, \log\sigma)$.
- Coral: natural gradient with the Fisher matrix of $(\mu, \sigma)$, which is $\mathrm{diag}(1/\sigma^2, 2/\sigma^2)$.
- Amber dashed: natural gradient with the Fisher matrix of $(\mu, \log\sigma)$, which is $\mathrm{diag}(1/\sigma^2, 2)$.

All paths are drawn in the same $(\mu, \sigma)$ plane.

**What moves.** The common step size $\eta$.

**What to notice.** The two GD paths are different curves, although they optimize the same model: renaming a parameter changed the algorithm. The two natural-gradient paths nearly lie on top of each other; the readout measures the largest distance between each pair of paths over the first 40 steps. With finite steps the invariance is only approximate; it is exact in the limit of small steps. One part is exact at any step size: natural gradient's mean update is $\mu \leftarrow \mu - \eta(\mu - 1)$ in both parameterizations, so the mean moves the same fraction $\eta$ of the way to the data mean at every step, whatever $\sigma$ is. Speed is not the point here. At the default step size, the two GD runs happen to reach the optimum in fewer steps than natural gradient; the readout counts them. The point is that natural gradient's path depends on the predictions, not on how the model is written down. Raise $\eta$ and GD in $(\mu, \sigma)$ can overshoot to $\sigma \le 0$, where the model breaks; it is then stopped.
:::

## 3. What the second-order theory really guarantees

Section IV-J of the survey summarizes the guarantees:
- **Full Newton**, on a sufficiently smooth, locally strongly convex problem: local quadratic convergence (chapter 10).
- **BFGS**, under appropriate line-search and regularity assumptions: local superlinear behaviour (chapter 11).
- **The basic non-convex GD bound** of chapter 7: it controls only the best gradient norm.
- **Trust-region and cubic-regularization methods**: stronger global complexity results for certain stationarity criteria, but at the cost of solving model subproblems.

None of these results says that a large network trains faster in wall-clock time, or generalizes better. A guarantee must be read together with its oracle assumptions, its local or global scope, and its objective class. The survey's added Table V lists all of them side by side (chapter 17).

:::problem The next step
For a network layer, $F$ has $(mn)^2$ entries. Section V of the survey begins with the first structured answer: K-FAC.
:::
