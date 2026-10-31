# AdaGrad Avoids Saddle Points

:::minute The paper in one minute
**Problem:** Classical GD has one fixed scalar step size, whereas AdaGrad changes the geometry of its updates using the *history of observed gradients*. Can a stable-manifold argument still show that adaptive methods avoid saddle points almost everywhere?

**Principal result:** AdaGrad's changing coordinate scales do not inherently destroy the strict-saddle avoidance phenomenon: the trajectories analyzed in the paper avoid saddles from almost all initializations.

**Scope:** It is an asymptotic avoidance statement, not a worst-case iteration bound for escaping a saddle; it also does not automatically transfer to Adam, RMSProp, large-step training or nonsmooth neural-network objectives.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. "Avoids saddles" is a statement about a thin set of starts

A **strict saddle** is a critical point $x^*$ ($\nabla f(x^*) = 0$) whose Hessian has at least one negative eigenvalue, $\lambda_{\min}(\nabla^2 f(x^*)) < 0$. Other eigenvalues may be zero. Along the negative direction the function curves down, so a gradient step pushes small offsets away from $x^*$.

Some starts still converge to the saddle: those with exactly zero component along every unstable direction. This **stable set** is a thin surface, of lower dimension than the space. Thin sets have zero volume, so a start drawn from any continuous distribution lands on one with probability zero. "Avoids saddles from almost any initialization" means exactly this: the bad starts form a zero-volume set.

### 2. A preconditioner is a step size per direction

The paper studies GD with a matrix-valued step size, the **preconditioner** $\Gamma_t$:

$$x_{t+1} = x_t - \Gamma_t \nabla f(x_t).$$

AdaGrad builds $\Gamma_t$ from all gradients seen so far. With $G_t = \delta_0^2 I + \sum_{s \le t} \nabla f(x_s)\nabla f(x_s)^\top$, the full-matrix version takes $\Gamma_t = G_t^{-1/2}$; the diagonal version keeps only the diagonal of the sum; the scalar version uses $1/\sqrt{\delta_0^2 + \sum_{s\le t}\|\nabla f(x_s)\|^2}$. Coordinates that have seen large gradients get small steps. Because $\Gamma_t$ is **positive definite**, $\nabla f^\top \Gamma_t \nabla f > 0$ whenever $\nabla f \neq 0$: the step still points downhill, measured in a different metric.

### 3. A change of metric keeps the count of negative directions

The **inertia** of a symmetric matrix is its number of positive, negative and zero eigenvalues. Sylvester's law of inertia says $D^{1/2} H D^{1/2}$ has the same inertia as $H$ for any $D \succ 0$. A preconditioner can tilt and stretch the unstable direction; it cannot remove it.

Take $H = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$, with eigenvalues $3$ and $-1$, and $D = \mathrm{diag}(1, 0.25)$. The product $DH$ has eigenvalues $\approx 1.693$ and $\approx -0.443$. The sizes change, the signs do not: one direction still has multiplier $1 + 0.443\,\eta > 1$ near the saddle.

### 4. A history-dependent step makes the map change over time

The classical avoidance proof (Lee et al.) applies to one fixed map $g(x) = x - \eta\nabla f(x)$, used at every step. AdaGrad's map $x \mapsto x - \Gamma_t \nabla f(x)$ changes with $t$, and $\Gamma_t$ depends on the whole past trajectory: a **non-autonomous** system. The fixed-map theorem does not apply as stated.

What rescues the argument is that $\Gamma_t$ settles down. Scalar AdaGrad on $f(x) = \tfrac12 x^2$ with $\delta_0 = 1$, $x_0 = 1$ takes the step $\Gamma_0 \approx 0.707$, then $0.692$, $0.691$, and stays at $\approx 0.691$: the gradients shrink fast enough that their squared sum stays finite. The paper proves this in general (its Proposition 1): $\sum_t \|\nabla f(x_t)\|^2 < \infty$ and $\Gamma_t$ converges to a positive definite limit.

:::key Putting the four ideas together
AdaGrad is GD in a metric $\Gamma_t$ that changes with time (ideas 2 and 4). In any fixed positive definite metric a strict saddle keeps an unstable direction (idea 3). Because $\Gamma_t$ converges to a positive definite limit, the late iterations look like a fixed preconditioned GD plus a small, shrinking remainder, and the bad starts again form a zero-volume set (idea 1). How the paper handles the history dependence is made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. The research problem

Classical GD has one fixed scalar step size, whereas AdaGrad changes the geometry of its updates using the *history of observed gradients*. Can a stable-manifold argument still show that adaptive methods avoid saddle points almost everywhere? The paper addresses this for the AdaGrad family, including scalar, diagonal and full-matrix preconditioning.

## 2. First principles: AdaGrad is a changing metric

For a simple diagonal AdaGrad rule, define the accumulated squared-gradient vector $s_k$ and preconditioner $D_k$:

$$s_k = s_{k-1} + g_k \odot g_k, \qquad D_k = \mathrm{diag}\big((s_k + \delta\mathbf{1})^{-1/2}\big), \qquad x_{k+1} = x_k - \eta D_k g_k.$$

These familiar equations are a *teaching representation* of diagonal AdaGrad; the paper analyzes more general variants. The preconditioner gives different effective steps to different coordinates, which can reduce the conditioning problem but complicates dynamical systems analysis because $D_k$ depends on the entire trajectory. At a strict saddle $x^*$ with Hessian $H$, a fixed positive-definite matrix $D$ would induce a local linearization

$$e_{k+1} \approx (I - \eta D H)\, e_k.$$

Even if $DH$ is not symmetric, it is similar to $D^{1/2} H D^{1/2}$, which is symmetric and has the same inertia as $H$ (for $D \succ 0$). Thus negative curvature still yields an unstable mode. The challenge is that *actual* AdaGrad has a time-dependent, history-dependent $D_k$.

## 3. Method and main results

The authors develop step-size stabilization arguments and a recursive representation for the adaptive preconditioner, allowing center-stable manifold methods to be applied despite the gradient-history dependence. They establish almost-everywhere saddle avoidance for AdaGrad variants, including scalar, diagonal and full-matrix versions, under the required regularity and initialization conditions.

:::key Principal result
AdaGrad's changing coordinate scales do not inherently destroy the strict-saddle avoidance phenomenon: the trajectories analyzed in the paper avoid saddles from almost all initializations.
:::

## 4. Why this is not just an application of the GD theorem

In GD with fixed $\eta$, the update $G(x) = x - \eta\nabla f(x)$ depends only on the present state. For AdaGrad, the state is better represented as *parameters plus optimizer history*. Two runs at the same $x$ but with distinct accumulated gradients may take different next steps. A theorem for constant $D$ therefore does not automatically apply: the authors' novel work is in controlling this additional dynamical state.

## 5. Assumptions and survey relevance

The theorem uses the paper's differentiability, step-size stabilization and almost-everywhere initialization hypotheses. It is an asymptotic avoidance statement, not a worst-case iteration bound for escaping a saddle; it also does not automatically transfer to Adam, RMSProp, large-step training or nonsmooth neural-network objectives. Global optimality requires a separate landscape theorem.

:::warn Scope
"AdaGrad avoids saddles" under the specified theory does not mean an adaptive preconditioner guarantees efficient escape from every degenerate saddle.
:::

This is particularly relevant to **RQ2** (adaptive dynamics), **RQ3** (negative curvature versus avoidance), and **RQ4** (modern-optimizer theory).

## 6. Our notes

This paper is a useful bridge from the GD saddle-avoidance results to adaptive optimizers. AdaGrad is harder to analyze because the update remembers past gradients, but that history does not automatically make saddle convergence likely. The result is reassuring for AdaGrad specifically, without claiming that the same theory already covers every Adam-style optimizer used in deep learning.

> **Original source:** [proceedings.mlr.press/v162/antonakopoulos22a.html](https://proceedings.mlr.press/v162/antonakopoulos22a.html). **Verification:** official ICML/PMLR version; diagonal update and fixed-metric example are pedagogical.
