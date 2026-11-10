## 1. Every step minimizes a model plus a penalty

This chapter follows Sections II-G and II-H of the survey. Chapter 2 showed that the gradient step lowers the linear model $g^\top\Delta$ fastest among steps of a fixed Euclidean length. "Length" was the choice. Measure it differently and the best step changes. The survey writes this as one subproblem: for a positive-definite matrix $M_t$, the **[[variable-metric]]** step is (survey Eq. 12)

$$\Delta_t = \arg\min_\Delta\left[g_t^\top\Delta + \frac{1}{2\eta}\Delta^\top M_t\Delta\right].$$

The first term is the linear model: it rewards moving downhill. The second is a penalty on the step's size, measured in the metric $M_t$: it is large for steps that $M_t$ considers long. Differentiate with respect to $\Delta$ and set the result to zero: $g_t + M_t\Delta/\eta = 0$, so (survey Eq. 13)

$$\boxed{\Delta_t = -\eta M_t^{-1}g_t.}$$

This single formula covers much of the survey, depending only on the choice of $M_t$:

- $M_t = I$: Euclidean gradient descent.
- $M_t$ diagonal and built from past gradients: per-coordinate adaptation, as in AdaGrad and Adam (chapter 8).
- $M_t = H$, a positive-definite Hessian: a Newton-like step (chapter 10).
- $M_t = F$, the Fisher matrix: natural gradient (chapter 12).

Another way to see the same step: among all steps on the ellipse $\Delta^\top M\Delta = r^2$, the one that lowers the linear model most is $\Delta \propto -M^{-1}g$. The metric decides what "equally long" means, and the step is the steepest one under that definition.

:::viz metric
:::

:::read How to read this visual
**What is drawn.** Level sets (grey) of a quadratic loss with curvatures $6$ and $0.6$ along axes turned by $30^\circ$; the green dot is the minimum. At the white point, the short white arrow is the direction $-g$. The coloured dashed ellipse is the set of steps of equal length in the chosen metric, $\Delta^\top M\Delta = r^2$. The thick coloured arrow is the step $\Delta = -\eta M^{-1}g$ of Eq. (13). The faint dashed line through its tip is a level line of the linear model $g^\top\Delta$.

**What moves.** The metric $M$ ($I$, $\mathrm{diag}(H)$, $H$, or $\mathrm{diag}(|g|)$), the step size $\eta$, and the point.

**What to try.** Switch between the four metrics at the same point. With $M = H$, set $\eta = 1$.

**What to notice.**
- **The tangency.** The faint line always just touches the ellipse at the arrow's tip. The step is exactly the point of the ellipse where the linear model is lowest.
- **$M = I$.** The ellipse is a circle and the step is $-\eta g$, perpendicular to the level set.
- **$M = \mathrm{diag}(H)$.** The ellipse is stretched along the axes only, so the step turns, but it cannot follow the tilted valley.
- **$M = H$.** The ellipse has the shape of the loss's own level sets. The step points at the minimum, and with $\eta = 1$ it lands on it: the readout's loss drops to zero.
- **$M = \mathrm{diag}(|g|)$.** This is what Adam's first step does (chapter 8). It gives $\Delta = -\eta\,\mathrm{sign}(g)$: a step of the same size in every coordinate, at $45^\circ$ to the axes.

The readout gives the angle between the step and $-g$, and the loss before and after.
:::

## 2. A lens, not an identity

The survey insists that Eq. (13) is a *conceptual* relationship. Two warnings:

- **Adam's statistics are not a Hessian.** Writing Adam as $M = \mathrm{diag}(\sqrt{\hat v_t})$ does not mean that its gradient moments estimate curvature. They are averages of squared gradients (chapter 8).
- **Muon is not described exactly.** Muon's geometry comes from the spectral norm of a matrix, which is generally not a quadratic metric $\Delta^\top M\Delta$ at all (chapter 16).

These distinctions are why the survey's Table I lists *what is constructed* and *what geometry it gives*, rather than calling every matrix "curvature":

| family | constructed quantity | geometric interpretation |
|---|---|---|
| SGD / momentum | gradient, past gradients | Euclidean update with history |
| Adam / Adafactor | squared-gradient statistics | coordinate-dependent scaling |
| Newton / BFGS | Hessian or secant approximation | local quadratic curvature |
| natural gradient / K-FAC | Fisher-information geometry | distributional displacement |
| Shampoo / SOAP | structured gradient statistics | tensor-coordinate preconditioning |
| Muon | orthogonalized momentum matrix | spectral-norm steepest direction |

## 3. What counts as cost

The survey separates the *information* an optimizer uses from what it *costs*. Storage first, for $d$ parameters:

- The parameter vector needs $O(d)$ numbers; a full Hessian needs $d^2$.
- For one $m \times n$ weight matrix, a full metric over its $mn$ entries needs $m^2n^2$ numbers.
- Separate row and column factors need $m^2 + n^2$ (K-FAC, Shampoo; chapters 13 and 14).
- A factored second moment needs about $m + n$ (Adafactor, chapter 9).

These are storage *orders*, not end-to-end byte counts. Copies of parameters, gradients, momentum, sharding across devices and mixed precision all change the real peak memory. Chapter 17 counts the actual bytes in a real run.

What is counted as an "oracle" call (one gradient, one Hessian–vector product, one linear solve) matters as much as the number of iterations.

## 4. What counts as convergence

Iteration counts are just as easy to misread. A rate of $O(1/T)$ for *function error* on a convex problem cannot be ranked against $O(1/\sqrt{T})$ for the *expected squared gradient norm* on a non-convex stochastic one. They measure different quantities, under different noise models and accuracy criteria. The survey compares theorems only after naming three things for each: its target quantity, its assumptions, and its oracle. Chapter 17 lines up every theorem in the survey this way.

:::problem The next step
The preliminaries are complete. Section III of the survey begins with the simplest choices of $M$: none, then a memory of past gradients.
:::
