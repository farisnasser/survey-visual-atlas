## 1. Start with ordinary gradient descent, in numbers

This chapter follows Section II-G of the survey. Even when the gradient is known exactly, an optimizer still has to decide how to turn it into a step. Ordinary gradient descent takes $\Delta = -\eta g$. With $g = (4, 2)^\top$ and $\eta = 0.1$,

$$\Delta = -0.1\begin{bmatrix}4\\2\end{bmatrix} = \begin{bmatrix}-0.4\\-0.2\end{bmatrix}.$$

This treats a move of a given size as equally "expensive" in every direction. The survey's question is: what if moving in some directions should cost more than in others?

## 2. A matrix that prices movement

The **[[variable-metric]]** idea uses a symmetric positive-definite matrix $M_t$ to say how strongly the optimizer may move in each direction. For example,

$$M = \begin{bmatrix}2 & 0\\ 0 & 1\end{bmatrix}$$

says that a step of a given length along the first coordinate is penalized twice as much as the same step along the second. The curves of equal penalty, $\Delta^\top M\Delta = \text{const}$, are circles when $M = I$ and ellipses otherwise. They are not the contours of the loss; they describe what the optimizer counts as a "small" step.

**Where the formula comes from.** Choose the step that makes the predicted change as negative as possible while paying for its size:

$$\Delta_t = \arg\min_\Delta\Big[\,g_t^\top\Delta + \frac{1}{2\eta}\Delta^\top M_t\Delta\,\Big].$$

The first term is the Taylor-predicted change in loss, which we want negative. The second penalizes large or "expensive" moves. Without it, we could make a negative-gradient step longer forever and the linear model would keep promising more, which is useless when Taylor is only local.

**One parameter first.** Set $g = 4$, $\eta = 0.1$ and $M = 2$. The problem becomes $\min_\Delta\,(4\Delta + 10\Delta^2)$. Differentiate, $4 + 20\Delta$, and set it to zero: $\Delta = -0.2$. Ordinary GD would have given $-0.4$. The movement price $M = 2$ halves the step.

**Now with matrices.** For symmetric $M$, the gradient of the bracket with respect to $\Delta$ is $g + \tfrac{1}{\eta}M\Delta$. Set it to zero, multiply by $\eta$, then by $M^{-1}$:

$$g + \tfrac{1}{\eta}M\Delta = 0 \;\Rightarrow\; M\Delta = -\eta g \;\Rightarrow\; \Delta_t = -\eta M_t^{-1}g_t, \qquad \theta_{t+1} = \theta_t + \Delta_t.$$

That is the survey's Eq. (12). Positive definiteness ($v^\top Mv > 0$ for every $v \ne 0$) makes the bracket strictly convex and $M$ invertible, so the step is unique.

**What $M^{-1}$ does in numbers.** Keep $g = (4, 2)^\top$ and $\eta = 0.1$, with $M = \mathrm{diag}(2, 1)$ and $M^{-1} = \mathrm{diag}(0.5, 1)$:

$$\Delta = -0.1\begin{bmatrix}0.5 & 0\\ 0 & 1\end{bmatrix}\begin{bmatrix}4\\2\end{bmatrix} = -0.1\begin{bmatrix}2\\2\end{bmatrix} = \begin{bmatrix}-0.2\\-0.2\end{bmatrix}.$$

The first gradient component was halved; the second was left alone. Transforming the gradient by $M^{-1}$ before stepping is what the survey calls **[[preconditioning]]**.

@@viz metric@@

## 3. One formula, many optimizers

The framework connects several families through their choice of $M_t$:

| method | interpretation of $M$ |
|---|---|
| gradient descent | $M = I$: every Euclidean direction is priced the same |
| diagonal adaptation (AdaGrad, Adam) | a diagonal, history-dependent $M$: one price per coordinate |
| Newton | $M = H$ when the Hessian is positive definite: the true local curvature |
| natural gradient | $M = F$ when the Fisher matrix is invertible: the geometry of the predictions |

These are conceptual connections, not identical algorithms in every implementation, and the survey is careful about the limits. The matrices carry different information: Adam's squared-gradient statistics are not estimates of the Hessian, and Muon's orthogonalization (Section V-G) changes the geometry of a matrix update without being a quadratic penalty of this kind at all.

The survey's Table I summarizes what each family constructs and what geometry it imposes, rather than calling every matrix "curvature":

| family | constructed quantity | geometric interpretation |
|---|---|---|
| SGD / momentum | gradient, past gradients | Euclidean update with history |
| Adam / Adafactor | squared-gradient statistics | coordinate-dependent scaling |
| Newton / BFGS | Hessian or secant approximation | local quadratic curvature |
| natural gradient / K-FAC | Fisher-information geometry | distributional displacement |
| Shampoo / SOAP | structured gradient statistics | tensor-coordinate preconditioning |
| Muon | orthogonalized momentum matrix | spectral-norm steepest direction |

:::idea In plain English
Every optimizer in the survey answers the same question, "how expensive is it to move in this direction?", in its own way. Gradient descent says all directions cost the same. Adam prices each coordinate separately. Newton prices directions by curvature, and natural gradient by how much the predictions would change.
:::

:::key What you should remember
The unifying step is $\Delta = -\eta M^{-1}g$, but the meaning of $M$ matters as much as the algebra. The value of the framework is not that all optimizers are equivalent; it is a common starting point for asking what each one changes about the gradient, and why.
:::

:::problem The next step
With these foundations in place, Section III begins with gradient descent itself and examines how first-order methods improve it. Chapter 7 starts there.
:::
