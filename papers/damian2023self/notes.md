# Self-Stabilization at the Edge of Stability

:::minute The paper in one minute
**Problem.** Cohen et al. observed *progressive sharpening*: the top Hessian eigenvalue rises until its product with the learning rate is around two, then often hovers near this boundary while training loss keeps falling non-monotonically. Damian et al. seek an analytical mechanism for this surprising behavior, not merely another measurement.

**Mechanism.** When $S > 2/\eta$, oscillations along $u$ grow; the quadratic-in-amplitude drift pushes toward lower sharpness. This is the self-stabilizing feedback: the instability itself helps change the landscape location.

**Main result.** With suitable analytical conditions, the authors show that GD near the edge of stability is closely approximated by dynamics resembling projected gradient descent constrained by $S(\theta) \le 2/\eta$.

**Caveat.** "GD tracks constrained descent" is an *approximate dynamical explanation*, not a universal theorem that any architecture, optimizer, stochastic minibatch or step size converges globally.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Past the stability line, GD zig-zags outward

Along a direction of curvature $\lambda$, one GD step multiplies the displacement $x$ by $1 - \eta\lambda$. With $\eta = 1$ and $\lambda = 1.8$ the multiplier is $-0.8$: from $x = 0.1$ the iterates go $-0.08$, $0.064$, a zig-zag that shrinks. With $\lambda = 2.2$ it is $-1.2$: $0.1$, $-0.12$, $0.144$, a zig-zag that grows. The **sharpness** $S$ is the largest Hessian eigenvalue, and $S = 2/\eta$ is the line between the two behaviours. A quadratic would stay on whichever side it started; a neural network loss is not a quadratic.

### 2. Curvature changes as the weights move

The sharpness depends on where we are: $S(\theta)$ has its own gradient $\nabla S$, which involves the third derivative of the loss. Take the toy loss $L(x, y) = \tfrac12(2 + y)x^2$, where $y$ is a slow coordinate. Along $x$ the curvature is $2 + y$, and $\partial L/\partial y = x^2/2$. So a large swing in $x$ creates a gradient in $y$ that lowers the curvature: with $\eta = 1$ and $x = 1$, one step moves $y$ by $-0.5$. The oscillation itself feeds back on the sharpness. This is what the notes' cubic term $\tfrac{x^2}{2}\nabla S(\theta)$ expresses.

### 3. Two coupled numbers: the loop the paper writes down

The paper condenses the mechanism into a two-variable map (its Eq. 3). Let $x_t$ be the displacement along the top eigenvector and $y_t = S(\theta_t) - 2/\eta$ how far the sharpness is above the line:

$$x_{t+1} = -(1 + \eta y_t)\,x_t, \qquad y_{t+1} = y_t + \eta\Big(\alpha - \beta\frac{x_t^2}{2}\Big).$$

Here $\alpha$ is the rate of progressive sharpening and $\beta = \|\nabla S\|^2$. With $\eta = 1$, $\alpha = 0.01$, $\beta = 1$: while $x \approx 0$, $y$ rises by $0.01$ per step; once $y > 0$, $|x|$ grows; at $|x| = 0.5$, $y$ changes by $0.01 - 0.125 = -0.115$ per step; the sharpness falls once $|x| > \sqrt{2\alpha/\beta} \approx 0.14$. Then $y < 0$, $|x|$ shrinks again, and the cycle starts over. In this two-number approximation the cycles grow from one to the next; the paper's theorem tracks the full dynamics over a finite horizon.

:::key Putting the three ideas together
Above $S = 2/\eta$ the iterates zig-zag outward along the top eigenvector (idea 1). The growing swing creates a gradient that lowers the sharpness (idea 2), and the coupled map shows the loop that results: sharpen, blow up, self-stabilize, return (idea 3). Averaged over the loop, GD behaves like descent constrained to $S \le 2/\eta$, which the notes below state as the paper's main result.
:::
:::

:::divider Study notes
:::

## 1. Problem: why does large-step gradient descent not simply diverge?

Cohen et al. observed *progressive sharpening*: the top Hessian eigenvalue rises until its product with the learning rate is around two, then often hovers near this boundary while training loss keeps falling non-monotonically. Damian et al. seek an analytical mechanism for this surprising behavior, not merely another measurement.

## 2. Derive the stability threshold

Near a stationary point $\theta^*$, write $e_t = \theta_t - \theta^*$ and approximate $\nabla L(\theta_t) \simeq He_t$. GD gives

$$e_{t+1} \simeq (I - \eta H)e_t.$$

Along a positive-curvature eigenvector with eigenvalue $\lambda$, the component multiplies by $1 - \eta\lambda$. Local stability needs $|1 - \eta\lambda| < 1$, or $0 < \eta\lambda < 2$. Define *sharpness* $S(\theta) = \lambda_{\max}(\nabla^2 L(\theta))$; the relevant top-direction boundary is $S \simeq 2/\eta$.

## 3. The third-order mechanism

Let $u$ denote a locally selected top eigenvector and $x$ the signed displacement in direction $u$. A local expansion in that direction includes terms beyond quadratic:

$$L(\theta + xu) \approx L(\theta) + x\,\nabla L(\theta)^T u + \frac{x^2}{2}S(\theta) + O(x^3).$$

Differentiating with respect to the slow coordinates produces a curvature-gradient coupling of approximate size

$$\nabla_\theta L(\theta + xu) \supset \frac{x^2}{2}\nabla S(\theta), \qquad \Delta\theta_{\text{slow}} \supset -\eta\frac{x^2}{2}\nabla S(\theta).$$

When $S > 2/\eta$, oscillations along $u$ grow; the quadratic-in-amplitude drift pushes toward lower sharpness. Oscillations can then contract when stability is restored. This is the self-stabilizing feedback: the instability itself helps change the landscape location.

:::theorem Main result
With suitable analytical conditions, the authors show that GD near the edge of stability is closely approximated by dynamics resembling projected gradient descent constrained by $S(\theta) \le 2/\eta$. They make quantitative predictions for loss, sharpness and deviations, tested theoretically in their regime and empirically on training examples.
:::

## 4. The important distinction: observed vs proved

The article's experiments motivate a broad phenomenon, but the proof uses a controlled local description, smoothness and other assumptions. "GD tracks constrained descent" is an *approximate dynamical explanation*, not a universal theorem that any architecture, optimizer, stochastic minibatch or step size converges globally.

:::plain Survey relevance (RQ2, RQ4)
Classical small-step arguments require descending loss each step. Here GD is often non-monotone; the coupled sharpness–oscillation dynamics explain why the longer-time loss can still decrease. Do not conflate the stability set $\{S \le 2/\eta\}$ with a valley floor or a set of globally optimal parameters.
:::

## 5. Our notes

The new idea is that going unstable in a sharp direction can automatically push GD into a less sharp part of the loss surface. That offers a mechanism for the edge of stability, but it depends on a particular local smooth model and should not be quoted as a blanket guarantee for Adam or all large-step training.

> **Original source:** [https://openreview.net/pdf/5ce767b4f951bb7db5b2c04af424d1c59c1e9387.pdf](https://openreview.net/pdf/5ce767b4f951bb7db5b2c04af424d1c59c1e9387.pdf).
