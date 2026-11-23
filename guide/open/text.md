## 1. When does better geometry repay its cost?

This chapter follows Sections VII and VIII of the survey. Section VII names open questions rather than declaring any family the winner. The first is economic. Suppose Shampoo makes each step more effective but takes longer to compute. When does the benefit outweigh the overhead? K-FAC and Shampoo face a refresh trade-off: recomputing their matrix statistics more often keeps them representative of the current loss landscape but adds overhead; refreshing less often is cheaper but staler. Hessian-free methods pay with inner iterations per step. Muon's orthogonalization runs well on accelerators, but its cost depends on the shapes of the weight matrices, and it usually shares the model with another optimizer for the remaining parameters.

So the value of extra geometric information cannot be read from the update rule alone. It depends on the architecture, the matrix shapes, the hardware, the phase of training and how much training remains. The open problem is not only better mathematics; it is deciding when more mathematical information is economically worth it, especially as models and hardware change.

## 2. Curvature under noise and non-convexity

Curvature changes as the network trains, so a curvature estimate can go stale. Estimated from minibatches, it is also noisy: BFGS infers curvature from gradient differences that sampling can corrupt, and a noisy or indefinite Hessian estimate can make an update unstable. The Fisher matrix and gradient statistics do not, in general, carry the same information as the Hessian. Damping and clipping control the damage, but they are extra choices, and stronger regularization can cancel the benefit of the curvature itself.

The open question: how accurate must a curvature estimate be to improve training at all, under different levels of gradient noise and different landscapes, and how does its quality change as optimization progresses?

Chapter 4 derived the threshold $\eta < 2/\lambda_{\max}$ from a fixed quadratic. On a fixed quadratic it is exact; on a real network, the curvature itself moves.

:::viz quad
:::

:::read How to read this visual
**What is drawn.** Gradient descent on the fixed quadratic $f = \tfrac12(20x^2 + y^2)$ in 3D, with the eigen-directions drawn in blue (curvatures $20$ and $1$). The chart shows the error $|x_t|$ along the sharp axis on a log scale.

**What moves.** The step size $\eta$, around the threshold $2/20 = 0.1$.

**What to notice.** Along the sharp axis, every step multiplies the error by $1 - 20\eta$. Below $\eta = 0.1$ it shrinks; above $0.1$ it flips sign and grows, and GD diverges, even though the flat axis would still converge. On a quadratic the Hessian never changes, so nothing can rescue the run.
:::

:::viz mlp
:::

:::read How to read this visual
**What is drawn.** An offline replication: a small tanh network (10–48–48–1, 128 synthetic regression points) trained by full-batch GD at four step sizes. The main plot is the **[[sharpness]]**, the largest Hessian eigenvalue, measured by power iteration with Hessian–vector products (chapter 11). The dashed lines mark $2/\eta$ for each step size. The side chart is the training loss on a log scale.

**What to notice.** At the smallest step size, sharpness rises and settles below $2/\eta$. At the three larger ones, it rises until it reaches $2/\eta$, overshoots once, and then hovers right at the line, while the loss keeps falling. Locally the iterates are unstable, yet training continues. The quadratic threshold describes a fixed local model, not the global nonlinear path. This is why the survey treats stability, noise and generalization as separate questions.
:::

## 3. Matrix-aware methods need their own theory

Muon changes singular values without estimating the Hessian; Shampoo and SOAP use gradient-statistical matrices; K-FAC approximates Fisher information. A convergence theorem proved for one of these constructions does not transfer to the others, and it is misleading to treat them all as "second-order". For Muon in particular, the spectral-norm picture of chapter 16 explains a single linearized step, but real updates are shaped by momentum, stochastic gradients and an approximate orthogonalization. There is room for theory that matches the actual update operation in stochastic non-convex training, and for understanding which parameters benefit from matrix-aware updates (Muon is usually applied to hidden-layer matrices, with AdamW for the rest), which could guide better hybrid designs.

## 4. Invariance and parameterization

The same network function can be written with different parameter values. Rescaling the weights of some layers can leave the output unchanged while changing the gradients, so gradient descent and coordinate-wise adaptive methods can follow different paths on what is really the same model. An optimizer is **invariant** to a reparameterization if it behaves the same either way. Natural gradient is designed around this, because it measures steps by their effect on the predicted distribution (chapter 12); matrix-aware methods have their own, different structural properties. Greater invariance does not guarantee faster training or better generalization, and normalization layers and parameter scaling change the picture again. Testing optimizers under equivalent parameterizations of the same network would help separate an optimizer's real principles from artifacts of one representation.

:::viz rescale
:::

:::read How to read this visual
**What is drawn.** A one-neuron ReLU network $y = \theta_2\,\mathrm{ReLU}(\theta_1x)$ fit to the single point $(1, 1)$, over the plane of its two weights. Every point of the green valley $\theta_1\theta_2 = 1$ is a minimum, and every one of them computes *exactly the same function*, because $\mathrm{ReLU}(a\theta_1x)\cdot\theta_2/a = \mathrm{ReLU}(\theta_1x)\cdot\theta_2$ for $a > 0$. The ring is the minimum $(a, 1/a)$; blue marks the curved direction across the valley, amber the flat direction along it.

**What moves.** $a$ (log scale): sliding along the valley.

**What to notice.** The curvature across the valley, $2(a^2 + 1/a^2)$, grows without bound as $a$ moves away from $1$, although the function never changes. "Sharp" and "flat" describe the parameterization, not the function (Dinh et al.). So a lower sharpness, or a lower training loss, does not by itself establish better generalization. The survey places generalization in the evaluation protocol, not among the guaranteed outputs of an optimizer.
:::

## 5. Conclusion: the whole survey in one argument

This section follows Section VIII. The survey begins with one decision: *given a gradient, how should an optimizer turn it into a parameter update?* Gradient descent uses a scalar learning rate. SGD makes the gradient cheaper but noisier. Momentum and Adam make better use of gradient history. Newton adds the Hessian to account for local curvature, but full curvature is expensive and can mislead in non-convex regions. BFGS, L-BFGS and Hessian-free methods address different parts of that cost. K-FAC, Shampoo, SOAP, Sophia and Muon show the much wider range of informed, matrix-structured updates.

:::key The survey's answers
**RQ1.** First-order methods extend gradient descent without estimating curvature. Momentum makes better use of previous gradient directions, adaptive methods scale each parameter by its gradient history, and newer designs cut memory and simplify the update (chapters 7 to 9).

**RQ2.** Second-order methods use curvature to account for how the gradient changes; natural-gradient methods use the sensitivity of the predicted distribution; structured and matrix-aware methods exploit how the parameters are organized to make otherwise expensive updates affordable. They do not all estimate the same quantity, even when their update rules look alike (chapters 10 to 16).

**The overall lesson.** More informative updates do not automatically mean more efficient training. The gain must be weighed against memory and compute, and a convergence guarantee is only as relevant as its assumptions. Understanding an optimizer means asking what information it uses and how that information shapes the update; that gives a way to choose methods for the problem and the resources at hand, rather than a ranking with a universal winner.
:::

:::idea One sentence to finish
The best optimization method is not decided by mathematical sophistication alone, but by the usefulness of its information, the geometry of its update, and its end-to-end cost for the problem being solved.
:::

:::problem Where to go from here
Read the survey itself (the PDF link at the top of every page) with this guide beside it. For the basics (gradients, step sizes, curvature, momentum, noise, Adam), the crash course builds each from scratch, and the paper library has a page per paper for many of the survey's references.
:::
