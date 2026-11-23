## 1. When does better geometry repay its cost?

This chapter follows Sections VII and VIII of the survey. Section VII names open questions rather than declaring any family the winner. The first is economic. Suppose Shampoo makes each step more effective but takes longer to compute. When does the benefit outweigh the overhead? K-FAC and Shampoo face a refresh trade-off: recomputing their matrix statistics more often keeps them representative of the current loss landscape but adds overhead; refreshing less often is cheaper but staler. Hessian-free methods pay with inner iterations per step. Muon's orthogonalization runs well on accelerators, but its cost depends on the shapes of the weight matrices, and it usually shares the model with another optimizer for the remaining parameters.

So the value of extra geometric information cannot be read from the update rule alone. It depends on the architecture, the matrix shapes, the hardware, the phase of training and how much training remains. The open problem is not only better mathematics; it is deciding when more mathematical information is economically worth it, especially as models and hardware change.

## 2. Curvature under noise and non-convexity

Curvature changes as the network trains, so a curvature estimate can go stale. Estimated from minibatches, it is also noisy: BFGS infers curvature from gradient differences that sampling can corrupt, and a noisy or indefinite Hessian estimate can make an update unstable. The Fisher matrix and gradient statistics do not, in general, carry the same information as the Hessian. Damping and clipping control the damage, but they are extra choices, and stronger regularization can cancel the benefit of the curvature itself.

The open question: how accurate must a curvature estimate be to improve training at all, under different levels of gradient noise and different landscapes, and how does its quality change as optimization progresses?

Chapter 4 derived the threshold $\eta < 2/\lambda_{\max}$ from a fixed quadratic. On a fixed quadratic it is exact; on a real network, the curvature itself moves.

@@viz quad@@

@@viz mlp@@

## 3. Matrix-aware methods need their own theory

Muon changes singular values without estimating the Hessian; Shampoo and SOAP use gradient-statistical matrices; K-FAC approximates Fisher information. A convergence theorem proved for one of these constructions does not transfer to the others, and it is misleading to treat them all as "second-order". For Muon in particular, the spectral-norm picture of chapter 16 explains a single linearized step, but real updates are shaped by momentum, stochastic gradients and an approximate orthogonalization. There is room for theory that matches the actual update operation in stochastic non-convex training, and for understanding which parameters benefit from matrix-aware updates (Muon is usually applied to hidden-layer matrices, with AdamW for the rest), which could guide better hybrid designs.

## 4. Invariance and parameterization

The same network function can be written with different parameter values. Rescaling the weights of some layers can leave the output unchanged while changing the gradients, so gradient descent and coordinate-wise adaptive methods can follow different paths on what is really the same model. An optimizer is **invariant** to a reparameterization if it behaves the same either way. Natural gradient is designed around this, because it measures steps by their effect on the predicted distribution (chapter 12); matrix-aware methods have their own, different structural properties. Greater invariance does not guarantee faster training or better generalization, and normalization layers and parameter scaling change the picture again. Testing optimizers under equivalent parameterizations of the same network would help separate an optimizer's real principles from artifacts of one representation.

@@viz rescale@@

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
