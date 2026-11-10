## 1. When does better geometry repay its cost?

This chapter follows Sections VII and VIII of the survey. The main practical question is whether expensive geometric information saves enough expensive training operations to pay for itself. The trade-off differs by method:

- **K-FAC and Shampoo:** factor quality, refresh frequency, matrix roots and distributed synchronization all interact.
- **Hessian-free optimization:** the inner CG solve may dominate.
- **Muon:** matrix products may be efficient on accelerators, but eligible matrices and auxiliary AdamW groups complicate the memory comparison.

The survey asks research to report both the idealized optimizer arithmetic and the complete training-system cost. The dashboard of chapter 17 is a small example of the second.

## 2. Curvature under noise and non-convexity

A stochastic Hessian estimate can be indefinite or noisy (chapter 15). An empirical Fisher can differ substantially from the true Fisher (chapter 12). A structured gradient covariance may reflect loss curvature only indirectly (chapter 14). Open questions:
- which approximations are accurate enough along real training paths;
- how much damping or clipping they need;
- whether apparent acceleration survives when gradients and curvature factors are estimated from minibatches of different sizes.

The right benchmark includes noisy, poorly conditioned and rotated problems, not only nearly quadratic convex ones.

## 3. Stability beyond the quadratic picture

Chapter 4 derived the threshold $\eta < 2/\lambda_{\max}$ from a fixed quadratic. Section VI-E of the survey warns that it is not a universal prohibition for networks. On a fixed quadratic it is exact:

:::viz quad
:::

:::read How to read this visual
**What is drawn.** Gradient descent on the fixed quadratic $f = \tfrac12(20x^2 + y^2)$ in 3D, with the eigen-directions drawn in blue (curvatures $20$ and $1$). The chart shows the error $|x_t|$ along the sharp axis on a log scale.

**What moves.** The step size $\eta$, around the threshold $2/20 = 0.1$.

**What to notice.** Along the sharp axis, every step multiplies the error by $1 - 20\eta$. Below $\eta = 0.1$ it shrinks; above $0.1$ it flips sign and grows, and GD diverges, even though the flat axis would still converge. On a quadratic the Hessian never changes, so nothing can rescue the run.
:::

But a network's Hessian changes as it trains. Cohen et al. observed that full-batch GD drives the sharpness up to $2/\eta$ and then keeps training there, at the **[[edge-of-stability|edge of stability]]**.

:::viz mlp
:::

:::read How to read this visual
**What is drawn.** An offline replication: a small tanh network (10–48–48–1, 128 synthetic regression points) trained by full-batch GD at four step sizes. The main plot is the **[[sharpness]]**, the largest Hessian eigenvalue, measured by power iteration with Hessian–vector products (chapter 11). The dashed lines mark $2/\eta$ for each step size. The side chart is the training loss on a log scale.

**What to notice.** At the smallest step size, sharpness rises and settles below $2/\eta$. At the three larger ones, it rises until it reaches $2/\eta$, overshoots once, and then hovers right at the line, while the loss keeps falling. Locally the iterates are unstable, yet training continues. The quadratic threshold describes a fixed local model, not the global nonlinear path. This is why the survey treats stability, noise and generalization as separate questions.
:::

## 4. Matrix-aware methods need their own theory

Muon shows that useful optimizer geometry need not be an estimated inverse Hessian (chapter 16). The spectral-norm view explains one aspect of its update, the steepest linearized step. It leaves open the effects of momentum, nonlinearity, approximate orthogonalization, stochastic gradients and changing layer shapes. A theoretical comparison with Shampoo and SOAP should treat their actual operators, rather than relabel every matrix transform as "second order". It also remains to understand *which* parameter blocks benefit from matrix-aware updates, since implementations keep AdamW for embeddings and non-matrix parameters.

## 5. Invariance and parameterization

An optimizer's path can change when the same model is written in different coordinates. Euclidean and diagonal methods use the stored coordinates directly. Natural gradient attempts a geometrically meaningful transformation of predictive distributions (chapter 12). Structured and matrix-aware methods introduce their own invariances or biases. These properties matter for networks with normalization, scale symmetries, and layers of very different widths. One consequence: sharpness itself is not invariant.

:::viz rescale
:::

:::read How to read this visual
**What is drawn.** A one-neuron ReLU network $y = \theta_2\,\mathrm{ReLU}(\theta_1x)$ fit to the single point $(1, 1)$, over the plane of its two weights. Every point of the green valley $\theta_1\theta_2 = 1$ is a minimum, and every one of them computes *exactly the same function*, because $\mathrm{ReLU}(a\theta_1x)\cdot\theta_2/a = \mathrm{ReLU}(\theta_1x)\cdot\theta_2$ for $a > 0$. The ring is the minimum $(a, 1/a)$; blue marks the curved direction across the valley, amber the flat direction along it.

**What moves.** $a$ (log scale): sliding along the valley.

**What to notice.** The curvature across the valley, $2(a^2 + 1/a^2)$, grows without bound as $a$ moves away from $1$, although the function never changes. "Sharp" and "flat" describe the parameterization, not the function (Dinh et al.). So a lower sharpness, or a lower training loss, does not by itself establish better generalization. The survey places generalization in the evaluation protocol, not among the guaranteed outputs of an optimizer.
:::

## 6. Toward fair, system-aware benchmarks

Optimizer studies should state exactly what they count:
- number of updates;
- examples or tokens processed;
- gradient evaluations;
- GPU-hours;
- peak memory;
- communication volume;
- final held-out performance.

These are related, but not interchangeable. Training-scale studies make the trade-offs central. Public comparisons are more informative when they release the optimizer implementations, hyperparameter searches, mixed-precision settings, parameter-group exclusions and hardware assumptions with the results.

## 7. Conclusion: the survey's answers

All the methods interpret one object, the gradient of the training loss, in different ways:
- **GD** follows it in Euclidean geometry, and **SGD** estimates it from a minibatch.
- **Momentum** accumulates its history.
- **Adam and Adafactor** use coordinate-dependent gradient statistics.
- **Newton and quasi-Newton methods** add loss curvature.
- **Natural gradient and K-FAC** use Fisher-information geometry.
- **Shampoo and SOAP** use structured matrix statistics.
- **Sophia** estimates diagonal Hessian curvature.
- **Muon** reshapes a momentum update along its singular directions.

These mechanisms are related through local optimization, yet they are not interchangeable.

:::key The survey's answers
**RQ1.** First-order methods progressively trade simple, memory-light updates for historical and coordinate-wise information. Their improvements depend on tuning, stochasticity and model class (chapters 7 to 9).

**RQ2.** Second-order, curvature-aware and matrix-aware methods can better reflect anisotropy or structure. They pay for it with matrix information, approximation errors, extra hyperparameters, or restrictions on which parameters they apply to (chapters 10 to 16).

**The central lesson is methodological.** Distinguish Hessians from gradient moments, theorems from observed speedups, and update counts from full training costs. These distinctions are the foundation for reading current optimizer research without treating the newest algorithm as automatically the best.
:::

:::problem Where to go from here
Read the survey itself (the PDF link at the top of every page), with this guide beside it. For the underlying basics (gradients, step sizes, curvature, momentum, noise, Adam), the crash course builds each from scratch. The paper library has a page per paper for the references the two surveys share, including Adam, AdaGrad, Shampoo, the heavy ball, cubic regularization and the edge of stability.
:::
