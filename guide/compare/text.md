## 1. Compare the mathematics, not the names

This chapter follows Section VI of the survey. Each family is a response to a different bottleneck:

- **GD** has the exact gradient direction but crawls in ill-conditioned valleys; **SGD** saves computation at the cost of noise.
- **Momentum** averages directions and can accelerate persistent motion, but it adds a dynamical state and possible overshoot.
- **Adam** changes per-coordinate step sizes using past gradients. That helps with heterogeneous features, but its statistics are not curvature estimates, and its guarantees depend on assumptions.
- **Newton** solves a local quadratic model with the Hessian; **L-BFGS** approximates the inverse curvature from secant pairs.
- **K-FAC, Shampoo and SOAP** exploit matrix structure; **Sophia** estimates diagonal Hessian curvature; **Muon** imposes a matrix-norm geometry.

The survey's central point: *all of these can be written as formulas, but not all formulas use the same kind of information*. Multiplying $g$ by $H^{-1}$ uses the second derivative of the loss. Multiplying by a diagonal built from squared gradients uses historical stochastic statistics. Replacing singular values by one optimizes a linear model under a spectral-norm budget. On a toy problem these can produce similar-looking trajectories, yet differ in invariance, response to noise and cost.

## 2. Count the actual tensors

The survey's Table II counts the persistent **[[optimizer-state|optimizer state]]**: what each method keeps between steps, excluding weights, gradients, activations, master copies and temporary buffers.

| method | persistent state | order | main per-step overhead |
|---|---|---|---|
| GD / minibatch SGD | none | $O(d^0)$ | one gradient evaluation |
| heavy-ball momentum | velocity vector | $O(d)$ | vector arithmetic |
| Adam / AdamW | first- and second-moment arrays | $2d$ | elementwise operations |
| Lion | one momentum array | $d$ | elementwise sign and moments |
| Adafactor (matrix, lean) | row and column second-moment sums | $m + n$ | factored reconstruction |
| dense Newton | full Hessian or factorization | $O(d^2)$ | second derivatives, linear solve |
| L-BFGS | $k$ recent pairs $(s_i, y_i)$ | $O(kd)$ | two-loop recursion, line search |
| K-FAC (one layer) | two Fisher factors | $O(m^2 + n^2)$ | factor updates and solves |
| Shampoo (one matrix) | row and column accumulators | $O(m^2 + n^2)$ | inverse roots and products |
| Sophia | momentum and diagonal-curvature estimates | $O(d)$ | occasional curvature estimation |
| Muon (eligible matrices) | momentum, plus temporary workspace | $O(d)$ | Newton–Schulz matrix products |

These are scalar counts, not bytes; bytes depend on precision, and mixed-precision training often keeps optimizer states in full precision. As chapter 14 showed, $m^2 + n^2$ is not always smaller than $mn$.

:::viz memory
:::

:::read How to read this visual
**What is drawn.** For one weight matrix with $m$ rows and $n$ columns, the number of values each optimizer keeps between steps, on a log scale, using the counts of the survey's Table II. The colours follow the legend from left to right; the number above each bar is its count. The dashed line is $mn$, the size of the weights themselves.

**What moves.** The number of rows $m$ and columns $n$, each a power of two from $16$ to $16{,}384$.

**What to try.** Start at $4096 \times 1024$, then make the matrix square, then very thin.

**What to notice.** Momentum, Lion and Muon keep one copy of the weights' size; Adam and Sophia keep two; Adafactor keeps only $m + n$, a tiny fraction of Adam's. K-FAC and Shampoo keep $m^2 + n^2$. That is astronomically less than the full curvature matrix of $(mn)^2$ values, but it is never less than Adam's $2mn$, because $m^2 + n^2 - 2mn = (m - n)^2 \ge 0$. It equals Adam's count only for a square matrix, and for a thin one it is far larger. This is the survey's point in Section VI-B: structured does not mean smaller than diagonal, and these counts still leave out activations, precision and temporary buffers.
:::

## 3. Arithmetic, communication and wall-clock time

Per-step cost is not captured by memory. SGD and Adam do elementwise work after the gradient. Newton solves a system, quasi-Newton methods manipulate vector histories, K-FAC and Shampoo maintain and factor matrices, and Muon adds repeated matrix multiplications. Matrix operations can run efficiently on GPUs, while "cheap" elementwise operations can be limited by memory bandwidth, so relative speed depends on shapes, precision, implementation and hardware. Distributed training adds communication: curvature factors may need averaging across workers, and techniques such as ZeRO move optimizer state between devices.

The fair question is therefore not only *how many iterations reach a target loss?* It is also *how many gradient evaluations, matrix products, accelerator seconds and memory bytes does that target require?* SOAP's own numbers make the point: over 40% fewer iterations, but over 35% less wall-clock time.

## 4. Guarantees must match the problem

The survey lines up four statements that are often juxtaposed:

- $O(1/T)$ function error for GD on smooth convex losses;
- $O(1/T^2)$ function error for accelerated GD on smooth convex losses;
- $O(1/T)$ for the *best squared gradient norm* in smooth deterministic non-convex GD (chapter 7);
- $O(1/\sqrt T)$ for some stochastic non-convex gradient-norm criteria (chapter 5).

They measure different things. Reaching $\|g\| \le \epsilon$ takes $O(\epsilon^{-2})$ iterations by Eq. (15), and $O(\epsilon^{-4})$ by the stochastic bound. These are worst-case statements about different oracle models, not evidence that practical SGD needs quadratically more time than GD. Similarly, Newton's local quadratic convergence (chapter 10), Shampoo's convex regret bound (chapter 14) and Adam-type non-convex results under extra conditions (chapter 8) cannot be ranked by their exponent of $T$. The survey's added Table V lists each statement with its objective class, success criterion and oracle. In the survey's words, comparing theorem statements without checking their hypotheses is a category error.

## 5. Stability, noise and generalization are separate questions

- **Stability.** The Hessian changes along the path, and the threshold $2/\lambda_{\max}$ of chapter 4 is a local statement. Training can continue at the edge of stability despite local instability (chapter 18), but large learning rates can also genuinely diverge.
- **Noise.** It can help escape strict saddles under suitable assumptions, yet it also disturbs curvature estimates. Matrix-aware and curvature-factor methods are sensitive to finite batches, regularization and update frequency, so their robustness needs testing across batches and initializations.
- **Generalization.** It is a third axis. Rescaling some network parameters can change Hessian sharpness without changing the function computed (chapter 18). A lower training loss, or a stronger optimization theorem, does not by itself establish better test accuracy.

## 6. A reproducible protocol, run

Section VI-F proposes a controlled study:
- one differentiable toy loss, with both an axis-aligned and a rotated ill-conditioned quadratic (chapters 4 and 6 did this);
- one small network task with fixed architecture, data split, initializations, loss, batch size and budget;
- comparable tuning effort for every optimizer;
- several seeds, with the implementation stated, including parameter groups and precision.

It asks for four plots: loss versus gradient evaluations, loss versus wall-clock time, peak memory versus model size, and performance versus learning rate. The survey's Fig. 9 is that dashboard, from real runs on this laptop.

:::survey Fig. 9 of the survey: recommended evaluation dashboard. Matching compute, data, initialization, and tuning effort is necessary before interpreting empirical optimizer comparisons.
public/survey3/fig9.webp
:::

:::read How to read this figure
**The setup** (printed under the panels, and in full in the survey's Table VIII). An MLP with layers 64–128–128–10 on the scikit-learn digits; 600 updates, batch 64, 5 seeds; fp32 on one laptop CPU. Each of the seven optimizers gets the same 9-point learning-rate grid.

**The four panels.**
- **(a)** Training loss against gradient evaluations at each optimizer's best learning rate: the median of 5 seeds, with a band from the best to the worst seed.
- **(b)** Loss against measured optimizer wall-clock on one thread.
- **(c)** Measured optimizer-state memory against model size.
- **(d)** Median final loss against the learning rate, divided by each optimizer's best.

**What to notice.**
- **The ordering depends on the axis.** In (a), Muon reaches the lowest loss per gradient evaluation. In (b) its extra matrix work costs time, and plain Shampoo, with an exact eigendecomposition every step, is slowest by far.
- **Memory.** In (c), Adafactor's factored state is about 60 times smaller than Adam's, and Shampoo's grows fastest.
- **Learning-rate sensitivity.** In (d), the V shapes show how forgiving each method is. Adafactor's relative step size makes it nearly flat to the right of its best.

None of this ranks optimizers in general. It is one small task, measured carefully, exactly as the survey's protocol warns.
:::

:::viz dashboard
:::

:::read How to read this visual
**What is drawn.** The same measured runs as Fig. 9, one panel at a time, with one colour per optimizer: SGD with momentum, Adam, Lion, Adafactor, Shampoo, Muon and Sophia-H (see the legend).

**What moves.** The panel, and in panel (a), the learning rate. The slider moves every optimizer the same number of grid steps (each a factor of $\sqrt{10}$) away from its own best.

**What to try.** In panel (a), move the learning rate one or two grid steps either way. Then compare panels (a) and (b).

**What to notice.**
- **Panel (a) is a snapshot of one tuning.** One grid step away from their best, several optimizers lose an order of magnitude or more in final loss, and further away some diverge; the readout gives each final loss. A comparison at one shared learning rate would mostly measure which optimizer that rate happens to suit, which is the survey's warning about unequal tuning.
- **Panel (b) re-weights panel (a) by cost per step.** The readout gives each method's total optimizer time.
- **Panel (c) measures real tensors**, not formulas. At width 128: about 0.10 MiB for SGD with momentum, Lion and Muon; 0.20 MiB for Adam and Sophia-H; 0.003 MiB for Adafactor; 0.53 MiB for Shampoo, including its cached inverse roots.
- **Panel (d) counts** how many of the 9 grid points land within a factor of 10 of each optimizer's best loss.
:::

## 7. What each family establishes, and what it does not

The survey's Table III ends the comparison:

| family | main strength | not automatically implied |
|---|---|---|
| SGD / momentum | simple, low auxiliary overhead | fastest wall-clock convergence |
| Adam / AdamW | coordinate adaptation | Hessian approximation or global optimum |
| Adafactor / Lion | lower auxiliary state in selected regimes | better loss on every task |
| Newton / L-BFGS | local or secant curvature | stable global non-convex progress |
| K-FAC | structured Fisher approximation | exact Hessian or cheap communication |
| Shampoo / SOAP | tensor geometry and rotated adaptation | lower total memory than all diagonal methods |
| Sophia | estimated diagonal Hessian and clipping | universal reported speedup |
| Muon | orthogonalized hidden-weight updates | a second-order Hessian guarantee |

Its compact summary is deliberately qualitative. A method should be chosen by matching its representation of geometry to the task's scale, parameter structure, noise and compute:
- a full-Hessian method is elegant but unusable for billions of parameters;
- a diagonal method is economical but misses rotations;
- a matrix-aware method uses structure but can carry substantial fixed overhead.

:::problem The next step
Chapter 18 closes the survey: the open problems, and its answers to RQ1 and RQ2.
:::
