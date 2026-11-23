## 1. Compare the mathematics, not the names

This chapter follows Section VI of the survey. Sections III to V asked how each method builds its update. Section VI asks a harder question: how should these methods be compared fairly? Every optimizer still produces $\theta_{t+1} = \theta_t + \Delta_t$, and its name says little about the information used to build $\Delta_t$:

- SGD and momentum use current and past gradients.
- Adam and Adafactor use gradient statistics to scale coordinates.
- Newton, BFGS and Sophia use exact or approximate *loss curvature*, obtained in different ways.
- Natural gradient and K-FAC use Fisher-based *statistical geometry*.
- Shampoo and SOAP use *structured gradient statistics*.
- Muon *orthogonalizes* a momentum matrix instead of approximating any curvature.

The survey's Table IV collects every update in one place. Its central caution is that **matrix-valued does not mean second order**. If you see an inverse matrix, ask: inverse of what? An eigenbasis: eigenvectors of which statistic? Singular values: is curvature being estimated, or is an update being normalized?

| method | update (one step) | remembers | what the remembered quantity is |
|---|---|---|---|
| GD / SGD | $\theta \leftarrow \theta - \eta g$ | nothing | — |
| heavy ball | $v \leftarrow \mu v + g$, $\theta \leftarrow \theta - \eta v$ | velocity $v$ | running sum of past gradients |
| Adam | $\theta \leftarrow \theta - \eta\hat m/\sqrt{\hat v + \varepsilon}$ | $m$, $v$ | smoothed direction; squared-gradient scale (not a Hessian) |
| Lion | $\theta \leftarrow \theta - \eta\,\mathrm{sign}(\text{momentum blend})$ | $m$ | momentum; only its sign is used |
| Newton | solve $H\Delta = -g$ | (the Hessian) | the exact local curvature |
| natural gradient | $\Delta = -\eta F^{-1}g$ | (the Fisher) | how much a step changes the predictions |
| K-FAC | step $\propto S^{-1}GA^{-1}$ | $A$, $S$ per layer | two small factors of a Fisher block |
| Shampoo | $\Delta W = -\eta L^{-1/4}GR^{-1/4}$ | $L$, $R$ per matrix | row and column gradient statistics |
| Muon | $\Delta W = -\eta\,UV^\top$ | momentum $M$ | no curvature: singular values set to one |

## 2. Optimizer-state memory: count the actual tensors

Training memory holds weights, gradients, activations, and each optimizer's persistent state. The survey's Table II counts only that last part, the **[[optimizer-state|optimizer state]]**; comparing optimizers means not mixing these categories. With $d$ parameters and a weight matrix $W \in \mathbb{R}^{m\times n}$:

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

**A worked example.** A $100\times100$ weight matrix has $mn = 10{,}000$ weights. Adam's two moment arrays hold $20{,}000$ values. Two full square factors hold $10{,}000 + 10{,}000 = 20{,}000$ values: *not less* than Adam. A full curvature block for the layer would be $10{,}000\times10{,}000$, $100$ million entries. That is the survey's nuance: structured factors are far smaller than full curvature, yet not smaller than coordinate-wise state. For a dense layer with $200$ outputs and $100$ inputs, the counts are $400{,}000{,}000$ for the full block, $200^2 + 100^2 = 50{,}000$ for two square factors, and $200 + 100 = 300$ for two vectors.

:::survey Fig. 9 of the survey: persistent optimizer state for one 4096 × 1024 weight matrix, counted as in Table II. K-FAC and Shampoo store far less than the full curvature matrix, but never less than Adam's two moment arrays, because m² + n² ≥ 2mn.
public/survey3/fig9.webp
:::

:::read How to read this figure
**What is drawn.** For one $4096\times1024$ weight matrix, the number of values each optimizer keeps between steps, on a logarithmic axis, with the count printed beside each bar. The dashed line is the size of the weights themselves, $mn \approx 4.2$ million.

**What to notice.** Momentum, Lion and Muon keep one copy of the weights' size; Adam and Sophia keep two; Adafactor keeps only $5{,}120$. K-FAC and Shampoo keep $m^2 + n^2 \approx 17.8$ million, more than Adam's $8.4$ million, while the full curvature matrix would need $17.6$ trillion. Since $m^2 + n^2 - 2mn = (m - n)^2 \ge 0$, two square factors never store less than Adam's two moment arrays.
:::

:::viz memory
:::

:::read How to read this visual
**What is drawn.** For one weight matrix with $m$ rows and $n$ columns, the number of values each optimizer keeps between steps, on a log scale, using the counts of the survey's Table II. The colours follow the legend from left to right; the number above each bar is its count. The dashed line is $mn$, the size of the weights themselves.

**What moves.** The number of rows $m$ and columns $n$, each a power of two from $16$ to $16{,}384$.

**What to try.** Start at $4096 \times 1024$, then make the matrix square, then very thin.

**What to notice.** Momentum, Lion and Muon keep one copy of the weights' size; Adam and Sophia keep two; Adafactor keeps only $m + n$, a tiny fraction of Adam's. K-FAC and Shampoo keep $m^2 + n^2$. That is astronomically less than the full curvature matrix of $(mn)^2$ values, but it is never less than Adam's $2mn$, because $m^2 + n^2 - 2mn = (m - n)^2 \ge 0$. It equals Adam's count only for a square matrix, and for a thin one it is far larger. This is the survey's point in Section VI-B: structured does not mean smaller than diagonal, and these counts still leave out activations, precision and temporary buffers.
:::

These are scalar counts, not bytes: buffers, factor refreshes, numerical precision and temporary workspace matter too, and mixed-precision or distributed training may keep extra copies.

## 3. Arithmetic, communication and wall-clock time

A method can need fewer steps and still train more slowly, because not every step costs the same. Two students cover the same distance: A takes $10$ steps of $20$ seconds, $200$ seconds in all; B takes $30$ steps of $2$ seconds, $60$ seconds. Fewer steps, much later arrival. In optimizer terms: if A needs $100$ updates at $0.5$ seconds and B needs $250$ updates at $0.1$ seconds, A takes $50$ seconds and B $25$.

- SGD mainly pays for one gradient and cheap arithmetic.
- Hessian-free methods may run many inner matrix–vector products per outer step.
- K-FAC builds, solves and periodically refreshes factors.
- Shampoo and SOAP need inverse roots or eigendecompositions.
- Muon needs several matrix multiplications to orthogonalize.

Matrix operations can run efficiently on GPUs, while "cheap" elementwise work can be limited by memory bandwidth, so actual speed depends on shapes, kernels and hardware. Distributed training adds communication to synchronize statistics or factors. SOAP's reported results show the gap: over $40\%$ fewer iterations, but over $35\%$ less wall-clock time.

:::viz budget
:::

:::read How to read this visual
**What is drawn.** Two optimizers racing to the same target loss. Each bar is the total time, the number of updates multiplied by the seconds per update; the split shows the two factors. The defaults are the worked example above: A needs $100$ updates at $0.5$ s, B needs $250$ updates at $0.1$ s.

**What moves.** Each optimizer's number of updates and its cost per update.

**What to try.** Give A even fewer updates, then raise its cost per update until it loses.

**What to notice.** The winner is set by the product, not by either factor. The readout gives the break-even cost per update: the most A can spend per step and still tie B. A method with a better direction wins only if its extra work per step stays below that line.
:::

:::key Remember VI-B and VI-C
Count persistent state separately from total memory, and count time, not iterations. A mathematically strong search direction is useful only if the time and resources needed to produce it are justified.
:::

## 4. Guarantees must match the problem

Theorems that sound comparable often measure different things. If a theorem says $L(\theta_T) - L_* \le C/T$, the left side is the gap in loss value after $T$ steps; with $C = 100$ the bound is $10$ at $T = 10$, $1$ at $T = 100$, $0.1$ at $T = 1000$. A bound is a guarantee, not the loss measured in a run.

- **Gradient descent on a smooth non-convex loss.** With a suitable learning rate and a loss bounded below, Eq. (15) bounds the smallest squared gradient norm over $T$ steps by $2[L(\theta_0) - L_*]/(\eta T)$: progress toward an approximately stationary point, not the global minimum.
- **Newton.** Quadratic convergence is a strong *local* statement near a suitable minimum.
- **BFGS.** Local superlinear results depend on regularity, the problem class and the line search; noisy minibatches can break their assumptions.
- **Trust-region and cubic methods.** Non-convex complexity results target first- or second-order stationarity under stated derivative access.

**Why $O(1/T)$ and $O(1/\sqrt{T})$ cannot be ranked by their exponents.** It is tempting to say $1/T$ shrinks faster, so that method must be better. But a convex theorem may bound the loss gap, $L(\theta_T) - L_* = O(1/T)$, while a stochastic non-convex theorem bounds an expected squared gradient norm, $\mathbb{E}\|\nabla L\|^2 = O(1/\sqrt{T})$. These are different targets under different assumptions, and a small gradient does not even mean a minimum. It is like comparing "distance left to the destination" with "walking speed".

The survey's Table V lists every guarantee it quotes with its setting and what it measures:

| result | holds when | what it measures |
|---|---|---|
| GD: error $O(1/T)$ | smooth convex loss | $L(\theta_T) - L_*$ |
| Nesterov: error $O(1/T^2)$ | smooth convex loss, the right coefficients | $L(\theta_T) - L_*$ |
| GD: $O(\epsilon^{-2})$ steps to $\lVert g\rVert \le \epsilon$ | $\beta$-smooth, bounded below, $\eta \le 1/\beta$ | smallest $\lVert g_t\rVert^2$ seen |
| SGD: $O(\epsilon^{-4})$ steps | as above, plus unbiased gradients with variance $\le \sigma^2/B$ | average $\mathbb{E}\lVert g_t\rVert^2$ |
| Newton: quadratic convergence | close to a minimum, $H \succ 0$, Lipschitz Hessian | $\lVert\theta_t - \theta^*\rVert$ |
| stability: $\eta < 2/\lambda_{\max}$ | fixed quadratic (local model) | does the error shrink? |
| Shampoo; AdaGrad, AMSGrad | stochastic or online convex problems | regret |
| random perturbations escape strict saddles | the cited papers' assumptions | approximate second-order stationarity |

:::idea In plain English
Before calling one method "more convergent", ask: what problem class, what starting point, what measure of error, what cost per step, and what assumptions about noise?
:::

## 5. Stability, noise and generalization are different questions

A run can show unstable updates, noisy minibatch gradients, or poor test accuracy, and these are three different phenomena. **Stability** is about how updates behave in the local geometry: chapter 4's threshold $\eta < 2/\lambda_{\max}$ comes from a fixed quadratic, yet studies of the **[[edge-of-stability|edge of stability]]** show networks that keep making progress while their curvature hovers at or above that threshold. That does not make large learning rates safe; it shows the local analysis does not describe nonlinear training completely. **Gradient noise** is the gap between minibatch and full gradients; it can help move away from saddles but can interfere with curvature estimates. **[[generalization|Generalization]]** is performance on unseen data. Lower training loss does not guarantee higher test accuracy, and measures of **[[sharpness]]** depend on how the network is parameterized, so they do not by themselves predict generalization.

## 6. A reproducible comparison protocol, run

If one experiment used a larger batch, longer training, another initialization or a kinder learning-rate schedule, an improvement cannot be credited to the optimizer. The survey's protocol starts from a quadratic whose curvature is known (an axis-aligned and a rotated valley, as in chapter 4), then a neural-network task with the same model, data split, budget and initializations, comparable tuning effort, and several seeds. It asks for more than one measure: training loss, wall-clock time, memory, and sensitivity to the learning rate. The survey's Fig. 10 is that dashboard, from real runs on this laptop.

:::survey Fig. 10 of the survey: recommended evaluation dashboard. Matching compute, data, initialization, and tuning effort is necessary before interpreting empirical optimizer comparisons.
public/survey3/fig10.webp
:::

:::read How to read this figure
**The setup** (printed under the panels). An MLP with layers 64–128–128–10 on the scikit-learn digits; 600 updates, batch 64, 5 seeds; fp32 on one laptop CPU. Each of the seven optimizers gets the same 9-point learning-rate grid.

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
**What is drawn.** The same measured runs as Fig. 10, one panel at a time, with one colour per optimizer: SGD with momentum, Adam, Lion, Adafactor, Shampoo, Muon and Sophia-H (see the legend).

**What moves.** The panel, and in panel (a), the learning rate. The slider moves every optimizer the same number of grid steps (each a factor of $\sqrt{10}$) away from its own best.

**What to try.** In panel (a), move the learning rate one or two grid steps either way. Then compare panels (a) and (b).

**What to notice.**
- **Panel (a) is a snapshot of one tuning.** One grid step away from their best, several optimizers lose an order of magnitude or more in final loss, and further away some diverge; the readout gives each final loss. A comparison at one shared learning rate would mostly measure which optimizer that rate happens to suit, which is the survey's warning about unequal tuning.
- **Panel (b) re-weights panel (a) by cost per step.** The readout gives each method's total optimizer time.
- **Panel (c) measures real tensors**, not formulas. At width 128: about 0.10 MiB for SGD with momentum, Lion and Muon; 0.20 MiB for Adam and Sophia-H; 0.003 MiB for Adafactor; 0.53 MiB for Shampoo, including its cached inverse roots.
- **Panel (d) counts** how many of the 9 grid points land within a factor of 10 of each optimizer's best loss.
:::

A small mathematical diagnostic shows which directions a preconditioner handles; a training experiment shows whether that benefit survives overhead and noise. The survey is a literature comparison: its contribution is the comparison framework, not a reproduction of every original benchmark.

## 7. What each family establishes, and what it does not

The survey's Table III summarizes each family's main strength and what it does *not* establish by itself:

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

:::key The central result of Section VI
Optimizers differ in what information they use to modify gradients. First-order methods use current and past gradients; second-order methods use curvature or its approximations; information-geometric methods measure how predictions change; structured methods use the matrix organization of the weights. These operations may improve a step, but they add memory, computation and implementation cost, so a fair comparison needs both the mathematics and an end-to-end account of resources.
:::

:::problem The next step
Chapter 18 turns to what remains unknown, and to the survey's answers to its two research questions.
:::
