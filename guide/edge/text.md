## 1. Why $2/\eta$ is a wall

This chapter follows Section IV-C of the survey. The **[[sharpness]]** of the loss at a point $\theta$ is its largest Hessian eigenvalue, $S(\theta) = \lambda_{\max}(\nabla^2 L(\theta))$: the curvature of the steepest direction. Chapter 5 showed that on a quadratic, GD is stable only while $S < 2/\eta$. The first visual makes that wall concrete on the example the edge-of-stability paper itself starts with.

:::viz quad
:::

:::read How to read this visual
**What is drawn.** The quadratic $f = \tfrac12(20x^2 + y^2)$, with curvature $20$ along $x$ (sharp) and $1$ along $y$ (flat). The blue tubes are the two eigen-directions. The surface height is drawn as $\log(1 + f)$ so the bowl stays visible. The chart plots $|x_t|$, the distance along the sharp direction, on a log scale.

**What moves.** The coral ball is plain gradient descent with step $\eta$, from a point near the flat axis.

**What to try.** Move the step size from $0.04$ to $0.12$, crossing $\eta = 0.1$, which is $2/20$.

**What to notice.** Along the sharp axis, each step multiplies $x$ by $1 - 20\eta$. Below $\eta = 0.1$ this is between $-1$ and $1$: the ball zig-zags inward. Above it, the factor is below $-1$: the ball bounces with growing amplitude and diverges, even though the flat direction is perfectly happy. The readout gives the factor and compares the sharpness $20$ with $2/\eta$. On the chart the dots follow the dashed prediction $|1 - 20\eta|^t$, a straight line on the log scale. **One direction that is too sharp is enough to break GD.**
:::

So a careful practitioner should keep $\eta < 2/S$ throughout training. Real training does not.

## 2. Progressive sharpening and the edge of stability

Cohen, Kaur, Li, Kolter & Talwalkar (2021) documented, across architectures and tasks, that full-batch gradient descent typically goes through two phases.

1. **[[progressive-sharpening|Progressive sharpening]].** While $S < 2/\eta$, the sharpness rises steadily. GD has an "overwhelming tendency" to move into sharper regions.
2. **The [[edge-of-stability|edge of stability]] (EoS).** The sharpness reaches approximately $2/\eta$ and then **hovers at or slightly above it**. Meanwhile the loss decreases non-monotonically over short time scales, but consistently over long ones.

With heavy-ball or Nesterov momentum, the same behavior occurs at the correspondingly modified stability threshold. GD therefore does not avoid sharp regions by choosing a small step size. **It self-tunes the sharpness to the largest value its step size can tolerate.** The paper is empirical: it documents the phenomenon on fully connected networks, VGG and ResNet on CIFAR-10 subsets, a Transformer on WikiText-2, and smaller models, with both the squared loss and the cross-entropy. It does not prove a convergence theorem.

:::viz mlp
:::

:::read How to read this visual
**What is drawn.** An offline replication: a tanh network (10–48–48–1) trained by full-batch GD on 128 synthetic regression points with the squared loss, for 3000 steps, at four step sizes. The main plot shows the sharpness $\lambda_{\max}(\nabla^2 L)$ against the iteration, measured every 10 steps by power iteration on exact Hessian-vector products. The dashed lines are $2/\eta$ for each step size, in the same color. The chart below shows the training loss on a log scale.

**Colors.** Blue $\eta = 0.1$, coral $0.3$, violet $0.36$, white $0.42$.

**What to notice.** At $\eta = 0.1$ the sharpness rises and saturates far below its threshold $2/\eta = 20$, and the loss falls smoothly. At the three larger steps, progressive sharpening carries the sharpness up to $2/\eta$. It overshoots once, comes back, and then hovers right on the dashed line. On the loss chart, those runs show spikes and short-term increases, yet keep falling over the long run. The readout compares the late sharpness of each run with $2/\eta$. They match to a few hundredths: training lives on the boundary of the stable band of chapter 5.
:::

:::survey Survey Fig. 7. Original simulation of the edge of stability: a tanh network (20 → 96 → 96 → 1) trained by full-batch GD on 256 synthetic regression points, for four step sizes.
public/survey/eos_mlp.webp
:::

:::read How to read this figure
**Panel (a).** The training loss (squared error) on a log scale against the full-batch GD iteration, for $\eta = 0.1$ (teal), $0.2$ (green), $0.27$ (yellow) and $0.35$ (red). The two smaller steps decrease smoothly. The two larger steps show loss spikes at the moments their sharpness crosses $2/\eta$, then recover and keep decreasing.

**Panel (b).** The sharpness $\lambda_{\max}(\nabla^2 L)$, estimated every 5 steps by power iteration. The dotted lines are the thresholds $2/\eta = 10$, $7.41$ and $5.71$. For $\eta = 0.1$ and $0.2$, the sharpness saturates near $8.6$, below $2/\eta$: small steps plateau below their threshold. For $\eta = 0.27$ and $0.35$, it overshoots, is pushed back, and then locks onto $2/\eta = 7.41$ and $5.71$, to within $0.4\%$. This is a second, independent run of the same phenomenon, with a different network and data set.
:::

## 3. With momentum, the edge moves

For heavy ball and Nesterov momentum, the multiplier of chapter 5 is replaced by the roots of a two-step recursion, and the largest stable curvature changes. Cohen et al.'s Eq. (1) gives the **maximum stable sharpness**:

$$\text{heavy ball: } \frac{2 + 2\beta}{\eta}, \qquad \text{Nesterov: } \frac{2 + 2\beta}{\eta\,(1 + 2\beta)}.$$

With $\beta = 0$ both reduce to $2/\eta$. They report that, with momentum, the sharpness plateaus at exactly these values.

:::viz mom
:::

:::read How to read this visual
**What is drawn.** The maximum stable sharpness (MSS) of a quadratic, multiplied by $\eta$, against the momentum $\beta$. The curves are Cohen et al.'s Eq. (1) for heavy ball and for Nesterov. The dots are computed numerically, live: for each $\beta$, the largest curvature for which the two-step iteration's spectral radius stays below one, found by bisection.

**What to try.** Move $\beta$ from $0$ to $0.99$.

**What to notice.** The dots sit on the curves: the formula is exactly the stability boundary of the momentum iteration. Heavy ball tolerates more curvature as $\beta$ grows, up to almost $4/\eta$. Nesterov's edge stays lower. The readout gives both edges at $\eta = 0.01$, against plain GD's $2/\eta = 200$.
:::

## 4. Unstable convergence: a theorem that covers both instabilities

Ahn, Zhang & Sra (2022) called this behavior **[[unstable-convergence|unstable convergence]]** and studied it from first principles. Their first result is the bridge between the two halves of the survey.

:::theorem Ahn, Zhang & Sra 2022, Theorem 1 (informal)
Under a mild non-degeneracy assumption, gradient descent with step size $\eta$, from almost every initialization, cannot converge to a stationary point whose Hessian has **either a negative eigenvalue or an eigenvalue above $2/\eta$**.
:::

The theorem treats strict saddles and overly sharp minima in exactly the same way, as unstable fixed points of the GD map. Its proof follows the stable-manifold argument of chapter 7. In the language of chapter 5: a fixed point with any multiplier outside $[-1, 1]$ cannot attract a set of positive measure. So when no flat enough stationary point lies near the trajectory, GD cannot settle anywhere, and it keeps moving.

Ahn et al. identified two causes of unstable convergence. The first is the lack of flat stationary points near the GD trajectory. The second is the existence of forward-invariant sets of the GD dynamics, sets that the iterates cannot leave, which keep the unstable iterates from diverging. They characterize the regime by two quantities. One is a relative-progress ratio, $\big(L(\theta - \eta\nabla L(\theta)) - L(\theta)\big)/\big(\eta\lVert\nabla L(\theta)\rVert^2\big)$, which oscillates near zero instead of staying negative. The other is a directional smoothness along the update, which oscillates near $2/\eta$.

:::viz wells
:::

:::read How to read this visual
**What is drawn.** A loss with two minima of equal depth: a sharp one on the left, with sharpness $20.8$, and a flat one on the right, with sharpness $1.97$. Between them lies a strict saddle, on the ridge that separates the two basins. The surface is $0.06(x^2 + y^2)$ minus two Gaussian wells of widths $0.22$ and $0.75$. The dots are 357 starting points on a grid, each colored by where gradient descent ends from it after 4000 steps: violet for the sharp well, green for the flat well, white if it never settles.

**What moves.** *Drop the ball* runs one GD trajectory (coral) from $(-1.25, 0.75)$.

**What to try.** Raise the step size $\eta$ in stages: $0.05$, $0.08$, then just past $2/20.8 \approx 0.096$ (for example $0.1$), then $0.4$.

**What to notice.** At $\eta = 0.05$ and $0.08$, the two wells split the starts by basin (87 sharp, 270 flat), and no start ends at the saddle: chapter 7's avoidance. Just past $0.096$, the sharp well loses every one of its starts. Its multiplier along the sharp direction is now below $-1$, so it cannot hold GD, exactly as Ahn et al.'s theorem says. At $\eta = 0.1$ those starts do not settle anywhere: they keep bouncing near the sharp well (white, "never settles"), a bouncing orbit of the kind chapter 12 studies. At $\eta = 0.4$ every start ends in the flat well, the only stationary point left that is stable at this step. The readout reports the counts. The same theorem forbids both the saddle and the sharp minimum.
:::

:::warn What the edge of stability breaks
Full-batch GD at practical learning rates operates at $\eta\lambda_{\max} \approx 2$, far outside the regime $\eta < 1/\ell$ (and hence $\eta\lambda_{\max} < 1$) assumed by the saddle-avoidance theorems of chapter 7, and outside the descent lemma of chapter 3. The loss is not monotone. The classical convergence analyses do not describe this regime. That is the problem the next two chapters address: why does training not blow up, and what path does it take?
:::

:::key What to keep from this chapter
- Sharpness $S = \lambda_{\max}(\nabla^2 L)$. On a quadratic, GD is stable only while $S < 2/\eta$.
- In real training with full-batch GD, the sharpness rises (progressive sharpening) until it reaches $2/\eta$, then hovers there (the edge of stability), while the loss keeps falling over the long run.
- With momentum, the edge moves to $(2 + 2\beta)/\eta$ (heavy ball) or $(2 + 2\beta)/(\eta(1 + 2\beta))$ (Nesterov).
- Ahn et al.: GD cannot converge to stationary points with an eigenvalue below $0$ or above $2/\eta$. Saddles and too-sharp minima are the same kind of obstacle.
:::

:::problem The next step
Past $2/\eta$, the linear analysis predicts divergence, yet training continues. Something beyond the linear term must be pushing the sharpness back down.
:::
