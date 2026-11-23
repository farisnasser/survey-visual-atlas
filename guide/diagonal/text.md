## 1. What a diagonal Hessian keeps, and what it loses

This chapter follows Sections V-D and V-E of the survey. K-FAC and Shampoo use structured matrices, which cost memory and compute. AdaHessian, Apollo and Sophia keep the efficiency of diagonal scaling but feed it curvature. For two parameters, the diagonal entries of $H$ are the curvature along each coordinate and the off-diagonal entries the interaction. With

$$H = \begin{bmatrix}10 & 3\\ 3 & 2\end{bmatrix}, \qquad D = \mathrm{diag}(10, 2), \qquad g = (10, 2)^\top,$$

a simplified diagonal preconditioner gives $D^{-1}g = (0.1\cdot10,\; 0.5\cdot2) = (1, 1)$. The first gradient component was five times larger, and so was its curvature estimate: after preconditioning the two match. A large gradient does not by itself justify a large step. (This is only the core idea; real Hessian diagonals can be zero or negative, so safeguards are needed.)

## 2. AdaHessian: estimate the diagonal with random probes

**AdaHessian** never builds $H$. It uses Hessian–vector products (chapter 11) on random probes. Draw $z$ with independent entries $\pm1$, a **[[hutchinson|Rademacher]]** vector, and form $z\odot(Hz)$, entry by entry. With the $H$ above and $z = (1, 1)^\top$, $Hz = (13, 5)$ and $z\odot(Hz) = (13, 5)$: not the diagonal $(10, 2)$. One probe is noisy. But in two dimensions we can list all four sign vectors:

| probe $z$ | $Hz$ | $z\odot(Hz)$ |
|---|---|---|
| $(1, 1)$ | $(13, 5)$ | $(13, 5)$ |
| $(1, -1)$ | $(7, 1)$ | $(7, -1)$ |
| $(-1, 1)$ | $(-7, -1)$ | $(7, -1)$ |
| $(-1, -1)$ | $(-13, -5)$ | $(13, 5)$ |
| **average** | | **$(10, 2)$** |

The off-diagonal contributions cancel on average, so

$$\mathbb{E}_z\big[z\odot(Hz)\big] = \mathrm{diag}(H),$$

because independent signs satisfy $\mathbb{E}[z_iz_j] = 0$ for $i \ne j$ and $\mathbb{E}[z_i^2] = 1$. A few probes give a noisy estimate, which AdaHessian smooths over time.

:::viz hutch
:::

:::read How to read this visual
**What is drawn.** A fixed $12\times12$ symmetric matrix $H$, standing in for a Hessian, with some negative diagonal entries. For each coordinate $i$, the white outline is the true $H_{ii}$ and the filled bar is Hutchinson's estimate averaged over the first $k$ probes (coral where positive, red where negative). The side chart shows the relative error of the whole estimated diagonal as $k$ grows (log-log), for one fixed sequence of probes. The dashed line is the exact expected error.

**What moves.** The number of probes $k$, from $1$ to $1000$.

**What to notice.** With one probe the estimate is rough, and several entries can even have the wrong sign; the readout counts them. The estimate is unbiased, so averaging more probes always helps. The error falls like $1/\sqrt k$, the dashed slope. The constant comes from the off-diagonal entries: one probe's error in coordinate $i$ has variance $\sum_{j\ne i}H_{ij}^2$, so a strongly coupled matrix needs more probes. The negative entries are real curvature, not noise. A method that divides by this estimate must deal with them, which is what Sophia's clipping does.
:::

## 3. Apollo: learn the diagonal from gradient differences

**Apollo** takes its curvature from quasi-Newton ideas instead (chapter 11's secant pairs, $y_t \approx Hs_t$). In one dimension, for $L(w) = 5w^2$, moving from $w = 3$ to $w = 2$ changes the gradient from $30$ to $20$: $s = -1$, $y = -10$, and $y/s = 10$ recovers the second derivative without differentiating twice. Apollo builds a more careful diagonal quasi-Newton approximation from the history of steps and gradient changes; it is not simple coordinate-wise division.

**The shared limitation.** Diagonal curvature stores about $O(d)$ numbers instead of $O(d^2)$, but it throws away the coupling. Chapter 4's rotated valley, $H = \begin{bmatrix}50.5 & -49.5\\ -49.5 & 50.5\end{bmatrix}$, has diagonal $(50.5, 50.5)$, so a diagonal method sees equal curvature in both coordinates, while the true eigenvalues are $1$ and $100$. The diagonal alone cannot reveal rotated directions. Diagonal methods can still be the right choice when structured operations cost more than they gain.

:::key Remember V-D
AdaHessian estimates the Hessian's diagonal with random Hessian–vector probes; Apollo learns a diagonal quasi-Newton approximation from changes in weights and gradients. Both save memory and lose cross-coordinate curvature.
:::

## 4. Sophia: curvature scaling with a hard cap

Why can dividing by curvature be dangerous? One-dimensional Newton takes $\Delta = -g/H$. With $g = 6$: $H = 6$ gives $\Delta = -1$, $H = 2$ gives $-3$, and $H = 0.1$ gives $-60$. A tiny curvature estimate can produce a huge step, the same danger that motivated damping and trust regions. **[[sophia|Sophia]]** combines momentum with a periodically refreshed diagonal Hessian estimate and **clips** the result. In the survey's simplified form, entry by entry,

$$\Delta\theta_t = -\eta_t\,\operatorname{clip}\!\Big(\frac{m_t}{\max(h_t, \epsilon)},\, -\rho,\, \rho\Big),$$

where $m_t$ is momentum, $h_t$ the diagonal Hessian estimate, $\epsilon > 0$ keeps the denominator positive, and $\rho$ is the clipping threshold. Smooth the gradient, scale by curvature, clip, multiply by the learning rate.

**A complete example.** Let $m = (6, 6)$, $h = (2, 0.1)$, $\epsilon = 0.01$, $\rho = 5$, $\eta = 0.1$. Then $\max(h, \epsilon) = (2, 0.1)$; dividing gives $(3, 60)$; clipping to $[-5, 5]$ gives $(3, 5)$; and $\Delta\theta = (-0.3, -0.5)$. Without clipping the second weight would have moved by $-6$; the cap holds it to $-0.5$. Curvature still shapes the step, but a tiny denominator cannot create an arbitrarily large component.

If an estimate is negative, say $h = -2$, then $\max(h, \epsilon) = 0.01$: the denominator stays positive, the ratio may be large, and the clip caps it. Neither safeguard guarantees that every step lowers the true loss; each limits one specific problem.

:::viz clip
:::

:::read How to read this visual
**What is drawn.** Why capping a step's size helps, in its simplest form. The loss $f(x, y) = x^2/2 + x^4/4 + y^2/2$ is steep far out along $x$ and gentle near the minimum (green), in 3D with a log height scale. Violet: gradient descent with one fixed step, which must be tiny to survive the steep start. Coral: clipped gradient descent, which takes the full step $\eta = 1$ near the minimum but at most a length $\gamma = 0.5$ when the gradient is large.

**What moves.** The start $x_0$ and the fixed step of the violet run.

**What to notice.** The fixed step is either too large for the steep region or too small for the flat one; the clipped step is safe in both. Sophia applies the same idea per coordinate, to a curvature-scaled step: where the curvature estimate is tiny or negative, $m_t/\max\{h_t, \epsilon\}$ would be huge, and the clip caps it at $\rho$. This visual clips plain gradient descent, not Sophia itself; it shows why the cap matters.
:::

**How it differs from Adam.** Both use momentum, but Adam's denominator is a squared-gradient average, while Sophia's is designed around a Hessian diagonal, and the two are not equal in general; clipping is also part of Sophia's definition. Its original study, on GPT-style models from 125 million to 1.5 billion parameters, reported about a twofold reduction in the resources needed to reach a matched target compared with Adam, a result tied to those settings, not a universal guarantee.

:::key Remember V-E
Sophia's distinctive mechanism is affordable diagonal curvature scaling plus a bound on each update component. It still treats parameter directions independently.
:::

:::problem The next step
Chapter 16 brings matrix structure back: SOAP performs Adam-style adaptation in a rotated basis, and Muon reshapes the update matrix itself.
:::
