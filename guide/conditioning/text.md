## 1. A narrow valley

This chapter follows Sections II-D and II-E of the survey, with the worked examples of Appendices A-A, C-A and D-B. Take the quadratic (survey Eq. 8)

$$L(x, y) = \tfrac12(100x^2 + y^2), \qquad H = \begin{bmatrix}100 & 0\\ 0 & 1\end{bmatrix}.$$

Its level sets are long thin ellipses: moving in $x$ is a hundred times more curved than moving in $y$. On a quadratic, gradient descent treats each eigen-direction separately (chapter 3): with step $\eta$ it multiplies $x$ by $1 - 100\eta$ and $y$ by $1 - \eta$. The survey's example takes $\eta = 0.01$. Then $x$ is multiplied by $1 - 100 \times 0.01 = 0$, so it is solved in one step, while $y$ is multiplied by $0.99$ per step and crawls.

There is no way out by choosing $\eta$ better. Stability in $x$ needs $\eta < 2/100$, and any such step is tiny for $y$. The ratio of the largest to the smallest curvature, the **[[condition-number|condition number]]** $\kappa = \lambda_{\max}/\lambda_{\min} = 100$, measures this mismatch. For strongly convex quadratics, even the best fixed step converges at a rate governed by $\kappa$: the number of steps grows in proportion to $\kappa$.

:::viz kappa
:::

:::read How to read this visual
**What is drawn.** For gradient descent on $\tfrac12(\mu u_1^2 + L u_2^2)$ from $u = (1, 1)$, the number of steps needed until both coordinates are below $10^{-6}$, against the condition number $\kappa = L/\mu$. Both axes are logarithmic. Coral: the step $\eta = 1/L$. Violet: the best fixed step $\eta = 2/(L + \mu)$. The dots are measured runs; the dashed lines are the predictions $\kappa\ln(10^6)$ and $(\kappa/2)\ln(10^6)$.

**What moves.** Drag $\kappa$; the white dots mark the current value.

**What to notice.** The measured counts sit on straight lines of slope one on the log-log axes: steps grow in proportion to $\kappa$. Ten times narrower means ten times slower, even with the best possible fixed step. The readout gives the per-step factors, $1 - 1/\kappa$ and $(\kappa - 1)/(\kappa + 1)$, that cause it.
:::

## 2. Rotate the valley

Now rotate the coordinates. Replace $H$ by $Q^\top\mathrm{diag}(100, 1)Q$ with an orthogonal matrix $Q$. The eigenvalues are unchanged, so the condition number and gradient descent's behaviour are unchanged too: GD does not care how the axes are drawn. But the principal axes no longer line up with the stored parameter coordinates. Appendix A-A of the survey works out the $45^\circ$ case (survey Eq. 70):

$$Q = \frac{1}{\sqrt2}\begin{bmatrix}1 & -1\\ 1 & 1\end{bmatrix}, \qquad H = Q^\top\begin{bmatrix}100 & 0\\ 0 & 1\end{bmatrix}Q = \begin{bmatrix}50.5 & -49.5\\ -49.5 & 50.5\end{bmatrix}.$$

The diagonal entries are now equal, $50.5$ each. The diagonal of $H$ contains no information about the coupling $-49.5$ that makes the valley narrow. This is the key fact behind every diagonal method in the survey (AdaGrad, RMSProp, Adam, AdaHessian, Sophia).

A **diagonal [[preconditioning|preconditioner]]** divides each coordinate of the gradient by its own scale $D_{jj}$: $\theta \leftarrow \theta - \eta D^{-1}g$. It can rescale how far we move along each coordinate. It cannot undo a coupling between coordinates.

:::viz precond
:::

:::read How to read this visual
**What is drawn.** Contour lines of $f = \tfrac12\theta^\top H\theta$ with curvatures $L/\kappa$ and $L = 10$ along axes turned by the chosen angle. Three methods start at $(-1.3, 1.1)$ and show their first 30 steps, each with its best fixed step. Violet: gradient descent. Coral: the diagonal preconditioner $\mathrm{diag}(H)^{-1}$, which divides each coordinate by its own curvature. White: Newton, which multiplies by the full $H^{-1}$. The green dot is the minimum.

**What moves.** The valley angle and $\kappa$ ($10$ or $100$).

**What to notice.** At $0^\circ$ the diagonal preconditioner makes the valley perfectly round, and it matches Newton. As the angle grows it helps less and less, and at $45^\circ$ not at all: the readout's "condition number after the diagonal rescaling" climbs back to $\kappa$. Gradient descent zig-zags at every angle; Newton lands on the minimum in one step at every angle. The step counts in the readout make the same point in numbers.
:::

The survey's Fig. 3 shows the same comparison with the exact $100:1$ matrices of Eqs. (8) and (70).

:::survey Fig. 3 of the survey: why conditioning and coordinate choice matter. The eigenvalues are unchanged by rotation, but diagonal rescaling is tied to the chosen coordinates.
public/survey3/fig3.webp
:::

:::read How to read this figure
**What is drawn.** Contour lines of the $100:1$ quadratic in two coordinate systems. (a) Axis-aligned, $H = \mathrm{diag}(100, 1)$. (b) Rotated by $45^\circ$, the matrix of Eq. (70). Each method takes 30 steps from the star, and the two stars are the same point in the valley's own axes, so both panels show the same problem. Blue: GD with $\eta = 0.018$. Green: the exact diagonal preconditioner $D = \mathrm{diag}(H)$ with $\eta = 1$. Vermillion: one Newton step.

**What to notice.** GD's path is identical up to rotation in both panels: it zig-zags across the valley and creeps along it. Newton goes straight to the minimum in both. Only the diagonal method changes. In (a) its step equals Newton's step (when $H$ is diagonal, $D = H$), so its path is hidden under Newton's. In (b) it bounces from wall to wall, because dividing by $50.5$ in both coordinates is just gradient descent with a step of $1/50.5$. That is close to the stability limit for the steep direction, as section 4 shows.
:::

## 3. One gradient, three steps

Appendix C-A of the survey makes the contrast exact with one step from $\theta_0 = (1, 0)$ on the rotated matrix (survey Eq. 72). All three methods see the same gradient, $g_0 = H\theta_0 = (50.5, -49.5)^\top$, at loss $L(\theta_0) = 25.25$:

| method | step | new point $\theta_1$ | new loss |
|---|---|---|---|
| GD, $\eta = 0.01$ | $-0.01\,g_0$ | $(0.495, 0.495)$ | $0.245025$ |
| exact diagonal, $D = 50.5I$, $\eta = 1$ | $-D^{-1}g_0$ | $(0,\ 0.9802)$ | about $24.26$ |
| Newton | $-H^{-1}g_0$ | $(0, 0)$ | $0$ |

GD happens to do well here, because $\eta = 0.01$ exactly cancels the steep eigen-direction (its multiplier is $1 - 0.01 \times 100 = 0$). The exact diagonal of the Hessian barely helps, because it discards the coupling. Newton solves the quadratic. As the survey stresses, no one-step ranking is meaningful without a fair choice of step for each method. The numbers show *what each optimizer adds* to the same gradient.

:::survey Fig. 11 of the survey: the construction of Appendix C-A. Solid arrows are the first steps, dotted lines the next 11.
public/survey3/fig11.webp
:::

:::read How to read this figure
**What is drawn.** Level sets of the rotated quadratic of Eq. (72), including the levels $25.25$, $24.26$ and $0.245$ quoted in the table. From the star $\theta_0 = (1, 0)$, the solid arrows are the three first steps; the dotted lines and dots are the next 11 steps of GD and of the diagonal method.

**What to notice.** The diagonal method's first step jumps across the valley to $(0, 0.9802)$, almost the same level as the start, and its later steps keep bouncing between the two walls. GD's first step lands near the valley floor, and later steps creep toward the minimum along it. Newton reaches the minimum in one step.
:::

## 4. Local stability: the multiplier $1 - \eta\lambda$

Near any stationary point $\theta^\star$, where $g(\theta^\star) = 0$, the gradient is approximately linear, $g(\theta^\star + e) \approx H^\star e$. So the error $e_t = \theta_t - \theta^\star$ of gradient descent evolves as (survey Eq. 9)

$$e_{t+1} \approx (I - \eta H^\star)\,e_t.$$

Along each Hessian eigenvector, the error is multiplied by the **[[multiplier]]** $1 - \eta\lambda$ at every step:

- $0 < \eta\lambda < 2$: it contracts;
- $\lambda < 0$: it expands without changing sign, which is how GD leaves a saddle;
- $\eta\lambda > 2$: it expands while flipping sign, so the iterates bounce outward.

The famous threshold $\eta < 2/\lambda_{\max}$ comes from this local quadratic analysis. It is not a universal rule for networks: Section VI-E of the survey returns to the "edge of stability", where training continues above it (chapter 18).

The same algebra covers preconditioned steps (Appendix D-B). With a fixed positive-definite $M$, the step $\theta_{t+1} = (I - \eta M^{-1}H)\theta_t$ (survey Eq. 85) behaves like gradient descent on the symmetric matrix $K = M^{-1/2}HM^{-1/2}$ (survey Eq. 86). It contracts in every eigen-direction if and only if (survey Eq. 87)

$$0 < \eta < \frac{2}{\lambda_{\max}(K)}.$$

For $M = I$, $K = H$ and this is the textbook threshold. For $M = H$ (Newton), $K = I$: a unit step sends every component to zero.

:::viz pgdstab
:::

:::read How to read this visual
**What is drawn.** For the rotated valley with curvatures $\kappa$ and $1$, the worst per-step factor $\max_i|1 - \eta\lambda_i(K)|$ as a function of the step size $\eta$, on a log axis, for three preconditioners. Violet: GD, $M = I$. Coral: the exact diagonal, $M = \mathrm{diag}(H)$. White: Newton, $M = H$. Below the dashed line at $1$ the step contracts the error; above it the error grows. The dotted vertical lines mark each method's stability limit $2/\lambda_{\max}(K)$; the dots mark the current $\eta$ (amber line).

**What moves.** The valley angle, $\kappa$, and $\eta$.

**What to try.** At $45^\circ$ and $\kappa = 100$, set $\eta = 1$. Then turn the angle to $0^\circ$.

**What to notice.** At $45^\circ$ the readout gives exactly the numbers behind the table above. $K$ has eigenvalues $1$ and $100$ for GD, so GD is stable only for $\eta < 0.02$. For the diagonal method they are $0.0198$ and $1.98$, so it is stable for $\eta < 1.01$; at $\eta = 1$ its factors are $\pm0.98$, which is the bouncing of Fig. 11. Newton's $K = I$ gives a factor of $0$ at $\eta = 1$. The coral and violet curves have the same shape, shifted along the $\eta$ axis: at $45^\circ$ the diagonal method is just rescaled GD. At $0^\circ$, the coral curve drops onto the white one: an aligned valley is exactly what diagonal scaling can fix. The readout also gives each method's step count at its own best $\eta$, which repeats the condition-number lesson of section 1.
:::

## 5. What gradient-based guarantees actually promise

For non-convex losses the survey uses weaker targets than "find the minimum". A point is an $\epsilon$-[[eps-stationary|first-order stationary point]] if $\|g(\theta)\| \le \epsilon$. It is approximately second-order stationary if, in addition, $\lambda_{\min}(H(\theta)) \ge -\gamma$: no direction curves down by more than $\gamma$. Gradient-based guarantees usually promise only the first: a small gradient, not global optimality. Escaping saddles requires separate assumptions, and sometimes random perturbations (Section II-E; the earlier survey of this course covered them in depth).

:::problem The next step
Every bound so far assumed the exact gradient. Chapter 5 asks what changes when it is estimated from a minibatch.
:::
