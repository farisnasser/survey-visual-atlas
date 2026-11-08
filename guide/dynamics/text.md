## 1. Gradient descent as a map

This chapter follows Sections II-G, II-H and II-I of the survey, the last part of the preliminaries. They build the vocabulary that Sections III and IV use: what it means for GD to "converge to a saddle with probability zero", what "isotropic noise" is, and how the sharpness of a real network is measured. As in the survey, we use one running example, the toy saddle

$$L_{\rm toy}(\theta) = \tfrac12\theta_1^2 - \tfrac12\theta_2^2 + \tfrac14\theta_2^4,$$

with a strict saddle at the origin (curvature $+1$ along $\theta_1$, $-1$ along $\theta_2$) and minima at $(0, \pm1)$. This is survey Eq. (12).

One GD step is a **map**: a rule that sends a point to a new point, $G(\theta) = \theta - \eta\nabla L(\theta)$. Running GD means applying the map again and again, $\theta_t = G(G(\cdots G(\theta_0)))$. A **[[fixed-point|fixed point]]** is a point the map does not move, $G(\theta) = \theta$. That happens exactly when $\nabla L(\theta) = 0$: **the fixed points of GD are the critical points of $L$.**

The map's Jacobian, the matrix of first derivatives from the course, is $DG = I - \eta\nabla^2 L$. Its eigenvalues are the multipliers $1 - \eta\lambda_j$ of chapter 5. A fixed point is **unstable** if some multiplier has magnitude larger than one. For the toy saddle, the step multiplies $\theta_1$ by $1 - \eta$ and maps $\theta_2$ to $\theta_2 + \eta(\theta_2 - \theta_2^3)$.

## 2. Stable sets, measure zero, almost surely

The **[[stable-set|stable set]]** of a fixed point is the set of starting points from which GD converges to it. For the toy saddle, near the saddle it is the line $\theta_2 = 0$. A start with $\theta_2 = 0$ keeps $\theta_2 = 0$ forever and slides into the saddle; a nearby start with $\theta_2 \ne 0$ is pushed away. With a finite step a few more lines belong to it: those that one GD step maps exactly onto $\theta_2 = 0$, such as $\theta_2 = \pm\sqrt{1 + 1/\eta}$, and their own preimages. For example, with $\eta = 0.1$, the start $\theta_2 = \sqrt{11}$ is mapped to $\sqrt{11}\,(1 + 0.1 - 0.1\cdot 11) = 0$ in one step.

A set has **[[measure-zero|measure zero]]** if it has zero length on a line, zero area in the plane, or zero volume in $\mathbb{R}^d$. A line in the plane has measure zero, and so does any countable collection of lines. A random start drawn from any distribution with a density lands in such a set with probability zero. An event of probability one happens **[[almost-surely|almost surely]]**.

:::viz stable
:::

:::read How to read this visual
**What is drawn.** A map seen from above of $f = \tfrac12y^2 + \cos(x - y^2/4) - 1$. It has a strict saddle at the center (red) and minima at $(\pm\pi, 0)$ (green). Every starting point in the box is tinted by the minimum that gradient descent reaches from it: one tint for the left minimum, another for the right. The amber curve between the two tints is the saddle's stable set, computed by bisection between the two basins. Here it is a curve, not a straight line, because the loss is not separable.

**What moves.** Drag the white start; the coral path is GD from there, with $\eta = 0.2$, below $1/L$ on this box as Lee et al.'s theorem requires.

**What to try.** Drag the start as close to the amber curve as you can.

**What to notice.** However close you get, the path slides towards the saddle and then still peels away to one side. Only starts exactly on the amber curve end at the saddle, and that curve has zero area. The readout reports that of 2,000 random starts in the box, none end at the saddle. This is "converges to a saddle with probability zero" in one picture.
:::

**The stable-manifold theorem.** Near a fixed point of a smooth invertible map where some multiplier has magnitude larger than one, the points whose iterates stay nearby forever lie on a smooth surface of dimension strictly less than $d$: the **center-stable manifold**, tangent to the directions whose multipliers have magnitude at most one. So the stable set has measure zero. The theorem needs $G$ to be a **[[diffeomorphism]]**, a smooth map with a smooth inverse, so that it never folds a region of positive volume onto a set of zero volume. The GD map is one when $\eta < 1/\ell$. Then every $|\eta\lambda_j| < 1$, every factor of $\det DG = \prod_j(1 - \eta\lambda_j)$ is positive, and $DG$ is invertible everywhere. Chapter 7 puts these pieces together into Lee et al.'s proof.

## 3. Gradient flow, invariant sets, oscillations

**[[gradient-flow|Gradient flow]].** If the step size is small, many steps move $\theta$ only a little. In the limit $\eta \to 0$, with time measured as $s = \eta t$, GD becomes the differential equation $d\theta/ds = -\nabla L(\theta)$, called **gradient flow** (survey Eq. 13). Counting time in steps, it reads $d\theta/dt = -\eta\nabla L(\theta)$. Along it the loss always decreases, by the chain rule: $dL/ds = -\lVert\nabla L\rVert^2 \le 0$. GD follows gradient flow closely when every multiplier lies between $0$ and $1$, that is, when the step is small compared with the curvature.

**Invariant sets.** A set is (forward) **invariant** if the dynamics never leave it once inside. The line $\theta_2 = 0$ of the toy saddle is invariant. In networks, symmetry creates invariant sets: two hidden units with identical weights get identical gradients and stay identical forever (chapter 2's mirror visual started a ball on such a set).

**Oscillations.** GD need not converge to a point. A **[[period-doubling|period-two orbit]]** is a pair of points $a \ne b$ with $G(a) = b$ and $G(b) = a$: the iterates bounce between them forever. On $f(x) = \tfrac\lambda2x^2$ with $\eta\lambda = 2$, the step is $x \mapsto (1 - 2)x = -x$, and every start bounces between $x_0$ and $-x_0$. As the step size crosses a critical value, a fixed point can lose its stability while a stable period-two orbit appears; this change is a **bifurcation**. Repeating it gives orbits of period 4, 8, and so on, a period-doubling cascade, and finally chaos, irregular iterates that never repeat. Chapter 12 shows this happening at the edge of stability.

:::viz orbit
:::

:::read How to read this visual
**What is drawn.** The loss $f(x, y) = \tfrac12(xy - 1)^2$. Its minima form the green hyperbola $xy = 1$. The sharpness at a minimum is $(x - y)^2 + 2$, so the flattest minimum is $(1, 1)$, with sharpness $2$. The white rings mark the two points of the period-two orbit when there is one. The top chart follows the gap $|x - y|$ on a log scale (zero means on the diagonal). The bottom chart follows the residual $xy - 1$ (zero means on the hyperbola of minima).

**What moves.** The coral ball is gradient descent from $(1.3, 0.5)$ with step $\eta = K$ (the slider; the curvature scale is $\mu = 1$).

**What to try.** Set $K$ below 1, then just above it, say $1.05$, then above $1.24$.

**What to notice.** For $K < 1$ the ball converges to a minimum on the hyperbola. For $K$ slightly above 1, every minimum is too sharp for the step ($\eta \cdot 2 > 2$), so no minimum is stable. Yet GD does not diverge. It is squeezed onto the diagonal $x = y$ and ends bouncing between the two white rings on either side of the valley: a stable period-two orbit. Chen & Bruna's theorem guarantees this up to $K \approx 1.121$; in this run it persists until $K \approx 1.237$ and then doubles into a period-four orbit. In the period-two range the readout gives the two points and the two-step multiplier, which has magnitude below one: the orbit is stable.
:::

**Time averages and projections.** When iterates oscillate, their average over a short window is easier to follow than the iterates themselves. The projection matrix $\Pi^\perp_v = I - vv^\top/\lVert v\rVert^2$ removes the component of a vector along $v$, as in the course's geometry of least squares. Moving along $-\Pi^\perp_v\nabla L$ follows the gradient as closely as possible while keeping unchanged, to first order, any quantity whose gradient is $v$. Chapter 13 uses exactly this with $v = \nabla S$.

## 4. Randomness: the shape of minibatch noise

A minibatch is a random subset $B$ of $S$ of the $N$ examples, and the minibatch gradient is $g = \tfrac1S\sum_{i\in B}\nabla L_i(\theta)$. By linearity of expectation it is **unbiased**: $\mathbb{E}[g] = \nabla L$. Its noise $\xi = g - \nabla L$ has mean zero and a **covariance matrix** $\Sigma = \mathbb{E}[\xi\xi^\top]$, as in the course. The variance of the noise along a unit direction $v$ is $v^\top\Sigma v$.

Noise is **[[isotropic-noise|isotropic]]** if $\Sigma = cI$, equally strong in every direction, and **anisotropic** otherwise. Real SGD noise is anisotropic, and its shape is tied to the Hessian. Isotropic noise has a weakness in high dimension. Take $u$ uniform on the unit sphere in $\mathbb{R}^d$. Then $\sum_j u_j^2 = 1$, and by symmetry every $\mathbb{E}[u_j^2]$ is the same, so each equals $1/d$. The noise along any one fixed direction, such as the single escape direction of a saddle, therefore has typical size only $1/\sqrt d$.

:::viz dim
:::

:::read How to read this visual
**What is drawn.** For a saddle in $d$ dimensions with a single escape direction, $-w_1^2/2 + w_1^4/4 + \lVert w_{\rm rest}\rVert^2/2$, the number of gradient-descent steps needed to escape after one random kick of the same energy $r^2$. The horizontal axis is the dimension $d$ on a log scale, and the vertical axis is the median number of steps over 60 runs. Two curves compare two kinds of kick: isotropic, spread over all directions, and CNC-like, which always gives half its energy to the escape direction. The values come from an offline computation.

**What to notice.** The isotropic curve rises as $d$ grows. An isotropic kick puts only $r^2/d$ of its energy on the escape direction, so the escape starts from a smaller offset, and by the escape-time formula the wait grows like $\log d$. The CNC curve is flat: noise that keeps a fixed share along the escape direction escapes equally fast in any dimension. This is why the noise conditions of chapter 8 talk about the noise *along the negative-curvature direction*, not its total size. The readout lists both medians for each $d$.
:::

**Batch size.** For a quantity $h_i$ that varies across examples, with mean $\bar h$ and variance $s^2$, the average over a batch of $S$ examples drawn without replacement has variance $\frac{s^2}{S}\cdot\frac{N - S}{N - 1}$ (survey Eq. 14). It equals $s^2$ for $S = 1$ and $0$ for the full batch. Chapter 10 applies this to the per-example curvatures to derive SGD's stability condition.

**Heavy tails.** A random variable is **[[heavy-tail|heavy-tailed]]** if $P(|X| > t)$ decays only like a power $t^{-\alpha}$, instead of like $e^{-t^2/2}$ as for a Gaussian. The exponent $\alpha$ is the tail index. If $\alpha \le 2$, the variance is infinite. **Large deviations** describe events that need a long run of unlikely noise, such as SGD climbing over a barrier of height $E$. For small steps, such probabilities decay like $e^{-E/\eta}$, so the expected waiting time grows like $e^{E/\eta}$. Both reappear in chapter 8.

## 5. Measuring curvature in practice

The **[[sharpness]]** at $\theta$ is the largest Hessian eigenvalue, $S(\theta) = \lambda_{\max}(\nabla^2 L(\theta))$. It changes as training moves $\theta$. When the top eigenvalue is not repeated, with unit eigenvector $u$, then $S(\theta) = u^\top\nabla^2 L(\theta)u$, and its gradient is obtained by differentiating the Hessian with $u$ held fixed: $\nabla S(\theta) = \nabla^3 L(\theta)[u, u]$, the third-derivative term of the Taylor expansion. $\nabla S$ points to where the loss gets sharper fastest. Chapter 12 needs exactly this.

A network's Hessian has millions of rows and cannot be stored. But its product with a vector can be computed from two gradients: $\nabla^2 L(\theta)v \approx \big(\nabla L(\theta + hv) - \nabla L(\theta)\big)/h$ for a small $h$. Automatic differentiation computes it exactly at the cost of a few gradients. This is a **[[hvp|Hessian-vector product]]** (HVP).

:::viz hvp
:::

:::read How to read this visual
**What is drawn.** The error of the HVP estimate against the step $h$, both on log scales, at one point of the smooth surface $0.3x^2 + 0.2y^2 + 0.4\sin 1.5x\cos y$, with direction $v = (1, 1)/\sqrt2$. "Error" is the distance from the estimate to the exact $\nabla^2 L\,v$. Coral: the forward difference $(\nabla f(x + hv) - \nabla f(x))/h$. Violet: the central difference $(\nabla f(x + hv) - \nabla f(x - hv))/(2h)$. The dashed amber line and the white dot mark the chosen $h$ and the forward error there.

**What to try.** Move $h$ from $1$ down to $10^{-12}$.

**What to notice.** Each curve is a V. On the right, a large $h$ lets the bend of $f$ spoil the estimate, the Taylor error: the forward difference's error shrinks like $h$, the central one's like $h^2$. On the left, a tiny $h$ makes the two gradients almost equal, and rounding errors in their difference take over. There is a best $h$ in between, and the central difference is more accurate there. Automatic differentiation avoids the trade-off, which is why real experiments use it.
:::

Repeating $v \leftarrow \nabla^2 L\,v/\lVert\nabla^2 L\,v\rVert$ is **[[power-method|power iteration]]**. In the eigenvectors, each step multiplies the component along $v_j$ by $\lambda_j$, so after $k$ steps the components are in the ratio $\lambda_j^k$. The vector turns towards the eigenvector with the largest $|\lambda_j|$, and $v^\top\nabla^2 L\,v$ converges to that eigenvalue. This is how the edge-of-stability experiments of chapter 11 measure the sharpness of real networks. The same trick, applied to $I - \eta\nabla^2 L$ instead of $\nabla^2 L$, finds the most *negative* curvature direction at a saddle.

:::viz power
:::

:::read How to read this visual
**What is drawn.** At a saddle in $d = 30$ dimensions, with one negative curvature $-\gamma$ and 29 nonnegative ones, the share of the vector $v$ that lies along the negative-curvature direction, $|\langle v, e_{\min}\rangle|$, against the iteration on a log axis. The update uses only gradients: $v \leftarrow v - (\nabla f(x + rv) - \nabla f(x))/r$, then rescale to length one. The difference of two gradients is $\approx \nabla^2 f\,v$, so this is power iteration on $I - \nabla^2 f$ (step 1, which equals $1/L$ here).

**What to try.** Change the strength of the negative curvature $\gamma$.

**What to notice.** The share starts small and climbs to 1: the vector finds the way down using gradients only. Along the negative direction the multiplier is $1 + \gamma$, the largest of all, so that component grows fastest. Its advantage over the others is only $\gamma$, so the readout's iteration count grows like $1/\gamma$. Weak negative curvature is slow to find, for the same reason it is slow to escape (chapter 5).
:::

**Conditioning, preconditioning, singular values.** The condition number of a positive definite matrix is $\kappa = \lambda_{\max}/\lambda_{\min}$; a large $\kappa$ means a long, narrow valley. For a preconditioned update $\theta \leftarrow \theta - \eta P^{-1}\nabla L$ (chapter 4), the same linearization as chapter 5 gives the matrix $I - \eta P^{-1}\nabla^2 L$, so its stability is decided by the **preconditioned Hessian** $P^{-1}\nabla^2 L$. For non-square matrices, such as the end-to-end map $W = W_H\cdots W_1$ of a linear network, the course's singular value decomposition replaces the eigendecomposition. The singular values measure how strongly $W$ maps each input direction, and the rank is their number of nonzeros. Inputs are **whitened** when they have been transformed so that their covariance is the identity, for instance by PCA.

:::key What to keep from this chapter
- GD is a map; its fixed points are the critical points; its Jacobian's eigenvalues are the multipliers $1 - \eta\lambda$.
- The starting points that converge to a strict saddle form a set of measure zero, so a random start avoids them almost surely, provided the GD map is a diffeomorphism ($\eta < 1/\ell$).
- Isotropic noise puts only a $1/d$ share of its variance on any one direction; heavy tails can make the variance infinite.
- Sharpness is measured by power iteration with Hessian-vector products; $\nabla S = \nabla^3 L[u, u]$.
:::

:::problem The next step
The preliminaries are complete. Section III of the survey begins: why saddles matter in high dimension, and Lee et al.'s proof that GD almost never stops at a strict one.
:::
