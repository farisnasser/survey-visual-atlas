## 1. One step size, many curvatures

Stop 8 ended on its own example: the line fit bends with curvature $0.45$ in one direction and $12.33$ in another. A single step size $\eta$ has to stay below $2/12.33$ to be stable along the steep direction, and then it crawls along the flat one. That is the zig-zag of stops 3 and 4 again, and momentum (stop 7) only softens it from $\kappa$ to $\sqrt\kappa$.

The parameters of a network make this worse. Weights in different layers, biases, embeddings of common and rare words: their gradients can differ by orders of magnitude, and one global step cannot suit all of them. The idea of this stop is to give **each direction its own step size**. Done exactly, that is Newton's method; done cheaply, coordinate by coordinate, it is AdaGrad, RMSProp and Adam.

## 2. Preconditioning: reshaping the landscape

:::def Preconditioned gradient descent
Multiply the gradient by a fixed symmetric positive definite matrix $P$ before stepping:

$$x_{t+1} = x_t - \eta\, P\, \nabla f(x_t).$$

$P$ is the **[[preconditioning|preconditioner]]**. With $P = I$ (the identity) this is gradient descent.
:::

Why does this help? Change coordinates by $x = P^{1/2} y$, where $P^{1/2}$ is the matrix square root. Plain GD on $y$ is exactly preconditioned GD on $x$ (multiply the GD step in $y$ by $P^{1/2}$ and use the chain rule of stop 2). In the new coordinates the Hessian is $P^{1/2}\, H\, P^{1/2}$. So a preconditioner is a change of coordinates, and a good one makes the landscape **rounder**: it lowers the condition number that, by stops 4 and 7, sets the speed.

The ideal choice on a quadratic is $P = H^{-1}$, the inverse Hessian. Then $P^{1/2} H P^{1/2} = I$, every curvature becomes $1$, and one step with $\eta = 1$ lands exactly on the minimum. That is **[[newton|Newton's method]]** (stop 6). It is also why Newton is not used to train networks: for $d$ parameters, $H$ has $d^2$ entries and solving with it costs about $d^3$ operations, and near a saddle the Newton step points toward the saddle (stop 6).

The cheapest useful preconditioner is **diagonal**: one number per coordinate, $P = \operatorname{diag}(p_1, \dots, p_d)$, so coordinate $i$ gets its own step $\eta\, p_i$. If $P$ is the inverse of the Hessian's diagonal, an **axis-aligned** valley becomes perfectly round. A rotated one does not: at $45°$ both diagonal entries of $H$ equal $(\mu + L)/2$, so the diagonal preconditioner only rescales $H$ and the condition number stays $\kappa$.

:::viz precond
:::

With $\kappa = 100$, the diagonal preconditioner reaches $f < 10^{-8}$ in 1 step when the valley lies along an axis, in 396 steps at $30°$ (the rescaled valley still has condition number $75.5$), and in 528 steps at $45°$, the same as GD. Every method in the rest of this stop except Shampoo is diagonal, so this picture is their main limitation.

## 3. AdaGrad: let the gradients set the steps

We do not know the Hessian. But the gradients we see along the way carry information about scale: a coordinate whose gradients are always large is probably steep, and a coordinate that rarely gets a gradient (a rare word, a rare feature) needs bigger steps when it does.

:::def AdaGrad
**[[adagrad|AdaGrad]]** (Duchi, Hazan & Singer 2011) keeps, for every coordinate $i$, the running sum of squared gradients $G_{t,i} = \sum_{s \le t} g_{s,i}^2$ and divides that coordinate's step by its square root:

$$x_{t+1,i} = x_{t,i} - \frac{\eta}{\sqrt{G_{t,i}} + \epsilon}\; g_{t,i}.$$

The small $\epsilon$ (for example $10^{-8}$) avoids dividing by zero.
:::

Two consequences. Coordinates with large gradients get small steps and rare coordinates get large ones. And because the sum only grows, every coordinate's step **shrinks over time** (like $1/\sqrt t$ for a steady gradient), which is what the convex theory wants and what long non-convex training does not.

[Duchi et al.](paper.html?id=duchi2011adaptive) analyze AdaGrad in **online convex optimization**, where the guarantee is a bound on *regret* (how much worse the method did, in total, than the best fixed point in hindsight). Their bound is as good as the best diagonal preconditioner chosen in hindsight, up to a factor (Corollary 1), and with sparse features it can be much smaller than plain online gradient descent's (their Section 1.3 example: $O(\log d\,\sqrt T)$ against $O(\sqrt{dT})$). These are convex guarantees; for non-convex problems, [Antonakopoulos et al. 2022](paper.html?id=antonakopoulos2022adagrad) show that deterministic AdaGrad, in its diagonal and full-matrix forms, still avoids strict saddles from almost every starting point (Theorem 2).

## 4. RMSProp: forget the distant past

AdaGrad's ever-growing sum makes its steps vanish. **[[rmsprop|RMSProp]]** (proposed by Tieleman and Hinton in a 2012 lecture) replaces the sum by an [[ema|exponential moving average]] (stop 7) of the squared gradients,

$$v_{t,i} = \beta_2\, v_{t-1,i} + (1 - \beta_2)\, g_{t,i}^2, \qquad x_{t+1,i} = x_{t,i} - \frac{\eta}{\sqrt{v_{t,i}} + \epsilon}\; g_{t,i},$$

so the scale estimate follows the recent gradients and the step no longer shrinks forever. $\sqrt{v_{t,i}}$ is a "root mean square" of recent gradients, which gives the method its name.

## 5. Adam: momentum plus RMSProp, with bias correction

:::def Adam
**[[adam|Adam]]** (Kingma & Ba 2015) keeps two EMAs per coordinate: the gradient itself (momentum, stop 7) and its square (RMSProp):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2,$$

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad x_{t+1} = x_t - \alpha\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon},$$

all operations coordinate by coordinate. The paper's defaults are $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
:::

The two divisions by $1 - \beta^t$ are the **[[bias-correction|bias correction]]**. Both EMAs start at $0$. If the gradient were the same number $g$ at every step, then $m_t = (1 - \beta_1)(g + \beta_1 g + \dots + \beta_1^{t-1} g) = (1 - \beta_1^t)\, g$: too small by exactly the factor $1 - \beta_1^t$, and likewise for $v_t$. Dividing undoes it. Without the correction the two errors do not cancel, because $\beta_1 \ne \beta_2$: the first step would be $0.1/\sqrt{0.001} \approx 3.16$ times $\alpha$, and the step would keep growing to $6.57\alpha$ at $t = 12$ before settling, the opposite of the careful start one wants.

:::viz bias
:::

For a steady gradient, $\hat m_t / \sqrt{\hat v_t} = g/\lvert g \rvert = \pm 1$: Adam's step is about $\alpha$ in each coordinate **whatever the size of the gradient**. Kingma & Ba make this precise (their Section 2.1: the step is bounded by about $\alpha$ in typical cases) and note that the update does not change if the gradient is multiplied by a constant. This is Adam's strength (no coordinate is stuck because its gradients are tiny) and the source of its quirks (near a minimum the step does not shrink by itself as the gradient does, unless $\alpha$ is decayed).

Here are the diagonal methods on the $\kappa = 100$ valley. Along an axis, AdaGrad and Adam leave GD far behind: after 400 steps GD is at $8.2 \times 10^{-7}$, Adam at $1.2 \times 10^{-18}$, AdaGrad at $5.6 \times 10^{-123}$. Turned by $45°$, the same valley defeats them: GD reaches $1.8 \times 10^{-6}$, AdaGrad $2.1 \times 10^{-5}$ and Adam $5.6 \times 10^{-5}$. Per-coordinate steps help exactly as much as the landscape's directions line up with the coordinates.

:::viz race
:::

## 6. What is known about Adam's convergence

Adam is the default optimizer for much of deep learning, so it is striking how long its theory lagged.

:::warn Adam can fail, even on convex problems
Kingma & Ba's convergence proof (Theorem 4.1, online convex) was later shown to be flawed. [Reddi, Kale & Kumar 2018](paper.html?id=reddi2018convergence) built an online convex problem on which Adam's average regret does not go to zero (Theorem 1): linear losses on $[-1, 1]$ whose gradient is a large $C$ once every three steps and $-1$ otherwise. The rare large gradient is shrunk by the second-moment average, so Adam drifts to the worst point. For any constant $\beta_1 < \sqrt{\beta_2}$ such a problem exists (Theorem 2), also in a stochastic version (Theorem 3). Their fix, **AMSGrad**, divides by the running *maximum* of $v_t$, so the effective step never grows.
:::

[Taniguchi et al. 2024](paper.html?id=taniguchi2024adopt) use a cousin of that example: minimize $f(\theta) = \theta$ on $[-1, 1]$ with stochastic gradient $k^2$ with probability $1/k$ and $-k$ otherwise. It is unbiased, $k^2 \cdot \frac1k - k\,(1 - \frac1k) = 1$, so the answer is $\theta = -1$. In our run with $k = 10$, Adam ends at the worst point $+1$ for $\beta_2 \le 0.9$ and near $-1$ only for $\beta_2 \ge 0.99$ (the paper's Figure 1 shows success only at $0.999$; seeds and run length differ). With $k = 50$ it fails for every $\beta_2$ we tried. ADOPT goes to $-1$ in every setting; AMSGrad does for $k = 10$ but is far slower for $k = 50$ (still at $-0.05$ after 400 000 steps).

:::viz adamfail
:::

:::theorem Convergence results for Adam-type methods
- [Li, Rakhlin & Jadbabaie 2023](paper.html?id=li2023convergence), Theorem 4.1: under a relaxed smoothness condition (curvature at most $L_0 + L_\rho \lVert \nabla f \rVert^\rho$ with $\rho < 2$; section 7) and bounded gradient noise, Adam with a suitably small step size and momentum weight (their $\beta$, which plays the role of $1 - \beta_1$, of order $\varepsilon^2$) keeps the gradient bounded along its path and reaches $\frac1T \sum_t \lVert \nabla f(x_t) \rVert^2 \le \varepsilon^2$ with $O(\varepsilon^{-4})$ stochastic gradients, with high probability and with no condition on $\beta_2$. The rate matches SGD's (stop 8); a variance-reduced variant reaches $O(\varepsilon^{-3})$.
- [Taniguchi et al. 2024](paper.html?id=taniguchi2024adopt) trace the failures to the current gradient sitting inside its own normalizer. **ADOPT** normalizes the new gradient by the *previous* second moment $v_{t-1}$, and applies momentum after the normalization. Theorem 4.1: for any $\beta_2$, a constant step of order $1/\sqrt T$ gives the optimal $O(1/\sqrt T)$ rate (in the form $\min_t \mathbb{E}[\lVert \nabla f \rVert^{4/3}]^{3/2}$), assuming only a bounded second moment of the noise.
- [Staib et al. 2019](paper.html?id=staib2019escaping): RMSProp-style preconditioning makes the gradient noise roughly isotropic near a stationary point, which helps escape saddles; with occasional large steps they prove a $(\tau, \sqrt{\rho\tau})$-second-order stationary point in $\tilde O(\tau^{-5})$ iterations (Theorem 4.4, Corollary 4.2).
:::

None of these says Adam reaches better points than tuned SGD with momentum; whether and when it does is an empirical question.

## 7. Gradient clipping and $(L_0, L_1)$-smoothness

Stop 3 built everything on one global smoothness constant $L$. Many losses do not have one. On $f(x, y) = x^2/2 + x^4/4 + y^2/2$ the curvature along $x$ is $1 + 3x^2$: $193$ at $x = 8$, unbounded as $x$ grows. No single step size is safe everywhere, and GD must use a step small enough for the steepest place it will ever visit.

:::def $(L_0, L_1)$-smoothness
[Zhang, He, Sra & Jadbabaie 2020](paper.html?id=zhang2020gradient) (Definition 1) call a twice-differentiable $f$ **[[l0l1-smooth|$(L_0, L_1)$-smooth]]** if

$$\lVert \nabla^2 f(x) \rVert \;\le\; L_0 + L_1\, \lVert \nabla f(x) \rVert .$$

The curvature may be large, but only where the gradient is large too. Polynomials of degree 3 or more satisfy it without being $L$-smooth (Lemma 2); the example above is $(1, 3)$-smooth on $[-6, 6]^2$. They motivate the condition by measurements on an LSTM language model, where curvature and gradient size grow together (their Figure 1).
:::

Under this condition the natural fix is to shrink the step where the gradient is large:

:::idea Gradient clipping
**[[clipping|Clipped GD]]** uses the step $h_t = \min\{\eta,\; \gamma\eta / \lVert \nabla f(x_t) \rVert\}$: the full step $\eta$ when the gradient is small, and a move of length at most $\gamma\eta$ when it is large. It is a one-line change, widely used when training recurrent and transformer language models.
:::

:::theorem Zhang et al. 2020, Theorems 3 and 4
For $(L_0, L_1)$-smooth $f$ with $\Delta = f(x_0) - f^\star$, clipped GD reaches $\lVert \nabla f \rVert \le \varepsilon$ in $O(L_0\Delta/\varepsilon^2 + L_1^2\Delta/L_0)$ iterations (Theorem 3). GD with any fixed step can need $\Omega(L_1 M \Delta / (\varepsilon^2 \log M))$, where $M$ is the largest gradient norm on the starting sublevel set (Theorem 4). When the start is steep ($M$ large), clipping can be arbitrarily faster.
:::

On the example, from $(x_0, 1)$ with $\varepsilon = 10^{-4}$: the best fixed step needs 6, 22, 81 and 322 iterations for $x_0 = 1, 2, 4, 8$, while clipped GD with one setting ($\eta = 1$, $\gamma = 0.5$) needs 4, 6, 10 and 18.

:::viz clip
:::

Adam's normalization $\hat m / \sqrt{\hat v}$ acts like a per-coordinate clip, which is one explanation for why adaptive methods cope with such landscapes. [Li et al. 2023](paper.html?id=li2023convergence) analyze Adam under the generalized condition $L_0 + L_\rho\lVert \nabla f \rVert^\rho$ (section 6).

## 8. Beyond the diagonal

Diagonal methods cannot see rotated valleys (section 2). Full-matrix preconditioners can, but cost $d^2$ memory. Methods in the Atlas sit in between.

:::def Shampoo
**[[shampoo|Shampoo]]** ([Gupta, Koren & Singer 2018](paper.html?id=gupta2018shampoo)) treats an $m \times n$ weight matrix as a matrix. It keeps $L_t = L_{t-1} + G_t G_t^\top$ ($m \times m$) and $R_t = R_{t-1} + G_t^\top G_t$ ($n \times n$) and steps

$$W_{t+1} = W_t - \eta\, L_t^{-1/4}\, G_t\, R_t^{-1/4}.$$

This is a Kronecker-factored approximation of full-matrix AdaGrad ($(L \otimes R^\top)\,\mathrm{vec}(G) = \mathrm{vec}(LGR)$, their Lemma 4). For a $1000 \times 1000$ layer it stores $2 \times 10^6$ numbers, against $10^6$ for a diagonal method and $10^{12}$ for full-matrix AdaGrad.
:::

Its guarantee is an $O(\sqrt T)$ regret bound in online convex optimization (Theorem 7); its benefit on networks is empirical (the paper's experiments on CIFAR and a language model).

Other directions in the Atlas:

- **Diagonal curvature.** AdaHessian and Sophia estimate the Hessian's diagonal itself instead of using squared gradients. [SASSHA](paper.html?id=shin2025sassha) (Shin et al. 2025) adds a sharpness-aware step (stop 10) and uses the square root of the absolute diagonal Hessian estimate, updated lazily. Its theorem is convex (Theorem 4.4). Its Table 1 (ResNet-32 on CIFAR-100, reported) shows why the next stop matters: the second-order methods ended at much **sharper** minima than SGD (largest Hessian eigenvalue 11 992 for AdaHessian, 22 797 for Sophia-H, 436 374 for Shampoo, against 265 for SGD) and generalized worse (68.06%, 67.76%, 64.08% accuracy against 69.32%), while SASSHA ended flatter (107) and better (72.14%).
- **Other norms.** [Crawshaw et al. 2026](paper.html?id=crawshaw2026exploration) view Adam and Muon as steepest descent in different norms (Propositions 3.4 to 3.6). Muon's step for a weight matrix is the "orthogonalized" momentum matrix, a steepest-descent step in the spectral norm.

:::atlas Where this shows up in the Atlas
- [Duchi et al. 2011](paper.html?id=duchi2011adaptive): AdaGrad. [Kingma & Ba 2015](paper.html?id=kingma2015adam): Adam and bias correction.
- [Reddi et al. 2018](paper.html?id=reddi2018convergence), [Li et al. 2023](paper.html?id=li2023convergence), [Taniguchi et al. 2024](paper.html?id=taniguchi2024adopt): when Adam converges and when it does not.
- [Antonakopoulos et al. 2022](paper.html?id=antonakopoulos2022adagrad) and [Staib et al. 2019](paper.html?id=staib2019escaping): adaptive methods and saddle points.
- [Zhang et al. 2020](paper.html?id=zhang2020gradient): $(L_0, L_1)$-smoothness and clipping.
- [Gupta et al. 2018](paper.html?id=gupta2018shampoo), [Shin et al. 2025](paper.html?id=shin2025sassha), [Crawshaw et al. 2026](paper.html?id=crawshaw2026exploration): beyond diagonal scaling.
- [Defazio et al. 2024](paper.html?id=defazio2024road): Schedule-Free variants of SGD and AdamW, which replace the decaying step schedule with iterate averaging.
:::

:::problem The next problem
SASSHA's table made it visible: different optimizers, with similar training loss, end at minima of very different **sharpness**, and the sharp ones generalized worse. A network has many minima. **Which one does training find, and does it matter?**
:::
