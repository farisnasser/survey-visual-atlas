## 1. BFGS: infer curvature from how the slope changes

This chapter follows Sections IV-E, IV-F and IV-G of the survey. A second derivative measures how fast a first derivative changes. So if a step moves a weight from $w_t$ to $w_{t+1}$ and the gradient changes from $g_t$ to $g_{t+1}$, then

$$\text{average curvature over that step} \approx \frac{g_{t+1} - g_t}{w_{t+1} - w_t}.$$

It is not the exact second derivative at either end, but it is information about curvature obtained without differentiating twice. For $L(w) = 2w^2$, with gradient $4w$ and Hessian $4$, move from $w_t = 3$ to $w_{t+1} = 2$: then $s = 2 - 3 = -1$ and $y = g(2) - g(3) = 8 - 12 = -4$, and the ratio $y/s = 4$ is exactly the curvature.

**With many parameters.** The survey calls these differences the secant pair (survey Eq. 34):

$$s_t = \theta_{t+1} - \theta_t, \qquad y_t = g_{t+1} - g_t.$$

On a locally quadratic loss $y_t \approx H_t s_t$: curvature turns a displacement into a change of slope. **[[bfgs|BFGS]]**, named after Broyden, Fletcher, Goldfarb and Shanno, keeps an approximation $B_t$ of the Hessian and updates it so that the new one explains the latest observation, the **[[secant]] condition**

$$B_{t+1}s_t = y_t.$$

:::fold The BFGS update, for reference | one rank-two correction that satisfies the secant condition
For $y_t^\top s_t > 0$, the standard update is

$$B_{t+1} = B_t - \frac{B_t s_t s_t^\top B_t}{s_t^\top B_t s_t} + \frac{y_t y_t^\top}{y_t^\top s_t}.$$

Multiply both sides by $s_t$: the middle term cancels the old prediction $B_t s_t$ and the last term adds the observation $y_t$, so $B_{t+1}s_t = y_t$. The condition $y_t^\top s_t > 0$ keeps $B_{t+1}$ positive definite whenever $B_t$ was, which makes $-B_{t+1}^{-1}g$ a descent direction. The survey describes this update in words; the formula is the classical one from Nocedal and Wright.
:::

**Why BFGS needs care.** The pair is informative only if the gradient change reflects the loss geometry. If $g_t$ and $g_{t+1}$ come from different noisy minibatches, part of $y_t$ is sampling noise: a large apparent change might mean different data, not a steep direction. Near negative curvature or under noise the positivity condition $y_t^\top s_t > 0$ may fail, and safeguards become necessary.

:::viz bfgs
:::

:::read How to read this visual
**What is drawn.** The curved valley $L = (1 - x)^2 + 10(y - x^2)^2$, a Rosenbrock-type test function, in 3D (heights log-scaled), with the minimum at $(1, 1)$ in green. Three balls start at $(-1.2, 1)$. Violet: gradient descent with $\eta = 0.01$, about as large as the steep walls allow. Coral: BFGS with the inverse update above, starting from $C = I$, with a backtracking line search. White: exact Newton. The chart plots each loss on a log scale.

**What to notice.** Gradient descent creeps along the curved floor: after 400 steps it is still far from the minimum. BFGS starts out like a gradient method, because $C = I$ is just GD. After a few steps its estimate $C$ has learned the valley's curvature from the pairs $(s_t, y_t)$, and it converges as fast as Newton does near the end: locally superlinear convergence. It never computes a second derivative. The readout also counts the extra loss evaluations its line search used, a real cost the survey warns about ("even the meaning of one step differs"). Newton, with the true Hessian, needs the fewest steps but the most information per step.
:::

:::key Remember IV-E
BFGS learns curvature by asking "I moved this much; how much did the slope change?" Its data are the pairs $(s_t, y_t)$.
:::

## 2. Inverse BFGS and limited-memory BFGS

Newton wants $H\Delta = -g$. With an approximation $C \approx H^{-1}$ the step is direct, $\Delta \approx -Cg$, so many implementations update an inverse approximation instead. The information is the same, learned from the same secant pairs.

**Why "no exact Hessian" is still not enough.** A dense $d\times d$ approximation holds $d^2$ numbers. With $d = 10{,}000$ that is $100{,}000{,}000$ values, about $400$ MB at 4 bytes each, for that one matrix; a real network is far larger.

**L-BFGS keeps only recent evidence.** It stores the latest $k$ pairs $(s_i, y_i)$, each two vectors of length $d$: about $2kd$ numbers, which the survey writes as $O(kd)$. With $d = 10{,}000$ and $k = 10$, that is $200{,}000$ values, about $0.8$ MB, against $400$ MB for the dense matrix. A short sequence of vector operations, the two-loop recursion, applies the implied inverse-curvature approximation to the gradient.

What remains: L-BFGS may still need reliable gradient differences, line searches and extra gradient evaluations per step, and minibatch noise can corrupt the stored pairs.

:::key Remember IV-F
Full BFGS remembers a curvature matrix. L-BFGS remembers recent evidence and acts as if it had an inverse-curvature approximation, at $O(kd)$ memory instead of $O(d^2)$.
:::

## 3. Hessian–vector products and Hessian-free optimization

Do we need every entry of $H$? Solving $H\Delta = -g$ only requires the ability to compute $Hv$ for chosen vectors $v$. With $H = \mathrm{diag}(4, 1)$ and $v = (2, 3)^\top$, $Hv = (8, 3)^\top$: it tells us how the Hessian transforms one direction. In a network, automatic differentiation computes such a **[[hvp|Hessian–vector product]]** without ever building the $d\times d$ matrix, at roughly the cost of a couple of gradients.

:::viz hvp
:::

:::read How to read this visual
**What is drawn.** The error of the finite-difference estimate of $Hv$, against the difference step $h$, both on log scales. The function is the curved surface of chapter 2 at $(1.1, 0.9)$, with $v = (1, 1)/\sqrt2$. Coral: the forward difference $(\nabla f(x + hv) - \nabla f(x))/h$. Violet: the central difference $(\nabla f(x + hv) - \nabla f(x - hv))/(2h)$.

**What moves.** The step $h$.

**What to notice.** Each curve is a V. Too large an $h$, and the curvature of $f$ spoils the estimate (the right side). Too small, and rounding errors in subtracting two almost equal gradients take over (the left side). The central difference is far more accurate at its best $h$. Automatic differentiation avoids the trade-off entirely, which is why real Hessian-free methods use it.
:::

**What "Hessian-free" means.** Not ignoring curvature, but using it through an operator that returns $Hv$ on demand: a calculator that answers "what would this matrix do to this vector?" without showing the matrix. An iterative solver such as **[[conjugate-gradient|conjugate gradients]]** starts from a guess and improves it with one such product per inner step. One outer step might: compute $g$; define the operator $v \mapsto (H + \lambda I)v$; run CG to solve $(H + \lambda I)\Delta = -g$ approximately; check or limit the step; set $\theta_{t+1} = \theta_t + \Delta$.

:::viz cg
:::

:::read How to read this visual
**What is drawn.** Solving $Hx = b$ for a 60-dimensional positive-definite $H$ whose eigenvalues are spread from $1$ to $\kappa$. The vertical axis is the error $\|x - x^*\|_H$ relative to the start, on a log scale; the horizontal axis counts iterations, each costing one product $Hv$. Coral: conjugate gradients. Violet: gradient descent with its best fixed step, $2/(1 + \kappa)$. Dashed lines: the classical worst-case bounds, $2\left(\tfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^k$ for CG and $\left(\tfrac{\kappa - 1}{\kappa + 1}\right)^k$ for GD.

**What moves.** The condition number $\kappa$.

**What to notice.** For the same number of Hessian–vector products, CG is dramatically faster, and the gap widens with $\kappa$: its rate depends on $\sqrt\kappa$ instead of $\kappa$. CG also stops: in exact arithmetic it solves a 60-dimensional system in at most 60 iterations, and the coral curve drops off a cliff before that. The readout gives the products each method needs to reach $10^{-6}$. This is how Hessian-free methods buy Newton-quality steps with only gradient-like operations; the price is a variable number of inner iterations per outer step.
:::

A non-convex Hessian may have negative eigenvalues, which breaks the standard positive-definite CG solve. Hessian-free methods therefore often use the generalized Gauss–Newton matrix instead, the subject of chapter 12, plus damping.

:::key Remember IV-G
Avoiding Hessian storage does not avoid curvature computation. Hessian-free methods trade memory for repeated products and inner solver work.
:::

:::problem The next step
We have now met the Hessian, the Gauss–Newton matrix and the Fisher matrix. Chapter 12 asks what each one actually measures, and what the theory really promises.
:::
