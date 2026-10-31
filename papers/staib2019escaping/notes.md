# Escaping Saddle Points with Adaptive Gradient Methods

:::minute The paper in one minute
**Question:** Can adaptive gradient preconditioning make the natural stochastic noise more effective for saddle escape?

**Model/Method:** Idealized adaptive methods update $x_{t+1} = x_t - \eta A_t g_t$, where $A_t$ is estimated from stochastic-gradient second moments, as in a full-matrix RMSProp-like method.

**Finding:** Near stationarity, an inverse-square-root preconditioner can approximately whiten gradient noise; the authors give a second-order convergence analysis for an estimable RMSProp-style variant.

**Why we care:** Adaptive methods can reshape the *directions of noise*, not just rescale parameter learning rates. This is a mechanism distinct from explicitly injected perturbations.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A matrix inverse square root acts on each eigen-direction separately

A symmetric positive-definite matrix $G$ can be written $G = Q\,\mathrm{diag}(\mu_1, \dots, \mu_d)\,Q^\top$ with orthonormal eigenvectors in $Q$. Its **inverse square root** is $G^{-1/2} = Q\,\mathrm{diag}(\mu_1^{-1/2}, \dots, \mu_d^{-1/2})\,Q^\top$: it shrinks each eigen-direction by the square root of its eigenvalue. If a random vector $g$ has second moment $G$, then $G^{-1/2}g$ has second moment $G^{-1/2}GG^{-1/2} = I$: it is **whitened**.

The notes' example has a diagonal $G$. A rotated one works the same way: $G = \begin{pmatrix} 5 & 4 \\ 4 & 5 \end{pmatrix}$ has eigenvalues $9$ and $1$ along $(1, 1)/\sqrt2$ and $(1, -1)/\sqrt2$, and $G^{-1/2} = \frac13\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$, which again gives $G^{-1/2}GG^{-1/2} = I$.

### 2. Escape needs noise along the escape direction

Near a saddle, the coordinate $y$ along a negative-curvature direction (eigenvalue $-\gamma$) evolves as $y_{t+1} = (1 + \eta\gamma)\,y_t - \eta\,\xi_t$, where $\xi_t$ is the gradient noise along that direction. If $\xi_t$ has zero variance and $y_0 = 0$, then $y_t = 0$ forever: the iterate stays on the saddle's stable set. The more noise lands along the escape direction, the sooner $y$ is pushed off zero and starts to grow.

The paper measures this with $\nu$, a lower bound on the preconditioned noise in *every* direction (Definition 4.2). For plain SGD, $\nu \le \lambda_{\min}(G)$. For $G = \mathrm{diag}(1, 0.01)$ that is $0.01$. For the RMSProp preconditioner with $\epsilon = 10^{-4}$, Proposition 4.4 gives $\nu = \lambda_{\min}(G)/(\lambda_{\min}(G) + \epsilon) = 0.01/0.0101 \approx 0.990$.

### 3. A moving average estimates a moving target

The algorithm cannot compute $G = \mathbb{E}[gg^\top]$; it averages recent outer products $g_tg_t^\top$ with weights $\beta^{t-i}$. That average remembers about $1/(1 - \beta)$ steps. A long memory lowers the variance of the estimate but mixes in statistics from points the iterate has already left. The paper balances the two by tying $\beta$ to the step size, $\beta = 1 - C\eta^{2/3}$ (Sec. 5.4): the usual $\eta = 0.001$ and $\beta = 0.99$ correspond to $C = 1$, a memory of about $100$ steps. A **burn-in** of $W = \Theta(\eta^{-2/3})$ steps fills the estimate before optimization starts.

### 4. Occasional large steps and the $\tau^{-5}$ rate

The paper's tolerance is $\tau$: it seeks a $(\tau, \sqrt{\rho\tau})$-stationary point, so the gradient is at most $\tau$ and every Hessian eigenvalue is at least $-\sqrt{\rho\tau}$. Its second-order algorithm takes ordinary steps $\eta$ and, every $t_{\mathrm{thresh}}$ iterations, one larger step $r$ that lets the noise move the iterate a useful distance. The proof follows Daneshmand et al. (2018) for SGD and gives the same order, $\widetilde{O}(\tau^{-5})$: $10^{10}$ steps at $\tau = 10^{-2}$, up to constants. The preconditioner enters only through those constants, including $\nu^{-4}$, which is why a larger $\nu$ can make the bound smaller.

:::key Putting the four ideas together
The RMSProp preconditioner is the inverse square root of the gradient second moment, which whitens the noise near a stationary point (idea 1). Whitened noise reaches every direction, including the escape direction, which is what saddle escape needs (idea 2). The matrix must be estimated on the fly with a moving average tuned to the step size (idea 3), and occasional larger steps turn the noise into escape at an $\widetilde{O}(\tau^{-5})$ rate whose constants depend on the preconditioner (idea 4), made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Noise geometry can help or hinder escape

Stochastic gradient $g_t$ satisfies $\mathbb{E}[g_t] = \nabla f(x_t)$. A preconditioned update has the form

$$x_{t+1} = x_t - \eta A_t g_t = x_t - \eta A_t\nabla f(x_t) - \eta A_t\big(g_t - \nabla f(x_t)\big).$$

The first term is the deterministic descent component; the second is noise. Near a saddle with small mean gradient, the noise term can dominate the escape behavior. If stochastic variability is negligible along the most-negative curvature direction, ordinary SGD may escape slowly.

### What does a preconditioner do?

A positive-definite matrix $A$ can scale different directions differently. Full-matrix RMSProp-style preconditioning uses

$$A = (G + \epsilon I)^{-1/2}, \qquad G = \mathbb{E}[gg^\top].$$

This $G$ is the second moment of the stochastic gradients, not always their covariance: $G = \mathrm{Cov}(g) + \nabla f\,\nabla f^\top$.

### Worked example: whitening two uneven noise directions

At a stationary point suppose a sampled gradient has mean zero and covariance

$$G = \begin{pmatrix} 9 & 0 \\ 0 & 1 \end{pmatrix}.$$

The raw noise standard deviations are 3 and 1. With negligible damping,

$$A = G^{-1/2} = \begin{pmatrix} 1/3 & 0 \\ 0 & 1 \end{pmatrix}, \qquad AGA^\top = I.$$

After preconditioning, both directions have unit noise variance. Thus a previously under-explored negative-curvature direction can receive substantial stochastic excitation. This is a *toy illustration*, not proof that any particular Adam implementation achieves isotropic noise.

## 2. From idealized preconditioning to an actual algorithm

### Part A: the idealized adaptive method (IAM)

The paper first analyzes what happens *if* an appropriate preconditioner is available. Its convergence constants depend on how $A$ affects the descent direction, gradient-noise magnitude, and the smallest eigenvalue of preconditioned stochastic second moments. Definition 4.2 summarizes these properties with parameters $\Lambda_1, \Lambda_2, \Gamma, \nu, \lambda_-$. These replace a universal claim that every adaptive rule helps.

### Part B: estimating the matrix

The algorithm must estimate $G = \mathbb{E}[gg^\top]$ online. In an RMSProp-style exponential estimate,

$$\widehat{G}_t = \beta\widehat{G}_{t-1} + (1 - \beta)g_tg_t^\top, \qquad \widehat{A}_t = (\widehat{G}_t + \epsilon I)^{-1/2}.$$

The analysis includes a *burn-in* to estimate a usable preconditioner, and extra care when the algorithm makes larger steps. A large parameter move changes the gradient distribution and can make a stale estimate unreliable.

### The main guarantee—and its scope

Theorems 4.3–4.4 analyze the idealized/noisy-preconditioner procedure, and Corollary 4.2 converts the analysis to a carefully specified RMSProp version. In the paper’s tolerance notation, the method finds a $(\tau, \sqrt{\rho\tau})$ approximate second-order stationary point in

$$\widetilde{O}(\tau^{-5})$$

steps, with potentially substantial dependence on the preconditioner constants, smoothness, objective gap and probability. It uses periodically enlarged steps as part of the saddle-escape argument.

:::warn
**Avoid overclaiming:** The paper analyzes particular *controlled* adaptive schemes, not a general proof that every ordinary implementation of Adam or RMSProp converges to local minima from any initialization.
:::

> Read Sections 3.1, 4.1–4.2, Definition 4.2, Algorithms 3–4 and Corollary 4.2; inspect the appendix for preconditioner-estimation details.

## 3. What it contributes and what it assumes

**Contributions.** A clean split between the ideal preconditioner and its online estimation; an exact covariance calculation explaining near-isotropic transformed noise; second-order convergence results for a preconditioned stochastic-gradient scheme; a theoretical lens on why adaptivity may improve saddle escape. **Assumptions.** Unbiased stochastic gradients, smooth objective with Hessian regularity, suitable bounded stochastic moments, positive-definite/regularized preconditioner with controlled spectrum and second-moment noise, careful parameter selection and estimation burn-in. **Limitations.** Full-matrix preconditioning is expensive in high dimensions; diagonal alternatives need their own constants and assumptions. The $\widetilde{O}(\tau^{-5})$ bound is not universally superior to SGD. Guarantees are approximate second-order stationarity, not global minima or practical superiority of every adaptive optimizer.

## Connections across the survey

**Daneshmand (2018)** requires natural SGD noise to be correlated with a negative-curvature direction (CNC); Staib et al. use *preconditioning* to make that excitation less direction-dependent. **Jin (2017)** adds external isotropic perturbations; here favorable noise geometry can arise from the optimizer’s own gradient statistics. **LENA (Ref 84)** instead controls noisy recursive-gradient estimation during occasional perturbation phases.

## Quick reference

| Concept | Meaning |
|---|---|
| $G = \mathbb{E}[gg^\top]$ | Gradient second moment; equals covariance only at zero mean. |
| $A = (G + \epsilon I)^{-1/2}$ | Preconditioner that equalizes directions when well estimated. |
| $AGA^\top \approx I$ | Noise approximately isotropic near stationarity, if damping small. |
| Burn-in | Gradient-statistics estimation before normal optimization. |
| $\widetilde{O}(\tau^{-5})$ | Second-order convergence order under controlled preconditioner assumptions. |

### Papers Master note

:::plain
Staib et al. view adaptive methods as preconditioned SGD. Full-matrix inverse-square-root second-moment scaling can whiten intrinsic gradient noise near stationary points, improving saddle-escape directions; a controlled RMSProp-style method gains conditional approximate second-order convergence guarantees.
:::

> **Source:** [original ICML 2019 PDF](https://proceedings.mlr.press/v97/staib19a/staib19a.pdf), Sections 3–4, Eq. (3), Definition 4.2, Theorem 4.4 and Corollary 4.2.
