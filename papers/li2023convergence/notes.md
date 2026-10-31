# Convergence of Adam Under Relaxed Assumptions

:::minute The paper in one minute
**Problem.** Adam works remarkably well in large-scale deep learning, but a convergence proof is not automatic: adaptive denominators couple all previous stochastic gradients, and earlier proofs often assume globally bounded gradients or a uniform global smoothness constant.

**Proof idea.** The central technical innovation is an argument showing that *along the Adam iterates*, gradient magnitudes stay bounded with high probability under the specified generalized smoothness and oracle assumptions.

**Reported result (authors).** Adam finds an $\epsilon$-stationary point, $\|\nabla f(x)\| \le \epsilon$, with $\mathcal{O}(\epsilon^{-4})$ stochastic-gradient complexity under the paper's relaxed conditions and parameter choices. The authors additionally analyze a variance-reduced Adam variant with $\mathcal{O}(\epsilon^{-3})$ complexity.

**Do not claim.** "Adam is proved to escape all saddle points" or "Adam converges to global minima." The target is first-order stationarity.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Adam keeps two running averages and corrects their start

Adam's $m_t$ and $v_t$ are **exponential moving averages**: each step keeps a fraction $\beta$ of the old value and mixes in $1 - \beta$ of the new gradient (or its square). Starting from $0$ they are biased low. With a constant gradient $g = 2$ and $\beta_1 = 0.9$, $m_t = 2(1 - 0.9^t)$: only $0.2$ after one step and about $1.30$ after ten. Dividing by $1 - \beta_1^t$ removes this bias exactly: $\widehat m_t = 2$ at every step. The second moment is corrected the same way with $\beta_2$. Each coordinate's step is then $\eta_t\widehat m_t/(\sqrt{\widehat v_t} + \delta)$: about $\eta_t$ in size when the gradient is steady, whatever its scale.

### 2. Stationarity targets and how complexity is counted

A point with $\|\nabla f(x)\| \le \epsilon$ is **$\epsilon$-stationary**. For non-convex $f$ this is the standard target: it rules out steep slopes but says nothing about minima versus saddles. **Complexity** counts how many stochastic gradients an algorithm needs to guarantee such a point, as a function of $\epsilon$. The difference between rates is large: for $\epsilon = 0.01$, $\epsilon^{-4} = 10^8$ but $\epsilon^{-3} = 10^6$, a hundred times fewer.

### 3. A wider smoothness class: curvature allowed to grow with the gradient

The paper bounds the Hessian norm by a function of the gradient norm that grows slower than a square, $L_0 + L_1\|\nabla f\|^q$ with $q < 2$. Simple functions show the range of exponents. For $e^x$, $f'' = f'$: $q = 1$. For $x^4/4$, $f' = x^3$ and $f'' = 3x^2 = 3|f'|^{2/3}$: $q = 2/3$. For $1/x$ on $x > 0$, $f'' = 2/x^3 = 2|f'|^{3/2}$: $q = 3/2$, outside the earlier $q = 1$ class of Zhang et al. but inside this paper's. None of these has a global Lipschitz gradient.

### 4. Bounded gradients on the path give a local smoothness constant

If the gradient stays below $G$ along the iterates, the smoothness condition caps the curvature there at $L_0 + L_1 G^q$; for example $1 + 3 \cdot 10 = 31$ when $L_0 = 1$, $L_1 = 3$, $q = 1$ and $G = 10$. The paper's Lemma 3.4 makes this precise: near a point with a bounded gradient, the gradient is locally Lipschitz. A local constant is all the classical descent argument needs, so the hard part is to show the gradients stay bounded along Adam's path, rather than assuming it everywhere.

:::key Putting the four ideas together
Adam rescales each coordinate by running averages (idea 1) and is judged by how many gradients it needs to reach an $\epsilon$-stationary point (idea 2). The paper allows any loss whose curvature grows sub-quadratically with the gradient (idea 3), and its proof shows that the gradients stay bounded along Adam's own trajectory, which supplies the local smoothness constant the classical analysis needs (idea 4). The notes below give the setting, the rates and what is not claimed.
:::
:::

:::divider Study notes
:::

## 1. The problem and why the paper exists

Adam works remarkably well in large-scale deep learning, but a convergence proof is not automatic: adaptive denominators couple all previous stochastic gradients, and earlier proofs often assume globally bounded gradients or a uniform global smoothness constant. The authors ask whether one can prove convergence of Adam to *first-order stationary points* under a more realistic class of objectives, without assuming in advance that every gradient is bounded.

## 2. Start from the optimization objective and Adam update

The paper studies non-convex stochastic optimization

$$\min_{x \in \mathbb{R}^d} f(x), \qquad f(x) = \mathbb{E}_\xi[f(x; \xi)], \qquad \mathbb{E}[g_t \mid x_t] = \nabla f(x_t).$$

For orientation, the usual Adam statistics and update have the form

$$\begin{aligned} m_t &= \beta_1 m_{t-1} + (1 - \beta_1)g_t, \\ v_t &= \beta_2 v_{t-1} + (1 - \beta_2)(g_t \odot g_t), \\ x_{t+1} &= x_t - \eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t} + \delta}, \end{aligned}$$

with coordinatewise division and bias corrections. The denominator makes the effective step size coordinate dependent. These generic equations introduce the method; the paper's theorem requires its own parameter choices and technical conditions.

## 3. Why global smoothness and bounded gradients are restrictive

A classical assumption is $\|\nabla^2 f(x)\| \le L$ everywhere. The paper instead allows local curvature to grow with the gradient norm: its generalized smoothness class bounds the Hessian norm (where it exists) by a *sub-quadratic* function of $\|\nabla f(x)\|$. A schematic example of this idea is

$$\|\nabla^2 f(x)\| \le L_0 + L_1\|\nabla f(x)\|^q, \qquad 0 \le q < 2,$$

where the exact regularity specification and constants are governed by the paper. This is *not* the assertion that every neural network satisfies the inequality globally.

**Teaching illustration, not an experiment from the paper.** For $f(x) = e^x$, $f'(x) = f''(x) = e^x$; no global constant $L$ bounds the second derivative over $\mathbb{R}$, although curvature scales directly with gradient size. A proof can exploit such local information instead of requiring a single worst-case curvature bound.

## 4. Proof idea: bound the actual trajectory rather than assume it

The central technical innovation is an argument showing that *along the Adam iterates*, gradient magnitudes stay bounded with high probability under the specified generalized smoothness and oracle assumptions. Conceptually the proof first controls possible excursions into high-gradient/high-curvature regions, then establishes sufficient descent in expectation through the adaptive update. This is different from assuming $\sup_x\|\nabla f(x)\| < \infty$ for the entire domain.

:::theorem Reported result (authors)
Adam finds an $\epsilon$-stationary point, $\|\nabla f(x)\| \le \epsilon$, with $\mathcal{O}(\epsilon^{-4})$ stochastic-gradient complexity under the paper's relaxed conditions and parameter choices. The authors additionally analyze a variance-reduced Adam variant with $\mathcal{O}(\epsilon^{-3})$ complexity. These are *first-order* guarantees; they do not assert negative-curvature escape or global optimality.
:::

## 5. Mathematical intuition behind the rates

A typical non-convex descent analysis tries to bound a sum of gradient norms:

$$\sum_{t<T}\eta_t\,\mathbb{E}\|\nabla f(x_t)\|^2 \;\lesssim\; f(x_0) - \inf f + \text{noise/adaptivity terms}.$$

For stochastic gradients, controlling the noise term can lead to $T$ of order $\epsilon^{-4}$ for an $\epsilon$-gradient target. Variance reduction makes gradient tracking more accurate as optimization proceeds, improving the oracle dependence to $\epsilon^{-3}$ in the paper's method. This display is an explanatory schematic, not the authors' verbatim theorem.

## 6. Assumptions, limits, and our survey's RQs

The relaxed curvature condition and stochastic-oracle controls remain substantive assumptions; the result does not justify unrestricted Adam hyperparameters on arbitrary losses. The bounded-trajectory conclusion is probabilistic and is not a global bounded-gradient property. There is no strict-saddle or SOSP statement. This work supports **RQ2** (adaptive optimization dynamics) and **RQ4** (relaxing unrealistic global smoothness) more directly than RQ3.

:::warn Do not claim
"Adam is proved to escape all saddle points" or "Adam converges to global minima." The target is first-order stationarity, and finite-time complexity depends on the paper's assumptions.
:::

## 7. Our notes

The contribution is not a new Adam update but a more realistic explanation of why Adam can converge. Instead of assuming gradients are bounded everywhere, the authors show how to control them along the actual trajectory, then obtain first-order rates. This is useful when discussing the gap between classical optimization assumptions and modern adaptive training.

> **Original source:** [https://papers.neurips.cc/paper_files/paper/2023/hash/a3cc50126338b175e56bb3cad134db0b-Abstract-Conference.html](https://papers.neurips.cc/paper_files/paper/2023/hash/a3cc50126338b175e56bb3cad134db0b-Abstract-Conference.html).
