# How Gaussian Data Can Produce Heavy-Tailed SGD Iterates

:::minute The paper in one minute
**Question:** Can SGD develop power-law fluctuations even when the input data have light tails?

**Model/Method:** View constant-step streaming least-squares SGD as a random affine recursion, with random curvature multiplying the current iterate.

**Finding:** Under contraction and nondegeneracy conditions, the stationary law has a power-law tail; its variance can be infinite despite finite moments of every order in the data.

**Why we care:** Noise is state dependent. Learning rate, batch size, dimension and curvature affect long-run fluctuations; this is not itself an efficient saddle-escape or generalization theorem.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. SGD on least squares multiplies the iterate by a random number

Take one data pair $(a, y)$ and the loss $\tfrac12(ax - y)^2$ in one dimension. Its gradient is $a(ax - y)$, so one SGD step with step size $\eta$ is

$$x^+ = x - \eta\, a(ax - y) = (1 - \eta a^2)\, x + \eta a y .$$

The iterate is **multiplied** by the random factor $M = 1 - \eta a^2$ and then shifted by $q = \eta a y$. This is a **random affine recursion**. The noise is not just added on top: it rescales whatever the iterate already is, and that is what can create heavy tails.

### 2. Typical growth and average growth of a product differ

After $k$ steps the start is multiplied by $M_k \cdots M_1$. Its logarithm is a sum, $\sum \log|M_j|$, so the product typically shrinks when $\mathbb{E}\log|M| < 0$. Averages of powers can still grow. Let $M$ be $0.2$ or $1.6$ with equal probability. Then $\mathbb{E}\log|M| \approx -0.57$: typical products shrink. But $\mathbb{E}M^2 = \tfrac12(0.04 + 2.56) = 1.3 > 1$: the average of the squared product grows like $1.3^k$, driven by rare runs of $1.6$s. The function $h(s) = \mathbb{E}|M|^s$ measures this: here $h(1) = 0.9$ and $h(2) = 1.3$.

### 3. A tail index says which moments exist

A **power-law tail** $P(|X| > t) \approx c\,t^{-\alpha}$ for large $t$ decays slowly; $\alpha$ is the **tail index**. Moments come from the tail through

$$\mathbb{E}|X|^p = \int_0^\infty p\,t^{p-1}\, P(|X| > t)\, dt ,$$

and with $t^{p-1} \cdot t^{-\alpha}$ in the integrand this is finite exactly when $p < \alpha$. So $1 < \alpha < 2$ means a finite mean and an infinite variance. A Gaussian has $P(|X| > t)$ falling faster than any power: all moments are finite.

### 4. The tail index is where $h(\alpha) = 1$

At stationarity $x$ and $Mx' + q$ have the same law, with $x'$ an independent copy. For huge $t$ the shift $q$ hardly matters, so $P(|x| > t) \approx \mathbb{E}\,P(|x'| > t/|M|)$. If $P(|x| > t) \approx c\,t^{-\alpha}$, the right side is $c\,t^{-\alpha}\,\mathbb{E}|M|^\alpha$, and the two match only if $\mathbb{E}|M|^\alpha = 1$. This is a heuristic; the paper's Theorem 2 makes it rigorous for matrices. For the two-point example, $h(\alpha) = 1$ at $\alpha \approx 1.35$, and a simulation of $x^+ = Mx + 1$ shows it: $P(x > 100) \approx 0.009$ and $P(x > 1000) \approx 0.0004$, a ratio of about $22 \approx 10^{1.35}$.

:::key Putting the four ideas together
Least-squares SGD multiplies the iterate by random curvature (idea 1). When products shrink typically but not on average (idea 2), the stationary law has a power-law tail whose index solves $h(\alpha) = 1$ (idea 4), and that index decides which moments are finite (idea 3). Gaussian data are enough to get $\alpha < 2$, an infinite variance, as the notes below work out.
:::
:::

:::divider Study notes
:::

## 1. Separate the data, the noise and the iterates

A light-tailed dataset does not force light-tailed optimizer iterates. The reason is multiplicative randomness: the sampled Hessian rescales the current parameter at every step. Repeated products of mostly shrinking, occasionally expanding factors can produce large excursions. An additive Gaussian approximation with a fixed deterministic contraction misses this mechanism.

$$F(x) = \tfrac12\,\mathbb{E}\big[(a^\top x - y)^2\big],$$

$$x_k = M_k x_{k-1} + q_k, \qquad M_k = I - \frac{\eta}{b}\sum_i a_i a_i^\top, \qquad q_k = \frac{\eta}{b}\sum_i a_i y_i.$$

Each new batch contains fresh, independent data. Thus the pairs $(M_k, q_k)$ are independent across $k$ and independent of the previous iterate. $M_k$ and $q_k$ within a batch need not be independent. This streaming assumption is central: a fixed finite dataset revisited for many epochs is a separate empirical setting.

## 2. What a tail index means

A power-law tail $P(|X| > t)$ behaving like $c\,t^{-\alpha}$ assigns much greater probability to extreme values than a Gaussian. Smaller $\alpha$ means heavier tails. Integrating the tail shows why a $p$th moment is finite below $\alpha$ and diverges at or above $\alpha$ for a genuine nonzero power-law tail. In particular, $1 < \alpha < 2$ permits a mean but no finite variance. A power-law stationary law need not itself be an exactly $\alpha$-stable distribution.

:::warn Convergence of laws, not parameter convergence
The distribution can approach a stationary law while individual constant-step iterates keep fluctuating. A stationary random parameter is not an optimizer that has settled at a single point.
:::

## 3. Worked example: stationary distribution, infinite variance

*Pedagogical illustration, constructed for these notes.*

Take independent $a, y \sim N(0, 1)$, scalar $x$, batch size $b = 1$ and $\eta = 1$. The population objective is $F(x) = \mathbb{E}[(ax - y)^2]/2 = (x^2 + 1)/2$, a strictly convex quadratic with minimizer zero. The stochastic update is $x^+ = (1 - a^2)x + ay$.

*Write $M = 1 - a^2$. Its second-moment multiplier is $\mathbb{E}[M^2] = 1 - 2\mathbb{E}[a^2] + \mathbb{E}[a^4] = 1 - 2 + 3 = 2$.*

A multiplier larger than one does not alone rule out a stationary distribution. Contraction of random products is controlled by $\mathbb{E}\log|M|$. Let $\varphi(z)$ be the standard Gaussian density. Since $\mathbb{E}[M] = 0$, symmetry and integration by parts give the following exact first absolute moment.

$$\mathbb{E}|1 - a^2| = 4\int_0^1 (1 - z^2)\,\varphi(z)\,dz = 4\big[z\varphi(z)\big]_0^1 = 4\varphi(1) \approx 0.967883 < 1.$$

Jensen gives $\mathbb{E}\log|M| \le \log\mathbb{E}|M| < 0$, so the multiplicative dynamics contract on average. In the scalar tail equation $h(s) = \mathbb{E}|M|^s$, we have $h(1) < 1$ and $h(2) = 2 > 1$. The nonzero root $h(\alpha) = 1$ therefore lies strictly between 1 and 2. The continuous Gaussian setup supplies the needed nondegeneracy and moment conditions.

Consequently the stationary SGD law has a finite mean and infinite variance, although both $a$ and $y$ have moments of every order. For finite $k$ and $x_0 = 0$, independence and $\mathbb{E}[y] = 0$ give $\mathbb{E}[x_k^2] = 2\mathbb{E}[x_{k-1}^2] + 1 = 2^k - 1$. Finite-time moments exist while growing without bound; there is no contradiction with convergence in distribution.

The scalar variance threshold in Proposition 5 is $\eta_{\mathrm{crit}} = 2b/[\sigma^2(d + b + 1)] = 2/3$. Our $\eta = 1$ lies above it, but stationarity is established separately by the absolute-moment calculation. A large step should not be assumed stable merely because it exceeds the variance threshold.

## 4. The stationary-law theorem

For matrix products $\Pi_k = M_k \cdots M_1$, define the top Lyapunov exponent as their asymptotic logarithmic growth rate and $h(s)$ as $\lim\big(\mathbb{E}\|\Pi_k\|^s\big)^{1/k}$. Theorem 2 states a unique stationary law and directional power-law tails when the exponent is negative and a unique positive $\alpha$ solves $h(\alpha) = 1$, under the paper's continuous-data and nondegeneracy setting. These conditions distinguish average contraction from rare expanding products.

With isotropic Gaussian $a \sim N(0, \sigma^2 I)$, Theorem 3 simplifies the product quantities to $\mathbb{E}\log\|Me_1\|$ and $\mathbb{E}\|Me_1\|^s$. Rotational symmetry makes the distribution of $\|Mv\|/\|v\|$ independent of the direction of $v$. This is a fixed-direction moment. Do not replace it by $\mathbb{E}\|M\|^s$, the one-step operator-norm moment, which is a different quantity and only supplies a general bound.

$$h(2) = 1 - 2\eta\sigma^2 + \frac{\eta^2\sigma^4}{b}\,(d + b + 1).$$

Solving $h(2) = 1$ gives $\eta_{\mathrm{crit}} = 2b/[\sigma^2(d + b + 1)]$. Below it, the Gaussian setting has $\alpha > 2$. At the threshold, $\alpha = 2$. Just above it, while average contraction still holds, $\alpha < 2$. Beyond stability, convergence is no longer guaranteed. Theorem 4 gives increasing $\alpha$ with batch size and decreasing $\alpha$ with step size or input variance when $\alpha \ge 1$, plus decreasing $\alpha$ with dimension. It does not say $\eta/b$ alone completely determines the distribution.

## 5. Rates and averaging have moment restrictions

Theorem 8 bounds the $p$-Wasserstein distance between the current and stationary laws by $h(p)^{k/p}$ times the initial distance, for $1 \le p < \alpha$. Wasserstein distance compares distributions by the cost of transporting one to another; $W_2$ is inappropriate when the stationary variance is infinite. Corollary 11 describes stable-law limits of appropriately centered sums when $\alpha < 2$, with a special normalization at $\alpha = 2$. It does not make every finite average exactly stable or Gaussian.

## 6. What the experiments measure

Synthetic experiments revisit fixed Gaussian datasets and estimate tail indices of centered, averaged runs. Figure 1 varies $\eta$, $b$, dimension and input variance; larger steps and smaller batches generally yield heavier estimated tails. RMSProp and SVRG comparisons concern the tested implementations and do not constitute a general theorem about those optimizers.

Figures 2-3 test fully connected ReLU networks on MNIST/CIFAR10 and an 11-layer VGG network on CIFAR10. Runs use 10,000 iterations and average the last 1,000. Because repeating whole-network experiments thousands of times is expensive, layer coordinates are treated as an approximately independent collection for tail estimation. This is a modeling approximation, not a proved finite-network independence result. Some layers show a stronger $\eta/b$ trend than others; VGG estimates above two are clipped to two in the plot.

## 7. What remains outside the theorem

The sharp theory is for constant-step streaming linear regression. Deep-network evidence is empirical; repeated finite-data sampling, changing learning rates and adaptive optimizers require further analysis. The paper discusses connections to flatness and generalization, but does not prove that choosing heavier tails always improves test error. It explicitly leaves the effect of tail index on saddle-escape time as future work. Several printed auxiliary formulas contain slips; these notes rely on the stated fixed-direction identity, relevant theorem qualifications and the independently derived scalar example, rather than certifying every supporting inequality.

## 8. Survey connection and reference table

Fang2019 controls finite-time escape using bounded, dispersive stochastic noise. This paper studies a stationary law generated by random curvature, where global bounded-noise assumptions may fail. Azizian2025 studies rare transitions and expected global hitting times under a different controlled-noise model. A stationary heavy tail cannot simply be inserted into either theorem without checking assumptions.

| Quantity | Interpretation |
|---|---|
| $\mathbb{E}\log\lVert\Pi_k\rVert / k$ | Typical product growth; negative supports stationarity. |
| $h(\alpha) = 1$ | Boundary between controlled and divergent moments. |
| $1 < \alpha < 2$ | Stationary mean exists; variance diverges. |
| $\eta_{\mathrm{crit}}$ | Variance boundary, not the upper stability limit. |

:::plain Papers Master - Our notes (copy)
Constant-step streaming least-squares SGD can have a power-law stationary law, including infinite variance despite light-tailed inputs. Random curvature supplies multiplicative noise. Gaussian-input tail-index identities and monotonicity have specific assumptions; finite-data neural-network evidence and generalization interpretations remain separate.
:::

Reading recommendation: Read the affine-recursion mechanism and distinguish streaming quadratic theorems from finite-data neural-network measurements. This adds a different perspective on SGD noise.

> **Source:** publisher main PDF 1-9, Theorems 2-4/8, Proposition 5, Corollary 11; supplement 1-16, especially C.1-C.6 and D. Scalar Gaussian calculation constructed for teaching. Gürbüzbalaban, Şimşekli & Zhu (2021), *The Heavy-Tail Phenomenon in SGD*, ICML 2021 (PMLR 139). [proceedings.mlr.press/v139/gurbuzbalaban21a/gurbuzbalaban21a.pdf](https://proceedings.mlr.press/v139/gurbuzbalaban21a/gurbuzbalaban21a.pdf)
