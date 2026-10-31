# First-order Stochastic Algorithms for Escaping From Saddle Points in Almost Linear Time

:::minute The paper in one minute
**Question:** Can stochastic first-order gradients reveal negative curvature with nearly linear cost in parameter dimension?

**Model/Method:** NEON/NEON+: a specially centered gradient iteration amplifies directions where the Hessian is negative, without explicitly computing Hessian-vector products.

**Finding:** NEON provides provable negative-curvature directions; combining it with first-order stochastic optimization yields high-probability second-order stationarity, including a stated $\widetilde{O}(d/\varepsilon^{3.5})$ runtime regime.

**Why we care:** The Hessian's escaping directions can be detected indirectly using the same primitive as backpropagation: gradients.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Repeated multiplication finds the most negative direction, at a speed set by $\gamma$

Multiplying a vector again and again by $I - \eta H$ is the **power method**: the component with the largest multiplier wins. With $0 < \eta \le 1/L$ (where $L$ bounds the Hessian's eigenvalues), the largest multiplier is $1 + \eta\gamma$ for the most negative eigenvalue $-\gamma$, but directions with eigenvalue near $0$ have multiplier near $1$ and compete. Take $H = \mathrm{diag}(1, 0, -\gamma)$, $\eta = 1$, and $u_0 = (1, 1, 1)/\sqrt{3}$. For $\gamma = 0.1$, the cosine between $u_t$ and the escape direction $e_3$ exceeds $0.99$ after $21$ steps; for $\gamma = 0.01$ it takes $196$. The step count grows like $1/\gamma$. Accelerated (momentum or Lanczos-type) iterations need only about $1/\sqrt{\gamma}$, which is what NEON+ achieves with gradients.

### 2. The search must start from a random point

A component that starts at exactly zero stays zero under $I - \eta H$: if $u_0$ has no part along the escape direction, no number of steps will find it. Starting from a point drawn uniformly on a small sphere gives that component a nonzero size with probability one, typically about $1/\sqrt{d}$ of the radius. This explicit randomness only seeds the search; the iteration itself is deterministic given the gradients.

### 3. A detected direction pays for itself

Once a unit vector $v$ with $v^\top H v \le -c\gamma$ is found, a step of length $c\gamma/L_2$ along $\pm v$ ($L_2$ is the Hessian's Lipschitz constant) lowers $f$ by at least $c^3\gamma^3/(3L_2^2)$, the paper's Lemma 1. With $c = 1$, $\gamma = 0.1$ and $L_2 = 1$ that is at least $3.3 \times 10^{-4}$, so an objective gap of $\Delta = 1$ can pay for at most about $3000$ such escapes. Bounding the number of escapes this way, and the cost of each search, gives the total complexity.

:::key Putting the three ideas together
A gradient difference stands in for a Hessian-vector product (notes §1), so iterating it from a random start (idea 2) runs a power method that singles out the most negative direction in about $1/\gamma$ steps, or $1/\sqrt{\gamma}$ with acceleration (idea 1). Each direction found buys a fixed decrease (idea 3). Interleaving these searches with a first-order stochastic optimizer gives the paper's almost-linear-in-$d$ runtime, as the notes below explain.
:::
:::

:::divider Study notes
:::

## 1. Why first-order information can reveal second-order geometry

A stochastic nonconvex objective has the form $F(x) = \mathbb{E}_\xi[f(x; \xi)]$. We might know only sampled gradients $\nabla f(x; \xi)$, not the full Hessian. At a stationary saddle, $\nabla F(x)$ is nearly zero while $\lambda_{\min}(\nabla^2 F(x)) < 0$. NEON asks whether we can discover this negative direction using gradient differences.

### Start with a centered auxiliary function

Define, at fixed reference point $x$,

$$\widehat{F}_x(u) = F(x + u) - F(x) - \nabla F(x)^T u.$$

By Taylor's theorem, when $u$ is small,

$$\widehat{F}_x(u) \approx \tfrac12 u^T H u, \qquad \nabla\widehat{F}_x(u) = \nabla F(x + u) - \nabla F(x) \approx Hu,$$

where $H = \nabla^2 F(x)$. Notice: two gradients give a local approximation to a Hessian-vector product. GD on this centered function therefore behaves near the origin as

$$u_{t+1} \approx (I - \eta H)\,u_t.$$

A negative Hessian eigenvalue $-\gamma$ becomes multiplier $1 + \eta\gamma > 1$, so its component grows faster than directions of positive curvature.

### Worked eigenvalue example

Let $H = \mathrm{diag}(2, -1)$, $u_0 = (0.1, 0.1)$ and $\eta = 0.2$. Then

$$u_1 = (0.6 \cdot 0.1,\; 1.2 \cdot 0.1) = (0.06, 0.12), \qquad u_2 = (0.036, 0.144).$$

The positive-curvature component shrinks while the negative one grows. After normalization, $u_t$ becomes increasingly aligned with the escaping eigenvector $(0, 1)$. This is the NEON intuition, not the paper's full stopping or confidence argument.

## 2. From intuition to NEON and NEON+

### NEON (Algorithm 1 and Theorem 1)

Start with a small random $u_0$ around the reference point $x$. Repeatedly apply gradient steps to the locally centered objective and monitor a specified decrease test. If the objective exhibits sufficient negative curvature, the iteration can find a vector $u$ with

$$\frac{u^T \nabla^2 F(x)\, u}{\|u\|^2} \le -\widetilde{\Omega}(\gamma)$$

with high probability when $\lambda_{\min}(\nabla^2 F(x)) \le -\gamma$, subject to the theorem's parameter conditions. If it returns the designated null result, it provides a corresponding negative-curvature absence conclusion to tolerance and confidence.

### Why NEON+ matters

Ordinary gradient iterations grow the negative-curvature component on a scale related to $1/\gamma$. NEON+ accelerates the centered dynamics using a momentum-style/Nesterov mechanism, improving the negative-curvature search's dependence toward $1/\sqrt{\gamma}$ (up to logarithmic and problem factors). This is important because a small negative eigenvalue is otherwise hard to detect.

### Use the escaping vector to descend

If $v$ is normalized and satisfies $v^T H v \le -c\gamma$, choose an appropriate signed step of scale $\gamma/L_2$ along $v$, where $L_2$ is Hessian Lipschitz constant. A Taylor bound implies expected or deterministic objective decrease of order $\gamma^3/L_2^2$ (Lemma 1). Thus the algorithm converts detected negative curvature into genuine improvement.

### Framework and rates

The paper combines NEON-type curvature search with first-order optimizers (SGD, stochastic momentum, SCSG). In its stated stochastic setting and runtime model, a NEON–SCSG variant finds an $(\varepsilon, \sqrt{\varepsilon})$-approximate second-order stationary point with high probability in $\widetilde{O}(d/\varepsilon^{3.5})$ time. The almost linear claim concerns dependence on $d$ and the chosen oracle-cost model; it is not an $O(\varepsilon^{-3.5})$ convergence rate without any dimensional or regularity factors.

:::plain What to read
Section 3's assumptions, Section 4's centered function / Algorithms 1–2 / Theorem 1, Section 5's framework and Table 1, then Figure 1 and experiments in Section 6. Appendix is needed only for proof constants.
:::

## 3. What the authors establish and where it stops

**Contributions.** (i) A general first-order way to detect Hessian negative curvature using gradient-difference dynamics; (ii) an accelerated NEON+ variant; (iii) combinations with stochastic first-order optimizers providing second-order guarantees; (iv) numerical comparisons with power and Lanczos methods in selected models.

**Assumptions.** Individual sample objectives require smooth gradients and Hessians, and the stochastic-gradient noise obeys a stated exponential-moment/bounded-variance-type condition (Assumption 1). The global complexity also assumes a finite initial objective gap. Parameter choices depend on smoothness constants and target curvature threshold. **Limitations.** These are oracle/runtime guarantees for smooth stochastic objectives, not proofs of global minima for arbitrary deep ReLU networks. Explicit randomization is still used to seed the negative-curvature search. Claimed speedups depend on cost models and dimension; full practical deep-learning speed is not proven.

### Important comparisons

Jin (2017) injects perturbations to induce escape directly. Daneshmand (2018) assumes intrinsic stochastic gradients align with negative curvature (CNC). Xu–Jin–Yang instead constructs a gradient-only negative-curvature search mechanism. NEON2 (Ref 33) replaces/strengthens this detection primitive with a different reduction, improving several oracle complexities.

### Quick reference

| Quantity | Interpretation |
|---|---|
| $\widehat{F}_x(u)$ | Centered local function; removes constant and linear terms. |
| $\nabla F(x + u) - \nabla F(x)$ | First-order proxy for $Hu$. |
| $(I - \eta H)u$ | Linearized iteration; negative eigendirections expand. |
| $u^T H u / \lVert u\rVert^2 < 0$ | Certificate of a negative-curvature direction. |
| $\widetilde{O}(d/\varepsilon^{3.5})$ | Particular stochastic high-probability runtime, not a universal GD bound. |

:::plain Papers Master note
Xu–Jin–Yang introduce NEON and NEON+, gradient-only negative-curvature detectors based on centered gradient differences. Coupled with stochastic first-order methods they obtain high-probability approximate second-order stationarity, including nearly linear-in-dimension runtimes under smoothness and stochastic-noise assumptions.
:::

> **Source:** original NeurIPS 2018, Sections 3–6, Theorem 1, Table 1. Xu, Jin & Yang (2018), *First-order Stochastic Algorithms for Escaping From Saddle Points in Almost Linear Time*, NeurIPS 2018. [papers.nips.cc (paper)](https://papers.nips.cc/paper_files/paper/2018/file/217e342fc01668b10cb1188d40d3370e-Paper.pdf)
