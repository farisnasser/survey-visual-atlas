# Gradients, Stochasticity, and Saddle Points

:::minute The paper in one minute
**Problem:** GD and SGD can find points with small gradient, but they may be saddles. Earlier escape guarantees often have polynomial dependence on parameter dimension $d$.

**Method:** Analyze carefully designed perturbed gradient methods, using the geometry of random perturbations near strict negative-curvature directions.

**Main result:** Perturbed GD/SGD reach approximate *second-order* stationary points with a *polylogarithmic* dependence on dimension and complexity comparable (up to logarithms and problem constants) to obtaining first-order stationarity in corresponding models.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Escape time grows only with the logarithm of how close you start

Near a strict saddle, the unstable coordinate is multiplied by $1 + \eta$ per GD step (in the notes' toy saddle). Starting at distance $|v_0|$ and stopping at distance $r$ takes $\log(r/|v_0|)/\log(1 + \eta)$ steps. With $\eta = 0.1$ and $r = 1$, a start at $10^{-3}$ needs about $72$ steps, and a start a million times closer, at $10^{-9}$, needs about $217$. Getting much closer to the saddle costs only additively more time.

### 2. In high dimension, a random kick has a small but sufficient component along any fixed direction

The paper's main algorithm adds Gaussian noise $\xi \sim \mathcal{N}(0, (r^2/d)I)$, whose squared length averages $r^2$. Its component along one fixed unit direction, such as the escape direction, is Gaussian with standard deviation $r/\sqrt d$: $0.01$ for $r = 1$ and $d = 10^4$. A component $100$ times smaller than a kick of size $1$ costs, by idea 1, about $48$ extra steps at $\eta = 0.1$. Only rarely is the component tiny: it falls below $10^{-4}$ with probability about $0.008$. This is how dimension enters only through logarithms.

### 3. Ordinary SGD noise need not point where it is needed

SGD noise has its own covariance, set by the data. It can be zero along the escape direction. On the toy saddle $\tfrac12u^2 - \tfrac12v^2$, noise that only moves $u$ never changes $v = 0$, so the iterates never leave the saddle. Isotropic noise reaches every direction. This is why the paper perturbs explicitly, and why its guarantees are stated for perturbed methods rather than for unmodified SGD.

### 4. Many rare failures add up at most linearly

A run may pass near many saddles. If each escape fails with probability at most $p$, then the chance that any of $K$ escapes fails is at most $Kp$ (the **union bound**). With $K = 1000$ and $p = 10^{-6}$, that is $10^{-3}$. In the paper's Lemma 20 the failure probability of one escape decays like $2^{-\iota}$ in a parameter $\iota$ that is only logarithmic in $d$ and the other constants, so keeping the total small costs only logarithmic factors.

:::key Putting the four ideas together
Escape time depends only logarithmically on how close to the saddle you start (idea 1), and an isotropic perturbation puts a component of size about $r/\sqrt d$ along the escape direction (idea 2), so dimension costs only logarithms. Explicit noise is needed because data noise may miss that direction (idea 3), and a union bound over many escapes keeps the total failure probability small (idea 4). The notes below explain the mechanism and its limits.
:::
:::

:::divider Study notes
:::

## 1. The distinction they are trying to close

A first-order stationary point (FOSP) meets $\|\nabla f(x)\| \le \epsilon$. A second-order stationary point (SOSP) additionally meets a lower Hessian bound, typically

$$\boxed{\|\nabla f(x)\| \le \epsilon, \qquad \lambda_{\min}(\nabla^2 f(x)) \ge -\sqrt{\rho\epsilon}},$$

where $\rho$ controls variation of the Hessian. A point $x$ with very small gradient may nevertheless have $\lambda_{\min}(\nabla^2 f(x)) \ll 0$: a negative-curvature direction permits descent.

## 2. Why a strict saddle can trap ordinary GD for a long time

Take the local toy saddle $f(u, v) = \frac12u^2 - \frac12v^2$. Its gradient is $(u, -v)$, hence

$$u_{t+1} = (1 - \eta)u_t, \qquad v_{t+1} = (1 + \eta)v_t.$$

Along $u$, GD contracts; along $v$, it expands. But if $v_0 = \delta$ is tiny, the number of steps needed to reach a visible distance $r$ is

$$T_{\mathrm{esc}} \approx \frac{\log(r/|\delta|)}{\log(1 + \eta)}.$$

So asymptotic avoidance of a strict saddle is not the same as a *uniform finite-time escape guarantee*.

## 3. Perturb only when the gradient is small

The paper's main PGD (Algorithm 1) adds a small isotropic Gaussian perturbation at every step, $x_{t+1} = x_t - \eta\big(\nabla f(x_t) + \xi_t\big)$ with $\xi_t \sim \mathcal{N}(0, (r^2/d)I)$. Its streamlined proof uses a variant (Algorithm 4) that runs descent normally, then injects a random vector when progress is small:

$$x \leftarrow x - \eta\xi, \quad \xi \sim \mathrm{Unif}(B(0, r)), \qquad x \leftarrow x - \eta\nabla f(x).$$

A random kick usually contains a nontrivial component along the unstable eigenvector. More subtly, the set of kicks from which GD remains stuck resembles a thin slab of the perturbation ball. Two nearby starting points separated sufficiently along the negative-curvature direction cannot both stay trapped for many steps, because their difference expands. A volume argument makes failure unlikely, with only logarithmic dimension dependence.

## 4. What the guarantees do and do not say

For deterministic smooth objectives with Lipschitz Hessians, perturbed GD can find an approximate SOSP in a number of steps comparable to GD's $\widetilde{O}(\epsilon^{-2})$ first-order rate. The paper extends the picture to stochastic settings and carefully tracks perturbation/error probabilities; stochastic-query rates depend on the noise and oracle assumptions. The *qualitative* claim shared by the results is that second-order stationarity need not incur polynomial dependence on $d$.

:::warn Do not overstate the conclusion.
A general SOSP may be a degenerate saddle or a suboptimal local minimum. Obtaining an SOSP is not global optimization without additional landscape structure. The theorem's smoothness and Hessian-Lipschitz conditions are not automatic for ReLU objectives. Nor does the analysis prove a generic fast escape from arbitrarily high-order flat saddles.
:::

## 5. Connection to earlier and later work

This 2021 journal treatment synthesizes and extends the line of efficient perturbation-based saddle escape that includes Jin et al. (2017), already represented in our master list. Whereas the 2017 work established the memorable perturbed-GD mechanism, this paper emphasizes a wider deterministic/stochastic picture and near-dimension-free efficiency. It does *not* mean ordinary unmodified SGD always has the required noise geometry.

:::key Survey interpretation.
RQ3 asks whether saddle escape requires a full Hessian. The answer here is no for approximate strict-saddle escape under smoothness assumptions: a well-timed random perturbation plus gradients can suffice. RQ4 asks whether this captures real network training; the structural assumptions remain the caveat.
:::

**Our takeaway.** The clever move is to add randomness only when the gradient becomes uninformative, and then let deterministic dynamics amplify that random displacement in an unstable direction.

> **Source:** [https://doi.org/10.1145/3418526](https://doi.org/10.1145/3418526); author version [https://arxiv.org/pdf/1902.04811](https://arxiv.org/pdf/1902.04811).
