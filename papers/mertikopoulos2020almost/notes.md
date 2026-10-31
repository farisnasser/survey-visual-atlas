# On the Almost Sure Convergence of Stochastic Gradient Descent in Non-Convex Problems

:::minute The paper in one minute
**Research problem:** A typical non-convex SGD theorem shows that the expected gradient norm becomes small at some iterate. That does not show that the *full training trajectory* converges, nor that it avoids saddles. The paper studies boundedness and almost-sure convergence properties of SGD under stochastic-gradient noise and decreasing step sizes, with an additional focus on strict saddles and their manifolds.

**Principal result:** Under the paper's regularity, stability and noise hypotheses, ordinary SGD does not converge to strict saddles or strict-saddle manifolds with positive probability. This differs from adding artificial isotropic perturbations to deterministic GD.

**Scope:** Almost-sure convergence toward a critical set and almost-sure strict-saddle avoidance are different logical steps. Neither is a blanket theorem for Adam or arbitrary constant-large-step training.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. "Small on average" is weaker than "converges along the run"

A bound on $\mathbb{E}\|\nabla f\|^2$ speaks about averages, not about the path one run actually takes. Example: let $X_n = 1$ with probability $1/n$ and $0$ otherwise, independently. Then $\mathbb{E}X_n = 1/n \to 0$, yet because $\sum_n 1/n = \infty$, the value $1$ keeps reappearing forever with probability one (on average about $12$ times among the first $10^5$ draws, since $\sum_{n \le 10^5} 1/n \approx 12.1$). With probability $1/n^2$ instead, only finitely many ones ever occur. **Almost-sure convergence** asks for the second kind of behaviour: the trajectory itself settles, with probability one.

### 2. Step sizes $1/n^p$: how fast may they shrink?

The classical Robbins–Monro conditions ask $\sum_n \gamma_n = \infty$ (enough total movement) and $\sum_n \gamma_n^2 < \infty$ (noise eventually averages out); for $\gamma_n \propto 1/n^p$ that means $p \in (1/2, 1]$. The paper goes beyond this. If the noise has finite $q$-th moments, its Theorem 2 allows $p \in (2/(q+2), 1]$: $p > 1/2$ for $q = 2$, $p > 1/3$ for $q = 4$, and any $p \in (0, 1]$ for bounded noise ($q = \infty$), which is also the setting of its saddle-avoidance Theorem 3.

### 3. Noise must push in the unstable direction

On the notes' saddle, the unstable coordinate follows $y_{k+1} = (1 + \gamma_k)y_k - \gamma_k\zeta^y_{k+1}$. If the noise had *no* $y$-component, a run started at $y_0 = 0$ would stay on the stable set forever and converge to the saddle, just like deterministic GD started there. With a $y$-component, the first kick moves $y$ off zero and the factor $1 + \gamma_k$ then amplifies it. This is why the paper assumes the noise is **uniformly exciting**: it has a nonzero component in every direction, so the trajectory cannot stay on the thin stable set.

### 4. Saddles can come in whole manifolds

Not every saddle is an isolated point. Take a radial function $\varphi(r) = r^4/4 - r^3 + r^2$ of $r = \|x\|$ in the plane. Its derivative $\varphi'(r) = r(r - 1)(r - 2)$ vanishes on the whole circle $r = 1$, where $\varphi''(1) = -1$: every point of the circle is a strict saddle, with Hessian eigenvalues $-1$ (radial) and $0$ (along the circle). The zero eigenvalue makes the Hessian singular, so results that assume an invertible Hessian at the saddle do not apply. The paper's Theorem 3 is stated for such **strict-saddle manifolds**.

:::key Putting the four ideas together
The paper asks for almost-sure statements about the trajectory, not averages (idea 1), under step sizes $1/n^p$ that may shrink more slowly or faster than Robbins–Monro depending on the noise's moments (idea 2). Noise that pushes in every direction keeps SGD off the thin stable set of a strict saddle (idea 3), even when the saddles form a whole manifold (idea 4). The notes below set up the update and state the result and its limits.
:::
:::

:::divider Study notes
:::

## 1. Research problem: what does ordinary noisy SGD approach?

A typical non-convex SGD theorem shows that the expected gradient norm becomes small at some iterate. That does not show that the *full training trajectory* converges, nor that it avoids saddles. The paper studies boundedness and almost-sure convergence properties of SGD under stochastic-gradient noise and decreasing step sizes, with an additional focus on strict saddles and their manifolds.

## 2. Start with the stochastic update

Write an unbiased stochastic gradient as $g(x_k, \xi_k) = \nabla f(x_k) + \zeta_{k+1}$, so

$$x_{k+1} = x_k - \gamma_k\big(\nabla f(x_k) + \zeta_{k+1}\big), \qquad \mathbb{E}[\zeta_{k+1} \mid \mathcal{F}_k] = 0.$$

Under a Lipschitz-gradient condition, the descent lemma implies

$$f(x_{k+1}) \le f(x_k) + \nabla f(x_k)^\top(x_{k+1} - x_k) + \frac{L}{2}\|x_{k+1} - x_k\|^2.$$

Taking conditional expectation and assuming $\mathbb{E}[\|\zeta_{k+1}\|^2 \mid \mathcal{F}_k] \le \sigma^2$ gives the illustrative inequality

$$\mathbb{E}[f(x_{k+1}) \mid \mathcal{F}_k] \le f(x_k) - \gamma_k\Big(1 - \frac{L\gamma_k}{2}\Big)\|\nabla f(x_k)\|^2 + \frac{L\gamma_k^2\sigma^2}{2}.$$

The last term is the noise penalty. It suggests why step-size control matters, but by itself is *not* the complete proof of the paper's stronger almost-sure results.

## 3. The paper's distinctive contribution

The authors analyze stochastic approximation trajectories and give conditions ensuring boundedness and almost-sure approach to the critical set. They then show that, with suitable nondegenerate stochastic excitation and step-size assumptions, SGD avoids *strict saddle points and saddle manifolds* almost surely. The paper also studies local convergence rates in neighborhoods of nondegenerate local minima.

:::key Principal result
Under the paper's regularity, stability and noise hypotheses, ordinary SGD does not converge to strict saddles or strict-saddle manifolds with positive probability. This differs from adding artificial isotropic perturbations to deterministic GD.
:::

## 4. Why noise can help rather than only hurt

For the model $f(x, y) = \frac{1}{2}x^2 - \frac{1}{2}y^2$, a noisy SGD step is

$$x_{k+1} = (1 - \gamma_k)x_k - \gamma_k\zeta^x_{k+1}, \qquad y_{k+1} = (1 + \gamma_k)y_k - \gamma_k\zeta^y_{k+1}.$$

The $x$ direction is locally contracting. The $y$ direction is locally expanding; noise with a nonzero component along $y$ makes it implausible for an entire stochastic trajectory to remain on the stable set $y = 0$. This is a *teaching example*. The actual analysis must control stochastic paths over time, not merely average the noise to zero.

## 5. Assumptions and what is not proved

The claims require precise smoothness, growth/stability, noise-excitation and decreasing-step conditions. They are not assertions that arbitrary SGD hyperparameters or heavy-tailed minibatch noise always satisfy these hypotheses. Avoidance of strict saddles does not automatically handle degenerate saddles, prove a finite-time escape rate, or guarantee global optimality for neural networks.

:::warn Scope
Almost-sure convergence toward a critical set and almost-sure strict-saddle avoidance are different logical steps. Neither is a blanket theorem for Adam or arbitrary constant-large-step training.
:::

This paper supports **RQ2** and **RQ4**, and gives context for the explicit escape techniques in **RQ3**.

## 6. Our notes

The useful idea here is that the stochasticity already present in SGD can help keep it from settling at a strict saddle. The paper moves beyond expected-gradient bounds to statements about entire trajectories. I would connect it to Panageas et al. on decreasing steps, while stressing that the answer depends on noise assumptions and still does not say that escaping a saddle is fast. **Original source:** [https://papers.nips.cc/paper/2020/file/0cb5ebb1b34ec343dfe135db691e4a85-Paper.pdf](https://papers.nips.cc/paper/2020/file/0cb5ebb1b34ec343dfe135db691e4a85-Paper.pdf). **Verification:** original NeurIPS manuscript; descent-lemma example is pedagogical.

> **Primary source:** Mertikopoulos, Hallak, Kavis & Cevher (2020), *On the Almost Sure Convergence of Stochastic Gradient Descent in Non-Convex Problems*, NeurIPS 2020. [papers.nips.cc](https://papers.nips.cc/paper/2020/file/0cb5ebb1b34ec343dfe135db691e4a85-Paper.pdf).
