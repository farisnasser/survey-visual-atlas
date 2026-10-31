# Escaping Saddles with Stochastic Gradients

:::minute The paper in one minute
**Question:** Does SGD need artificial isotropic noise to escape saddles, or can the randomness of sampling data be enough?

**Model/Method:** Introduce *Correlated Negative Curvature* (CNC): stochastic gradients must have nontrivial squared projection onto the most negative Hessian direction. Analyze GD interrupted by stochastic steps and an SGD variant with periodic large steps.

**Finding:** Under CNC and smoothness conditions, the SGD variant reaches an approximate second-order stationary point in $\widetilde{O}(\varepsilon^{-4})$ iterations, with no *explicit* dimension factor; a GD–SGD hybrid takes $\widetilde{O}(\varepsilon^{-2})$.

**Why we care:** SGD's *intrinsic*, directionally structured noise can replace externally injected perturbations. But the guaranteed SGD algorithm uses periodic learning-rate boosts; it is not a theorem for arbitrary constant-step SGD.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A second-order stationary point also rules out strong negative curvature

A small gradient does not separate a minimum from a saddle. A **second-order stationary point** adds a curvature test: $\|\nabla f(w)\| \le \varepsilon_g$ and $\lambda_{\min}(\nabla^2 f(w)) \ge -\varepsilon_h$. At the origin of $\tfrac12 x^2 - \tfrac12 y^2$ the gradient is zero but $\lambda_{\min} = -1$, so the origin fails the test for any $\varepsilon_h < 1$.

The two tolerances are tied to one accuracy $\varepsilon$ through a power. A larger tolerance is a weaker guarantee. For $\varepsilon = 0.01$: $\varepsilon^{1/2} = 0.1$, $\varepsilon^{2/5} \approx 0.158$, $\varepsilon^{1/4} \approx 0.316$. So a curvature target $\varepsilon^{2/5}$ is looser than $\varepsilon^{1/2}$ and tighter than $\varepsilon^{1/4}$.

### 2. Noise in one direction is a quadratic form

For a random vector $g$ and a unit vector $v$, the **second moment along $v$** is

$$\mathbb{E}\big[\langle v, g\rangle^2\big] = v^\top\, \mathbb{E}[g g^\top]\, v .$$

So it depends on the matrix $\mathbb{E}[gg^\top]$ and on the direction. For a vector $u$ uniform on a sphere of radius $r$ in $\mathbb{R}^d$, symmetry makes $\mathbb{E}[uu^\top]$ a multiple of $I$, and its trace is $\mathbb{E}\|u\|^2 = r^2$. Hence $\mathbb{E}[uu^\top] = (r^2/d)\, I$: every direction gets the same $r^2/d$. Sampled gradients need not be spread out like this; their energy can sit in a few directions.

### 3. Negative curvature amplifies a kick

Near a saddle, a GD step multiplies the component along an eigenvector with eigenvalue $\lambda$ by $1 - \eta\lambda$. For $\lambda < 0$ that is $1 + \eta|\lambda| > 1$, and the loss along that direction falls like $-\tfrac12|\lambda|\,y^2$ as the component $y$ grows. With $\lambda = -1$, $\eta = 0.1$ and a kick to $y_0 = 0.2$, twenty GD steps give $y = 0.2 \cdot 1.1^{20} \approx 1.35$ and, on the quadratic model, a loss change of $-\tfrac12(1.35^2 - 0.2^2) \approx -0.88$. What matters is that the kick has a component along the negative direction at all.

### 4. Returning a random iterate turns an average into a probability

The paper's algorithms return an iterate chosen uniformly at random from $w_1, \dots, w_T$. If at most $30$ of $T = 1000$ iterates miss the target, the returned point hits it with probability at least $1 - 30/1000 = 0.97$. The proof bounds the expected fraction of bad iterates, using the fact that the loss cannot decrease by more than $f(w_0) - f^*$ in total, and this counting turns that bound into a success probability $1 - \delta$.

:::key Putting the four ideas together
The target is second-order stationarity (idea 1). Escaping a saddle needs a kick with a component along the negative-curvature direction, which then grows (idea 3). Isotropic noise gives that direction only $r^2/d$ (idea 2); CNC assumes the sampled gradients give it a fixed amount $\gamma$, which is why the bounds carry no explicit $d$. Counting bad iterates and returning a random one gives the high-probability guarantee (idea 4), made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Start with the paradox: zero full gradient, nonzero sample gradients

For a population loss $f(w) = \mathbb{E}_z[f_z(w)]$, ordinary GD uses the average $\nabla f(w)$, while SGD samples one $\nabla f_z(w)$ at a time:

$$\underbrace{w^+ = w - \eta\nabla f(w)}_{\text{GD}} \qquad \underbrace{w^+ = w - \eta\nabla f_z(w)}_{\text{SGD}}.$$

At a saddle, $\nabla f(w) = 0$ can hold while *individual* stochastic gradients are nonzero and point in opposite directions. Their average cancels; their fluctuations do not. The question is whether those fluctuations point in an *escape direction*.

## 2. The new assumption: Correlated Negative Curvature (CNC)

Let $v_{\min}$ be a unit eigenvector corresponding to the minimum Hessian eigenvalue. Equation (6) asks for some $\gamma > 0$ such that

$$\boxed{\mathbb{E}_z\big[\langle v_{\min}, \nabla f_z(w)\rangle^2\big] > \gamma.}$$

This is a bound on the *second moment of a projection*, not a requirement for noise in every direction. At an exact stationary point, the projected mean is zero, so this second moment also equals projected variance.

:::key Translate the equation into English
Find the direction in which curvature is most negative. Project the sampled gradient onto that direction. CNC says that this projection is not almost always tiny. Then sampling can kick the iterates along a direction where the saddle is unstable.
:::

## 3. Tiny worked example: the randomness was there all along

Take $f(x, y) = \tfrac12 x^2 - \tfrac12 y^2$ near $(0, 0)$. The Hessian is $\mathrm{diag}(1, -1)$, so $v_{\min} = (0, 1)$. As an *illustrative local model*, suppose individual gradients at the origin are $(0, 2)$ or $(0, -2)$ with equal probability. Then

$$\mathbb{E}[g_z] = (0, 0), \qquad \mathbb{E}[\langle v_{\min}, g_z\rangle^2] = \tfrac12\big(2^2 + (-2)^2\big) = 4.$$

GD stays at the origin; one SGD step moves $y$ to $\pm 2\eta$, and negative curvature can then amplify the displacement. This toy is pedagogical, *not* a demonstration that the paper's global bounded-gradient assumptions hold for this quadratic.

## 4. Why targeted stochastic noise can beat isotropic noise

Imagine a $d$-dimensional parameter vector with only one useful escape direction. An isotropic perturbation $u$ uniformly distributed on a sphere of radius $r$ has

$$\mathbb{E}[\langle u, v\rangle^2] = \frac{r^2}{d} \qquad (\|v\| = 1).$$

Thus, as $d$ grows, less of its energy falls along a *particular* negative-curvature eigenvector. But CNC only demands a fixed lower bound $\gamma$ on the squared projection of the *sample gradient* along that direction.

:::key Numerical scale comparison
With $r = 1$ and $d = 1000$, isotropic noise has mean squared projection $0.001$ onto one fixed unit direction. If sampling gradients instead gives projected values $+0.5$ or $-0.5$ with equal probability, the second moment is $0.25$, without spreading that energy across all $1000$ coordinates. These are illustrative numbers, not measurements from the paper.
:::

## 5. What do the two algorithms actually do?

| | Algorithm 1: CNC-PGD | Algorithm 2: CNC-SGD |
|---|---|---|
| Routine step | Full gradient: $w^+ = w - \eta\nabla f(w)$ | Sample gradient: $w^+ = w - \eta\nabla f_z(w)$ |
| Escape step | If gradient is sufficiently small and enough time passed, take $w^+ = w - r\nabla f_z(w)$ | Every $t_{\mathrm{thres}}$ iterations, take a larger step $w^+ = w - r\nabla f_z(w)$, with $r > \eta$ |
| Added random vector? | No externally sampled isotropic perturbation | No externally sampled isotropic perturbation |
| Cost caveat | Requires full gradients and threshold logic | Uses stochastic gradients, but requires a prescribed periodic step-size schedule |

:::warn A nuance that matters for the survey
The paper calls Algorithm 2 SGD *without perturbation / vanilla SGD* because it does not add a separate random vector. Nevertheless, its *proved method* periodically enlarges the learning rate; do not describe Theorem 2 as applying to arbitrary fixed-step, off-the-shelf SGD.
:::

## 6. The convergence results, with correct targets

An $(\varepsilon_g, \varepsilon_h)$ second-order stationary point satisfies

$$\|\nabla f(w)\| \le \varepsilon_g, \qquad \lambda_{\min}(\nabla^2 f(w)) \ge -\varepsilon_h.$$

For $\varepsilon_g = \varepsilon$ and $\varepsilon_h = \sqrt{\rho}\,\varepsilon^{2/5}$, the main results are:

| Result | Method | Complexity | Success |
|---|---|---|---|
| Theorem 1 | CNC-PGD | $\widetilde{O}(\varepsilon^{-2})$ | $1 - \delta$ |
| Theorem 2 | CNC-SGD | $\widetilde{O}(\varepsilon^{-4})$ | $1 - \delta$ |

These abbreviations suppress constants (including $L, \rho, \ell, \gamma, \delta$) and logarithmic terms. The displayed bounds have *no explicit $d$ factor*, but that does not mean every underlying constant is automatically independent of dimension in every application.

:::warn What the guarantee is NOT
An approximate second-order stationary point is not necessarily a global optimum. Degenerate saddle points can satisfy approximate first- and second-order conditions. A strict-saddle landscape assumption would be needed for stronger claims about proximity to local minima.
:::

## 7. Why does CNC hold for at least one real model class?

The authors do more than assume CNC: they verify it analytically for *learning half-spaces* of the form

$$f(w) = \mathbb{E}_z[\phi(w^\top z)], \qquad \nabla f_z(w) = \phi'(w^\top z)\, z.$$

The key extra condition (Assumption 3) is $|\phi''(a)| \le c\,|\phi'(a)|$, along with bounded input support $\|z\| \le 1$. **Lemma 4:** for any unit Hessian eigenvector $v$ with a *negative* eigenvalue $\lambda$,

$$\boxed{\mathbb{E}_z\big[\langle \nabla f_z(w), v\rangle^2\big] \ge (\lambda/c)^2.}$$

A bigger magnitude of negative curvature supplies a stronger lower bound on the stochastic-gradient projection. Notice the *square*: the displayed theoretical bound is proportional to $\lambda^2$, not literally to $|\lambda|$.

## 8. Proof intuition: three regimes, not one magical noise effect

The convergence reasoning separates the trajectory into three cases:

1. **Gradient appreciably large:** ordinary descent decreases the objective, by the usual smoothness inequality.
2. **Gradient small; Hessian strongly negative:** a stochastic-gradient kick has enough negative-direction energy, under CNC, to trigger progress away from a saddle; subsequent descent decreases the objective in expectation.
3. **Gradient small; negative curvature mild:** the point already meets the approximate first- and second-order target, so only limited short-run loss change needs to be controlled.

The proof then tracks expected loss decreases across iterations and returns a uniformly randomly selected visited iterate to obtain a high-probability guarantee.

:::key Why is a bigger step inserted periodically?
Small standard SGD steps may not move far enough along the escape direction before a useful descent episode. The occasional $r$-step magnifies the directional stochastic variation; the smaller $\eta$-steps exploit the escape. This uses sampling noise *already contained in gradients*, instead of introducing an independent perturbation vector.
:::

## 9. What the experiments actually show

**Figure 1:** In feed-forward MNIST classification networks, the measured stochastic-gradient energy along the most negative Hessian direction remains appreciable as width/depth increases, whereas isotropic noise projections decrease with the dimension. These are *empirical* CNC observations, not a universal neural-network theorem.

**Figure 2:** On a small, synthetic learning-halfspaces task, SGD and perturbed methods escape the tested saddle region faster than GD. The figure is proof of concept, not a broad benchmark proving superiority on all tasks.

**Figure 3:** A scatter plot relates curvature magnitude and measured projected stochastic-gradient variation in networks with 8, 16, and 32 hidden layers; fitted trends are positive. This supports a possible alignment between SGD noise and curvature, not exact proportionality in every network.

## 10. Contributions: keep the four claims distinct

1. Define the weaker, directional CNC assumption in place of isotropic noise.
2. Prove dimension-free-in-form iteration bounds for a GD–SGD hybrid and scheduled-step SGD (Theorems 1–2).
3. Prove CNC for a specified learning-halfspaces model (Lemma 4).
4. Provide experimental evidence that aligned gradient fluctuations appear in the tested neural networks.

## 11. Assumptions and limitations you must not omit

**Assumptions.** The objective is twice differentiable with $L$-Lipschitz gradient and $\rho$-Lipschitz Hessian; individual stochastic gradients are bounded by $\ell$; the CNC lower bound holds at relevant points; and the prescribed learning rates / thresholds are used. The proved halfspace example needs a condition on $\phi$ and bounded input support.

**Limitations.** CNC is *not* proved for general deep neural networks; this extension is supported by experiments only. The CNC-SGD algorithm has periodic large steps, not an arbitrary fixed step size. The second-order condition $\varepsilon_h = \sqrt{\rho}\,\varepsilon^{2/5}$ is weaker than certain isotropic-perturbation guarantees; *dimension-free* refers to the form of the bound. The result does not ensure global minima, fast wall-clock training, or superior test accuracy.

## 12. Survey connections: locate this among the other papers

| Paper | Difference worth stating in the survey |
|---|---|
| Lee et al. (2016) | Deterministic GD avoids convergence to strict saddles almost surely, but with no general finite escape-time bound. |
| Jin et al. (2017) | Adds *external isotropic perturbations* to GD and proves finite-time escape; its principal bound has polylogarithmic dimension dependence. |
| Daneshmand et al. (2018) | Uses *intrinsic stochastic-gradient directions* under CNC and obtains bounds without explicit dimension dependence; periodic step-size boosts remain essential to the proved SGD version. |
| Dauphin et al. (2014) | Uses Hessian curvature directly in saddle-free Newton, while Daneshmand avoids explicit Hessian computation in its updates. |

## 13. Quick reference

| Concept | Meaning / formula |
|---|---|
| Intrinsic SGD noise | $\nabla f_z(w) - \nabla f(w)$; randomness comes from sampling. |
| CNC, Eq. (6) | $\mathbb{E}[\langle v_{\min}, \nabla f_z(w)\rangle^2] > \gamma$. |
| Isotropic benchmark | Expected squared projection onto one direction: $r^2/d$. |
| Second-order target | $\lVert\nabla f\rVert \le \varepsilon$, $\lambda_{\min}(H) \ge -\sqrt{\rho}\,\varepsilon^{2/5}$. |
| Algorithm 1 | GD steps; occasional sampled-gradient kick; $\widetilde{O}(\varepsilon^{-2})$. |
| Algorithm 2 | SGD with periodic large steps; $\widetilde{O}(\varepsilon^{-4})$. |
| Halfspace result | Negative-$\lambda$ eigenvector receives squared projection $\ge (\lambda/c)^2$. |

## 14. What to read in the original paper

**Must read:** p. 2 Assumption 1 and contributions; pp. 3–6 Algorithms 1–2 and Theorems 1–2; p. 7 Lemma 4 and Figure 1; p. 8 Figures 2–3 and conclusion. **Skim:** proof details on pp. 4–5 and the related-work table. The supplemental appendix is useful only for verifying a particular proof.

:::plain Papers Master — 1–3-line note
Daneshmand et al. (2018) show that directional SGD noise can replace external isotropic perturbations under CNC. Their GD–SGD hybrid and scheduled-step SGD attain approximate second-order stationarity with no explicit dimension factor; CNC is proved for a halfspace model and supported empirically, but not proved for general deep networks.
:::

> **Source:** Daneshmand, Kohler, Lucchi & Hofmann, ICML 2018, original pp. 1–8. [Full paper (PMLR)](https://proceedings.mlr.press/v80/daneshmand18a/daneshmand18a.pdf). Teaching examples are original illustrations, not results attributed to the authors.
