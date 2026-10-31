# SGD Can Escape Faster When We Analyze Whole Local Runs

:::minute The paper in one minute
**Question:** Can the usual stochastic-gradient update reach approximate second-order stationarity in fewer than the classical $\varepsilon^{-4}$ oracle calls?

**Model/Method:** Constant-step SGD with a ball-controlled restart/stopping test, averaged output and dispersive stochastic noise.

**Finding:** Theorem 5 gives $\widetilde{O}(\varepsilon^{-7/2})$ stochastic-gradient complexity under smoothness, a finite objective gap and noise assumptions.

**Why we care:** The improvement comes from analyzing a whole neighborhood visit, not demanding descent at every noisy step. The stopping rule and noise shape are part of the result.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Complexity counts stochastic-gradient calls, and step-by-step analysis gives $\varepsilon^{-4}$

An **oracle-complexity** bound counts how many stochastic gradients $\nabla F(x; \zeta)$ a method needs to reach accuracy $\varepsilon$. Exponents matter: for $\varepsilon = 0.01$, $\varepsilon^{-4} = 10^8$ and $\varepsilon^{-7/2} = 10^7$, ten times fewer.

The classical $\varepsilon^{-4}$ comes from asking every single step to make progress. With an $L$-Lipschitz gradient and noise of size $\sigma$, one SGD step gives

$$\mathbb{E}\, f(x^+) \le f(x) - \eta\Big(1 - \tfrac{L\eta}{2}\Big)\|\nabla f(x)\|^2 + \tfrac{L\eta^2}{2}\,\sigma^2 .$$

The last term is the price of noise. While $\|\nabla f\| \approx \varepsilon$, the gain beats it only if $\eta \lesssim \varepsilon^2/(L\sigma^2)$, and then each step gains only about $\varepsilon^4$. A finite gap $\Delta$ allows about $\Delta/\varepsilon^4$ such steps.

### 2. Sums of noise grow like $\sqrt{K}$; averages shrink like $1/\sqrt{K}$

Add $K$ independent mean-zero noise vectors of size about $\sigma$. Their directions mostly cancel: the sum has size about $\sigma\sqrt{K}$, and the average about $\sigma/\sqrt{K}$. With $\sigma = 1$ and $K = 10^4$, the average has size about $0.01$.

Two consequences fit the paper's scales. Averaging $K_0 \approx \varepsilon^{-2}$ iterates leaves noise of order $\sigma\varepsilon$, small enough for an $\varepsilon$-accurate gradient. And noise alone moves an SGD iterate with step $\eta \approx \varepsilon^{3/2}$ by about $\eta\sigma\sqrt{K_0} \approx \sigma\varepsilon^{1/2}$ in $K_0$ steps: the same order as the ball radius $B \approx \varepsilon^{1/2}$. (This is a scale check, not the paper's proof.)

### 3. A Lipschitz Hessian makes the quadratic model accurate inside a small ball

If the Hessian is $\rho$-Lipschitz, the quadratic Taylor model around an anchor $x_0$ has a gradient error of at most $\tfrac{\rho}{2}\|x - x_0\|^2$. Inside a ball of radius $B$ that is at most $\rho B^2/2$. For $f(x) = \cos x$ (here $\rho = 1$) around $x_0 = 0$, the model's gradient at $x = 0.1$ is $-0.1$, the true gradient is $-\sin 0.1 \approx -0.09983$: an error of $1.7 \times 10^{-4}$, below the bound $0.005$. This is why a small ball lets the analysis treat the landscape as a quadratic.

:::key Putting the three ideas together
Demanding descent at every noisy step costs $\varepsilon^{-4}$ (idea 1). The paper instead looks at a whole run of $K_0$ steps inside a ball of radius $B$, where the quadratic model is accurate (idea 3) and noise sums and averages behave predictably (idea 2). Either the run leaves the ball, which buys a guaranteed decrease, or it stays, and its average is a certified approximate second-order stationary point. That trade gives $\widetilde{O}(\varepsilon^{-7/2})$, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. A small gradient is only the first test

For $f(x) = \mathbb{E}[F(x; \zeta)]$, an unbiased stochastic gradient is $\nabla F(x; \zeta)$. At a saddle, the average gradient can vanish although moving along a negative Hessian eigenvector lowers $f$. The paper seeks both a small gradient and no substantially negative curvature. Its accuracy target is second-order stationarity; a degenerate saddle can still satisfy such tests.

$$\|\nabla f(x)\| \le \varepsilon, \qquad \lambda_{\min}(\nabla^2 f(x)) \ge -\sqrt{\rho\varepsilon}.$$

Here $\rho$ bounds how quickly the Hessian changes with position. The gradient tolerance and curvature tolerance have different units and scales. Converting this output into proximity to an actual local minimum needs extra landscape structure, such as an appropriate strict-saddle condition.

## 2. The assumptions do substantive work

Assumption 1 requires every stochastic component gradient to be $L$-Lipschitz, and the expected objective Hessian to be $\rho$-Lipschitz. Component smoothness controls the difference between two trajectories using the same sampled data. Assumption 2 gives a finite gap $\Delta = f(x_0) - \inf f$. The bounded-noise version of Assumption 3 requires $\|\nabla F(x; \zeta) - \nabla f(x)\| \le \sigma$ almost surely.

Assumption 4 is stronger than positive variance. For every unit direction $v$, any set too narrow along $v$ must receive probability at most $1/4$ from the stochastic gradient. The specified width is $\sigma/(4\sqrt{d})$. This prevents the first step from concentrating inside the thin region from which saddle escape is unlikely. Uniform ball noise is one proved example; arbitrary finite-sum sampling is not automatically dispersive.

## 3. Worked example: why a thin trapped region cannot hold two paths

*Pedagogical illustration, constructed for these notes.*

Use the local quadratic $q(u, v) = (-u^2 + v^2)/2$. Its gradient is $(-u, v)$, and its Hessian is $\mathrm{diag}(-1, 1)$. The $u$-direction is unstable for descent; the $v$-direction is stable. This local model is not lower bounded globally, so it illustrates the escape mechanism rather than satisfying the complete global theorem.

*An SGD step with $\eta = 0.1$ and additive noise $\xi = (\xi_u, \xi_v)$ is $u^+ = 1.1u - 0.1\xi_u$ and $v^+ = 0.9v - 0.1\xi_v$.*

Start two paths at $(0.001, 0.002)$ and $(-0.001, 0.002)$, both inside the ball of radius $B = 0.01$ about zero. Couple them by using identical future noises. Subtract their updates: the additive noises cancel exactly.

$$\Delta u_k = 0.002\,(1.1)^k, \qquad \Delta v_k = 0.$$

At $k = 25$, their distance is $0.0216694$, exceeding the ball diameter $2B = 0.02$. Both paths cannot still be inside the ball. The first integer with this property is $\lceil \log 10 / \log 1.1 \rceil = 25$. This conclusion concerns the coupled paths, not a guarantee that each individual path escapes by step 25.

Appendix C extends this argument beyond exact quadratics. Hessian Lipschitzness bounds nonlinear rotation; component-gradient smoothness bounds the coupled noise difference. After rescaling by the unstable expansion, concentration keeps a significant projection along the negative-curvature direction. Thus the bad starting region is narrow. Dispersive noise makes entering that region unlikely.

:::key Why covariance alone is insufficient
The proof needs an anti-concentration property of an entire narrow set. A distribution can have a nonzero second moment in a direction while placing most of its probability near a small collection of points. That is a different requirement from the paper's dispersiveness condition.
:::

## 4. The actual algorithm and its output

Algorithm 2 fixes an anchor $x_0$ and performs ordinary SGD steps. If an iterate travels farther than $B$ from the current anchor, that iterate becomes the next anchor and the local counter restarts. If $K_0$ consecutive steps remain in the ball, the algorithm stops and returns the arithmetic mean of the preceding $K_0$ iterates. It neither returns an arbitrary last iterate nor checks the true objective after every step.

## 5. Three statements explain the improved rate

Proposition 7: sufficiently negative anchor curvature forces a ball exit with high probability. Proposition 9: any exit within $K_0$ steps decreases the objective by at least $B^2/(7\eta K_0)$, on the stated event. Proposition 10: a full non-exiting run has an averaged output with small gradient and controlled negative curvature. Appendix D splits the local quadratic into positive and nonpositive spectral subspaces; this avoids charging the same noise penalty to both.

$$B = \widetilde{O}(\varepsilon^{1/2}), \qquad \eta = \widetilde{O}(\varepsilon^{3/2}), \qquad K_0 = \widetilde{O}(\varepsilon^{-2}).$$

Ignoring logarithms and fixed problem parameters, each exit lowers $f$ by order $\varepsilon^{3/2}$. A finite objective gap permits order $\varepsilon^{-3/2}$ exits, each costing at most order $\varepsilon^{-2}$ gradients. Multiplying gives $\varepsilon^{-7/2}$. Halving $\varepsilon$ multiplies this leading work by $2^{7/2} \approx 11.31$; this is an accuracy exponent, not a measured runtime ratio.

## 6. Why averaging certifies the gradient

Summing $x_{k+1} - x_k = -\eta(\nabla f(x_k) + \xi_{k+1})$ makes the updates telescope. The average gradient is bounded by endpoint displacement divided by $\eta K_0$, plus average noise. For a quadratic, its gradient at the average equals the average gradient exactly. Hessian Lipschitzness bounds the nonlinear discrepancy by order $\rho B^2$, which is order $\varepsilon$. This is why the specified average appears in the theorem.

Theorem 5 states $\|\nabla f(\bar{x})\| \le 18\rho B^2$ and $\lambda_{\min}(\nabla^2 f(\bar{x})) \ge -17\delta$, with $\delta = \sqrt{\rho\varepsilon}$, and success at least $1 - (T_1 + 1)p$. To obtain total failure $\alpha$, budget $p \le \alpha/(T_1 + 1)$; $p$ is not itself the final failure probability. Constant rescaling yields the customary accuracy notation.

## 7. Evidence and limits

This is a theoretical paper. Table 1 compares published oracle-complexity results, not experimental learning curves. There is no new neural-network benchmark establishing that this stopping rule improves practical training time. The result counts stochastic-gradient evaluations with logarithmic dimension dependence and fixed $L$, $\rho$, $\sigma$ and objective gap; it is not a global-optimum or generalization guarantee.

## 8. Read the assumptions and printed formulas carefully

The leading rate depends on Hessian regularity and dispersive noise. It is not a replacement for the classical first-order bound under only gradient smoothness. Algorithm 3 schedules artificial noise if intrinsic dispersiveness is absent; that changes the method. A literal implementation should not copy the paper without reconciliation: the printed parameter expression uses $\log(0.7)$, while the positive repetition count in Appendix C uses $\log(0.7^{-1})$. Appendix E also prints $\nabla f$ where Hessian eigenvalues require $\nabla^2 f$. These notes use the positive amplification argument, the correct Hessian object and the stated asymptotic scales.

## 9. Survey connection

Compared with Ge2015, this analysis obtains a sharper second-order stationary-point complexity under a different noise condition and output rule. Compared with Tripuraneni2018 in this batch, its leading $\varepsilon^{-7/2}$ exponent uses stochastic gradients rather than sampled Hessian-vector products. Equal exponents do not make the assumptions or operations interchangeable. Heavy-tail stationary-law analysis addresses a separate long-run question.

| Symbol | Meaning |
|---|---|
| $B$ | Neighborhood radius; bounds Taylor error. |
| $K_0$ | Length of a non-exiting local run. |
| Dispersive noise | Avoids placing too much mass in any narrow trapped set. |
| $\widetilde{O}(\varepsilon^{-7/2})$ | Gradient-oracle bound for the specified averaged output. |

:::plain Papers Master - Our notes (copy)
Ball-controlled SGD with averaged output reaches approximate second-order stationarity in $\widetilde{O}(\varepsilon^{-7/2})$ gradient calls under component smoothness, Hessian regularity, bounded/dispersive noise and a finite gap. The stopping rule, noise condition and total failure budget matter; this is not a last-iterate or global-optimum theorem.
:::

Reading recommendation: Read Algorithm 2, Assumptions 1-4 and the three-part proof. This explains what improved SGD analysis needs beyond an ordinary training loop.

> **Source:** main PDF pages 1-11; proof appendices A-F, pages 16-43. Main Theorem 5, Algorithm 2 and Propositions 7, 9, 10. Coupled-path calculation constructed for teaching. Fang, Lin & Zhang (2019), *Sharp Analysis for Nonconvex SGD Escaping from Saddle Points*, COLT 2019 (PMLR 99). [proceedings.mlr.press/v99/fang19a/fang19a.pdf](https://proceedings.mlr.press/v99/fang19a/fang19a.pdf)
