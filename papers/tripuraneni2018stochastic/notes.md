# Cubic Regularization When Both Derivatives Are Noisy

:::minute The paper in one minute
**Question:** Can cubic regularization certify approximate second-order stationarity using only stochastic gradients and stochastic Hessian-vector products?

**Model/Method:** Build sampled cubic models, solve them approximately with perturbed gradient descent, and refine the final model more accurately when its predicted decrease is small.

**Finding:** Theorem 1 and Corollary 1 give leading total oracle complexity $\widetilde{O}(\varepsilon^{-7/2})$ under smoothness, concentration and subsolver conditions.

**Why we care:** The method controls how derivative errors affect model progress and stopping. It does not require the exact objective to be evaluated for each acceptance decision.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. With exact derivatives, the cubic model is an upper bound

If the Hessian is $\rho$-Lipschitz, Taylor's theorem bounds the true change by the cubic model:

$$f(x + s) - f(x) \le \nabla f(x)^\top s + \tfrac12 s^\top \nabla^2 f(x)\, s + \tfrac{\rho}{6}\|s\|^3 .$$

So minimizing the model can only promise too little, never too much: whatever decrease the exact model predicts, $f$ achieves at least that. For $f(s) = \cos s$ at $s = 0$ with $\rho = 1$, the model $-s^2/2 + |s|^3/6$ is minimized at $s = \pm 2$ with value $-2/3$, and the true change there is $\cos 2 - 1 \approx -1.416$, below the prediction as promised. This is the classical (exact) cubic regularization of Nesterov and Polyak; the paper asks what survives when $\nabla f$ and $\nabla^2 f$ are only estimated.

### 2. A Hessian-vector product costs about as much as a gradient

The model's gradient $g + Bs + \tfrac{\rho}{2}\|s\|s$ needs only the product $Bs$, never the matrix $B$. A **Hessian-vector product** can be computed from gradients alone, for example $\nabla^2 f(x)\, v \approx \big(\nabla f(x + hv) - \nabla f(x)\big)/h$ for small $h$, or exactly by automatic differentiation at a cost of a small multiple of one gradient. For $f(x, y) = x^2 y + y^3$ at $(1, 2)$ and $v = (1, 0)$, the exact product is $(4, 2)$, and the difference quotient with $h = 10^{-6}$ matches it to six digits. A $d \times d$ Hessian is never formed.

### 3. Averaging $n$ samples shrinks the error like $1/\sqrt{n}$

An average of $n$ independent estimates with standard deviation $\sigma$ has error about $\sigma/\sqrt{n}$. To get a gradient error of order $\varepsilon$ takes $n_1 \approx \sigma_1^2/\varepsilon^2$ samples; a Hessian error of order $\sqrt{\rho\varepsilon}$ takes only $n_2 \approx \sigma_2^2/(\rho\varepsilon)$. With $\sigma_1 = \sigma_2 = \rho = 1$ and $\varepsilon = 0.01$ that is $10^4$ gradient samples but $100$ Hessian samples per model. The curvature only has to be accurate to the looser scale $\sqrt{\rho\varepsilon}$, which is why Hessian batches can be much smaller.

:::key Putting the three ideas together
Exact cubic models give guaranteed progress (idea 1). The paper builds them from sampled gradients and sampled Hessian-vector products (idea 2), with batch sizes chosen so that both errors are small at the right scales (idea 3), and it proves that the sampled models still make enough progress and certify the final point. Counting the batches gives the $\widetilde{O}(\varepsilon^{-7/2})$ total in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Why an indefinite quadratic needs a stabilizer

At $x$, a second-order Taylor model uses a linear gradient term and a quadratic Hessian term. If the Hessian has a negative eigenvalue, the quadratic model can decrease without bound along that direction. Adding a positive norm-cubed term makes large moves expensive. This is a regularized second-order Taylor model, not a claim that $\|s\|^3$ is the true third-order Taylor derivative.

$$m(s) = g^\top s + \tfrac12 s^\top B s + \tfrac{\rho}{6}\|s\|^3.$$

Here $g$ estimates $\nabla f(x)$, and $B$ estimates $\nabla^2 f(x)$. Access to $B$ is through products $Bv$, computed from a fixed Hessian minibatch for the current model. No dense Hessian is required. The model gradient is $g + Bs + (\rho/2)\|s\|s$. The cubic coefficient convention matters: it is $\rho/6$ here, whereas Carmon-Duchi's subproblem paper uses $\rho/3$.

## 2. Conditions and the target

The expected $f$ is lower bounded, has an $\ell$-Lipschitz gradient and a $\rho$-Lipschitz Hessian. Assumption 2 controls stochastic gradient and Hessian errors by variance bounds $\sigma_1^2, \sigma_2^2$ and almost-sure bounds $M_1, M_2$. Independent minibatches estimate the two derivatives. The target is $\|\nabla f\| \le \varepsilon$ and $\lambda_{\min}(\nabla^2 f) \ge -\sqrt{\rho\varepsilon}$. Despite the source's approximate-local-minimum terminology, these inequalities alone do not exclude every degenerate saddle.

:::key Two approximation errors
Sampling approximates the true derivatives. The inner solver then only approximately minimizes that sampled model. A theorem must control both; a small inner gradient alone is not enough to justify the complete outer algorithm.
:::

## 3. Worked example: what noisy curvature does to a cubic step

*Pedagogical illustration, constructed for these notes.*

Use $f(s) = \cos s$ at $x = 0$. Then $f'(0) = 0$ and $f''(0) = -1$, while global gradient and Hessian Lipschitz constants can both be chosen as one. The exact centered cubic with $\rho = 1$ is $q(s) = -s^2/2 + |s|^3/6$. Its nonzero minimizers are $s = \pm 2$, each with $q = -2/3$.

Now suppose independent derivative estimates give $g = 0.01$ and $B = -0.9$. The sampled model is $m(s) = 0.01s - 0.45s^2 + |s|^3/6$. These values can arise from bounded unbiased component perturbations $F(s; \zeta) = (1 + \zeta_h)\cos s + \zeta_g \sin s$, with symmetric $\zeta_h = \pm 0.1$ and $\zeta_g = \pm 0.01$.

The positive $g$ selects the negative direction. Put $s = -r$, $r \ge 0$. Setting the derivative of $-0.01r - 0.45r^2 + r^3/6$ to zero gives $r^2/2 - 0.9r - 0.01 = 0$.

$$r = 0.9 + \sqrt{0.83} \approx 1.811043; \qquad m(-r) \approx -0.504055.$$

The exact cubic at this point is $q(-r) \approx -0.649939$, while the actual loss change is $\cos(r) - 1 \approx -1.237943$. Thus this particular sampled step makes real progress; a random sampled model is not automatically an upper bound on the true loss.

The model-error bound is $|m(s) - q(s)| \le |g|r + |B + 1|r^2/2 = 0.01r + 0.05r^2 \approx 0.182104$. Its exact difference here is about $0.145883$, safely below the bound.

Near termination the characteristic step length is $\sqrt{\varepsilon/\rho}$. Gradient error of order $\varepsilon$ times this length and Hessian error of order $\sqrt{\rho\varepsilon}$ times its squared length are both order $\varepsilon^{3/2}/\sqrt{\rho}$. That is the same scale as useful cubic decrease, so the hidden accuracy constants must be small. Fixed arbitrary derivative errors cannot support arbitrarily precise certification.

## 4. The outer method uses predicted progress

At each outer iteration, Algorithm 1 draws fresh gradient and Hessian minibatches, fixes the sampled model, and asks Cubic-Subsolver for a step and its model change $\Delta m$. Large predicted decreases imply true objective decreases on the derivative-concentration event and under Condition 1. When $\Delta m$ is too small, Cubic-Finalsolver refines the model before returning a point. Simply returning the last inexact step can fail the desired stationarity test.

Condition 1 allows either sufficiently large model and true-function decrease, or a controlled step length plus sufficiently accurate model minimization in the relevant regime. The condition is an analytical contract; it does not require the implementation to query the unknown expected objective. The proposed subsolver supplies the contract using gradients of the sampled cubic.

## 5. Why the total cost is $\varepsilon^{-7/2}$

Lemma 2 in the supplement bounds the sampled gradient error by $c_1\varepsilon$ and the Hessian operator error by $c_2\sqrt{\rho\varepsilon}$. Matrix concentration is uniform over $v$, so the same Hessian batch can be reused at adaptively chosen inner directions. In the small-$\varepsilon$ variance-dominated regime, the leading counts are:

| Work per scale | Leading dependence |
|---|---|
| Outer model iterations | $\widetilde{O}(\sqrt{\rho}\,\Delta_f\,\varepsilon^{-3/2})$. |
| Gradient minibatch $n_1$ | $\widetilde{O}(\sigma_1^2\varepsilon^{-2})$. |
| Hessian minibatch $n_2$ | $\widetilde{O}(\sigma_2^2/(\rho\varepsilon))$. |
| Inner products per model $T(\varepsilon)$ | $\widetilde{O}(\ell/\sqrt{\rho\varepsilon})$. |

Each gradient batch is computed once per model; each $Bv$ query costs $n_2$ component products. Thus total work is outer count times $[n_1 + n_2T(\varepsilon)]$. With fixed parameters, the gradient term scales as $\varepsilon^{-7/2}$, and the main inner-product term as $\varepsilon^{-3}$. The exact theorem also keeps $M_1/\varepsilon$ and $M_2/\sqrt{\rho\varepsilon}$ terms when the variance-dominated simplification does not apply. Final refinement is accounted for and does not change the leading total exponent.

For a requested whole-run failure $\delta$, Appendix B.5 divides the probability budget among concentration and subsolver events over all outer iterations. A per-batch failure probability cannot simply be declared to be the whole-run failure probability.

## 6. Solving the sampled model

Algorithm 2 takes an analytic Cauchy step along $-g$ when the gradient is large. Otherwise it perturbs $g$ slightly and runs gradient descent on the cubic. The perturbation addresses a hard case: at $g = 0$, a zero-initialized deterministic inner gradient method stays at zero even when $B$ has negative curvature. This is randomness in the inner model, distinct from data-sampling noise. Small predicted decrease triggers Algorithm 3's more accurate final solve.

## 7. What the experiments support

The synthetic problem combines a one-dimensional W-shaped piecewise cubic with $10x_2^2$ and has saddle Hessian eigenvalues $-0.2$ and $20$ at zero. Figure 2 compares progress by oracle calls. The MNIST experiment uses a softplus autoencoder with encoder $784 \to 512 \to 256 \to 128 \to 32$ and symmetric decoder, final sigmoid and reconstruction squared loss. Figure 3 compares cubic, SGD, AdaGrad and SVRG+Oja.

Supplement D reports ten inner iterations in experiments, gradient batch 100 and Hessian batch 10 for the autoencoder, and a 55,000/5,000 training-validation split. Hyperparameters are selected under a two-million-oracle-call budget; the full batch-size grid was not searched for every method. The curves show favorable tested performance. Plateaus alone do not independently certify that every displayed point is a saddle, and oracle-call comparisons are not equal wall-clock comparisons.

## 8. Limits and survey connections

The guarantees require calibrated derivative accuracy and a suitable inner solver. The finite experimental inner count is not a blanket certificate that Condition 1 holds on every model. The output is approximate second-order stationarity, not a universal global minimum. Unlike Kohler2017's acceptance test using exact objective values, this fully stochastic construction reasons from model decrease. Compared with Fang2019, its leading exponent uses a stronger curvature oracle under different noise assumptions.

## 9. Quick reference

| Item | Meaning |
|---|---|
| $\rho/6$ | Cubic coefficient; model gradient uses $\rho/2$. |
| $n_1 + n_2T$ | One gradient batch, repeated sampled Hessian products. |
| Finalsolver | Needed to certify the returned inexact point. |

:::plain Papers Master - Our notes (copy)
Fully stochastic cubic regularization attains approximate second-order stationarity with leading total $\widetilde{O}(\varepsilon^{-7/2})$ gradient/Hessian-vector calls. Requires $\varepsilon$-scale gradient accuracy, $\sqrt{\rho\varepsilon}$-scale Hessian accuracy, an appropriate inner solver and final refinement. Model progress replaces exact objective acceptance; experiments do not certify universal global optimality.
:::

Reading recommendation: Read the derivative-accuracy scales, meta-algorithm and final refinement. This is the fully stochastic bridge from cubic models to finite-time second-order stationarity.

> **Source:** main PDF 1-9, Theorem 1/Corollary 1 and Algorithms 1-3; all publisher supplement 1-15 (printed 11-25), including proof and experimental protocols. Numerical cosine example constructed for teaching. Tripuraneni, Stern, Jin, Regier & Jordan (2018), *Stochastic Cubic Regularization for Fast Nonconvex Optimization*, NeurIPS 2018. [papers.nips.cc (paper)](https://papers.nips.cc/paper_files/paper/2018/file/db1915052d15f7815c8b88e879465a1e-Paper.pdf)
