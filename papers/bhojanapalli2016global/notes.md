# When Local Search Becomes Global

:::minute The paper in one minute
**Question:** Can randomly initialized local optimization recover the correct low-rank matrix even though its factorized objective is non-convex?

**Model/Method:** Positive-semidefinite matrix sensing, $X = UU^\top$, with measurements satisfying restricted isometry (RIP). Analyze local minima, Hessian curvature, and a perturbed stochastic update.

**Finding:** Under suitable RIP, every local minimum is globally correct in the noiseless case; nonoptimal stationary points have negative curvature. A suitable randomized algorithm converges near a global solution in polynomial time. Noisy or nearly low-rank targets admit quantitative error bounds.

**Why we care:** The *geometry* of an objective can turn local search into a global-recovery method, without a specially engineered starting point.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Factorizing a matrix saves parameters but makes the loss non-convex

A symmetric PSD $n \times n$ matrix of rank $r$ can be written $X = UU^\top$ with $U \in \mathbb{R}^{n\times r}$. That uses $nr$ numbers instead of the $n(n+1)/2$ entries of a symmetric matrix: for $n = 100$, $r = 5$, it is $500$ instead of $5050$. The price is symmetry: $U$ and $UR$ give the same $X$ for every orthogonal $R$ (for $r = 1$, just $U$ and $-U$). In the notes' $2 \times 2$ example, $U = (1, 0)^\top$ and $U = (-1, 0)^\top$ both have loss $0$, but their midpoint $U = 0$ has loss $1$. A convex function could not do that, so the factorized loss is non-convex even though the measurements are linear in $X$.

### 2. Random measurements preserve sizes on average; RIP asks for it on all low-rank matrices at once

A measurement $\langle A_i, Z\rangle = \sum_{jk} (A_i)_{jk} Z_{jk}$ with independent $\mathcal{N}(0, 1)$ entries is a Gaussian with variance $\|Z\|_F^2$. So $\frac{1}{m}\sum_i \langle A_i, Z\rangle^2$ has mean exactly $\|Z\|_F^2$, and its ratio to $\|Z\|_F^2$ has standard deviation $\sqrt{2/m}$: $0.1$ for $m = 200$. RIP is the much stronger demand that this ratio stays within $[1 - \delta_s, 1 + \delta_s]$ for *every* matrix of rank at most $s$ simultaneously. Covering all low-rank matrices costs a number of measurements proportional to their degrees of freedom, which is why Gaussian sensing needs $m = O(nr)$.

### 3. Two conditions every local minimum must pass

At a local minimum of a smooth $f$, the gradient is zero (**first-order** condition) and the Hessian has no negative eigenvalue (**second-order** condition). A point that fails the second test is a **strict saddle**: it has a direction of negative curvature, and perturbed or second-order methods can find it and move away. So two facts together make local search global: every point that passes both tests is a global minimum (Theorem 3.1 / 4.1), and every wrong stationary point fails the second test (Theorem 3.2). The notes' origin example fails it with Hessian eigenvalue $-4$.

### 4. Hard in the worst case does not mean hard here

Without any condition on the measurements, this recovery problem is NP-hard: with each $A_i$ reading a single entry it becomes matrix completion, for which hardness is known. The paper's result does not contradict this. It identifies a *structured* class of instances (RIP measurements) where the landscape is benign; its Section 5 argues that outside that class no such guarantee can be expected, since an efficient method for every measurement operator is unlikely. (The explicit counterexample it gives there does not hold as printed; see the correction in the notes' §8.)

:::key Putting the four ideas together
Factorization makes the loss non-convex (idea 1). RIP guarantees that the measurements see every low-rank error (idea 2), and under it the paper shows that the only points passing the first- and second-order tests are correct, while every wrong stationary point has negative curvature (idea 3). That is why a randomized local method succeeds from a random start inside the RIP class, while no such guarantee is expected outside it (idea 4). The notes below give the theorems and the proof's structure.
:::
:::

:::divider Study notes
:::

## 1. Start with the problem, not the theorems

Suppose a large, unknown matrix $X^\star$ is positive semidefinite (PSD) and has rank at most $r$. We cannot observe every entry; instead we see measurements $y_i = \langle A_i, X^\star\rangle + w_i$. We want to reconstruct $X^\star$. Rather than optimize over all $n^2$ entries of $X$, write the estimate as $X = UU^\top$ with $U \in \mathbb{R}^{n\times r}$, and minimize

$$f(U) = \sum_{i=1}^{m}\big(\langle A_i, UU^\top\rangle - y_i\big)^2.$$

This reduces the variables from about $n^2$ to $nr$, but $UU^\top$ is quadratic in $U$. Thus the objective is non-convex even though the measurements are linear in $X$.

:::key What the paper is really asking
An unlucky initialization could land near a wrong stationary point. Can we prove that any apparent *local minimum* must reconstruct $X^\star$? If so, only saddles remain as obstacles, and a suitable saddle-escaping method can finish the job.
:::

## 2. Worked example: zero gradient need not mean a bad minimum

Use an idealized *fully observed* $2 \times 2$ PSD target (this is a teaching example, not the paper's partially observed setting):

$$X^\star = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \qquad U = \begin{bmatrix} a \\ b \end{bmatrix}, \qquad UU^\top = \begin{bmatrix} a^2 & ab \\ ab & b^2 \end{bmatrix}.$$

With squared Frobenius error, expand the loss and its gradient:

$$f(a, b) = \|UU^\top - X^\star\|_F^2 = (a^2 - 1)^2 + 2a^2b^2 + b^4, \qquad \nabla f = \begin{bmatrix} 4a(a^2 + b^2 - 1) \\ 4b(a^2 + b^2) \end{bmatrix}.$$

The origin has gradient zero yet loss $f(0, 0) = 1$. But $f(0.1, 0) = 0.9801 < 1$; moving toward the true first coordinate reduces loss. In fact $\nabla^2 f(0, 0) = \mathrm{diag}(-4, 0)$: the origin has a negative-curvature direction. By contrast $U = (\pm 1, 0)^\top$ yields $UU^\top = X^\star$ and zero loss.

## 3. What RIP means and why it matters

A measurement system should not confuse two distinct low-rank matrices. The paper formalizes that using the *Restricted Isometry Property* (Definition 2.1): for all matrices $Z$ of rank at most $s$,

$$(1 - \delta_s)\|Z\|_F^2 \;\le\; \frac{1}{m}\sum_{i=1}^{m}\langle A_i, Z\rangle^2 \;\le\; (1 + \delta_s)\|Z\|_F^2.$$

**Plain English:** the measurement vector preserves the size of any low-rank difference up to a small distortion $\delta_s$. If two candidate matrices are very different but their measurements look identical, successful recovery is impossible. RIP prevents that sort of information loss on low-rank directions.

:::key Why $4r$?
In the proof, combinations of candidate, optimal, and comparison matrices can have ranks larger than $r$. The main theorems therefore assume $(4r, \delta_{4r})$-RIP rather than merely rank-$r$ RIP. This is a structural assumption about the sensing operator, not a generic property of arbitrary training datasets.
:::

## 4. Three theorems worth remembering

**Theorem 3.1 (noisy local minima).** If $\mathrm{rank}(X^\star) \le r$, the measurement noise is Gaussian with standard deviation $\sigma_w$, and $\delta_{4r} < 1/10$, then every local minimum obeys, with high probability,

$$\|UU^\top - X^\star\|_F \le 20\sqrt{\frac{\log n}{m}}\;\sigma_w.$$

In particular, $\sigma_w = 0$ forces $UU^\top = X^\star$. For the *noiseless* case, Theorem 4.1 relaxes the RIP constant to $\delta_{4r} < 1/5$. Notice the precise conclusion: noisy local minima are *close to the true matrix*, not necessarily exactly globally optimal.

**Theorem 3.2 (strict-saddle geometry).** Under the noiseless conditions with $\delta_{4r} < 1/10$, every first-order critical point with $UU^\top \neq X^\star$ has a strictly negative Hessian eigenvalue. Thus a wrong stationary point is not a local minimum: an escaping direction exists.

**Theorem 3.3 (algorithmic implication).** Combining this geometry with prior saddle-escape analysis gives polynomial-iteration high-probability convergence near an optimal *factor* from random initialization. The analyzed update includes *explicit random perturbation* and projection to a bounded ball; it is not a blanket guarantee for arbitrary vanilla SGD.

:::key Key implication
Theorem 3.1 says *where the local minima are*; Theorem 3.2 says *what happens at the other stationary points*; Theorem 3.3 uses both to justify a *specific randomized optimization method*. These are three different levels of claim.
:::

## 5. Why a good initialization is not needed here

Earlier analyses often used an SVD-based first step to place the parameters near the answer, followed by local gradient descent. Here, once the *whole relevant factorized landscape* is shown to have no bad local minima and strict nonoptimal saddles, the optimizer does not need to start near the correct factor. The randomness helps it avoid unfavorable stationary points, while the sensing geometry ensures there is no bad valley to settle into.

For Gaussian sensing, the paper highlights a measurement requirement $m = O(nr)$ under its assumptions: this matches the order of the unknown low-rank degrees of freedom, up to constants. It is *not* a universal sample-complexity claim for every measurement design.

## 6. Proof intuition: first-order + second-order conditions

The proof of the noiseless result is a contradiction: assume a point $U$ is a local minimum, yet $UU^\top \neq X^\star$.

1. **Align factors first.** Because $UR(UR)^\top = UU^\top$ for any orthogonal $R$, different factors can represent the same matrix. Choose an appropriate rotation $R$ before comparing $U$ with the true factor $U^\star$.
2. **Use the zero-gradient condition.** Lemma 4.2 shows that a stationary point's reconstruction error is tightly controlled *within the column space of $U$*. But that alone does not rule out missed directions orthogonal to this space.
3. **Use nonnegative Hessian curvature at a local minimum.** Corollary 4.1 and Lemma 4.4 constrain precisely the errors that the first-order argument could not control, using the aligned difference between $U$ and $U^\star$.
4. **Combine both and invoke RIP.** With $\delta_{4r} < 1/5$, the resulting inequalities cannot hold for a nonzero reconstruction error. Hence $UU^\top = X^\star$.

:::key Don't memorize the constants in the lemmas
Remember why gradient information alone is insufficient, why the proof needs Hessian information, and why factor rotation matters. Those three ideas explain the mathematics without reproducing pages of algebra.
:::

## 7. What Figure 1 actually tells us

The authors compare *random initialization* against *SVD initialization* across ranks and numbers of Gaussian measurements ($n = 100$). White regions mean successful recovery; dark regions indicate failures under the numerical stopping criterion. The success regions are visually very similar, with a reported empirical transition near $m \approx 2nr$.

This corroborates the claim that elaborate initialization may not be necessary *in the tested regime*; it is not itself a proof of global convergence, nor does it show that two initializations always behave identically.

## 8. An essential counterexample: RIP is not decorative

Section 5 constructs measurement operators outside the favorable RIP setting for which, the paper states, a rank-constrained recovery problem *does have* a nonglobal local minimum. One example has a rank-one true matrix $X^\star = \mathrm{diag}(1, 0)$ with zero objective, and the paper states that $X = \mathrm{diag}(0, 1)$ is a suboptimal rank-one local minimum with loss 1. **Correction (our check):** as printed, $\mathrm{diag}(0, 1)$ is not a local minimum: the rank-one PSD matrices $\begin{bmatrix} s & \sqrt{s(1-s)} \\ \sqrt{s(1-s)} & 1-s \end{bmatrix}$ tend to it as $s \to 0$ and have loss $1 - s^2 < 1$ (the paper's second example fails in the same way). The section's other argument stands: without RIP the problem is NP-hard.

**Interpretation:** When measurements do not preserve low-rank directions well, factorization can have misleading valleys. More parameters or a non-convex objective by themselves neither guarantee success nor imply failure. The sensing assumptions determine the landscape.

## 9. Contributions vs. scope

**Contribution:** noiseless all-local-minima-are-global; noisy recovery-error bound; negative-curvature guarantee at nonoptimal critical points; a randomized global-convergence consequence; experiments contrasting random and SVD initialization.

**Not proved:** convergence of ordinary unperturbed SGD under arbitrary hyperparameters; global minima for general neural networks; robustness to an arbitrary measurement operator; exact matrix recovery in the presence of noise.

## 10. Important assumptions and limitations

| Item | What it means for our survey |
|---|---|
| PSD low-rank target | Main theory studies $X^\star \succeq 0$ with rank at most $r$; it is not arbitrary rectangular matrix factorization. |
| RIP / measurements | $(4r, \delta_{4r})$-RIP is essential to the positive geometry guarantees; the authors demonstrate bad local minima may exist otherwise. |
| Noise assumptions | Theorem 3.1 uses independent Gaussian measurement noise; its error bound is probabilistic. |
| Optimization method | The convergence result is for a specific projected, randomized perturbation update, with suitable algorithm parameters. |
| Neural network transfer | Matrix factorization resembles a very small linear network, but there is no theorem here for generic deep nonlinear architectures. |

## 11. Cross-paper connections: what changes?

| Paper | Relation to Bhojanapalli et al. (2016) |
|---|---|
| Lee (2016) | General smooth strict-saddle avoidance is asymptotic; here the matrix-sensing geometry lets saddle escape support global recovery. |
| Jin (2017) | Perturbed GD finds approximate second-order stationary points in general smooth objectives; under benign geometry these can lie near global solutions. Different perturbation theorem. |
| Ge, Jin & Zheng (2017) | Extends favorable landscape analysis across sensing, completion and robust PCA; compare assumptions and geometry rather than treating the two papers as identical. |
| Safran & Shamir (2018) | Proves actual bad local minima in certain shallow ReLU networks: favorable matrix-sensing geometry does not transfer automatically. |
| Du / Allen-Zhu (2019) | Global training-loss convergence under strong overparameterization assumptions; distinct structures produce distinct reasons non-convex training succeeds. |

## 12. Quick reference

| Concept | One-line reminder |
|---|---|
| Factorized sensing | $X = UU^\top$, $f(U) = \sum_i(\langle A_i, UU^\top\rangle - y_i)^2$. |
| RIP | A approximately preserves Frobenius lengths of low-rank matrices. |
| No bad local minima | In noiseless RIP regime, every local minimum satisfies $UU^\top = X^\star$. |
| Strict nonoptimal saddle | A stationary point with $UU^\top \neq X^\star$ has some negative Hessian curvature. |
| Global convergence | Requires algorithmic noise/perturbation and bounded, appropriately chosen updates. |

:::plain Papers Master note (copy)
Bhojanapalli et al. prove that PSD matrix sensing under suitable RIP has no spurious local minima (noiseless), with recovery-error bounds under Gaussian noise and negative curvature at incorrect stationary points. Together with a specific randomized projected update, this yields polynomial-time convergence from random initialization. Results do not generalize to arbitrary measurements or deep neural networks.
:::

> **Source:** Bhojanapalli, Neyshabur & Srebro (2016), NeurIPS 29, Sections 2–6, Theorems 3.1–3.4 and 4.1. [https://papers.nips.cc/paper_files/paper/2016/file/b139e104214a08ae3f2ebcce149cdf6e-Paper.pdf](https://papers.nips.cc/paper_files/paper/2016/file/b139e104214a08ae3f2ebcce149cdf6e-Paper.pdf). Worked $2 \times 2$ example is pedagogical, not a theorem from the paper.
