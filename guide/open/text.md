## 1. From gaps to questions

This chapter follows Sections VI and VII of the survey. The comparison of chapter 14 pointed to specific gaps between what is proved and what is observed. The survey turns each gap into an open problem: what the evidence says now, and a concrete direction for future work.

## 2. A. Degenerate and non-smooth saddles

The efficient-escape theory of chapters 6 and 7 assumes strict saddles and Lipschitz Hessians. Neural networks violate both. Deep linear networks have saddles without negative curvature (Kawaguchi 2016). Overparameterized students converge slowly in degenerate valleys: Xu & Du (2023) show that a student wider than its single-neuron teacher slows GD from an exponential rate to a polynomial one. And ReLU networks are not twice differentiable at all.

A general theory of degenerate saddles cannot exist: certifying fourth-order local optimality is NP-hard (Anandkumar & Ge 2016). The useful question is narrower. **Do the degenerate saddles that arise from the structure of neural networks admit efficient escape?** Those come from permutation and rescaling symmetries, zero initialization, and neuron duplication. Cheridito et al.'s results for shallow ReLU networks (chapter 6) and the invariant-manifold picture of saddle-to-saddle dynamics (chapter 8) suggest that structured degeneracy may be tractable where generic degeneracy is not.

:::viz race
:::

:::read How to read this visual
**What is drawn.** Two surfaces side by side, the race of chapter 6. Left: a one-hidden-layer scalar network, whose origin is a strict saddle (red direction). Right: two hidden layers, whose origin is a degenerate saddle (amber, flat directions only).

**What moves.** *Restart race* starts both balls at the same small distance from their saddle, with the same step size.

**What to notice.** The strict saddle lets go quickly, because its multiplier is above one. The degenerate one has a multiplier of exactly one in every direction, so the ball creeps, pushed only by higher-order terms. No escape-time theorem of chapter 7 covers the right-hand ball. That is open problem A in one picture.
:::

## 3. B. A theory of stochastic and adaptive large-step training

The optimizers that train modern models combine everything the theory handles separately: Adam variants with warmup and decay, large learning rates, momentum and minibatches. For exactly these, the large-step theory is thinnest. Central flows (chapter 12) describe deterministic GD and RMSProp without momentum. The adaptive edge of stability and the edge of stochastic stability (chapter 13) are established mainly empirically.

A natural next step is a **central flow for SGD**, in which the minibatch quantity of Eq. (14), the batch sharpness, rather than the full-batch sharpness is held at its threshold. Another is a central flow for **Adam**, in which the preconditioned sharpness is the state variable. Cohen et al. point out a real obstacle: because of the randomness, it is not clear that such a deterministic description exists for stochastic optimizers. The geometry of non-Euclidean optimizers such as Shampoo and Muon adds a further dimension. For them, stability thresholds and saddle escape should be measured in the norm the optimizer uses, not in the Euclidean norm.

## 4. C. Does real SGD noise satisfy the escape conditions?

Plain SGD provably escapes saddles under the CNC or dispersive-noise conditions (chapter 7). These conditions are plausible, but they have been verified only on limited models and training stages. Meanwhile, SGD noise can be heavy-tailed, and its covariance changes as the sharpness evolves: Jastrzębski et al.'s break-even point (chapter 9) shows that the early step size shapes the noise for the rest of training. Systematic measurements across architectures, batch sizes and training phases, of the noise variance along the bottom Hessian eigenvectors, would settle whether the noise-based escape guarantees describe practice. A theory for heavy-tailed noise near saddles is also largely missing.

:::viz alpha
:::

:::read How to read this visual
**What is drawn.** For streaming linear regression trained by SGD, the tail index $\alpha$ of the iterates (vertical) against the step size $\eta$ (horizontal), one curve per batch size. A smaller $\alpha$ means a heavier tail. $\alpha = 2$ is the boundary between finite and infinite variance. The dots mark $\eta_{\rm crit} = 2b/(\sigma^2(d + b + 1))$, where $\alpha$ crosses 2 exactly (the paper's Proposition 5). Each curve stops where no stationary distribution exists. The values are an offline Monte Carlo computation.

**What to try.** Change the dimension $d$.

**What to notice.** $\alpha$ falls as the step grows and rises with the batch size (the paper's Theorem 4). Past the dot, the iterates have infinite variance even though the data are Gaussian. The bounded-variance assumption of most SGD analyses fails in the simplest model, at ordinary step sizes. How such tails interact with saddles is open problem C.
:::

## 5. D. Saddles as features: escape speed against incremental learning

Saddle-to-saddle dynamics (chapter 8) show that the plateaus in which GD lingers can be **stages of learning**. Each escape adds one unit of complexity: a singular value, a neuron, a kink, a head. This view conflicts with the goal of the escape literature, which is to leave every saddle as quickly as possible. **Is there a trade-off between escape speed and the simplicity bias that incremental learning provides?** A precise answer would require coupling the escape-time formula (chapter 5) with a measure of the complexity of the function learned at each stage.

:::viz plateaus
:::

:::read How to read this visual
**What is drawn.** Saxe et al.'s exact mode strengths $u(t)$ for four modes, as in chapter 8. The lines are the exact solution and the dots are discrete gradient descent.

**What to try.** Move the initial strength $u_0$.

**What to notice.** The plateaus are where the order of learning is decided: strongest mode first, weakest last. An optimizer that removed the plateaus by escaping instantly would learn all modes at once, and lose that ordering. Whether that would help or hurt generalization in real networks is the open question.
:::

## 6. E. When the two instabilities meet

The two halves of the survey have been studied separately. In practical training, though, they can occur together. Training at the edge of stability oscillates along the top Hessian eigenvectors, while in other directions the iterate may pass near saddles. The analyses reviewed in the survey treat one unstable regime or the other, never both at once.

- Does the oscillation along high-curvature directions change the escape time along a negative-curvature direction, for example through the third-order coupling of Eq. (12)?
- Does saddle-to-saddle learning from small initialization proceed differently at large step sizes, where the plateaus may be crossed while the sharpness is held at $2/\eta$?

A model that contains both a strict saddle and a sharp valley, analyzed with the central-flow method, would be a natural first step.

## 7. F. Global exploration and hitting times

Almost all results in the survey are local: they describe escape from one saddle, or the behavior near one valley. Azizian et al. (chapter 7) took a global view. Using large-deviations theory, they bounded the expected time for SGD to reach the global minimizers, which scales as $\exp(E/\eta)$ in an energy $E$ set by the barriers SGD may have to climb. Extending such results to the anisotropic, state-dependent noise of minibatch SGD, and to large step sizes at which the iterates oscillate, would show which minima SGD reaches in realistic training times.

## 8. Conclusion

The survey traced how unmodified gradient methods behave when the gradient-descent map is locally unstable. **A single multiplier, $1 - \eta\lambda$, organizes the story.**

- **When $\lambda < 0$, GD leaves strict saddles.** It does so from almost every initialization (Lee et al. 2016, 2019), but possibly only after exponential time (Du et al. 2017), unless randomness places it away from the stable manifold. That randomness can be injected (Jin et al. 2017), or already present in SGD under suitable conditions (Daneshmand et al. 2018; Fang et al. 2019). From small initialization, the sequence of saddles GD visits is itself the mechanism of incremental learning (Saxe et al. 2014; Zhang et al. 2026).
- **When $\lambda > 2/\eta$, GD cannot settle either** (Ahn et al. 2022). Instead it oscillates, and the third-order feedback of self-stabilization holds the sharpness at the edge of stability (Cohen et al. 2021; Damian et al. 2023). Central flows now describe this behavior with high accuracy for deterministic training (Cohen et al. 2025), and it has adaptive and stochastic counterparts (Cohen et al. 2022; Andreyev & Beneventano 2024).

Across both halves, three patterns recur. Every guarantee is relative to a notion of success, and the links between notions hold only under assumptions that deep networks rarely satisfy exactly. The most predictive analyses are increasingly about **trajectories rather than landscapes**. And quantities that classical theory treats as obstacles turn out to be mechanisms that shape what a network learns: negative curvature, large step sizes, and gradient noise.

:::key The take-away of the survey
A theory of training that treats **instability as part of the algorithm**, rather than as a failure to be avoided, appears to be the most promising direction for explaining why simple gradient methods succeed on non-convex losses.
:::

:::plain Where to go from here
- The [crash course](learn.html) builds every basic term used in this guide from scratch, in twelve stops.
- The [paper library](library.html) keeps the per-paper visual pages built for our first, broader survey, including most of the papers this guide draws on.
- Every number in the guide's new visuals is recomputed by `scripts/check-math/guide_*.py` in the repository.
:::
