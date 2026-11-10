## 1. One instability, two faces

This chapter follows Section V of the survey. Chapters 7 to 14 followed each literature in roughly chronological order. Here we step back and compare them. We ask three questions. What do the two kinds of instability have in common, and where do they differ? What does each result actually guarantee? And which of its assumptions hold for the networks that are trained in practice?

The common structure was made precise by Ahn, Zhang & Sra (2022), chapter 11: **strict saddles and minima sharper than $2/\eta$ are both unstable fixed points of the GD map**, and GD avoids both from almost every initialization. In the language of chapter 5, both have a multiplier $1 - \eta\lambda$ outside $[-1, 1]$: above $+1$ for a saddle, below $-1$ for a too-sharp minimum. The first visual is that multiplier again. Keep it in view while reading the table.

:::viz multiplier
:::

:::read How to read this visual
**What is drawn.** The same multiplier explorer as in chapter 5. The top plot is the distance $u_t$ along one eigenvector over 30 steps; the middle chart is the multiplier line $m = 1 - \eta\lambda$ with its three bands; the bottom chart is $|u_t|$ on a log scale.

**What to try.** Compare the two presets that lie outside the stable band: *saddle −0.1* and *too sharp 2.2*.

**What to notice.** Both make $|u_t|$ grow: on the log chart, both are rising straight lines. They differ in one thing only, the sign pattern. The saddle grows monotonically, so the iterate leaves in one direction. The too-sharp minimum grows with alternating sign, so the iterate bounces across the valley. That single difference, monotone escape against oscillation, is the root of every row of the table below.
:::

| | **Negative curvature** (saddles, chapters 7–9) | **Excess curvature** (edge of stability, chapters 10–14) |
|---|---|---|
| Condition | $\lambda < 0$ along some eigenvector | $\lambda > 2/\eta$ along some eigenvector |
| Multiplier $1 - \eta\lambda$ | $> 1$: monotone growth | $< -1$: growth with alternating sign |
| What GD does | leaves along the negative-curvature direction; its stable set has measure zero (Lee et al. 2016, 2019) | oscillates along the top eigenvector; cannot converge to such points (Ahn et al. 2022) |
| What limits progress | time near the stable manifold, $\propto \log(1/\delta)$; exponential in $d$ in the worst case (Du et al. 2017) | progressive sharpening against oscillation-induced sharpness reduction (Damian et al. 2023; Cohen et al. 2025) |
| Higher-order effect | decides where the escaping trajectory goes | restoring drift $-\eta\tfrac{x^2}{2}\nabla S$ (survey Eq. 18) |
| Role of noise | supplies a component along the escape direction (Ge et al. 2015; Daneshmand et al. 2018; Fang et al. 2019) | makes the threshold stricter (Eq. 17); batch sharpness locks to $2/\eta$ (Andreyev & Beneventano 2024) |
| Role in learning | saddle-to-saddle simplicity bias (Saxe et al. 2014; Zhang et al. 2026) | selects flatter regions; catapults (Lewkowycz et al. 2020; Kalra & Barkeshli 2024); can change the destination (Mohtashami et al. 2023) |
| Status of the theory | asymptotic theorems; polynomial-time escape with noise; degenerate saddles open | rigorous in specific settings (Damian et al. 2023; Arora et al. 2022; Chen & Bruna 2023); accurate but informal flows (Cohen et al. 2025) |

This is the survey's Table V. Three differences stand out.

**The question asked.** For saddles, the question is how to leave as fast as possible, and the answer is measured in escape time. For the edge of stability, the question is how to make progress while unstable, and the answer is a description of the time-averaged trajectory. The saddle literature counts iterations; the edge-of-stability literature predicts curves.

**The role of higher-order terms.** At a saddle, the linear instability is the whole story near the critical point, and higher-order terms matter only after escape. At the edge, the linear instability alone would predict divergence, and it is the third-order term of Eq. (18) that keeps training alive. The saddle analysis is local; the edge-of-stability analysis is a feedback loop between the iterate and its own curvature.

**The role of noise.** Noise helps at saddles, because it supplies a component along the direction of escape. Noise does not help stability: it tightens it. Under condition (17), smaller batches make the curvature threshold stricter. Under Eq. (20), SGD locks a minibatch quantity, not the full-batch sharpness, to $2/\eta$. **The same randomness that speeds up escape from negative curvature pushes training away from regions of large positive curvature.** The next two visuals show the two effects side by side.

:::viz noisesaddle
:::

:::read How to read this visual
**What is drawn.** The saddle $f = x^2/2 + y^4/4 - y^2/2$, with the escape direction $y$ in red and the upward direction $x$ in blue, and minima at $(0, \pm1)$ (green). Both balls start at $(1.5, 0)$, exactly on the line that leads into the saddle.

**What moves.** White: full-batch GD. Coral: SGD, whose gradient on each step comes from one of two training examples, $f \pm s\cdot y$ (noise along the escape direction) or $f \pm s\cdot x$ (noise across it).

**What to try.** Switch *noise points along* between $y$ and *x only*, and change the noise size $s$.

**What to notice.** GD slides into the saddle and stays. With noise along $y$, SGD is knocked off the stable line and escapes to a minimum. With noise along $x$ only, however large it is, SGD stays on the line and stops at the saddle too. The readout says whether, and when, each run escaped. **At a saddle, noise helps, but only its component along the negative-curvature direction counts**: the CNC condition of chapter 8.
:::

:::viz sgdstab
:::

:::read How to read this visual
**What is drawn.** The SGD stability map of chapter 10. The horizontal axis is $\eta\lambda$ (step times average curvature) and the vertical axis is $\eta s$ (step times the spread of the per-example curvatures). Each curve bounds the stable region of one batch size; the full-batch region is the strip $0 \le \eta\lambda \le 2$. The chart runs real SGD at the dragged point.

**What to try.** Keep the dot at a moderate $\eta\lambda$, around $1.5$, and lower the batch size from 100 to 1.

**What to notice.** The stable region shrinks as the batch shrinks: the same minimum that holds full-batch GD can reject small-batch SGD. **Around a minimum, noise hurts stability**, and the effect is to push SGD towards flatter, more uniform regions. Put this next to the previous visual: one kind of randomness, helpful on the negative-curvature side and restrictive on the positive-curvature side.
:::

## 2. What is guaranteed

Results in both literatures claim success in different senses, and comparing them without care leads to confusion.

1. **First-order stationarity**, $\lVert\nabla L\rVert \le \epsilon$. This is the guarantee of GD and SGD under Assumptions 1 and 3 (chapter 4). An $\epsilon$-FOSP may be a saddle point.
2. **Asymptotic avoidance.** Lee et al.'s theorem and its extensions guarantee that the limit, if it exists, is not a strict saddle. They give no rate.
3. **Second-order stationarity.** This is the guarantee of perturbed GD and of the noise-based analyses (chapter 8). It implies approximate local optimality only under the strict-saddle property.
4. **Progress at the edge.** No classical guarantee applies, because the step size violates the descent lemma. The available results are the monotone decrease of the central-flow loss (chapter 13), which is derived informally, and rigorous analyses in specific settings (Damian et al.; Arora et al.; Chen & Bruna).

A fifth target, implicit in Section IV, is reaching a **good** minimum: flat enough to be stable at the chosen step size and to generalize. No theorem reviewed in the survey guarantees this for practical networks. The self-stabilization and catapult analyses explain how the step size biases which minimum is reached.

## 3. What is proved against what is observed

| Assumption | Used by | Status in practice |
|---|---|---|
| Global $\ell$-smoothness, $\eta < 1/\ell$ | avoidance (Lee et al.), most escape bounds | violated: ReLU kinks; smoothness grows with the gradient norm; practical GD runs at $\eta\lambda_{\max} \approx 2$ |
| Lipschitz Hessian | perturbed GD, noise-based escape | plausible for smooth activations; undefined at ReLU kinks |
| Strict-saddle property | efficient escape (Ge et al.; Jin et al.) | proved for matrix and tensor problems; fails for deep linear networks and degenerate valleys |
| Isotropic injected noise | perturbed GD and SGD | not used in practice; SGD noise is anisotropic |
| Bounded noise variance | SGD analyses | heavy tails can arise even in linear regression (Gürbüzbalaban et al.) |
| CNC / dispersive noise | Daneshmand et al.; Fang et al. | closer to practice; verified empirically only in limited settings |
| Infinitesimal step (gradient flow) | Saxe et al.; Zhang et al. | discrete large steps change the trajectory and the destination (Mohtashami et al.; Galli et al.) |
| Full batch, no momentum | edge-of-stability mechanisms, central flows | practical training is stochastic and uses momentum; partial extensions exist (Cohen et al. 2022; Andreyev & Beneventano) |

This is the survey's Table VI. Three mismatches stand out.

**Smoothness and step size.** Nearly all saddle-escape results assume a global Lipschitz constant $\ell$ and a step $\eta \le 1/\ell$, which makes GD a descent method. Practical networks violate both. ReLU activations make the gradient discontinuous at their kinks. Smoothness is better described as growing with the gradient norm (Assumption 4). Most strikingly, full-batch GD at practical learning rates operates at $\eta\lambda_{\max} \approx 2$, far outside the regime assumed by the avoidance theorem. **The saddle-avoidance theorems are proved in a regime that practical training leaves early.**

**The nature of the noise.** Perturbation methods inject isotropic noise, which practitioners never do. Real SGD noise is anisotropic, its structure is tied to the Hessian, and it can be heavy-tailed even in linear regression. CNC and dispersive noise describe the noise SGD already has, but they have been checked only on a limited range of models and training stages.

**The geometry itself.** The strict-saddle property fails for deep linear networks with two or more hidden layers, which have saddles without negative curvature, and wherever symmetries or overparameterization create connected sets of degenerate critical points. **The property on which efficient escape rests holds most reliably in the models that are least like deep networks.** On the edge-of-stability side, the mechanisms assume a well-defined, differentiable sharpness and a gap between the top eigenvalues, and the central flow needs the oscillations to be fast compared with the drift of the eigenvectors.

These mismatches do not make the theory irrelevant. They shift the burden of proof. For problems that satisfy the assumptions, the saddle-escape theory is close to complete. For deep networks, both literatures supply **mechanisms**, whose relevance must be confirmed empirically: exponential divergence along negative curvature, stuck regions of small volume, and self-stabilization.

## 4. Strengths, weaknesses, and four lessons

The **saddle literature** is mathematically mature. Its theorems are rigorous and its assumptions explicit, and for problems that satisfy them, avoidance is almost sure and escape is almost free in iteration count. Its weakness is reach: small step sizes, strict saddles and noise conditions that are rarely checked, all of which practical training violates. The **edge-of-stability literature** has the opposite profile. Its observations are robust across architectures and tasks, and its best models predict real training curves with high accuracy. Its weakness is rigor. Its proofs cover simplified losses or small models, its most predictive tool is derived informally, and its extensions to SGD and Adam are largely empirical.

:::key Four cross-cutting lessons (survey Section V-E)
**From landscapes to trajectories.** Early work asked global questions about the landscape: are all local minima global, are all saddles strict? For deep networks the answers are often negative or unknown. Later work asks what the trajectory of a specific algorithm, with a specific step size and initialization, actually visits. These results are narrower in scope but much closer to practice.

**Step size and noise are mechanisms, not nuisances.** The step size selects the region, flat or sharp, in which training ends. The geometry of SGD noise drives escape and global exploration.

**Instability is a resource.** Negative curvature carries the iterate from one saddle to the next, implementing a simplicity bias. Excess curvature carries it out of sharp regions, implementing an implicit sharpness penalty. Stability in the classical sense, $\eta\lambda_{\max} < 2$ with monotone descent, is the exception during training rather than the rule.

**Degeneracy is the rule.** Symmetries, overparameterization, homogeneous activations and depth all create degenerate critical points and connected valleys of minima. Strict saddles are a convenient idealization. Saddle-to-saddle dynamics and slow convergence in degenerate valleys suggest that the escape question should become a question about time spent near degenerate sets.
:::

:::problem The next step
The gaps found here, between what is proved and what is observed, are exactly the open problems of the survey's last section.
:::
