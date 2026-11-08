## 1. The puzzle

Training a neural network means choosing numbers. A network $f(x;\theta)$ has a long list of adjustable numbers, its weights and biases, stacked into one vector $\theta \in \mathbb{R}^d$: the [[parameters]]. $d$ is the number of parameters, from thousands to billions. A [[loss]] $L(\theta)$ measures how badly the network with parameters $\theta$ fits the training data: zero would be a perfect fit. Training means making $L(\theta)$ small.

If $L$ were a single bowl, this would be easy: walk downhill and you reach the bottom. It is not a bowl. A neural-network loss is **non-convex**: its [[landscape]], the graph of $L$ over all possible $\theta$, has many valleys, ridges, plateaus and passes. Section 2 of the next chapter shows why this cannot be avoided.

Non-convexity is not a small inconvenience. Classical optimization theory treats it with caution, for good reasons:

- Deciding whether a given point is even a local minimum of a general smooth function is [[np-hard|NP-hard]] (Murty & Kabadi 1987). "NP-hard" means that no known algorithm solves every instance in a time that grows only polynomially with the size of the problem.
- Training a network with only three nodes, two hidden units and one output unit, is NP-complete in the worst case (Blum & Rivest 1992).

And yet, plain [[gd|gradient descent]] (GD) and its cheaper, noisy cousin [[sgd|stochastic gradient descent]] (SGD) routinely drive the training loss of modern networks close to zero. Both methods only ever look at the local slope. This is the puzzle our survey is about: **why do simple gradient methods succeed on losses that are hard in the worst case?**

The visual below is the simplest picture of the difficulty: a loss with one adjustable number $\theta$ and three valleys.

:::viz valleys
:::

:::read How to read this visual
**What is drawn.** The horizontal axis is the single parameter $\theta$. The vertical axis is the loss $\mathcal L(\theta) = 0.1\theta^2 + 0.5\sin 3\theta$ at that $\theta$. The blue curve is therefore the whole landscape of this one-parameter problem. The three dots on the curve are its local minima. The green one is the global minimum (the lowest point); the two white ones are local minima that are not global.

**What moves.** The coral ball is gradient descent: at every step it moves $\theta$ a small amount downhill, against the slope. The faint coral dots are the path it has taken.

**What to try.** Set the start $\theta_0$ with the slider and press *Roll downhill*. Start at $\theta_0 = 2.6$, then at $\theta_0 = -0.3$, then at $\theta_0 = -2.8$.

**What to notice.** The ball always stops in the valley directly below its start, and only some starts lead to the global minimum. The readout says which minimum it reached and how many steps it took. In one dimension, a valley that is not the best one is a trap. The surprise of deep learning is that in very high dimension this kind of trap seems to matter much less than this picture suggests.
:::

## 2. A layered explanation

The explanation that has emerged over the past decade is not a single theorem. It is a layered story, and our survey follows one thread through it.

1. **Are there bad valleys?** Early work found that for linear networks every local minimum is a global minimum (Baldi & Hornik 1989). Non-convex does not have to mean "full of traps".
2. **Saddles, not valleys.** Dauphin et al. (2014) argued that in high dimension the dominant obstacle is the [[saddle|saddle point]]: a flat point that curves up in some directions and down in others, surrounded by long plateaus. A theory of *strict* saddles followed: if every saddle has a direction that curves strictly down, gradient methods avoid them from almost every starting point (Lee et al. 2016), and with a little randomness they escape them efficiently (Jin et al. 2017).
3. **Width.** In parallel, the overparameterization literature showed that very wide networks can be trained to a global minimum without reasoning about saddles at all. This branch is outside our scope.
4. **The dynamics that actually happen.** Most recently, attention turned to what training does in practice. Large step sizes push training to the edge of stability (Cohen et al. 2021), loss curves show sudden spikes called catapults (Lewkowycz et al. 2020), and networks pass through a sequence of saddles while learning progressively more complex functions (Zhang, Saxe & Latham 2026).

## 3. The single thread: one multiplier

Here is the observation that organizes the whole survey. Near a point where the slope is zero, a gradient-descent step with step size $\eta$ multiplies the distance to that point, measured along each "curvature direction", by the same number:

$$1 - \eta\lambda,$$

where $\lambda$ is the curvature in that direction. Chapter 5 derives this from scratch. For now, keep three facts in mind:

- If $|1 - \eta\lambda| < 1$, the distance shrinks: gradient descent settles.
- If $\lambda < 0$ (the loss curves *down* in that direction), then $1 - \eta\lambda > 1$, the distance grows, and gradient descent **leaves**. This is how it escapes saddles. Section III of the survey is about this case.
- If $\lambda > 2/\eta$ (the loss curves up *too sharply* for the step size), then $1 - \eta\lambda < -1$, the distance grows while flipping sign, and gradient descent **bounces**. This is the starting point of catapults and the edge of stability. Section IV of the survey is about this case.

The two cases have been studied by largely separate communities, with different questions and different tools. The saddle literature asks whether, and how fast, GD escapes. The large-step literature asks why training still converges when it is unstable. Both describe the same object: the instability of the gradient-descent map. Ahn, Zhang & Sra (2022) proved a single theorem that covers both (chapter 10). Our survey puts the two literatures side by side.

## 4. Two research questions

:::key The two research questions of the survey
**RQ1 (Saddles).** When do unmodified GD and SGD avoid or escape saddle points, how long does escape take, and what roles do initialization and gradient noise play? (Survey Section III, chapters 6 to 8 here.)

**RQ2 (Stability).** What happens when the step size exceeds the classical stability limit $2/\lambda_{\max}$, and why does training still make progress? (Survey Section IV, chapters 9 to 13 here.)
:::

"Unmodified" is the key word of the scope. The survey studies the methods practitioners actually run: GD, SGD, momentum and adaptive methods such as Adam. It does not survey algorithms built specifically to escape saddles, such as cubic regularization, Hessian-free negative-curvature search or variance reduction. The one exception is perturbed gradient descent, the smallest change that makes escape provably fast. Overparameterization theory and the debate on whether flat minima generalize better are discussed only where they bear directly on optimization.

## 5. How this guide is built

Each chapter follows one part of the survey, in order. The map on the home page shows which survey section each chapter belongs to.

| Survey section | What it does | Guide chapters |
|---|---|---|
| I. Introduction | the puzzle, existing surveys, the two RQs | 1 |
| II. Preliminaries | loss, gradient, Hessian, flat points, assumptions, algorithms, the multiplier | 2–5 |
| III. Saddles (RQ1) | avoidance, escape time, noise, saddle-to-saddle learning | 6–8 |
| IV. Stability (RQ2) | catapults, edge of stability, mechanisms, central flows, Adam and SGD | 9–13 |
| V. Comparison | the two faces side by side; assumptions against practice | 14 |
| VI–VII. Open problems, conclusion | where the evidence runs out | 15 |

Every visual in the guide is followed by a **How to read this visual** box. It says what the axes are, what the colors mean, what moves, what to try, and what to notice. Every surface is a real function, and every ball runs a real optimizer on it. Every number shown is recomputed by a script in the repository.

:::plain Existing surveys, and why ours exists
Several surveys cover parts of this field: Bottou, Curtis & Nocedal (2018) on stochastic methods; Jain & Kar (2017) on structured non-convex problems; Jin et al. (2021) on perturbed gradient methods; Sun (2020) and Sun et al. (2020) on practical optimization and the landscape of neural networks; Danilova et al. (2022) on complexity and adaptive methods. All of them were written before, or without a focus on, the large-learning-rate phenomena: catapults, the edge of stability, self-stabilization, and their adaptive and stochastic versions. None treats saddle escape and large-step instability as two faces of the same linear-stability question. That is the gap our survey fills.
:::

:::problem The next step
Before any of this can be made precise, we need the objects themselves: what exactly is minimized, why it is non-convex, and how the slope and the curvature of the loss describe it near a point.
:::
