# Over-Parameterization Exponentially Slows Down GD for a Single Neuron

:::minute The paper in one minute
**Question:** Does adding student neurons always speed up gradient descent, even for learning one ReLU teacher?

**Model/Method:** Gaussian-input teacher–student squared-loss training with one ReLU teacher and $n \ge 1$ student neurons; analyze GD and gradient flow from small Gaussian initialization.

**Finding:** For $n \ge 2$, the authors prove a $O(T^{-3})$ discrete-GD upper bound and a matching-order $\Omega(T^{-3})$ gradient-flow lower bound, whereas the $n = 1$ case has an exponential-in-$T$ loss decay in the cited setting.

**Why we care:** More parameters can make a task *slower* even when they preserve realizability and global convergence.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Exponential versus algebraic convergence

A loss that decays **exponentially**, $L(t) \le e^{-ct}$, needs time proportional to $\log(1/\varepsilon)$ to reach $\varepsilon$. A loss that decays **algebraically**, $L(t) \approx t^{-3}$, needs time $\varepsilon^{-1/3}$. With $c = 1$, reaching $10^{-6}$ takes $t \approx 13.8$ against $t = 100$; reaching $10^{-12}$ takes $27.6$ against $10^4$. The gap itself grows without bound as $\varepsilon$ shrinks, which is the sense of "exponentially slower": $\log(1/\varepsilon)$ versus a power of $1/\varepsilon$.

### 2. How fast the gradient vanishes decides the rate

Along gradient flow, $\frac{dL}{dt} = -\|\nabla L\|^2$. If $\|\nabla L\|^2 \ge \mu L$ (a Polyak–Łojasiewicz inequality), then $L$ decays exponentially. If instead the gradient is only as large as a higher power of the loss, $\|\nabla L\|^2 \ge c\,L^\alpha$ with $\alpha > 1$, the decay is algebraic, $L(t) \approx t^{-1/(\alpha - 1)}$. The paper's local bound (its Theorem 7) is $\|\nabla L\| \ge \Omega(L^{2/3})$, so $\alpha = 4/3$ and the rate is $t^{-3}$; the notes' surrogate ODE below is exactly this case. The earlier bound $\|\nabla L\| \ge \Omega(L)$ (Zhou et al., 2021), $\alpha = 2$, only gave $t^{-1}$.

### 3. Redundant neurons create a continuum of global minima

ReLU is positively homogeneous: $[c\,z]_+ = c\,[z]_+$ for $c \ge 0$. So with two students, $w_1 = \alpha v$ and $w_2 = (1 - \alpha)v$ reproduce the teacher $[v^\top x]_+$ exactly for *every* $\alpha \in [0, 1]$. The global minima form a segment rather than a point, and the loss is exactly flat along it. A flat direction at a minimum means the Hessian has a zero eigenvalue there, so the quadratic bowl that gives exponential convergence for $n = 1$ is gone.

### 4. A loss that is cubic in the misalignment gives a $t^{-3}$ tail

Near that segment, what remains to be learned is how the two students' components *orthogonal* to $v$ cancel. The paper measures this with a potential $Z(t) = \sum_{i<j}\|z_i - z_j\|$, where $z_i$ is the part of $w_i$ orthogonal to $v$, and shows $L \ge \Omega(Z^3)$ and $Z = \Omega(t^{-1})$. A one-dimensional caricature shows why such flatness is slow: for $L(b) = |b|^3$, gradient flow $\dot{b} = -3b^2$ gives $b(t) = 1/(b_0^{-1} + 3t)$, so $L(t) = (b_0^{-1} + 3t)^{-3}$. From $b_0 = 1$, at $t = 100$ the loss is still $(301)^{-3} \approx 3.7 \times 10^{-8}$; a quadratic loss $b^2$ would be at $e^{-400}$.

:::key Putting the four ideas together
The rate of a gradient method is set by how the gradient shrinks with the loss (idea 2): proportionally gives exponential decay, more slowly gives algebraic decay (idea 1). With extra students, the global minima form a flat continuum (idea 3), and the remaining misalignment between students costs only a cubic amount of loss, which gradient descent removes at rate $t^{-3}$ (idea 4). The paper proves both the $O(T^{-3})$ upper bound for GD and the $\Omega(T^{-3})$ lower bound for gradient flow, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. An apparent paradox about width

Many overparameterization papers show that sufficiently wide networks can avoid difficult optimization geometry. Xu and Du ask a different question: *how fast* does a student learn when a target is already representable with one neuron? The teacher is $f^*(x) = [v^\top x]_+$; the student uses $n$ ReLU neurons:

$$f_W(x) = \sum_{i=1}^n [w_i^\top x]_+, \qquad L(W) = \tfrac12\mathbb{E}_{x\sim\mathcal{N}(0, I)}\big(f_W(x) - f^*(x)\big)^2.$$

For $n = 1$ the student has exactly the teacher’s representational size. For $n \ge 2$ it has redundant parameterizations of the same target, for example $w_1 = 0.4v$ and $w_2 = 0.6v$ achieve $f_W = f^*$ by positive homogeneity.

### Worked example: many distinct global solutions

Set $v = (1, 0)$ and input $x = (2, -1)$. Then teacher output $[2]_+ = 2$. Both $w_1 = v$ alone and the pair $(0.4v, 0.6v)$ yield 2, since

$$[0.4(2)]_+ + [0.6(2)]_+ = 0.8 + 1.2 = 2.$$

One function can correspond to many student weight configurations. Extra degrees of freedom can create slow interactions among the students, even though the target function is simple.

### What the paper measures

The performance metric is population squared loss along the gradient-descent trajectory, *not* test accuracy, wall-clock time, or a comparison across arbitrary width distributions. The slow-down claim compares $n = 1$ with $n \ge 2$ under the same tightly controlled teacher–student model.

:::key
**Intuition:** Extra neurons are not just extra descent directions; they can move relative to one another. These “splitting” or alignment degrees of freedom may relax only algebraically even while the predictor approaches the teacher.
:::

## 2. Why the rate becomes algebraic

### Theorems 1 and 2: two distinct trajectories

With sufficiently high input dimension, small Gaussian initialization, and specified step size, the first main theorem bounds *discrete GD* with $n \ge 2$ by

$$L(W_T) \le C\,T^{-3}$$

where $C$ depends on model, initialization and learning-rate parameters. The second main theorem proves for *continuous gradient flow* a lower bound $L(W_t) \ge (c_1t + c_2)^{-3}$ in the stated high-probability regime. These establish the same *order of decay across related but different dynamics*; do not call this an identical finite-step upper/lower theorem.

### How the three-phase analysis works

**Phase 1: orientation.** Student directions rotate toward teacher $v$ while remaining relatively small.

**Phase 2: alignment and radial correction.** Angular errors stay small, and an aggregate residual decreases rapidly.

**Phase 3: slow local relaxation.** Interactions between extra student neurons govern the long-time tail; the authors prove a stronger local gradient lower bound and a potential-function lower bound.

### Worked intuition: gradient magnitude controls the rate

A simple surrogate differential inequality

$$\frac{dL}{dt} = -c\,L^{4/3}$$

integrates to $L^{-1/3}(t) = L^{-1/3}(0) + (c/3)t$, hence

$$L(t) = \big[L(0)^{-1/3} + (c/3)t\big]^{-3}.$$

This *pedagogical ODE*, not an equation asserted verbatim by the paper, illustrates why a vanishing gradient condition can yield a $t^{-3}$ tail rather than exponential contraction $\dot{L} = -cL$.

:::warn
**Do not overgeneralize:** “Exponentially slows down” compares exponential loss decay in the exactly parameterized case against algebraic decay when extra neurons are added. It is not a claim that runtime increases exponentially with the number of neurons.
:::

## 3. Contributions, assumptions, and limitations

**Contributions.** A global GD convergence proof beyond exact parameterization for this single-ReLU teacher; an $O(T^{-3})$ rate; a corresponding gradient-flow $\Omega(T^{-3})$ obstruction; analysis of self-balancing among student neurons and a potential measuring their pairwise misalignment. **Assumptions.** Gaussian population inputs, squared loss, one teacher ReLU with fixed output coefficient, $n \ge 2$ student ReLUs, sufficiently large input dimension, small Gaussian initialization, and suitably controlled GD learning rate. The precise high-probability rates use these explicit hypotheses. **Limitations.** This is one specialized teacher–student architecture, not all neural-network losses. The rate comparison is primarily theoretical, with separate upper (discrete GD) and lower (gradient flow) results. It does not imply overparameterization is generally harmful, nor that it slows learning in the NTK regime.

## Cross-paper connections

**Du and Allen-Zhu (2019)** prove successful training with enough width under different models; this paper cautions that successful global convergence does not imply a fast rate. **Li–Ma–Zhang (Ref 95)** uses width for richer feature learning in a structured teacher; here width creates a slow tail. **Safran–Shamir (2018)** shows extra neurons can empirically reduce bad local-minimum frequency, another distinct outcome.

## Quick reference

| Object | Interpretation |
|---|---|
| $n = 1$ | Exactly parametrized single-neuron teacher model. |
| $n \ge 2$ | Overparameterized student with interacting neurons. |
| $O(T^{-3})$ | GD loss upper bound (under theorem conditions). |
| $\Omega(T^{-3})$ | Gradient-flow loss lower bound (related conditions). |
| Three phases | Initial angular alignment, residual reduction, slow local tail. |

### Papers Master note

:::plain
Xu–Du (2023) prove that adding student ReLU neurons to a single-neuron Gaussian teacher problem can change exponential convergence (exact width) to algebraic loss decay. They establish $O(T^{-3})$ for discrete GD and an order-matching $\Omega(T^{-3})$ lower bound for gradient flow, revealing a conditional optimization cost of redundancy.
:::

> **Source:** [COLT 2023 original](https://proceedings.mlr.press/v195/xu23a/xu23a.pdf), Theorems 1–2, technical overview Sections 2, 4–5. Toy ODE pedagogical.
