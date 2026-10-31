# Learning Over-Parametrized Two-Layer ReLU Networks beyond NTK

:::minute The paper in one minute
**Question:** Can trained neural networks learn a target that polynomial-size fixed-feature or NTK methods provably cannot learn as accurately?

**Model/Method:** A two-layer teacher with orthonormal directions, absolute-value activation and Gaussian inputs; a wide student trained by a *two-stage truncated gradient descent* method.

**Finding:** The student attains population error $O(d^{-1-Q})$ for a small constant $Q > 0$ with polynomial resources, whereas the specified polynomial-size kernel/feature methods have $\Omega(1/d)$ error lower bounds.

**Why we care:** Feature evolution can produce provable benefits unavailable to the NTK’s fixed-kernel approximation.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. A kernel method is linear regression on features chosen in advance

A **fixed-feature method** picks a map $\phi(x)$ before seeing the data and fits only the coefficients $c$ in $c^\top\phi(x)$, a linear least-squares problem. A **kernel method** is the same thing with features given implicitly by a kernel $K(x, x')$. The **neural tangent kernel** uses as features the network's gradients at initialization. Whatever the features, if they cannot express part of the target, no amount of fitting recovers it. A neural network trained beyond the NTK regime can change its features.

### 2. Gaussian inputs split a target into orders, and low orders can miss a lot

For a Gaussian input $z \sim N(0, 1)$, every function can be expanded in **Hermite polynomials** $1, z, z^2 - 1, \dots$, which are orthogonal under the Gaussian. For a neuron-like target the expansion splits the loss into one term per order; for the paper this gives a sum of tensor-decomposition problems (its Eq. 4).

Take the teacher's activation $|z|$. Its mean is $\sqrt{2/\pi} \approx 0.798$ and its variance $1 - 2/\pi \approx 0.363$. The best fit by $a + b(z^2 - 1)$ uses $b = \tfrac12\sqrt{2/\pi} \approx 0.399$ and explains $2b^2 = 1/\pi \approx 0.318$ of that variance. The remaining $0.045$, about $12\%$, lives in orders $4$ and higher. Features limited to low orders leave that part unfitted.

### 3. Second moments cannot tell an orthonormal basis apart from its rotations; fourth moments can

The second moment of the teacher directions, $\tfrac12(e_1e_1^\top + e_2e_2^\top) = I/2$, is the same for every orthonormal basis of the plane. Rotate the basis by $45°$ to $u_1 = (1, 1)/\sqrt2$, $u_2 = (1, -1)/\sqrt2$: still $I/2$. The fourth-order tensor tells them apart. Evaluated at $e_1$ in all four slots, $\tfrac12(e_1^{\otimes4} + e_2^{\otimes4})$ gives $\tfrac12$, while $\tfrac12(u_1^{\otimes4} + u_2^{\otimes4})$ gives $\tfrac14$. Orthogonal decompositions of tensors of order $3$ or more are unique, which is why fitting the fourth-order term identifies the teacher's directions.

### 4. Tensor power steps start slowly, then accelerate

Recovering a direction from a fourth-order orthogonal tensor resembles the **tensor power method**: each coordinate of the current guess is replaced by its cube, then the guess is normalized. The ratio of the two leading coordinates is therefore cubed at every step. Starting from a near-tie of $1.1$, it becomes $1.33$, then $2.36$, then $13.1$, then about $2.3 \times 10^3$. Progress is invisible for a while and then sudden. The paper notes that its second stage shows the same long plateau followed by faster progress, as observed for randomly initialized tensor power methods.

:::key Putting the four ideas together
A fixed-feature or kernel method only fits coefficients on features chosen in advance (idea 1). For this Gaussian teacher, the low Hermite orders that such methods capture well leave a residual in the higher orders (idea 2), and only the fourth-order structure pins down the teacher's directions (idea 3). A trained network can move its neurons onto those directions, after a plateau (idea 4), and so beats the $\Omega(1/d)$ kernel lower bound, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. What exactly is the NTK limitation?

A kernel method uses a fixed feature map $\phi(x)$ and learns only linear coefficients $c$, giving $f(x) = c^\top\phi(x)$. In the neural tangent kernel (NTK) approximation the effective features are the network’s derivatives at initialization; parameters can move, but the learned predictor is modeled by a *fixed* kernel. A nonlinear neural network can instead modify the feature directions themselves. Li, Ma and Zhang construct a teacher–student problem where this extra freedom matters quantitatively.

### The special target used in the theorem

The input $x \sim \mathcal{N}(0, I_d)$ and the teacher is

$$f^*(x) = \sum_{j=1}^d a_j\,|w_j^{*\top}x|, \qquad \{w_j^*\}_{j=1}^d \text{ orthonormal}, \qquad a_j \ge 0.$$

Because $|z| = \mathrm{ReLU}(z) + \mathrm{ReLU}(-z)$, this absolute-value teacher can be represented by paired ReLU units. The nonnegative coefficients are bounded in the paper within a condition-number parameter $\kappa$.

### Worked example: what low-order moments can miss

Take $d = 2$, teacher directions $e_1, e_2$, and $a_1 = a_2 = 1/2$. Then

$$f^*(x) = \tfrac12\big(|x_1| + |x_2|\big).$$

For unit directions $e_1, e_2$, the second-moment matrix $\tfrac12(e_1e_1^\top + e_2e_2^\top) = I/2$ treats every rotation similarly; it does not identify the individual coordinate axes. A fourth-order tensor contains $\tfrac12(e_1^{\otimes4} + e_2^{\otimes4})$, which can reveal the two axes under suitable uniqueness conditions. This simplified calculation illustrates why recovering *higher-order structure* can require more than kernel-style linearized learning.

:::key
**Big idea:** An overparameterized network can reorganize the distribution of feature directions; a fixed NTK approximation cannot perform the same feature learning.
:::

## 2. How does the proof establish a separation?

### Algorithm 1 is not unrestricted vanilla GD

The theoretical procedure is *truncated gradient descent*: neurons with norm above a stage-dependent threshold are not updated. It uses small random Gaussian initial weights, a wide ReLU student with norm-dependent output coefficients, and two stages with different truncation levels. The truncation is technically important; do not replace the guaranteed method with arbitrary SGD or ordinary GD.

### Stage-like feature discovery

First, training matches lower-order population statistics (the zeroth- and second-order tensor components). Then higher-order tensors, particularly the fourth-order structure, resolve the teacher’s orthogonal directions. The authors’ analysis features an initial slow plateau followed by faster progress, consistent with their numerical illustrations.

### The two statements that must be compared

The learning theorem yields, under Gaussian input, orthonormal target weights, polynomially many neurons and samples, and specified truncation/step choices,

$$L(\widehat{W}) \le O(d^{-1-Q}), \qquad 0 < Q < 1/100.$$

Theorem 2 proves that a polynomial-size fixed feature map, or a kernel method given polynomial samples, incurs a lower bound of the form

$$L_{\mathrm{kernel}} \ge \Omega(1/d)$$

for a suitable family of target weights. Hence the neural network achieves an asymptotically smaller *population* loss in this construction.

### Worked comparison for intuition

With $Q = 0.005$ (illustrative), $d^{-1-Q}/d^{-1} = d^{-0.005}$. At $d = 10^4$ this ratio is $10^{-0.02} \approx 0.95$: the asymptotic ratio shrinks with $d$. Big-$O$ and $\Omega$ contain constants, so this is *not* a predicted experiment or numerical guarantee.

:::warn
**Interpret theorem scope exactly:** The separation is for a special Gaussian teacher class and resource regime; it is not proof that every trainable neural network outperforms every kernel method on realistic datasets.
:::

## 3. Contributions, assumptions, and limitations

**Contributions.** Provable population-risk improvement beyond polynomial-resource fixed-kernel models, a stage-by-stage explanation via higher-order tensor decomposition, and a finite-sample/finite-width transfer from a population dynamical analysis. **Assumptions.** Gaussian inputs; teacher weights forming an orthonormal basis; controlled positive teacher coefficients; particular two-layer ReLU student parameterization; large polynomial width and sample counts; carefully chosen learning rates and truncation thresholds. **Limitations.** Truncated GD differs from unrestricted GD in practice. The comparison is asymptotic and target-specific; the target/teacher is highly structured. The results do not prove global training convergence for arbitrary deep ReLU networks, nor superiority over every adaptive kernel technique.

## Cross-paper connections

**Mei et al. (Ref 93)** describes moving neuron distributions, with kernel regression as a limiting regime. **Arora et al. (Ref 91)** explains optimization in a nearly frozen NTK regime; Li et al. prove why that regime can be insufficient. **Saxe (2014)** provides another stage-like mode-learning explanation in deep linear networks, though the mathematical models differ. **Xu–Du (Ref 96)** cautions that feature freedom and extra neurons do not necessarily speed convergence.

## Quick reference

| Object | Interpretation |
|---|---|
| $f^*(x) = \sum_j a_j\lvert w_j^{*\top}x\rvert$ | Structured Gaussian teacher with orthogonal directions. |
| Truncated GD | Gradient update gated by neuron-norm thresholds. |
| Zeroth/second moments | Initial coarse target information. |
| Fourth-order tensor | Helps identify orthogonal teacher directions. |
| $O(d^{-1-Q})$ vs $\Omega(1/d)$ | Special neural-network vs polynomial-resource kernel loss comparison. |

### Papers Master note

:::plain
Li–Ma–Zhang (2020) show that a wide two-layer ReLU student trained by staged truncated GD learns a special Gaussian orthogonal teacher to population error $O(d^{-1-Q})$, while polynomial-size kernel/fixed-feature methods face $\Omega(1/d)$ lower bounds. The separation illustrates feature learning beyond a fixed NTK regime.
:::

> **Source:** [COLT 2020 original](https://proceedings.mlr.press/v125/li20a/li20a.pdf), Algorithm 1, Theorems 1–2, Section 2. Worked moment example pedagogical.
