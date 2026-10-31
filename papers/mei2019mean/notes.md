# Mean-Field Theory of Two-Layer Neural Networks

:::minute The paper in one minute
**Question:** Can a very wide two-layer network be analyzed by tracking a *distribution* of neurons instead of every weight, without width requirements scaling with input dimension?

**Model/Method:** Represent the $N$ neurons by an empirical probability measure; compare one-pass SGD with a transport PDE (and a diffusion PDE for injected noise).

**Finding:** The authors derive stronger finite-width mean-field approximation bounds, dimension-free in important regimes; extend to some unbounded activations and noisy SGD; recover kernel regression as a special scaling limit.

**Why we care:** Wide-network training has a *feature-learning, distributional* description; the frozen-kernel regime is only one limiting case.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. The $1/N$ scaling turns a network into an average

A two-layer network can be normalized in two common ways. With a $1/\sqrt{N}$ prefactor (the NTK scaling), the output of $N$ random neurons fluctuates by order $1$ at initialization and each neuron barely moves during training. With a $1/N$ prefactor (the **mean-field scaling**, used in this paper), the output is an *average* of neuron outputs, and each neuron moves by order $1$, so the features themselves change.

For example, take $N = 10^4$ neurons whose outputs are independent $\pm1$ coin flips. Their sum divided by $\sqrt{N}$ has standard deviation $1$; divided by $N$ it has standard deviation $1/\sqrt{N} = 0.01$. Under the $1/N$ scaling the network output is close to its expected value, which depends only on the *distribution* of the neurons.

### 2. Averages of $N$ samples are accurate to $1/\sqrt{N}$, whatever the dimension

The law of large numbers says an average of $N$ independent copies of a quantity with variance $s^2$ has error of order $s/\sqrt{N}$. The dimension of each copy does not enter, only its variance. Example: neurons $\mathrm{relu}(w^\top x)$ with $w \sim N(0, I_d)$ and a unit input $x$. Each output is $\mathrm{relu}(z)$ with $z \sim N(0, 1)$ in every dimension $d$, so the mean is $1/\sqrt{2\pi} \approx 0.399$ and the variance $\tfrac12 - \tfrac{1}{2\pi} \approx 0.341$. An $N$-neuron average then errs by about $0.58/\sqrt{N}$, for $d = 10$ and for $d = 1000$ alike. This is the mechanism behind the paper's "dimension-free" $N^{-1/2}$ term.

### 3. Particles following a velocity field move their distribution: the continuity equation

If each particle at $\theta$ moves with velocity $v(\theta)$, mass is conserved and the density obeys the **continuity equation** $\partial_t\rho_t = -\nabla_\theta\cdot(\rho_t v)$. Gradient descent on the neurons gives $v = -2\xi(t)\nabla_\theta\Psi(\theta; \rho_t)$, which yields the notes' transport PDE. The velocity depends on $\rho_t$ itself because each neuron's gradient depends on what all the others are doing.

The simplest case has no interaction: $\Psi(\theta) = \theta^2/2$ in one dimension, so each particle follows $\dot\theta = -\theta$ and $\theta_t = \theta_0 e^{-t}$. A standard normal starting distribution stays normal and shrinks: at time $t$ its standard deviation is $e^{-t}$, about $0.368$ at $t = 1$.

### 4. Injected noise adds a Laplacian, which spreads the distribution

Adding independent Gaussian noise to every particle at every step makes the density obey a **diffusion** (heat) equation: a Laplacian term $\Delta_\theta\rho$ appears next to the transport term. Without a drift, diffusion with coefficient $c$ in one dimension grows the variance linearly, by $2ct$. With the drift of idea 3 as well, $d\theta = -\theta\,dt + \sqrt{2\tau}\,dB_t$, the two balance and the distribution settles at $N(0, \tau)$. The paper's noisy-SGD PDE has this transport-plus-diffusion shape.

:::key Putting the four ideas together
Under the $1/N$ scaling the network output is an average over a distribution of neurons (idea 1), so a finite network approximates its infinite-width limit with an error of order $1/\sqrt{N}$ that need not depend on the dimension (idea 2). Training moves the distribution by a continuity equation (idea 3), and injected noise adds diffusion (idea 4). The paper bounds the gap between finite-width SGD and these PDEs, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. Replace a list of neurons by a population

A two-layer network is a sum of feature-producing neurons. In the mean-field normalization,

$$\widehat{f}_N(x) = \frac{1}{N}\sum_{i=1}^N \sigma_*(x; \theta_i), \qquad \widehat{\rho}_N = \frac{1}{N}\sum_{i=1}^N \delta_{\theta_i}.$$

The Dirac mass $\delta_{\theta_i}$ is a point mass at neuron $i$’s parameters. Thus the same predictor can be written as an integral:

$$\widehat{f}_N(x) = \int \sigma_*(x; \theta)\,\widehat{\rho}_N(d\theta).$$

**Interpretation.** Moving from $N$ weights to a probability measure $\rho$ is like studying the density of particles rather than following every particle. The distribution *moves*: this is different from freezing initial features in an NTK approximation.

### Worked example: empirical measure in one dimension

Let each neuron output $\sigma_*(x; w) = wx$ and take two weights $w_1 = 1$, $w_2 = 3$. Their empirical distribution is $\widehat{\rho} = \tfrac12\delta_1 + \tfrac12\delta_3$. For input $x = 2$,

$$\widehat{f}_2(2) = \tfrac12(1\cdot 2 + 3\cdot 2) = 4 = \int 2w\,\widehat{\rho}(dw).$$

If a gradient update moves both neurons, the locations of the point masses change, so $\rho_t$ evolves. A continuum description replaces these discrete masses with a smoothly evolving distribution when the network is wide enough.

### From gradient flow to a distributional PDE

The potential $\Psi(\theta; \rho)$ describes how the population risk changes when probability mass moves at $\theta$. Its characteristic velocity is proportional to $-\nabla_\theta\Psi$. Conservation of mass yields a transport equation of the form

$$\partial_t\rho_t = 2\xi(t)\,\nabla_\theta\cdot\big(\rho_t\nabla_\theta\Psi(\theta; \rho_t)\big).$$

This is a *Wasserstein gradient-flow* viewpoint, not ordinary gradient flow in a fixed vector of parameters.

## 2. What is genuinely improved by this paper?

### The finite-width comparison

Section 3 compares the actual population risk of finite-width, discrete SGD, $R_N(\theta^k)$, with the risk $R(\rho_{k\varepsilon})$ predicted by the mean-field PDE. Theorem 1 provides high-probability approximation bounds for noiseless SGD. In the fixed-second-layer-coefficient regime, the bound has the schematic shape

$$\sup_{k \le T/\varepsilon}\big|R_N(\theta^k) - R(\rho_{k\varepsilon})\big| \;\lesssim\; C_T\left(\frac{\sqrt{\log N} + z}{\sqrt{N}} + \sqrt{\varepsilon}\,\big(\sqrt{D} + \sqrt{\log N} + z\big)\right).$$

This display is a *schematic simplification* of Eq. (9), not a substitute for its precise constants, conditions, and logarithms. Crucially, the $N^{-1/2}$ term does not have a dimension multiplier in important regimes. A small enough step size remains necessary and can itself depend on $D$.

### Noisy SGD introduces diffusion

When the SGD update includes deliberately injected Gaussian noise, the limiting PDE contains a Laplacian term:

$$\partial_t\rho_t = 2\xi(t)\nabla\cdot(\rho_t\nabla\Psi_\lambda) + \frac{2\xi(t)\tau}{D}\Delta_\theta\rho_t.$$

The transport term moves the population according to risk; the Laplacian spreads it. Theorem 4 extends the approximation, but *its general-coefficient noisy case is not dimension-free* (Remark 5).

### When does mean-field reduce to a kernel method?

In Section 4, increasing a model-output scaling parameter makes the neuron distribution move only weakly over a rescaled time interval. Then the evolving kernel operator $H_{\rho_t}$ is approximately its initial value $H_{\rho_0}$, and the prediction residual obeys

$$\partial_t u_t \simeq -H_{\rho_0}u_t.$$

This is linear kernel dynamics, and an associated limit recovers kernel regression. It is a *special limit* of mean-field dynamics, not the entire mean-field theory.

:::warn
**Read correctly:** “Dimension-free” refers to particular width-dependent approximation terms under bounded regularity constants. It is not a claim that all sample, step-size and computational requirements are independent of dimension.
:::

## 3. Contributions, assumptions, and limits

**Contributions.** (i) Stronger approximation guarantees for the finite-neuron SGD dynamics, including width bounds that can avoid explicit dimension dependence; (ii) extensions beyond bounded activations and to noisy stochastic gradients; (iii) identification of kernel regression as a limiting version of the same distributional model. **Assumptions.** One-pass i.i.d. sampling, suitable small steps, independent neuron initialization, bounded or sub-Gaussian data-related quantities, regularity and Lipschitz properties of the potentials $V, U$, with extra initialization/smoothness conditions for noisy SGD (Assumptions A1–A6). Some claims fix output-layer coefficients; others update them. **Limitations.** Results depend on distributional regularity and time horizons; approximation errors may grow with time. The different noisy/general-coefficient cases do not share the same dimension-free rate. The paper is not a universal convergence-to-global-minimum result for deep networks.

## Survey connections

**Lu et al. (Ref 94)** transports distributional reasoning from two-layer networks into a special continuous-depth ResNet. **Li–Ma–Zhang (Ref 95)** shows that networks that actually move their features can learn targets inaccessible to polynomial-size fixed kernels. **Arora et al. (Ref 91)** studies a frozen-kernel spectrum; this paper clarifies where that description arises.

## Quick reference

| Concept | Meaning |
|---|---|
| $\widehat{\rho}_N = N^{-1}\sum_i\delta_{\theta_i}$ | Distribution of the finite set of neuron parameters. |
| $\rho_t$ | Mean-field population distribution evolving in training time. |
| Transport PDE | Conservation of probability mass under neuron drift. |
| Diffusion term $\Delta\rho$ | Spread from injected noise. |
| $H_{\rho_0}$ | Fixed initial kernel in the small-movement limit. |

### Papers Master note

:::plain
Mei–Misiakiewicz–Montanari (2019) sharpen mean-field approximations of two-layer SGD by tracking distributions of neurons. Width-dependent approximation terms can avoid explicit dimension dependence under regularity conditions; they also treat noisy SGD and show that fixed-kernel regression is a special scaling limit.
:::

> **Source:** [COLT 2019 original](https://proceedings.mlr.press/v99/mei19a/mei19a.pdf), Sections 3–4, Theorems 1 and 4, Remark 5. Toy example and schematic bound are explanatory.
