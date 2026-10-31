# Gradient Descent Escapes Saddles in Shallow ReLU Training

:::minute The paper in one minute
**Problem:** Most strict-saddle avoidance theorems assume a twice-continuously differentiable objective. A ReLU network is piecewise linear in its input and generally creates a parameter-space loss that need not satisfy classical global smoothness hypotheses. The authors ask whether a suitably weakened dynamical-systems result can still explain escape near saddle points in concrete shallow ReLU-like training problems.

**Principal result:** For the specified shallow ReLU-like problems, saddle avoidance can be proved without assuming that the full training objective obeys the classical smoothness regime; global convergence needs additional conditions on the initialization/limiting loss.

**Scope:** This paper closes part of the *smoothness-assumption* gap for a structured shallow ReLU class, not the entire practical-neural-network theory gap.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Near a saddle, one GD step is a linear map

Write one GD step as a map, $G(\theta) = \theta - \eta\nabla\mathcal{L}(\theta)$. A critical point $\theta^*$ is a **fixed point**: $G(\theta^*) = \theta^*$. Close to it, a Taylor expansion gives

$$G(\theta^* + v) \approx \theta^* + (I - \eta\nabla^2\mathcal{L}(\theta^*))\, v .$$

A Hessian eigenvalue $\lambda < 0$ (a **strict saddle**) gives the Jacobian an eigenvalue $1 - \eta\lambda > 1$: offsets along that eigenvector grow. The eigenvectors with Jacobian eigenvalue of size at most one span the **center-stable space** $E^{cs}$. Only motion inside it can stay near the saddle.

### 2. The starts that stay near a saddle form a thin graph

The nonlinear terms bend $E^{cs}$ into a curved set, a **center-stable manifold**. Every orbit that stays in a small ball around the saddle lies on it. It is the graph of a function over $E^{cs}$, a space of dimension at most $d - 1$, so it has zero volume. Classical versions need the GD map to be invertible near the saddle, which earlier papers get from $\mathcal{L} \in C^2$ with a globally Lipschitz gradient. The paper's version (its Theorem 2.2) drops that, at the price of a graph that is only Lipschitz, which still has zero volume.

### 3. Pulling a thin set back needs a map that does not crush volume

A trajectory can wander for many steps before it reaches the saddle's neighbourhood. To conclude that its *start* lies in a zero-volume set, the argument pulls the thin set back through $G$ again and again. That only works if $G$ never maps a set of positive volume onto a set of zero volume.

The ReLU itself shows how this can fail. The map $x \mapsto \max(x, 0)$ sends the whole half-line $(-\infty, 0]$ to the single point $0$. The preimage of the zero-length set $\{0\}$ has infinite length, and the derivative is $0$ on all of it. Classical proofs avoid this by assuming a globally Lipschitz gradient, so that a small step makes $G$ invertible everywhere. The paper needs less: a nonzero Jacobian determinant on an open set whose complement has zero volume (its Lemma 2.4).

### 4. An integral loss smooths some kinks, not all

Take one ReLU neuron with input $x$ uniform on $[0, 1]$, the target $g(x) = x$, and train only the bias $b$:

$$\mathcal{L}(b) = \int_0^1 \big(\max(x + b, 0) - x\big)^2\, dx = \begin{cases} b^2, & b \ge 0,\\ b^2 + \tfrac{2}{3}b^3, & -1 \le b \le 0,\\ \tfrac13, & b \le -1. \end{cases}$$

At $b = 0$ the kink moves through the edge of the data, and both pieces have $\mathcal{L} = 0$, $\mathcal{L}' = 0$, $\mathcal{L}'' = 2$: the integral has smoothed the kink to second order. At $b = -1$ the neuron dies on the whole interval: $\mathcal{L}'' = -2$ from the right and $0$ from the left. The loss is $C^1$ but not $C^2$ there, and for every $b < -1$ it is flat, so all those points are critical. Such dead-neuron regions are exactly where the classical hypotheses break.

:::key Putting the four ideas together
A strict saddle has an unstable direction, so the starts that stay near it lie on a thin graph (ideas 1 and 2). Making that a statement about all starts needs a GD map that does not crush volume, which ReLU kinks threaten (idea 3), and the ReLU loss is not $C^2$ everywhere (idea 4). The paper proves a center-stable manifold theorem that tolerates both, then applies it to shallow ReLU networks with affine targets, as the notes below explain.
:::
:::

:::divider Study notes
:::

## 1. Problem: what if the network is not smoothly differentiable?

Most strict-saddle avoidance theorems assume a twice-continuously differentiable objective. A ReLU network is piecewise linear in its input and generally creates a parameter-space loss that need not satisfy classical global smoothness hypotheses. The authors ask whether a suitably weakened dynamical-systems result can still explain escape near saddle points in concrete shallow ReLU-like training problems.

## 2. Mathematical setup and intuition

A shallow scalar-input network can be expressed schematically as

$$f_\theta(x) = c + \sum_{j=1}^{m} a_j\,\sigma(w_j x + b_j), \qquad \sigma(z) = \max\{0, z\},$$

with a square integral loss against an affine target $g(x)$:

$$\mathcal{L}(\theta) = \int \big(f_\theta(x) - g(x)\big)^2\, \mathrm{d}\mu(x).$$

The exact paper covers specified ReLU and leaky-ReLU parameterizations, target classes and measures; these equations illustrate the problem family rather than quote a single universal model. Since $\sigma$ has a kink at zero, the update $G(\theta) = \theta - \eta\nabla\mathcal{L}(\theta)$ may fail to satisfy regularity conditions of a standard $C^2$ stable-manifold theorem.

## 3. Paper method: weaken the stable-manifold regularity

The authors establish a version of the center-stable manifold theorem with relaxed differentiability requirements and apply it after a detailed structural classification of critical points in shallow scalar-input ReLU and leaky-ReLU training with affine targets. This allows them to prove that GD *circumvents most saddle points* in the analyzed family. They additionally establish global-minimum convergence under favorable initialization criteria expressed using an explicit threshold involving the limiting loss.

:::key Principal result
For the specified shallow ReLU-like problems, saddle avoidance can be proved without assuming that the full training objective obeys the classical smoothness regime; global convergence needs additional conditions on the initialization/limiting loss.
:::

## 4. An illustrative kink and a key subtlety

For the scalar ReLU $\sigma(z) = \max(0, z)$, the derivative is $0$ for $z < 0$ and $1$ for $z > 0$, but it is not classically defined at $z = 0$. However, integrating a composed loss over an input distribution can sometimes smooth parts of the objective: *a ReLU activation does not by itself prove that every parameter-space expected loss is everywhere nonsmooth*. That is why the paper analyzes the actual loss and its critical points rather than simply declaring all smooth arguments invalid.

## 5. What the theorem does not establish

The target functions are affine; the networks are shallow with scalar input; the loss is the square integral loss under specified conditions. Thus the result is not an unconditional statement for arbitrary deep ReLU, CNNs, Transformers, high-dimensional inputs or finite-data cross-entropy. "Most saddles" and "global minima under favorable initialization" must not be collapsed into universal global convergence. No general polynomial saddle-escape complexity is provided.

:::warn Scope
This paper closes part of the *smoothness-assumption* gap for a structured shallow ReLU class, not the entire practical-neural-network theory gap.
:::

This makes it valuable to **RQ1**, **RQ2** and especially **RQ4**.

## 6. Our notes

This is a good counterpoint to the classical strict-saddle papers: it tackles an actual ReLU-like model instead of assuming a globally smooth objective. The authors do not claim every deep ReLU network escapes all saddles. Their theoretical advance is a weaker stable-manifold tool plus a detailed analysis of a very specific shallow training problem, with an extra condition needed for global convergence.

> **Original source:** [doi.org/10.1007/s10957-024-02513-3](https://doi.org/10.1007/s10957-024-02513-3). **Verification:** open-access journal version and paper text; schematic model labelled as illustration.
