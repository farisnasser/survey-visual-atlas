# Global Convergence Using Optimal Transport

:::minute The paper in one minute
**Problem.** In a wide two-layer network, output is a sum of many nonlinear neuron features. Optimizing finite neuron locations and amplitudes is non-convex.

**Idea.** Chizat and Bach reformulate the task in terms of an *unknown measure*: once the distribution of neurons is the optimization variable, the objective can become convex even while its finite-particle parameterization remains non-convex.

**Actual contribution.** Under specified regularity, homogeneity and initialization/support assumptions, if the appropriate infinite-particle gradient flow converges, its limit is a global minimizer of the convex-in-measure objective. The paper also reports numerical evidence that the many-particle picture is informative at moderate finite width.

**Limits.** The guarantee concerns the relevant *continuous-time many-particle limit* and its hypotheses. It is not a blanket theorem that arbitrary finite ReLU/Transformer networks trained by Adam at large learning rates reach global minima.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. A loss can be non-convex in the weights and convex in another variable

Take $L(a, b) = (ab - 1)^2$. At $(a, b) = (0, 0)$ its Hessian is $\begin{pmatrix} 0 & -2 \\ -2 & 0 \end{pmatrix}$, with eigenvalues $-2$ and $+2$, so $L$ is not convex in $(a, b)$. Yet $L$ depends on the weights only through the product $p = ab$, and $(p - 1)^2$ is a convex function of $p$. The non-convexity comes from how the weights build the quantity that matters, not from the loss itself. The paper finds such a variable for a whole network: the distribution of its neurons.

### 2. Averaging networks is a convex operation

A two-layer network with $m$ neurons computes $f(x) = \frac1m\sum_{j=1}^m a_j\,\sigma(\theta_j \cdot x)$: a **measure** $\mu$ that puts weight $a_j/m$ at each neuron position $\theta_j$, integrated against the feature $\phi(\theta) = \sigma(\theta \cdot x)$. The predictor $\int \phi\, d\mu$ is **linear** in $\mu$: mixing two measures half and half mixes their predictors half and half. A convex loss $R$ of a linear map is convex. With $R(f) = (f - 1)^2$, two networks predicting $0$ and $2$ each have loss $1$, but their half-half mixture predicts $1$ and has loss $0 \le \tfrac12 \cdot 1 + \tfrac12 \cdot 1$. In the variable $\mu$ there are no bad local minima to get stuck in.

### 3. Gradient descent moves a cloud of particles

Think of each neuron $(a_j, \theta_j)$ as a **particle**. Gradient descent on the $m$ particles moves all of them at once, each by a velocity that depends on its own position and on the current predictor. With the $\frac1m$ scaling, duplicating every neuron leaves the predictor unchanged, and each copy then moves exactly like the original (with the step size scaled by $m$, as in the paper's normalization). The motion depends on the distribution of the particles, not on how many there are. As $m \to \infty$ the cloud becomes a continuous distribution that flows: the **Wasserstein gradient flow** of the notes.

:::key Putting the three ideas together
The network objective is non-convex in the individual neurons but convex in their distribution (ideas 1 and 2), and gradient descent on many neurons is a flow of that distribution (idea 3). The paper shows that, with homogeneous features and an initialization whose support covers all directions, a limit of this flow, if it exists, is a global minimizer. The notes below give the formulation and its limits.
:::
:::

:::divider Study notes
:::

## 1. The problem: why can non-convex particle training find a global solution?

In a wide two-layer network, output is a sum of many nonlinear neuron features. Optimizing finite neuron locations and amplitudes is non-convex. Chizat and Bach reformulate the task in terms of an *unknown measure*: once the distribution of neurons is the optimization variable, the objective can become convex even while its finite-particle parameterization remains non-convex.

## 2. The exact modeling idea

Their broad formulation is a convex loss of a feature mixture plus a convex measure regularizer,

$$J(\mu) = R\left(\int_\Theta \phi(\theta)\, d\mu(\theta)\right) + G(\mu).$$

$\Theta$ is the neuron/particle parameter space; $\phi(\theta)$ is the feature represented by a neuron; $\mu$ is a (possibly signed) measure over neurons; and $R$ is convex in the *resulting predictor*. If $\mu$ is represented by finitely many particles, then

$$\mu_m = \sum_{j=1}^m a_j \delta_{\theta_j}, \qquad \int \phi\, d\mu_m = \sum_{j=1}^m a_j \phi(\theta_j).$$

This recovers finite-width feature sums, such as single-hidden-layer network models, but the parameters $\{a_j, \theta_j\}$ still enter nonlinearly.

## 3. Why optimal transport appears

The method follows the continuous-time evolution of particle masses and positions. In the many-particle limit, those trajectories are described by a *Wasserstein gradient flow*, an evolution of a distribution rather than a single vector. A schematic continuity equation is

$$\partial_t \mu_t + \nabla \cdot (\mu_t v_t) = 0, \quad v_t \propto -\nabla_\theta \frac{\delta J}{\delta \mu}(\theta),$$

where $\delta J/\delta\mu$ is a first variation of the objective. This is explanatory notation: the precise formulation depends on the measure model, homogeneity and regularizer used in the article.

:::theorem Actual contribution
Under specified regularity, homogeneity and initialization/support assumptions, if the appropriate infinite-particle gradient flow converges, its limit is a global minimizer of the convex-in-measure objective. The paper also reports numerical evidence that the many-particle picture is informative at moderate finite width.
:::

## 4. Why this is not just the neural tangent kernel

NTK analysis often keeps features nearly frozen around random initialization. In mean-field/transport regimes, particles can move nontrivially in parameter space, thereby changing the learned feature distribution. Thus "wide network" alone is insufficient to identify which theory applies: the limiting scaling and initialization matter.

## 5. Assumptions, qualifications and implications

The guarantee concerns the relevant *continuous-time many-particle limit* and its hypotheses. It is not a blanket theorem that arbitrary finite ReLU/Transformer networks trained by Adam at large learning rates reach global minima. Nor does convexity in $\mu$ by itself make the finite particle objective convex in $(a_j, \theta_j)$.

:::plain Survey relevance (RQ4)
This is a second route to favorable optimization, distinct from explicit saddle escape. Its price is a measure-valued limit plus support/initialization assumptions. Contrast with lazy NTK convergence and with saddle-to-saddle dynamics at small initialization.
:::

## 6. Our notes

The useful insight is that a neural network can be non-convex in the individual neurons but convex when the whole neuron distribution is the variable. The authors show global behavior in an idealized infinitely wide particle flow; they do not say that every practical finite network behaves like this.

> **Original source:** [https://papers.neurips.cc/paper_files/paper/2018/hash/a1afc58c6ca9540d057299ec3016d726-Abstract.html](https://papers.neurips.cc/paper_files/paper/2018/hash/a1afc58c6ca9540d057299ec3016d726-Abstract.html).
