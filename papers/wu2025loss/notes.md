# Loss Landscape of Shallow ReLU-like Neural Networks

:::minute The paper in one minute
**Question:** ReLU-like activations are non-differentiable at the origin, so their empirical squared-loss landscapes contain boundaries at which ordinary gradients do not exist. The authors ask which stationary points actually behave like minima, which admit an escape direction, and what happens when a narrow network is embedded inside a wider one.

**Paper result:** Escape-neuron conditions distinguish saddle-like stationary configurations from local minima in one-hidden-layer ReLU-like models despite non-smooth activation boundaries.

**Do not conflate:** directional stationary points, Clarke stationary points, strict Hessian saddles and escape-neuron configurations are different objects. A nonsmooth landscape may not even have a Hessian at the relevant point.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. At a kink, use one-sided slopes

Where a function has a corner, there is no gradient, but every direction $d$ still has a **one-sided directional derivative**

$$L'(\theta; d) = \lim_{\alpha \downarrow 0} \frac{L(\theta + \alpha d) - L(\theta)}{\alpha} .$$

For $f(x) = |x|$ at $0$ it is $|d|$: positive both ways. For $f(x) = -|x|$ it is $-|d|$: negative both ways. The paper calls a point **stationary** when $L'(\theta; d) \ge 0$ for *every* direction: no first-order way down. Where $L$ is smooth this is the usual $\nabla L = 0$.

### 2. Clarke stationarity is a weaker test

Another common definition, **Clarke stationarity**, asks whether $0$ lies in the convex hull of the nearby gradients. Both $|x|$ and $-|x|$ pass it at $0$: the nearby slopes are $\pm 1$, and their hull $[-1, 1]$ contains $0$. Yet $-|x|$ has a clear way down. The paper uses this exact example to explain why it works with directional stationarity: only the origin of $|x|$ stalls gradient descent.

### 3. First-order flat does not mean minimum: look one order higher

Take one neuron, one data point $x = 1$, target $1$: $L(a, w) = \tfrac12\big(a\,\mathrm{ReLU}(w) - 1\big)^2$. At $(a, w) = (0, 0)$ every one-sided slope is $0$, so the point is stationary. But moving both weights together, $a = w = \varepsilon > 0$, gives $L = \tfrac12(\varepsilon^2 - 1)^2 \approx \tfrac12 - \varepsilon^2$: at $\varepsilon = 0.1$ the loss is $0.49005 < 0.5$. The decrease is second order, $C\varepsilon^2$ with $C < 0$, the same shape as the paper's Corollary 4.4. A neuron whose joint growth lowers the loss like this is the idea behind an **escape neuron**. By contrast, for $w < 0$ the neuron is **dead**: it outputs $0$ on the data and no small change of $a$ or $w$ alone moves the loss.

### 4. Gradient flow conserves a balance between layers

For a ReLU neuron, $w\,\partial_w L = a\,\partial_a L$ (both equal the residual times $a\,\mathrm{ReLU}(w^\top x)$, by positive homogeneity). Under gradient flow, $\theta' = -\nabla L$, this gives

$$\frac{d}{dt}\big(\|w\|^2 - a^2\big) = -2w^\top\partial_w L + 2a\,\partial_a L = 0 .$$

So $\|w\|^2 - a^2$ never changes: this is the paper's Fact 4.5. Starting from a tiny initialization, $\|w\| = |a|$ for the whole run, so a neuron with small input weights also has a small output weight. That is why, near a saddle reached from small initialization, the neurons that can escape are the small but living ones.

### 5. Small initialization turns training into plateaus and drops

Near a saddle, a small escaping component grows roughly geometrically, by a factor $1 + \eta\lambda$ per step for some growth rate $\lambda$. Starting from size $\delta$ it needs about $\ln(c/\delta)/\ln(1 + \eta\lambda)$ steps to reach size $c$. Example: $\delta = 5 \times 10^{-6}$ (the paper's initialization scale), $c = 0.1$, $\eta\lambda = 0.001$: about $9900$ steps of almost no visible progress, then a quick drop. The loss curve becomes a staircase, **saddle-to-saddle** training. (A linearized estimate; the paper's experiment shows the staircase, not this formula.)

:::key Putting the five ideas together
At ReLU kinks, stationarity must be defined with one-sided slopes (idea 1), which is stricter than the Clarke test (idea 2). A stationary point can still be escaped at second order by growing a neuron (idea 3), and the paper's escape-neuron criterion detects exactly this. The conservation law (idea 4) says which neurons can do the escaping after small initialization, and the escapes appear as drops between plateaus (idea 5). The notes below set up the model and state the results.
:::
:::

:::divider Study notes
:::

## 1. The question and the obstacle

In a smooth objective, $\nabla L(\theta) = 0$ is the natural definition of stationarity. ReLU-like activations are non-differentiable at the origin, so their empirical squared-loss landscapes contain boundaries at which ordinary gradients do not exist. The authors ask which stationary points actually behave like minima, which admit an escape direction, and what happens when a narrow network is embedded inside a wider one. They analyze a *one-hidden-layer* model, not arbitrary deep networks.

## 2. First-principles mathematical setup

A representative scalar-output two-layer ReLU-like network has the form

$$f_\theta(x) = \sum_{j=1}^{m} a_j\,\sigma(w_j^\top x), \qquad L(\theta) = \frac{1}{2n}\sum_{i=1}^{n}\big(f_\theta(x_i) - y_i\big)^2.$$

For ReLU, $\sigma(z) = \max\{0, z\}$. Away from a kink, differentiation through a unit gives

$$\frac{\partial L}{\partial w_j} = \frac{1}{n}\sum_{i=1}^{n}\big(f_\theta(x_i) - y_i\big)\,a_j\,\mathbf{1}\{w_j^\top x_i > 0\}\,x_i.$$

But at $w_j^\top x_i = 0$, the indicator can change discontinuously. The paper uses *directional stationarity* to work with these boundaries, rather than treating Clarke stationarity as interchangeable. The formula above is an explanatory smooth-region calculation, not the authors' full directional-stationarity definition.

## 3. What is an escape neuron?

A unit can have a parameter direction that would improve the squared loss even when the network is at a directional stationary point according to the paper's conditions. Such a unit is called an *escape neuron*; the authors characterize it with first-order criteria tailored to the activation's piecewise structure. Their central conclusions are logically asymmetric:

- At a stationary point *without* escape neurons, the point is a local minimum.
- For *scalar-output* networks, the presence of an escape neuron certifies that the stationary point is *not* a local minimum.

These results are not a claim that all local minima are global. Nor does identifying an escape neuron alone prove that plain gradient descent will escape within a particular iteration complexity.

:::key Paper result
Escape-neuron conditions distinguish saddle-like stationary configurations from local minima in one-hidden-layer ReLU-like models despite non-smooth activation boundaries.
:::

## 4. Network embedding and saddle-to-saddle training

Embedding takes a narrower network and represents the same function with a wider architecture. For example, in a positively homogeneous model, splitting one neuron into two equal-output copies can preserve the prediction:

$$a\,\sigma(w^\top x) = \frac{a}{2}\sigma(w^\top x) + \frac{a}{2}\sigma(w^\top x).$$

This algebra is a teaching example; the paper develops a more complete account of how stationary configurations behave under embedding. It also narrows the types of saddle escape compatible with training from *vanishing initialization*. Its achievement is a more precise description of possible training paths, not a general guarantee of global convergence.

## 5. Assumptions, limitations, relevance

The model has one hidden layer, ReLU-like activations and empirical squared loss; the scalar-output qualifier matters for one direction of the escape-neuron result. The analysis does not cover arbitrary Transformer objectives or provide a universal complexity rate for escaping non-smooth saddles. This is important for **RQ1** (critical-point geometry), **RQ2** (saddle-to-saddle dynamics), and **RQ4** (the gap between smooth theory and ReLU networks).

:::warn Do not conflate
directional stationary points, Clarke stationary points, strict Hessian saddles and escape-neuron configurations are different objects. A nonsmooth landscape may not even have a Hessian at the relevant point.
:::

## 6. Our notes

The main thing here is that ReLU kinks force us to be more careful about what "stationary" means. The escape-neuron idea gives a concrete way to tell whether a shallow network has a way to improve its loss, and embedding explains why the same learned function can look different in a wider network. It helps us discuss real activation functions without pretending the smooth strict-saddle theory applies directly. **Original source:** [https://openreview.net/pdf?id=ogKE7LcvW6](https://openreview.net/pdf?id=ogKE7LcvW6). **Verification:** original ICLR 2025 PDF and abstract checked; examples are marked as teaching derivations.

> **Primary source:** F. Z. Wu, B. Şimşek & F. G. Ged (2025), *Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escape, and Network Embedding*, ICLR 2025. [openreview.net/pdf?id=ogKE7LcvW6](https://openreview.net/pdf?id=ogKE7LcvW6).
