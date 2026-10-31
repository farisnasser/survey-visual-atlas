# Why Networks Learn One Unit at a Time

:::minute The paper in one minute
**Question:** Why do networks often learn simple functions first, pausing at plateaus before suddenly fitting more complex patterns?

**Model/Method:** Analyze fixed points, invariant manifolds and gradient-flow dynamics across fully connected, convolutional and attention-based models.

**Finding:** Training can move between nested saddles, increasing the number of *effective* neurons, kernels or attention heads. Linear models develop low-rank structure from data-dependent growth rates; quadratic-weight models develop sparse units from initialization-dependent growth rates.

**Why we care:** Plateaus can mark successive stages of representation learning rather than merely failed optimization.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Gradient flow is gradient descent with an infinitely small step

The paper analyzes **gradient flow**, the differential equation $\dot\theta = -\nabla L(\theta)$, where $\dot\theta$ is the time derivative. It is the limit of GD $\theta_{k+1} = \theta_k - \eta\nabla L(\theta_k)$ as $\eta \to 0$, with time $t = \eta k$. On $L = \theta^2/2$, GD gives $\theta_k = (1 - \eta)^k\theta_0$ and gradient flow gives $\theta(t) = e^{-t}\theta_0$. With $\eta = 0.01$ and $k = 100$ (so $t = 1$), the two factors are $0.366$ and $0.368$. Continuous time makes exact statements possible, such as "this set is never left".

### 2. A network's weights are a sum of units, and identical units count once

A two-layer linear network computes $W = \sum_i v_i u_i^\top$, a sum of one rank-one term per hidden unit. If two units share the same input weights, $u_1 = u_2 = u$, their terms merge: $v_1 u^\top + v_2 u^\top = (v_1 + v_2)u^\top$. Two units then do the work of one. The **effective width** of a network is the smallest number of units that could express its current function, which can be far below the number it has.

### 3. Growth from a tiny start takes time proportional to $\log(1/\varepsilon)$

Near zero, many weights grow like $\varepsilon e^{st}$, where $\varepsilon$ is the initial scale and $s$ a growth rate. Reaching size $1$ takes time $\log(1/\varepsilon)/s$: about $6.9$ for $\varepsilon = 10^{-3}$, $s = 1$, and about $13.8$ for $\varepsilon = 10^{-6}$. Two directions with rates $s_1 > s_2$ finish at different times, and the smaller the initialization, the longer the wait and the clearer the separation.

### 4. A plateau is time spent near a saddle

Take one mode of the loss, $L(a) = \tfrac12(s - a^2)^2$, with $s = 1$. The point $a = 0$ is a saddle-like flat point with loss $0.5$. Gradient flow is $\dot a = 2a(1 - a^2)$; started at $a = 0.01$, the loss is still $0.462$ at $t = 1.5$, crosses the halfway point near $t = 2.3$, and is $0.0017$ by $t = 3$. Most of the time goes into leaving the saddle, which shows up as a flat stretch of the loss curve followed by a sudden drop.

:::key Putting the four ideas together
Under gradient flow (idea 1), units that start alike stay alike, so a wide network can behave like a narrow one (idea 2). Directions or units grow at different rates from a small start (idea 3), so they leave the saddle at zero one at a time, each departure a sudden drop after a plateau (idea 4). The notes below show which of these steps the paper proves and for which architectures.
:::
:::

:::divider Study notes
:::

## 1. First understand the strange observation: a large network can behave like a small one

Suppose a trained width-10 network uses only one effective neuron. It may be capable of fitting a more complex function, but its current weights implement a function expressible with a width-1 network. The authors call this *simplicity*: the smallest number of active/effective units required to express the current input–output map, *not* low test error or a small parameter count.

:::figure
<svg viewBox="0 0 420 92" role="img" aria-label="Schematic: a box 'Near a width-1 saddle, one effective unit' with the label 'Loss plateau', an arrow to a box 'Near a width-2 saddle, two effective units' with the label 'Rapid loss decrease, then plateau'.">
<defs><marker id="ntZhAr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#8e99bd"/></marker></defs>
<rect x="40" y="8" width="132" height="44" rx="7" fill="rgba(124,196,255,0.08)" stroke="#7cc4ff" stroke-width="1.2"/>
<text x="106" y="27" fill="#e8ecf8" font-size="9" text-anchor="middle">Near a width-1 saddle</text>
<text x="106" y="42" fill="#8e99bd" font-size="8" text-anchor="middle">one effective unit</text>
<line x1="172" y1="30" x2="246" y2="30" stroke="#8e99bd" stroke-width="1.3" marker-end="url(#ntZhAr)"/>
<rect x="248" y="8" width="132" height="44" rx="7" fill="rgba(255,93,115,0.06)" stroke="#ff5d73" stroke-width="1.2"/>
<text x="314" y="27" fill="#e8ecf8" font-size="9" text-anchor="middle">Near a width-2 saddle</text>
<text x="314" y="42" fill="#8e99bd" font-size="8" text-anchor="middle">two effective units</text>
<text x="106" y="72" fill="#8e99bd" font-size="8" text-anchor="middle">Loss plateau</text>
<text x="314" y="72" fill="#8e99bd" font-size="8" text-anchor="middle">Rapid loss decrease, then plateau</text>
</svg>
:::

The sequence can continue: each stage often adds one or a few effective units. The schematic illustrates the mechanism, not a universal training trajectory. See the paper's **Figure 1**, pp. 2–3.

## 2. A worked example: how a width-1 solution sits inside width-2

For a scalar-output two-layer model, write

$$f(x) = v_1\phi(x; u_1) + v_2\phi(x; u_2).$$

Start with a stationary *width-1* solution $f_*(x) = v\phi(x; u)$. Create a wider representation by duplicating its unit:

$$u_1 = u_2 = u, \qquad v_1 = v_2 = \frac{v}{2}.$$

Then $f(x) = \frac{v}{2}\phi(x; u) + \frac{v}{2}\phi(x; u) = f_*(x)$: the network is wider, but the represented function is unchanged. Because the old unit had zero derivatives and the new units share or scale those derivatives, the duplicated configuration is also stationary under the paper's setup.

:::key Why this example matters
A stationary point of a narrower model can reappear in a larger model. That is the **embedded fixed-point** principle (Theorem 1). But an embedded fixed point is *not automatically a saddle* in every architecture; it may instead be a local minimum.
:::

## 3. Geometry: embedded fixed points versus invariant manifolds

**Theorem 1 (embedded fixed points).** For the network class in Eq. (1), any fixed point of a width-$(H-1)$ model can be represented as a fixed point of a width-$H$ model with the same input–output map. Corollary 2 extends this to additional units and layers. Different constructions require different properties of the activation (zero outputs, homogeneity or linearity).

**Theorem 3 (invariant manifolds).** Certain relations between units are preserved by *gradient flow*. For example, if two units start with identical parameter vectors $\theta_1 = \theta_2$, then their gradients are identical, so

$$\frac{d}{dt}(\theta_1 - \theta_2) = -\nabla_{\theta_1}L + \nabla_{\theta_2}L = 0.$$

Thus identical units remain identical at all later times. A wider network restricted to that manifold behaves like a narrower network. For certain activations, zero, proportional or linearly dependent units also form such manifolds.

:::key Difference to memorize
A **fixed point** is a single stationary configuration ($\dot\theta = 0$). An **invariant manifold** is a whole set of configurations that gradient flow cannot leave once it enters. The paper's story uses both: saddles define the stages; approximately invariant low-width paths help explain the transitions.
:::

## 4. Where the mathematics becomes dynamical: linear two-layer networks

The network's effective linear map is

$$f(x) = \sum_{i=1}^H v_iu_i^\top z(x) = Wz(x), \qquad W = \sum_i v_iu_i^\top.$$

Under squared-error gradient flow (paper's Eq. 9),

$$\dot v_i = (\Sigma_{yz} - W\Sigma_{zz})u_i, \qquad \dot u_i = (\Sigma_{yz} - W\Sigma_{zz})^\top v_i.$$

When every weight is initially tiny, $W = O(\varepsilon^2)$; the residual terms involving $W$ can initially be neglected. The resulting linearized dynamics grow along singular directions of $\Sigma_{yz}$ at rates determined by its singular values $s_k$.

## 5. Worked numbers: why data modes learn at different speeds

In an illustrative two-mode case, let $s_1 = 3$, $s_2 = 1$ and suppose each unstable growing-mode amplitude begins at $0.01$. The early linearized dynamics give

$$a_1(t) \approx 0.01e^{3t}, \qquad a_2(t) \approx 0.01e^{t}.$$

The first mode reaches unit size when $t = \log(100)/3 \approx 1.535$. At that same time,

$$a_2(t) \approx 0.01e^{1.535} \approx 0.0464.$$

So one direction has become large while the other remains small. The network's weights are *approximately low-rank* and it behaves like an effectively narrower model.

:::theorem What Theorem 4 proves beyond this toy calculation
If the top singular value has multiplicity $r < D$ and the initialization is independent small Gaussian noise, then when the top-$r$ component becomes $O(1)$, the other component is $O(\varepsilon^{1 - s_{r+1}/s_1})$ almost surely for the *linearized* system. Distinct singular values produce a separation of learning times. This is not a theorem for every nonlinear network.
:::

## 6. The second mechanism: initialization makes one unit win first

The linear case separates *data directions*: a large singular value learns first. In the paper's quadratic-weight case (including its linear self-attention construction), the early growth can separate *individual units*. Proposition 5 formalizes this under specified symmetry/eigenvalue and small-Gaussian-initialization conditions.

To see the intuition, the authors consider the simplified scalar differential equation $\dot v_i = v_i^2$. Separate variables:

$$\frac{dv_i}{v_i^2} = dt \quad\Longrightarrow\quad v_i(t) = \frac{1}{1/v_i(0) - t}.$$

Imagine $v_1(0) = 0.10$ and $v_2(0) = 0.05$. By $t = 9$,

$$v_1(9) = \frac{1}{10 - 9} = 1, \qquad v_2(9) = \frac{1}{20 - 9} \approx 0.091.$$

The slightly larger unit races ahead while the other stays relatively small. Unlike the exponential mode story, this is a *rich-get-richer* mechanism driven by initial differences between units. The scalar example is pedagogical; the paper's exact quadratic-network equations and Proposition 5 are more involved.

:::warn Do not mix the two mechanisms
**Linear-weight case:** different data singular values $\to$ different growth rates across directions $	o$ low-rank structure.

**Quadratic-weight case:** different initial unit magnitudes $\to$ different growth rates across units $\to$ sparse use of units.
:::

## 7. What Figure 1 and Figure 2 actually demonstrate

| Figure / test | Correct interpretation |
|---|---|
| Fig. 1A | Cartoon: near an embedded saddle the loss plateaus; a change of effective-width manifold allows another stage. |
| Fig. 1B–G | Examples across linear fully connected/convolutional, ReLU fully connected/convolutional, linear self-attention and quadratic networks. The illustrated transition goes from one to two effective units. |
| Fig. 2A | Increasing width barely changes the stage timing in the tested linear case, but can shorten plateaus for the tested quadratic / attention case. |
| Fig. 2B | Closing the data singular-value gaps shortens or eliminates intermediate plateaus in linear networks; in quadratic models the initialization mechanism can preserve them. |
| Fig. 2C–D | Starting near a low-width manifold, and the scale of initialization, influence whether/how prominently stages appear. Larger random initialization weakens plateaus. |

## 8. Main contributions worth remembering

1. **A cross-architecture geometric framework:** nested fixed points of narrower subnetworks and invariant manifolds of lower effective width.
2. **A mechanism for simplicity bias:** stage-like learning arises when trajectories remain near simple-function manifolds and intermittently recruit extra effective units.
3. **Two explanations of timescale separation:** data-induced rank bias versus initialization-induced sparsity.
4. **Testable predictions:** data spectrum, architecture width and initialization control plateau duration and occurrence, with supporting numerical experiments.

## 9. Important assumptions and limitations

| Aspect | What you must qualify in the survey |
|---|---|
| Training rule | Formal invariance is analyzed for continuous-time **gradient flow**, a small-step limit of GD, not general minibatch SGD. |
| Architecture | The general Eq. (1) has additive unit structure; different fixed-point constructions require specific activation properties. |
| Dynamics | Detailed growth-rate results in Section 5 focus on **two-layer** linear- or quadratic-in-weights models with squared-error loss and small random initialization. |
| Saddle geometry | Embedded *fixed points* exist generally; their saddle character and attraction of trajectories need further model-dependent conditions. |
| Generality | A full dynamical theory for arbitrary deep models/transformers is **not** proved; some extensions are explicitly conjectural. |
| Not inevitable | The paper discusses tanh and large-initialization settings in which saddle-to-saddle stages may be absent. |

:::warn One subtlety for a good survey
Theorem 1/Corollary 2 and Theorem 3 are general *structural existence/invariance* claims for the stated architecture class. The claim that training *typically follows* these low-dimensional manifolds is developed using specific dynamical mechanisms and experiments. Never present the latter as a universal convergence theorem.
:::

## 10. Connections to the papers we have already read

| Earlier paper | How Zhang's perspective differs |
|---|---|
| Saxe (2014) | Stage-like mode learning in deep linear nets; Zhang unifies it with effective-unit recruitment. |
| Dauphin (2014) | Saddle plateaus as obstacles; Zhang adds their role in progressively learned representations. |
| Lee (2016) | Avoiding *convergence* to saddles differs from repeatedly passing *near* saddles. |
| Jin (2017) | Perturbations provide escape bounds; Zhang studies natural structured transitions without intervention. |
| Li (2018) | Architecture shapes the visible loss landscape; Zhang explains architecture-dependent trajectories. |
| Du (2019) | Width can enable successful optimization; Zhang studies which units become effectively active. |
| Galli (2026) | Large steps and flatness change dynamics; Zhang instead studies progressive complexity. |

## 11. Quick reference and reading route

| Term / item | Remember this |
|---|---|
| Fixed point | $\dot\theta = -\nabla L(\theta) = 0$; stationary, not necessarily globally optimal. |
| Invariant manifold | A set preserved by gradient flow (e.g., equal units remain equal). |
| Effective width | Minimum number of units needed to express the current learned function. |
| Theorems 1 and 3 | Narrow fixed points embed into wider networks; some simple-weight relations remain invariant. |
| Theorem 4 / Prop. 5 | Different data directions grow at different rates / different units grow at different rates. |
| Read first | Abstract and Figure 1; Sections §3–§6; discussion §7. Use Appendices E–H only to verify proof steps. |

:::plain Papers Master — 2-line note
Zhang et al. (2026) explain progressive complexity learning through embedded stationary points and approximately low-width invariant manifolds. Linear models show data-induced rank growth; quadratic-weight models show initialization-induced unit sparsity. Detailed trajectory guarantees require restricted settings.
:::

> **Original paper:** [ICLR 2026 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5e96d63b20f335f57999a1aca7fa0474-Paper-Conference.pdf). Worked examples are teaching adaptations, not quoted numerical experiments.
