# When Bad Local Minima Really Exist

:::minute The paper in one minute
**Question:** Can even simple ReLU networks have bad local minima when a network of the same size can perfectly generate the target outputs?

**Model/Method:** A two-layer, fixed-output-weight ReLU model trained on Gaussian inputs. The authors combine gradient-descent searches with rigorous computer-assisted verification.

**Finding:** Proven spurious local minima exist for equal student/teacher width $6 \le k \le 20$, even with orthogonal teacher weights. GD often reaches these minima; adding hidden units helps markedly in the tested cases.

**Why we care:** Saddle points are not the whole story. A point can be a genuine *local minimum* with *positive* training error; width can change this optimization picture.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. With Gaussian inputs the population loss has a closed form

The loss is an expectation over $x \sim \mathcal{N}(0, I_d)$, so it involves averages of products of two ReLUs. Because a Gaussian looks the same in every direction, such an average depends only on the two weight lengths and the angle $\theta$ between them. The paper uses the formula

$$\mathbb{E}\big[[w^\top x]_+ [v^\top x]_+\big] = \frac{\|w\|\,\|v\|}{2\pi}\big(\sin\theta + (\pi - \theta)\cos\theta\big).$$

Example: for $w = v$ of length one, $\theta = 0$ and the value is $\tfrac12$ (the notes' $\mathbb{E}[x^2 \mathbf{1}_{x>0}] = \tfrac12$); for orthogonal unit vectors it is $1/(2\pi) \approx 0.159$. So $F(W)$ can be evaluated, and differentiated, exactly, without sampling.

### 2. Swapping neurons gives many global minima, and that forces non-convexity

The student's output does not change if two hidden neurons swap places. With $k = n$ there are $k!$ zero-loss settings: $720$ for $k = 6$. Example with $k = 2$, teacher $v_1 = e_1$, $v_2 = e_2$: the students $(e_1, e_2)$ and $(e_2, e_1)$ both have loss $0$, but their midpoint, both neurons at $(e_1 + e_2)/2$, has loss about $0.091$. A convex function could not do this, so the landscape is non-convex even in this tiny realizable case.

### 3. A minimum is trapped inside any sphere whose values exceed the center

A continuous function on a closed ball attains its minimum somewhere in the ball. If every point on the boundary sphere has a larger value than the center, that minimum cannot be on the boundary, so it is an interior point: a genuine **local minimum**. The notes' Taylor bound is a way to prove "the sphere is higher than the center" with only finitely many checked numbers. Example with made-up constants $\varepsilon = 0.001$, $\lambda_{\min} = 0.1$, $B = 1$: the lower bound $-\varepsilon t + \tfrac{\lambda_{\min}}{2}t^2 - \tfrac{B}{6}t^3$ is about $+5.4 \times 10^{-5}$ at $t = 0.05$, so a local minimum lies within distance $0.05$ of the candidate. At $t = 0.01$ the bound is still negative: the radius must be large enough for the quadratic term to beat the small gradient.

### 4. A frequency from 1000 runs has an error bar

The experiments count how many of $1000$ random initializations end at a spurious minimum. A fraction $\hat p$ from $N$ independent runs has standard error $\sqrt{\hat p(1 - \hat p)/N}$. For $3/1000$ that is about $0.17$ percentage points; for $346/1000$ about $1.5$; for $940/1000$ about $0.75$. The jumps in the notes' table (from $0.3\%$ to $94\%$) are far larger than these error bars, so the trend with width is real for the tested sizes.

### 5. Extra neurons add room to move

ReLU is positively homogeneous: $[\tfrac12 w^\top x]_+ = \tfrac12 [w^\top x]_+$. So one neuron $w$ can be replaced by two copies of $w/2$ with exactly the same output. With $n > k$ the student has whole families of zero-loss settings and more directions along which a stuck configuration can rearrange itself. This is one intuition for why $n = k + 1$ or $k + 2$ helps in the experiments; the paper observes the effect but does not prove it.

:::key Putting the five ideas together
The loss can be computed exactly (idea 1) and is non-convex because of neuron symmetry (idea 2). GD finds candidate points, and a sphere argument turns finitely many verified numbers into a proof that a real local minimum with positive loss sits nearby (idea 3). Counting outcomes over 1000 runs shows how often GD lands there, with error bars far smaller than the effects (idea 4), and extra neurons make such traps rarer in the tested cases (idea 5). The notes below give the theorem, the certificate and the tables.
:::
:::

:::divider Study notes
:::

## 1. Start with the network — what is being optimized?

The student and teacher compute a sum of ReLU outputs (no trained output weights or bias terms):

$$\widehat{y}(x) = \sum_{i=1}^{n}[w_i^\top x]_+, \qquad y^*(x) = \sum_{j=1}^{k}[v_j^\top x]_+, \qquad [z]_+ = \max\{0, z\}.$$

The input is $x \sim \mathcal{N}(0, I_d)$. The *population* loss is

$$F(W) = \frac{1}{2}\,\mathbb{E}_x\Big[\big(\widehat{y}(x) - y^*(x)\big)^2\Big].$$

Here $k$ is the **teacher width**, $n$ the **student width**. If $n = k$ and $w_i = v_i$ (up to permutation), then $F(W) = 0$. So the best possible loss is known exactly.

## 2. Why "spurious" is different from a saddle

A **spurious local minimum** is locally unbeatable by a sufficiently small move, yet worse than the global optimum. For this realizable loss, it has $F(W_*) > 0$ even though a zero-loss configuration exists. At a smooth *strict* local minimum, $\nabla F(W_*) = 0$ and $\lambda_{\min}(\nabla^2 F(W_*)) > 0$; there is no nearby direction of negative curvature to exploit.

:::key The contrast with Lee (2016)
Strict-saddle avoidance concerns points with *negative* Hessian curvature. It does *not* prevent convergence to the positive-curvature bad minima constructed here. This is why the two papers do not conflict.
:::

## 3. Worked example: the loss is perfectly realizable

Before tackling the proof, take a *one-dimensional teaching example* with one teacher and one student neuron: $v = 1$, $w \ge 0$, and $x \sim \mathcal{N}(0, 1)$. Their outputs are $[x]_+$ and $[wx]_+ = w[x]_+$. Therefore,

$$\begin{aligned} F(w) &= \frac{1}{2}\,\mathbb{E}\Big[\big(w[x]_+ - [x]_+\big)^2\Big] = \frac{1}{2}(w - 1)^2\,\mathbb{E}[x^2\mathbf{1}_{x>0}] \\ &= \frac{1}{4}(w - 1)^2 \qquad \Big(\text{because } \mathbb{E}[x^2\mathbf{1}_{x>0}] = \tfrac{1}{2}\Big). \end{aligned}$$

At $w = 0.6$, $F(w) = 0.04$; at $w = 1$, $F(w) = 0$. **Point:** a nonzero loss is not automatically a bad local minimum — here gradient descent can simply decrease it. The paper's counterexamples require coupled hidden neurons ($k \ge 6$ in the certified cases), where the landscape becomes much more complicated.

## 4. Theorem 1: existence of genuinely bad minima

With orthonormal teacher vectors, Gaussian inputs and $d = k$, the authors rigorously verify spurious local minima for:

- $n = k$ for every integer $k \in \{6, 7, \dots, 20\}$;
- certain $n = k + 1$ pairs: $(8, 9)$ and $(k, k + 1)$ for $10 \le k \le 19$.

**Corollary 1** extends the constructions to any input dimension $d > k$. For independent Gaussian teacher vectors, in sufficiently high dimension the conclusion holds with probability at least $1 - \exp[-\Omega(d)]$. This is a probability over the *teacher vectors*, not a statement about GD reaching minima.

## 5. How can a computer-assisted proof be rigorous?

The authors first use GD to locate a promising candidate $W_0$ with small gradient, positive Hessian curvature, and loss above zero. They then use guaranteed-precision calculations and a Taylor bound to certify a *real local minimum nearby*, not merely a numerical approximation.

$$F(W_0 + tu) \;\ge\; F(W_0) - \varepsilon t + \frac{\lambda_{\min}}{2}t^2 - \frac{B}{6}t^3, \qquad \|u\| = 1,$$

where $\|\nabla F(W_0)\| \le \varepsilon$, $\nabla^2 F(W_0) \succeq \lambda_{\min} I$ with $\lambda_{\min} > 0$, and $B$ controls the third-order remainder.

1. **Tiny linear term:** $-\varepsilon t$ permits a little improvement near the candidate because the computed gradient is not exactly zero.
2. **Positive quadratic term:** $\lambda_{\min} t^2/2$ raises the loss in every direction once $t$ is large enough.
3. **Controlled cubic term:** $-Bt^3/6$ bounds the approximation error. With a suitable radius, the sphere around $W_0$ has strictly larger loss than its center.
4. **Compactness:** the loss attains a minimum *inside* that sphere. A separate bound ensures that the enclosed minimum still has positive loss, so it is non-global.

:::key A proof certificate from the paper ($n = k = 6$)
The verified Hessian eigenvalue is at least $0.004699$ at the candidate. The argument certifies a local minimum within $1.12 \times 10^{-7}$ of it, with loss at least $0.02508$. Since a global solution has loss 0, this is an actual *spurious* local minimum, not a saddle or rounding artifact.
:::

## 6. Experiments: how often does ordinary GD reach these minima?

The authors run **1,000 independent GD initializations per configuration**, using population loss, step size 0.1, and Xavier-style Gaussian initial weights. They verify the promising non-global candidates rather than labeling a high-loss point a local minimum by appearance alone.

| Teacher $k$ | Student $n$ | Certified hits | Approx. runs | Interpretation |
|--:|--:|--:|--:|---|
| 6 | 6 | 0.3% | 3/1000 | Rare at this width |
| 10 | 10 | 34.6% | 346/1000 | Already frequent |
| 15 | 15 | 80.3% | 803/1000 | Most runs |
| 20 | 20 | 94.0% | 940/1000 | Very common |
| 15 | 16 | 10.1% | 101/1000 | One extra neuron helps |
| 19 | 20 | 49.1% | 491/1000 | One extra is not always enough |

**Read this carefully:** In the equal-width experiments ($n = k$), the observed chance of ending at a spurious minimum rose sharply with width. *For the same teacher*, giving the student extra width often helped. These are two different comparisons and should not be confused.

## 7. Does overparameterization remove the bad minima?

The authors report that, among tested configurations up to width 20, adding two or more student neurons ($n \ge k + 2$) yielded **no encountered spurious local minima**. They do *not* prove that such minima cannot exist, or that two extra neurons always suffice at larger widths.

:::key Survey-worthy contrast
Safran–Shamir show that exact-width shallow ReLU models can have genuine bad minima. Du (2019) and Allen-Zhu et al. (2019) prove favorable optimization guarantees in *sufficiently wide*, differently specified regimes. This is a change of assumptions and model, not a contradiction.
:::

## 8. Figure 1 — what it adds beyond the tables

The figure plots the empirical fraction of GD runs reaching a minimum with loss below a chosen threshold. A curve with more mass at positive losses indicates that the algorithm is more likely to terminate at suboptimal solutions. The tables quantify how often certified spurious minima were reached; the figure displays the *distribution of attained loss values* across runs.

## 9. What is proved versus only observed?

| Rigorous guarantee | Empirical observation / conjecture |
|---|---|
| Spurious minima exist for the specified $(k, n)$ pairs; they extend to higher input dimension. | Random GD often finds these minima; the frequency increases for $n = k$ in tested sizes. |
| The computer-assisted certificates bound an actual local minimum near each verified candidate. | Mild extra student width greatly reduces the observed failure frequency; absence of bad minima for $n \ge k + 2$ is *not* proved. |

## 10. Important assumptions and limitations

- **Architecture:** shallow (two-layer) ReLU network; trainable hidden weights; fixed unit output coefficients and no biases in the studied formulation.
- **Data and target:** standard Gaussian inputs; realizable teacher with orthonormal or approximately orthogonal hidden weights; *population* squared-error loss, not a finite sample.
- **Range:** rigorous constructions for selected neuron counts up to 20; the experiments test configurations only up to that size.
- **Methodological limit:** numerical searching discovers candidates, while verification proves existence. The work does not provide a closed-form classification of every critical point.
- **Scope:** there is no claim that modern deep networks generally have these same failure probabilities or that overparameterization guarantees no bad minima without additional assumptions.

## 11. Survey connections — the argument we can build

| Paper | Main focus | Connection to Safran–Shamir |
|---|---|---|
| **Dauphin (2014)** | Saddles / Hessian | Saddles are serious obstacles, but not the only obstacles. |
| **Lee (2016)** | Strict-saddle avoidance | Almost-sure avoidance does not exclude convergence to *strict bad minima*. |
| **Jin (2017)** | Perturbed GD | Approximate second-order stationarity is not equivalent to global optimality. |
| **Du (2019)** | Wide-network convergence | Stronger width / smoothness / architecture assumptions give positive convergence results. |
| **Allen-Zhu (2019)** | Wide ReLU / GD-SGD | Local gradient dominance near initialization can rule out bad stationary points along certain trajectories. |
| **Wu (2025)** | Shallow ReLU geometry | Both papers study what ReLU stationary points mean, under different architectural setups. |

:::key The narrative for the survey
"Avoiding saddles" is only part of the story. Some shallow ReLU losses contain isolated, positive-curvature *non-global* minima. Thus, convergence to second-order stationarity alone cannot ensure a global solution; architectural and width assumptions can change which landscape features obstruct training.
:::

## 12. Quick reference

| Term | Meaning to remember |
|---|---|
| Spurious local min. | Locally optimal but $F(W_*) > \inf_W F(W) = 0$ in this realizable model. |
| Population loss | $F(W) = \frac{1}{2}\mathbb{E}[(\widehat{y} - y^*)^2]$ over $x \sim \mathcal{N}(0, I_d)$. |
| Teacher / student width | $k$ target neurons; $n$ trained neurons; $n > k$ means overparameterized student. |
| Certificate | Small $\lVert\nabla F\rVert$, positive $\lambda_{\min}(H)$, bounded third derivatives, positive loss. |
| Core result | Theorem 1 + Corollary 1: bad minima exist in particular simple Gaussian/ReLU regimes. |

## 13. Papers Master note (copy into the purple column)

:::plain
Safran–Shamir (2018) rigorously certify positive-loss strict local minima in realizable two-layer ReLU population-risk problems, for specified teacher/student widths up to 20. In 1,000-run GD experiments, equal-width models increasingly reach these minima as width grows; small student overparameterization substantially reduces such failures in tested configurations, but absence of bad minima with extra width is not proved. This clarifies why strict-saddle avoidance alone does not guarantee global optimization.
:::

> **Source:** Safran and Shamir, ICML 2018, Sections 1–4, Theorem 1, Corollary 1, Tables 1–2. [Original paper](https://proceedings.mlr.press/v80/safran18a/safran18a.pdf). Explanatory one-neuron calculation is pedagogical, not a construction from the paper.
