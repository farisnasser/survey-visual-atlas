# Why a Rare Detour Can Dominate Global Optimization Time

:::minute The paper in one minute
**Question:** How long does constant-step SGD take to enter a neighborhood of a global minimizer in a nonconvex landscape?

**Model/Method:** A large-deviation action defines transition costs between critical components; directed forests and prunings determine an energy exponent.

**Finding:** The paper reports exponential-scale expected hitting-time bounds of the form $\exp[(E \pm a)/\eta]$. Matching lower bounds require sufficiently strong target attraction.

**Why we care:** The mean time can be controlled by a rare detour into a deep trap. This differs from efficient saddle escape; an appendix proof step is explicitly flagged in these notes.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Waiting for a rare event takes about one over its probability

If an event happens at each step independently with probability $p$, the number of steps until it first happens is geometric, with mean $1/p$. When $p = e^{-c/\eta}$, the mean wait is $e^{c/\eta}$, and $\eta \log(\text{mean wait}) = c$ exactly. For $c = 2$ and $\eta = 0.5$ that is $e^4 \approx 54.6$ steps; halving $\eta$ squares it. This is why the paper's times read $\exp(\text{cost}/\eta)$, and why it reports the cost, the **exponent**, rather than the time itself.

### 2. Large deviations: rare events have an exponential cost

A **large-deviation** estimate says how fast the probability of an atypical outcome shrinks: like $e^{-n I}$, with a rate $I$ set by the outcome. Toss a fair coin $n$ times and ask for at least 75% heads. The rate is $I = 0.75\log 1.5 + 0.25\log 0.5 \approx 0.131$. The exact probabilities give $-\tfrac1n \log P \approx 0.151$, $0.138$, $0.133$ for $n = 100$, $400$, $1600$: the exponent settles at $I$, while the probability itself becomes astronomically small.

For SGD, $1/\eta$ plays the role of $n$: with a small step, many small noisy steps must conspire for the iterate to go against the gradient. The **action** of a path is its rate, and the cheapest path between two regions sets the transition cost.

### 3. Deep traps cost exponentially more time

Leaving a basin means climbing out of it against the drift. With Gaussian gradient noise of variance $\sigma^2$ per coordinate, the notes (§7) show the cheapest climb of height $\Delta f$ costs $2\Delta f/\sigma^2$. The waiting time is then about $\exp\big(2\Delta f/(\sigma^2\eta)\big)$. Doubling the depth from $1$ to $2$ with $\sigma^2 = 1$ and $\eta = 0.25$ multiplies the wait by $e^{8} \approx 2981$.

### 4. A mean can be ruled by rare, long runs

Suppose a run takes $10$ steps with probability $0.99$ and $10^6$ steps with probability $0.01$. The typical run takes $10$ steps; the mean is $0.99 \cdot 10 + 0.01 \cdot 10^6 \approx 10{,}010$. An expected hitting time $\mathbb{E}_x[\tau]$ is a mean of this kind, so one unlikely detour into a deep trap can decide its exponent.

:::key Putting the four ideas together
Each transition between critical regions of the landscape is a rare event with a large-deviation cost (idea 2), and waiting for it takes about $\exp(\text{cost}/\eta)$ steps (idea 1). Deep spurious minima are expensive to leave (idea 3), and the mean time picks up even unlikely visits to them (idea 4). The paper's forest-and-pruning energy adds up these costs over the whole graph of transitions, as the notes below explain.
:::
:::

:::divider Study notes
:::

## 1. Reaching the global target is a different question

Finding a small gradient or escaping negative curvature does not ensure that a better basin has been reached. Let $Q = \arg\min f$ and define $\tau_\delta = \inf\{n \ge 0 : \mathrm{dist}(x_n, Q) \le \delta\}$. The paper asks about $\mathbb{E}_x[\tau_\delta]$, the mean first time constant-step SGD gets close to $Q$ from $x$. It does not claim that every later iterate stays at a global minimum or that $Q$ can be cheaply recognized in practice.

$$x_{n+1} = x_n - \eta\, G(x_n; \omega_n), \qquad G(x; \omega) = \nabla f(x) + Z(x; \omega).$$

Noise seeds are independent across iterations. The blanket objective assumptions are $C^2$ regularity, coercive loss and gradient norm at infinity, a globally Lipschitz gradient, and finitely many suitably connected critical components. Noise is unbiased, has positive-definite covariance, smooth controlled growth and a uniform sub-Gaussian bound. Finite-action transitions must connect the components. These are not automatic for arbitrary unregularized ReLU networks or every minibatch oracle.

## 2. A cost for unlikely trajectories

The Hamiltonian $H(x, p) = \log\mathbb{E}\exp(-p \cdot G(x; \omega))$ encodes the full local noise distribution. Its convex conjugate $L(x, v)$ measures the cost of realizing velocity $v$. Integrating $L$ along a path $\gamma$ gives its action $S[\gamma]$. Downhill gradient flow has zero action; forcing a path against the drift generally costs positive action. Transition probabilities then have an exponential scale $\exp(-\text{cost}/\eta)$. This is a small-step asymptotic description, not an exact formula at any chosen learning rate.

## 3. Worked example: enumerate the graph, not just the shortest route

*Pedagogical illustration, constructed for these notes.*

Consider a pedagogical Markov model with non-target states $A, B$ and absorbing target $G$. Assign direct transition costs $A \to G: 2$, $A \to B: 3$, $B \to A: 5$, $B \to G: 8$. A cost $c$ means probability $\exp(-c/\eta)$; the remaining probability is a self-loop. All outgoing sums are below one for the numerical step sizes used below. This finite chain illustrates the graph mechanism, not a claimed neural-network construction.

A forest reaching $G$ must select one outgoing edge from each non-target and contain no cycle. The three choices are $\{A \to G, B \to A\}$, $\{A \to G, B \to G\}$, and $\{A \to B, B \to G\}$, with costs $7, 10, 11$. Thus the minimum forest cost is $E(G) = 7$.

A pruning has one edge and no path from $A$ to $G$. The eligible edges are $A \to B$, $B \to A$, $B \to G$, with costs $3, 5, 8$. The minimum is $3$, giving $E(G \mid A) = 7 - 3 = 4$. Merely looking at the direct cost $A \to G = 2$ misses this exponent.

Write $u = e^{-2/\eta}$, $v = e^{-3/\eta}$, $w = e^{-5/\eta}$, $z = e^{-8/\eta}$. First-step conditioning gives $(u + v)T_A - vT_B = 1$ and $-wT_A + (w + z)T_B = 1$. Solving yields $T_A = (w + z + v)/(uw + uz + vz)$. The leading numerator is $e^{-3/\eta}$; the leading denominator is $e^{-7/\eta}$. Hence $T_A$ has scale $e^{4/\eta}$, confirming the forest calculation independently.

| $\eta$ | Exact $T_A$ and logarithmic scale |
|---|---|
| 0.5 | $T_A \approx 3027.17$; $\eta\log T_A \approx 4.00769$. |
| 0.25 | $T_A \approx 8{,}889{,}035.88$; $\eta\log T_A \approx 4.00008$. |

At $A$, the chance of choosing the rarer $B$ detour before the direct target move has scale $e^{-1/\eta}$. Residence near $B$ has scale $e^{5/\eta}$. Their product has scale $e^{4/\eta}$, which dominates the typical direct route. A mean can therefore be governed by rare, unusually long trajectories. Minimum-path closure of the direct costs leaves this example's exponent unchanged.

## 4. From critical components to the published exponent

Nodes represent connected critical components, not individual samples or parameter coordinates. $B_{ij}$ is the least action of moving between components while avoiding the target beforehand. A transition forest directs every non-target node toward $Q$ without cycles. A pruning deletes one outgoing edge so that the selected starting node $p$ cannot reach $Q$. The energy difference is $E(Q \mid p)$ = minimum forest cost $-$ minimum pruning cost. For general $x$, the paper further adjusts for initial transition costs to possible traps.

$$E(Q \mid x) = \max_p\,\big[E(Q \mid p) - B(x, p)\big]_+ .$$

## 5. State both sides with their qualifications

Theorem 1 reports, for sufficiently small target radius $\delta$ and step $\eta$ and any logarithmic tolerance $a > 0$, the following bounds. The upper bound uses Assumptions 1-3. The matching lower bound additionally requires a sufficiently strong attracting neighborhood of $Q$; Appendix D.3 makes this requirement quantitative.

$$\exp\big[(E(Q \mid x) - a)/\eta\big] \le \mathbb{E}_x[\tau_\delta] \le \exp\big[(E(Q \mid x) + a)/\eta\big].$$

The tolerance $a$ concerns the exponential rate, whereas $\delta$ concerns distance to the global target. Neither is the gradient tolerance $\varepsilon$ of a second-order stationarity theorem. If $E$ is positive, the mean time can be exponentially large in $1/\eta$. If no spurious locally minimizing component exists, Proposition D.1 yields $E = 0$ and subexponential upper scaling. Subexponential does not mean polynomial or practical.

## 6. How the proof is organized

Appendix B states the path large-deviation principle. Appendix C constructs a chain of visits to critical neighborhoods and estimates its transition probabilities. Appendix D applies finite-state forest formulas to the induced chain, then translates its time scale back to actual SGD steps. Retaining enough attraction near $Q$ is the extra step used for the lower bound. Our exact three-state calculation isolates the finite-chain algebra without assuming the delicate continuous-state reduction.

## 7. A Gaussian action makes the geometry intuitive

For an ideal additive Gaussian oracle with covariance $\sigma^2 I$, $H(x, p) = -p \cdot \nabla f(x) + \sigma^2\|p\|^2/2$ and $L(x, v) = \|v + \nabla f(x)\|^2/(2\sigma^2)$. Completing squares gives $L(x, v) - L(x, -v) = 2v \cdot \nabla f(x)/\sigma^2$. Along a reversed gradient-flow connection, uphill action is therefore $2\Delta f/\sigma^2$ while downhill action is zero. The general controlled-noise theorem uses its actual cumulant function; the Gaussian warmup does not remove the blanket regularity and growth requirements.

## 8. What the numerical evidence covers

Figure 2 uses a modified two-dimensional three-hump camel function. For each $\eta$ the authors run 500 trajectories with Gaussian noise $\sigma = 50$ and target distance $10^{-2}$, comparing log hitting times with inverse step size. Appendix F adds two-dimensional Styblinski-Tang and Himmelblau examples. These support the predicted scale on selected toy landscapes, not deep-network runtime or generalization. Polynomial test functions also exceed global Lipschitz smoothness at infinity, so local numerical agreement should be separated from blanket theorem coverage.

## 9. Source audit and limitations

The published lower-bound proof needs further justification at Lemma D.7. Its weighted sum $M_n$ in (D.44) obeys $\mathbb{E}[M_n \mid \text{past}] = (1 - \mu\eta)M_{n-1}$, so it is generally not a martingale as asserted before the Doob-inequality step (D.46). The guide reports the lower-bound statement with its attraction condition, flags this unresolved proof step, and does not certify a repaired proof. Several other displayed sign/index slips should likewise not be copied into code. Knowing the graph and all transition costs can itself be hard; this is an analysis framework, not an efficient global optimizer.

## 10. Survey connection

Fang2019 addresses efficient approximate second-order stationarity. Gürbüzbalaban2021 addresses stationary heavy tails under streaming random curvature. Here the target is first entrance near an actual global optimum, with controlled-noise and geometric conditions. These different targets explain why an exponential hitting time does not contradict a polynomial local-stationarity bound.

| Quantity | Do not confuse it with |
|---|---|
| $\tau_\delta$ | Permanent convergence or a practical stopping test. |
| $E(Q \mid x)$ | Only the cheapest path to the target. |
| $E = 0$ | A proved polynomial complexity bound. |

:::plain Papers Master - Our notes (copy)
Large-deviation transition costs and forest/pruning energies describe small-step expected global-target hitting times. Matching lower bounds add a strong-attraction requirement; the guide flags an unresolved martingale step in that proof. This is not an efficient global optimizer, and subexponential scaling does not imply polynomial time.
:::

Reading recommendation: Read the transition-forest construction to distinguish global hitting time from local stationarity. Keep the lower-bound qualification and unresolved proof step visible.

> **Source:** ICML/PMLR original PDF 1-9; selected necessary appendices B-E and F-G. Central locators: Theorem 1, Eq. 13-23, D.1-D.3, Proposition D.1. Toy Markov example constructed and checked independently. Azizian, Iutzeler, Malick & Mertikopoulos (2025), *The Global Convergence Time of Stochastic Gradient Descent in Non-Convex Landscapes: Sharp Estimates via Large Deviations*, ICML 2025. [arxiv.org/abs/2503.16398](https://arxiv.org/abs/2503.16398)
