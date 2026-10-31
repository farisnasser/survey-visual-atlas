# Lower Bounds for Non-Convex Stochastic Optimization

:::minute The paper in one minute
**Problem:** For non-convex stochastic learning, many algorithms prove that they eventually find a point where the gradient is small. But are their iteration rates loose analysis, or genuine limits of the information available?

**Approach:** This paper constructs *lower bounds*: worst-case smooth optimization problems on which every method using a specified stochastic-gradient oracle needs many queries to find even an approximately stationary point.

**Main result:** Under bounded-variance stochastic gradient information, the classical $\epsilon^{-4}$ worst-case rate is minimax optimal. With mean-squared smoothness of the oracle, the correct frontier improves to $\epsilon^{-3}$. The result is about *first-order stationarity*, not escaping every saddle.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. Complexity counts oracle queries

The paper measures cost in **queries to an oracle**: a black box that, given a point $x$, returns a random vector $g$ whose average is $\nabla F(x)$. An algorithm may do any computation it likes between queries; only the number of queries counts. The **complexity** of a function class is the number of queries the best algorithm needs on the hardest function in that class to reach $\|\nabla F(x)\| \le \epsilon$. It is a min over algorithms of a max over functions, so it is called **minimax**.

### 2. Squared-gradient rates turn into $\epsilon^{-4}$

The notes' §2 bound controls the *squared* gradient: about $T^{-1/2}$ after $T$ queries. Asking for $\|\nabla F\| \le \epsilon$ means asking for $\|\nabla F\|^2 \le \epsilon^2$, so $T^{-1/2} \le \epsilon^2$ needs $T \ge \epsilon^{-4}$. The exponent matters a great deal:

$$\epsilon = 0.1:\quad \epsilon^{-4} = 10^4,\ \ \epsilon^{-3} = 10^3; \qquad \epsilon = 0.01:\quad \epsilon^{-4} = 10^8,\ \ \epsilon^{-3} = 10^6 .$$

Halving $\epsilon$ multiplies an $\epsilon^{-4}$ cost by $16$ and an $\epsilon^{-3}$ cost by $8$.

### 3. A lower bound hides information one piece at a time

The hard functions are built as a **chain**: the gradient at $x$ only reveals the coordinate right after the last one the algorithm has already discovered. To reach a small gradient, the algorithm must discover $T$ coordinates in order, so it needs at least $T$ queries even with exact gradients. In high dimension it cannot guess the hidden coordinates. Choosing $T$ of order $\Delta L/\epsilon^2$ (the gap $\Delta$ and smoothness $L$ fix how long a chain fits) already gives a deterministic lower bound of order $\epsilon^{-2}$.

### 4. Noise makes each piece cost $1/p$ queries

The stochastic oracle reveals the next coordinate only with probability $p$ per query, and otherwise reports zero there. To stay unbiased it multiplies that entry by $z/p$ with $z \sim \text{Bernoulli}(p)$; the factor has mean $1$ and variance $(1-p)/p$. Each discovery then waits a geometric number of queries with mean $1/p$, so the whole chain needs about $T/p$ queries. Example: $T = 10$ and $p = 0.1$ give $100$ queries on average; the factor $z/p$ has variance $9$. In the paper the hard function's gradient entries have size about $\epsilon$, so a variance budget $\sigma^2$ lets $p$ shrink to about $\epsilon^2/\sigma^2$. Then $T/p \approx (\Delta L/\epsilon^2)(\sigma^2/\epsilon^2) = \Delta L\sigma^2/\epsilon^4$: the extra factor lifts $\epsilon^{-2}$ to $\epsilon^{-4}$. (This is a sketch: the paper's oracle and constants are more careful.)

### 5. Using the same sample at two points can cancel noise

Some oracles let you query two points with the *same* random sample $z$. If $g(x,z) = \nabla F(x) + z$ (additive noise), the difference $g(x,z) - g(y,z) = \nabla F(x) - \nabla F(y)$ has no noise at all. With two independent samples, the difference has variance $2\sigma^2$. **Mean-squared smoothness** is the assumption that same-sample differences shrink with $\|x - y\|$. Variance-reduced methods such as SPIDER track the gradient by adding up these cheap differences, which is why they reach $\epsilon^{-3}$, and why the matching lower bound needs that stronger oracle.

:::key Putting the five ideas together
Complexity is a game against every algorithm (idea 1), and the exponent of $\epsilon$ decides the cost (idea 2). A hidden chain forces sequential discovery (idea 3), and oracle noise makes each discovery rare (idea 4): together they give $\Omega(\epsilon^{-4})$, matching SGD. A same-sample oracle can cancel noise (idea 5), which changes the frontier to $\epsilon^{-3}$. Both results are made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. The problem: is the slow rate of SGD unavoidable?

For non-convex stochastic learning, many algorithms prove that they eventually find a point where the gradient is small. But are their iteration rates loose analysis, or genuine limits of the information available? This paper constructs *lower bounds*: worst-case smooth optimization problems on which every method using a specified stochastic-gradient oracle needs many queries to find even an approximately stationary point. It does not ask for global or local minimizers.

$$\text{Target: } \|\nabla F(x)\| \le \epsilon, \qquad F : \mathbb{R}^d \longrightarrow \mathbb{R}.$$

## 2. Rebuild the standard SGD rate from first principles

Assume a smooth objective with Lipschitz gradient $L$ and a lower bound $F_*$; we also have an unbiased stochastic gradient $g_t$ satisfying

$$\mathbb{E}[g_t \mid x_t] = \nabla F(x_t), \qquad \mathbb{E}[\|g_t - \nabla F(x_t)\|^2 \mid x_t] \le \sigma^2.$$

The smoothness (descent) inequality and SGD update $x_{t+1} = x_t - \eta g_t$ imply

$$\begin{aligned} \mathbb{E}_t F(x_{t+1}) &\le F(x_t) - \eta\|\nabla F(x_t)\|^2 + \frac{L\eta^2}{2}\mathbb{E}_t\|g_t\|^2 \\ &\le F(x_t) - \left(\eta - \frac{L\eta^2}{2}\right)\|\nabla F(x_t)\|^2 + \frac{L\eta^2\sigma^2}{2}. \end{aligned}$$

For $\eta \le 1/L$, telescope over $t = 0, \dots, T-1$ and define $\Delta = F(x_0) - F_*$:

$$\frac{1}{T}\sum_{t<T} \mathbb{E}\|\nabla F(x_t)\|^2 \le \frac{2\Delta}{\eta T} + L\eta\sigma^2.$$

Choosing $\eta$ on the order of $T^{-1/2}$ makes the *expected squared gradient* decay as $O(T^{-1/2})$, so achieving it below $\epsilon^2$ takes $T = O(\epsilon^{-4})$ with other parameters fixed. This is a teaching derivation of the familiar upper bound, not the paper's lower-bound proof.

## 3. The actual new contribution: two matching impossibility results

The authors prove that, for general $L$-smooth non-convex objectives with an unbiased stochastic gradient oracle having bounded variance, the order $\epsilon^{-4}$ cannot be uniformly improved: a suitable worst-case instance needs

$$\boxed{\Omega(\epsilon^{-4})\ \text{stochastic queries}}\,.$$

They also analyze a *more restrictive oracle*: the same random sample can give gradients that vary smoothly in mean square with the query location, typically expressed as

$$\mathbb{E}_z\|g(x,z) - g(y,z)\|^2 \le \widetilde{L}^2\|x - y\|^2.$$

In this class they establish a sharper lower bound of $\Omega(\epsilon^{-3})$. This agrees with the best recursive variance-reduced gradient methods of the corresponding class (such as SPIDER-type estimators). Smoothness of the *expected* function alone is weaker than mean-squared smoothness of the individual stochastic oracle.

:::key Main result
Under bounded-variance stochastic gradient information, the classical $\epsilon^{-4}$ worst-case rate is minimax optimal. With mean-squared smoothness of the oracle, the correct frontier improves to $\epsilon^{-3}$. The result is about *first-order stationarity*, not escaping every saddle.
:::

## 4. Why constructing lower bounds is different from convergence analysis

An upper bound describes an algorithm and proves *it succeeds* within a specified number of queries on all admissible functions. A lower bound must show that *every possible oracle algorithm* faces a hard admissible function; one counterexample for ordinary SGD would not suffice. The paper uses high-dimensional, carefully hidden information and noisy oracle constructions. A picture of a long narrow valley can build intuition, but it is *not* a substitute for that information-theoretic argument.

## 5. Assumptions, limits, connection to our survey

The guarantee is minimax and asymptotic in the stationarity tolerance, with variance, smoothness, initial gap, and dimension relevant to complete statements. It does not predict convergence time for a particular neural net, incorporate GPU parallelism, or require the hard instance to resemble practical ReLU losses. It does not prove second-order stationarity or any particular saddle escape mechanism. This is core **RQ3** material for the oracle complexity comparison and **RQ4** for explaining why strong performance claims depend on the oracle model.

:::warn Important distinction
Do not conflate the $\Omega(\epsilon^{-3})$ lower bound under *mean-squared oracle smoothness* with an $\epsilon^{-3}$ bound under mere bounded variance. They are different assumptions.
:::

## 6. Our notes

This paper is the reason we should not call SGD's $\epsilon^{-4}$ rate "outdated" just because SPIDER can get $\epsilon^{-3}$. SPIDER relies on a stronger way of comparing noisy gradients at nearby points. Under the weaker stochastic oracle, the slower rate is actually optimal in the worst case. It also keeps our table honest: small gradient does not mean we have escaped a saddle. **Original source:** [https://doi.org/10.1007/s10107-022-01822-7](https://doi.org/10.1007/s10107-022-01822-7). **Source check:** full author manuscript accessible via the NSF Public Access Repository.

> **Primary source:** Arjevani, Carmon, Duchi, Foster, Srebro & Woodworth (2023), *Lower Bounds for Non-Convex Stochastic Optimization*, Mathematical Programming 199. [doi.org/10.1007/s10107-022-01822-7](https://doi.org/10.1007/s10107-022-01822-7).
