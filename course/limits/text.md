## 1. Upper bounds, lower bounds and oracles

Every stop so far has quoted rates: GD needs $O(\varepsilon^{-2})$ steps, perturbed accelerated GD $\tilde O(\varepsilon^{-7/4})$, SGD $O(\varepsilon^{-4})$. Stop 11 ended by asking the question these numbers raise: **what is the best any algorithm could possibly do, and how close do the methods in the Atlas come?** To answer it, we first need to say precisely what an algorithm is allowed to know, and what kind of statement a rate is.

:::def Oracle
An algorithm learns about $f$ only by asking an **[[oracle]]** for information at points of its choice. A *first-order oracle* answers a query $x$ with $f(x)$ and $\nabla f(x)$; a *second-order* one also gives $\nabla^2 f(x)$ (or Hessian-vector products, stop 4); a *stochastic* one returns an unbiased noisy estimate, like a minibatch gradient (stop 8). The cost of an algorithm is the number of queries it needs to reach its target.
:::

Counting queries ignores everything else an algorithm does (arithmetic, memory), which makes the count a clean, machine-independent measure. Two kinds of statements are then possible.

- An **[[upper-bound|upper bound]]** says that *one particular algorithm* reaches the target within so many queries on *every* function of a class. Stop 3's result is one: on any $L$-smooth $f$, GD with $\eta = 1/L$ finds an $\varepsilon$-stationary point within $2L(f(x_0) - f^\star)/\varepsilon^2$ steps, which is $20{,}000$ for $L = 1$, $f(x_0) - f^\star = 1$, $\varepsilon = 0.01$.
- A **[[lower-bound|lower bound]]** says that *every* algorithm of a class needs at least so many queries on *some* function of the class. It is written with $\Omega$ (omega): $\Omega(g)$ means "at least a constant times $g$", the mirror image of $O(g)$.

Both are **[[worst-case]]** statements: an upper bound must hold for the hardest function in the class, and a lower bound only needs one hard function. A real problem can be far easier than the worst case, as every "guarantee versus real run" visual of this course showed. When the upper bound of a method matches the lower bound of its class (same power of $\varepsilon$), the method is **optimal** for that class: no algorithm can do better in the worst case, up to constants.

Papers often hide logarithmic factors: **[[soft-o|$\tilde O$]]** ("soft O") means big-O up to factors like $\log d$ or $\log(1/\varepsilon)$. Jin et al. 2017's perturbed GD, for example, needs $O\big(\ell(f(x_0) - f^\star)/\varepsilon^2 \cdot \log^4(\cdot)\big)$ iterations, written $\tilde O(\varepsilon^{-2})$.

## 2. A hard function: the zero chain

How can one prove that *every* algorithm is slow? The idea of Carmon, Duchi, Hinder & Sidford 2020 is to build a function that reveals itself one coordinate at a time.

Consider an algorithm that starts at $x_0 = 0$ and only moves within the span of the gradients it has seen so far. GD, momentum and coordinate-wise adaptive methods such as Adam behave like this, as do most methods in this course; such methods are called *zero-respecting*. Now suppose $f$ has this property: **if $x_j = 0$ for all $j \ge i$, then $\nabla f(x)$ is zero in every coordinate beyond $i$**. Then each query can make at most one new coordinate non-zero. A function with this property is a **[[zero-chain|zero chain]]**.

Carmon et al.'s chain on $T$ coordinates is

$$\bar f_T(x) = -\Psi(1)\,\Phi(x_1) + \sum_{i=2}^{T} \Big[\Psi(-x_{i-1})\,\Phi(-x_i) - \Psi(x_{i-1})\,\Phi(x_i)\Big],$$

built from two one-dimensional pieces:

$$\Psi(t) = \begin{cases} 0 & t \le \tfrac12 \\ \exp\!\Big(1 - \dfrac{1}{(2t - 1)^2}\Big) & t > \tfrac12 \end{cases}, \qquad \Phi(t) = \sqrt{e} \int_{-\infty}^{t} e^{-s^2/2}\, ds .$$

$\Psi$ is a smooth switch: exactly $0$ until $\tfrac12$, then rising to $\Psi(1) = 1$. $\Phi$ is a smooth increasing step with slope $\Phi'(0) = \sqrt{e} \approx 1.649$. Because $\Psi$ is exactly zero near $0$, coordinate $x_i$ does not affect the gradient at all until its left neighbour $x_{i-1}$ has grown past $\tfrac12$ in size. Three properties are proved in the paper:

- **Lemma 1**: $\Psi(x)\,\Phi'(y) > 1$ whenever $x \ge 1$ and $\lvert y \rvert < 1$.
- **Lemma 2**: $\lVert \nabla \bar f_T(x) \rVert > 1$ unless *every* coordinate satisfies $\lvert x_i \rvert \ge 1$.
- **Lemma 3**: $\bar f_T(0) - \inf \bar f_T \le 12T$, and the derivatives of $\bar f_T$ are Lipschitz, with constants that do not depend on $T$.

So the gradient stays above $1$ until all $T$ coordinates have been uncovered, and uncovering them takes at least $T$ queries.

:::viz chain2
:::

With $\eta = 0.05$, $x_2$ stays exactly $0$ for seven steps; it first moves at step 8, once $x_1 = 0.554$ has passed $\tfrac12$. The gradient norm drops below $1$ only at step 29. For every step size on the slider, $x_2$ waits for $x_1$ to cross $\tfrac12$.

## 3. From the chain to $\Omega(\varepsilon^{-2})$

The chain has gradient threshold $1$ and gap $12T$; a lower bound for tolerance $\varepsilon$ needs threshold $\varepsilon$ and gap at most $\Delta = f(x_0) - f^\star$. Rescaling does it. Let $\ell_1$ be the smoothness constant of $\bar f_T$ from Lemma 3, and set

$$f(x) = \frac{L\lambda^2}{\ell_1}\, \bar f_T\!\Big(\frac{x}{\lambda}\Big), \qquad \lambda = \frac{\ell_1\, \varepsilon}{L}.$$

Three lines of calculus give: the Hessian of $f$ is $L/\ell_1$ times that of $\bar f_T$, so $f$ is $L$-smooth; its gradient is $\nabla f(x) = \frac{L\lambda}{\ell_1}\nabla \bar f_T(x/\lambda) = \varepsilon\, \nabla \bar f_T(x/\lambda)$, so $\lVert \nabla f \rVert > \varepsilon$ until all coordinates are uncovered; and its gap is at most $\frac{L\lambda^2}{\ell_1} \cdot 12T = 12 T \ell_1 \varepsilon^2 / L$. Choosing the largest $T$ with gap $\le \Delta$,

$$T = \frac{\Delta\, L}{12\, \ell_1\, \varepsilon^2} \qquad \Longrightarrow \qquad \text{every zero-respecting method needs } \Omega\big(\Delta L\, \varepsilon^{-2}\big) \text{ queries.}$$

This matches GD's upper bound from stop 3 up to a constant: **GD is optimal** among first-order methods for finding $\varepsilon$-stationary points of $L$-smooth functions.

:::theorem Carmon, Duchi, Hinder & Sidford 2020, Theorems 1 and 2
For any $p \ge 1$, every deterministic (Theorem 1) or randomized (Theorem 2) algorithm that uses derivatives up to order $p$ needs at least

$$c_p\, \Delta\, L_p^{1/p}\, \varepsilon^{-(p+1)/p}$$

queries to find an $\varepsilon$-stationary point of some function with $p$-th derivative $L_p$-Lipschitz and gap $\Delta$, in a high enough dimension. The bounds are sharp: GD attains $p = 1$, cubic-regularized Newton (Nesterov & Polyak 2006) attains $p = 2$.
:::

| derivatives used | $p$ | lower bound | at $\varepsilon = 0.01$ |
|---|---|---|---|
| gradients | 1 | $\varepsilon^{-2}$ | $10^4$ |
| gradients and Hessians | 2 | $\varepsilon^{-3/2}$ | $10^3$ |
| up to third derivatives | 3 | $\varepsilon^{-4/3}$ | $464$ |

Two fine points. The hard function lives in dimension at least $T$, which grows like $\varepsilon^{-2}$; the bounds are about *dimension-free* rates, the kind every theorem in the Atlas states. And the class matters: with the extra assumption that the Hessian is $\rho$-Lipschitz, gradient methods can beat $\varepsilon^{-2}$ (perturbed accelerated GD, $\tilde O(\varepsilon^{-7/4})$, stop 7) without contradicting Theorem 1, which is about the larger class of all $L$-smooth functions.

:::viz chain
:::

With exact gradients, GD needs about 6.5 more steps for every extra coordinate: 71 queries for $T = 8$ and 121 for $T = 16$. The linear growth in $T$, after rescaling, is the $\varepsilon^{-2}$.

## 4. Noise makes it harder: stochastic lower bounds

With a stochastic oracle (stop 8), Arjevani, Carmon, Duchi, Foster, Srebro & Woodworth 2023 make the chain slower to uncover. Their oracle returns the exact gradient, except in the next undiscovered coordinate, which it reveals only with probability $p$, multiplied by $1/p$ so that the answer is still unbiased (their Eq. 17). Its variance is at most $23^2(1 - p)/p$ (Lemma 3). Now each new coordinate costs about $1/p$ queries, so uncovering $T$ of them costs about $T/p$ (Lemma 1).

The numbers balance as follows (up to constants). After the rescaling of section 3, the gradient entries are of size $\varepsilon$, and the oracle's variance is about $\varepsilon^2/p$. Allowing variance $\sigma^2$ means choosing $p \approx \varepsilon^2/\sigma^2$, and then

$$\frac{T}{p} \;\approx\; \frac{\Delta L}{\varepsilon^2} \cdot \frac{\sigma^2}{\varepsilon^2} \;=\; \frac{\Delta L\, \sigma^2}{\varepsilon^4}.$$

:::theorem Arjevani et al. 2023, Theorems 1–3
Every zero-respecting algorithm with a stochastic first-order oracle of variance $\sigma^2$ needs $\Omega(\Delta L \sigma^2 \varepsilon^{-4})$ queries to find an $\varepsilon$-stationary point of an $L$-smooth function (Theorem 1). This matches SGD (Ghadimi & Lan 2013, stop 8), so **SGD is optimal** in this setting. If the stochastic gradients are also smooth on average (*mean-squared smoothness*, constant $\bar L$), the bound becomes $\Omega(\Delta \bar L \sigma \varepsilon^{-3} + \sigma^2 \varepsilon^{-2})$ (Theorem 2), which variance-reduced methods such as SPIDER attain (stop 8). Theorem 3 extends both to all randomized algorithms and any batch size.
:::

In the visual above, switch the oracle to $p = 0.2$ or $p = 0.05$. For $T = 8$ the counts go from 71 queries (exact) to 90 and 191; for $T = 16$, from 121 to 156 and 529. Each count is one seeded run, so the numbers fluctuate, but the cost of each coordinate grows like $1/p$.

Second-order information helps less with noise. Arjevani et al. 2020 give an algorithm that reaches an $\varepsilon$-stationary point with $O(\varepsilon^{-3})$ stochastic gradients and Hessian-vector products, and prove $\Omega(\varepsilon^{-3})$ for *every* stochastic method of order $p \ge 2$: with noise, going from first to second order improves $\varepsilon^{-4}$ to $\varepsilon^{-3}$, and higher orders do not improve it further. Without noise the exponents keep improving with $p$, as in the table above. The authors call this the "elbow".

## 5. A different kind of hard: NP-hardness

The bounds above are about approximate stationarity, which is reachable in polynomially many queries. Harder targets are a different story. **[[np-hard|NP-hard]]** means at least as hard as every problem in the class NP; no polynomial-time algorithm is known for any of them, and finding one would prove P = NP.

- Murty & Kabadi 1987: for quadratic programs with simple constraints and integer data, deciding that a given point is *not* a local minimum, or that the objective is *not* bounded below, is NP-complete. Through a reduction from the subset-sum problem, computing a global minimum of a smooth non-convex quadratic program is NP-hard. As a corollary, checking that a matrix is *not* copositive ($v^\top Q v \ge 0$ for all $v \ge 0$) is NP-complete.
- Deciding whether a point is a local minimum is hard even without constraints: for a quartic polynomial whose Hessian at the point is zero, the answer is a copositivity question (via Ahmadi & Zhang's restatement).
- Blum & Rivest 1992 (Theorem 2): deciding whether a network with two hidden threshold units and one threshold output unit can fit a given training set exactly is NP-complete.
- Anandkumar & Ge 2016 (Theorem 10): finding a fourth-order local minimum is NP-hard. Arjevani et al. 2020 note, citing Murty & Kabadi, that it is NP-hard to decide whether a second-order stationary point is a local minimum or a higher-order saddle.

The smallest case of the second point fits on screen: $f(u) = u_1^4 + 2q\, u_1^2 u_2^2 + u_2^4$. At the origin the gradient and the Hessian are zero for every $q$, so the tests of stops 2–5 are blind. Because $f(r u) = r^4 f(u)$, the origin is a local minimum exactly when $f \ge 0$ on the unit circle, whose minimum is $(1 + q)/2$ (on the diagonals) for $q \le 1$. So the origin is a local minimum if and only if $q \ge -1$; for $q = -1.2$ the circle dips to $-0.1$. Writing $v = (u_1^2, u_2^2)$, $f(u) = v^\top Q v$ with $Q = \begin{bmatrix} 1 & q \\ q & 1 \end{bmatrix}$, and the test "$q \ge -1$" is exactly "$Q$ is copositive". It is easy in two dimensions; in $d$ dimensions it is NP-hard.

:::viz quartic
:::

At $q = -1.3$ the ball slides down a diagonal and leaves the square after 30 steps. For $q \ge -1$ it creeps toward the origin, slowly, because the minimum is degenerate (the gradient vanishes like $\lVert u \rVert^3$).

:::key The idea to keep
"Find a stationary point" is easy in the worst case: polynomially many queries, with known optimal rates. "Find a global minimum", or even "certify a local one", is NP-hard in general. This is why the papers in the Atlas aim for $\varepsilon$-FOSPs and $\varepsilon$-SOSPs (stop 5), and why the landscape results of stop 11 matter: they say when those cheap targets are already global minima.
:::

## 6. The complexity ladder

Here are the main rates of the course in one picture, grouped by the oracle. The length of a bar is the exponent $a$ in $\varepsilon^{-a}$: shorter is faster.

:::viz ladder
:::

Read it row by row and three patterns appear.

1. **Matched pairs are optimal.** GD and the first-order lower bound ($\varepsilon^{-2}$); cubic regularization and the second-order lower bound ($\varepsilon^{-3/2}$); SGD and the stochastic lower bound ($\varepsilon^{-4}$); SPIDER under mean-squared smoothness ($\varepsilon^{-3}$); stochastic second-order methods ($\varepsilon^{-3}$).
2. **Second-order targets come almost for free.** Perturbed GD reaches an $\varepsilon$-SOSP in $\tilde O(\varepsilon^{-2})$, the same power as GD's first-order rate (stop 6).
3. **Each extra assumption buys speed.** Hessian-Lipschitz smoothness lets acceleration reach $\varepsilon^{-7/4}$; strong convexity or PL turns powers of $1/\varepsilon$ into $\log(1/\varepsilon)$: with $\kappa = 100$ and $\varepsilon = 10^{-3}$, $\kappa \log(1/\varepsilon) \approx 691$ against $\sqrt{\kappa} \log(1/\varepsilon) \approx 69$ with momentum (stop 7).

At $\varepsilon = 10^{-3}$, $\varepsilon^{-2} = 10^6$ and $\varepsilon^{-4} = 10^{12}$; these are worst-case counts with constants dropped, not predictions of training time.

## 7. How to read a paper in the Atlas

Every theorem in the Atlas can be decoded with the same six questions. Take Jin, Ge, Netrapalli, Kakade & Jordan 2017, Theorem 3, as the example.

| question | what to look for | in Jin et al. 2017 |
|---|---|---|
| 1. Which functions? | assumptions: $L$-smooth, $\rho$-Hessian-Lipschitz, bounded variance $\sigma^2$, PL, width… | gradient $\ell$-Lipschitz (their $\ell$ is our $L$) and Hessian $\rho$-Lipschitz |
| 2. Which target? | $\varepsilon$-FOSP, $\varepsilon$-SOSP (with which curvature tolerance), global minimum, loss $\le \varepsilon$ | $\varepsilon$-SOSP: $\lVert \nabla f \rVert \le \varepsilon$ and $\lambda_{\min}(\nabla^2 f) \ge -\sqrt{\rho\varepsilon}$ |
| 3. Which oracle? | exact or stochastic gradients, Hessian-vector products, Hessians; what is counted | exact gradients; iterations (one gradient each) |
| 4. How fast, really? | the power of $\varepsilon$, hidden constants, logs in $\tilde O$, dependence on $d$, success probability | $O\big(\ell (f(x_0) - f^\star)/\varepsilon^2 \cdot \log^4(d\ell\Delta_f/(\varepsilon^2\delta))\big)$, with probability $1 - \delta$: polylogarithmic in $d$ |
| 5. Upper or lower bound? | is it matched by a bound in the other direction? | upper bound; same power as the $\Omega(\varepsilon^{-2})$ lower bound for first-order points |
| 6. Proved or observed? | theorem with assumptions, or experiments | proved |

Questions 2 and 6 catch most misreadings. "Converges" may mean to a stationary point, to a local minimum or to a global one; and "SAM finds flatter minima" (stop 10) or "monotone interpolation" (stop 11) are observations, not theorems. Each paper page in the Atlas states its result in this form, with the toy model and what the picture simplifies.

## 8. Where to go next

The journey started with "make the loss small" and ends here, with the vocabulary of the field: gradients and curvature, saddles and their escape, momentum, noise and adaptivity, sharpness, landscapes and limits. The survey that the Atlas accompanies is organized by the same questions, in its Fig. 5 taxonomy:

- [III. Geometry of critical points](index.html#b-III): which critical points exist, and when every local minimum is global. Prepared by stops 4, 5 and 11.
- [IV. Dynamics of GD/SGD near saddles](index.html#b-IV): asymptotic avoidance, escape times, plateaus, noise. Prepared by stops 6 and 8.
- [V. Algorithms designed to escape saddles](index.html#b-V): second-order and cubic methods, perturbation, acceleration, Hessian-free negative curvature, variance reduction and the lower bounds of this stop. Prepared by stops 6, 7, 8 and 12.
- [VI. Overparameterization](index.html#b-VI): NTK and lazy training, mean field, beyond lazy. Prepared by stop 11.
- [VII. Dynamics of practical training](index.html#b-VII): initialization and momentum, large learning rates and the edge of stability, sharpness-aware methods, modern optimizers. Prepared by stops 7, 9 and 10.

Pick a branch, open a paper, and read its theorem with the six questions of section 7. When a word is unclear, the [glossary](learn.html?ch=glossary) sends you back to the stop that builds it.

:::atlas Where this shows up in the Atlas
- Deterministic lower bounds: [Carmon, Duchi, Hinder & Sidford 2020](paper.html?id=carmon2020lower).
- Stochastic lower bounds: [Arjevani et al. 2023](paper.html?id=arjevani2023lower), [Arjevani et al. 2020](paper.html?id=arjevani2020second); a finite-sum lower bound appears in [Fang et al. 2018](paper.html?id=fang2018spider) (Theorem 3).
- Hardness: [Murty & Kabadi 1987](paper.html?id=murty1987some), [Blum & Rivest 1992](paper.html?id=blum1992training), [Anandkumar & Ge 2016](paper.html?id=anandkumar2016efficient).
- The upper bounds on the ladder: [Nesterov 2004](paper.html?id=nesterov2004introductory), [Jin et al. 2017](paper.html?id=jin2017escape), [Jin, Netrapalli & Jordan 2018](paper.html?id=jin2018accelerated), [Nesterov & Polyak 2006](paper.html?id=nesterov2006cubic), [Ghadimi & Lan 2013](paper.html?id=ghadimi2013stochastic), [Fang et al. 2018](paper.html?id=fang2018spider), [Fang, Lin & Zhang 2019](paper.html?id=fang2019sharp), [Tripuraneni et al. 2018](paper.html?id=tripuraneni2018stochastic), [Allen-Zhu 2018](paper.html?id=allenzhu2018natasha).
:::
