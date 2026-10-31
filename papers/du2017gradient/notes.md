# Gradient Descent Can Take Exponential Time to Escape Saddles

:::minute The paper in one minute
**Main problem:** Lee et al. showed that generic random initialization does not lead GD to converge to a strict saddle, under appropriate assumptions. This paper asks the computational question that theorem leaves open: *can GD still take an enormous number of iterations to make useful progress?* The authors construct smooth non-convex problems where the answer is yes.

**Principal result:** There are smooth non-pathological function classes and common random initialization schemes for which plain GD takes exponential time to escape saddles, even though it asymptotically avoids strict saddle convergence. Perturbation-based GD can achieve polynomial-time approximate local-minimum finding under its separate assumptions.

**Critical distinction:** Probability zero of *converging* to a strict saddle says nothing by itself about the number of iterations required to escape its neighborhood.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. "Where GD ends" and "how long it takes" are different questions

A **saddle-avoidance** theorem says where GD ends: from almost every start, not at a strict saddle. A **complexity** bound says how many iterations it takes to get somewhere useful. The first can hold while the second is terrible.

Complexity is measured against the dimension $d$. **Polynomial** time means a bound like $d^3$ iterations. **Exponential** time means something like $2^d$: if each of $d$ saddles in a row takes twice as long as the one before, the last one alone costs $2^{10} = 1024$ times the first for $d = 10$, and $2^{30} \approx 1.07 \times 10^9$ times for $d = 30$.

### 2. Near a quadratic saddle, each coordinate is multiplied by a fixed number

The paper's building block is $f(x_1, x_2) = -\gamma x_1^2 + L x_2^2$ with $L > \gamma > 0$. Its gradient is $(-2\gamma x_1,\; 2L x_2)$, so one GD step with step size $\eta$ gives

$$x_1 \leftarrow (1 + 2\eta\gamma)\, x_1, \qquad x_2 \leftarrow (1 - 2\eta L)\, x_2 .$$

The escape coordinate $x_1$ grows geometrically; the stable coordinate $x_2$ shrinks geometrically. How long the escape takes depends on how small $x_1$ starts: the count grows like $\log(1/x_1)$.

### 3. Escaping one saddle can set up a slow escape from the next

The paper's key observation (its Eq. 3) glues two such saddles. Around the first, $x_1$ escapes and $x_2$ is stable. Around the second, the roles swap: $x_2$ is now the escape direction. But $x_2$ has been shrinking the whole time GD was escaping the first saddle.

Take $\gamma = L = 1$, $\eta = 0.1$, start $(x_1, x_2) = (0.01, 0.5)$. The multipliers are $1.2$ and $0.8$. Leaving the first region ($x_1 \ge 1$) takes $t_1 = 26$ steps; by then $x_2 = 0.5 \cdot 0.8^{26} \approx 0.0015$. Growing $x_2$ back to $1$ at rate $1.2$ takes $36$ more steps, so the second escape ends at $t_2 = 62$, above the paper's bound $t_2 \ge \tfrac{L + \gamma}{\gamma}\, t_1 = 52$. Chain $d$ saddles like this and the waits multiply by $(L + \gamma)/\gamma$ each time: exponential in $d$.

### 4. A random kick replaces a tiny offset by a typical one

**Perturbed GD** adds a small random vector, drawn uniformly from a ball of radius $r$, when the gradient is small. Its component along the escape direction is then rarely tiny. In $d$ dimensions the chance that it is below $w\,r$ is about $w\sqrt{2d/\pi}$ for small $w$: for $d = 100$ and $w = 10^{-6}$, about $8 \times 10^{-6}$. Each saddle then costs roughly the same number of steps, instead of more and more.

:::key Putting the four ideas together
Saddle avoidance (idea 1) leaves the escape time open. Near a saddle, time is set by the log of the escape coordinate (idea 2), and a chain of saddles can shrink that coordinate before it is needed, multiplying the wait at every stage (idea 3). That is the paper's exponential lower bound for plain GD; random kicks reset the offset (idea 4), which is why perturbed GD stays polynomial. The notes below make this precise.
:::
:::

:::divider Study notes
:::

## 1. Main problem: "almost surely" does not mean "quickly"

Lee et al. showed that generic random initialization does not lead GD to converge to a strict saddle, under appropriate assumptions. This paper asks the computational question that theorem leaves open: *can GD still take an enormous number of iterations to make useful progress?* The authors construct smooth non-convex problems where the answer is yes.

## 2. A first-principles escape-time calculation

Consider the toy objective

$$f(x, y) = \tfrac12 x^2 - \tfrac12 y^2 + \tfrac14 y^4,$$

with a strict saddle at $(0, 0)$ and minima at $(0, \pm 1)$. GD with a small step $\eta$ has

$$x_{k+1} = (1 - \eta)\,x_k, \qquad y_{k+1} = y_k + \eta\,(y_k - y_k^3).$$

Near the saddle, $|y_k| \ll 1$, the cubic term is negligible and

$$y_{k+1} \approx (1 + \eta)\,y_k, \qquad T_{\mathrm{esc}} \approx \frac{\log(r/\delta)}{\log(1 + \eta)},$$

where $\delta = |y_0|$ is the initial distance from the stable set $y = 0$ and $r$ is an escape radius. An initial offset $\delta > 0$ does eventually grow, but if $\delta$ is extremely tiny, the waiting time can be very long. This is an *illustrative example*, not the authors' exponential lower-bound construction.

## 3. How the actual construction produces exponential slowdown

The paper constructs high-dimensional smooth functions containing a carefully arranged sequence of saddle regions. Escape from one region sends the unperturbed GD trajectory extremely close to the attracting/stable directions of the next. This repeatedly suppresses the component of motion along the next escape direction. As the number of coupled saddles grows with dimension, the waiting time can become *exponential in the dimension* for natural random initializations.

:::key Principal result
There are smooth non-pathological function classes and common random initialization schemes for which plain GD takes exponential time to escape saddles, even though it asymptotically avoids strict saddle convergence. Perturbation-based GD can achieve polynomial-time approximate local-minimum finding under its separate assumptions.
:::

## 4. Why random perturbations change the story

A random kick $x \leftarrow x + \xi$ near a saddle can restore a non-negligible component in a negative-curvature direction. In the toy model, a nonzero vertical perturbation changes $\delta$ from an arbitrarily tiny value to a typical random scale, reducing $\log(r/\delta)$. More sophisticated perturbed GD analyses quantify how unlikely it is for a perturbation to land in the thin "stuck" set. This is the connection to Jin et al. [8] and our survey's Fig. 10, not a statement that any arbitrary perturbation works.

## 5. Assumptions, limitations and relevance

The lower bound is *existential*: it constructs hard objectives, not proof that training every large network with plain GD requires exponential time. The complementary polynomial guarantees rely on function smoothness, curvature controls and perturbation schedules. It does not characterize practical Adam/SGD trajectories or degenerate-saddle escape.

:::warn Critical distinction
Probability zero of *converging* to a strict saddle says nothing by itself about the number of iterations required to escape its neighborhood.
:::

This is crucial for **RQ2** (trajectory behavior), **RQ3** (why perturbation methods matter), and **RQ4** (worst-case constructions versus neural-network practice).

## 6. Our notes

I think this is the clearest reason we need separate sections for saddle avoidance and saddle escape. GD can be guaranteed not to settle at a strict saddle and still spend an impractically long time moving from one saddle region to another. Random perturbations change the story by keeping the trajectory from repeatedly landing almost exactly on stable directions. The exponential lower bound is a worst-case example, not a prediction for every neural network.

> **Original source:** [papers.neurips.cc/paper/6707-gradient-descent-can-take-exponential-time-to-escape-saddle-points.pdf](https://papers.neurips.cc/paper/6707-gradient-descent-can-take-exponential-time-to-escape-saddle-points.pdf). **Verification:** original NeurIPS manuscript; quartic example is pedagogical.
