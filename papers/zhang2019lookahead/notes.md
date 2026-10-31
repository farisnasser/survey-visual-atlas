# Lookahead Optimizer: k Steps Forward, 1 Step Back

:::minute The paper in one minute
**Question:** Can a simple wrapper stabilize an existing SGD or Adam optimizer without redesigning its gradients?

**Model/Method:** Maintain *fast weights* for $k$ inner optimizer steps and *slow weights* updated by interpolation toward the final fast parameters.

**Finding:** Lookahead reduces stationary variance in a noisy quadratic model under assumptions and often improves training robustness and accuracy in tested vision and language tasks.

**Why we care:** Training dynamics can be improved by a low-cost two-timescale update, even without explicitly detecting saddles or Hessian curvature.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Stochastic gradients leave a noise floor

Take a one-dimensional quadratic whose centre is noisy: $\hat L(x) = \tfrac12 a (x - c)^2$ with $c$ drawn fresh at every step, mean $0$ and variance $\sigma^2$. SGD with step $\gamma$ moves to $x_{t+1} = (1 - \gamma a)\,x_t + \gamma a\, c_t$. The mean goes to $0$, but every step injects new noise, so the iterate never settles. Its variance $V$ stops changing when the shrinkage balances the injection:

$$V = (1 - \gamma a)^2 V + \gamma^2 a^2 \sigma^2 \quad\Longrightarrow\quad V = \frac{\gamma^2 a^2 \sigma^2}{1 - (1 - \gamma a)^2}.$$

With $a = 1$, $\gamma = 0.5$, $\sigma^2 = 1$ this gives $V = 0.25 / 0.75 = 1/3$. This is the **stationary variance**: the width of the cloud the iterates keep wandering in. It does not shrink with more steps at a fixed step size.

### 2. Expected loss = mean error + spread + irreducible noise

On the same quadratic, the expected loss of an iterate $\theta$ splits into three parts (this is the paper's Eq. 5, one coordinate at a time):

$$\mathbb{E}\big[\hat L(\theta)\big] = \tfrac12 a \big(\mathbb{E}[\theta]^2 + \mathbb{V}[\theta] + \sigma^2\big).$$

Once the mean has converged, only the **variance** term is still under the optimizer's control; $\sigma^2$ comes from the data. So the loss an optimizer settles at, its **steady-state risk**, is lower exactly when its stationary variance is lower. This is the quantity Proposition 2 compares for SGD and Lookahead.

### 3. Interpolating between two points cancels an oscillation

In a sharp direction a large step overshoots: with $a = 1.9$ and $\gamma = 1$, each GD step multiplies the error by $1 - \gamma a = -0.9$, so the iterate flips sign every step and shrinks slowly. After $k = 5$ steps the error is multiplied by $(-0.9)^5 \approx -0.590$. Lookahead then moves the slow weights only a fraction $\alpha$ of the way to that endpoint, so one round multiplies the error by

$$1 - \alpha + \alpha\,(1 - \gamma a)^k = 0.5 + 0.5 \times (-0.590) \approx 0.205 \qquad (\alpha = 0.5).$$

Stopping halfway between a point and its overshoot lands near the middle of the oscillation. This deterministic toy only shows the mechanism; the paper's own deterministic analysis (Sec. 3.2) wraps classical momentum, not plain GD.

### 4. An exponential moving average forgets geometrically

The update $\phi \leftarrow \phi + \alpha(\theta - \phi) = (1 - \alpha)\phi + \alpha\theta$ is an **exponential moving average (EMA)**. Unrolled, the newest input gets weight $\alpha$, the one before $\alpha(1 - \alpha)$, then $\alpha(1 - \alpha)^2$, and so on. With $\alpha = 0.5$ the weights are $0.5, 0.25, 0.125, \dots$; after three rounds they add up to $0.875$, and the starting value keeps the remaining $(1 - \alpha)^3 = 0.125$. No past iterate has to be stored: the running value carries the whole history.

:::key Putting the four ideas together
SGD on a noisy problem settles into a cloud whose width sets its steady-state loss (ideas 1 and 2). Lookahead runs the inner optimizer for $k$ steps and then moves a slow copy of the weights only partway toward the endpoint, which damps overshoot (idea 3) and averages the noisy endpoints with geometric weights (idea 4). On the paper's noisy quadratic this gives a strictly smaller stationary variance than SGD at the same learning rate, made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. The algorithm in one picture: fast steps, slow decision

A conventional optimizer updates one parameter vector after each minibatch. Lookahead instead maintains two:

- **Fast weights** $\theta_{t,i}$: updated $k$ times by an inner optimizer $A$ such as SGD or Adam.
- **Slow weights** $\phi_t$: updated once after those $k$ steps, then used to restart the next fast trajectory.

The outer update (Algorithm 1, Eq. 1) is

$$\boxed{\phi_{t+1} = \phi_t + \alpha(\theta_{t,k} - \phi_t), \qquad 0 < \alpha \le 1.}$$

If $\alpha = 1$ it accepts the final fast weights; if $\alpha$ is small it moves cautiously toward them.

### Worked numerical example: $k = 3$, $\alpha = 0.5$

Take scalar starting slow weight $\phi_0 = 10$. Suppose the inner optimizer proposes fast weights 9, 7, then 8 (three potentially noisy steps). Lookahead ignores the individual steps when computing the slow update and uses the final fast weight 8:

$$\phi_1 = 10 + 0.5(8 - 10) = 9.$$

The next group of three fast steps *restarts at 9*, not 8. If the next group ends at 7, the new slow weight is $\phi_2 = 9 + 0.5(7 - 9) = 8$. Thus the fast path explores while the slow path averages accepted progress.

### Exponential smoothing without storing every iterate

Repeated substitution gives

$$\phi_t = (1 - \alpha)^t \phi_0 + \alpha \sum_{j=0}^{t-1} (1 - \alpha)^{t-1-j}\,\theta_{j,k}.$$

Recent final fast proposals receive larger weights. This is an exponential moving average *of final inner-loop endpoints*, not an average of every SGD iterate.

:::key Core intuition
Use fast exploration to find direction, then move the stable anchor only partway. This can damp stochastic oscillation while preserving progress in gentler directions.
:::

## 2. What is actually proved or measured?

### Noisy quadratic analysis (Section 3.1)

The paper considers stochastic quadratic losses $\hat L(x) = \tfrac12 (x - c)^\top A (x - c)$ with noise assumptions that permit a joint eigenbasis for $A$ and the noise covariance. For matching SGD step sizes in a stable range, Proposition 2 derives explicit stationary covariance matrices for SGD and Lookahead; under the analyzed conditions Lookahead yields smaller expected steady-state risk. This is *not* a theorem that Lookahead always beats Adam or SGD on arbitrary nonconvex functions.

### Worked variance intuition

Let a scalar fast endpoint be $\theta_{t,k} = \mu + \epsilon_t$, with independent mean-zero fluctuations of variance $\sigma^2$, and suppose it is independent of the current slow-weight noise. An EMA $\phi_{t+1} = (1 - \alpha)\phi_t + \alpha(\mu + \epsilon_t)$ has stationary variance

$$V = (1 - \alpha)^2 V + \alpha^2 \sigma^2 \quad\Rightarrow\quad V = \frac{\alpha}{2 - \alpha}\,\sigma^2.$$

For $\alpha = 0.5$, $V = \tfrac13 \sigma^2$. This independent-endpoint approximation is a *teaching model*, not the paper's coupled fast/slow covariance formula.

### Empirical evidence (Section 5)

Lookahead wrapped around SGD or Adam is evaluated on CIFAR-10/100, ImageNet, Penn Treebank language models and WMT translation. For the reported CIFAR comparison (Table 1), Lookahead achieves $95.27 \pm 0.06\%$ and $78.34 \pm 0.05\%$ on CIFAR-10 and CIFAR-100, respectively. The differences from SGD in that table are small, so do not present them as dramatic universal improvements.

:::warn Not a saddle-escape theorem
The contribution is stabilizing noisy training updates and improving robustness to hyperparameters. It offers no general approximate second-order-stationary-point guarantee.
:::

## 3. Contributions, assumptions, limitations, and survey relevance

**Contributions.** A flexible two-timescale optimization wrapper; an exact slow-weight exponential-average expression; stationary-risk results for a noisy quadratic; and empirical demonstrations of stable training and improved hyperparameter robustness across tasks. **Assumptions.** The implementation uses two parameter copies, inner-loop length $k$ and slow interpolation $\alpha$. The formal risk analysis relies on a quadratic model, stable learning rates and covariance/curvature alignment assumptions. Experiment-specific comparisons use particular datasets, models and tuning choices. **Limitations.** Lookahead needs an extra weight copy and synchronization; conclusions from the noisy quadratic do not prove a global convergence rate for general neural networks. Results are not direct saddle-escape guarantees, and reported advantages vary by task and baseline.

### Cross-paper connections

**Kalra (2025) and Galli (2026)** study how learning-rate choice creates sharpness and edge-of-stability phenomena; Lookahead offers a separate smoothing mechanism but does not prove elimination of EoS or chaos. **Wang et al. (Ref 77)** incorporates momentum in a cubic-regularized second-order optimizer; Lookahead is a generic wrapper over first-order or adaptive methods. **Staib et al. (Ref 89)** analyzes adaptive preconditioners for saddle escape; Lookahead reduces update variance without that guarantee.

### Quick reference

| Quantity | Interpretation |
|---|---|
| $k$ | Number of inner fast-optimizer updates. |
| $\theta_{t,k}$ | Final fast weights of round $t$. |
| $\alpha$ | Slow-weight interpolation fraction. |
| $\phi_{t+1} = \phi_t + \alpha(\theta_{t,k} - \phi_t)$ | Slow-weight outer update. |
| Noisy quadratic | Setting for the paper's variance/risk analysis. |

### Papers Master note

:::plain
Lookahead (Zhang et al., 2019) wraps SGD/Adam with $k$ fast optimizer steps followed by slow interpolation of weights. It reduces steady-state variance in an analyzed noisy quadratic model and often improves empirical training stability, but provides no general saddle-escape or global-convergence theorem.
:::

> **Source:** [NeurIPS 2019 original](https://proceedings.neurips.cc/paper_files/paper/2019/file/90fd4f88f588ae64038134f1eeaa023f-Paper.pdf), Algorithm 1, Eqs. (1)–(2), Proposition 2, Table 1. Numerical and EMA examples pedagogical.
