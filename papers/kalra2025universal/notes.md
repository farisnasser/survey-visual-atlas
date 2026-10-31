# Why Neural-Network Sharpness Changes During Training

:::minute The paper in one minute
**Question:** Why does neural-network sharpness sometimes fall, later rise, and eventually oscillate near the edge of stability? Can it become chaotic?

**Model/Method:** Study discrete gradient descent on a two-layer *linear* UV network with one training example. Rewrite its updates using a residual and a curvature-related variable, then analyze fixed points and their stability.

**Finding:** Initial conditions and parameterization determine whether training shows early sharpness reduction, progressive sharpening, and edge-of-stability oscillations. Above a model-specific learning-rate threshold, an oscillatory manifold can become attracting; increasing the learning rate can produce period doubling and chaos.

**Why we care:** Sharpness is a *dynamical variable*, not a fixed property of the loss landscape. Large-step training can oscillate without immediately diverging, but the toy model's dynamics do not transfer unchanged to real networks.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Two numbers can summarize all the weights

GD updates every weight, but the loss on one example depends only on the **residual** $r = f - y$. How fast $r$ responds to a step is set by $\lambda = \|\nabla_\theta f\|^2$, the squared size of the output's gradient (the scalar **neural tangent kernel**, which the paper writes as $\lambda := \operatorname{Tr} H$ and uses as its curvature proxy). To first order one GD step changes the output by $-\eta\lambda r$. For $f(u, v) = uv$, $y = 1$, $u = v = 0.1$ and $\eta = 0.2$: $\lambda = 0.02$, $r = -0.99$, so the first-order change is $+0.00396$. The actual new output is $0.1198^2 \approx 0.014352$, a change of $0.004352$; the extra $0.000392$ is exactly the second-order term $\eta^2 r^2 (r + y)$ of the paper's Equation 1. The pair $(r, \lambda)$ is a closed description: no other information about the weights is needed.

### 2. Fixed points, their stability, and period doubling

A map $z_{t+1} = g(z_t)$ has a **fixed point** where $g(z^*) = z^*$. Near it the error is multiplied by $g'(z^*)$ every step, so the fixed point attracts when $|g'(z^*)| < 1$ and repels when $|g'(z^*)| > 1$. When the multiplier passes $-1$, the iterates start alternating between two values: a **two-cycle**. Take the notes' cubic GD map $z_{t+1} = z_t + \eta z_t(1 - z_t^2)$, with multiplier $1 - 2\eta$ at $z^* = 1$. At $\eta = 0.9$ it settles at $1$. At $\eta = 1.1$ it alternates between $0.815$ and $1.116$. At $\eta = 1.25$ it cycles through four values. At $\eta = 1.35$ it wanders without a short repeating cycle. This cascade, $1 \to 2 \to 4 \to \cdots \to$ irregular, is the **period-doubling route to chaos**; a plot of the late-time values against $\eta$ is a **bifurcation diagram**.

### 3. Width and parameterization move the threshold

The UV model writes $f = v^\top U x/\sqrt{n_{\text{eff}}}$ with effective width $n_{\text{eff}} = n^{1-p}$. The choice $p = 0$ (the NTK parameterization, NTP) gives $n_{\text{eff}} = n$; $p = 1$ ($\mu$P) gives $n_{\text{eff}} = 1$. The paper's necessary condition for the edge of stability is $\eta > \eta_c = \sqrt{n_{\text{eff}}}/(\|x\|y)$. With $\|x\| = 1$, $y = 2$ and width $n = 512$: $\eta_c \approx 11.3$ for $p = 0$ but $\eta_c = 0.5$ for $p = 1$. The same architecture can reach the edge of stability easily in one parameterization and not at all in the other, if training diverges before $\eta_c$.

:::key Putting the three ideas together
The UV model's GD reduces exactly to a two-number map in $(r, \lambda)$ (idea 1). Its fixed points and their stability decide whether sharpness falls, rises or settles, and on the edge-of-stability line the map becomes one-dimensional and period-doubles as $\eta$ grows (idea 2). Where the threshold sits depends on width and parameterization (idea 3). The notes below work through the model and say which parts carry over to real networks.
:::
:::

:::divider Study notes
:::

## 1. Start with the four training regimes

The paper organizes training into four stages. *Not every network exhibits all four*: initialization scale, learning rate, and parameterization change which stages appear.

| Stage | What happens to Hessian sharpness $\lambda_{\max}(H)$? |
|---|---|
| T1: early transient | Can decrease; at large step sizes loss and sharpness may temporarily spike (a "catapult"). |
| T2: saturation | Sharpness changes little for a while. |
| T3: progressive sharpening | Sharpness grows as the network moves through parameter space. |
| T4: edge of stability (EoS) | Sharpness fluctuates near the classical quadratic boundary $2/\eta$. |

## 2. The first mathematical intuition: why $2/\eta$ appears

Take the simple local quadratic $L(z) = \tfrac12 a z^2$ with positive curvature $a$. Gradient descent makes

$$z_{t+1} = z_t - \eta a z_t = (1 - \eta a)z_t.$$

For the coordinate to contract, $|1 - \eta a| < 1$, which means $0 < \eta a < 2$. Exactly at $\eta a = 2$, the multiplier is $-1$: the coordinate flips sign on every step without contracting in the ideal quadratic. Above 2, its magnitude grows.

:::warn Important precision
This $2/\eta$ boundary follows from a *fixed quadratic approximation*. Neural networks are nonquadratic; their curvature changes during the update. Oscillation at EoS does *not* by itself mean the full neural-network training run has diverged.
:::

## 3. The UV network: simple input-output mapping, nonlinear weight dynamics

The authors use a two-layer linear network with input $x \in \mathbb{R}^d$ and one target $y$:

$$f(x; U, v) = \frac{v^\top U x}{\sqrt{n_{\text{eff}}}}, \qquad n_{\text{eff}} = n^{1-p}, \quad p \in [0, 1], \qquad \mathcal{L} = \tfrac12(f - y)^2.$$

Here $U$ and $v$ are both trainable. The input-output function is linear in $x$, but the product $v^\top U$ makes GD nonlinear in the separate weights. The parameter $p$ interpolates between parameterization regimes; $n_{\text{eff}}$ is the *effective width*.

## 4. Worked example: curvature changes even when the model is linear

Use one scalar input $x = 1$, one unit ($n_{\text{eff}} = 1$), and let $f(u, v) = uv$ with $y = 1$. Then

$$\mathcal{L}(u, v) = \tfrac12(uv - 1)^2, \qquad \frac{\partial\mathcal{L}}{\partial u} = (uv - 1)v, \qquad \frac{\partial\mathcal{L}}{\partial v} = (uv - 1)u.$$

**Step 1.** Take $u = v = 0.10$ with $\eta = 0.2$. The prediction is $uv = 0.01$ and residual $r = -0.99$. Both updates are

$$u^+ = v^+ = 0.10 - 0.2(-0.99)(0.10) = 0.1198.$$

**Step 2.** Here the scalar tangent-kernel quantity is $\lambda = u^2 + v^2$; it rises from $0.0200$ to $2(0.1198)^2 \approx 0.02870$. The curvature proxy *increases* even as the loss falls.

**Step 3.** Contrast $u = v = 2$ with $\eta = 0.1$. Now $r = 3$ and $u^+ = v^+ = 2 - 0.1(3)(2) = 1.4$. The same quantity decreases from $\lambda = 8$ to $\lambda^+ = 3.92$: *reduction of the curvature proxy*. Thus its direction of change depends on *where training starts*.

:::warn What the toy example does — and does not — show
It illustrates why a product-parameterized linear network can sharpen or flatten. It is *not* a proof that all realistic networks follow those trajectories, nor is $u^2 + v^2$ always equal to the top Hessian eigenvalue. In the paper the authors analyze $\lambda = \operatorname{Tr} H$ as a *proxy* for measured sharpness.
:::

## 5. Reduce an enormous parameter problem to two variables

Let the residual be $r_t = f_t - y$ and let

$$\lambda_t = \frac{\|Ux\|^2 + \|v\|^2\|x\|^2}{n_{\text{eff}}}.$$

The full UV dynamics can be represented by the following *closed* update equations in $(r_t, \lambda_t)$:

$$r_{t+1} = r_t\left(1 - \eta\lambda_t + \frac{\eta^2\|x\|^2}{n_{\text{eff}}} r_t(r_t + y)\right),$$

$$\lambda_{t+1} = \lambda_t + \frac{\eta\|x\|^2}{n_{\text{eff}}} r_t^2\left(\eta\lambda_t - 4\frac{r_t + y}{r_t}\right).$$

These are the paper's key technical reduction (Equations 1–2). By studying where both updates vanish, the authors identify fixed points and a zero-loss fixed line. They use the stability of these sets to explain when curvature falls, rises, or oscillates.

## 6. Why a saddle can cause reduction, then sharpening

In the two-variable diagram, the point $(r, \lambda) = (-y, 0)$ corresponds to all UV parameters being zero. It is a saddle of the gradient dynamics. Some initializations move *toward* that region: weight norms and the curvature proxy decrease. As training later leaves the saddle neighborhood, the curvature proxy starts to increase. The resulting pattern is

$$\boxed{\text{approach saddle} \;\longrightarrow\; \text{sharpness reduction} \;\longrightarrow\; \text{plateau} \;\longrightarrow\; \text{progressive sharpening.}}$$

This is one mechanism, not a universal claim that every loss plateau is caused by a saddle.

## 7. A critical step size for the edge of stability

The zero-loss fixed line ($r = 0$) has stable points where $\eta\lambda < 2$. The paper derives a *necessary* condition for its UV model to exhibit EoS:

$$\boxed{\eta > \eta_c, \qquad \eta_c = \frac{\sqrt{n_{\text{eff}}}}{\|x\|y}} \qquad (y > 0).$$

**Worked numerical check.** With $n_{\text{eff}} = 16$, $\|x\| = 1$, and $y = 2$, the threshold is $\eta_c = 4/2 = 2$.

- At $\eta = 1$, this necessary condition is *not* met, so the proposed UV EoS mechanism is excluded.
- At $\eta = 2.5$, the condition is met, but *EoS is not guaranteed*: training must still remain in a trainable regime and encounter the corresponding attracting dynamics.

For $\eta > \eta_c$, the paper analyzes a line segment in the $(r, \lambda)$ plane that acts as an *EoS attractor*. Rather than settling at a fixed point on the zero-loss line, the discrete trajectory can keep oscillating nearby.

:::warn Do not confuse a necessary condition with a sufficient guarantee
The paper also notes a maximum trainable learning rate $\eta_{\max}$ dependent on initialization. If $\eta_{\max} < \eta_c$, then EoS cannot be observed before divergence.
:::

## 8. Where the "route to chaos" comes from

On the EoS manifold the model reduces further to a *one-dimensional cubic map* in the residual. Increasing $\eta$ changes the map's attracting trajectory: one repeated value can become a two-cycle, then four values, then more complicated irregular fluctuations. This is called *period doubling*.

For an intuition-only analogue, consider $F(z) = \tfrac14(z^2 - 1)^2$ and perform GD:

$$z_{t+1} = z_t - \eta F'(z_t) = z_t + \eta z_t(1 - z_t^2).$$

The map is cubic. Around the solution $z^* = 1$, its local multiplier is $1 - 2\eta$; at $\eta = 1$ it reaches $-1$, the threshold where a fixed point can lose stability through a flip bifurcation. *This illustrative cubic is not the UV paper's exact Equation 3.*

The paper's *actual* EoS-manifold map is

$$r_{t+1} = r_t + \frac{\eta r_t}{\eta_c y}\left(\frac{\eta r_t}{\eta_c y} - 2\right)(r_t + y).$$

The bifurcation diagram in its Figure 3 shows the period-doubling route in the UV setting as learning rate rises.

## 9. What is established, and what is observed?

| Evidence type | Takeaway |
|---|---|
| **Toy-model analysis** | Closed two-variable UV dynamics; classification of fixed points; critical EoS condition and EoS-manifold cubic dynamics. |
| **Neural-network observations** | FCNs/CNNs/ResNets display parameterization-dependent sharpness trends; Figure 4 shows an observed association between EoS and test accuracy in the investigated setting. |
| **Synthetic versus real data** | Clear period doubling appears for certain synthetic data; real-image data produce broad, correlated bands of sharpness fluctuations rather than the same clean bifurcation pattern (Figure 5). |

## 10. Contributions, assumptions, limitations

### Main contributions

- Unifies early sharpness reduction, intermediate saturation, progressive sharpening, and EoS in one analytically tractable model.
- Explains how fixed-point locations, stability, initialization, and parameterization change the training trajectory.
- Derives an EoS threshold in the UV model and relates its oscillations to a cubic-map period-doubling route.
- Tests selected model predictions in larger architectures on real and synthetic datasets.

**Important assumptions** Exact analytic dynamics use a two-layer *linear* network, one training example, squared loss, and *full-batch, discrete* GD. The main critical threshold assumes $y > 0$; initialization and effective width influence which dynamics are accessible.

**Limitations** The UV model does not explain nonmonotone *decrease* of loss at EoS in realistic networks, does not reproduce long-range correlations of real-data sharpness fluctuations, and can miss a loss catapult at small effective width. The clean route to chaos is not established for arbitrary nonlinear networks on real datasets. This is not a universal convergence theorem or a prescription that chaotic training improves generalization.

## 11. Survey relevance and reading priority

**IV — Dynamics:** This is a useful recent mechanistic treatment of step size, initialization, EoS, and nonmonotone trajectories. **III — Geometry:** It links Hessian statistics and saddle neighborhoods to dynamical motion. **VI / theory versus practice:** It sharply illustrates the gap between a controlled toy model and real architectures.

**Read first:** Sections 3–5 and Figures 1–3 (especially Equations 1–3, Result 1, and Figure 3). **Then skim:** Section 6 and Figures 4–5. **Refer to appendices:** only for the details of the threshold, EoS attractor, and experiments.

## 12. Quick reference and Papers Master note

| Term | Meaning |
|---|---|
| Sharpness | $\lambda_{\max}(H)$, the largest eigenvalue of the loss Hessian. |
| UV curvature proxy | $\lambda = \operatorname{Tr} H$, analyzed as a tractable proxy; not identical to the top eigenvalue in general. |
| EoS boundary | $\eta\lambda_{\max}(H) \approx 2$ (classical positive-curvature quadratic stability threshold). |
| UV EoS condition | $\eta > \sqrt{n_{\text{eff}}}/(\lVert x\rVert y)$; *necessary*, not sufficient. |
| Period doubling | Fixed behavior changes to 2-cycle, 4-cycle, etc. as the learning rate increases. |

:::plain Papers Master — short note
The UV model explains sharpness reduction, progressive sharpening, and EoS through fixed-point stability and parameterization-dependent trajectories. It derives a necessary EoS learning-rate threshold and exhibits period doubling/chaos in a controlled setting; realistic networks show only partially matching, data-dependent behavior.
:::

> **Original paper:** [Kalra, He & Barkeshli, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c8cd1b78cdae751265c88efc136e5bd-Paper-Conference.pdf). Mathematical examples in this guide are pedagogical adaptations; model results and limitations are checked against the authors' paper.
