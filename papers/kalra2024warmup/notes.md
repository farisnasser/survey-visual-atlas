# Why Warmup the Learning Rate? Underlying Mechanisms and Improvements

:::minute The paper in one minute
**Question:** Why does gradually increasing the learning rate help SGD and Adam, and can warmup be shortened?

**Model/Method:** Experiments across FCNs, ResNets and Transformers, with curvature/sharpness measurements and SGD/Adam variants.

**Finding:** The dominant empirical benefit is enabling larger *target* learning rates; early changes in ordinary or preconditioned sharpness make those rates tolerable. Catapult-informed starts and GI-Adam can reduce warmup needs.

**Why we care:** Warmup is partly a curvature-management intervention, not merely a ritual to "estimate early moments."
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. A schedule moves the stability line during training

A **learning-rate schedule** chooses a step size $\eta_t$ for every step $t$. **Linear warmup** raises it from $\eta_{\text{init}}$ to $\eta_{\text{trgt}}$ over $T_{\text{wrm}}$ steps, $\eta_t = \eta_{\text{init}} + (\eta_{\text{trgt}} - \eta_{\text{init}})\,t/T_{\text{wrm}}$, then holds it. With $\eta_{\text{init}} = 0.01$, $\eta_{\text{trgt}} = 0.3$ and $T_{\text{wrm}} = 100$, step $50$ uses $\eta_{50} = 0.155$. On a quadratic, GD is stable while $\eta\lambda < 2$, where $\lambda$ is the largest curvature (the **sharpness**). Warmup therefore moves the line $2/\eta_t$ down slowly, from $200$ to about $6.7$ here, while the network's own sharpness also changes.

### 2. A catapult: an unstable step that lands somewhere flatter

Take the two-weight model $L(u, v) = \tfrac12(uv - 1)^2$. Every point with $uv = 1$ is a minimum, and the sharpness there is $u^2 + v^2$, so $(1, 1)$ is the flattest one (sharpness $2$). Start near $(2.2, 0.45)$, where the loss is only $0.00005$ but the sharpness is about $5.0$, and run GD with $\eta = 0.6$, so $2/\eta \approx 3.3$. The step is too large for this spot: the loss shoots up to about $1.5$, the weights swing, and GD then settles at a minimum with sharpness about $2.0$, below $3.3$. This loss spike followed by a move to a flatter region is a **catapult**. It happens in this toy model as in the paper's networks; the paper measures it, the toy only shows the shape of the event.

### 3. Adam's first step ignores the gradient's size

Adam divides each coordinate of the step by a running root-mean-square of that coordinate's gradient. At the very first step the bias-corrected second moment is exactly $g_0^2$, so the step is $\eta\,\mathrm{sign}(g_0)$ in every coordinate (ignoring the tiny $\epsilon$). A coordinate with gradient $10$ and one with gradient $0.001$ both move by $\eta$. Measured in Adam's rescaled coordinates, small-gradient directions therefore look extremely sharp at the start. The paper's GI-Adam starts the second moment at $v_0 = g_0^2$ instead of $0$; with the usual bias correction its first step is $\sqrt{1 - \beta_2}$ times smaller, a factor $0.0316$ for $\beta_2 = 0.999$, which acts like a built-in warmup.

:::key Putting the three ideas together
Warmup lowers the stability line slowly (idea 1), giving the network time to move, sometimes through a catapult (idea 2), to regions flat enough for the target step. For Adam the relevant sharpness is measured in its rescaled coordinates, and it starts very high because the first steps ignore gradient size (idea 3). The notes below state what the paper's experiments establish about both.
:::
:::

:::divider Study notes
:::

## 1. What is warmup really doing?

A typical schedule increases $\eta_t$ from a small $\eta_{\text{init}}$ to a desired $\eta_{\text{trgt}}$ over $T_{\text{wrm}}$ steps. Intuitively, the starting loss landscape may be too sharp for the target step size. Initial updates can move the model to a better-conditioned region.

The paper isolates this from another often-repeated claim, that Adam simply needs time to estimate enough moments. It finds that the ability to run a *larger final learning rate* explains most observed warmup benefits in its experiments. A longer warmup at a fixed target rate tends to have smaller additional value.

## 2. Worked example: curvature changes the allowable step

For $L(z) = \tfrac12\lambda z^2$, the update factor is $1 - \eta\lambda$. The quadratic stability requirement is $\eta < 2/\lambda$. Suppose initially $\lambda = 20$, so $\eta < 0.1$ is the local threshold. A desired target rate $\eta = 0.3$ would be unstable in that fixed initial geometry. If early learning changes the local sharpness to $\lambda = 5$, the corresponding threshold becomes $2/5 = 0.4$, and $0.3$ is locally admissible.

**Interpretation:** Warmup need not make the target step intrinsically smaller. It can help the network *change the landscape it encounters* before applying that target step. These numbers are illustrative, not experimental measurements from the paper.

## 3. Two warmup regimes and the catapult idea

The authors distinguish **progressive sharpening** (sharpness initially increases) from **natural sharpness reduction** (sharpness initially decreases). If learning rate rises enough to violate a local stability boundary, the loss may briefly spike and sharpness may subsequently fall; the authors call such behavior a *catapult*. Warmup can make this transition more gradual, and a loss-based line search can choose a more useful starting learning rate.

For adaptive Adam, let $P_t$ denote the positive diagonal preconditioner implicit in the update $\Delta\theta = -\eta P_t g_t$. The curvature relevant to this update is the largest eigenvalue of

$$P_t^{1/2} H_t P_t^{1/2},$$

not simply $\lambda_{\max}(H_t)$. This *preconditioned sharpness* can start extremely high, even when the ordinary Hessian is not particularly sharp.

:::key GI-Adam: a concrete modification
Initialize Adam's second moment with the first squared gradient, $v_0 = g_0^2$, rather than zero. The authors show this reduces initial preconditioned sharpness and can act like an implicit warmup under their bias-correction convention.
:::

## 4. What experiments actually establish

Figures 1–4 compare warmup durations and achievable target learning rates across parameterizations, optimizers, models and datasets including CIFAR, TinyImageNet and WikiText. The main reported benefit is robust tolerance to larger $\eta_{\text{trgt}}$. For fixed target rate, improvements from longer warmup are usually smaller. In some configurations, a catapult-informed start shortens or eliminates warmup; GI-Adam improves failure boundaries in their tests.

These are *empirical mechanisms and algorithm proposals*, not a theorem guaranteeing all warmup schedules will improve training.

## 5. Why Adam needs a different curvature scale

Suppose $H = \mathrm{diag}(20, 1)$ and an illustrative positive preconditioner is $P = \mathrm{diag}(0.01, 1)$. Raw largest curvature is $20$, but the transformed matrix is

$$P^{1/2} H P^{1/2} = \mathrm{diag}(0.2, 1),$$

whose largest eigenvalue is $1$. The curvature relevant to a preconditioned update must be assessed in *rescaled coordinates*. This explains why judging Adam by ordinary Hessian sharpness alone can mislead. The paper finds the reverse initial problem can also occur: Adam's preconditioned sharpness may start very high, motivating warmup or modified initialization.

The toy matrices are pedagogical, not experimental measurements.

## 5. Assumptions and limitations

**Setting:** Specific SGD and Adam implementations, architectures, parameterizations, initialization scales, and experimental stability criteria. Linear/quadratic thresholds are only local intuitions for nonlinear training.

**Limits:** There is no universal optimal warmup duration or general convergence theorem for arbitrary neural networks. Catapult effects and GI-Adam improvements depend on task/hyperparameters; extra warmup can still help for reasons not isolated in the selected experiments.

## 6. Connections to the survey

**Li–Wang–Li (2022)** supplies a progressive-sharpening explanation; **Kalra et al. (2025)** later studies sharpness regimes via fixed-point dynamics. **Galli (2026)** explores adaptive large steps and flat saddle regions, while this paper focuses on the *warmup path* to a larger target rate.

## 7. Quick reference

| Concept | Meaning / equation |
|---|---|
| Linear warmup | $\eta_t : \eta_{\text{init}} \to \eta_{\text{trgt}}$ |
| GD quadratic threshold | $\eta\lambda_{\max}(H) < 2$ |
| Adam curvature | $\lambda_{\max}(P^{1/2} H P^{1/2})$ |
| GI-Adam initialization | $v_0 = g_0^2$ |
| Original to read | Sections 3–5, Figures 1–4 |

## 8. Papers Master note

:::plain Copy into Our notes
Finds empirically that warmup chiefly permits larger target learning rates by managing ordinary or preconditioned sharpness. Distinguishes progressive sharpening from initial sharpness reduction; proposes catapult-informed starts and GI-Adam. No universal training guarantee.
:::

> **Original paper:** [View original research PDF](https://papers.neurips.cc/paper_files/paper/2024/file/ca98452d4e9ecbc18c40da2aa0da8b98-Paper-Conference.pdf) *Worked numerical examples are pedagogical unless explicitly attributed to the authors.*
