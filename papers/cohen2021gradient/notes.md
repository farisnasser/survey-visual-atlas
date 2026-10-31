# Gradient Descent at the Edge of Stability

:::minute The paper in one minute
**Problem:** Classical convergence arguments assume that a sufficiently small gradient-descent step leads to a steady decrease in loss. Do neural networks actually train in this small-step regime?

**Method:** The authors train networks by *full-batch gradient descent*, measuring both loss and the top eigenvalue of the loss Hessian throughout training.

**Main finding:** The top eigenvalue often rises to around $2/\eta$ and hovers just above it while short-term loss fluctuates yet decreases over longer periods. This is an *empirical observation*, not a general convergence theorem.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Sharpness can be measured without forming the Hessian

The **sharpness** is the largest Hessian eigenvalue $\lambda_{\max}$. A network has far too many parameters to write the Hessian down, but **power iteration** needs only Hessian-vector products: repeat $v \leftarrow Hv/\|Hv\|$ and read off $v^\top Hv$. For $H = \mathrm{diag}(20, 1)$ started at $v = (1, 1)$, the estimate is $19.95$ after one step and $19.9999$ after two. The paper tracks sharpness this way along whole training runs.

### 2. Two step-size thresholds: $1/\ell$ and $2/\lambda$

If the gradient is $\ell$-Lipschitz, one GD step changes the loss by at most $-\eta\big(1 - \tfrac{\ell\eta}{2}\big)\|\nabla L\|^2$, a guaranteed decrease for $\eta < 2/\ell$ and the textbook choice $\eta \le 1/\ell$. On a quadratic, the eigen-direction with curvature $\lambda$ is multiplied by $1 - \eta\lambda$ per step, which is stable exactly when $\eta\lambda < 2$. The paper's own quadratic example (its Fig. 2) has curvatures $20$ and $1$ with $\eta = 0.11$, so $2/\eta \approx 18.2$: the multipliers are $-1.2$ and $0.89$, and GD diverges along the sharp direction while converging along the flat one.

### 3. Sharpness depends on where you stand

In a network, the Hessian changes as the weights move. A two-parameter example shows it. For $L(u, v) = \tfrac12(uv - 1)^2$, every point with $uv = 1$ is a global minimum, and the Hessian there has eigenvalues $0$ and $u^2 + v^2$. With $\eta = 0.6$ the threshold is $2/\eta \approx 3.33$: the minimum $(2, 0.5)$ has sharpness $4.25$ and is unstable for GD, while $(1, 1)$ has sharpness $2$ and is stable. The same step size can be too large in one region and fine in another.

### 4. Just past the threshold, GD oscillates instead of exploding

When $\eta\lambda$ is slightly above $2$, the multiplier $1 - \eta\lambda$ is slightly below $-1$. The error flips sign every step and grows slowly: for $\eta\lambda = 2.02$ it grows by $2\%$ per step. The iterate bounces between the walls of a valley rather than leaving at once. On a fixed quadratic this still diverges; in a network, the bouncing moves the weights, and with them the curvature (idea 3).

:::key Putting the four ideas together
Measured sharpness (idea 1) tends to rise during training until it meets the stability threshold $2/\eta$ (idea 2). Past it, GD oscillates along the sharp direction (idea 4) instead of failing, and the curvature it sees keeps changing (idea 3). The paper documents that training then hovers at the edge while the loss still falls over longer periods; the notes below separate what the quadratic proves from what the experiments show.
:::
:::

:::divider Study notes
:::

## 1. Why the threshold $2/\eta$ appears

Start from a one-dimensional quadratic loss about its minimum $x^*$:

$$L(x) = \tfrac12 a(x - x^*)^2, \qquad a > 0.$$

GD with step size $\eta$ updates

$$x_{t+1} = x_t - \eta a(x_t - x^*).$$

Write the error $e_t = x_t - x^*$. Then

$$\boxed{e_{t+1} = (1 - \eta a)e_t, \qquad e_t = (1 - \eta a)^t e_0}.$$

Errors shrink if $|1 - \eta a| < 1$, i.e. $0 < \eta a < 2$. If $\eta a > 2$, the error changes sign each iteration *and grows*: oscillatory instability. For a positive definite Hessian $H$, decompose the local error into Hessian eigenvectors. Each coordinate scales by $(1 - \eta\lambda_i)$, so the limiting sharpness is $\lambda_{\max}(H) \approx 2/\eta$.

:::warn What this calculation proves:
A *fixed quadratic* becomes unstable when its largest eigenvalue exceeds $2/\eta$. It does *not* prove that nonlinear neural-network training diverges whenever the instantaneous Hessian crosses that threshold, because both the position and curvature change during training.
:::

## 2. What Cohen et al. actually measured

The paper defines *sharpness* precisely as the top Hessian eigenvalue,

$$S(\theta_t) = \lambda_{\max}(\nabla^2 L(\theta_t)).$$

For constant-step full-batch GD, experiments across several architectures and tasks reveal a recurring trajectory: (i) **progressive sharpening**, when $S(\theta_t)$ increases; (ii) **edge of stability**, when $S(\theta_t)$ stays around or slightly above $2/\eta$ even as training continues. Short-run loss need not decrease at every step. The empirical pattern challenges the idea that training always follows the descent-lemma regime $\eta \le 1/\ell$.

## 3. Interpret the local dynamics correctly

Near a changing parameter point, a quadratic Taylor approximation gives

$$L(\theta_t + \delta) \approx L(\theta_t) + \nabla L(\theta_t)^\top\delta + \tfrac12\delta^\top H_t\delta.$$

When $\eta S_t$ approaches 2, perturbations along the sharpest eigenvector oscillate strongly. But since $H_t$ changes as the weights move, a period of instability may be followed by lower sharpness. *The paper documents this feedback but does not establish a complete theorem for the mechanism*; later self-stabilization studies pursue that explanation.

:::key Connection to our survey.
This paper motivates Section VII: practical full-batch GD can succeed outside the regime assumed in many smooth, monotonic-descent analyses. The relevant variables are not just the gradient magnitude but also evolving curvature and the chosen step size.
:::

## 4. Assumptions, scope, and limits

The central evidence is experimental, with *full-batch* GD and prescribed step sizes. The authors explicitly caution against inferring an established link between their Hessian sharpness measure and generalization. The observed edge should not be stated as a universal law of minibatch SGD or Adam; in fact the paper notes that the sharpness for SGD need not flatline at the same threshold. The $2/\eta$ boundary is exact for fixed positive quadratic curvature, heuristic as a moving local marker for nonlinear training. **Our takeaway.** The striking message is not that GD becomes unstable and fails, but that it can train successfully while repeatedly approaching a boundary where its local quadratic approximation would become unstable.

> **Source:** [https://arxiv.org/pdf/2103.00065](https://arxiv.org/pdf/2103.00065) (esp. pp. 1–3 and experimental sections).
