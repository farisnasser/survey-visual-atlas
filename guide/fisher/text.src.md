## 1. Three matrices that answer three different questions

This chapter follows Sections IV-H, IV-I and IV-J of the survey. Papers often call any matrix that multiplies a gradient "second order". The survey separates three of them, each answering its own question:

- **Hessian:** how does the gradient of *this loss* change when the parameters move?
- **Generalized Gauss–Newton:** what curvature arises when parameter changes move the model's *outputs*, which then pass through the loss?
- **Fisher information:** how sensitively does the model's *predicted probability distribution* change when the parameters move?

**The Hessian**, $H = \nabla^2L(\theta)$, is the curvature of the training objective itself. For $L(x) = x^2$, $H = 2$, a bowl; for $L(x) = -x^2$, $H = -2$, a hill. It can be positive in some directions and negative in others, which is exactly what made undamped Newton unsafe in chapter 10.

**The [[ggn|generalized Gauss–Newton]] matrix.** A network's loss is a composition: parameters $\theta$ produce outputs $f_\theta(x)$, and a loss $\ell$ scores them. The full Hessian contains both the loss's curvature in the outputs and the network's own second derivatives. The GGN keeps only the first, structured part, $G = J^\top H_\ell J$, where $J$ is the Jacobian of the outputs with respect to the parameters and $H_\ell$ the loss's curvature in the outputs. When the loss is convex in its outputs, as squared error and cross-entropy are, $G$ is positive semidefinite: it cannot produce the downward directions of a general Hessian. That makes solving with it better behaved, at the price of omitting some information.

**The [[fisher|Fisher information]]** belongs to probabilistic models. Picture a binary classifier predicting $P(y = 1) = 0.51$. One small parameter change moves that to $0.52$; another moves it to $0.95$. The second changes the predicted distribution far more. The Fisher matrix encodes this local sensitivity:

$$F(\theta) = \mathbb{E}_{x,\;y\sim p_\theta(\cdot\mid x)}\Big[\nabla_\theta\log p_\theta(y\mid x)\,\nabla_\theta\log p_\theta(y\mid x)^\top\Big],$$

an average of outer products of score gradients, with the labels drawn *from the model*. What distribution the average uses matters.

**Why the empirical Fisher needs care.** Averaging the same outer products with the *observed* labels gives the **[[empirical-fisher|empirical Fisher]]**, a common shortcut. It is not, in general, the Fisher, the GGN, or the Hessian. Special identities hold under particular assumptions, but a squared-gradient statistic should not simply be renamed "curvature".

| matrix | built from | always $\succeq 0$? | relation to $H$ |
|---|---|---|---|
| Hessian $H$ | $\nabla^2 L$ | no | itself |
| Gauss–Newton $G$ | $J^\top H_\ell J$ | yes, when $H_\ell \succeq 0$ | $H$ without the network's own second derivatives |
| Fisher $F$ | score outer products, labels from the model | yes | related only through identities in special cases |
| empirical Fisher | gradient outer products, observed labels | yes | not $F$, $G$ or $H$ in general |

:::key Remember IV-H
Hessian = curvature of the loss. GGN = a structured output-loss curvature term. Fisher = geometry of the predicted distributions. Matrices of the same shape do not automatically encode the same information.
:::

## 2. Natural gradient: measure steps by how much the predictions change

Ordinary gradient descent measures a step with Euclidean distance in parameter space: $0.1$ in one weight counts like $0.1$ in another. But a predictor cares about how much its outputs change. A change of $0.01$ in one weight may swing a predicted probability a lot, while $0.1$ in another barely moves anything. Equal-length steps can have very unequal effects.

**[[natural-gradient|Natural gradient]]** fixes this with the Fisher matrix as the metric of chapter 6's variable-metric step (survey Eq. 35):

$$\Delta_t = -\eta F(\theta_t)^{-1}g_t.$$

Here $g_t$ is the gradient of the loss and $F$ says which parameter movements are large or small *in predictive terms*. The inverse is notation: if $F$ is singular or badly conditioned, real methods damp it, use a pseudoinverse, or solve approximately.

**A numerical picture.** Purely for illustration, take $g = (2, 2)^\top$, $F = \mathrm{diag}(100, 1)$ and $\eta = 0.1$. Ordinary GD gives $\Delta = (-0.2, -0.2)^\top$. Natural gradient computes $F^{-1}g = (0.02, 2)^\top$, so $\Delta = (-0.002, -0.2)^\top$. The first weight moves a hundred times less, because this $F$ says the predictions are very sensitive in that direction.

**Not Newton with another name.** Both transform the gradient with a matrix, but Newton uses the curvature of the chosen loss, while natural gradient uses the geometry of the model's distributions. Its main theoretical appeal is that a smooth change of coordinates for the same model, for example from $\sigma$ to $\log\sigma$, does not change the ideal direction in distribution space, to first order. Finite steps, damping and approximations weaken this property.

@@viz natgrad@@

For a layer with millions of weights, the full Fisher matrix is far too large to store or invert. That is what K-FAC addresses in chapter 13.

:::key Remember IV-I
Natural gradient asks "how much would this step change the model's predictions?", not just "how far does the parameter vector move?".
:::

## 3. What the theory really guarantees

Three claims sound alike but are not:

- **"Newton converges quadratically."** A *local* result near a suitable minimum, under smoothness and nonsingularity assumptions; not a claim about arbitrary network initializations.
- **"BFGS can converge superlinearly."** Under appropriate conditions and line searches, its progress accelerates near the solution; noisy minibatch implementations do not automatically inherit this.
- **"Trust-region and cubic methods have non-convex guarantees."** These theorems reach a kind of *stationarity*, under smoothness assumptions and access to derivatives or approximate subproblem solutions; they do not promise the global optimum.

**Fewer iterations, not necessarily less time.** If method A needs $20$ updates at $3$ seconds each and method B needs $100$ updates at $0.1$ seconds each, then $T_A = 20\times3 = 60$ s and $T_B = 100\times0.1 = 10$ s. A takes five times fewer steps and six times longer.

Theorems also answer different questions: one bounds the loss gap of a convex function, another the gradient norm of a non-convex one, a third only applies once close to a minimum. Comparing their rates without this context is misleading, and a lower training loss does not by itself mean better test accuracy.

| stage | question | main idea | remaining limitation |
|---|---|---|---|
| Newton | can curvature choose a better step? | minimize a local quadratic model | Hessian may be unreliable or expensive |
| local convergence | when is Newton exceptionally fast? | quadratic error reduction near a minimum | only local, under assumptions |
| damping | what if curvature is negative? | shift the Hessian to a safer model | damping must be chosen |
| trust region, cubic | how far to trust the model? | restrict or penalize large steps | costlier subproblems |
| BFGS | curvature without second derivatives? | learn from $(s, y)$ pairs | large approximation; noise |
| L-BFGS | less storage? | keep only recent pairs | approximation quality, line searches |
| Hessian-free | curvature without a stored matrix? | Hessian–vector products and CG | repeated inner iterations |
| GGN, Fisher | which matrix is really used? | separate loss curvature from prediction geometry | different meanings and assumptions |
| natural gradient | distance as change in predictions? | precondition with the Fisher | full Fisher too expensive |

:::key Remember IV-J
Read a convergence theorem together with its objective class, its starting assumptions, the quantity it bounds, and the cost of each step. Theory says what is possible; fair experiments establish the practical trade-offs.
:::

:::problem The next step
Full curvature matrices are too large for real layers. Section V asks how the matrix shape of a layer can make these ideas affordable, starting with K-FAC in chapter 13.
:::
