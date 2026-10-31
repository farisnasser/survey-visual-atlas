# Global Optimality in Neural Network Training

:::minute The paper in one minute
**Problem:** The training of a neural net is non-convex in its layer parameters, but some overparameterized structures have a convex interpretation after regrouping their components. Haeffele and Vidal ask for *sufficient conditions* ensuring that certain local minima are global, and for a non-increasing path from an arbitrary starting parameterization toward a global solution.

**Main result:** Under structured parallel architecture, convex loss in the output, appropriate homogeneity/regularization, and sufficient model size, special local minima (notably those containing an inactive subnetwork) admit a global-optimality certificate. The paper also analyzes paths toward global solutions.

**Scope:** The result does *not* say all local minima are global for general ReLU/Transformer training. The global statement is conditional and depends on the network's parallel factorization and homogeneous regularizer.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the five ideas the paper stands on, built from scratch.
### 1. Bias-free ReLU networks are positively homogeneous

A function is **positively homogeneous of degree $p$** when scaling its input by $c \ge 0$ scales its output by $c^p$. Because $\mathrm{ReLU}(cz) = c\,\mathrm{ReLU}(z)$, a bias-free ReLU unit with input weights $u$ and output weight $v$, $\phi(u, v) = v\,\mathrm{ReLU}(u^\top x)$, has degree $2$ in $(u, v)$ jointly. Example: $x = (1, 2)$, $u = (1, 1)$, $v = 2$ gives $2 \cdot 3 = 6$; scaling both $u$ and $v$ by $c = 3$ gives $(3 \cdot 2)\,\mathrm{ReLU}(3 \cdot 3) = 54 = 3^2 \cdot 6$. Each layer contributes one factor of $c$, so a $K$-layer bias-free ReLU path has degree $K$. Weight decay $\tfrac12(\|u\|^2 + v^2)$ also has degree $2$: the degrees match.

### 2. Balancing the layers turns weight decay into a convex penalty

A product of parameters can be rescaled without changing the output: $(a, b) \to (ta, b/t)$ keeps $ab$ fixed. Among all such rescalings, weight decay is smallest when the two factors are balanced:

$$\min_{ab = x} \tfrac12(a^2 + b^2) = |x| .$$

Example: for $x = 4$ the minimum is at $a = b = 2$, with value $4$. The penalty seen *by the output* is $|x|$, which is convex, even though the problem in $(a, b)$ is not. Matched degrees are what make this work; the paper builds the general convex function of the output (its $\Omega_{\phi,\theta}$) the same way.

### 3. With matched degrees, the origin is a saddle

Take the smallest example: $f(u, v) = \tfrac12(1 - uv)^2 + \tfrac{\lambda}{2}(u^2 + v^2)$, output $uv$ of degree $2$, regularizer of degree $2$. Its Hessian at the origin is

$$\nabla^2 f(0, 0) = \begin{bmatrix} \lambda & -1 \\ -1 & \lambda \end{bmatrix}, \qquad \text{eigenvalues } \lambda - 1 \text{ and } \lambda + 1 .$$

For $\lambda < 1$ one eigenvalue is negative, so "all zero" is a saddle, not a trap. By idea 2 the problem is equivalent to the convex $\tfrac12(1 - x)^2 + \lambda|x|$, minimized at $x = 1 - \lambda$ with value $\lambda - \lambda^2/2$. Example: $\lambda = 0.2$ gives $x = 0.8$ and $f^\star = 0.18$.

### 4. With mismatched degrees, the origin traps

Replace the regularizer by $\lambda(|u| + |v|)$, of degree $1$. Moving from the origin a distance $s$ along $u = v$ changes the loss by about $-s^2$ but adds $2\lambda s$ to the penalty, and for small $s$ the linear term wins. Example: $\lambda = 0.15$, $s = 0.1$: the loss falls by $0.00995$, the penalty rises by $0.03$, so the objective goes up. The origin, with value $0.5$, is a local minimum, yet $u = v = 1$ gives $0.3$. This is the notes' §4 counterexample in two numbers.

### 5. A zero unit lets you test every possible new unit

At a local minimum where one subnetwork is entirely zero, that subnetwork can be switched on in *any* direction by an infinitesimal amount. Local minimality says none of these tiny new units lowers the objective. With matched degrees, that is exactly the optimality condition of the equivalent convex problem: no new atom can help, so the point is globally optimal. This is the intuition behind Theorem 1, not its proof; the proof works with the paper's convex function $\Omega_{\phi,\theta}$.

:::key Putting the five ideas together
Bias-free ReLU subnetworks are homogeneous (idea 1), and a regularizer of the same degree becomes a convex penalty on the output once layers are balanced (idea 2). Then the zero point is a saddle (idea 3), and a local minimum with a spare zero subnetwork passes the convex optimality test (idea 5). A degree mismatch breaks all of this and creates a spurious minimum at zero (idea 4). The notes below state the model and the supplement's counterexample precisely.
:::
:::

:::divider Study notes
:::

## 1. Problem: when does a non-convex neural objective have a global certificate?

The training of a neural net is non-convex in its layer parameters, but some overparameterized structures have a convex interpretation after regrouping their components. Haeffele and Vidal ask for *sufficient conditions* ensuring that certain local minima are global, and for a non-increasing path from an arbitrary starting parameterization toward a global solution. This does not claim that all stationary points of arbitrary deep nets are global minima.

## 2. Factorized parallel networks and positive homogeneity

A representative structured model is a sum of $r$ parallel subnetworks:

$$\Phi_r(W) = \sum_{i=1}^{r} \phi(w_i), \qquad \Theta_r(W) = \sum_{i=1}^{r} \vartheta(w_i).$$

Training takes the schematic form

$$\boxed{\min_W\ \ell\!\left(Y, \sum_{i=1}^{r}\phi(w_i)\right) + \lambda\sum_{i=1}^{r}\vartheta(w_i).}$$

The mapping $\phi$ and its size-controlling regularizer $\vartheta$ are assumed *positively homogeneous with matching degree*:

$$\phi(cw) = c^p\phi(w), \qquad \vartheta(cw) = c^p\vartheta(w), \qquad c \ge 0.$$

A ReLU neuron satisfies $\mathrm{ReLU}(cz) = c\,\mathrm{ReLU}(z)$ for $c \ge 0$, but a general network with biases and arbitrary normalization does not automatically satisfy the *joint parameter* homogeneity required here. The degree and the regularizer must match.

## 3. Why adding a zero component can help

A zero-valued subnetwork can create a direction in the *factorization space* through which infinitesimal parameter changes realize a new functional component. The paper proves global-optimality conditions for certain local minima in sufficiently large homogeneous parallel models when such a component is present; it also discusses a non-increasing path to a global solution. *Geometric intuition, not a full proof:* imagine you have a representation $X = \sum_i \phi(w_i)$ and can split or reallocate the representation across more parallel components without raising the regularized objective. Extra components make a non-convex parameterization flexible enough to connect to an equivalent convex-gauge description. The original proof uses specific atomic/gauge constructions and duality, not merely the assertion that "more neurons improve optimization."

## 4. Why the matching degree matters: derivation from the supplement

The official supplemental material gives a particularly instructive counterexample when the output map has degree $p$ but the regularizer has smaller degree $p' < p$. For a fixed nonzero direction $W$ and small $s > 0$,

$$\phi(sW) = s^p\phi(W), \qquad \vartheta(sW) = s^{p'}\vartheta(W).$$

For convex $\ell$ with a subgradient at the origin, a first-order lower bound gives the *form*

$$\ell(s^p\phi(W)) + s^{p'}\vartheta(W) - \ell(0) \;\ge\; s^p\langle q, \phi(W)\rangle + s^{p'}\vartheta(W).$$

As $s \downarrow 0$, the positive regularization term of order $s^{p'}$ dominates the potentially negative loss change of order $s^p$. Thus the origin can become a *local minimum* even when it is not globally best. This demonstrates why the theory cannot be transferred to a deep ReLU model with an arbitrary degree-mismatched weight penalty.

:::key Main result
Under structured parallel architecture, convex loss in the output, appropriate homogeneity/regularization, and sufficient model size, special local minima (notably those containing an inactive subnetwork) admit a global-optimality certificate. The paper also analyzes paths toward global solutions.
:::

## 5. What was and was not verified

The full CVPR paper and its *official eight-page proof supplement* are available. The supplement is directly relevant to the homogeneity restrictions; its Proposition 2 supplies the mismatch counterexample summarized above. This note explains the central theorem mechanism and its limits, but it is not a theorem-by-theorem proof check of every proposition. Do not replace the specified model class with "any sufficiently wide nonlinear neural net."

## 6. Survey relevance and limitations

Use in **RQ1** (sufficient conditions for benign minima) and **RQ4** (nontrivial assumptions needed to transfer such results to realistic models). General feedforward networks with unstructured depth, biases, arbitrary regularizers, or insufficient width are not covered by this claim. Nor does the result imply every generic SGD trajectory efficiently finds an optimal solution in wall-clock time.

:::warn Scope
The result does *not* say all local minima are global for general ReLU/Transformer training. The global statement is conditional and depends on the network's parallel factorization and homogeneous regularizer.
:::

## 7. Our notes

The useful idea here is that extra homogeneous subnetworks can make a non-convex parameterization behave like a convex problem, but only when the architecture and regularization fit together precisely. The proof supplement even shows why using the wrong degree of regularization can create a spurious minimum at zero. That's a strong example for our theory-versus-practice discussion. **Paper:** [https://openaccess.thecvf.com/content_cvpr_2017/html/Haeffele_Global_Optimality_in_CVPR_2017_paper.html](https://openaccess.thecvf.com/content_cvpr_2017/html/Haeffele_Global_Optimality_in_CVPR_2017_paper.html). **Supplement:** [https://openaccess.thecvf.com/content_cvpr_2017/supplemental/Haeffele_Global_Optimality_in_2017_CVPR_supplemental.pdf](https://openaccess.thecvf.com/content_cvpr_2017/supplemental/Haeffele_Global_Optimality_in_2017_CVPR_supplemental.pdf).

> **Primary source:** B. D. Haeffele & R. Vidal (2017), *Global Optimality in Neural Network Training*, IEEE/CVF CVPR, pp. 7331–7339. [openaccess.thecvf.com](https://openaccess.thecvf.com/content_cvpr_2017/html/Haeffele_Global_Optimality_in_CVPR_2017_paper.html).
