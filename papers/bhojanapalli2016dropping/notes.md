# Dropping Convexity for Faster Semi-definite Optimization

:::minute The paper in one minute
**Question:** Many problems ask us to minimize a convex function $f(X)$ under the constraint $X \succeq 0$. Writing $X = UU^\top$ with $U \in \mathbb{R}^{n\times r}$ reduces storage when $r \ll n$, but the map from $U$ to $f(UU^\top)$ becomes non-convex. Can gradient descent on the factors still converge rapidly?

**Paper result:** Factorization can trade the original convex geometry for a lower-dimensional non-convex problem without losing local convergence rates under suitable conditions. The analysis provides an algorithmic route, not a theorem that every critical point of every factorized objective is globally optimal.

**Do not overstate:** "FGD converges locally at convex-like rates" is different from "all local minima are global"; the latter is a separate landscape question and is not the universal theorem of this work.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. The chain rule through a product

The notes' gradient $\nabla g(U) = 2\nabla f(UU^\top)U$ is the matrix form of a one-line scalar fact. With $n = r = 1$, $g(u) = f(u^2)$ and

$$g'(u) = 2u\,f'(u^2) .$$

Example: $f(x) = \tfrac12(x - 4)^2$ is convex with its minimum at $x = 4$. Then $g(u) = \tfrac12(u^2 - 4)^2$ has $g'(1) = 2 \cdot 1 \cdot (1 - 4) = -6$, minima at $u = \pm 2$, and a stationary point at $u = 0$ where $f'(0) = -4 \neq 0$: the factor $u$ in front creates a critical point that $f$ itself does not have. Matrix inner products $\langle A, B\rangle = \mathrm{tr}(A^\top B)$ turn the same computation into the notes' $dg = \langle \nabla f(X), dU\,U^\top + U\,dU^\top\rangle$.

### 2. What "convex rates" means

For gradient descent on a convex, $L$-smooth function, the error after $k$ steps is at most of order $L\|x_0 - x^*\|^2/k$: an $O(1/k)$ rate. If the function is also $\mu$-strongly convex, each step shrinks the error by a fixed factor such as $1 - \mu/L$: a **linear** (exponential) rate. Example with $\mu/L = 0.1$: after $50$ steps $0.9^{50} \approx 0.0052$, while $1/k$ gives $0.02$, and after $200$ steps $0.9^{200} \approx 7 \times 10^{-10}$ against $0.005$. The paper's claim is that FGD keeps these two rates, now measured near the factored solution.

### 3. Local means "starting close enough"

A local rate holds only inside some radius around the solution. Because $U$ and $UR$ give the same $X$ for every orthogonal $R$, closeness is measured up to rotation, $\mathrm{DIST}(U, U^\star) = \min_R \|U - U^\star R\|_F$. In the scalar example, $g(u) = \tfrac12(u^2 - 4)^2$ has $g''(u) = 6u^2 - 8$: near $u = 2$ the curvature is $16 > 0$ and gradient descent converges linearly, but at $u = 0$ it is $-8$ (a local maximum). That is why the paper pairs its local theorems with an **initialization** that lands inside the radius; only then do the guarantees become global.

### 4. The factored problem's curvature grows with the solution's size

In the scalar example with target $a^2$, $g(u) = \tfrac12(u^2 - a^2)^2$ has curvature $g''(a) = 4a^2$ at the solution. A fixed step must satisfy $\eta < 2/(4a^2) = 1/(2a^2)$ to converge there: $\eta < 0.125$ for $a = 2$, so $\eta = 0.2$ diverges even though the same step is fine for $a = 1$. The curvature of $g$ depends on $\|U\|^2$, not only on the smoothness of $f$. This is why the paper's step size rule uses the size of the starting matrix, $\eta = 1/\big(16(M\|X_0\|_2 + \|\nabla f(X_0)\|_2)\big)$, instead of a plain $1/M$.

:::key Putting the four ideas together
Gradients pass through the product by the chain rule (idea 1). The paper shows that, with a step size scaled to the starting point (idea 4), gradient descent on the factors keeps the convex rates $O(1/k)$ and linear (idea 2) from any start close enough to a solution, measured up to rotation, and supplies an initialization that gets there for (restricted) strongly convex $f$ (idea 3). The notes below state these results and their limits.
:::
:::

:::divider Study notes
:::

## 1. The question: why optimize a non-convex factorization?

Many problems ask us to minimize a *convex* function $f(X)$ under the constraint $X \succeq 0$. A positive-semidefinite $n \times n$ matrix can be expressed as $X = UU^\top$ for $U \in \mathbb{R}^{n\times r}$. This reduces storage when $r \ll n$, but the map from $U$ to $f(UU^\top)$ becomes non-convex. Can gradient descent on the factors still converge rapidly?

## 2. The method: factored gradient descent

Write the new objective $g(U) = f(UU^\top)$. Assuming differentiability and a symmetric gradient of $f$ with respect to $X$, differential calculus gives

$$dX = dU\,U^\top + U\,dU^\top, \quad dg = \langle \nabla f(X), dX\rangle, \quad \boxed{\nabla g(U) = 2\nabla f(UU^\top)U.}$$

Therefore the factor-based gradient step becomes

$$\boxed{U_{k+1} = U_k - 2\eta_k\nabla f(U_kU_k^\top)U_k.}$$

For the teaching example $f(X) = \frac{1}{2}\|X - X_*\|_F^2$, the factor gradient is

$$\nabla g(U) = 2(UU^\top - X_*)U.$$

Although $f$ is convex, $g$ is generally not. For instance, $U$ and $UQ$ define the same PSD matrix for every orthogonal $Q$, so the factor problem has symmetries and non-isolated solutions.

## 3. Precisely what the authors prove

The authors design a stepsize rule for *factored gradient descent* (FGD) based on assumptions on the original convex objective $f$. They prove *local* convergence rates mirroring the ordinary convex rates:

$$\text{smooth } f:\ O(1/k), \qquad \text{(restricted) strongly convex } f:\ \text{exponential/linear convergence in } k.$$

For the restricted strongly convex setting, they also give an initialization procedure using first-order oracle access to $f$; for several problem instances, pairing this with FGD provides global convergence assurances. The distinctions between *local convergence* and *global convergence after suitable initialization* are essential.

:::key Paper result
Factorization can trade the original convex geometry for a lower-dimensional non-convex problem without losing local convergence rates under suitable conditions. The analysis provides an algorithmic route, not a theorem that every critical point of every factorized objective is globally optimal.
:::

## 4. How this links to neural networks

A two-layer linear network computes $W_2W_1x$; PSD factorization uses the multiplicative parameterization $UU^\top$. Both create non-convexity by writing a matrix as a product. The comparison helps explain why non-convex *parameterizations* need not make optimization hopeless. However, the constraints and model classes differ: PSD optimization is not generic deep learning.

## 5. Assumptions, limits, survey relevance

The theorems assume smoothness, appropriate convexity or restricted strong convexity of $f$, stepsize control, and—in the global guarantees—appropriate initialization. Rank selection matters. The paper does not give a universal strict-saddle classification for neural-network losses, nor an efficient SOSP result for arbitrary $g$. Use primarily for **RQ1** (benign factorized geometry) and **RQ3** (computationally economical optimization), with the structured-versus-realistic-model caution of **RQ4**.

:::warn Do not overstate
"FGD converges locally at convex-like rates" is different from "all local minima are global"; the latter is a separate landscape question and is not the universal theorem of this work.
:::

## 6. Our notes

This is a useful counterexample to the idea that non-convexity automatically makes gradient descent slow. By writing a PSD matrix as $UU^\top$, we give up convexity but save memory and can keep strong convergence rates with the right initialization and assumptions. It is a structured matrix problem, so we should use it as a bridge to neural-network factorization rather than a claim about every neural network. **Original source:** [https://proceedings.mlr.press/v49/bhojanapalli16.html](https://proceedings.mlr.press/v49/bhojanapalli16.html). **Verification:** official COLT abstract and methodology checked; displayed gradients are transparent teaching derivations.

> **Primary source:** Bhojanapalli, Kyrillidis & Sanghavi (2016), *Dropping Convexity for Faster Semi-definite Optimization*, COLT 2016 (PMLR 49). [proceedings.mlr.press/v49/bhojanapalli16.html](https://proceedings.mlr.press/v49/bhojanapalli16.html).
