# Stable-Manifold Foundations: Global Stability of Dynamical Systems — targeted Chapter 5 reading

:::minute The paper in one minute
**Question:** What determines which starting points converge to a fixed point when some directions contract and others expand?

**Model/Method:** Analyze a differentiable dynamical map near hyperbolic fixed points using invariant stable and unstable manifolds; extend to center-stable manifolds when neutral directions appear.

**Finding:** Stable sets near hyperbolic fixed points are smooth manifolds tangent to the contracting eigenspaces (Theorem 5.20). They have lower dimension when any expanding direction exists.

**Why we care:** These tools underlie geometric proofs that randomly initialized GD almost surely avoids strict saddles. Shub's book supplies mathematics, not an optimizer experiment or a neural-network result.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the book stands on, built from scratch.
### 1. Powers of a matrix follow its eigenvectors

Iterating a linear map $A$ means computing $A^k u$. If $v$ is an eigenvector, $A v = \lambda v$, then $A^k v = \lambda^k v$: the component along $v$ is multiplied by $\lambda$ at every step. The eigenvectors need not be the coordinate axes. For

$$A = \begin{bmatrix} 0.8 & 0.3 \\ 0 & 1.2 \end{bmatrix},$$

the eigenvalue $0.8$ has eigenvector $(1, 0)$ and the eigenvalue $1.2$ has eigenvector $(0.6, 0.8)$. A starting point converges to $0$ exactly when it has no component along $(0.6, 0.8)$, that is, when it lies on the line spanned by $(1, 0)$. The set of points that converge is the span of the **contracting eigenvectors**, whatever directions they point in.

### 2. A manifold is a smooth curved version of a subspace

A **manifold** of dimension $k$ is a set that, near each of its points, looks like a bent piece of $\mathbb{R}^k$: a smooth curve is a 1-dimensional manifold in the plane, a smooth surface a 2-dimensional one in space. For a nonlinear map the stable set usually bends. Take

$$g(x, y) = \big(0.8x,\; 1.2y - 0.28x^2\big).$$

Its derivative at the origin is $\mathrm{diag}(0.8, 1.2)$, as in idea 1. But the points that converge to $0$ form the parabola $y = 0.5x^2$, not the $x$-axis: starting at $(1, 0.5)$ the iterates are $(0.8, 0.32)$, $(0.64, 0.2048)$, $(0.512, 0.131)$, each still on $y = 0.5x^2$, while any point off the parabola drifts away from it by a factor $1.2$ per step. The parabola is **tangent** to the $x$-axis at the origin: near the fixed point the curved stable set looks like the linear one. This is the picture the stable-manifold theorem makes general.

### 3. Thin sets stay thin under smooth invertible maps

A set has **Lebesgue measure zero** when it has no area (in the plane) or no volume (in higher dimension): a point, a line, a smooth curve. Three facts let a local statement become a global one. A countable union of measure-zero sets has measure zero (countably many lines still have no area). A smooth map sends measure-zero sets to measure-zero sets. And a smooth map with a smooth inverse (a **diffeomorphism**) therefore also has measure-zero *preimages*. Invertibility matters: the map $(x, y) \mapsto (x, 0)$ squashes the whole plane onto a line, so the preimage of that thin line is everything.

### 4. When an eigenvalue is exactly one

A fixed point is **hyperbolic** when no eigenvalue of its derivative has modulus exactly $1$. A neutral eigenvalue changes the speed, not the picture. Take $g(x, y) = (x - x^3,\, 1.2y)$: along $x$ the multiplier at $0$ is exactly $1$, yet $x$ still creeps toward $0$ because of the cubic term: from $x_0 = 0.5$, after $100$ steps $x \approx 0.069$, close to the rate $1/\sqrt{2k}$. Along $y$ the multiplier $1.2$ still repels. The points that converge all lie on the line $y = 0$ (and every point of it with $|x| < 1$ does). Such sets, which include the neutral directions, are **center-stable manifolds**; they are still lower-dimensional whenever some direction genuinely expands.

:::key Putting the four ideas together
Near a fixed point, the starting points that converge lie along the contracting directions (idea 1); for a nonlinear map they form a curved manifold tangent to those directions (idea 2), or a center-stable manifold when some eigenvalues equal one (idea 4). If at least one direction expands, that manifold is thin, and smooth invertible maps keep thin sets thin (idea 3). This is the machinery behind "GD almost surely avoids strict saddles", made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. A book, not a neural-network paper

Shub's *Global Stability of Dynamical Systems* covers general dynamical-systems theory. Our survey needs a focused portion: Chapter 4 (hyperbolicity), Chapter 5 (stable manifolds), and Appendix III (center-stable manifolds). The whole monograph need not be treated as an ML benchmark.

## 2. Fixed points and the derivative map

A point $p$ is fixed for $g$ if $g(p) = p$. Near it, with small displacement $u$, first-order linearization gives

$$g(p + u) - p \approx Dg(p)\, u.$$

Eigenvalues of $Dg(p)$ tell us how different directions evolve. A magnitude below 1 contracts, above 1 expands, and exactly 1 is neutral at linear order. A *hyperbolic* fixed point has no eigenvalues on the unit circle.

## 3. Worked example: the stable set is a line

Consider the simple linear map

$$g(x, y) = (0.8x,\, 1.2y).$$

After $k$ iterations,

$$(x_k, y_k) = (0.8^k x_0,\, 1.2^k y_0).$$

The $x$ component approaches zero, but $y$ grows unless $y_0 = 0$. Thus the set of starting points converging to the origin is

$$W^s(0) = \{(x, 0) : x \in \mathbb{R}\}.$$

It is a one-dimensional line inside a two-dimensional parameter space; it has zero 2D Lebesgue measure. A random initialization with a density lands on it with probability zero.

:::key Why this matters for gradient descent
For $f(x, y) = \tfrac12(x^2 - y^2)$ and step size $\eta = 0.2$, GD is exactly $g(x, y) = (0.8x, 1.2y)$. The saddle at the origin attracts only an exceptional line of initializations. This is a pedagogical reduction, not a theorem about all neural networks.
:::

## 4. What the stable-manifold theorem adds

The linear example was easy because the eigenvectors never change. For a *nonlinear* differentiable map, the stable set generally curves. The stable-manifold theorem identifies a smooth manifold passing through the fixed point, tangent to its contracting eigenspace. For a hyperbolic fixed point, Shub's Theorem 5.20 describes local stable and unstable manifolds and their global immersed extensions.

### Proof intuition: straighten the dynamics locally

Write $g(p + u) = p + Au + r(u)$ with $A = Dg(p)$ and $r(u)$ smaller than the linear term near $p$. One constructs an invariant graph over the stable eigenspace. The *graph transform* sends candidate graphs to new ones under the dynamics; a contraction argument identifies the invariant graph. The result persists from the linear picture because the nonlinear correction is locally controlled.

## 5. What if an eigenvalue is exactly one?

Hyperbolicity excludes neutral directions. A neural-network Hessian may have a zero eigenvalue, which produces an eigenvalue 1 in the GD Jacobian. Appendix III discusses *center* and *center-stable* manifolds that include neutral directions. A center-stable manifold can still have lower dimension than the ambient space if a genuinely expanding direction exists.

:::warn Do not overstate measure-zero arguments
The measure-zero conclusion needs an expanding direction, suitable smooth invertible local dynamics, and an absolutely continuous initialization. When moving from a *local* manifold to all starting points, additional regularity of preimages is needed.
:::

## 6. Application to strict-saddle avoidance in GD

If $g(w) = w - \eta\nabla f(w)$, then

$$Dg(w_*) = I - \eta\nabla^2 f(w_*).$$

For a strict saddle, some Hessian eigenvalue $\lambda_H < 0$ yields a GD eigenvalue $1 - \eta\lambda_H > 1$: the saddle has a locally expanding direction. Lee et al. (2016) make this argument rigorous under their differentiability, Lipschitz-gradient, step-size and random-initialization assumptions.

## 7. What this source does and does not prove

**Mathematical contribution.** The book develops rigorous invariant-manifold and stability theory for maps; the cited chapters make the geometry of contracting and expanding directions precise. It does not propose any gradient-descent algorithm. **Limitations for our survey.**

- A theorem about a hyperbolic fixed point cannot automatically be applied to a degenerate saddle with neutral eigenvalues; the center-stable extension matters.
- The dynamical-systems result is local. Global claims for GD require control of the gradient map and preimages, as supplied in later optimization analyses.
- Almost-sure non-convergence to a strict saddle does not give a finite-time escape bound, nor does it guarantee reaching a global minimum.

## 8. Where to read

**Prioritize:** Chapter 4 (pp. 20–32), Chapter 5 (pp. 33–70), particularly Theorem 5.20, and Appendix III (pp. 61–70) for center-stable manifolds. **Optional:** Chapter 6 and Appendix IV, which extend stable-manifold ideas to invariant sets. Page references are to the book's printed pagination.

## 9. Quick reference

| | |
|---|---|
| **Fixed point** | $g(p) = p$. |
| **Linearization** | $g(p + u) \approx p + Dg(p) u$. |
| **Stable direction** | $\lvert\lambda(Dg(p))\rvert < 1$. |
| **Unstable direction** | $\lvert\lambda(Dg(p))\rvert > 1$. |
| **Hyperbolicity** | No Jacobian eigenvalue of modulus 1. |
| **Stable manifold** | Invariant set tangent to stable eigenspace (Thm. 5.20). |
| **GD Jacobian** | $I - \eta\nabla^2 f(w_*)$. |
| **Exception measure** | Lower-dimensional stable set has zero ambient volume. |

### Papers Master note

Shub's stable- and center-stable-manifold theory provides the dynamical-systems foundations for almost-sure GD avoidance of strict saddles: a negative Hessian eigenvalue yields an expanding GD-map direction, making saddle-convergent initializations exceptional. Cite as mathematical background; not a neural-network optimization result.

> **Book:** [https://link.springer.com/book/10.1007/978-1-4757-1947-5](https://link.springer.com/book/10.1007/978-1-4757-1947-5). **Author copy:** [https://shub.ccny.cuny.edu/articles/1986-Global_Stability_of_Dynamical_Systems.pdf](https://shub.ccny.cuny.edu/articles/1986-Global_Stability_of_Dynamical_Systems.pdf).
