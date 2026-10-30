## escape | Lemma 1 and Theorem 1
**Definition 2.** $f$ is $\mu$-one-point-strongly-convex (OPSC) w.r.t. $x^\star$ over $M$ if $\langle\nabla f(x),x-x^\star\rangle\ge\mu\lVert x-x^\star\rVert^2$ for $x\in M$.
**Lemma 1.** $f$ is $L_{\text{global}}$-smooth with global minimum $x^\star$; around a local minimum $x^\dagger$ it is $\mu^\dagger$-OPSC w.r.t. $x^\dagger$ over a set $M$ of diameter $r$; on a surrounding shell $P(M)$ it is $L$-smooth and $\mu^\star$-OPSC w.r.t. $x^\star$; $x^\dagger$ is far enough from $x^\star$. Then for
$$\frac{2}{\mu^\dagger}<\gamma<\frac{\mu^\star}{L^2}$$
GD that enters $M$ leaves it to a point closer to $x^\star$.
**Theorem 1 (informal).** On a class $C_l$ of such functions, GD with a large step from a random start converges to $x^\star$ almost surely, whereas a small step converges to $x^\dagger$ with positive probability. The gradient norm can rise temporarily on the way.
### On screen
$L(z)=0.15\lVert z-c_g\rVert^2-0.35\,e^{-\lVert z-c_s\rVert^2/(2\cdot0.15^2)}$.
> Check script: spurious minimum $L=0.100$, $\lambda_{\max}=15.46$ ($2/\lambda=0.129$); global $L=0$, $\lambda=0.3$. From $(-0.55,0.62)$: $\eta=0.05$ ends at the spurious minimum, $\eta=0.3$ and $0.8$ at the global one; $\lVert\nabla f\rVert$ rises from 1.41 to 1.75 during the escape.

## odds | Theorem 2 and the experiments
**Theorem 2 (informal).** For a region $X$ far from $x^\star$, the probability that randomly initialised GD never visits $X$ is lower-bounded by a quantity increasing in the learning rate and decreasing in the volume of $X$ (equal to 1 if $X$ is finite).
- Experiments (reported): the benefit of large steps persists in full-batch GD, so SGD noise alone does not explain it; GD on a network can escape one minimum and converge to another while the gradient norm temporarily grows (Sec. 5).
> Check script (600 starts): $\eta\le0.12$: about 15% end at the spurious minimum; $\eta=0.15,0.2$: 0% spurious, 14% oscillate inside its well; $\eta\ge0.3$: 100% global; no divergence below $2/0.3$.
### Notes' example
Sharp valley $a=10$: $\eta=0.1$ gives multiplier $1-\eta a=0$, $\eta=0.25$ gives $-1.5$ (locally repelling).
