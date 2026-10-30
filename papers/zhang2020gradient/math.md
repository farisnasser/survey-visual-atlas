## steep | (L0, L1)-smoothness (Definition 1)
$$\lVert\nabla^2f(x)\rVert\le L_0+L_1\lVert\nabla f(x)\rVert$$
Motivated by measurements along AWD-LSTM training on PTB, where local smoothness correlates with the gradient norm (Fig. 1). Strictly weaker than $L$-smoothness: **Lemma 2**, polynomials of degree $\ge3$ are $(L_0,L_1)$-smooth but not $L$-smooth.
**Clipped GD (Alg. 5):** $x_{k+1}=x_k-h_c\nabla f(x_k)$, $h_c=\min\{\eta_c,\ \gamma\eta_c/\lVert\nabla f(x_k)\rVert\}$.
### On screen
$f=\tfrac12x^2+\tfrac14x^4+\tfrac12y^2$; $\nabla^2f=\mathrm{diag}(1+3x^2,1)$.
> Check script: $\lVert\nabla^2f\rVert\le1+3\lVert\nabla f\rVert$ on $[-6,6]^2$ (max ratio 0.84). From $(4,1)$: best fixed-step GD 81 iterations ($h=0.111$), clipped GD 10.

## scaling | Theorems 3, 4, 6
With $\Delta=f(x_0)-f^\star$ and $M=\sup\{\lVert\nabla f(x)\rVert: f(x)\le f(x_0)\}$:
- **Theorem 3 (clipped GD):** $\displaystyle O\Big(\frac{L_0\Delta}{\epsilon^2}+\frac{L_1^2\Delta}{L_0}\Big)$ iterations, independent of $M$.
- **Theorem 4 (fixed-step GD, lower bound):** $\displaystyle\Omega\Big(\frac{L_1M\,(\Delta-\dots)}{\epsilon^2\log M}\Big)$.
- **Theorem 6 (fixed-step GD, upper bound):** $O\big((ML_1+L_0)\Delta/\epsilon^2\big)$ with $h\propto1/(ML_1+L_0)$.
So clipping can be arbitrarily faster when $M$ is large (poor initialization). Theorems 7–8 give the stochastic versions (clipped SGD vs SGD).
> Check script ($x_0=1,2,3,4,6,8$; $M=2.2\dots520$): best fixed-step GD 6, 22, 47, 81, 189, 322 iterations; clipped GD 4, 6, 8, 10, 14, 18.
