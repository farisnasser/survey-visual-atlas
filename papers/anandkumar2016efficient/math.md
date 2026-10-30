## third | Higher-order local minima
### Definition 1 ($p$-th order local minimum)
A critical point $x$ with constants $C,\epsilon>0$ such that $f(y)\ge f(x)-C\lVert y-x\rVert^{p+1}$ whenever $\lVert y-x\rVert\le\epsilon$. Every point with $\nabla f=0$, $\nabla^2f\succeq0$ is a second-order local minimum.
### Definition 4 (third-order necessary condition)
$$\nabla f(x)=0,\qquad\nabla^2f(x)\succeq0,\qquad u^\top\nabla^2f(x)u=0\ \Rightarrow\ \nabla^3f(x)[u,u,u]=0$$
checkable in polynomial time (Claim 1).
### Results (COLT 2016)
- Theorem 9: the algorithm (cubic regularization alternated with a third-order step in the competitive subspace, Def. 6) converges to a point satisfying Definition 4; Theorem 8 / Corollary 1: an approximate one in polynomial time, with $\lVert\nabla f\rVert<c_1$, $\lambda_{\min}(\nabla^2f)>-c_2$, competitive third-order norm $<c_3$ after $\mathrm{poly}\cdot\max\{c_1^{-1.5},c_2^{-3},c_3^{-4.5}\}$ iterations
- Assumption 2: Lipschitz third derivatives
### Toy on screen
$$f_3=x^2+y^3+y^4,\qquad\nabla^2f_3(0)=\mathrm{diag}(2,0),\qquad\partial_y^3f_3(0)=6\ne0$$
The origin is a second-order but not a third-order local minimum; the minimum is $(0,-\tfrac34)$, $f=-\tfrac{27}{256}$.
> Check script: GD (step 0.05) from $(0.8,0.6)$ has $y=0.00164$ after 4000 steps ($1/(3\eta k)=0.00167$): it converges to the saddle. With the third-order step (taken once, at iteration 90) it ends at $(0,-0.75)$.

## fourth | Fourth order is NP-hard
- Theorem 10: finding a fourth-order local minimum is NP-hard, even for well-behaved functions
- Proof idea (Theorem 11, Nesterov 2000; Hillar–Lim 2013): deciding whether a degree-4 homogeneous polynomial is non-negative is NP-hard, and $g(x)$ built from it has $0$ as its unique fourth-order local minimizer iff the polynomial is non-negative (Claims 2–4)
### Toy on screen
$$f_4=x^2-y^4+y^6,\qquad\nabla^2f_4(0)=\mathrm{diag}(2,0),\qquad\partial_y^3f_4(0)=0$$
The origin passes Definition 4 (a third-order local minimum) yet $f_4(0,t)=-t^4+t^6<0$: not a fourth-order minimum. Minima at $(0,\pm\sqrt{2/3})$, $f=-\tfrac4{27}$.
> Check script: GD from $(0.5,y_0)$ reaches $|y|=0.5$ after 1013, 256, 63 steps for $y_0=0.05,0.1,0.2$, against $1/(8\eta y_0^2)=1000,250,62$. The third-order test takes no step near the origin.
