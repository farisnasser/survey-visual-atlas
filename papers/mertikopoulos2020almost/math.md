## ridge | Almost-sure convergence and saddle avoidance
$$X_{n+1}=X_n-\gamma_n\big(\nabla f(X_n)+Z_n\big),\qquad\gamma_n\propto1/n^p$$
### Assumptions
- 1: $f$ is $G$-Lipschitz and $L$-smooth; 2: bounded sublevel sets; 3: bounded gradient sublevel sets $\{\lVert\nabla f\rVert\le\varepsilon\}$
- 4: zero-mean noise with finite $q$-th moments, $\mathbb E\lVert Z\rVert^q\le\sigma^q$, $q\ge2$
- 5: uniformly exciting noise, $\mathbb E[\langle Z,u\rangle_+]\ge c>0$ for every unit $u$
### Results (NeurIPS 2020)
- Theorem 1: the induced asymptotic pseudotrajectories are bounded a.s.; Theorem 2: for $p\in(\frac2{q+2},1]$, $X_n$ converges a.s. to a connected component of crit $f$ on which $f$ is constant (no boundedness assumption; steps need not be square-summable)
- Theorem 3: $p\in(0,1]$, Assumptions 1–5 with $q=\infty$: $P(X_n\to S)=0$ for every **strict saddle manifold** $S$ (a smooth component of crit $f$ whose negative eigenvalues stay below $-c_-$ and positive ones above $c_+$)
### Toy on screen
$$f=\varphi(r),\quad\varphi(r)=\tfrac{r^4}4-r^3+r^2,\quad\varphi'(r)=r(r-1)(r-2)$$
Ridge $r=1$: Hessian eigenvalues $\varphi''(1)=-1$ (radial) and $\varphi'(1)/1=0$ (tangential), a strict saddle manifold. Minima: $r=0$ and the ring $r=2$.
> Check script: GD started exactly on the ridge stays there. SGD from 200 points on the ridge, 20 000 steps: $p=0.5$: 44% middle, 56% outer ring; $p=0.75$: 38% / 62%; $p=1$: 37.5% / 62.5%; none on the ridge.
The toy is $L$-smooth but its gradient is unbounded at infinity (Assumption 1 asks for $G$-Lipschitz $f$); the runs stay inside $r<2.6$.

## rate | Local rate near a regular minimizer
- Theorem 4: $x^\ast$ a regular minimizer, $\gamma_n=\gamma/(n+m)^p$, $p\in(\frac2{q+2},1]$, $\gamma,m$ large enough: with probability $\ge1-\delta$ the iterates stay in a neighbourhood $U$, and conditioned on that $\mathbb E[\lVert X_n-x^\ast\rVert^2\mid\Omega_U]=O(1/n^p)$
> Check script (noise 0.1, $\gamma=0.5$, $m=10$, 2000 runs from $x^\ast=0$, all stayed in $\lVert x\rVert<0.5$): fitted slopes over $n\in[10^3,10^5]$: $-0.509$ ($p=0.5$), $-0.754$ ($p=0.75$), $-1.004$ ($p=1$).
