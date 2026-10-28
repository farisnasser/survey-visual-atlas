## stable | Theorem 4
### Strict saddle (Def. 2)
A critical point $x^*$ with $\lambda_{\min}(\nabla^2 f(x^*))<0$ (zero eigenvalues allowed).
### Theorem 4
If $f\in C^2$, $\nabla f$ is $L$-Lipschitz, $x^*$ is a strict saddle, $0<\alpha<1/L$ and $x_0$ has an absolutely continuous distribution, then
$$\Pr\big(\lim_k x_k=x^*\big)=0$$
### Paper's example (Sec. 3)
$$f(x,y)=\tfrac12x^2+\tfrac14y^4-\tfrac12y^2,\qquad \nabla^2f=\begin{pmatrix}1&0\\0&3y^2-1\end{pmatrix}$$
Saddle $(0,0)$ with $\lambda=1,-1$; minima $(0,\pm1)$. The stable set of the saddle is the $x$-axis, which has measure zero.
> $L$ is only bounded on a region here ($L\le 6.68$ for $|y|\le1.6$); the balls use $\alpha=0.1<1/L$.

## escape | What the theorem does not give
The result is asymptotic. Near the saddle $y_{k+1}\approx(1+\alpha)y_k$, so
$$t_{\text{escape}}\approx\frac{\ln(c/|y_0|)}{\ln(1+\alpha)}\ \xrightarrow{\ y_0\to0\ }\ \infty$$
No finite-time bound holds uniformly over starts. Jin et al. (2017) add perturbations to get one.

## map | Why measure zero survives (Prop. 8)
GD is the map $g(x)=x-\alpha\nabla f(x)$ with Jacobian $Dg=I-\alpha\nabla^2 f$. For $\alpha<1/L$, $Dg$ is invertible everywhere and $g$ is a diffeomorphism, so preimages of null sets are null:
$$W^s(x^*)\subseteq\bigcup_{k\ge0}g^{-k}\big(W^{s}_{\text{loc}}\big)$$
For this $f$: $\det Dg=(1-\alpha)\big((1+\alpha)-3\alpha y^2\big)$, which vanishes on $|y|=\sqrt{(1+\alpha)/(3\alpha)}$.
### Corollary 9, Theorem 11
Countably many saddles, all strict ⇒ $\Pr(\text{converge to a saddle})=0$; if $\lim x_k$ exists it is a local minimizer a.s.
