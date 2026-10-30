## framework | Algorithm 1
Repeat: $(y_t,z_t)=\text{GFO}(x_{t-1})$; $u_t=y_t$ w.p. $p$, else $z_t$; $(x_t,\tau)=\text{HFO}(u_t)$; stop when HFO certifies ($\tau=\emptyset$).
### Requirements
- G.1 $\mathbb Ef(y)\le f(x)$; G.2 $\mathbb E\lVert\nabla f(z)\rVert^2\le\frac1{g(n,\epsilon)}\mathbb E[f(x)-f(y)]$
- H.1 $\mathbb Ef(y)\le f(x)$; H.2 $\mathbb Ef(y)\le f(x)-h(n,\epsilon,\gamma)$ whenever $\lambda_{\min}(\nabla^2f(x))\le-\gamma$
- Theorem 1: the rate is governed by the slower of the two; the best $p$ satisfies $\frac1p=\frac1{\epsilon^2g}+\frac1h$
### Instantiation
GFO = non-convex SVRG (Alg. 2, $\eta=\frac1{4Ln^{2/3}}$, epoch $m=n$); HFO = HessianDescent (Alg. 3).
### Toy on screen
$$f_i(x,y)=\tfrac12x^2-\tfrac12y^2+\tfrac14y^4+a_ix+c_i\tfrac{x^2-y^2}2$$
Every $\nabla f_i$ has zero $y$-part on $y=0$, so SVRG alone never leaves the saddle's stable set.
> Check script: SVRG alone converges to the saddle; the framework reaches $f=-\tfrac14$ after 4 HessianDescent moves (162 rounds).

## lemma2 | HessianDescent and Lemma 2
Unit $v$ with $v^\top\nabla^2f(x)v\le\lambda_{\min}+\frac\gamma2$:
$$\alpha=\frac{|v^\top\nabla^2f(x)v|}{M},\qquad u=x-\alpha\,\mathrm{sign}(v^\top\nabla f(x))\,v,\qquad y=\arg\min\{f(u),f(x)\}$$
Lemma 2: if $\lambda_{\min}(\nabla^2f(x))\le-\gamma$ then $f(y)\le f(x)-h$ with
$$h(n,\epsilon,\gamma)=\frac{\gamma^3}{24M^2}$$
### Toy
$f=\tfrac12x^2-\tfrac\gamma2y^2+\tfrac M6|y|^3$ at 0: exact eigenvector gives $f(0)-f(0,\gamma/M)=\frac{\gamma^3}{3M^2}$.
> Check script: worst admissible decrease 0.00586 vs $h=0.00521$ ($\gamma=0.5$, $M=1$); same ordering for $(0.2,2)$ and $(1,3)$.
