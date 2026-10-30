## camel | Global convergence time (Sec. 2 warm-up, Theorem 1)
$$x_{n+1}=x_n-\eta\big(\nabla f(x_n)+Z_n\big),\qquad \tau=\inf\{n:\operatorname{dist}(x_n,Q)\le\delta\}$$
with $Q=\arg\min f$. Theorem 1 (Assumptions 1–3; small $\eta,\delta$; any tolerance $a>0$):
$$\exp\!\Big(\frac{E(Q\mid x)-a}{\eta}\Big)\le\mathbb E_x[\tau]\le\exp\!\Big(\frac{E(Q\mid x)+a}{\eta}\Big)$$
the lower bound requiring $Q$ to be attracting enough. $E\equiv0$ iff there are no spurious minima (then $\tau$ is subexponential).
### Test function (Fig. 1)
$$f(x_1,x_2)=\tfrac{2x_1^6}{13}+\tfrac{x_1^5}8-\tfrac{91x_1^4}{64}-\tfrac{x_1^3}2+\tfrac{42x_1^2}{16}+\tfrac{5x_2^2}4+x_1x_2$$
Critical points ($x_2=-2x_1/5$): minima $p_1=(-2.533,1.013)$ $f=-7.248$ (global), $p_3=(0,0)$ $f=0$, $p_5=(2.078,-0.831)$ $f=-3.297$; saddles $p_2=(-1.116,0.447)$ $f=1.590$, $p_4=(0.894,-0.358)$ $f=0.823$.
> The paper prints $p_1\approx(-2.573,1.029)$; the printed formula's minimizer is at $x_1=-2.533$. We use the formula.

## graph | Action, transition costs, forests and prunings
Hamiltonian $\mathcal H(x,p)=\log\mathbb E\,e^{-\langle p,G(x;\omega)\rangle}$, Lagrangian $\mathcal L$ = its convex conjugate, action $\mathcal S[\gamma]=\int\mathcal L(\gamma,\dot\gamma)$. $B(p,q)$ = least action between critical components. For Gaussian noise $\mathcal N(0,\sigma^2I)$:
$$\mathcal L(x,v)=\frac{\lVert v+\nabla f(x)\rVert^2}{2\sigma^2},\qquad \text{climbing }\Delta f\text{ costs }\frac{2\Delta f}{\sigma^2},\ \text{descending is free}$$
$$E(Q\mid p)=\min_{\text{forests}}\sum B-\min_{\text{prunings of }p}B,\qquad E(Q\mid x)=\max_p\,[E(Q\mid p)-B(x,p)]$$
### Camel (units $1/\sigma^2$)
$B(p_3\!\to\!p_1)=3.181$, $B(p_3\!\to\!p_5)=1.645$, $B(p_5\!\to\!p_3)=8.239$: forest $11.420$, pruning $1.645$, $E(Q\mid p_3)=9.775$.
### Notes' three-state chain
Costs $AG=2$, $AB=3$, $BA=5$, $BG=8$: forests 7, 10, 11; prunings 3, 5, 8; $E(G\mid A)=4$; exact $T=\frac{w+z+v}{uw+uz+vz}$ with $u,v,w,z=e^{-2/\eta},e^{-3/\eta},e^{-5/\eta},e^{-8/\eta}$.

## scaling | Checking the exponent
> Check script ($\sigma=11$, 400 runs per $\eta\in[0.0098,0.02]$): $\mathbb E[\tau]$ from 1 264 to 162 841 steps; slope of $\log(\eta\,\mathbb E\tau)$ against $1/\eta$ = 0.0793 vs predicted $E=9.775/\sigma^2=0.0808$.
### Source audit (from our notes, verified)
Lemma D.7's $M_n=\sum_{k\le n}(1-\gamma\mu)^{n-k}\langle u_k,p_k-y_k\rangle$ (D.44) satisfies $\mathbb E[M_{n+1}\mid\mathcal F_n]=(1-\gamma\mu)M_n$, so it is not a martingale as asserted before Doob's inequality (D.46). We report the theorem as stated and flag this step; we do not certify a repair.
