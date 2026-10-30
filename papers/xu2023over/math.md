## split | The problem and why the rate becomes algebraic
Teacher $f^\star(x)=[v^\top x]_+$, student $f_w(x)=\sum_{i=1}^n[w_i^\top x]_+$, $x\sim N(0,I_d)$:
$$L(w)=\mathbb E_x\tfrac12\Big(\sum_j[w_j^\top x]_+-[v^\top x]_+\Big)^2,\qquad \mathbb E\big[[u^\top x]_+[v^\top x]_+\big]=\frac{\lVert u\rVert\lVert v\rVert\sin\phi+(\pi-\phi)\,u^\top v}{2\pi}$$
For $n\ge2$ the global minima form a set (e.g. $w_1=cv$, $w_2=(1-c)v$). Two students at angle $\phi$ with fixed sum differ from the teacher only through $\sin\phi-\phi\cos\phi\approx\phi^3/3$, so the loss is cubic across the minimum set and gradient flow gives $\dot b\propto-b^2$, $b\sim1/t$, $L\sim t^{-3}$ — the paper proves this through a potential on pairwise distances.
### Slice shown
$d=2$, $v=e_1$; left $w=(a,b)$ ($n=1$); right $w_1=(a,b)$, $w_2=(1-a,-b)$.
> Check script: on the right slice $L(0.5,b)\propto|b|^{3.000}$ near $b=0$; the whole segment $b=0$, $0\le a\le1$ has $L=0$; GD (step 0.5) on the slice decays with fitted slope $-3.00$.

## rates | Theorems 1 and 2
- Gradient (Eq. 3, Safran & Shamir): $\nabla_i=\tfrac12\big(\sum_jw_j-v\big)+\tfrac1{2\pi}\Big[\big(\sum_{j\ne i}\lVert w_j\rVert\sin\theta_{ij}-\lVert v\rVert\sin\theta_i\big)\bar w_i-\sum_{j\ne i}\theta_{ij}w_j+\theta_iv\Big]$.
- Theorem 1 (informal): $d=\Omega(\log(n/\delta))$, small init $\sigma\sqrt d=\mathrm{poly}(n^{-1})\lVert v\rVert$, $\eta=\mathrm{poly}(\cdot)$: w.p. $1-\delta$ GD converges globally with $L(w(t))\le\mathrm{poly}(n,\lVert v\rVert,\eta^{-1})\,t^{-3}$.
- Theorem 2 (informal): for $n\ge2$, gradient flow satisfies $L(w(t))\ge(\Gamma_1t+\Gamma_2)^{-3}$ w.h.p. With $n=1$ the rate is $\exp(-\Omega(t))$ (Yehudai & Shamir 2020).
> Check script (Fig. 1 setting $\sigma=0.1$, $\eta=0.05$, $\lVert v\rVert=1$, $d=10$): Eq. (3) matches finite differences to $4\times10^{-10}$; $n=1$ reaches $L<10^{-28}$ in 750 steps; slopes $-2.99,-2.99,-2.98$ for $n=2,3,5$; $\log L/\log(\eta T)$ at $T=60\,000$: $-2.96,-3.01,-3.10$.
