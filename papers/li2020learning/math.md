## neurons | Teacher, student and Algorithm 1
Teacher (Eq. 1.1), $x\sim N(0,I_d)$, orthonormal $\{w_i^\star\}$, $a_i\in[\frac1{\kappa d},\frac\kappa d]$, $\sum a_i=1$:
$$f^\star(x)=\sum_{i=1}^da_i\,|w_i^{\star\top}x|=\sum_ia_i\big(\mathrm{ReLU}(w_i^{\star\top}x)+\mathrm{ReLU}(-w_i^{\star\top}x)\big)$$
Student (Eq. 1.2): $f_W(x)=\frac1m\sum_{i=1}^m\lVert w_i\rVert\,\mathrm{ReLU}(w_i^\top x)$. Algorithm 1: $w_i^{(0)}\sim N(0,I/d)$; truncated GD $w_i\leftarrow w_i-\eta\,\mathbb 1\{\lVert w_i\rVert^2\le1/\lambda\}\nabla_{w_i}\hat L(W)$, stage 1 with $\lambda_0$, stage 2 with $\lambda_1$.
### Why it is a tensor problem (Eq. 2.1)
In the Hermite basis the population loss is a sum of tensor-decomposition losses $c_{2j}\big\lVert\frac1m\sum_i\lVert w_i\rVert^2\bar w_i^{\otimes 2j}-\sum_ia_ie_i^{\otimes 2j}\big\rVert_F^2$: low orders are matched first, the fourth-order term pins down the directions.
### Here
$d=3$, $W^\star=I$, $a_i=\tfrac13$, $m=60$, $\eta=2$, $1/\lambda=25$; loss computed exactly with $\mathbb E[\mathrm{relu}(u\cdot x)\mathrm{relu}(v\cdot x)]=\frac{\lVert u\rVert\lVert v\rVert\sin\phi+(\pi-\phi)\,u\cdot v}{2\pi}$.
> Check script: kernel and loss match Monte Carlo; gradient matches finite differences to $4\times10^{-11}$; loss $0.234\to3.4\times10^{-5}$ in 6 000 steps; neuron mass within 15° of some $\pm e_i$: 4.4% at init, 97.0% at the end.

## vsfixed | Theorem 1.1 vs Theorem 1.2
- Theorem 1.1: with $N=\mathrm{poly}_\kappa(d)$ samples, $m\ge\mathrm{poly}(d)/\mathrm{poly}(\lambda_1)$ and suitable $\eta,\lambda_0,\lambda_1$, w.p. $1-1/\mathrm{poly}(d)$ the output of Algorithm 1 has $L(\hat W)\le O(1/d^{1+Q})$ ($0<Q<1/100$).
- Theorem 1.2: any feature map $\phi:\mathbb R^d\to\mathbb R^N$ or kernel method with $N=\mathrm{poly}(d)$ (features or samples) has $L=\Omega(1/d)$ for some admissible $W^\star, a$. So the trained network beats every polynomial-size kernel, NTK included.
> Check script: the best population fit on the 60 frozen initial features has loss $1.2\times10^{-3}$, 35× the trained network's. (A same-width illustration at $d=3$; the theorem's separation is asymptotic in $d$.)
- Notes call the lower bound "Theorem 2"; in the paper it is Theorem 1.2.
