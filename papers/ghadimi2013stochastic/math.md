## rsg | The RSG method
### Problem and assumptions (A1)
$\min_x f(x)=\mathbb E_\xi F(x,\xi)$ with $f$ $L$-smooth and bounded below, and a stochastic gradient with
$$\mathbb E[G(x,\xi)]=\nabla f(x),\qquad \mathbb E\lVert G(x,\xi)-\nabla f(x)\rVert^2\le\sigma^2$$
### Algorithm
Draw $R$ with $P(R=k)\propto 2\gamma_k-L\gamma_k^2$, run $x_{k+1}=x_k-\gamma_kG(x_k,\xi_k)$ for $k<R$, output $x_R$.
### Landscape on screen
$$f(x,y)=0.12(x^2+y^2)+\sum_j a_j\,e^{-\lVert (x,y)-c_j\rVert^2/s_j}$$
Globally $L$-smooth and bounded below, with two minima, two saddles and one maximum.

## rate | Theorem 2.1 and Corollary 2.2
$$\frac1L\,\mathbb E\lVert\nabla f(x_R)\rVert^2\le\frac{D_f^2+\sigma^2\sum_k\gamma_k^2}{\sum_k(2\gamma_k-L\gamma_k^2)},\qquad D_f=\Big[\tfrac{2(f(x_1)-f^*)}{L}\Big]^{1/2}$$
With $\gamma_k=\min\{\tfrac1L,\tfrac{\tilde D}{\sigma\sqrt N}\}$:
$$\frac1L\,\mathbb E\lVert\nabla f(x_R)\rVert^2\le B_N=\frac{LD_f^2}{N}+\Big(\tilde D+\frac{D_f^2}{\tilde D}\Big)\frac{\sigma}{\sqrt N}$$
So $O(1/\sqrt N)$ with noise, $O(1/N)$ without. The guarantee is **first-order** stationarity only.
> Offline check: at $N=1000$, empirical $0.143\le$ bound $0.249$.

## smallgrad | What the guarantee does not say
A point with $\lVert\nabla f\rVert^2\le\epsilon$ can be a minimum, a saddle or a maximum. Ruling out saddles needs second-order information or noise arguments (Ge 2015, Jin 2017, Lee 2016).

## zeroth | RSGF (Sec. 3)
Gaussian smoothing $f_\mu(x)=\mathbb E_{u\sim\mathcal N(0,I)}f(x+\mu u)$ and
$$G_\mu(x,\xi,u)=\frac{F(x+\mu u,\xi)-F(x,\xi)}{\mu}\,u,\qquad \mathbb E\,G_\mu=\nabla f_\mu(x)$$
Theorem 3.1: $\lVert\nabla f_\mu-\nabla f\rVert\le\tfrac{\mu}{2}L(n+3)^{3/2}$. The complexity picks up a factor of the dimension $n$.
