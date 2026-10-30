## slice | Shallow ReLU networks with scalar input
$$\mathcal N_\theta(x)=c+\sum_{j=1}^Nv_j\max\{w_jx+b_j,0\},\qquad L(\theta)=\int_\alpha^\beta\big(\mathcal N_\theta(x)-f(x)\big)^2dx,\quad f\ \text{affine}$$
$L$ is not differentiable everywhere; GD uses the left gradient (any value at kinks works in the main text).
### Theorem 1.1 (center-stable manifold, weakened regularity)
If the GD map is $C^1$ with locally Lipschitz derivative on an open set $U$, has non-zero Jacobian determinant on a full-measure open $V\subseteq U$, and at every point of $S$ its Jacobian is symmetric with an eigenvalue of modulus $>1$, then the starts whose iterates converge to $S$ form a null set. (Theorem 2.2: a Lipschitz center-stable manifold under Assumption 2.1.)
### Critical points (Proposition 4.5)
- no non-global critical point is a local maximum
- local minima that are not global: $c=f(\tfrac{\alpha+\beta}2)$ and every neuron inactive or semi-inactive with $f'(\alpha)v_jw_j<0$ (dead networks)
- saddles: classified in (iii); Lemma 4.6: the saddles in $S$ are strict for the relevant restricted maps
### Toy on screen: a slice of the real loss
$c=\tfrac12$, one neuron $v\max\{b-x,0\}$ ($w=-1$), other neurons dead, $[\alpha,\beta]=[0,1]$, $f(x)=x$:
$$L(b,v)=\int_0^{m}(A-Bx)^2dx+\int_m^1(\tfrac12-x)^2dx,\quad m=\min(b,1),\ A=\tfrac12+vb,\ B=1+v$$
and $L=\tfrac1{12}$ for $b\le0$. Here $f'(\alpha)vw=-v$: $v>0$ gives the spurious local minima of (ii), $v<0$ saddles.
> Check script: closed form matches a 4200-cell midpoint rule to $10^{-7}$; $L(0.05,-1)<\tfrac1{12}<L(0.05,1)$.

## levels | Quantized critical values and the global result
- Proposition 4.5 (iv): a non-global critical point has $L(\theta)=\dfrac{f'(\alpha)^2(\beta-\alpha)^3}{12(n+1)^4}$ for some even $n\le N$, and $\mathcal N_\theta$ is a sawtooth with $n$ interior kinks at $\alpha+i\frac{\beta-\alpha}{n+1}$
- Theorem 4.7: for almost every step size, the starts converging to the strict saddles in $S$ form a null set
- Theorem 1.2 ($N$ even): for a.e. step and a.e. start whose GD iterates converge with $\lim L<\dfrac{f'(\alpha)^2(\beta-\alpha)^3}{12(N-1)^4}$, the limit loss is $0$
- Section 5 (Theorem 5.4, Corollary 5.5): the same for leaky ReLU with small enough leak $\zeta$
> Check script: sawtooth losses $1/12$, $1.029\times10^{-3}$, $1.333\times10^{-4}$ for $n=0,2,4$; the $n=2$ sawtooth as a 4-neuron net (kink signs $(-1)^{i+1}$, $v_jw_j=\tfrac23$, $c=\tfrac12$) has gradient $7\times10^{-17}$ and Hessian eigenvalue $-0.024$: a strict saddle. 40 GD runs (N = 4, step 0.2, small output weights): all 40 end below the threshold $1.029\times10^{-3}$, with loss $\le3.5\times10^{-8}$.
