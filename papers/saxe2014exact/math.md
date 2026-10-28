## hyperbolas | One mode, two weights
### Decoupled modes (Eq. 8–9)
With orthogonal inputs and weights aligned to the SVD of $\Sigma^{31}=\sum_\alpha s_\alpha u_\alpha v_\alpha^\top$, each mode is a pair of scalars $(a,b)$ doing gradient flow on
$$E(a,b)=\tfrac12\,(s-ab)^2,\qquad \tau\dot a=b\,(s-ab),\quad \tau\dot b=a\,(s-ab)$$
### Conserved quantity
$E$ is invariant to $a\to\lambda a,\ b\to b/\lambda$, so $\tfrac{d}{dt}(a^2-b^2)=0$: trajectories are hyperbolas $a^2-b^2=\text{const}$ ending on $ab=s$.
- origin: unstable fixed point, Hessian $\begin{pmatrix}0&-s\\-s&0\end{pmatrix}$
- every point of $ab=s$ is a global minimum
> Small-step GD conserves $a^2-b^2$ up to $O(\eta)$ (drift $5\times10^{-4}$ at $\eta=0.005$ in the check script).

## plateaus | The sigmoid learning curve
With $a=b$ and $u=ab$ (Eq. 10–12):
$$\tau\dot u=2u(s-u),\qquad u(t)=\frac{s\,e^{2st/\tau}}{e^{2st/\tau}-1+s/u_0}$$
Learning time from $u_0=\epsilon$ to within $\epsilon$ of $s$:
$$t\approx\frac{\tau}{s}\ln\frac{s}{\epsilon}$$
Stronger modes are learned first; a smaller start gives a longer plateau. A plateau here is a saddle at the origin, not a bad local minimum.
> Dots are discrete GD; they match Eq. 12 to within 0.03 in the check script.

## orthogonal | Products of random matrices
Deep linear net: $W_{\text{tot}}=W_{D}\cdots W_1$.
- scaled Gaussian $W_{ij}\sim\mathcal N(0,1/N)$: $\mathbb E\,\lVert W v\rVert^2=\lVert v\rVert^2$, but singular values of $W_{\text{tot}}$ spread out with depth (most tiny, a few large)
- orthogonal $W^\top W=I$: $W_{\text{tot}}$ is orthogonal, every singular value is 1 (dynamical isometry)
### Paper's claim (Sec. 3, Fig. 6)
Random orthogonal initialization, like greedy pretraining, gives depth-independent learning times on MNIST; scaled Gaussian initialization does not.
