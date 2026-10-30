## neuron | Landscape of a single neuron (Sec. IV-A)
$$F(v,w)=\big(y-v\,\varphi(wx)\big)^2,\qquad(x,y)=(1,1)$$
### Proposition IV.1
For $x,y\neq0$, $F$ has no sub-optimal local minima if and only if every zero $t$ of $\varphi$ is neither a local minimum nor a local maximum of $\varphi$. So ReLU and $\varphi(t)=t^2$ give sub-optimal local minima; activations like tanh do not.
### Fig. 2
$\varphi(z)=-(z-1)^2/8$: the point $(v,w)=(1,1)$ is a sub-optimal local minimum, not strict (flat along one direction).
### The rest of the overview
Linear networks (every local minimum global under conditions), over-parameterized networks (wide nets can still have sub-optimal local minima; "no bad basin" / no spurious valley results such as Theorem IV.1 for a last hidden layer at least as wide as the sample size), modifications that remove bad minima, empirical landscape studies, and links to convergence results.
> Check script: $F\ge1=F(1,1)$ on the disc of radius 0.3 around $(1,1)$; Hessian eigenvalues $0$ and $0.5$; $(v,1)$ is a minimum for $v>0$ and a saddle for $v<0$. GD (step 0.05, 6000 steps) from a $41\times41$ grid on $[-3,3]^2$: quadratic activation 69.6% at zero loss, 30.4% at $F=1$; tanh 99.9% at zero loss, the remaining start being the saddle $(0,0)$.
