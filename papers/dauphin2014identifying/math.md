## saddle | GD, Newton and saddle-free Newton (Sec. 4)
Near a critical point $\theta^*$ write $\Delta\theta=\theta-\theta^*$ in the Hessian eigenbasis $(\lambda_i,e_i)$:
$$f(\theta^*+\Delta\theta)\approx f(\theta^*)+\tfrac12\sum_i\lambda_i\,\Delta v_i^2$$
- GD step along $e_i$: $-\lambda_i\Delta v_i$ — correct direction, but tiny when $|\lambda_i|$ is small
- Newton step: $-\Delta v_i$ — for $\lambda_i<0$ it moves **toward** the saddle, which becomes an attractor
- saddle-free Newton: $-\tfrac{\lambda_i}{|\lambda_i|}\Delta v_i$ — always downhill, unit rate
### Surface on screen (paper Fig. 2a)
$$f(x,y)=5x^2-y^2,\qquad \nabla^2 f=\mathrm{diag}(10,-2)$$
> Offline: from $(0.9,0.12)$ SFN leaves the box in 7 steps, GD in 18, Newton converges to the saddle.

## weak | Weak negative curvature
$$f(x,y)=5x^2-\varepsilon y^2$$
GD multiplies $y$ by $(1+2\eta\varepsilon)$ per step: escape time $\propto 1/(\eta\varepsilon)$. SFN multiplies $y$ by $(1+\eta)$ regardless of $\varepsilon$ (until the damping floor $\delta$).
> Offline: $\varepsilon=0.01$: GD 1621 steps, SFN 28.

## monkey | Degenerate saddle (Fig. 2b)
$$f(x,y)=x^3-3xy^2,\qquad \nabla^2 f(0)=0$$
Three descending valleys; second-order information vanishes at the origin, so damping matters.

## index | Index and error (Sec. 2)
Index $\alpha=$ fraction of negative Hessian eigenvalues at a critical point. For random Gaussian fields (Bray & Dean 2007) critical points concentrate on a monotone curve in the $(\alpha,\varepsilon)$ plane; the Hessian spectrum is a Wigner semicircle shifted by an amount set by $\varepsilon$:
$$\rho(\lambda)=\frac{\sqrt{4-(\lambda-m)^2}}{2\pi}$$
Minima ($\alpha=0$) have error near the global minimum; high-error critical points are saddles. Fig. 1 measures the same trend in small MLPs on downsampled MNIST/CIFAR-10.
### Trust-region view (Eq. 4–5)
$$|s^\top Hs|\le s^\top|H|s\ \Rightarrow\ \Delta\theta=-|H|^{-1}\nabla f,\qquad |H|=Q|\Lambda|Q^\top$$
At scale: Krylov subspace (Lanczos) + damping (Algorithm 1). Empirical method, no finite-time guarantee.
