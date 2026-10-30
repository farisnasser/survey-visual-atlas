## flow | The mean-field ResNet (Eq. 2.1–2.2)
Residual blocks become particles of a distribution $\rho(\theta,t)$ over parameters and depth $t\in[0,1]$:
$$\dot X_\rho(x,t)=\int f(X_\rho(x,t),\theta)\,\rho(\theta,t)\,d\theta,\qquad X_\rho(x,0)=w_2x,\qquad E(\rho)=\mathbb E_x\tfrac12\big(\langle w_1,X_\rho(x,1)\rangle-y(x)\big)^2$$
A ResNet with $L$ layers of $M$ blocks is its Euler/particle discretization (Sec. 4): $X_{n+1}=X_n+\frac1{LM}\sum_jf(X_n,\theta_{nj})$.
### Here
$x\in[-1,1]$, $w_2x=(x,0)\in\mathbb R^2$, $w_1=(1,0)$ fixed, $y=\sin(\pi x)$, $f(X,\theta)=a\tanh(w\cdot X+b)$ with $\theta=(a,w,b)\in\mathbb R^5$, $L=16$, $M=32$, GD step $0.6LM$.
> Check script: loss $0.107\to3.1\times10^{-5}$ in 3 000 steps; gradients by backprop through depth agree with finite differences to $2\times10^{-11}$. Notes' Euler example $(1+1/L)^L$: 2.25, 2.594, $\to e$ — checked.

## depth | Theorems 2–4 (ICML numbering)
- Theorem 2: $\frac{\delta E}{\delta\rho}(\theta,t)=\mathbb E_x\,f(X_\rho(x,t),\theta)\,p_\rho(x,t)$, with the adjoint $\dot p=-p\int\nabla_Xf(X_\rho,\theta)\rho(\theta,t)d\theta$, $p(x,1)=(\langle w_1,X(1)\rangle-y)w_1$ — backprop in continuous depth.
- Lemma 1: $\lVert p_\rho(\cdot,t)\rVert_\mu\ge e^{-(C_1+C_2r)}E(\rho)$ for all $t$: the gradient signal cannot vanish through depth while the loss is positive.
- Theorem 3: if $E(\rho)>0$ there is $\nu$ with $\langle\frac{\delta E}{\delta\rho},\nu-\rho\rangle<0$, so **every local minimizer is global** (zero loss, using universal approximation).
- Theorem 4 (informal): for positively homogeneous blocks, if the Wasserstein gradient flow converges and its support separates two spheres, the limit has $E=0$.
> Check script: the smallest $\lVert p(t)\rVert/\lVert p(1)\rVert$ over depth and 10 checkpoints is 0.34; after 1 500 steps the loss is $8.0\times10^{-5}$, $1.6\times10^{-4}$, $1.2\times10^{-4}$ for $L=4,16,64$ (the $L=16$ and $L=64$ curves agree within a factor $e$ throughout).
> arXiv version numbers these Theorem 3.1, Theorem 3.2 and Theorem 3.9.
