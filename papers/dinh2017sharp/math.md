## rescale | Non-negative homogeneity
$$\mathrm{ReLU}(\alpha z)=\alpha\,\mathrm{ReLU}(z)\ (\alpha>0)\ \Longrightarrow\ \phi_{\rm rect}(x\cdot\alpha\theta_1)\cdot\alpha^{-1}\theta_2=\phi_{\rm rect}(x\cdot\theta_1)\cdot\theta_2$$
Definition 5: the $\alpha$-scale transformation $T_\alpha(\theta_1,\theta_2)=(\alpha\theta_1,\alpha^{-1}\theta_2)$ leaves the network function, hence its generalization, unchanged.
### Flatness notions
- Definition 1 (volume $\epsilon$-flatness): volume of the largest connected set $C(L,\theta,\epsilon)$ around $\theta$ with $L<L(\theta)+\epsilon$
- Definition 2 ($\epsilon$-sharpness): $\max_{\theta'\in B_2(\epsilon,\theta)}\big(L(\theta')-L(\theta)\big)/\big(1+L(\theta)\big)$, related to $\lVert\nabla^2L\rVert$
### Toy on screen
$$L(\theta_1,\theta_2)=\big(\theta_2\,\mathrm{ReLU}(\theta_1)-1\big)^2,\qquad\nabla^2L(a,1/a)=2\begin{pmatrix}a^{-2}&1\\1&a^2\end{pmatrix},\quad\text{eigenvalues }0,\ 2(a^2+a^{-2})$$
> Check script: $\lambda_{\max}=32.125,\ 8.5,\ 4,\ 8.5,\ 32.125$ and $\epsilon$-sharpness ($\epsilon=0.1$) $0.161,\ 0.044,\ 0.021,\ 0.044,\ 0.161$ for $a=0.25,0.5,1,2,4$; the output is identical.

## curve | The theorems
- Theorem 2: for a one-hidden-layer ReLU net and a minimum with $\theta_1,\theta_2\neq0$, $C(L,\theta,\epsilon)$ has infinite volume for every $\epsilon>0$
- Theorem 3: $\nabla^2L(T_\alpha\theta)=D_\alpha\nabla^2L(\theta)D_\alpha$, $D_\alpha=\mathrm{diag}(\alpha^{-1}I_{n_1},\alpha I_{n_2})$
- Theorem 4: for every $M$ there is $\alpha$ with $\lVert\nabla^2L(T_\alpha\theta)\rVert_2\ge M$; Theorem 5: for deep ReLU nets, $r-\min_k n_k$ Hessian eigenvalues can be made $>M$
- Sec. 5: reparameterizing the input space or the parameters changes flatness too
> Check script: Theorem 3's identity holds at $a=3$ to $2\times10^{-3}$ (finite differences).
