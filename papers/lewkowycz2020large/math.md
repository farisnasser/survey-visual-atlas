## spike | The warmup model (Sec. 2.1)
One input $x=1$, label $y=0$, width $n$:
$$f=n^{-1/2}\,v^\top u,\qquad L=\tfrac12f^2,\qquad \lambda=\Theta(1,1)=n^{-1}\big(\lVert u\rVert^2+\lVert v\rVert^2\big)$$
$\lambda$ is the (scalar) neural tangent kernel; on the zero-loss set it equals the top Hessian eigenvalue.
### GD in parameter space (Eq. 4)
$$u_{t+1}=u_t-\eta n^{-1/2}f_tv_t,\qquad v_{t+1}=v_t-\eta n^{-1/2}f_tu_t$$
### On screen: $n=1$
$L(u,v)=\tfrac12(uv)^2$, minima on both axes with Hessian eigenvalues $0$ and $u^2+v^2=\lambda$. Start $(2,0.35)$, $\lambda_0=4.12$.
> Check script: $\eta\lambda_0=2.6$: loss $0.245\to0.484$ peak, ends at $\lambda=1.21<2/\eta=3.17$. $\eta\lambda_0=3.4$: peak 1.05, ends at $\lambda=0.14$. $\eta\lambda_0=4.2$ diverges.

## width | Exact function-space dynamics (Eqs. 6–7)
$$f_{t+1}=\Big(1-\eta\lambda_t+\frac{\eta^2f_t^2}{n}\Big)f_t,\qquad \lambda_{t+1}=\lambda_t+\frac{\eta f_t^2}{n}\,(\eta\lambda_t-4)$$
No terms neglected. For $n\to\infty$ at fixed $t$: $f_{t+1}=(1-\eta\lambda)f_t$, $\lambda$ constant (NTK).
### Catapult phase ($2/\lambda_0<\eta<4/\lambda_0$)
$|1-\eta\lambda|>1$, so $|f|$ grows geometrically; $\eta\lambda-4<0$, so every kernel update is negative. After $\sim\log n$ steps $f^2/n=O(1)$, $\lambda$ falls below $2/\eta$ and the loss converges. If $\eta\lambda_0>4$ the kernel updates are positive and the loss diverges, so $\eta_{\max}=4/\lambda_0$ in this model.
> Check script: Eqs. (6)–(7) match GD on $(u,v)$ to $6\times10^{-15}$; loss peak at step 2, 6, 9 for $n=10^2,10^4,10^6$.

## phases | Three phases and experiments (Sec. 3)
- Lazy $\eta<2/\lambda_0$: kernel constant, linear-model dynamics.
- Catapult $2/\lambda_0<\eta<\eta_{\max}$: kernel ends $O(1)$ smaller, flatter minimum.
- Divergent $\eta>\eta_{\max}$.
- Experiments (wide fully connected ReLU nets on MNIST/CIFAR-10, Wide ResNet 28-10 on CIFAR-10; MSE; SGD): $\eta_{\text{crit}}=2/\lambda_0$ matches; $\eta_{\max}=c/\lambda_0$ with $c\approx12$ for ReLU networks; best test accuracy often in the catapult phase (reported, not reproduced).
> Check script ($n=1000$): $\eta\lambda_0=1.5$ keeps $\lambda=0.995$; 2.5 and 3.5 end at $\lambda=0.51$ and $0.26$; 3.98 converges, 4.02 diverges.
